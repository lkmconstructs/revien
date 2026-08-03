"""
Tests for the BM25 lexical lane (REVIEN_LEXICAL=bm25).

Three tiers:
  1. Module math — ranking correctness, tokenize/stopword behavior,
     deterministic tie-break, validation errors, empty query/docs. Runs
     against revien.retrieval.bm25 directly, no engine involved.
  2. Engine wiring — REVIEN_LEXICAL=bm25 finds what the substring keyword
     lane misses, bm25_score surfaces in score_breakdown only when the lane
     is selected, unset env stays byte-identical to the shipped keyword
     path, and the RRF+BM25 combination returns results.
  3. Scoring-blend contract — WITH the semantic layer enabled (a stub, so
     similarities are fixed known numbers): a bm25-only node uses bm25_score
     as its query_relevance term; a node with BOTH signals takes their MAX,
     not their sum; and the bounded (score/(score+1)) relevance term never
     exceeds 1.0. Tier 2's tests alone never exercise the both-signals
     branch (semantic is always disabled there), so this tier exists to
     make the CHANGELOG's "MAX not sum" claim an actually-checked fact.

House rule: assert effects (result contents, scores), not absence of
exceptions.
"""

import os
import tempfile

import pytest

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.bm25 import bm25_rank, tokenize
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex


# ── Tier 1: module math ─────────────────────────────────────────────────


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_hyphens_are_boundaries_underscore_is_a_word_char(self):
        # \w+ treats '-' as a boundary but '_' as part of the token (regex
        # word-char class) — matching Python's own \w semantics exactly.
        assert tokenize("zx-9942-omega") == ["zx", "9942", "omega"]
        assert tokenize("feature_flag_name") == ["feature_flag_name"]

    def test_stopwords_dropped(self):
        assert tokenize("the quick fox and the lazy dog") == [
            "quick", "fox", "lazy", "dog"
        ]

    def test_empty_and_non_string_input(self):
        assert tokenize("") == []
        assert tokenize(123) == ["123"]


class TestBM25Rank:
    def test_rare_exact_term_beats_common_repeated_terms(self):
        docs = [
            ("common", "project launch meeting project planning"),
            ("target", "heliotrope launch code orchid seven"),
        ]
        ranked = bm25_rank("What was the heliotrope launch code?", docs)
        assert ranked[0][0] == "target"

    def test_zero_overlap_query_returns_empty(self):
        docs = [("a", "apples and oranges"), ("b", "bananas and pears")]
        assert bm25_rank("unrelated-zorblatt-term", docs) == []

    def test_empty_query_returns_empty(self):
        docs = [("a", "some content")]
        assert bm25_rank("", docs) == []
        assert bm25_rank("the and a", docs) == []  # all-stopword query

    def test_empty_documents_returns_empty(self):
        assert bm25_rank("anything", []) == []

    def test_top_n_zero_returns_empty_list_distinct_from_none(self):
        docs = [("a", "apple pie"), ("b", "apple tart")]
        assert bm25_rank("apple", docs, top_n=0) == []
        assert len(bm25_rank("apple", docs, top_n=None)) == 2

    def test_top_n_truncates_ranked_list(self):
        docs = [("a", "apple pie"), ("b", "apple tart"), ("c", "apple crumble")]
        ranked = bm25_rank("apple", docs, top_n=2)
        assert len(ranked) == 2

    def test_deterministic_tiebreak_by_input_order(self):
        # Identical text -> identical score; input order must decide, not
        # dict/set iteration order.
        docs = [("z", "shared term here"), ("a", "shared term here")]
        ranked = bm25_rank("shared term", docs)
        assert [nid for nid, _ in ranked] == ["z", "a"]

    def test_scores_are_positive_and_descending(self):
        docs = [
            ("a", "cats and dogs are common pets"),
            ("b", "cats are extremely common household pets everywhere"),
            ("c", "a rare zoological specimen unrelated to pets"),
        ]
        ranked = bm25_rank("cats pets", docs)
        scores = [score for _, score in ranked]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("bad_k1", [0, -1, "x", True, float("nan"), float("inf")])
    def test_invalid_k1_raises(self, bad_k1):
        with pytest.raises((TypeError, ValueError)):
            bm25_rank("query", [("a", "text")], k1=bad_k1)

    @pytest.mark.parametrize("bad_b", [-0.1, 1.1, "x", True, float("nan")])
    def test_invalid_b_raises(self, bad_b):
        with pytest.raises((TypeError, ValueError)):
            bm25_rank("query", [("a", "text")], b=bad_b)

    @pytest.mark.parametrize("bad_top_n", [-1, 1.5, "x", True])
    def test_invalid_top_n_raises(self, bad_top_n):
        with pytest.raises((TypeError, ValueError)):
            bm25_rank("query", [("a", "text")], top_n=bad_top_n)


# ── Tier 2: engine wiring ────────────────────────────────────────────────


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _add_fact(store, label, content, access_count=0):
    return store.add_node(Node(
        node_type=NodeType.FACT,
        label=label,
        content=content,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        access_count=access_count,
    ))


class TestUnsetGateStaysKeyword:
    def test_default_lexical_mode_is_keyword(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_LEXICAL", raising=False)
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.lexical_mode == "keyword"

    def test_unset_and_non_bm25_values_are_byte_identical_to_keyword(
        self, store, monkeypatch
    ):
        _add_fact(store, "PostgreSQL decision",
                  "We decided to use PostgreSQL for the enterprise tier.")
        _add_fact(store, "pricing", "Pricing is $499/month for enterprise.")

        def snapshot(resp):
            return [
                (r.node_id, round(r.score, 12), tuple(sorted(r.score_breakdown)))
                for r in resp.results
            ]

        monkeypatch.delenv("REVIEN_LEXICAL", raising=False)
        base = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        baseline = snapshot(base.recall("What database did we choose?", top_n=5))

        for other in ("", "0", "off", "keyword"):
            monkeypatch.setenv("REVIEN_LEXICAL", other)
            eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
            resp = eng.recall("What database did we choose?", top_n=5)
            assert snapshot(resp) == baseline

    def test_no_bm25_score_key_leaks_when_lane_unselected(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_LEXICAL", raising=False)
        _add_fact(store, "PostgreSQL", "We chose PostgreSQL as the database.")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("Tell me about PostgreSQL database", top_n=5, min_score=0.0)
        for r in resp.results:
            assert "bm25_score" not in r.score_breakdown


class TestBM25LaneWiring:
    def test_bm25_lane_finds_identifier_keyword_lane_misses(self, store, monkeypatch):
        """A trailing-punctuation identifier query: the shipped keyword lane
        splits on whitespace only, so 'zx-9942-omega?' never substring-
        matches content that has 'zx-9942-omega' without the question mark
        — a total miss, no anchor at all. BM25 tokenizes both sides on
        \\w+ (hyphens/punctuation are boundaries), so 'zx', '9942', 'omega'
        overlap and the node is found."""
        target = _add_fact(
            store, "incident ticket",
            "Internal reference zx-9942-omega for ops incident review.",
        )

        monkeypatch.delenv("REVIEN_LEXICAL", raising=False)
        baseline = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        miss = baseline.recall("zx-9942-omega?", top_n=5, min_score=0.0)
        assert all(r.node_id != target.node_id for r in miss.results)

        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        hit = eng.recall("zx-9942-omega?", top_n=5, min_score=0.0, debug=True)
        got = {r.node_id: r for r in hit.results}
        assert target.node_id in got
        assert got[target.node_id].score_breakdown["bm25_score"] > 0
        assert eng.lexical_mode == "bm25"
        assert target.node_id in hit.diagnostics["bm25_scores"]

    def test_bm25_relevance_beats_popularity_hub(self, store, monkeypatch):
        """access_count=100 on the hub makes frequency (a real scoring
        factor, weight 0.30) the thing BM25 relevance has to overcome, not
        just anchor-list recency ordering — without that popularity
        disadvantage this test passed even when the bm25 lane never
        dispatched (keyword lane happened to rank the newer 'target' node
        first regardless). The keyword-lane baseline assertion below proves
        the hub's popularity actually DOES win without the bm25 signal, so
        bm25 overturning it is the discriminating behavior under test."""
        common = _add_fact(
            store, "common hub",
            "launch project meeting launch project planning launch",
            access_count=100,
        )
        target = _add_fact(
            store, "target",
            "heliotrope launch code orchid seven",
        )

        monkeypatch.delenv("REVIEN_LEXICAL", raising=False)
        baseline_eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        baseline = baseline_eng.recall(
            "heliotrope launch code", top_n=5, min_score=0.0
        )
        assert baseline.results[0].node_id == common.node_id, (
            "keyword-lane baseline: the popularity hub must rank FIRST here "
            "(no query-relevance signal to overcome frequency) — otherwise "
            "bm25 'winning' below proves nothing"
        )

        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall(
            "heliotrope launch code", top_n=5, min_score=0.0, debug=True
        )
        assert resp.results[0].node_id == target.node_id

    def test_rrf_plus_bm25_fused_path_returns_results(self, store, monkeypatch):
        target = _add_fact(store, "widget", "widget alpha bravo charlie delta")
        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.hybrid_mode == "rrf"
        assert eng.lexical_mode == "bm25"
        resp = eng.recall("widget alpha", top_n=5, min_score=0.0)
        assert any(r.node_id == target.node_id for r in resp.results)


# ── Tier 3: scoring-blend contract (semantic layer ENABLED via a stub) ──


class _StubSemantic:
    """Deterministic stand-in for the semantic layer — fixes EXACTLY which
    node gets which similarity, so the bm25/semantic blend contract (MAX,
    not sum; bm25_score only surfaces under REVIEN_LEXICAL=bm25) is checked
    against known numbers instead of whatever a real/mock embedder happens
    to produce. Mirrors the production overlay's own StubSemantic pattern."""

    def __init__(self, ranked=(), enabled=True):
        self.ranked = list(ranked)
        self.is_enabled = enabled

    def search(self, query, top_k=10):
        return self.ranked[:top_k] if self.is_enabled else []

    def inactive_reason(self):
        return None if self.is_enabled else "stub-disabled"

    def pending_note(self):
        return None


class TestScoringBlendContract:
    def test_bm25_only_node_uses_bm25_as_relevance_term(self, store, monkeypatch):
        """A node with a bm25 hit but NO semantic hit: query_relevance in the
        final-score formula must be its bm25_score, not silently dropped to
        0. Reconstruct final_score from the OTHER score_breakdown fields and
        assert it matches the documented formula exactly — (query_relevance
        + graph_refine * base_composite + community_boost) *
        effective_confidence."""
        target = _add_fact(store, "distinct", "zeta quorlex distinct marker text")
        _add_fact(store, "filler", "unrelated filler content about nothing")
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = RetrievalEngine(store, semantic=_StubSemantic(ranked=(), enabled=True))
        resp = eng.recall("zeta quorlex", top_n=5, min_score=0.0, debug=True)

        result = next(r for r in resp.results if r.node_id == target.node_id)
        bd = result.score_breakdown
        assert bd["semantic_sim"] == 0.0, "no semantic hit for this node"
        bm25_score = resp.diagnostics["bm25_scores"][target.node_id]
        assert bm25_score > 0.0
        assert bd["bm25_score"] == bm25_score

        expected = (
            bm25_score + eng.graph_refine * bd["base_composite"] + bd["community_boost"]
        ) * bd["effective_confidence"]
        assert result.score == pytest.approx(expected, rel=1e-9)

    def test_both_signals_present_takes_max_not_sum(self, store, monkeypatch):
        """Same node scores on BOTH lanes with DIFFERENT magnitudes: the
        final score must reflect max(sim, bm25_score) — a mutant that
        summed the two signals, or that used sim alone and ignored bm25,
        must fail this."""
        target = _add_fact(store, "distinct", "zeta quorlex distinct marker text")
        _add_fact(store, "filler", "unrelated filler content about nothing")
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        # Must clear the semantic_sim_floor (0.30 default) to register as a
        # semantic hit at all, but still stay below the bm25 score this
        # query yields.
        stub_sim = 0.31
        eng = RetrievalEngine(
            store, semantic=_StubSemantic(ranked=[(target.node_id, stub_sim)], enabled=True)
        )
        resp = eng.recall("zeta quorlex", top_n=5, min_score=0.0, debug=True)

        result = next(r for r in resp.results if r.node_id == target.node_id)
        bd = result.score_breakdown
        bm25_score = resp.diagnostics["bm25_scores"][target.node_id]
        assert bd["semantic_sim"] == pytest.approx(stub_sim)
        assert bm25_score > stub_sim, "test setup: bm25 score must exceed the stub sim"

        expected_max = (
            max(stub_sim, bm25_score)
            + eng.graph_refine * bd["base_composite"] + bd["community_boost"]
        ) * bd["effective_confidence"]
        expected_sum = (
            (stub_sim + bm25_score)
            + eng.graph_refine * bd["base_composite"] + bd["community_boost"]
        ) * bd["effective_confidence"]
        assert result.score == pytest.approx(expected_max, rel=1e-9)
        assert result.score != pytest.approx(expected_sum, rel=1e-6)

    def test_blended_relevance_never_exceeds_one(self, store, monkeypatch):
        """The bound (score / (score + 1)) must keep the query-relevance
        term under 1.0 even for an overwhelming raw BM25 score (extreme term
        repetition) — a mutant that dropped the bound would let it grow
        unboundedly and dominate every other scoring factor."""
        target = _add_fact(
            store, "distinct",
            " ".join(["quorlex"] * 50),  # extreme repetition -> large raw bm25 score
        )
        _add_fact(store, "filler", "unrelated filler content about nothing")
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))

        _ids, scores = eng._bm25_candidates("quorlex")
        assert target.node_id in scores
        assert 0.0 < scores[target.node_id] < 1.0
