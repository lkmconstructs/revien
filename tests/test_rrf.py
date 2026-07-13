"""
Tests for LEG P1 — RRF hybrid fusion (REVIEN_HYBRID=rrf, experimental).

Three tiers:
  1. Fusion math — rrf_score = Σ 1/(k + rank) on synthetic lists, exact
     values, ordering, top-N truncation, deterministic tie-break.
  2. Gate-off byte-identity — with REVIEN_HYBRID unset (or any value other
     than "rrf") the anchor-selection path is the shipped one: keyword search
     stays a fallback (never runs when entity anchors exist), responses are
     identical to a gate-unaware engine, and no new keys leak into
     score_breakdown or diagnostics (mirrors tests/test_semantic.py's
     byte-identity pattern).
  3. RRF path wiring — under REVIEN_HYBRID=rrf the anchor set is fused from
     BOTH lists (keyword-only hits AND semantic-only hits are anchors),
     keyword search runs ALWAYS, and REVIEN_RRF_K is honored.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine, rrf_fuse
from revien.semantic.index import SemanticIndex

from tests.test_semantic import _InMemoryVectorIndex, _MockEmbedder


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


def _add_fact(store, label, content):
    now = datetime.now(timezone.utc)
    node = Node(
        node_type=NodeType.FACT,
        label=label,
        content=content,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        created_at=now,
        last_accessed=now,
    )
    return store.add_node(node)


# ── Tier 1: fusion math on synthetic lists ─────────────────────────────

class TestRRFFusionMath:
    def test_known_ranks_exact_scores_and_order(self):
        # list A: a(1) b(2) c(3); list B: b(1) c(2) d(3); k=60
        fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60.0)
        # b = 1/62 + 1/61 ≈ .03253 > c = 1/63 + 1/62 ≈ .03200
        #   > a = 1/61 ≈ .01639 > d = 1/63 ≈ .01587
        assert fused == ["b", "c", "a", "d"]

    def test_score_values_match_formula(self):
        # Reconstruct the scores independently and check the head choice.
        lists = [["x", "y"], ["y", "z"], ["y"]]
        fused = rrf_fuse(lists, k=60.0)
        score_y = 1 / 62 + 1 / 61 + 1 / 61
        score_x = 1 / 61
        score_z = 1 / 62
        assert score_y > score_x > score_z
        assert fused == ["y", "x", "z"]

    def test_single_list_preserves_order(self):
        assert rrf_fuse([["p", "q", "r"]], k=60.0) == ["p", "q", "r"]

    def test_top_n_truncates(self):
        fused = rrf_fuse([["a", "b", "c", "d"]], k=60.0, top_n=2)
        assert fused == ["a", "b"]

    def test_k_shapes_the_blend(self):
        # 'solo' is rank 1 in ONE list; 'both' is rank 4 in TWO lists.
        # Small k: a single top rank dominates. Large k: consensus wins.
        lists = [["solo", "x", "y", "both"], ["p", "q", "r", "both"]]
        sharp = rrf_fuse(lists, k=1.0)
        flat = rrf_fuse(lists, k=1000.0)
        # k=1: 1/2 (a single top rank) > 2/5 (deep consensus).
        assert sharp.index("solo") < sharp.index("both")
        # k=1000: 2/1004 (consensus) > 1/1001 (single top rank).
        assert flat[0] == "both"
    def test_deterministic_tiebreak_by_id(self):
        # Identical scores (rank 1 in one list each) break by item id.
        assert rrf_fuse([["b"], ["a"]], k=60.0) == ["a", "b"]

    def test_empty_input(self):
        assert rrf_fuse([], k=60.0) == []
        assert rrf_fuse([[], []], k=60.0) == []

    def test_duplicate_within_and_across_lists(self):
        fused = rrf_fuse([["a", "b"], ["a"]], k=60.0)
        assert fused == ["a", "b"]
        assert len(fused) == len(set(fused))


# ── Tier 2: gate-off byte-identity (shipped path untouched) ────────────

class TestGateOffByteIdentity:
    def _snapshot(self, resp):
        """Full comparable serialization of a RetrievalResponse."""
        return [
            (r.node_id, r.label, round(r.score, 12),
             tuple(sorted(r.score_breakdown)), tuple(r.path))
            for r in resp.results
        ]

    def test_env_unset_matches_non_rrf_values(self, store, monkeypatch):
        """Unset, empty, and any non-'rrf' value must all run the identical
        shipped anchor path — only the literal 'rrf' flips the experiment."""
        _add_fact(store, "PostgreSQL decision",
                  "We decided to use PostgreSQL for the enterprise tier.")
        _add_fact(store, "pricing", "Pricing is $499/month for enterprise.")

        monkeypatch.delenv("REVIEN_HYBRID", raising=False)
        base = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp_unset = base.recall("What database did we choose?", top_n=5, debug=True)

        for other in ("", "0", "off", "weighted"):
            monkeypatch.setenv("REVIEN_HYBRID", other)
            eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
            resp = eng.recall("What database did we choose?", top_n=5, debug=True)
            assert self._snapshot(resp) == self._snapshot(resp_unset)
            assert resp.diagnostics["anchors"] == resp_unset.diagnostics["anchors"]

    def test_gate_off_keyword_stays_a_fallback(self, store, monkeypatch):
        """Shipped contract: keyword search runs ONLY when entity extraction
        finds no anchors. The RRF leg must not change that when gated off."""
        monkeypatch.delenv("REVIEN_HYBRID", raising=False)
        _add_fact(store, "PostgreSQL", "We chose PostgreSQL as the database.")

        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        calls = {"n": 0}
        orig = store.search_nodes_keyword

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(store, "search_nodes_keyword", counting)

        # Entity anchor exists ("PostgreSQL" label match) -> no keyword call.
        resp = eng.recall("Tell me about PostgreSQL", top_n=5, debug=True)
        assert resp.diagnostics["anchors"]["entity"]
        assert calls["n"] == 0

        # No entity anchor -> exactly one fallback keyword call.
        eng.recall("enterprise database choice", top_n=5)
        assert calls["n"] >= 1

    def test_gate_off_no_new_breakdown_keys(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_HYBRID", raising=False)
        _add_fact(store, "PostgreSQL", "We chose PostgreSQL as the database.")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("Tell me about PostgreSQL", top_n=5, debug=True)
        for r in resp.results:
            assert "rrf" not in " ".join(r.score_breakdown)
        assert set(resp.diagnostics["anchors"]) == {
            "entity", "keyword", "semantic", "all"
        }


# ── Tier 3: RRF path wiring ────────────────────────────────────────────

class TestRRFPath:
    def test_anchors_fused_from_both_lists(self, store, monkeypatch):
        """A keyword-only hit AND a semantic-only hit must BOTH land in the
        fused anchor set — the point of the leg."""
        # Keyword-only: 'zorblatt' matches by substring, embeds to nothing
        # (not in the mock vocab).
        kw_node = _add_fact(store, "zorblatt gadget",
                            "the zorblatt gadget shipped last week")
        # Semantic-only: query says 'puppy', node says 'dog' — zero literal
        # overlap, tied via the mock embedder's shared dimension.
        sem_node = _add_fact(store, "dog", "A friendly dog at the park.")
        _add_fact(store, "bread", "A sourdough bread recipe.")

        sem = _InMemoryVectorIndex(store, _MockEmbedder())
        sem.reindex_all()

        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        eng = RetrievalEngine(store, semantic=sem)
        assert eng.hybrid_mode == "rrf"
        resp = eng.recall("puppy zorblatt", top_n=5, debug=True,
                          include_context=True)

        anchors = resp.diagnostics["anchors"]
        assert kw_node.node_id in anchors["all"], "keyword-list hit not fused"
        assert sem_node.node_id in anchors["all"], "semantic-list hit not fused"
        assert kw_node.node_id in anchors["keyword"]
        assert sem_node.node_id in anchors["semantic"]
        # Both fused anchors are retrievable results too.
        got = {r.node_id for r in resp.results}
        assert kw_node.node_id in got and sem_node.node_id in got

    def test_keyword_search_runs_always_under_rrf(self, store, monkeypatch):
        """Under rrf, keyword search is a fusion list, not a fallback — it
        must run even when entity anchors would have existed."""
        _add_fact(store, "PostgreSQL", "We chose PostgreSQL as the database.")
        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))

        calls = {"n": 0, "limits": []}
        orig = store.search_nodes_keyword

        def counting(keywords, limit=10, **kw):
            calls["n"] += 1
            calls["limits"].append(limit)
            return orig(keywords, limit=limit, **kw)

        monkeypatch.setattr(store, "search_nodes_keyword", counting)
        eng.recall("Tell me about PostgreSQL", top_n=5)
        assert calls["n"] == 1
        # Widened to the semantic list length, not the fallback cap of 10.
        assert calls["limits"] == [eng.semantic_top_k]

    def test_consensus_node_ranks_first_in_fused_anchors(self, store, monkeypatch):
        """A node present in BOTH lists must outrank single-list nodes in the
        fused anchor ordering (RRF consensus)."""
        both = _add_fact(store, "dog", "the dog was at the park")  # kw 'dog' + semantic
        _add_fact(store, "bread", "A sourdough bread recipe.")

        sem = _InMemoryVectorIndex(store, _MockEmbedder())
        sem.reindex_all()

        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        eng = RetrievalEngine(store, semantic=sem)
        resp = eng.recall("dog", top_n=5, debug=True, include_context=True)
        assert resp.diagnostics["anchors"]["all"][0] == both.node_id

    def test_rrf_works_with_semantic_disabled(self, store, monkeypatch):
        """Keyword-only fusion still produces anchors when the semantic layer
        is off (single-list RRF degenerates to the keyword ranking)."""
        node = _add_fact(store, "zorblatt gadget", "the zorblatt gadget shipped")
        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("zorblatt status", top_n=5, debug=True)
        assert node.node_id in resp.diagnostics["anchors"]["all"]

    def test_rrf_k_env_override(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        monkeypatch.setenv("REVIEN_RRF_K", "30")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.rrf_k == 30.0
        monkeypatch.delenv("REVIEN_RRF_K")
        eng2 = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng2.rrf_k == 60.0

    def test_gate_value_is_case_insensitive(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_HYBRID", " RRF ")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.hybrid_mode == "rrf"
