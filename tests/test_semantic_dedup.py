"""
Semantic dedup: after the lexical layers miss, merge a
candidate into an existing SAME-TYPE node when their embeddings sit at or
above the cosine threshold. OPT-IN via REVIEN_SEMANTIC_DEDUP=1 — the unset
default must be byte-identical to lexical-only dedup. Merges are
non-destructive (survivor reinforced, candidate never created) and leave a
`semantic_merge` audit entry; contradictions (negation-marker parity) are
never merged — they belong to supersession.
"""

import os
import tempfile

import pytest

from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.dedup import Deduplicator

from tests.test_semantic import _InMemoryVectorIndex


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


class _PhraseEmbedder:
    """Deterministic embedder with controllable pairwise cosine. Any text
    containing a registered key embeds to that key's vector; unknown text
    gets its own orthogonal axis, so unrelated nodes never collide."""

    def __init__(self, phrases):
        # phrases: {key_substring: vector}
        self.phrases = phrases
        self._unknown_axis = {}
        self._dim = len(next(iter(phrases.values())))

    @property
    def dim(self):
        return self._dim

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        out = []
        for t in texts:
            tl = (t or "").lower()
            vec = None
            for key, v in self.phrases.items():
                if key in tl:
                    vec = list(v)
                    break
            if vec is None:
                # Orthogonal one-hot in an extended tail per unknown text.
                axis = self._unknown_axis.setdefault(
                    tl, len(self._unknown_axis)
                )
                vec = [0.0] * self._dim
                vec[axis % self._dim] = 1.0
            out.append(vec)
        return out


class _DedupVectorIndex(_InMemoryVectorIndex):
    """In-memory index + the true-cosine find_similar the dedup layer uses."""

    def find_similar(self, text, top_k=8):
        if not text.strip() or not self._vectors:
            return []
        import math
        q = self._get_embedder().embed([text])[0]

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = sorted(
            ((nid, cos(q, v)) for nid, v in self._vectors.items()),
            key=lambda x: x[1], reverse=True,
        )
        return scored[:top_k]


# Vectors: "pacing-a"/"pacing-b" are paraphrase-close (cos ~0.985),
# "pacing-far" is related but below any sane merge floor (cos ~0.6).
_VECTORS = {
    "reader dislikes when pacing slows": [1.0, 0.0, 0.0, 0.0],
    "reader felt the pacing died mid-book": [0.985, 0.17, 0.0, 0.0],
    "reader mentioned pacing once": [0.6, 0.8, 0.0, 0.0],
    # Lexically DISTANT contradiction pair (Levenshtein ratio well under the
    # fuzzy layer's 0.75 floor) so these exercise the SEMANTIC layer's guard —
    # a near-identical surface form like "does not like present tense" would
    # merge at the fuzzy layer before semantic dedup ever runs.
    "enjoys present tense": [0.0, 0.0, 1.0, 0.0],
    "reading in the present tense is never enjoyable": [0.0, 0.0, 0.99, 0.14],
}


def _mk_node(label, node_type=NodeType.PREFERENCE):
    return Node(
        node_type=node_type, label=label, content=label,
        source_type=SourceType.EXTRACTED, confidence=1.0,
    )


def _dedup(store, monkeypatch=None, gate="1", **env):
    sem = _DedupVectorIndex(store, _PhraseEmbedder(_VECTORS))
    if monkeypatch is not None:
        if gate is not None:
            monkeypatch.setenv("REVIEN_SEMANTIC_DEDUP", gate)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    return Deduplicator(store, GraphOperations(store), semantic=sem), sem


def _seed(store, sem, label, node_type=NodeType.PREFERENCE):
    node = store.add_node(_mk_node(label, node_type))
    sem.index_node(node.node_id, node.label, node.content)
    return node


class TestGateOff:
    def test_unset_gate_is_lexical_only(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_SEMANTIC_DEDUP", raising=False)
        dedup, sem = _dedup(store, gate=None)
        existing = _seed(store, sem, "reader dislikes when pacing slows")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader felt the pacing died mid-book")
        )
        assert is_new is True
        assert got.node_id != existing.node_id


class TestSemanticMerge:
    def test_paraphrase_merges_into_existing(self, store, monkeypatch):
        dedup, sem = _dedup(store, monkeypatch)
        existing = _seed(store, sem, "reader dislikes when pacing slows")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader felt the pacing died mid-book")
        )
        assert is_new is False
        assert got.node_id == existing.node_id
        # Survivor reinforced, not rewritten.
        merged = store.get_node(existing.node_id)
        assert merged.access_count == 1
        assert merged.label == "reader dislikes when pacing slows"
        # Audit trail: the reviewable surface for false merges.
        ops = [row["op"] for row in store.get_node_audit(existing.node_id)]
        assert "semantic_merge" in ops
        entry = next(
            row for row in store.get_node_audit(existing.node_id)
            if row["op"] == "semantic_merge"
        )
        assert "pacing died mid-book" in entry["actor"]
        assert "cosine=" in entry["actor"]

    def test_below_threshold_creates_new_node(self, store, monkeypatch):
        dedup, sem = _dedup(store, monkeypatch)
        _seed(store, sem, "reader dislikes when pacing slows")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader mentioned pacing once")
        )
        assert is_new is True

    def test_threshold_env_knob(self, store, monkeypatch):
        # Floor dropped to 0.5: the far paraphrase (cos ~0.6) now merges.
        dedup, sem = _dedup(
            store, monkeypatch, REVIEN_SEMANTIC_DEDUP_THRESHOLD="0.5"
        )
        existing = _seed(store, sem, "reader dislikes when pacing slows")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader mentioned pacing once")
        )
        assert is_new is False
        assert got.node_id == existing.node_id

    def test_different_type_never_merges(self, store, monkeypatch):
        dedup, sem = _dedup(store, monkeypatch)
        _seed(store, sem, "reader dislikes when pacing slows",
              node_type=NodeType.FACT)
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader felt the pacing died mid-book",
                     node_type=NodeType.PREFERENCE)
        )
        assert is_new is True

    def test_superseded_node_never_absorbs(self, store, monkeypatch):
        from datetime import datetime, timezone
        dedup, sem = _dedup(store, monkeypatch)
        existing = _seed(store, sem, "reader dislikes when pacing slows")
        store.update_node(
            existing.node_id,
            invalidated_at=datetime.now(timezone.utc),
            _audit_op="supersede",
        )
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader felt the pacing died mid-book")
        )
        assert is_new is True

    def test_allow_semantic_false_skips_layer(self, store, monkeypatch):
        # The capture path (defer_embed) must never embed inline.
        dedup, sem = _dedup(store, monkeypatch)
        _seed(store, sem, "reader dislikes when pacing slows")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reader felt the pacing died mid-book"),
            allow_semantic=False,
        )
        assert is_new is True


class TestFuzzyNegationGuard:
    """The fuzzy layer merged near-identical negation pairs — the
    edit distance IS the negation. Guard is unconditional (corruption fix,
    not an experiment). No semantic index involved: pure lexical path."""

    def test_negation_pair_survives_fuzzy_layer(self, store):
        dedup = Deduplicator(store, GraphOperations(store))
        existing = store.add_node(_mk_node("likes present tense"))
        got, is_new = dedup.deduplicate_node(
            _mk_node("does not like present tense")
        )
        assert is_new is True
        assert got.node_id != existing.node_id

    def test_same_polarity_fuzzy_still_merges(self, store):
        # The guard must not break legitimate fuzzy merges (typo class).
        dedup = Deduplicator(store, GraphOperations(store))
        existing = store.add_node(_mk_node("likes present tense"))
        got, is_new = dedup.deduplicate_node(
            _mk_node("likes present tensse")
        )
        assert is_new is False
        assert got.node_id == existing.node_id

    def test_both_negated_fuzzy_still_merges(self, store):
        dedup = Deduplicator(store, GraphOperations(store))
        existing = store.add_node(_mk_node("never reads novellas"))
        got, is_new = dedup.deduplicate_node(
            _mk_node("never reads novelas")
        )
        assert is_new is False
        assert got.node_id == existing.node_id


class TestNegationGuard:
    def test_contradiction_not_merged(self, store, monkeypatch):
        # cos ~0.99 but opposite polarity: supersession's case, not merge's.
        dedup, sem = _dedup(store, monkeypatch)
        _seed(store, sem, "enjoys present tense")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reading in the present tense is never enjoyable")
        )
        assert is_new is True

    def test_guard_can_be_disabled_for_sweeps(self, store, monkeypatch):
        dedup, sem = _dedup(
            store, monkeypatch, REVIEN_SEMANTIC_DEDUP_NEGATION_GUARD="0"
        )
        existing = _seed(store, sem, "enjoys present tense")
        got, is_new = dedup.deduplicate_node(
            _mk_node("reading in the present tense is never enjoyable")
        )
        assert is_new is False
        assert got.node_id == existing.node_id
