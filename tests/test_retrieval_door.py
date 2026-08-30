"""
Retrieval-door leg: vector-union over-fetch (edge-poor non-CONTEXT nodes
outside the old vector top-k become reachable), the ghost-vector delete leak
on fresh-opened indexes, and the honest top_n cap. Diagnosed from per-user
conversational benches against the shipped engine; over-fetch factor 1.0 is
the byte-identical kill switch.
"""

import os
import tempfile

import pytest

from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex

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


def _add(store, label, content, node_type=NodeType.FACT):
    return store.add_node(Node(
        node_type=node_type, label=label, content=content,
        source_type=SourceType.EXTRACTED, confidence=1.0,
    ))


# ── Item: vector-union over-fetch ──────────────────────────────────────

class _RankedEmbedder:
    """Similarity is controlled per registered key: any text containing the
    key embeds to a vector whose cosine against the query axis is the given
    value. Unregistered text lands on an orthogonal axis."""

    def __init__(self, ranks):
        self.ranks = ranks  # {key: cosine_vs_query}

    @property
    def dim(self):
        return 3

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        import math
        out = []
        for t in texts:
            tl = (t or "").lower()
            vec = [0.0, 0.0, 1.0]  # orthogonal default
            if "query-axis" in tl:
                vec = [1.0, 0.0, 0.0]
            else:
                for key, cos in self.ranks.items():
                    if key in tl:
                        vec = [cos, math.sqrt(max(0.0, 1.0 - cos * cos)), 0.0]
                        break
            out.append(vec)
        return out


def _door_setup(store):
    """Two CONTEXT turns hug the query; the edgeless preference sits third —
    outside a top-2 vector fetch, invisible to the old path."""
    emb = _RankedEmbedder({
        "turn one": 0.99,
        "turn two": 0.98,
        "planted preference": 0.95,
    })
    ctx1 = _add(store, "turn one", "verbatim turn one", NodeType.CONTEXT)
    ctx2 = _add(store, "turn two", "verbatim turn two", NodeType.CONTEXT)
    pref = _add(store, "planted preference", "planted preference rule",
                NodeType.PREFERENCE)
    sem = _InMemoryVectorIndex(store, emb)
    sem.reindex_all()
    return pref, sem


class TestVectorUnion:
    def test_edge_poor_preference_reachable_via_overfetch(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC_TOP_K", "2")
        monkeypatch.setenv("REVIEN_VECTOR_OVERFETCH", "3")
        pref, sem = _door_setup(store)
        eng = RetrievalEngine(store, semantic=sem)
        results = eng.recall("query-axis").results
        assert any(r.node_id == pref.node_id for r in results), (
            "non-CONTEXT node outside the vector top-k must union in"
        )

    def test_kill_switch_restores_old_miss(self, store, monkeypatch):
        # Documents the defect the leg fixes: at overfetch=1.0 the CONTEXT
        # turns eat the whole fetch and the preference never surfaces.
        monkeypatch.setenv("REVIEN_SEMANTIC_TOP_K", "2")
        monkeypatch.setenv("REVIEN_VECTOR_OVERFETCH", "1")
        pref, sem = _door_setup(store)
        eng = RetrievalEngine(store, semantic=sem)
        results = eng.recall("query-axis").results
        assert not any(r.node_id == pref.node_id for r in results)

    def test_include_context_skips_overfetch(self, store, monkeypatch):
        # With include_context=True nothing is structurally discarded, so
        # the fetch geometry stays exactly the old one.
        monkeypatch.setenv("REVIEN_SEMANTIC_TOP_K", "2")
        monkeypatch.setenv("REVIEN_VECTOR_OVERFETCH", "3")
        pref, sem = _door_setup(store)
        eng = RetrievalEngine(store, semantic=sem)
        results = eng.recall("query-axis", include_context=True).results
        ids = {r.node_id for r in results}
        assert pref.node_id not in ids

    def test_cosine_floor_gates_tail_admission(self, store, monkeypatch):
        # A tail hit below the (raw-cosine) floor must NOT union in — this
        # is the branch that makes SEMANTIC_SIM_FLOOR measure something.
        monkeypatch.setenv("REVIEN_SEMANTIC_TOP_K", "2")
        monkeypatch.setenv("REVIEN_VECTOR_OVERFETCH", "3")
        monkeypatch.setenv("REVIEN_SEMANTIC_SIM_FLOOR", "0.97")
        pref, sem = _door_setup(store)  # pref cosine 0.95 < 0.97
        eng = RetrievalEngine(store, semantic=sem)
        results = eng.recall("query-axis").results
        assert not any(r.node_id == pref.node_id for r in results)


# ── Item: top_n honest cap ─────────────────────────────────────────────

class TestTopNCap:
    def test_top_n_beyond_20_is_honored(self, store):
        # A hub anchor with 25 spokes: the walk yields 26 candidates, so the
        # only thing between the caller and 25 results is the old silent
        # min(top_n, 20). (The keyword lane's own limit=10 anchor cap is a
        # separate, documented bound — routed around via the hub edges.)
        from revien.graph.schema import Edge, EdgeType
        hub = _add(store, "docker", "docker hub node", NodeType.ENTITY)
        for i in range(25):
            spoke = _add(store, f"note {i}", f"detail {i}")
            store.add_edge(Edge(
                edge_type=EdgeType.RELATED_TO,
                source_node_id=hub.node_id,
                target_node_id=spoke.node_id,
                weight=0.5,
            ))
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        results = eng.recall("docker", top_n=25).results
        assert len(results) >= 25

    def test_cap_at_200_is_loud(self, store, capsys):
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        eng.recall("anything", top_n=500)
        err = capsys.readouterr().err
        assert "top_n=500 capped to 200" in err


# ── Neural gate: explicit opt-in, never dependency-presence ───────────

class TestNeuralGate:
    """An ambient trained model (~/.revien/models) plus an env that happens
    to have sklearn silently reranked per-user stores with a model trained
    on the whole machine's traffic: -20pts generic recall on the per-user
    bench. Neural must activate only on explicit request."""

    def _spy_loop(self, eng):
        calls = []
        class _Spy:
            def log_retrieval(self, **kw):
                calls.append(("log", kw))
            def mark_used(self, node_id, query):
                calls.append(("mark", node_id))
        eng.training_loop = _Spy()
        return calls

    def test_default_is_off(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_NEURAL", raising=False)
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.neural_enabled is False

    def test_env_opts_in(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_NEURAL", "1")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.neural_enabled is True

    def test_explicit_model_dir_is_intent(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_NEURAL", raising=False)
        model_dir = tempfile.mkdtemp()  # tmp_path fixture is broken on this box
        eng = RetrievalEngine(
            store, model_dir=model_dir,
            semantic=SemanticIndex(store, enabled=False),
        )
        assert eng.neural_enabled is True

    def test_no_training_writes_when_off(self, store, monkeypatch):
        monkeypatch.delenv("REVIEN_NEURAL", raising=False)
        node = _add(store, "docker note", "docker detail")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        calls = self._spy_loop(eng)
        resp = eng.recall("docker")
        assert resp.neural_active is False
        eng.mark_used(node.node_id)
        assert calls == [], "gated-off engine must not accumulate signals"

    def test_training_writes_when_on(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_NEURAL", "1")
        node = _add(store, "docker note", "docker detail")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        calls = self._spy_loop(eng)
        eng.recall("docker")
        eng.mark_used(node.node_id)
        assert ("mark", node.node_id) in calls
        assert any(c[0] == "log" for c in calls)


# ── Item: ghost-vector delete leak ─────────────────────────────────────

sqlite_vec = pytest.importorskip(
    "sqlite_vec", reason="ghost-vector repro needs the real vec0 table"
)


class _TinyEmbedder:
    @property
    def dim(self):
        return 4

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _vec_count(store, node_id):
    conn = store._get_conn()
    return conn.execute(
        "SELECT count(*) FROM vec_nodes WHERE node_id = ?", (node_id,)
    ).fetchone()[0]


class TestGhostVectorLeak:
    def test_forget_on_fresh_index_removes_vector(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # Session 1: ingest + embed, then close. Vector lands in vec0.
            s1 = GraphStore(db_path=path)
            sem1 = SemanticIndex(s1, embedder=_TinyEmbedder(), enabled=True)
            node = _add(s1, "secret fact", "the thing to forget")
            assert sem1.index_node(node.node_id, node.label, node.content)
            assert _vec_count(s1, node.node_id) == 1
            s1.close()

            # Session 2: open -> forget -> never search. The fresh index has
            # _table_ready=False; before the fix its delete listener
            # early-returned and the embedding survived the forget forever.
            s2 = GraphStore(db_path=path)
            sem2 = SemanticIndex(s2, embedder=_TinyEmbedder(), enabled=True)
            assert sem2.is_enabled
            ops = GraphOperations(s2)
            ops.forget_node(node.node_id)
            assert s2.get_node(node.node_id) is None
            assert _vec_count(s2, node.node_id) == 0, (
                "right-to-forget must remove the embedding, not just the node"
            )
            s2.close()
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass
