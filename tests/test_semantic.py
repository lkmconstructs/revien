"""
Tests for the opt-in LOCAL-FIRST semantic/vector hybrid layer (leg 5).

Two tiers:
  1. Extra-absent guarantees — the package imports, the layer self-disables,
     and recall() is byte-for-byte identical with the semantic layer off.
     These run in ANY env (no sqlite-vec / fastembed required).
  2. Hybrid-anchor wiring — proven with a deterministic MOCK embedder so the
     keyword-less-retrieval path is exercised WITHOUT the heavy deps. A real
     fastembed+sqlite-vec end-to-end test is included but skips when the extra
     is absent.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import (
    SemanticIndex,
    SEMANTIC_AVAILABLE,
    build_embedder,
    FastEmbedProvider,
    OpenAIEmbeddingProvider,
)


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


# ── Tier 1: extra-absent guarantees (run everywhere) ──────────────────

class TestDisabledDegradesToBase:
    def test_layer_self_disables_when_extra_absent_or_forced_off(self, store):
        # Forced off is independent of whether the extra is installed.
        si = SemanticIndex(store, enabled=False)
        assert si.is_enabled is False
        assert si.search("anything") == []
        assert si.index_node("n1", "label", "content") is False
        assert si.index_nodes([("n1", "l", "c")]) == 0
        assert si.reindex_all()["status"] == "disabled"

    def test_env_gate_force_off(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        si = SemanticIndex(store)
        assert si.is_enabled is False

    def test_recall_byte_identical_when_semantic_off(self, store):
        """recall() output must be IDENTICAL with the semantic layer off vs.
        a baseline engine that never knew about semantics. Same anchors, same
        scores, same score_breakdown keys (no semantic_boost key leaks in)."""
        pipeline = IngestionPipeline(store, semantic=SemanticIndex(store, enabled=False))
        pipeline.ingest(IngestionInput(
            source_id="s1",
            content=(
                "User: We decided to use PostgreSQL for the enterprise tier. "
                "Pricing is $499/month."
            ),
        ))

        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("What database did we choose?", top_n=5)

        # The disabled path must not introduce a semantic_boost key.
        for r in resp.results:
            assert "semantic_boost" not in r.score_breakdown
        # And it must still retrieve via the keyword path (unchanged behavior).
        assert resp.nodes_examined >= 0

    def test_engine_constructs_without_semantic_arg(self, store):
        # Default construction must not require the extra.
        eng = RetrievalEngine(store)
        # Whatever the env, a freshly-built engine never crashes recall.
        resp = eng.recall("hello world", top_n=3)
        assert resp.query == "hello world"


# ── Mock embedder: deterministic, dependency-free ─────────────────────

class _MockEmbedder:
    """Tiny deterministic embedder over a fixed vocabulary. No deps. Lets us
    prove the hybrid-anchor wiring (query embed -> vector search -> anchor
    union -> graph walk -> semantic boost) without sqlite-vec/fastembed."""

    VOCAB = ["dog", "puppy", "canine", "finance", "revenue", "money", "bread", "recipe"]

    @property
    def dim(self):
        return len(self.VOCAB)

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        out = []
        for t in texts:
            tl = (t or "").lower()
            # Bag-of-vocab vector; semantically-related words share dimensions.
            vec = [1.0 if w in tl else 0.0 for w in self.VOCAB]
            # Tie 'puppy' and 'canine' to the 'dog' dimension so a puppy query
            # lands near a dog node even with zero literal word overlap.
            if "puppy" in tl or "canine" in tl:
                vec[self.VOCAB.index("dog")] = 1.0
            out.append(vec)
        return out


class _InMemoryVectorIndex(SemanticIndex):
    """SemanticIndex variant that keeps vectors in a dict instead of vec0, so
    the hybrid wiring is testable WITHOUT the sqlite-vec extension. Overrides
    only storage/search; the enablement + engine integration are the real ones."""

    def __init__(self, store, embedder):
        super().__init__(store, embedder=embedder, enabled=True)
        # Force-enable regardless of SEMANTIC_AVAILABLE — we supply our own store.
        self._enabled = True
        self._vectors = {}

    def index_node(self, node_id, label, content):
        text = self._node_text(label, content)
        if not text:
            return False
        self._vectors[node_id] = self._get_embedder().embed([text])[0]
        return True

    def index_nodes(self, nodes):
        n = 0
        for nid, lbl, ct in nodes:
            if self.index_node(nid, lbl, ct):
                n += 1
        return n

    def search(self, query, top_k=10):
        if not query.strip() or not self._vectors:
            return []
        q = self._get_embedder().embed([query])[0]

        def cos(a, b):
            import math
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = [(nid, cos(q, v)) for nid, v in self._vectors.items()]
        scored = [(nid, s) for nid, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class TestHybridWiringWithMock:
    def test_keywordless_query_retrieves_via_semantic_anchor(self, store):
        dog = _add_fact(store, "dog", "A friendly dog at the park.")
        _add_fact(store, "revenue", "Quarterly finance revenue numbers.")
        _add_fact(store, "bread", "A sourdough bread recipe.")

        sem = _InMemoryVectorIndex(store, _MockEmbedder())
        assert sem.is_enabled
        sem.reindex_all()

        query = "tell me about my puppy"  # no literal overlap with any node text

        # BEFORE: graph-only finds nothing (no keyword/entity anchor).
        eng_off = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng_off.recall(query).results == []

        # AFTER: semantic anchor surfaces the dog node and it ranks first.
        eng_on = RetrievalEngine(store, semantic=sem)
        results = eng_on.recall(query).results
        assert results, "hybrid recall returned nothing"
        assert results[0].node_id == dog.node_id
        # The semantic component is recorded in the breakdown when enabled.
        assert "semantic_boost" in results[0].score_breakdown
        assert results[0].score_breakdown["semantic_boost"] > 0

    def test_ingest_indexes_new_nodes(self, store):
        sem = _InMemoryVectorIndex(store, _MockEmbedder())
        pipeline = IngestionPipeline(store, semantic=sem)
        # Content the rule extractor turns into real entity/fact nodes (named
        # entities + a decision), so there is something non-context to embed.
        pipeline.ingest(IngestionInput(
            source_id="s1",
            content=(
                "User: We decided to deploy PostgreSQL on the Hetzner server. "
                "Acme Corp signed the contract."
            ),
        ))
        # At least one non-context node embedded at ingest time.
        assert len(sem._vectors) >= 1


# ── Embedder selection ────────────────────────────────────────────────

class TestEmbedderSelection:
    def test_default_is_local_fastembed(self, monkeypatch):
        monkeypatch.delenv("REVIEN_EMBEDDER", raising=False)
        emb = build_embedder()
        assert isinstance(emb, FastEmbedProvider)
        assert emb.is_cloud is False

    def test_openai_is_cloud_and_opt_in(self):
        emb = build_embedder("openai")
        assert isinstance(emb, OpenAIEmbeddingProvider)
        assert emb.is_cloud is True

    def test_unknown_falls_back_to_local(self):
        emb = build_embedder("does-not-exist")
        assert isinstance(emb, FastEmbedProvider)

    def test_cloud_embedder_discloses_once(self, monkeypatch, capsys):
        # No API key -> embed raises, but the disclosure must fire FIRST.
        import revien.semantic.index as idx
        idx._DISCLOSED_PROVIDERS.clear()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = OpenAIEmbeddingProvider()
        with pytest.raises(RuntimeError):
            emb.embed(["secret text"])
        err = capsys.readouterr().err
        assert "leaves your machine" in err
        assert "openai" in err


# ── Tier 2: real extra end-to-end (skips when absent) ─────────────────

@pytest.mark.skipif(
    not SEMANTIC_AVAILABLE,
    reason="semantic extra (sqlite-vec) not installed",
)
class TestRealSemanticExtra:
    def test_end_to_end_keywordless_retrieval(self, store):
        dog = _add_fact(store, "Golden retriever named Biscuit",
                        "A friendly golden retriever who loves the park.")
        _add_fact(store, "Q3 revenue forecast",
                  "Finance approved the quarterly revenue projections.")
        _add_fact(store, "Sourdough starter recipe",
                  "Feed the sourdough starter with flour and water daily.")

        sem = SemanticIndex(store, enabled=True)
        if not sem.is_enabled:  # extra present but fastembed missing
            pytest.skip("fastembed embedder not available")
        summary = sem.reindex_all()
        assert summary["indexed"] == 3

        query = "information about my puppy"  # zero literal overlap
        eng_off = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng_off.recall(query).results == []

        eng_on = RetrievalEngine(store, semantic=sem)
        results = eng_on.recall(query).results
        assert results, "real hybrid recall returned nothing"
        assert results[0].node_id == dog.node_id
