"""
Capture leg (P3): deferred embedding + remote-capture token gate.

defer_embed contract: an interactive capture (bookmarklet, phone shortcut)
persists immediately and NEVER waits on a cold embedding model. Queued nodes
become recallable at the next semantic recall (search drains the queue first)
or the ~30s idle sweep, and the recall response says what happened
(semantic_note) instead of silently doing background work. NOTE: a verbatim-
only capture is NOT keyword-anchorable in the gap — keyword anchor search
excludes CONTEXT nodes (engine._keyword_search, exclude_context=True); the
drain-at-search path is what closes the gap, not the keyword path.

Auth contract: loopback capture is unchanged; remote capture is refused
outright unless REVIEN_CAPTURE_TOKEN is configured AND presented as a bearer.
"""

import asyncio
import os
import tempfile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import check_capture_auth, create_app
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex


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


class _MockEmbedder:
    """Bag-of-vocab embedder — deterministic, no model, no network."""

    VOCAB = ["dog", "revenue", "bread", "postgres", "capture", "sunset"]

    @property
    def dim(self):
        return len(self.VOCAB)

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        return [
            [1.0 if w in (t or "").lower() else 0.0 for w in self.VOCAB]
            for t in texts
        ]


class _QueueTestIndex(SemanticIndex):
    """Dict-backed vectors so the REAL pending-queue plumbing (defer_nodes /
    pending_count / drain_pending / pending_note) is exercised without the
    sqlite-vec extension. Only vector storage/search are overridden."""

    def __init__(self, store, embedder=None):
        super().__init__(store, embedder=embedder or _MockEmbedder(), enabled=True)
        self._enabled = True  # force past SEMANTIC_AVAILABLE
        self._vectors = {}
        self._register_store_listener()

    def remove_node(self, node_id):
        self._vectors.pop(node_id, None)

    def index_node(self, node_id, label, content):
        text = self._node_text(label, content)
        if not text:
            return False
        self._vectors[node_id] = self._get_embedder().embed([text])[0]
        return True

    def index_nodes(self, nodes):
        return sum(
            1 for nid, lbl, ct in nodes if self.index_node(nid, lbl, ct)
        )

    def search(self, query, top_k=10):
        # Mirror the base class's drain-at-search hook.
        self._last_search_drained = self.drain_pending()
        if not query.strip() or not self._vectors:
            return []
        import math

        q = self._get_embedder().embed([query])[0]

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = [(nid, cos(q, v)) for nid, v in self._vectors.items()]
        scored = [(nid, s) for nid, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


CAPTURE_TEXT = "User: Note to self — the sunset capture idea for the dog park photo essay."


# ── Deferred embedding ─────────────────────────────────────


class TestDeferEmbed:
    def test_defer_queues_instead_of_embedding(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        out = pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        assert out.context_node_id
        assert sem._vectors == {}, "defer_embed must not embed inline"
        assert sem.pending_count() >= 1

    def test_default_path_unchanged(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(IngestionInput(source_id="desk", content=CAPTURE_TEXT))
        assert len(sem._vectors) >= 1, "default ingest embeds inline"
        assert sem.pending_count() == 0

    def test_drain_embeds_and_clears_queue(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        queued = sem.pending_count()
        drained = sem.drain_pending()
        assert drained == queued > 0
        assert sem.pending_count() == 0
        assert len(sem._vectors) >= 1

    def test_recall_drains_and_reports_in_semantic_note(self, store):
        """Capture on the phone, ask at the desk — ONE query, and the response
        says the drain happened rather than doing it silently."""
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        eng = RetrievalEngine(store, semantic=sem)
        # include_context: a phone capture IS a verbatim turn — the context
        # node is the capture. Without it only extracted nodes would surface.
        resp = eng.recall("sunset dog photo", include_context=True)
        assert resp.results, "deferred capture must be recallable on first query"
        assert resp.semantic_active is True
        assert resp.semantic_note and "deferred capture" in resp.semantic_note
        # Once drained, the note goes away — shape returns to the common case.
        resp2 = eng.recall("sunset dog photo", include_context=True)
        assert resp2.semantic_note is None

    def test_pending_note_reports_backlog_without_drain(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        note = sem.pending_note()
        assert note and "pending embedding" in note

    def test_queue_survives_process_restart(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        # A fresh index over the same store (new process) sees the queue.
        sem2 = _QueueTestIndex(store)
        assert sem2.pending_count() >= 1
        assert sem2.drain_pending() >= 1
        assert sem2.pending_count() == 0

    def test_deleted_node_ghost_is_cleared_not_embedded(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        out = pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        # Delete everything that was queued, then drain: nothing embedded,
        # nothing left behind to re-drain forever.
        conn = store._get_conn()
        ids = [
            r[0]
            for r in conn.execute(
                f"SELECT node_id FROM {SemanticIndex.PENDING_TABLE}"
            ).fetchall()
        ]
        for nid in ids:
            store.delete_node(nid)
        assert sem.drain_pending() == 0
        assert sem.pending_count() == 0
        assert sem._vectors == {}

    def test_disabled_layer_defers_nothing(self, store):
        sem = SemanticIndex(store, enabled=False)
        assert sem.defer_nodes([("n1", "l", "c")]) == 0
        assert sem.pending_count() == 0
        assert sem.drain_pending() == 0
        assert sem.pending_note() is None

    def test_idle_sweep_drains(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        pipeline.ingest(
            IngestionInput(source_id="phone", content=CAPTURE_TEXT, defer_embed=True)
        )
        scheduler = SyncScheduler(pipeline=pipeline)
        drained = asyncio.run(scheduler.drain_pending_embeds())
        assert drained >= 1
        assert sem.pending_count() == 0


# ── Capture token gate ─────────────────────────────────────


class TestCaptureAuth:
    def test_loopback_passes_without_token(self, monkeypatch):
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        for host in ("127.0.0.1", "::1", "localhost", "testclient", "", None):
            check_capture_auth(host, "")  # must not raise

    def test_loopback_passes_even_when_token_configured(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        check_capture_auth("127.0.0.1", "")  # local adapters unchanged

    def test_remote_refused_when_no_token_configured(self, monkeypatch):
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        with pytest.raises(HTTPException) as e:
            check_capture_auth("100.64.0.7", "")
        assert e.value.status_code == 403

    def test_remote_with_correct_bearer_passes(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        check_capture_auth("100.64.0.7", "Bearer s3cret")  # must not raise

    def test_remote_with_wrong_or_missing_bearer_401(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        for header in ("", "Bearer wrong", "s3cret", "bearer s3cret "):
            with pytest.raises(HTTPException) as e:
                check_capture_auth("100.64.0.7", header)
            assert e.value.status_code == 401


# ── Endpoint round-trip ────────────────────────────────────


class TestIngestEndpointCapturePath:
    @pytest.fixture
    def client(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        app = create_app(db_path=path)
        with TestClient(app) as c:
            yield c
        try:
            os.unlink(path)
        except PermissionError:  # pragma: no cover - Windows WAL handle race
            pass

    def test_defer_embed_accepted_and_persists(self, client):
        resp = client.post(
            "/v1/ingest",
            json={
                "source_id": "browser",
                "content": CAPTURE_TEXT,
                "defer_embed": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["context_node_id"]
        assert data["total_nodes_in_graph"] >= 1

    def test_local_ingest_unaffected_by_configured_token(self, client, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        resp = client.post(
            "/v1/ingest",
            json={"source_id": "local", "content": CAPTURE_TEXT},
        )
        assert resp.status_code == 200
