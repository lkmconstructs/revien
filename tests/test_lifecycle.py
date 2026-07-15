"""
Lifecycle / close-path tests (repair leg R2).

One acceptance criterion, applied literally everywhere: after the close or
exit under test, ``os.unlink(<temp db>)`` MUST succeed. On Windows an open
SQLite handle keeps the file locked (WinError 32), so the unlink succeeding
IS the assertion that the handle was released. No PermissionError guard
appears anywhere in this module — a guard here would mask the exact bug
these tests exist to pin.

Covered close paths:
  - FastAPI lifespan teardown (create_app + TestClient exit)
  - daemon-ish teardown (create_app + scheduler attached + lifespan shutdown)
  - RevienDaemon.close() (idempotent)
  - MCP own-stack close (build_mcp_server(db_path=...) + close_mcp_server)
  - OpenAI / Ollama / LangChain adapter close() and context-manager exit
  - Hermes provider shutdown (existing behavior — pinned)
  - GraphStore double-close no-op + close-under-contention
"""

import asyncio
import json
import os
import queue
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

from revien.daemon.daemon import RevienDaemon
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.store import GraphStore


@pytest.fixture
def db_path():
    """Temp db whose teardown unlink is UNGUARDED — if a close path under
    test leaked the handle, the fixture itself fails the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ── GraphStore close contract ─────────────────────────────


class TestStoreClose:
    def test_close_releases_db_file(self, db_path):
        s = GraphStore(db_path=db_path)
        s.count_nodes()
        s.close()
        os.unlink(db_path)

    def test_double_close_is_noop(self, db_path):
        s = GraphStore(db_path=db_path)
        s.count_nodes()
        s.close()
        s.close()  # must not raise
        os.unlink(db_path)

    def test_use_after_close_raises(self, db_path):
        """Closed is closed: any use after close() raises instead of
        silently reopening the connection."""
        s = GraphStore(db_path=db_path)
        s.close()
        with pytest.raises(RuntimeError, match="closed"):
            s.count_nodes()
        os.unlink(db_path)

    def test_use_after_close_does_not_resurrect_handle(self, db_path):
        """The structural guarantee: a post-close call must NOT recreate the
        sqlite handle — the unlink after the attempted use is the proof."""
        s = GraphStore(db_path=db_path)
        s.count_nodes()
        s.close()
        for attempt in (s.count_nodes, s.count_edges):
            with pytest.raises(RuntimeError, match="closed"):
                attempt()
        assert s._conn is None  # nothing resurrected
        os.unlink(db_path)  # file handle really is gone

    def test_close_waits_for_lock_holder(self, db_path):
        """close() acquires the store lock — a close racing another thread's
        work blocks until that thread releases, never yanks the connection
        out from under it."""
        s = GraphStore(db_path=db_path)
        holding = threading.Event()
        release = threading.Event()

        def hold():
            with s._lock:
                holding.set()
                release.wait(timeout=10)

        holder = threading.Thread(target=hold)
        holder.start()
        assert holding.wait(timeout=10)

        closer = threading.Thread(target=s.close)
        closer.start()
        closer.join(timeout=0.2)
        assert closer.is_alive(), "close() must block while the lock is held"

        release.set()
        closer.join(timeout=10)
        assert not closer.is_alive()
        holder.join(timeout=10)
        os.unlink(db_path)


# ── FastAPI lifespan teardown ─────────────────────────────


class TestLifespanTeardown:
    def test_testclient_exit_releases_db(self, db_path):
        """The canonical pattern every API test uses: create_app ->
        TestClient -> exit -> unlink. The lifespan shutdown must close the
        store or Windows holds the file."""
        app = create_app(db_path=db_path)
        with TestClient(app) as c:
            resp = c.post("/v1/ingest", json={
                "source_id": "lifecycle-test",
                "content": "We decided the daemon must release its database on shutdown.",
            })
            assert resp.status_code == 200
            assert c.get("/v1/health").json()["status"] == "healthy"
        os.unlink(db_path)

    def test_daemonish_teardown_with_scheduler_attached(self, db_path):
        """The daemon's shape: create_app, attach a scheduler on app.state,
        run, shut down. The lifespan STARTUP must start the scheduler inside
        the running event loop (AsyncIOScheduler.start() outside one raises
        on py3.13 — the daemon's old pre-uvicorn start), and the teardown
        must stop it and close the store."""
        app = create_app(db_path=db_path)
        scheduler = SyncScheduler(pipeline=app.state.pipeline)
        app.state.scheduler = scheduler
        assert scheduler.is_running is False  # not started before the app runs
        with TestClient(app) as c:
            # Lifespan startup ran inside the app's loop — the scheduler is
            # genuinely running, so the stop assertion below is real.
            assert scheduler.is_running is True
            assert c.post("/v1/sync").status_code == 200
        assert scheduler.is_running is False
        os.unlink(db_path)

    def test_jobs_queued_before_start_register_at_lifespan_startup(self, db_path):
        """The daemon queues its dream job before uvicorn runs; the lifespan
        startup's scheduler.start() must register it for real."""
        app = create_app(db_path=db_path)
        scheduler = SyncScheduler(pipeline=app.state.pipeline)
        app.state.scheduler = scheduler
        assert scheduler.add_interval_job("lifecycle_probe", lambda: None, hours=1.0)
        with TestClient(app):
            assert scheduler._scheduler.get_job("lifecycle_probe") is not None
        os.unlink(db_path)

    def test_second_teardown_is_noop(self, db_path):
        """Lifespan cleanup pieces are idempotent — running them again after
        a TestClient exit must not raise."""
        app = create_app(db_path=db_path)
        scheduler = SyncScheduler(pipeline=app.state.pipeline)
        app.state.scheduler = scheduler
        with TestClient(app):
            pass
        # Same calls the lifespan teardown just made:
        scheduler.stop()
        app.state.store.close()
        os.unlink(db_path)


# ── RevienDaemon.close() ──────────────────────────────────


class TestDaemonClose:
    def _daemonish(self, db_path):
        """A RevienDaemon wired the way start() wires it, minus uvicorn."""
        daemon = RevienDaemon(db_path=db_path)
        daemon._app = create_app(db_path=db_path)
        daemon._scheduler = SyncScheduler(pipeline=daemon._app.state.pipeline)
        daemon._app.state.scheduler = daemon._scheduler
        return daemon

    def test_close_releases_db(self, db_path):
        daemon = self._daemonish(db_path)
        daemon.close()
        os.unlink(db_path)

    def test_close_is_idempotent(self, db_path):
        daemon = self._daemonish(db_path)
        daemon.close()
        daemon.close()  # must not raise
        os.unlink(db_path)

    def test_close_before_start_is_safe(self):
        # Nothing built yet — close() on a never-started daemon is a no-op.
        RevienDaemon(db_path="unused.db").close()


# ── MCP own-stack close ───────────────────────────────────


class TestMCPOwnStackClose:
    @pytest.fixture(autouse=True)
    def _require_mcp(self):
        pytest.importorskip(
            "mcp", reason="mcp extra not installed (pip install revien[mcp])"
        )

    def test_close_mcp_server_releases_db(self, db_path):
        from revien.mcp_server import build_mcp_server, close_mcp_server

        server = build_mcp_server(db_path=db_path)
        # Exercise the stack through a real tool call so the connection is
        # demonstrably live before the close.
        res = asyncio.run(server.call_tool("revien_store", {
            "content": "Decision: lifecycle leg closes every store it opens.",
        }))
        content = res[0] if isinstance(res, tuple) else res
        assert json.loads(content[0].text)["total_nodes_in_graph"] >= 1

        close_mcp_server(server)
        close_mcp_server(server)  # double close is a no-op
        os.unlink(db_path)

    def test_ride_along_mode_owns_nothing(self, db_path):
        """engine/pipeline injected: the server must NOT claim the store —
        close_mcp_server is a no-op and the daemon's stack stays usable."""
        from revien.mcp_server import build_mcp_server, close_mcp_server

        app = create_app(db_path=db_path)
        server = build_mcp_server(
            engine=app.state.engine, pipeline=app.state.pipeline
        )
        assert server._revien_store is None
        close_mcp_server(server)  # no-op, must not close the daemon's store
        assert app.state.store.count_nodes() == 0  # store still usable
        app.state.store.close()
        os.unlink(db_path)

    def test_build_failure_closes_owned_store(self, db_path, monkeypatch):
        """If wiring fails AFTER the own-stack GraphStore opened, the
        factory must close it before re-raising — the unlink is the proof
        the handle didn't leak."""
        import revien.semantic.index as semantic_index

        def boom(store):
            raise RuntimeError("semantic wiring exploded")

        monkeypatch.setattr(semantic_index, "SemanticIndex", boom)
        from revien.mcp_server import build_mcp_server

        with pytest.raises(RuntimeError, match="semantic wiring exploded"):
            build_mcp_server(db_path=db_path)
        os.unlink(db_path)


# ── Adapters ──────────────────────────────────────────────


class TestOpenAIAdapterClose:
    def test_close_releases_db(self, db_path):
        from revien.adapters.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(graph_path=db_path)
        assert adapter.store.count_nodes() == 0
        adapter.close()
        adapter.close()  # double close is a no-op
        os.unlink(db_path)

    def test_context_manager_releases_db(self, db_path):
        from revien.adapters.openai_adapter import OpenAIAdapter

        with OpenAIAdapter(graph_path=db_path) as adapter:
            assert adapter.store.count_nodes() == 0
        os.unlink(db_path)


class TestOllamaAdapterClose:
    def test_close_releases_db(self, db_path):
        from revien.adapters.ollama_adapter import OllamaAdapter

        adapter = OllamaAdapter(graph_path=db_path)
        assert adapter.store.count_nodes() == 0
        adapter.close()
        adapter.close()  # double close is a no-op
        os.unlink(db_path)

    def test_context_manager_releases_db(self, db_path):
        from revien.adapters.ollama_adapter import OllamaAdapter

        with OllamaAdapter(graph_path=db_path) as adapter:
            assert adapter.store.count_nodes() == 0
        os.unlink(db_path)


class TestLangChainAdapterClose:
    @pytest.fixture(autouse=True)
    def _require_langchain(self):
        pytest.importorskip(
            "langchain_core", reason="langchain-core not installed"
        )

    def test_close_releases_db(self, db_path):
        from revien.adapters.langchain_adapter import RevienMemory

        memory = RevienMemory(graph_path=db_path)
        assert memory.load_memory_variables({"input": "anything"}) == {"history": ""}
        memory.close()
        memory.close()  # double close is a no-op
        os.unlink(db_path)

    def test_context_manager_releases_db(self, db_path):
        from revien.adapters.langchain_adapter import RevienMemory

        with RevienMemory(graph_path=db_path) as memory:
            memory.load_memory_variables({"input": "anything"})
        os.unlink(db_path)


# ── Hermes provider (existing behavior — pinned) ──────────


class TestHermesShutdown:
    def _bypass_provider(self, db_path):
        """Provider with its stack prewired, built without the SDK-gated
        __init__ (same pattern as tests/test_hermes_provider.py)."""
        from revien.hermes_provider import RevienMemoryProvider
        from revien.ingestion.pipeline import IngestionPipeline
        from revien.retrieval.engine import RetrievalEngine
        from revien.semantic.index import SemanticIndex

        store = GraphStore(db_path=db_path)
        semantic = SemanticIndex(store)
        prov = object.__new__(RevienMemoryProvider)
        prov._db_path = db_path
        prov._session_id = ""
        prov._store = store
        prov._pipeline = IngestionPipeline(store, semantic=semantic)
        prov._engine = RetrievalEngine(store, semantic=semantic)
        prov._sync_queue = queue.Queue()
        prov._sync_worker = None
        prov._worker_lock = threading.Lock()
        return prov

    def test_shutdown_releases_db(self, db_path):
        prov = self._bypass_provider(db_path)
        assert prov._store.count_nodes() == 0
        prov.shutdown()
        prov.shutdown()  # double shutdown is a no-op
        os.unlink(db_path)
