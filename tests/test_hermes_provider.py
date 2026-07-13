"""
Hermes memory-provider tests.

Two layers, matching the integration's two dependency surfaces:

  1. Revien-side mapping — the part that must be correct regardless of whether
     the Hermes SDK is installed. Built WITHOUT the SDK by constructing the
     provider's in-process stack directly (object.__new__ bypasses the SDK-gated
     __init__) and driving the real hook methods: prefetch -> recall, sync_turn
     -> deferred ingest, get_tool_schemas / handle_tool_call. This is the leg
     that ships, so it is tested on this box (where hermes-agent is absent).

  2. SDK-present paths — construction, register(ctx), is_available — guarded by
     importorskip("agent.memory_provider"); they skip when the SDK is absent.

Temp-HOME pattern (Path.home monkeypatch + tempfile) per the box's broken
pytest tmp_path basetemp. conftest forces REVIEN_SEMANTIC=0 / REVIEN_RERANK=0,
so the default stack is graph-only and offline; the defer test injects an
explicitly-enabled queue index to exercise the defer_embed path.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex
from revien.hermes_provider import (
    HERMES_AVAILABLE,
    RevienMemoryProvider,
    _resolve_db_path,
    register,
)

# Reuse the dict-backed queue index that exercises the REAL pending-queue
# plumbing (defer_nodes / pending_count / drain_pending) without sqlite-vec.
try:  # pytest prepend-import mode puts tests/ on sys.path
    from test_capture import _QueueTestIndex
except ImportError:  # pragma: no cover - fallback if collected as a package
    from tests.test_capture import _QueueTestIndex


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def temp_home(monkeypatch):
    """Temp HOME so any ~/.revien resolution is sandboxed (basetemp is broken)."""
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        yield home


@pytest.fixture
def store(temp_home):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _bypass_provider(store, semantic):
    """A genuine RevienMemoryProvider with its stack prewired, built WITHOUT
    invoking __init__ (which is SDK-gated). The real hook methods run against
    this — _ensure_stack sees _store set and no-ops, so nothing is rebuilt."""
    prov = object.__new__(RevienMemoryProvider)
    prov._db_path = store.db_path
    prov._session_id = ""
    prov._store = store
    prov._pipeline = IngestionPipeline(store, semantic=semantic)
    prov._engine = RetrievalEngine(store, semantic=semantic)
    import queue
    import threading
    prov._sync_queue = queue.Queue()
    prov._sync_worker = None
    prov._worker_lock = threading.Lock()
    return prov


# ── Revien-side mapping (no Hermes SDK required) ──────────


class TestPrefetchMapping:
    def test_prefetch_returns_seeded_content(self, store):
        """prefetch -> engine.recall: seeded memory surfaces before a turn."""
        semantic = SemanticIndex(store)  # disabled under REVIEN_SEMANTIC=0
        prov = _bypass_provider(store, semantic)
        # A declarative fact yields a recallable non-CONTEXT entity node; recall
        # (graph-only keyword/anchor path) finds it. prefetch injects no context
        # nodes (include_context defaults off), so seed a fact, not a bare turn.
        prov._pipeline.ingest(
            IngestionInput(
                source_id="seed",
                content="We chose Postgres over SQLite because of concurrent writes.",
                content_type="note",
            )
        )
        block = prov.prefetch("Postgres database decision")
        assert isinstance(block, str)
        assert "Postgres" in block, block

    def test_prefetch_empty_query_returns_empty(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        assert prov.prefetch("") == ""
        assert prov.prefetch("   ") == ""


class TestSyncTurnMapping:
    def test_sync_turn_defers_embed_and_persists(self, store):
        """sync_turn -> pipeline.ingest(defer_embed=True) on a daemon thread:
        the turn is persisted but NOT embedded inline — it lands in the pending
        queue, mirroring the P3 capture contract."""
        sem = _QueueTestIndex(store)  # explicitly enabled queue index
        prov = _bypass_provider(store, sem)
        before = store.count_nodes()

        prov.sync_turn(
            "Let's switch the queue backend to Redis Streams.",
            "Done — Redis Streams it is.",
        )
        # Non-blocking: the write runs on the serial worker. Flush the queue so
        # the assertions are deterministic.
        prov._flush_sync()

        assert store.count_nodes() > before, "turn must be persisted"
        assert sem._vectors == {}, "defer_embed must NOT embed inline"
        assert sem.pending_count() >= 1, "turn must be queued for embedding"

    def test_sync_turn_serializes_concurrent_turns(self, store):
        """The race the review caught: many back-to-back turns must all persist
        with no interleaved/lost writes. One serial worker guarantees it."""
        sem = _QueueTestIndex(store)
        prov = _bypass_provider(store, sem)
        before = store.count_nodes()
        for i in range(12):
            prov.sync_turn(f"Decision {i}: option {i} chosen.", f"Ack {i}.")
        prov._flush_sync()
        # Every turn ingested (distinct content dedups to distinct nodes); none
        # dropped to an IntegrityError from an interleaved dedup race.
        assert store.count_nodes() > before
        assert sem.pending_count() >= 12

    def test_sync_turn_empty_is_noop(self, store):
        sem = _QueueTestIndex(store)
        prov = _bypass_provider(store, sem)
        before = store.count_nodes()
        prov.sync_turn("", "")
        assert prov._sync_queue.empty()
        assert store.count_nodes() == before

    def test_on_session_end_drains_pending(self, store):
        """on_session_end flushes deferred embeds (no second summary ingest)."""
        sem = _QueueTestIndex(store)
        prov = _bypass_provider(store, sem)
        prov.sync_turn("Pin the tokenizer to 0.15.", "Pinned tokenizer==0.15.")
        prov._flush_sync()
        assert sem.pending_count() >= 1
        prov.on_session_end(messages=[])
        assert sem.pending_count() == 0, "session end must drain the queue"
        assert len(sem._vectors) >= 1, "drained nodes are now embedded"


class TestToolSchemas:
    def test_schemas_well_formed(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        schemas = prov.get_tool_schemas()
        assert isinstance(schemas, list) and len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"revien_recall", "revien_store"}
        for s in schemas:
            assert s["description"]
            params = s["parameters"]
            assert params["type"] == "object"
            assert "properties" in params and params["properties"]
            assert isinstance(params.get("required", []), list)
        recall = next(s for s in schemas if s["name"] == "revien_recall")
        assert "query" in recall["parameters"]["properties"]
        store_tool = next(s for s in schemas if s["name"] == "revien_store")
        assert "content" in store_tool["parameters"]["properties"]

    def test_handle_tool_call_returns_json_string(self, store):
        """The ABC contracts handle_tool_call -> str (not dict). Result must be
        a JSON string that parses back to the payload."""
        prov = _bypass_provider(store, SemanticIndex(store))
        raw = prov.handle_tool_call(
            "revien_store",
            {"content": "The staging server is blue.example.com."},
        )
        assert isinstance(raw, str), "Hermes ABC requires a string result"
        stored = json.loads(raw)
        assert stored["context_node_id"]
        assert stored["total_nodes_in_graph"] >= 1

        raw2 = prov.handle_tool_call(
            "revien_recall", {"query": "staging server", "top_n": 5}
        )
        assert isinstance(raw2, str)
        recalled = json.loads(raw2)
        assert recalled["query"] == "staging server"
        assert isinstance(recalled["results"], list)
        assert "semantic_active" in recalled

    def test_handle_unknown_tool(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        raw = prov.handle_tool_call("revien_nope", {})
        assert isinstance(raw, str)
        out = json.loads(raw)
        assert "error" in out


class TestConfigAndPromptMapping:
    def test_config_schema_shape(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        schema = prov.get_config_schema()
        assert isinstance(schema, list) and schema
        field = schema[0]
        assert field["key"] == "db_path"
        assert "description" in field
        # Local-first: no secret token field (in-process, no network).
        assert not any(f.get("secret") for f in schema)

    def test_system_prompt_block_mentions_tools(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        block = prov.system_prompt_block()
        assert "revien_recall" in block and "revien_store" in block

    def test_name_is_revien(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        assert prov.name == "revien"

    def test_is_available_no_network(self, store):
        prov = _bypass_provider(store, SemanticIndex(store))
        assert prov.is_available() is True


class TestDbResolution:
    def test_explicit_wins(self):
        assert _resolve_db_path("/tmp/x.db") == "/tmp/x.db"

    def test_env_used_when_no_arg(self, temp_home, monkeypatch):
        monkeypatch.setenv("REVIEN_DB", "/tmp/env.db")
        assert _resolve_db_path(None) == "/tmp/env.db"

    def test_default_under_revien_home(self, temp_home, monkeypatch):
        monkeypatch.delenv("REVIEN_DB", raising=False)
        path = _resolve_db_path(None)
        assert path.endswith(os.path.join(".revien", "revien.db"))


# ── Import guard (SDK absent) ─────────────────────────────


@pytest.mark.skipif(
    HERMES_AVAILABLE, reason="hermes-agent installed — stub path not active"
)
def test_construction_without_sdk_raises():
    """Without the SDK, constructing the provider must fail with a clear
    ImportError (langchain-adapter stub pattern), not an obscure AttributeError."""
    with pytest.raises(ImportError, match="hermes-agent is required"):
        RevienMemoryProvider()


# ── SDK-present paths (skip when absent) ──────────────────


class TestWithHermesSDK:
    """Paths that need the real ABC/loader. Skipped on this box (no SDK)."""

    def setup_method(self):
        pytest.importorskip("agent.memory_provider")

    def test_full_construction_and_initialize(self, store):
        prov = RevienMemoryProvider(db_path=store.db_path)
        assert prov.is_available() is True
        prov.initialize(session_id="sess-1")
        assert prov.name == "revien"
        assert prov.get_tool_schemas()
        prov.shutdown()

    def test_register_registers_one_provider(self):
        captured = []

        class _Ctx:
            def register_memory_provider(self, provider):
                captured.append(provider)

        register(_Ctx())
        assert len(captured) == 1
        assert isinstance(captured[0], RevienMemoryProvider)
