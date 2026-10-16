"""
Origin Layer (WS0) tests — Leg A.

Covers: derive_origin's source_id -> Origin convention table (every adapter
convention + unknown), the 003 migration's column-add + backfill (idempotent),
the pipeline's stamp loop (every produced node, including CONTEXT, and the
fallback-derive-from-source_id path when a caller omits origin_runtime),
export/import round-trip, and that every adapter puts the expected origin
values on its IngestionInput/Node output — unit-level, no network.
"""

import asyncio
import importlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.codex import CodexAdapter
from revien.adapters.file_watcher import FileWatcherAdapter
from revien.adapters.generic_api import GenericAPIAdapter
from revien.adapters.obsidian import ObsidianVaultAdapter
from revien.adapters.openai_adapter import OpenAIAdapter
from revien.adapters.ollama_adapter import OllamaAdapter
from revien.graph.origin import Origin, RUNTIMES, SOURCES, derive_origin, validate_origin
from revien.graph.schema import Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


@pytest.fixture
def pipeline(store):
    return IngestionPipeline(store)


# ── derive_origin: the convention table, plus unknown ──────────────────────


class TestDeriveOrigin:
    def test_claude_code(self):
        assert derive_origin("claude-code:Fernweh-Core:sess-1") == Origin(
            "claude-code", "live", "Fernweh-Core", "sess-1"
        )

    def test_codex(self):
        assert derive_origin("codex:Fernweh-Core:rollout-abc") == Origin(
            "codex", "live", "Fernweh-Core", "rollout-abc"
        )

    def test_hermes(self):
        assert derive_origin("hermes") == Origin("hermes", "live", None, None)

    def test_openai(self):
        assert derive_origin("openai:conversation:conv_abc123") == Origin(
            "openai", "import", None, "conv_abc123"
        )

    def test_obsidian_vault(self):
        assert derive_origin("vault:notes/Mara.md#intro") == Origin(
            "obsidian", "vault", None, None
        )

    def test_file_watcher(self):
        assert derive_origin("file:theo-notes.txt") == Origin(
            "file", "watch", None, None
        )

    def test_generic_api(self):
        assert derive_origin("api:https://sam.example.com/feed") == Origin(
            "api", "api", None, None
        )

    def test_ollama_history(self):
        assert derive_origin("ollama_history") == Origin("ollama", "live", None, None)

    def test_ollama_chat(self):
        assert derive_origin("ollama_chat") == Origin("ollama", "live", None, None)

    def test_langchain_bare(self):
        assert derive_origin("langchain") == Origin("langchain", "live", None, None)

    def test_langchain_session_scoped_is_unrecognized(self):
        # A caller-chosen session_scope has no recognizable prefix — honest
        # unknown, never a guess.
        assert derive_origin("my-custom-session-42") == Origin(None, None, None, None)

    def test_unknown_prefix(self):
        assert derive_origin("mystery:thing") == Origin(None, None, None, None)

    def test_empty_string(self):
        assert derive_origin("") == Origin(None, None, None, None)

    def test_claude_code_missing_session_segment(self):
        # Malformed but prefixed: project present, session absent.
        assert derive_origin("claude-code:Fernweh-Core") == Origin(
            "claude-code", "live", "Fernweh-Core", None
        )


# ── Migration 003: column add + backfill, idempotent ────────────────────────


def _load_migration():
    return importlib.import_module("revien.graph.migrations.003_origin_layer")


class TestMigration003:
    def test_backfill_on_pre_003_db(self, store):
        """Build a db with the CURRENT store (so it already has the origin
        columns), then simulate a legacy pre-003 row by NULLing its origin
        columns via raw SQL. migrate() must repopulate it from source_id;
        running it again must report 0 backfilled."""
        node = Node(
            node_type=NodeType.CONTEXT,
            label="legacy session",
            content="User: We decided to use SQLite.\nAssistant: Noted.",
            source_id="claude-code:Fernweh-Core:sess-legacy",
        )
        store.add_node(node)
        store.close()

        # Simulate a pre-003 row: NULL the origin columns directly.
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            "UPDATE nodes SET origin_runtime=NULL, origin_source=NULL, "
            "project_key=NULL, session_key=NULL WHERE node_id=?",
            (node.node_id,),
        )
        conn.commit()
        conn.close()

        migrate = _load_migration().migrate
        summary = migrate(store.db_path)
        assert summary["backfilled"] == 1

        conn = sqlite3.connect(store.db_path)
        row = conn.execute(
            "SELECT origin_runtime, origin_source, project_key, session_key "
            "FROM nodes WHERE node_id=?",
            (node.node_id,),
        ).fetchone()
        conn.close()
        assert row == ("claude-code", "live", "Fernweh-Core", "sess-legacy")

        # Idempotent: nothing left to backfill.
        summary2 = migrate(store.db_path)
        assert summary2["backfilled"] == 0

    def test_columns_added_on_a_truly_pre_003_table(self):
        """A raw sqlite db with a pre-origin-layer nodes table (every column
        this store has ever had EXCEPT the four origin columns) gets them
        added AND populated in one pass. migrate() now delegates to
        GraphStore's own migration chain, so the fixture must be a table
        that store can actually open — a table missing columns from
        earlier migrations (confidence, modality, ...) is not a state this
        store produces and isn't what migration 003 is responsible for."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            conn.execute(
                """CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    source_type TEXT DEFAULT 'inferred',
                    confidence REAL DEFAULT 0.5,
                    pinned INTEGER DEFAULT 0,
                    confidence_set_at TEXT,
                    confidence_set_by TEXT DEFAULT '',
                    source_context TEXT DEFAULT '',
                    last_referenced TEXT,
                    invalidated_at TEXT,
                    source_modality TEXT DEFAULT 'text',
                    answerable_by_text INTEGER DEFAULT 1,
                    vision_processed INTEGER DEFAULT 0,
                    recorded_at TEXT,
                    event_time_start TEXT,
                    event_time_end TEXT,
                    event_time_granularity TEXT,
                    event_time_confidence REAL,
                    event_time_text TEXT DEFAULT '',
                    valid_from TEXT,
                    valid_until TEXT
                )"""
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO nodes (node_id, node_type, label, content, source_id, "
                "created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("n1", "context", "note", "body", "vault:notes/Theo.md#top", now_iso, now_iso),
            )
            conn.commit()
            conn.close()

            migrate = _load_migration().migrate
            summary = migrate(path)
            assert set(summary["columns_added"]) == {
                "origin_runtime", "origin_source", "project_key", "session_key",
            }
            assert summary["backfilled"] == 1

            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT origin_runtime, origin_source, project_key, session_key "
                "FROM nodes WHERE node_id='n1'"
            ).fetchone()
            conn.close()
            assert row == ("obsidian", "vault", None, None)
        finally:
            os.unlink(path)

    def test_unrecognized_source_id_stays_null(self, store):
        node = Node(
            node_type=NodeType.FACT,
            label="mystery",
            content="something",
            source_id="totally-unrecognized-source",
        )
        store.add_node(node)
        store.close()

        migrate = _load_migration().migrate
        summary = migrate(store.db_path)
        assert summary["backfilled"] == 0

        conn = sqlite3.connect(store.db_path)
        row = conn.execute(
            "SELECT origin_runtime FROM nodes WHERE node_id=?", (node.node_id,)
        ).fetchone()
        conn.close()
        assert row == (None,)


# ── store._ensure_db in-place backfill (same guard, runs on connect) ───────


class TestStoreEnsureDbBackfill:
    def test_reopening_pre_3_store_backfills_legacy_rows(self):
        """A row written with NULL origin columns (bypassing the pipeline
        stamp) on a db that hasn't reached user_version 3 yet gets
        backfilled the next time a GraphStore opens the db — _ensure_db's
        guard runs the same derive_origin backfill as the standalone
        migration. (F8: once user_version reaches 3, later opens skip this
        scan for performance — see TestOriginBackfillUserVersionMarker.)"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = GraphStore(db_path=path)
            node = Node(
                node_type=NodeType.CONTEXT,
                label="legacy",
                content="body",
                source_id="codex:Fernweh-Core:rollout-legacy",
            )
            s.add_node(node)
            s.close()

            conn = sqlite3.connect(path)
            conn.execute(
                "UPDATE nodes SET origin_runtime=NULL, origin_source=NULL, "
                "project_key=NULL, session_key=NULL WHERE node_id=?",
                (node.node_id,),
            )
            # Simulate a genuinely pre-origin-layer db: user_version 0 (a
            # fresh GraphStore() sets it to 3 on the very first open, since
            # nothing needed backfilling — a real legacy db never had this
            # PRAGMA touched at all).
            conn.execute("PRAGMA user_version = 0")
            conn.commit()
            conn.close()

            # Reopening runs _ensure_db -> _migrate_add_origin_columns again.
            s2 = GraphStore(db_path=path)
            got = s2.get_node(node.node_id)
            s2.close()
            assert got.origin_runtime == "codex"
            assert got.origin_source == "live"
            assert got.project_key == "Fernweh-Core"
            assert got.session_key == "rollout-legacy"
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass


# ── Pipeline: stamp loop + fallback derive ──────────────────────────────────


class TestPipelineStamping:
    def test_explicit_origin_stamps_every_produced_node_incl_context(
        self, store, pipeline
    ):
        out = pipeline.ingest(IngestionInput(
            source_id="claude-code:Fernweh-Core:sess-1",
            content=(
                "User: We decided to use PostgreSQL for Fernweh-Core.\n"
                "Assistant: Noted, PostgreSQL it is."
            ),
            origin_runtime="claude-code",
            origin_source="live",
            project_key="Fernweh-Core",
            session_key="sess-1",
        ))
        assert out.nodes_created >= 1
        all_nodes = store.list_nodes(limit=999)
        assert len(all_nodes) >= 1
        for n in all_nodes:
            assert n.origin_runtime == "claude-code"
            assert n.origin_source == "live"
            assert n.project_key == "Fernweh-Core"
            assert n.session_key == "sess-1"
        # The CONTEXT node specifically must also carry it.
        ctx = [n for n in all_nodes if n.node_type == NodeType.CONTEXT]
        assert ctx, "expected a CONTEXT node"
        assert ctx[0].origin_runtime == "claude-code"

    def test_fallback_derives_from_source_id_when_omitted(self, store, pipeline):
        """Caller omits origin_runtime entirely — the pipeline derives it
        from source_id via derive_origin."""
        out = pipeline.ingest(IngestionInput(
            source_id="codex:Sam-Repo:rollout-9",
            content="User: We decided to use SQLite.\nAssistant: Confirmed.",
        ))
        assert out.nodes_created >= 1
        all_nodes = store.list_nodes(limit=999)
        for n in all_nodes:
            assert n.origin_runtime == "codex"
            assert n.origin_source == "live"
            assert n.project_key == "Sam-Repo"
            assert n.session_key == "rollout-9"

    def test_fallback_unrecognized_source_id_stays_none(self, store, pipeline):
        out = pipeline.ingest(IngestionInput(
            source_id="mystery-source",
            content="User: We decided something.\nAssistant: OK.",
        ))
        assert out.nodes_created >= 1
        for n in store.list_nodes(limit=999):
            assert n.origin_runtime is None
            assert n.origin_source is None
            assert n.project_key is None
            assert n.session_key is None

    def test_refresh_keyed_path_also_stamps(self, store, pipeline):
        """The ingest_key refresh path (grown session content) stamps origin
        on the newly-extracted nodes too, not just the initial ingest."""
        key = "claude-code:Fernweh-Core:sess-grow"
        pipeline.ingest(IngestionInput(
            source_id=key,
            content="User: We decided to use PostgreSQL.\nAssistant: Noted.",
            ingest_key=key,
            origin_runtime="claude-code",
            origin_source="live",
            project_key="Fernweh-Core",
            session_key="sess-grow",
        ))
        pipeline.ingest(IngestionInput(
            source_id=key,
            content=(
                "User: We decided to use PostgreSQL.\nAssistant: Noted.\n"
                "User: Also we decided to use Redis for caching.\n"
                "Assistant: Confirmed."
            ),
            ingest_key=key,
            origin_runtime="claude-code",
            origin_source="live",
            project_key="Fernweh-Core",
            session_key="sess-grow",
        ))
        for n in store.list_nodes(limit=999):
            assert n.origin_runtime == "claude-code"
            assert n.session_key == "sess-grow"


# ── Export / import round-trip ──────────────────────────────────────────────


class TestExportImportRoundTrip:
    def test_origin_fields_survive_export_import(self, store, pipeline):
        pipeline.ingest(IngestionInput(
            source_id="claude-code:Fernweh-Core:sess-1",
            content="User: We decided to use PostgreSQL.\nAssistant: Noted.",
            origin_runtime="claude-code",
            origin_source="live",
            project_key="Fernweh-Core",
            session_key="sess-1",
        ))
        graph = store.export_graph()
        assert graph.nodes
        for n in graph.nodes:
            assert n.origin_runtime == "claude-code"
            assert n.session_key == "sess-1"

        fd, path2 = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store2 = GraphStore(db_path=path2)
            store2.import_graph(graph, mode="refuse")
            for n in store2.list_nodes(limit=999):
                assert n.origin_runtime == "claude-code"
                assert n.origin_source == "live"
                assert n.project_key == "Fernweh-Core"
                assert n.session_key == "sess-1"
            store2.close()
        finally:
            try:
                os.unlink(path2)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass


# ── Adapters: each one's IngestionInput/Node carries the expected origin ────


class TestAdapterOrigin:
    def test_claude_code_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "projects" / "Fernweh-Core"
            project_dir.mkdir(parents=True)
            session = project_dir / "sess-mara.jsonl"
            with open(session, "w") as f:
                f.write(json.dumps({
                    "type": "human", "content": "We decided to ship Friday.",
                    "timestamp": "2026-07-01T10:00:00Z",
                }) + "\n")

            adapter = ClaudeCodeAdapter(session_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) == 1
            item = results[0]
            assert item["origin_runtime"] == "claude-code"
            assert item["origin_source"] == "live"
            assert item["project_key"] == "Fernweh-Core"
            assert item["session_key"] == "sess-mara"

    def test_codex_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            day_dir = Path(tmpdir) / "2026" / "07" / "01"
            day_dir.mkdir(parents=True)
            rollout = day_dir / "rollout-2026-07-01T10-00-00-theo.jsonl"
            with open(rollout, "w") as f:
                f.write(json.dumps({
                    "timestamp": "2026-07-01T10:00:00Z", "type": "session_meta",
                    "payload": {"cwd": "/home/theo/Fernweh-Core"},
                }) + "\n")
                f.write(json.dumps({
                    "timestamp": "2026-07-01T10:00:01Z", "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text",
                                     "text": "We decided to use SQLite."}],
                    },
                }) + "\n")

            adapter = CodexAdapter(session_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) == 1
            item = results[0]
            assert item["origin_runtime"] == "codex"
            assert item["origin_source"] == "live"
            assert item["project_key"] == "Fernweh-Core"
            assert item["session_key"] == rollout.stem

    def test_obsidian_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir)
            note = vault / "Mara.md"
            note.write_text("# Mara\nSome note body about Sam.", encoding="utf-8")

            adapter = ObsidianVaultAdapter(vault_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) >= 1
            for item in results:
                assert item["origin_runtime"] == "obsidian"
                assert item["origin_source"] == "vault"

    def test_file_watcher_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "theo-notes.txt"
            f.write_text("We decided to use Redis.", encoding="utf-8")

            adapter = FileWatcherAdapter(watch_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) == 1
            assert results[0]["origin_runtime"] == "file"
            assert results[0]["origin_source"] == "watch"

    def test_generic_api_default_parser(self):
        adapter = GenericAPIAdapter(url="https://sam.example.com/feed")
        results = adapter._default_parser({
            "conversations": [{"content": "We decided to use Kafka."}]
        })
        assert len(results) == 1
        assert results[0]["origin_runtime"] == "api"
        assert results[0]["origin_source"] == "api"

    def test_ollama_history_ingest(self, store):
        adapter = OllamaAdapter(graph_path=store.db_path)
        try:
            adapter.ingest_ollama_history([
                {"role": "user", "content": "We decided to use SQLite."},
                {"role": "assistant", "content": "Noted."},
            ])
            nodes = adapter.store.list_nodes(source_id="ollama_history", limit=999)
            assert nodes
            for n in nodes:
                assert n.origin_runtime == "ollama"
                assert n.origin_source == "live"
        finally:
            adapter.close()

    def test_openai_adapter_nodes(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            adapter = OpenAIAdapter(graph_path=path)
            conv = {
                "id": "conv_theo1",
                "title": "Decision",
                "create_time": 1711234567.0,
                "update_time": 1711234999.0,
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None,
                              "children": ["u1"]},
                    "u1": {
                        "id": "u1",
                        "message": {
                            "id": "m1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text",
                                        "parts": ["We decided to use Kafka."]},
                            "create_time": 1711234600.0,
                        },
                        "parent": "root", "children": [],
                    },
                },
            }
            adapter._ingest_conversation_data(conv, "conv_theo1")
            nodes = adapter.store.list_nodes(
                source_id="openai:conversation:conv_theo1", limit=999
            )
            assert nodes
            for n in nodes:
                assert n.origin_runtime == "openai"
                assert n.origin_source == "import"
                assert n.session_key == "conv_theo1"
        finally:
            adapter.close()
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

    def test_langchain_adapter_session_key(self):
        pytest.importorskip("langchain_core")
        from revien.adapters.langchain_adapter import RevienMemory

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            mem = RevienMemory(graph_path=path, session_scope="sess-sam")
            mem.save_context(
                {"input": "We decided to use Kafka."},
                {"output": "Noted, Kafka it is."},
            )
            nodes = mem._store.list_nodes(source_id="sess-sam", limit=999)
            assert nodes
            for n in nodes:
                assert n.origin_runtime == "langchain"
                assert n.origin_source == "live"
                assert n.session_key == "sess-sam"
            mem.close()
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

    def test_hermes_provider_session_key(self, store):
        """Bypass-provider construction (SDK-independent), same pattern as
        tests/test_hermes_provider.py — mirrors what Hermes hands the
        provider: initialize(session_id=...) -> self._session_id, which now
        rides onto ingested nodes as session_key."""
        from revien.hermes_provider import RevienMemoryProvider
        from revien.semantic.index import SemanticIndex
        import queue
        import threading

        semantic = SemanticIndex(store)
        prov = object.__new__(RevienMemoryProvider)
        prov._db_path = store.db_path
        prov._session_id = "hermes-session-42"
        prov._store = store
        prov._pipeline = IngestionPipeline(store, semantic=semantic)
        prov._engine = None
        prov._sync_queue = queue.Queue()
        prov._sync_worker = None
        prov._worker_lock = threading.Lock()

        prov._tool_store(content="We decided to use Postgres.")
        nodes = store.list_nodes(source_id="hermes", limit=999)
        assert nodes
        for n in nodes:
            assert n.origin_runtime == "hermes"
            assert n.origin_source == "live"
            assert n.session_key == "hermes-session-42"


# ── Daemon / MCP passthrough ─────────────────────────────────────────────


class TestDaemonPassthrough:
    def test_v1_ingest_accepts_and_stores_origin_fields(self):
        from fastapi.testclient import TestClient
        from revien.daemon.server import create_app

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            app = create_app(db_path=path)
            client = TestClient(app)
            resp = client.post("/v1/ingest", json={
                "source_id": "custom-source",
                "content": "We decided to use Kafka for the event bus.",
                "origin_runtime": "claude",
                "origin_source": "import",
                "project_key": "Fernweh-Core",
                "session_key": "sess-passthrough",
            })
            assert resp.status_code == 200, resp.text

            store = GraphStore(db_path=path)
            nodes = store.list_nodes(source_id="custom-source", limit=999)
            assert nodes
            for n in nodes:
                assert n.origin_runtime == "claude"
                assert n.origin_source == "import"
                assert n.project_key == "Fernweh-Core"
                assert n.session_key == "sess-passthrough"
            store.close()
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass


# ── F3: origin validation ────────────────────────────────────────────────


class TestValidateOrigin:
    def test_every_known_runtime_and_source_passes(self):
        for runtime in RUNTIMES:
            validate_origin(runtime, None)
        for source in SOURCES:
            validate_origin(None, source)

    def test_none_always_passes(self):
        validate_origin(None, None)

    def test_unknown_runtime_raises_with_offending_value(self):
        with pytest.raises(ValueError, match="TOTALLY-MADE-UP"):
            validate_origin("TOTALLY-MADE-UP", None)

    def test_unknown_source_raises_with_offending_value(self):
        with pytest.raises(ValueError, match="not-a-real-source"):
            validate_origin("claude-code", "not-a-real-source")


class TestPipelineOriginValidation:
    def test_unknown_declared_runtime_raises_value_error(self, store, pipeline):
        with pytest.raises(ValueError, match="TOTALLY-MADE-UP"):
            pipeline.ingest(IngestionInput(
                source_id="claude-code:Fernweh-Core:sess-9",
                content="Theo said otherwise.",
                origin_runtime="TOTALLY-MADE-UP",
                origin_source="not-a-real-source",
            ))

    def test_unknown_declared_source_raises_value_error(self, store, pipeline):
        with pytest.raises(ValueError):
            pipeline.ingest(IngestionInput(
                source_id="claude-code:Fernweh-Core:sess-9",
                content="Theo said otherwise.",
                origin_runtime="claude-code",
                origin_source="not-a-real-source",
            ))

    def test_fallback_derivation_never_validated(self, store, pipeline):
        """Omitted origin_runtime falls back to derive_origin, which only
        ever returns vocabulary values or None — never raises, even for a
        source_id shape nobody recognizes."""
        out = pipeline.ingest(IngestionInput(
            source_id="mystery-source",
            content="User: We decided something.\nAssistant: OK.",
        ))
        assert out.nodes_created >= 1

    def test_omitted_runtime_keeps_declared_project_and_session(self, store, pipeline):
        """Correctness fix: when origin_runtime is omitted but the caller
        DID declare project_key/session_key, those must survive — only
        runtime/source fall back to source_id derivation. Previously the
        whole declared tuple was discarded in favor of derive_origin's,
        even though this source_id (unrecognized) derives to project=None,
        session=None -- the bug would have silently wiped the caller's
        values instead of keeping them."""
        pipeline.ingest(IngestionInput(
            source_id="mystery-source",
            content="User: We decided something notable.\nAssistant: OK.",
            project_key="Fernweh-Core",
            session_key="sess-declared-1",
        ))
        nodes = store.list_nodes(limit=999)
        assert nodes
        for n in nodes:
            assert n.project_key == "Fernweh-Core"
            assert n.session_key == "sess-declared-1"
            # runtime/source still come from derive_origin (unrecognized here)
            assert n.origin_runtime is None
            # Declared-project/session alone doesn't set origin_declared --
            # that flag is gated on origin_runtime specifically (unchanged).
            assert "origin_declared" not in (n.metadata or {})

    def test_omitted_runtime_derived_project_session_fill_gaps(self, store, pipeline):
        """When the caller declares only ONE of project_key/session_key,
        the other is backfilled from derive_origin -- a per-field merge,
        not an all-or-nothing swap."""
        pipeline.ingest(IngestionInput(
            source_id="claude-code:Fernweh-Core:sess-derived",
            content="User: We decided to use SQLite.\nAssistant: Noted.",
            session_key="sess-declared-override",
        ))
        nodes = store.list_nodes(limit=999)
        assert nodes
        for n in nodes:
            # project_key wasn't declared -- backfilled from source_id.
            assert n.project_key == "Fernweh-Core"
            # session_key WAS declared -- kept, not overwritten by the
            # source_id-derived "sess-derived".
            assert n.session_key == "sess-declared-override"
            assert n.origin_runtime == "claude-code"

    def test_origin_declared_true_when_caller_supplies_runtime(self, store, pipeline):
        pipeline.ingest(IngestionInput(
            source_id="claude-code:Fernweh-Core:sess-1",
            content="User: We decided to use PostgreSQL.\nAssistant: Noted.",
            origin_runtime="claude-code",
            origin_source="live",
        ))
        nodes = store.list_nodes(limit=999)
        assert nodes
        for n in nodes:
            assert n.metadata.get("origin_declared") is True

    def test_origin_declared_absent_when_derived_from_source_id(self, store, pipeline):
        pipeline.ingest(IngestionInput(
            source_id="claude-code:Fernweh-Core:sess-1",
            content="User: We decided to use PostgreSQL.\nAssistant: Noted.",
        ))
        nodes = store.list_nodes(limit=999)
        assert nodes
        for n in nodes:
            assert "origin_declared" not in (n.metadata or {})


class TestDaemonOriginValidation:
    def test_v1_ingest_unknown_runtime_is_400(self):
        from fastapi.testclient import TestClient
        from revien.daemon.server import create_app

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            app = create_app(db_path=path)
            client = TestClient(app)
            resp = client.post("/v1/ingest", json={
                "source_id": "custom-source",
                "content": "We decided to use Kafka for the event bus.",
                "origin_runtime": "TOTALLY-MADE-UP",
            })
            assert resp.status_code == 400
            assert "TOTALLY-MADE-UP" in resp.json()["detail"]
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass


class TestMCPStoreCannotClaimVault:
    def test_revien_store_has_no_origin_source_parameter(self):
        import inspect
        from revien.mcp_server import _build_server

        src = inspect.getsource(_build_server)
        tool_src = src.split("def revien_store(")[1].split(") -> Dict[str, Any]:")[0]
        assert "origin_source" not in tool_src, (
            "revien_store must not accept an origin_source parameter"
        )

    def test_revien_store_always_stamps_api_source(self, store):
        """Exercise the pipeline call the tool makes directly (no MCP SDK
        dependency needed): whatever origin_runtime is passed, origin_source
        is hard-set to 'api', never settable to 'vault'."""
        from revien.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(store)
        # Mirrors mcp_server.revien_store's call shape exactly.
        result = pipeline.ingest(IngestionInput(
            source_id="mcp",
            content="A durable fact from an LLM tool call.",
            content_type="note",
            defer_embed=False,
            origin_runtime="obsidian",  # even claiming a vault-associated runtime
            origin_source="api",  # hard-set by the tool, never caller-controlled
            project_key=None,
            session_key=None,
        ))
        node = store.get_node(result.context_node_id)
        assert node.origin_source == "api"
        assert node.origin_source != "vault"


# ── F8: user_version skip marker (second open does not rescan) ────────────


class TestOriginBackfillUserVersionMarker:
    def test_second_open_sets_user_version_and_skips_scan(self):
        """First open on a fresh db backfills (0 rows, but still runs the
        pass) and sets PRAGMA user_version to 3. A second open must see
        user_version >= 3 and skip the NULL-scan outright — proven here by
        patching derive_origin to explode if it's ever called again."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = GraphStore(db_path=path)
            s.close()

            conn = sqlite3.connect(path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
            assert version >= 3

            import revien.graph.store as store_module

            def _boom(source_id):
                raise AssertionError(
                    "derive_origin must not be called when user_version >= 3 "
                    "and no column needed to be added"
                )

            original = store_module.derive_origin
            store_module.derive_origin = _boom
            try:
                s2 = GraphStore(db_path=path)  # must NOT scan/backfill
                s2.close()
            finally:
                store_module.derive_origin = original
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

    def test_columns_readded_after_version_marker_still_backfills(self):
        """Edge case: if the origin columns are somehow missing again on a
        db that already carries user_version >= 3 (e.g. dropped out-of-band
        after the marker was set), the ALTER re-adds them AND the backfill
        still runs — the schema in front of us always wins over a stale
        marker."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = GraphStore(db_path=path)
            node = s.add_node(Node(
                node_type=NodeType.CONTEXT,
                label="legacy",
                content="body",
                source_id="codex:Fernweh-Core:rollout-legacy",
            ))
            s.close()

            conn = sqlite3.connect(path)
            for ix in ("idx_nodes_origin_runtime", "idx_nodes_origin_source", "idx_nodes_project", "idx_nodes_session"):
                conn.execute(f"DROP INDEX IF EXISTS {ix}")
            for col in ("origin_runtime", "origin_source", "project_key", "session_key"):
                conn.execute(f"ALTER TABLE nodes DROP COLUMN {col}")
            conn.commit()
            version_before = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
            assert version_before >= 3  # the stale marker from the first open

            s2 = GraphStore(db_path=path)
            got = s2.get_node(node.node_id)
            s2.close()
            assert got.origin_runtime == "codex"
            assert got.project_key == "Fernweh-Core"
            assert got.session_key == "rollout-legacy"
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

    def test_migration_003_standalone_sets_user_version(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # A store-openable pre-003 table: everything but the origin
            # columns (see TestMigration003.test_columns_added_on_a_truly_
            # pre_003_table — migrate() delegates to GraphStore, so the
            # fixture must be something GraphStore can actually open).
            s = GraphStore(db_path=path)
            s.close()
            conn = sqlite3.connect(path)
            for ix in ("idx_nodes_origin_runtime", "idx_nodes_origin_source", "idx_nodes_project", "idx_nodes_session"):
                conn.execute(f"DROP INDEX IF EXISTS {ix}")
            for col in ("origin_runtime", "origin_source", "project_key", "session_key"):
                conn.execute(f"ALTER TABLE nodes DROP COLUMN {col}")
            conn.execute("PRAGMA user_version = 0")
            conn.commit()
            conn.close()

            migrate = _load_migration().migrate
            migrate(path)

            conn = sqlite3.connect(path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
            assert version == 4  # chain ends at 4 (recorded_at_source backfill)
        finally:
            try:
                os.unlink(path)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass
