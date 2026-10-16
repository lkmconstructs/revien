"""`revien sync reset <adapter>`: store method, CLI, and the safety of the
full re-read it triggers.

Evidence for the two design questions:
  * a full re-read of already-ingested Claude Code sessions refreshes the one
    context node (ingest_key), it does not duplicate (TestReingestSafety);
  * the usage observer's own cursor does not need the reset, because it never
    goes through the adapter's session parser (TestPreExistingUsageObserver,
    which documents existing behaviour and is not a test of the reset).
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.cli import main
from revien.daemon.scheduler import EPOCH, SyncScheduler
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionPipeline
from revien.usage.observer import USAGE_CURSOR, UsageObserver


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(db_path):
    s = GraphStore(db_path=db_path)
    yield s
    s.close()


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


T = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _write_real_session(root, turns):
    """A session in the REAL nested Claude Code shape (message.content)."""
    d = root / "projects" / "proj-x"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "sess-1.jsonl"
    with open(path, "w") as f:
        for role, text in turns:
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
                "timestamp": "2026-07-01T10:00:00Z",
            }) + "\n")
    return path


# -- store ---------------------------------------------------------------

class TestStoreMethod:
    def test_delete_returns_whether_row_existed(self, store):
        assert store.delete_sync_cursor("claude-code") is False
        store.set_sync_cursor("claude-code", T)
        assert store.delete_sync_cursor("claude-code") is True
        assert store.get_sync_cursor("claude-code") is None
        assert store.delete_sync_cursor("claude-code") is False

    def test_delete_touches_only_that_adapter(self, store):
        store.set_sync_cursor("claude-code", T)
        store.set_sync_cursor("codex", T)
        store.delete_sync_cursor("claude-code")
        assert store.get_sync_cursor("codex") == T
        assert store.list_sync_cursor_names() == ["codex"]


# -- CLI -----------------------------------------------------------------

CC = {"claude-code": {"type": "claude_code", "session_dir": "/nonexistent"}}


class TestCLI:
    def _invoke(self, monkeypatch, db_path, name, adapters=None, extra=()):
        monkeypatch.setattr(
            "revien.cli._load_config",
            lambda: {"adapters": CC if adapters is None else adapters,
                     "db_path": db_path},
        )
        return CliRunner().invoke(
            main, ["sync", "reset", name, "--db", db_path, *extra])

    def test_reset_clears_cursor_and_explains(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code")
        assert r.exit_code == 0, r.output
        assert "Cleared the sync position for 'claude-code'" in r.output
        assert "next sync" in r.output
        assert "6 hours" in r.output
        assert "cannot undo this reset" in r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") is None
        finally:
            s.close()

    def test_output_is_ascii(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code")
        assert r.exit_code == 0, r.output
        r.output.encode("cp1252")

    def test_connected_adapter_without_cursor_is_a_noop_success(
            self, store, db_path, monkeypatch):
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code")
        assert r.exit_code == 0, r.output
        assert "already reads its full history" in r.output

    def test_unknown_name_is_an_error_listing_known(
            self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude_code")  # typo
        assert r.exit_code == 1
        assert "No connected adapter named 'claude_code'" in r.output
        assert "Connected adapters: claude-code" in r.output

    def test_cursor_without_connected_adapter_is_refused(
            self, store, db_path, monkeypatch):
        """Type cannot be verified, so a bare cursor name is not enough."""
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code", adapters={})
        assert r.exit_code == 1
        assert "No connected adapter named" not in r.output  # not contradictory
        assert "saved sync position but is not a connected adapter" in r.output
        assert "Connected adapters: none" in r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") == T
        finally:
            s.close()

    def test_missing_database(self, tmp_path, monkeypatch):
        missing = str(tmp_path / "nope.db")
        r = self._invoke(monkeypatch, missing, "claude-code")
        assert r.exit_code == 1
        assert "No Revien database found" in r.output


# -- only provably idempotent adapters ------------------------------------

class TestOnlyKeyedAdapters:
    @pytest.mark.parametrize("name,atype", [
        ("watch", "file_watcher"), ("vault", "obsidian"),
        ("api", "generic_api"),
    ])
    def test_unkeyed_adapter_is_refused_and_cursor_kept(
            self, store, db_path, monkeypatch, name, atype):
        store.set_sync_cursor(name, T)
        store.close()
        r = CliRunner()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {name: {"type": atype}}, "db_path": db_path})
        res = r.invoke(main, ["sync", "reset", name, "--db", db_path])
        assert res.exit_code == 1, res.output
        assert "Refusing to reset" in res.output
        assert "as new nodes" in res.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor(name) == T
        finally:
            s.close()

    def test_codex_is_accepted(self, store, db_path, monkeypatch):
        store.set_sync_cursor("codex", T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {"codex": {"type": "codex"}}, "db_path": db_path})
        res = CliRunner().invoke(main, ["sync", "reset", "codex", "--db", db_path])
        assert res.exit_code == 0, res.output

    def test_capability_is_declared_on_the_adapter_classes(self):
        from revien.adapters import (
            CodexAdapter, FileWatcherAdapter, GenericAPIAdapter,
            ObsidianVaultAdapter, adapter_type_rereads_idempotently)
        assert ClaudeCodeAdapter.IDEMPOTENT_REREAD and CodexAdapter.IDEMPOTENT_REREAD
        for cls in (FileWatcherAdapter, GenericAPIAdapter, ObsidianVaultAdapter):
            assert not cls.IDEMPOTENT_REREAD
        assert adapter_type_rereads_idempotently("claude_code")
        assert not adapter_type_rereads_idempotently("obsidian")
        assert not adapter_type_rereads_idempotently("")


# -- type problems are config problems, not "no stable key" ------------------

class TestConfigTypeMessages:
    @pytest.mark.parametrize("entry", [
        {"type": "no_such_type"}, {"type": ""}, {}, {"type": None},
        {"type": ["x"]}, "claude-code",
    ])
    def test_bad_type_is_a_config_error(self, store, db_path, monkeypatch, entry):
        store.set_sync_cursor("mystery", T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {"mystery": entry}, "db_path": db_path})
        res = CliRunner().invoke(main, ["sync", "reset", "mystery", "--db", db_path])
        assert res.exit_code == 1, res.output
        assert "no stable key" not in res.output
        assert "Refusing to reset" not in res.output
        assert "Fix the entry in the config file" in res.output
        assert "known types: claude_code, codex" in res.output
        assert "Traceback" not in res.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("mystery") == T
        finally:
            s.close()

    def test_known_unkeyed_type_still_says_no_stable_key(
            self, store, db_path, monkeypatch):
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {"v": {"type": "obsidian"}}, "db_path": db_path})
        res = CliRunner().invoke(main, ["sync", "reset", "v", "--db", db_path])
        assert res.exit_code == 1
        assert "no stable" in res.output and "config" not in res.output.lower()


# -- non-adapter names ------------------------------------------------------

class TestUsageCursorRefused:
    def test_usage_cursor_is_refused_with_pointer(self, store, db_path, monkeypatch):
        store.set_sync_cursor(USAGE_CURSOR, T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": CC, "db_path": db_path})
        r = CliRunner().invoke(main, ["sync", "reset", USAGE_CURSOR, "--db", db_path])
        assert r.exit_code == 1
        assert "revien usage scan --full" in r.output
        assert "next daemon sync" not in r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor(USAGE_CURSOR) == T
        finally:
            s.close()

    def test_reset_of_adapter_leaves_usage_cursor_alone(
            self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.set_sync_cursor(USAGE_CURSOR, T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": CC, "db_path": db_path})
        r = CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", db_path])
        assert r.exit_code == 0, r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") is None
            assert s.get_sync_cursor(USAGE_CURSOR) == T
        finally:
            s.close()


# -- bad --db paths ---------------------------------------------------------

OLDEST_REVIEN_SCHEMA = """
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
    content TEXT NOT NULL, source_id TEXT DEFAULT '', created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL, access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}', source_type TEXT DEFAULT 'inferred',
    confidence REAL DEFAULT 0.5, pinned INTEGER DEFAULT 0,
    confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '',
    source_context TEXT DEFAULT '', last_referenced TEXT, invalidated_at TEXT,
    source_modality TEXT DEFAULT 'text', answerable_by_text INTEGER DEFAULT 1,
    vision_processed INTEGER DEFAULT 0, recorded_at TEXT,
    event_time_start TEXT, event_time_end TEXT, event_time_granularity TEXT,
    event_time_confidence REAL, event_time_text TEXT DEFAULT '',
    valid_from TEXT, valid_until TEXT
);
CREATE TABLE edges (
    edge_id TEXT PRIMARY KEY, edge_type TEXT NOT NULL,
    source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL,
    weight REAL DEFAULT 0.5, created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}', confidence REAL DEFAULT 0.5,
    confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '',
    source_context TEXT DEFAULT '',
    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT NOT NULL,
    op TEXT NOT NULL, actor TEXT DEFAULT '', ts TEXT NOT NULL,
    before_json TEXT, after_json TEXT
);
"""
# (copied from `git show 7757085:revien/graph/store.py`, the first commit)


class TestBadDbPaths:
    def _run(self, monkeypatch, path):
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": CC, "db_path": str(path)})
        return CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", str(path)])

    def test_zero_byte_file_is_untouched(self, tmp_path, monkeypatch):
        p = tmp_path / "empty.db"
        p.write_bytes(b"")
        r = self._run(monkeypatch, p)
        assert r.exit_code == 1
        assert "empty" in r.output and "Traceback" not in r.output
        assert p.stat().st_size == 0

    def test_directory(self, tmp_path, monkeypatch):
        r = self._run(monkeypatch, tmp_path)
        assert r.exit_code == 1
        assert "directory" in r.output
        assert r.exception is None or isinstance(r.exception, SystemExit)

    def test_non_sqlite_file_is_untouched(self, tmp_path, monkeypatch):
        p = tmp_path / "notes.db"
        p.write_text("just some text, definitely not sqlite\n" * 10)
        before = p.read_bytes()
        r = self._run(monkeypatch, p)
        assert r.exit_code == 1
        assert "not a SQLite database" in r.output
        assert p.read_bytes() == before

    def _bytes_and_run(self, monkeypatch, p):
        before = p.read_bytes()
        r = self._run(monkeypatch, p)
        assert r.exit_code == 1, r.output
        assert "Traceback" not in r.output
        assert len(r.output.strip().splitlines()) == 1, r.output
        assert p.read_bytes() == before  # byte-identical
        assert sorted(x.name for x in p.parent.iterdir()) == [p.name]  # no journal
        return r

    def test_foreign_sqlite_file_is_refused_and_untouched(self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / "photos.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT)")
        c.execute("INSERT INTO photos (path) VALUES ('a.jpg')")
        c.commit()
        c.close()
        r = self._bytes_and_run(monkeypatch, p)
        assert "not a Revien database" in r.output

    def test_foreign_sync_cursors_shape_is_refused_and_untouched(
            self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / "other.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE sync_cursors (name TEXT, pos INTEGER)")
        c.commit()
        c.close()
        r = self._bytes_and_run(monkeypatch, p)
        assert "not a Revien database" in r.output

    def test_foreign_nodes_table_is_refused_and_untouched(self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / "graph.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE nodes (id TEXT, x TEXT)")
        c.execute("CREATE TABLE edges (a TEXT, b TEXT)")
        c.execute("CREATE TABLE audit_log (id INTEGER, node_id TEXT, op TEXT, "
                  "actor TEXT, ts TEXT, before_json TEXT, after_json TEXT)")
        c.commit()
        c.close()
        r = self._bytes_and_run(monkeypatch, p)
        assert "'nodes' table that is not Revien's" in r.output

    def test_valid_sqlite_header_but_corrupt_body_is_refused(self, tmp_path, monkeypatch):
        p = tmp_path / "bad.db"
        p.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4000)
        self._bytes_and_run(monkeypatch, p)

    def test_real_revien_database_is_accepted(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        from revien.cli import _sqlite_problem
        assert _sqlite_problem(db_path) is None
        assert self._run(monkeypatch, db_path).exit_code == 0

    def test_old_revien_database_without_later_tables_is_accepted(
            self, tmp_path):
        """A database with the exact nodes/edges/audit_log schema the oldest
        real Revien (commit 7757085, 0.3.0) created, and none of the later
        tables, is accepted; GraphStore then upgrades it in place."""
        import sqlite3
        p = tmp_path / "old.db"
        c = sqlite3.connect(p)
        c.executescript(OLDEST_REVIEN_SCHEMA)
        c.commit()
        c.close()
        from revien.cli import _sqlite_problem
        assert _sqlite_problem(str(p)) is None
        s = GraphStore(db_path=str(p))  # upgrades in place without error
        s.close()

    def test_minimal_nodes_edges_shape_is_not_revien(self, tmp_path):
        """The 4-column nodes/edges shape no real Revien version produced
        used to pass the check, and GraphStore then wrote into the file."""
        import sqlite3
        p = tmp_path / "shape.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT, "
                  "label TEXT, content TEXT)")
        c.execute("CREATE TABLE edges (edge_id TEXT PRIMARY KEY, edge_type TEXT, "
                  "source_node_id TEXT, target_node_id TEXT)")
        c.commit()
        c.close()
        before = p.read_bytes()
        from revien.cli import _sqlite_problem
        assert _sqlite_problem(str(p))
        assert p.read_bytes() == before

    def test_foreign_wal_database_leaves_no_sidecars_and_no_changes(
            self, tmp_path, monkeypatch):
        """Read-only behaviour is observable: bytes and the directory listing
        are identical afterwards (mode=rwc, or plain mode=ro on a WAL file,
        would add -wal/-shm files)."""
        import sqlite3
        p = tmp_path / "foreign.db"
        c = sqlite3.connect(p)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE people (id INTEGER, name TEXT)")
        c.execute("INSERT INTO people VALUES (1, 'a')")
        c.commit()
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        for side in ("-wal", "-shm"):  # closing may or may not remove them
            (tmp_path / ("foreign.db" + side)).unlink(missing_ok=True)
        listing = sorted(os.listdir(tmp_path))
        before = p.read_bytes()
        r = self._run(monkeypatch, p)
        assert r.exit_code == 1
        assert "not a Revien database" in r.output
        assert sorted(os.listdir(tmp_path)) == listing
        assert p.read_bytes() == before

    def test_readonly_uri_cannot_write_and_makes_no_sidecars_while_open(
            self, tmp_path):
        """Sidecars are removed again on close, so the directory is checked
        WHILE the connection is open; a write attempt must also fail (this is
        what catches mode=rwc)."""
        import sqlite3
        from pathlib import Path
        from revien.cli import _readonly_uri
        p = tmp_path / "w.db"
        c = sqlite3.connect(p)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE t (a)")
        c.commit()
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        for side in ("-wal", "-shm"):
            (tmp_path / ("w.db" + side)).unlink(missing_ok=True)
        listing = sorted(os.listdir(tmp_path))
        ro = sqlite3.connect(_readonly_uri(Path(p)), uri=True)
        try:
            ro.execute("SELECT name FROM sqlite_master").fetchall()
            assert sorted(os.listdir(tmp_path)) == listing  # none created
            with pytest.raises(sqlite3.OperationalError):
                ro.execute("CREATE TABLE x (a)")
        finally:
            ro.close()

    def test_foreign_wal_database_with_revien_names_is_also_untouched(
            self, tmp_path, monkeypatch):
        import sqlite3
        p = tmp_path / "shape-wal.db"
        c = sqlite3.connect(p)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT, "
                  "label TEXT, content TEXT)")
        c.execute("CREATE TABLE edges (edge_id TEXT PRIMARY KEY, edge_type TEXT, "
                  "source_node_id TEXT, target_node_id TEXT)")
        c.commit()
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        for side in ("-wal", "-shm"):
            (tmp_path / ("foreign.db" + side)).unlink(missing_ok=True)
            (tmp_path / ("shape-wal.db" + side)).unlink(missing_ok=True)
        listing = sorted(os.listdir(tmp_path))
        before = p.read_bytes()
        r = self._run(monkeypatch, p)
        assert r.exit_code == 1 and "not a Revien database" in r.output
        assert sorted(os.listdir(tmp_path)) == listing
        assert p.read_bytes() == before

    def test_mistyped_name_does_not_migrate_an_older_revien_db(
            self, tmp_path, monkeypatch):
        """Name/type are validated before a GraphStore is opened, so a typo
        cannot write the newer schema into an older Revien database."""
        import sqlite3
        p = tmp_path / "old.db"
        c = sqlite3.connect(p)
        c.executescript(OLDEST_REVIEN_SCHEMA)
        c.commit()
        c.close()
        before = p.read_bytes()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": CC, "db_path": str(p)})
        for argv in (["claude_code"],            # typo
                     ["claude-code", "--orphaned"],  # connected + --orphaned
                     ["nothing", "--orphaned"]):     # no recorded state
            r = CliRunner().invoke(
                main, ["sync", "reset", *argv, "--db", str(p)])
            assert r.exit_code == 1, r.output
            assert p.read_bytes() == before, argv
        # unknown-type and unkeyed-type refusals also precede the open
        for entry in ({"type": "bogus"}, {"type": "obsidian"}):
            monkeypatch.setattr("revien.cli._load_config", lambda e=entry: {
                "adapters": {"x": e}, "db_path": str(p)})
            r = CliRunner().invoke(main, ["sync", "reset", "x", "--db", str(p)])
            assert r.exit_code == 1
            assert p.read_bytes() == before

    def test_locked_database_is_one_clean_line(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        conn = store._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            monkeypatch.setattr("revien.graph.store.sqlite3.connect",
                                _fast_timeout_connect)
            r = self._run(monkeypatch, db_path)
        finally:
            conn.execute("ROLLBACK")
        assert r.exit_code == 1, r.output
        assert "Traceback" not in r.output
        assert len(r.output.strip().splitlines()) == 1
        assert isinstance(r.exception, SystemExit)


def _fast_timeout_connect(*a, **k):
    import sqlite3
    k["timeout"] = 0.05
    return _real_connect(*a, **k)


import sqlite3 as _sqlite3
_real_connect = _sqlite3.connect


# -- re-ingest safety ----------------------------------------------------

def _sync(store, root):
    sched = SyncScheduler(pipeline=IngestionPipeline(store))
    sched.register_adapter("claude-code", ClaudeCodeAdapter(session_dir=str(root)))
    return run_async(sched.sync_one("claude-code"))


def _session_contexts(store):
    return [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
            if n.source_id == "claude-code:proj-x:sess-1"]


class TestReingestSafety:
    def test_reset_then_full_reread_refreshes_not_duplicates(self, store, tmp_path):
        path = _write_real_session(tmp_path, [
            ("user", "We decided to use PostgreSQL for the backend database."),
            ("assistant", "Noted. PostgreSQL for the backend it is."),
        ])
        assert _sync(store, tmp_path)["items_ingested"] == 1
        (first,) = _session_contexts(store)
        total = store.count_nodes()

        # Cursor has moved past the file: a second sync reads nothing.
        assert _sync(store, tmp_path)["items_ingested"] == 0

        # Reset, then a full re-read of the identical session.
        assert store.delete_sync_cursor("claude-code") is True
        assert _sync(store, tmp_path)["items_ingested"] == 1
        (again,) = _session_contexts(store)
        assert again.node_id == first.node_id
        assert store.count_nodes() == total

        # Reset after the session GREW: the same node is refreshed in place.
        with open(path, "a") as f:
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "Also use Redis for caching."},
            }) + "\n")
        store.delete_sync_cursor("claude-code")
        assert _sync(store, tmp_path)["items_ingested"] == 1
        (grown,) = _session_contexts(store)
        assert grown.node_id == first.node_id
        assert "Redis" in grown.content


# -- pre-existing behaviour (not part of the reset feature) -----------------

class TestPreExistingUsageObserver:
    """Characterization only: passes on the parent commit too."""

    def test_observer_never_uses_the_adapter_parser(self, store, tmp_path, monkeypatch):
        """The parse bug fixed in 38addd1 lived in _parse_session_log. The
        usage observer reads raw files via extract_recall_episodes, so its
        cursor never skipped anything because of that bug."""
        _write_real_session(tmp_path, [("user", "hello"), ("assistant", "hi")])
        adapter = ClaudeCodeAdapter(session_dir=str(tmp_path))

        def boom(*a, **k):
            raise AssertionError("observer must not use the ingest parser")

        monkeypatch.setattr(adapter, "_parse_session_log", boom)
        report = UsageObserver(store, mode="shadow").run_claude_code(adapter)
        assert report.files_scanned == 1
        assert store.get_sync_cursor(USAGE_CURSOR) is not None


# -- capability drift --------------------------------------------------------

def _fetch(adapter):
    return run_async(adapter.fetch_new_content(EPOCH))


def _make_adapters(tmp_path):
    """One real instance of every adapter class, over real files, plus the
    items each fetches from the beginning of time."""
    import httpx
    from revien.adapters import (
        CodexAdapter, FileWatcherAdapter, GenericAPIAdapter,
        ObsidianVaultAdapter)
    from tests.test_codex_adapter import MOCK_CODEX_ROLLOUT, _write_rollout

    cc_root = tmp_path / "cc"
    _write_real_session(cc_root, [("user", "We chose PostgreSQL for storage."),
                                  ("assistant", "Noted, PostgreSQL.")])
    codex_root = tmp_path / "codex"
    _write_rollout(codex_root / "2026" / "07" / "01" / "rollout-a.jsonl",
                   MOCK_CODEX_ROLLOUT)
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "note.md").write_text("We decided to adopt Redis for caching.")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Plan.md").write_text("# Plan\n\nWe will migrate to Postgres.\n\n"
                                   "## Risks\n\nDowntime during the cutover.\n")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return [{"content": "an api item", "timestamp": T.isoformat()}]

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): return _Resp()

    return {
        "claude_code": ClaudeCodeAdapter(session_dir=str(cc_root)),
        "codex": CodexAdapter(session_dir=str(codex_root)),
        "file_watcher": FileWatcherAdapter(watch_dir=str(watch)),
        "obsidian": ObsidianVaultAdapter(vault_dir=str(vault)),
        "generic_api": GenericAPIAdapter(url="http://example.invalid/x"),
    }, _Client, httpx


class TestCapabilityDrift:
    def test_flag_matches_what_each_adapter_actually_emits(
            self, tmp_path, monkeypatch):
        from revien.adapters import _ADAPTER_CLASSES
        adapters, client, httpx = _make_adapters(tmp_path)
        monkeypatch.setattr(httpx, "AsyncClient", client)
        assert set(adapters) == set(_ADAPTER_CLASSES)  # a new class needs a case here
        for atype, adapter in adapters.items():
            assert type(adapter) is _ADAPTER_CLASSES[atype]
            items = _fetch(adapter)
            assert items, f"{atype} fetched nothing: the test would prove nothing"
            keyed = [bool((i.get("ingest_key") or "").strip()) for i in items]
            if type(adapter).IDEMPOTENT_REREAD:
                assert all(keyed), f"{atype} is flagged idempotent but emits unkeyed items"
            else:
                assert not any(keyed), f"{atype} emits ingest_key but is not flagged"

    def test_codex_without_a_known_project_is_still_keyed(self, tmp_path):
        """The flag promises EVERY item is keyed, including a rollout whose
        session_meta carries no cwd (project unknown)."""
        from revien.adapters.codex import CodexAdapter
        from tests.test_codex_adapter import MOCK_CODEX_ROLLOUT, _write_rollout

        lines = [dict(l) for l in MOCK_CODEX_ROLLOUT]
        for line in lines:
            if line.get("type") == "session_meta":
                line["payload"] = {k: v for k, v in line["payload"].items()
                                   if k != "cwd"}
        root = tmp_path / "codex-nocwd"
        _write_rollout(root / "2026" / "07" / "01" / "rollout-z.jsonl", lines)
        items = _fetch(CodexAdapter(session_dir=str(root)))
        assert items, "nothing fetched: the test would prove nothing"
        assert all(i["source_id"].startswith("codex:unknown:") for i in items)
        assert all((i.get("ingest_key") or "").strip() for i in items)
        assert CodexAdapter.IDEMPOTENT_REREAD

    def test_keys_are_stable_across_reads(self, tmp_path):
        adapters, _, _ = _make_adapters(tmp_path)
        for atype in ("claude_code", "codex"):
            a = [i["ingest_key"] for i in _fetch(adapters[atype])]
            b = [i["ingest_key"] for i in _fetch(adapters[atype])]
            assert a == b and len(set(a)) == len(a)

    def test_registry_agrees_with_build_adapter_from_config(self, tmp_path):
        import inspect
        import re
        from revien.adapters import _ADAPTER_CLASSES, build_adapter_from_config
        entries = {
            "claude_code": {"session_dir": str(tmp_path)},
            "codex": {"session_dir": str(tmp_path)},
            "file_watcher": {"watch_dir": str(tmp_path)},
            "generic_api": {"url": "http://example.invalid/x"},
            "obsidian": {"vault_dir": str(tmp_path)},
        }
        # Every registered type is buildable and builds that very class...
        assert set(entries) == set(_ADAPTER_CLASSES)
        for atype, extra in entries.items():
            built = build_adapter_from_config({"type": atype, **extra})
            assert type(built) is _ADAPTER_CLASSES[atype], atype
        # ...and the builder handles no type the registry lacks.
        src = inspect.getsource(build_adapter_from_config)
        handled = set(re.findall(r'adapter_type == "([^"]+)"', src))
        assert handled == set(_ADAPTER_CLASSES)
        assert build_adapter_from_config({"type": "no_such_type"}) is None


# -- real codex re-read through the CLI + scheduler --------------------------

class TestCodexRereadIdempotent:
    def test_codex_reset_then_reread_refreshes_not_duplicates(
            self, store, db_path, tmp_path, monkeypatch):
        from revien.adapters.codex import CodexAdapter
        from tests.test_codex_adapter import MOCK_CODEX_ROLLOUT, _write_rollout

        root = tmp_path / "sessions"
        _write_rollout(root / "2026" / "07" / "01" / "rollout-a.jsonl",
                       MOCK_CODEX_ROLLOUT)
        _write_rollout(root / "2026" / "07" / "02" / "rollout-b.jsonl",
                       MOCK_CODEX_ROLLOUT)
        # The cursor is stamped at sync start, and a file modified in the same
        # clock tick is re-read on purpose (never missed). Windows' coarse
        # datetime.now() can land at or before a just-written mtime, so age the
        # files so "cursor moved on" tests the cursor, not the clock.
        old = datetime.now(timezone.utc).timestamp() - 60
        for f in root.rglob("*.jsonl"):
            os.utime(f, (old, old))

        def sync():
            sched = SyncScheduler(pipeline=IngestionPipeline(store))
            sched.register_adapter("codex", CodexAdapter(session_dir=str(root)))
            return run_async(sched.sync_one("codex"))

        def contexts():
            return [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
                    if (n.source_id or "").startswith("codex:")]

        first = sync()
        assert first["items_ingested"] >= 1
        ids = sorted(n.node_id for n in contexts())
        total = store.count_nodes()
        assert ids and sync()["items_ingested"] == 0  # cursor moved on

        for _ in range(3):  # reset through the real CLI, repeatedly
            store._get_conn().commit()
            monkeypatch.setattr("revien.cli._load_config", lambda: {
                "adapters": {"codex": {"type": "codex"}}, "db_path": db_path})
            r = CliRunner().invoke(main, ["sync", "reset", "codex", "--db", db_path])
            assert r.exit_code == 0, r.output
            assert store.get_sync_cursor("codex") is None
            assert sync()["items_ingested"] == first["items_ingested"]
            assert sorted(n.node_id for n in contexts()) == ids
            assert store.count_nodes() == total
