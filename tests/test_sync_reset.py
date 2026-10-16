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
        assert "in the middle of a sync" in r.output
        assert "run this again" in r.output
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
        assert "claude-code" in r.output

    def test_cursor_without_connected_adapter_is_refused(
            self, store, db_path, monkeypatch):
        """Type cannot be verified, so a bare cursor name is not enough."""
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code", adapters={})
        assert r.exit_code == 1
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
        ("api", "generic_api"), ("mystery", "no_such_type"),
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


# -- right-to-forget warning ----------------------------------------------

class TestForgetWarning:
    def _forgotten_db(self, store):
        store.set_sync_cursor("claude-code", T)
        store.record_audit("n1", "forget", actor="tester")
        store.close()

    def _run(self, monkeypatch, db_path, *extra):
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": CC, "db_path": db_path})
        return CliRunner().invoke(
            main, ["sync", "reset", "claude-code", "--db", db_path, *extra])

    def test_tombstones_block_without_yes(self, store, db_path, monkeypatch):
        self._forgotten_db(store)
        r = self._run(monkeypatch, db_path)
        assert r.exit_code == 1
        assert "right-to-forget" in r.output
        assert "--yes" in r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") == T
        finally:
            s.close()

    def test_yes_proceeds(self, store, db_path, monkeypatch):
        self._forgotten_db(store)
        r = self._run(monkeypatch, db_path, "--yes")
        assert r.exit_code == 0, r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") is None
        finally:
            s.close()

    def test_no_tombstones_needs_no_yes(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.record_audit("n1", "invalidate", actor="tester")  # not a forget
        store.close()
        assert self._run(monkeypatch, db_path).exit_code == 0

    def test_count_forget_tombstones(self, store):
        assert store.count_forget_tombstones() == 0
        store.record_audit("n1", "forget", actor="x")
        store.record_audit("n2", "update", actor="x")
        assert store.count_forget_tombstones() == 1


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
