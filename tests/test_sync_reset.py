"""`revien sync reset <adapter>`: store method, CLI, and the safety of the
full re-read it triggers.

Evidence for the two design questions:
  * a full re-read of already-ingested Claude Code sessions refreshes the one
    context node (ingest_key), it does not duplicate (TestReingestSafety);
  * the usage observer's own cursor does not need the reset, because it never
    goes through the adapter's session parser (TestUsageCursorBound).
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

class TestCLI:
    def _invoke(self, monkeypatch, db_path, name, adapters=None):
        monkeypatch.setattr(
            "revien.cli._load_config",
            lambda: {"adapters": adapters or {}, "db_path": db_path},
        )
        return CliRunner().invoke(main, ["sync", "reset", name, "--db", db_path])

    def test_reset_clears_cursor_and_explains(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code")
        assert r.exit_code == 0, r.output
        assert "Cleared the sync position for 'claude-code'" in r.output
        assert "full history" in r.output
        assert "not duplicated" in r.output
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor("claude-code") is None
        finally:
            s.close()

    def test_output_is_ascii(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude-code")
        r.output.encode("cp1252")

    def test_connected_adapter_without_cursor_is_a_noop_success(
            self, store, db_path, monkeypatch):
        store.close()
        r = self._invoke(monkeypatch, db_path, "codex", adapters={"codex": {}})
        assert r.exit_code == 0, r.output
        assert "already reads its full history" in r.output

    def test_unknown_name_is_an_error_listing_known(
            self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._invoke(monkeypatch, db_path, "claude_code")  # typo
        assert r.exit_code == 1
        assert "No adapter or sync position named 'claude_code'" in r.output
        assert "claude-code" in r.output

    def test_missing_database(self, tmp_path, monkeypatch):
        missing = str(tmp_path / "nope.db")
        r = self._invoke(monkeypatch, missing, "claude-code")
        assert r.exit_code == 1
        assert "No Revien database found" in r.output


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


# -- usage cursor bound --------------------------------------------------

class TestUsageCursorBound:
    def test_sync_reset_leaves_usage_cursor_alone(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.set_sync_cursor(USAGE_CURSOR, T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config",
                            lambda: {"adapters": {}, "db_path": db_path})
        CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", db_path])
        s = GraphStore(db_path=db_path)
        try:
            assert s.get_sync_cursor(USAGE_CURSOR) == T
        finally:
            s.close()

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
