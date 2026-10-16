"""Orphaned sync reset and reset/sync generation fencing (compare-and-set on
sync_cursors.generation)."""

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.cli import main
from revien.daemon.scheduler import SyncScheduler
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

T = datetime(2026, 7, 1, tzinfo=timezone.utc)
SID = "claude-code:proj-x:sess-1"


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


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


def write_session(root, text="We chose PostgreSQL for storage."):
    d = root / "projects" / "proj-x"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "sess-1.jsonl"
    with open(path, "w") as f:
        for role, t in (("user", text), ("assistant", "Noted, " + text)):
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": [{"type": "text", "text": t}]},
                "timestamp": "2026-07-01T10:00:00Z",
            }) + "\n")
    return path


def make_sched(store, root):
    sched = SyncScheduler(pipeline=IngestionPipeline(store))
    sched.register_adapter("claude-code", ClaudeCodeAdapter(session_dir=str(root)))
    return sched


def session_nodes(store):
    return [n for n in store.list_nodes(limit=9999) if n.source_id == SID]


def touch_newer(path):
    st = path.stat()
    os.utime(path, (st.st_atime + 100, st.st_mtime + 100))


# -- orphaned reset -----------------------------------------------------------

class TestOrphanedReset:
    def _run(self, monkeypatch, db_path, name, *extra, adapters=None):
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": adapters if adapters is not None else {}, "db_path": db_path})
        return CliRunner().invoke(
            main, ["sync", "reset", name, "--db", db_path, *extra])

    def test_refused_without_flag_and_points_at_it(self, store, db_path, monkeypatch):
        store.set_sync_cursor("gone", T)
        store.close()
        r = self._run(monkeypatch, db_path, "gone")
        assert r.exit_code == 1 and "--orphaned" in r.output
        s = GraphStore(db_path=db_path)
        assert s.get_sync_cursor("gone") == T
        s.close()

    def test_flag_resets_recorded_state(self, store, db_path, monkeypatch):
        store.set_sync_cursor("gone", T)
        store.close()
        r = self._run(monkeypatch, db_path, "gone", "--orphaned")
        assert r.exit_code == 0, r.output
        assert "Cleared the sync position for 'gone'" in r.output
        assert "type is unknown" in r.output
        s = GraphStore(db_path=db_path)
        assert s.get_sync_cursor("gone") is None and s.get_sync_generation("gone") == 1
        s.close()

    def test_flag_without_recorded_state_is_refused(self, store, db_path, monkeypatch):
        store.close()
        r = self._run(monkeypatch, db_path, "ghost", "--orphaned")
        assert r.exit_code == 1 and "nothing to reset" in r.output

    def test_flag_on_a_connected_adapter_is_refused(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        r = self._run(monkeypatch, db_path, "claude-code", "--orphaned",
                      adapters={"claude-code": {"type": "claude_code"}})
        assert r.exit_code == 1 and "does not apply" in r.output


# -- generation fencing --------------------------------------------------------

class TestGenerationFence:
    def test_cas_semantics(self, store):
        c1 = datetime(2026, 7, 2, tzinfo=timezone.utc)
        assert store.get_sync_state("a") == (None, 0)
        assert store.set_sync_cursor_if_generation("a", T, 0)       # absent row
        assert store.set_sync_cursor_if_generation("a", c1, 0)
        assert store.reset_sync_cursor("a") is True
        assert store.get_sync_state("a") == (None, 1)
        assert not store.set_sync_cursor_if_generation("a", c1, 0)  # stale
        assert store.get_sync_state("a") == (None, 1)
        assert store.set_sync_cursor_if_generation("a", c1, 1)
        assert store.get_sync_state("a") == (c1, 1)
        assert store.reset_sync_cursor("nope") is None

    def test_plain_set_keeps_generation(self, store):
        store.set_sync_cursor("a", T)
        store.reset_sync_cursor("a")
        store.set_sync_cursor("a", T)
        assert store.get_sync_generation("a") == 1

    def test_reset_during_sync_is_not_undone(self, store, tmp_path):
        """sync starts -> reset happens -> sync finishes: cursor stays reset."""
        write_session(tmp_path)
        sched = make_sched(store, tmp_path)
        run_async(sched.sync_one("claude-code"))
        assert store.get_sync_cursor("claude-code") is not None
        touch_newer(tmp_path / "projects" / "proj-x" / "sess-1.jsonl")

        adapter = sched._adapters["claude-code"]
        real_fetch = adapter.fetch_new_content

        async def fetch_then_reset(since):
            items = await real_fetch(since)
            # the reset lands while the sync is mid-flight (a separate
            # connection, like the CLI in another process)
            other = GraphStore(db_path=store.db_path)
            try:
                assert other.reset_sync_cursor("claude-code") is True
            finally:
                other.close()
            return items

        adapter.fetch_new_content = fetch_then_reset
        res = run_async(sched.sync_one("claude-code"))
        assert res["status"] == "reset_during_sync"
        assert store.get_sync_cursor("claude-code") is None  # reset stands
        assert store.get_sync_generation("claude-code") == 1

        # the next sync starts from EPOCH and persists normally
        adapter.fetch_new_content = real_fetch
        res = run_async(sched.sync_one("claude-code"))
        assert res["status"] == "ok"
        assert store.get_sync_cursor("claude-code") is not None

    def test_reset_during_empty_sync_is_not_undone(self, store, tmp_path):
        write_session(tmp_path)
        sched = make_sched(store, tmp_path)
        run_async(sched.sync_one("claude-code"))
        adapter = sched._adapters["claude-code"]

        async def empty_then_reset(since):
            store.reset_sync_cursor("claude-code")
            return []

        adapter.fetch_new_content = empty_then_reset
        assert run_async(sched.sync_one("claude-code"))["status"] == "reset_during_sync"
        assert store.get_sync_cursor("claude-code") is None

    def test_cli_reset_bumps_generation(self, store, db_path, monkeypatch):
        store.set_sync_cursor("claude-code", T)
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {"claude-code": {"type": "claude_code"}}, "db_path": db_path})
        r = CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", db_path])
        assert r.exit_code == 0, r.output
        s = GraphStore(db_path=db_path)
        assert s.get_sync_generation("claude-code") == 1
        s.close()

    def test_existing_database_migrates_in_place(self, tmp_path):
        p = tmp_path / "pre.db"
        s = GraphStore(db_path=str(p))
        s.close()
        c = sqlite3.connect(p)
        c.execute("DROP TABLE sync_cursors")
        c.execute("CREATE TABLE sync_cursors (adapter_name TEXT PRIMARY KEY, "
                  "cursor TEXT NOT NULL, updated_at TEXT NOT NULL)")
        c.execute("INSERT INTO sync_cursors VALUES ('x', ?, ?)",
                  (T.isoformat(), T.isoformat()))
        c.commit()
        c.close()
        s = GraphStore(db_path=str(p))
        try:
            assert s.get_sync_state("x") == (T, 0)
            assert s.reset_sync_cursor("x") is True
            assert s.get_sync_generation("x") == 1
        finally:
            s.close()
        s = GraphStore(db_path=str(p))  # idempotent second open
        s.close()
