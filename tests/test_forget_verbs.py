"""Three verbs, never folded together (docs/FORGET_SPEC.md §1, invariant 4),
and the old-daemon refusal and generation fencing around forget (§8)."""
import json
import os
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon import heartbeat as hb
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests import forget_seed as fs
from tests.test_old_daemon_detection import OLD_HEALTH, fake_daemon, free_port

T = datetime(2026, 7, 1, tzinfo=timezone.utc)


def ops_of(db, op):
    s = GraphStore(db_path=db)
    try:
        return [a for a in s.get_all_audit() if a["op"] == op]
    finally:
        s.close()


def target(seeded):
    """The node a user would forget to take the secret out."""
    return fs.holders(seeded.db, seeded.secret)[0]


def nodes(db):
    s = GraphStore(db_path=db)
    try:
        return s.list_nodes(limit=1000)
    finally:
        s.close()


class TestDelete:
    def test_delete_writes_a_delete_entry_and_no_suppression(self, tmp_path, monkeypatch):
        """Invariant 4 (delete): REST DELETE is operational."""
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.delete(f"/v1/nodes/{seeded.ctx_ids[0]}")
            assert r.status_code == 200 and r.json()["status"] == "deleted"
        s = GraphStore(db_path=seeded.db)
        try:
            assert s.list_suppressions(None) == []
            entries = [a for a in s.get_node_audit(seeded.ctx_ids[0]) if a["op"] == "delete"]
            assert len(entries) == 1 and entries[0]["actor"] == "api:delete"
            assert entries[0]["before"] is None
            assert seeded.secret not in json.dumps(entries[0])
            assert not any(a["op"] == "forget" for a in s.get_all_audit())
        finally:
            s.close()

    def test_a_deleted_unit_can_be_captured_again(self, tmp_path, monkeypatch):
        """That is what makes delete a different verb from forget."""
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        for n in s.list_nodes(limit=100):
            GraphOperations(s).delete_node(n.node_id)
        s.close()
        assert nodes(seeded.db) == []
        seeded.recapture()
        assert nodes(seeded.db)

    def test_delete_of_a_missing_node_is_a_404(self, tmp_path):
        with TestClient(create_app(db_path=str(tmp_path / "d.db"))) as client:
            assert client.delete("/v1/nodes/nope").status_code == 404

    def test_langchain_clear_is_delete_not_forget(self, tmp_path):
        from revien.adapters.langchain_adapter import LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("langchain-core is not installed")
        from revien.adapters import RevienMemory
        db = str(tmp_path / "lc.db")
        with RevienMemory(graph_path=db, session_scope="s1") as mem:
            mem.save_context({"input": "remember that I prefer tabs."}, {"output": "Noted."})
            mem.clear()
        assert ops_of(db, "delete") and not ops_of(db, "forget")
        s = GraphStore(db_path=db)
        try:
            assert s.list_suppressions(None) == []
        finally:
            s.close()


class TestSweepsAndCleanup:
    def _stale(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "s.db"))
        old = datetime.now(timezone.utc) - timedelta(days=400)
        n = s.add_node(Node(node_type=NodeType.FACT, label="stale", content="STALE WORDS",
                            created_at=old, last_referenced=old))
        return s, n

    def test_user_triggered_sweep_is_delete(self, tmp_path):
        """Invariant 4 (user-triggered sweep): REST /v1/retention/sweep."""
        db = str(tmp_path / "sw.db")
        s = GraphStore(db_path=db)
        old = datetime.now(timezone.utc) - timedelta(days=400)
        n = s.add_node(Node(node_type=NodeType.FACT, label="stale", content="STALE WORDS",
                            created_at=old, last_referenced=old))
        s.close()
        with TestClient(create_app(db_path=db)) as client:
            r = client.post("/v1/retention/sweep", json={"mode": "expire", "days": 90})
            assert r.status_code == 200 and r.json()["expired"] == 1
        assert [a["actor"] for a in ops_of(db, "delete")] == ["api:retention_sweep"]
        assert not ops_of(db, "forget") and not ops_of(db, "cleanup")
        s = GraphStore(db_path=db)
        try:
            assert s.list_suppressions(None) == []
            assert "STALE WORDS" not in json.dumps(ops_of(db, "delete"))
        finally:
            s.close()

    def test_automatic_cleanup_is_its_own_verb(self, tmp_path):
        """Invariant 4 (automatic cleanup)."""
        s, n = self._stale(tmp_path)
        res = GraphOperations(s).run_scheduled_cleanup(mode="expire", days=90)
        assert res["expired"] == 1
        audit = s.get_node_audit(n.node_id)
        assert [a["op"] for a in audit if a["op"] in ("cleanup", "delete", "forget")] == ["cleanup"]
        assert [a["actor"] for a in audit if a["op"] == "cleanup"] == ["scheduler:retention"]
        assert s.list_suppressions(None) == []
        s.close()

    def test_only_forget_writes_a_suppression_record(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post(f"/v1/nodes/{target(seeded)}/forget", json={})
            assert r.status_code == 200, r.text
            body = r.json()
        assert body["verb"] == "forget" and body["bound_to"]
        s = GraphStore(db_path=seeded.db)
        try:
            assert len(s.list_suppressions("active")) == 1
        finally:
            s.close()
        assert ops_of(seeded.db, "forget")

    def test_forget_rest_not_found_and_refusal(self, tmp_path):
        db = str(tmp_path / "r.db")
        s = GraphStore(db_path=db)
        from revien.graph.schema import Node
        ctx = s.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: my code is 4471"),
                         owner=("session:claude-code:A", 1, 1))
        para = s.add_node(Node(node_type=NodeType.FACT, label="combo", content="four four seven one"),
                          owner=("session:claude-code:A", 1, 1))
        s.close()
        with TestClient(create_app(db_path=db)) as client:
            assert client.post("/v1/nodes/nope/forget", json={}).status_code == 404
            r = client.post(f"/v1/nodes/{para.node_id}/forget", json={})
            assert r.status_code == 409
            detail = r.json()["detail"]
            assert detail["status"] == "refused" and ctx.node_id in detail["message"]
            # a conversation is removed with delete, not forget
            whole = "To remove a whole conversation or file, use delete; to keep a specific detail out, forget that detail."
            r = client.post(f"/v1/nodes/{ctx.node_id}/forget", json={})
            assert r.status_code == 409 and r.json()["detail"]["message"] == whole
            prev = client.get(f"/v1/nodes/{ctx.node_id}/forget/preview")
            assert prev.status_code == 200 and prev.json()["status"] == "refused"
            assert prev.json()["message"] == whole
            assert client.delete(f"/v1/nodes/{ctx.node_id}").status_code == 200


@pytest.mark.real_probe
class TestForgetRefusesWhileAnOldDaemonMayBeRunning:
    def _setup(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {}, "db_path": seeded.db})
        return seeded

    def _forget(self, seeded, *extra):
        return CliRunner().invoke(main, [
            "forget", target(seeded), "--yes", "--db", seeded.db, *extra])

    def test_a_daemon_without_a_features_list_is_refused(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        httpd, url = fake_daemon(OLD_HEALTH)
        try:
            r = self._forget(seeded, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 1
        assert "Restart required" in r.output and "forget protection" in r.output
        assert "Nothing was changed." in r.output
        assert len(nodes(seeded.db)) == 3 and not ops_of(seeded.db, "forget")

    def test_a_daemon_with_the_reset_fence_but_no_forget_gate_is_refused(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        httpd, url = fake_daemon({**OLD_HEALTH, "features": ["sync_reset_fence", "usage_receipts"]})
        try:
            r = self._forget(seeded, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 1 and "does not report forget protection" in r.output
        assert len(nodes(seeded.db)) == 3

    def test_a_stale_heartbeat_in_the_database_is_refused(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        s.set_meta("daemon_heartbeat:4242", json.dumps({
            "features": ["sync_reset_fence"], "pid": 4242,
            "time": datetime.now(timezone.utc).isoformat()}))
        s.close()
        r = self._forget(seeded, "--daemon-url", f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 1 and "pid 4242" in r.output and "Restart required" in r.output
        assert len(nodes(seeded.db)) == 3

    def test_a_non_loopback_daemon_url_is_refused(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        r = self._forget(seeded, "--daemon-url", "http://example.invalid:7437")
        assert r.exit_code == 1 and "loopback" in r.output
        assert len(nodes(seeded.db)) == 3

    def test_no_daemon_means_proceed(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        r = self._forget(seeded, "--daemon-url", f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 0, r.output
        assert "keeps its wording out of later captures" in r.output
        assert [n.node_type.value for n in nodes(seeded.db)] == ["context"]

    def test_a_current_daemon_is_not_a_reason_to_refuse(self, tmp_path, monkeypatch):
        seeded = self._setup(tmp_path, monkeypatch)
        httpd, url = fake_daemon({**OLD_HEALTH, "features": list(hb.DAEMON_FEATURES)})
        try:
            r = self._forget(seeded, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 0, r.output

    def test_the_rest_route_refuses_when_another_old_daemon_serves_the_database(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        s.set_meta("daemon_heartbeat:4242", json.dumps({
            "features": ["sync_reset_fence"], "pid": 4242,
            "time": datetime.now(timezone.utc).isoformat()}))
        s.close()
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post(f"/v1/nodes/{target(seeded)}/forget", json={})
            assert r.status_code == 409
            assert r.json()["detail"]["reason"] == "restart_required"
        assert len(nodes(seeded.db)) == 3

    def test_features_a_current_daemon_reports(self, tmp_path):
        assert set(hb.FORGET_REQUIRES) <= set(hb.DAEMON_FEATURES)
        with TestClient(create_app(db_path=str(tmp_path / "h.db"))) as client:
            assert "forget_suppression" in client.get("/v1/health").json()["features"]


class TestFencingAndSyncReset:
    def test_a_sync_that_started_before_the_forget_cannot_write_the_wording_back(self, tmp_path, monkeypatch):
        """The forget lands between a sync's fetch and its write: the gate is
        asked at write time, so the wording does not come back."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        from revien.daemon.scheduler import SyncScheduler
        from revien.ingestion.pipeline import IngestionPipeline
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        adapter = ClaudeCodeAdapter(session_dir=str(seeded.extra["root"] / "projects"))
        store = GraphStore(db_path=seeded.db)
        try:
            store.delete_sync_cursor("claude-code")
            sched = SyncScheduler(IngestionPipeline(store))
            sched.register_adapter("claude-code", adapter)
            real_fetch = adapter.fetch_new_content

            async def fetch_then_forget(since):
                items = await real_fetch(since)          # the sync has its items
                GraphOperations(store).forget_node(target(seeded))
                return items                             # ...and writes them now

            adapter.fetch_new_content = fetch_then_forget
            result = fs.run_async(sched.sync_one("claude-code"))
            assert result["items_ingested"] == 1 and not result.get("items_suppressed")
            assert all(seeded.secret not in n.label + n.content
                       for n in store.list_nodes(limit=100))
            assert store.count_nodes() >= 1
        finally:
            store.close()

    def test_a_sync_counts_a_capture_refused_whole_as_suppressed_not_ingested(self, tmp_path, monkeypatch):
        """Only a capture that IS wording banned by forget --everywhere is
        refused whole; the scheduler says so."""
        from revien.daemon.scheduler import SyncScheduler
        from revien.ingestion.pipeline import IngestionPipeline
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        wording = fs.decision_text(seeded.db, seeded.secret)
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(target(seeded), everywhere=True)

        class Feed:
            async def fetch_new_content(self, since):
                return [{"content": wording, "source_id": "feed-1"},
                        {"content": "A plain remark about lunch plans.", "source_id": "feed-2"}]

            async def health_check(self):
                return True

        try:
            sched = SyncScheduler(IngestionPipeline(s))
            sched.register_adapter("feed", Feed())
            result = fs.run_async(sched.sync_one("feed"))
        finally:
            s.close()
        assert result["items_suppressed"] == 1 and result["items_ingested"] == 1
        assert all(seeded.secret not in n.label + n.content for n in nodes(seeded.db))

    def _cfg(self, monkeypatch, db):
        monkeypatch.setattr("revien.cli._load_config", lambda: {
            "adapters": {"claude-code": {"type": "claude_code", "session_dir": "/nonexistent"}},
            "db_path": db})

    def test_sync_reset_needs_yes_once_a_forget_exists(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        self._cfg(monkeypatch, seeded.db)
        # before any forget: no --yes needed
        ok = CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", seeded.db,
                                       "--daemon-url", f"http://127.0.0.1:{free_port()}"])
        assert ok.exit_code == 0, ok.output
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(target(seeded))
        s.close()
        before = GraphStore(db_path=seeded.db)
        gen = before.get_sync_state("claude-code")
        before.close()
        refused = CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", seeded.db,
                                            "--daemon-url", f"http://127.0.0.1:{free_port()}"])
        assert refused.exit_code == 1 and "--yes" in refused.output
        assert "Nothing was changed." in refused.output
        s = GraphStore(db_path=seeded.db)
        assert s.get_sync_state("claude-code") == gen
        s.close()
        done = CliRunner().invoke(main, ["sync", "reset", "claude-code", "--db", seeded.db, "--yes",
                                         "--daemon-url", f"http://127.0.0.1:{free_port()}"])
        assert done.exit_code == 0, done.output

    def test_a_reset_then_a_full_reread_still_keeps_the_wording_out(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(target(seeded))
        s.reset_sync_cursor("claude-code")
        s.close()
        seeded.recapture()
        assert nodes(seeded.db)
        assert all(seeded.secret not in n.label + n.content for n in nodes(seeded.db))


def test_secure_delete_is_switched_on_for_every_connection(tmp_path, monkeypatch):
    """docs/FORGET_SPEC.md §4: freed pages are overwritten, not left behind.
    Some SQLite builds default it on, so the test reads what Revien asks for."""
    import sqlite3
    asked = []

    class Recording(sqlite3.Connection):
        def execute(self, sql, *a, **kw):
            asked.append(" ".join(str(sql).split()).lower())
            return super().execute(sql, *a, **kw)

    real = sqlite3.connect
    monkeypatch.setattr("revien.graph.store.sqlite3.connect",
                        lambda *a, **kw: real(*a, factory=Recording, **kw))
    s = GraphStore(db_path=str(tmp_path / "sd.db"))
    try:
        s.count_nodes()
    finally:
        s.close()
    assert "pragma secure_delete = on" in asked


def test_forget_reports_what_it_did_to_the_storage(tmp_path, monkeypatch):
    seeded = fs.seed("pipeline", tmp_path, monkeypatch)
    s = GraphStore(db_path=seeded.db)
    try:
        res = GraphOperations(s).forget_node(target(seeded))
    finally:
        s.close()
    assert res["storage"]["secure_delete"] is True
    assert res["storage"]["vacuumed"] is True and res["storage"]["wal_truncated"] is True
    wal = Path(seeded.db + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
