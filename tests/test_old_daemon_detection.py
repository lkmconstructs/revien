"""Operations that depend on sync-reset fencing refuse while a daemon from
before fencing may be running, and say "restart required".

Cases: a current daemon (real uvicorn app with heartbeat), an old daemon (one
that answers /v1/health without a features list), no daemon at all, a
non-loopback --daemon-url, plus the heartbeat rows themselves.
"""
import json
import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon import heartbeat as hb
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.store import GraphStore

pytestmark = pytest.mark.real_probe

T = datetime(2026, 7, 1, tzinfo=timezone.utc)
CC = {"claude-code": {"type": "claude_code", "session_dir": "/nonexistent"}}
OLD_HEALTH = {"status": "healthy", "node_count": 1, "edge_count": 0,
              "uptime_seconds": 5.0, "version": "0.3.0"}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "revien.db")


@pytest.fixture
def seeded(db_path):
    s = GraphStore(db_path=db_path)
    s.set_sync_cursor("claude-code", T)
    s.close()
    return db_path


def fake_daemon(health):
    """A stand-in HTTP server on loopback answering /v1/health."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(health).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, args=(0.02,), daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def reset(monkeypatch, db_path, *extra, adapters=None):
    monkeypatch.setattr("revien.cli._load_config", lambda: {
        "adapters": CC if adapters is None else adapters, "db_path": db_path})
    return CliRunner().invoke(
        main, ["sync", "reset", "claude-code", "--db", db_path, *extra])


def cursor_state(db_path):
    s = GraphStore(db_path=db_path)
    try:
        return s.get_sync_state("claude-code")
    finally:
        s.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TestNoDaemon:
    def test_nothing_answering_means_proceed(self, monkeypatch, seeded):
        r = reset(monkeypatch, seeded, "--daemon-url",
                  f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 0, r.output
        assert "restart" not in r.output.lower()
        assert cursor_state(seeded) == (None, 1)


class TestOldDaemon:
    def test_an_old_daemon_is_refused_with_restart_required(
            self, monkeypatch, seeded):
        httpd, url = fake_daemon(OLD_HEALTH)
        try:
            r = reset(monkeypatch, seeded, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 1
        assert "Restart required" in r.output
        assert "before sync-reset protection existed" in r.output
        assert "Nothing was changed." in r.output
        assert cursor_state(seeded) == (T, 0)  # untouched

    def test_orphaned_resets_are_refused_too(self, monkeypatch, db_path):
        s = GraphStore(db_path=db_path)
        s.set_sync_cursor("gone", T)
        s.close()
        httpd, url = fake_daemon(OLD_HEALTH)
        try:
            monkeypatch.setattr("revien.cli._load_config", lambda: {
                "adapters": {}, "db_path": db_path})
            r = CliRunner().invoke(main, [
                "sync", "reset", "gone", "--orphaned", "--db", db_path,
                "--daemon-url", url])
        finally:
            httpd.shutdown()
        assert r.exit_code == 1 and "Restart required" in r.output
        s = GraphStore(db_path=db_path)
        assert s.get_sync_state("gone") == (T, 0)
        s.close()

    def test_refusal_happens_before_the_database_is_migrated(
            self, monkeypatch, tmp_path):
        """A database made by main (user_version 4, no generation column) is
        left exactly as it was: nothing opened a GraphStore."""
        path = str(tmp_path / "main.db")
        s = GraphStore(db_path=path)
        s.close()
        c = sqlite3.connect(path)
        c.execute("DROP TABLE sync_cursors")
        c.execute("CREATE TABLE sync_cursors (adapter_name TEXT PRIMARY KEY, "
                  "cursor TEXT NOT NULL, updated_at TEXT NOT NULL)")
        c.execute("INSERT INTO sync_cursors VALUES ('claude-code', ?, ?)",
                  (T.isoformat(), T.isoformat()))
        c.execute("PRAGMA user_version = 4")
        c.commit()
        c.close()
        before = open(path, "rb").read()
        httpd, url = fake_daemon(OLD_HEALTH)
        try:
            r = reset(monkeypatch, path, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 1 and "Restart required" in r.output
        assert open(path, "rb").read() == before

    def test_a_daemon_that_reports_features_without_the_fence_is_refused(
            self, monkeypatch, seeded):
        httpd, url = fake_daemon({**OLD_HEALTH, "features": ["something_else"]})
        try:
            r = reset(monkeypatch, seeded, "--daemon-url", url)
        finally:
            httpd.shutdown()
        assert r.exit_code == 1 and "Restart required" in r.output
        assert cursor_state(seeded) == (T, 0)

    def test_after_restart_the_reset_goes_through(self, monkeypatch, seeded):
        httpd, url = fake_daemon(OLD_HEALTH)
        try:
            assert reset(monkeypatch, seeded, "--daemon-url", url).exit_code == 1
        finally:
            httpd.shutdown()
        httpd2, url2 = fake_daemon({**OLD_HEALTH, "features": list(hb.DAEMON_FEATURES)})
        try:
            r = reset(monkeypatch, seeded, "--daemon-url", url2)
        finally:
            httpd2.shutdown()
        assert r.exit_code == 0, r.output
        assert cursor_state(seeded) == (None, 1)


class TestNonLoopback:
    @pytest.mark.parametrize("url", [
        "http://192.0.2.1:7437", "http://10.0.0.5", "http://192.168.1.2",
        "http://[::ffff:10.0.0.5]:80", "http://0.0.0.0:7437",
        "http://[2001:db8::1]:80", "https://127.0.0.1:7437",
        "http://example.com", "http://my-box.test:7437"])
    def test_a_non_loopback_url_refuses_the_reset_and_is_never_contacted(
            self, monkeypatch, seeded, url):
        made, resolved = [], []

        class NoSocket:
            def __init__(self, *a, **k):
                made.append(1)
                raise AssertionError("a socket was created")

        def no_resolve(*a, **k):
            resolved.append(a)
            raise AssertionError("the resolver was called")

        monkeypatch.setattr(socket, "socket", NoSocket)
        monkeypatch.setattr(socket, "create_connection", NoSocket)
        monkeypatch.setattr(socket, "getaddrinfo", no_resolve)
        r = reset(monkeypatch, seeded, "--daemon-url", url)
        assert (made, resolved) == ([], [])
        assert r.exit_code == 1, r.output
        assert "Use a loopback --daemon-url" in r.output
        assert "Nothing was changed." in r.output
        monkeypatch.undo()
        assert cursor_state(seeded) == (T, 0)

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:99999", "http://127.0.0.1:abc", "http://127.0.0.1:-1",
        "http://127.0.0.1:0"])
    def test_a_bad_port_refuses_the_reset(self, monkeypatch, seeded, url):
        r = reset(monkeypatch, seeded, "--daemon-url", url)
        assert r.exit_code == 1, r.output
        assert "Use a loopback --daemon-url" in r.output
        assert "Nothing was changed." in r.output
        assert cursor_state(seeded) == (T, 0)

    def test_localhost_and_loopback_literals_are_accepted(
            self, monkeypatch, seeded):
        for url in ("http://localhost:%d" % free_port(),
                    "http://127.0.0.1:%d" % free_port()):
            r = reset(monkeypatch, seeded, "--daemon-url", url)
            assert r.exit_code == 0, r.output
            seeded_again = GraphStore(db_path=seeded)
            seeded_again.set_sync_cursor("claude-code", T)
            seeded_again.close()


class _CurrentDaemon:
    """A real current daemon: create_app + scheduler under uvicorn, on an
    ephemeral loopback port."""

    def __init__(self, db_path):
        import uvicorn
        self.app = create_app(db_path=db_path)
        self.app.state.scheduler = SyncScheduler(
            pipeline=self.app.state.pipeline, interval_hours=6.0)
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=0,
                             log_level="error", lifespan="on")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(300):
            if self.server.started:
                break
            time.sleep(0.02)
        assert self.server.started
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


class TestCurrentDaemon:
    def test_a_current_daemon_is_accepted_and_the_reset_takes_effect(
            self, monkeypatch, db_path):
        s = GraphStore(db_path=db_path)
        s.set_sync_cursor("claude-code", T)
        s.close()
        d = _CurrentDaemon(db_path)
        try:
            from revien.loopback_probe import probe_daemon
            seen = probe_daemon(d.url)
            assert seen and set(hb.RESET_REQUIRES) <= set(seen["features"])
            r = reset(monkeypatch, db_path, "--daemon-url", d.url)
        finally:
            d.stop()
        assert r.exit_code == 0, r.output
        assert "Restart required" not in r.output
        assert cursor_state(db_path) == (None, 1)

    def test_the_daemon_wrote_a_heartbeat_and_removed_it_on_exit(self, db_path):
        d = _CurrentDaemon(db_path)
        try:
            time.sleep(0.2)
            s = GraphStore(db_path=db_path)
            beats = hb.live_heartbeats(s)
            s.close()
            assert len(beats) == 1
            assert beats[0]["pid"] == os.getpid()
            assert set(hb.DAEMON_FEATURES) == set(beats[0]["features"])
        finally:
            d.stop()
        s = GraphStore(db_path=db_path)
        try:
            assert hb.live_heartbeats(s) == []
            assert s.list_meta(hb.HEARTBEAT_PREFIX) == {}
        finally:
            s.close()

    def test_a_sync_running_during_the_reset_cannot_undo_it(
            self, monkeypatch, db_path):
        """The reason for the whole check, end to end on the current build."""
        import asyncio
        from revien.adapters.base import RevienAdapter
        from revien.ingestion.pipeline import IngestionPipeline

        store = GraphStore(db_path=db_path)
        store.set_sync_cursor("claude-code", T)

        class Slow(RevienAdapter):
            IDEMPOTENT_REREAD = True

            async def fetch_new_content(self, since):
                # the reset lands while this fetch is in flight
                r = reset(monkeypatch, db_path)
                assert r.exit_code == 0, r.output
                return []

            async def health_check(self):
                return True

        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("claude-code", Slow())
        res = asyncio.new_event_loop().run_until_complete(
            sched.sync_one("claude-code"))
        assert res["status"] == "reset_during_sync"
        assert store.get_sync_state("claude-code") == (None, 1)
        store.close()


class TestHealthEndpoint:
    def test_reports_features_and_nothing_path_derived(self, tmp_path):
        path = str(tmp_path / "secret-name-revien.db")
        with TestClient(create_app(db_path=path)) as client:
            body = client.get("/v1/health").json()
            raw = client.get("/v1/health").text
        assert body["features"] == list(hb.DAEMON_FEATURES)
        assert set(body) == {"status", "node_count", "edge_count",
                             "uptime_seconds", "version", "features"}
        assert "secret-name" not in raw and str(tmp_path) not in raw
        assert "db_id" not in body

    def test_no_heartbeat_without_a_scheduler(self, tmp_path):
        """Bare create_app (tests, ASGI) is not a sync daemon."""
        path = str(tmp_path / "r.db")
        with TestClient(create_app(db_path=path)):
            s = GraphStore(db_path=path)
            try:
                assert s.list_meta(hb.HEARTBEAT_PREFIX) == {}
            finally:
                s.close()


class TestHeartbeatRows:
    def test_write_refresh_clear(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "r.db"))
        try:
            hb.write_heartbeat(s)
            hb.write_heartbeat(s)
            assert len(hb.live_heartbeats(s)) == 1
            hb.clear_heartbeat(s)
            assert hb.live_heartbeats(s) == []
        finally:
            s.close()

    def test_stale_and_garbled_rows_are_ignored_and_dropped(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "r.db"))
        try:
            old = datetime.now(timezone.utc) - timedelta(seconds=hb.STALE_AFTER + 5)
            s.set_meta(hb.HEARTBEAT_PREFIX + "111", json.dumps({
                "features": list(hb.DAEMON_FEATURES), "pid": 111,
                "time": old.isoformat()}))
            s.set_meta(hb.HEARTBEAT_PREFIX + "222", "not json")
            assert hb.live_heartbeats(s) == []
            hb.write_heartbeat(s)
            assert set(s.list_meta(hb.HEARTBEAT_PREFIX)) == {
                hb.heartbeat_key()}
        finally:
            s.close()

    def test_a_fresh_heartbeat_without_the_feature_refuses(
            self, monkeypatch, seeded):
        s = GraphStore(db_path=seeded)
        s.set_meta(hb.HEARTBEAT_PREFIX + "4242", json.dumps({
            "features": ["something_else"], "pid": 4242,
            "time": datetime.now(timezone.utc).isoformat()}))
        s.close()
        r = reset(monkeypatch, seeded, "--daemon-url",
                  f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 1
        assert "Restart required" in r.output and "pid 4242" in r.output
        assert cursor_state(seeded) == (T, 0)

    def test_a_fresh_heartbeat_with_the_feature_does_not_refuse(
            self, monkeypatch, seeded):
        s = GraphStore(db_path=seeded)
        hb.write_heartbeat(s)
        s.close()
        r = reset(monkeypatch, seeded, "--daemon-url",
                  f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 0, r.output

    def test_a_stale_heartbeat_does_not_refuse(self, monkeypatch, seeded):
        s = GraphStore(db_path=seeded)
        old = datetime.now(timezone.utc) - timedelta(seconds=hb.STALE_AFTER * 3)
        s.set_meta(hb.HEARTBEAT_PREFIX + "4242", json.dumps({
            "features": [], "pid": 4242, "time": old.isoformat()}))
        s.close()
        r = reset(monkeypatch, seeded, "--daemon-url",
                  f"http://127.0.0.1:{free_port()}")
        assert r.exit_code == 0, r.output

    def test_a_database_without_store_meta_has_no_heartbeats(self, tmp_path):
        p = tmp_path / "old.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE nodes (node_id TEXT)")
        c.commit()
        c.close()
        assert hb.live_heartbeats_readonly(f"file:{p}?mode=ro") == []
