"""Suppression tombstones, orphaned reset, and reset/sync generation fencing."""

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.cli import main
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Graph, Node, NodeType
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


# -- suppression on forget ---------------------------------------------------

class TestForgetSuppresses:
    def _synced(self, store, tmp_path):
        path = write_session(tmp_path)
        sched = make_sched(store, tmp_path)
        assert run_async(sched.sync_one("claude-code"))["items_ingested"] == 1
        return path, sched

    def test_user_forget_then_changed_file_is_not_recaptured(self, store, tmp_path):
        path, sched = self._synced(store, tmp_path)
        ctx = [n for n in session_nodes(store) if n.node_type == NodeType.CONTEXT][0]
        out = GraphOperations(store).forget_node(
            ctx.node_id, reason="private", construct_id="me")
        assert SID in out["suppressed_identities"]
        assert session_nodes(store) == [] or all(
            n.node_id != ctx.node_id for n in session_nodes(store))
        before = store.count_nodes()

        # the session file CHANGES (grows) and is synced: the identity is
        # stable across that, so it is skipped, logged and counted
        write_session(tmp_path, "We chose PostgreSQL. Also secret plan zeta.")
        touch_newer(path)
        res = run_async(sched.sync_one("claude-code"))
        assert res["status"] == "ok"
        assert res["items_suppressed"] == 1 and res["items_ingested"] == 0
        assert store.count_nodes() == before
        row = store.list_ingest_suppressions()
        assert row and max(r["skipped_count"] for r in row) >= 1

    def test_suppression_row_is_content_free(self, store, tmp_path):
        self._synced(store, tmp_path)
        ctx = [n for n in session_nodes(store) if n.node_type == NodeType.CONTEXT][0]
        GraphOperations(store).forget_node(ctx.node_id, reason="private")
        conn = store._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ingest_suppressions)")}
        assert cols == {"identity", "kind", "created_at", "reason", "actor",
                        "skipped_count", "last_skipped_at"}
        dump = json.dumps([list(r) for r in conn.execute(
            "SELECT * FROM ingest_suppressions")])
        assert "PostgreSQL" not in dump
        audit = json.dumps([a for a in store.get_all_audit()
                            if a["node_id"] in (ctx.node_id, f"suppression:{SID}")])
        assert "PostgreSQL" not in audit

    def test_forgetting_one_extracted_node_suppresses_its_whole_session(
            self, store, tmp_path):
        path, sched = self._synced(store, tmp_path)
        extracted = [n for n in session_nodes(store)
                     if n.node_type != NodeType.CONTEXT]
        assert extracted, "extractor produced nothing; test proves nothing"
        out = GraphOperations(store).forget_node(extracted[0].node_id, reason="x")
        assert SID in out["suppressed_identities"]
        # the session's other nodes stay (only future capture stops)
        assert any(n.node_type == NodeType.CONTEXT for n in session_nodes(store))
        write_session(tmp_path, "brand new unrelated turn")
        touch_newer(path)
        assert run_async(sched.sync_one("claude-code"))["items_suppressed"] == 1

    def test_retention_expiry_never_suppresses(self, store):
        n = Node(node_type=NodeType.FACT, label="old", content="old fact",
                 source_id=SID,
                 last_accessed=datetime(2020, 1, 1, tzinfo=timezone.utc),
                 created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        store.add_node(n)
        ops = GraphOperations(store)
        ops.apply_retention(mode="expire", days=1)
        assert store.get_node(n.node_id) is None  # it really expired
        assert store.list_ingest_suppressions() == []
        n2 = Node(node_type=NodeType.FACT, label="o2", content="o2", source_id=SID)
        store.add_node(n2)
        ops.forget_node(n2.node_id, reason="retention_expire")
        assert store.list_ingest_suppressions() == []

    def test_shared_channel_source_cannot_be_tombstoned_and_says_so(self, store):
        n = Node(node_type=NodeType.FACT, label="m", content="c", source_id="mcp")
        store.add_node(n)
        out = GraphOperations(store).forget_node(n.node_id, reason="x")
        assert out["suppressed_identities"] == []
        assert out["suppression_unavailable"][0]["node_id"] == n.node_id
        assert store.list_ingest_suppressions() == []
        # ...so "mcp" ingestion is not blocked
        pipe = IngestionPipeline(store)
        r = pipe.ingest(IngestionInput(source_id="mcp", content="We use Redis."))
        assert not r.suppressed

    def test_keyed_identity_survives_a_source_id_change(self, store):
        """ingest_key is bound too, so a renamed source_id with the same key
        is still suppressed."""
        pipe = IngestionPipeline(store)
        pipe.ingest(IngestionInput(source_id="claude-code:p:s", content="We use Redis.",
                                   ingest_key="key-1"))
        ctx = store.find_context_node_by_ingest_key("key-1")
        GraphOperations(store).forget_node(ctx.node_id, reason="x")
        r = pipe.ingest(IngestionInput(source_id="claude-code:other:s",
                                       content="We use Redis. More.",
                                       ingest_key="key-1"))
        assert r.suppressed and r.nodes_created == 0

    def test_file_watcher_style_identity_is_stable_across_changes(self, store):
        pipe = IngestionPipeline(store)
        pipe.ingest(IngestionInput(source_id="file:notes.md", content="We use Redis."))
        n = [x for x in store.list_nodes(limit=99) if x.source_id == "file:notes.md"][0]
        GraphOperations(store).forget_node(n.node_id, reason="x")
        r = pipe.ingest(IngestionInput(source_id="file:notes.md",
                                       content="Totally different edited text."))
        assert r.suppressed
        r2 = pipe.ingest(IngestionInput(source_id="file:other.md", content="Hi there."))
        assert not r2.suppressed

    def test_mcp_and_rest_ingest_are_also_gated(self, store):
        pipe = IngestionPipeline(store)
        store.add_ingest_suppression(SID, "source_id", "x", "me")
        r = pipe.ingest(IngestionInput(source_id=SID, content="anything"))
        assert r.suppressed and r.suppressed_identity == SID
        assert store.count_nodes() == 0

    def test_forget_marks_audit_status_for_warning(self, store):
        n = Node(node_type=NodeType.FACT, label="a", content="b", source_id=SID)
        store.add_node(n)
        GraphOperations(store).forget_node(n.node_id, reason="x")
        assert store.count_unprotected_forgets() == 0
        m = Node(node_type=NodeType.FACT, label="a2", content="b2", source_id="mcp")
        store.add_node(m)
        GraphOperations(store).forget_node(m.node_id, reason="x")
        assert store.count_unprotected_forgets() == 1
        store.record_audit("legacy", "forget", after={"reason": "old"})
        assert store.count_unprotected_forgets() == 2


# -- restore ----------------------------------------------------------------

class TestRestore:
    def test_restore_lifts_and_is_audited(self, store, tmp_path):
        store.add_ingest_suppression(SID, "source_id", "x", "me")
        store.add_ingest_suppression(SID, "ingest_key", "x", "me")
        assert store.lift_ingest_suppression(SID, actor="cli") == 2
        assert store.list_ingest_suppressions() == []
        ops = [a["op"] for a in store.get_all_audit()]
        assert "suppression_lift" in ops
        assert store.lift_ingest_suppression(SID) == 0
        r = IngestionPipeline(store).ingest(
            IngestionInput(source_id=SID, content="We use Redis."))
        assert not r.suppressed and r.nodes_created > 0

    def test_cli_restore_and_list(self, store, db_path, monkeypatch):
        store.add_ingest_suppression(SID, "source_id", "private", "me")
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {"db_path": db_path})
        r = CliRunner().invoke(main, ["suppressions", "--db", db_path])
        assert r.exit_code == 0 and SID in r.output and "private" in r.output
        r = CliRunner().invoke(main, ["restore", SID, "--db", db_path])
        assert r.exit_code == 0 and "Restored" in r.output
        r = CliRunner().invoke(main, ["restore", SID, "--db", db_path])
        assert r.exit_code == 1 and "No suppression" in r.output
        r = CliRunner().invoke(main, ["suppressions", "--db", db_path])
        assert "No suppressed sources" in r.output


# -- graph import (explicit reimport) ---------------------------------------

class TestImportOverride:
    def _graph(self):
        a = Node(node_type=NodeType.CONTEXT, label="a", content="ca", source_id=SID)
        b = Node(node_type=NodeType.FACT, label="b", content="cb", source_id="file:ok.md")
        from revien.graph.schema import Edge, EdgeType
        e = Edge(edge_type=EdgeType.RELATED_TO, source_node_id=a.node_id,
                 target_node_id=b.node_id)
        return Graph(nodes=[a, b], edges=[e]), a, b

    def test_import_skips_suppressed_by_default(self, store):
        g, a, b = self._graph()
        store.add_ingest_suppression(SID, "source_id", "x", "me")
        res = store.import_graph(g, mode="refuse")
        assert res["suppressed_nodes"] == 1 and res["suppressed_edges"] == 1
        assert res["suppressed_identities"] == [SID]
        assert store.get_node(a.node_id) is None and store.get_node(b.node_id)
        assert store.list_ingest_suppressions()[0]["skipped_count"] == 1

    def test_override_imports_and_lifts_audited(self, store):
        g, a, b = self._graph()
        store.add_ingest_suppression(SID, "source_id", "x", "me")
        res = store.import_graph(g, mode="refuse", suppressions="override",
                                 actor="cli:import")
        assert res["lifted_identities"] == [SID] and res["suppressed_nodes"] == 0
        assert store.get_node(a.node_id) is not None
        assert store.list_ingest_suppressions() == []
        assert any(r["op"] == "suppression_lift" and r["actor"] == "cli:import"
                   for r in store.get_all_audit())

    def test_cli_flag(self, store, db_path, tmp_path, monkeypatch):
        g, a, _ = self._graph()
        f = tmp_path / "g.json"
        f.write_text(json.dumps(g.model_dump(mode="json")))
        store.add_ingest_suppression(SID, "source_id", "x", "me")
        store.close()
        monkeypatch.setattr("revien.cli._load_config", lambda: {"db_path": db_path})
        r = CliRunner().invoke(main, ["import", str(f), "--db", db_path, "--merge"])
        assert r.exit_code == 0 and "--override-suppressions" in r.output
        r = CliRunner().invoke(
            main, ["import", str(f), "--db", db_path, "--merge",
                   "--override-suppressions"])
        assert r.exit_code == 0 and "Lifted suppression for: " + SID in r.output


# -- REST -------------------------------------------------------------------

class TestRest:
    @pytest.fixture
    def client(self, db_path):
        app = create_app(db_path=db_path)
        with TestClient(app) as c:
            yield c

    def test_delete_writes_tombstone_and_ingest_is_suppressed(self, client):
        body = {"source_id": SID, "content": "We chose PostgreSQL for storage."}
        r = client.post("/v1/ingest", json=body)
        nid = r.json()["context_node_id"]
        d = client.delete(f"/v1/nodes/{nid}")
        assert d.status_code == 200
        assert d.json()["suppression"] == "created"
        assert SID in d.json()["suppressed_identities"]
        again = client.post("/v1/ingest", json={**body, "content": body["content"] + " More."})
        assert again.json()["suppressed"] is True
        assert again.json()["nodes_created"] == 0
        listed = client.get("/v1/suppressions").json()["suppressions"]
        assert listed[0]["identity"] == SID and listed[0]["skipped_count"] == 1
        assert client.post("/v1/suppressions/restore",
                           json={"identity": SID}).status_code == 200
        ok = client.post("/v1/ingest", json=body)
        assert ok.json()["suppressed"] is False and ok.json()["nodes_created"] > 0

    def test_delete_of_shared_channel_node_reports_the_gap(self, client):
        r = client.post("/v1/ingest", json={"source_id": "mcp", "content": "We use Redis."})
        d = client.delete(f"/v1/nodes/{r.json()['context_node_id']}")
        assert d.json()["suppression"] == "unavailable"
        assert d.json()["suppression_gap"]
        assert client.get("/v1/suppressions").json()["suppressions"] == []

    def test_forget_endpoint_suppresses(self, client):
        r = client.post("/v1/ingest", json={"source_id": SID, "content": "We use Redis."})
        nid = r.json()["context_node_id"]
        f = client.post(f"/v1/nodes/{nid}/forget", json={"reason": "private"})
        assert SID in f.json()["suppressed_identities"]

    def test_restore_unknown_is_404(self, client):
        assert client.post("/v1/suppressions/restore",
                           json={"identity": "nope"}).status_code == 404

    def test_import_override_param(self, client):
        g = Graph(nodes=[Node(node_type=NodeType.FACT, label="a", content="c",
                              source_id=SID)], edges=[])
        client.app.state  # app built; seed a suppression through the API
        client.post("/v1/ingest", json={"source_id": SID, "content": "We use Redis."})
        nid = client.get("/v1/nodes", params={"source_id": SID}).json()
        first = (nid if isinstance(nid, list) else nid.get("nodes"))[0]["node_id"]
        client.delete(f"/v1/nodes/{first}")
        data = g.model_dump(mode="json")
        r = client.post("/v1/graph/import?mode=merge", json=data)
        assert r.json()["suppressed_nodes"] == 1
        r = client.post("/v1/graph/import?mode=merge&override_suppressions=true",
                        json=data)
        assert r.json()["nodes"] == 1 and r.json()["lifted_identities"] == [SID]


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

    def test_orphaned_reset_still_needs_yes_for_unprotected_forgets(
            self, store, db_path, monkeypatch):
        store.set_sync_cursor("gone", T)
        store.record_audit("n1", "forget", after={"reason": "old"})
        store.close()
        assert self._run(monkeypatch, db_path, "gone", "--orphaned").exit_code == 1
        assert self._run(monkeypatch, db_path, "gone", "--orphaned", "--yes").exit_code == 0


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


# -- right-to-forget still holds ----------------------------------------------

class TestNoContentSurvives:
    def test_forget_leaves_no_content_in_audit_or_suppression(self, store):
        pipe = IngestionPipeline(store)
        r = pipe.ingest(IngestionInput(source_id=SID,
                                       content="My secret code word is zeppelin."))
        GraphOperations(store).forget_node(r.context_node_id, reason="private",
                                           cascade=True)
        conn = store._get_conn()
        blob = json.dumps(store.get_all_audit()) + json.dumps(
            [list(x) for x in conn.execute("SELECT * FROM ingest_suppressions")])
        assert "zeppelin" not in blob
        # the history that the node existed is kept
        assert any(a["op"] == "forget" for a in store.get_all_audit())

    def test_rest_delete_leaves_no_content_in_audit(self, db_path):
        with TestClient(create_app(db_path=db_path)) as c:
            r = c.post("/v1/ingest", json={"source_id": SID,
                                           "content": "My secret code word is zeppelin."})
            nid = r.json()["context_node_id"]
            assert c.delete(f"/v1/nodes/{nid}").status_code == 200
        s = GraphStore(db_path=db_path)
        try:
            assert "zeppelin" not in json.dumps(s.get_all_audit())
        finally:
            s.close()
