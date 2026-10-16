"""Follow-ups to suppression tombstones: stable identities per adapter, the
OpenAI/skills gates, the retention channel, privacy-grade scrubbing of forgotten
content, the stdout contract of the stdio MCP server, and what the CLI may say
about a running daemon.

Session fixtures here use REAL adapter identities (a UUID session file under a
project folder; a Codex rollout whose project comes from session_meta.cwd).
"""

import asyncio
import http.server
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.codex import CodexAdapter
from revien.adapters.file_watcher import FileWatcherAdapter
from revien.cli import main
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.suppression import session_identity_for
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

REPO = Path(__file__).resolve().parents[1]
UUID1 = "3f2b8c1e-5d4a-4e7b-9a10-0c6d2e8f1a42"
UUID2 = "9a7e6d5c-1b2a-4c3d-8e4f-5a6b7c8d9e0f"
SECRET = "Zanzibar quokka hunter2 bluefin"


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


def bump(path, seconds=100):
    st = path.stat()
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


def cc_session(root, project, stem, text):
    d = root / "projects" / project
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.jsonl"
    with open(p, "w") as f:
        for role, t in (("user", text), ("assistant", "Noted, " + text)):
            f.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": [{"type": "text", "text": t}]},
                "timestamp": "2026-07-01T10:00:00Z",
            }) + "\n")
    return p


def codex_rollout(root, day, stem, cwd, text):
    d = root / "2026" / "07" / day
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"timestamp": "2026-07-01T10:00:00Z",
                            "type": "session_meta",
                            "payload": {"cwd": cwd}}) + "\n")
        f.write(json.dumps({
            "timestamp": "2026-07-01T10:00:01Z", "type": "response_item",
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]},
        }) + "\n")
    return p


def hits(store, needle="Zanzibar"):
    return [n for n in store.list_nodes(limit=9999) if needle in (n.content or "")]


def forget_context(store, source_id_prefix):
    ctx = [n for n in store.list_nodes(limit=9999)
           if n.node_type == NodeType.CONTEXT
           and n.source_id.startswith(source_id_prefix)][0]
    return GraphOperations(store).forget_node(
        ctx.node_id, reason="private", construct_id="me", cascade=True)


# ---------------------------------------------------------------- identities

class TestStableIdentities:
    """Evidence per adapter of what identity survives a move or rename."""

    def test_session_identity_only_for_real_uuids(self):
        assert session_identity_for(f"claude-code:proj-a:{UUID1}") == \
            f"claude-code-session:{UUID1}"
        # same session, different project folder -> same identity
        assert session_identity_for(f"claude-code:proj-b:{UUID1}") == \
            session_identity_for(f"claude-code:proj-a:{UUID1}")
        # codex: the thread uuid is the tail of rollout-<ts>-<uuid>
        assert session_identity_for(
            f"codex:alpha:rollout-2026-07-01T10-00-00-{UUID2}") == \
            f"codex-session:{UUID2}"
        # free-form stems are not globally unique: no cross-project identity
        assert session_identity_for("claude-code:proj-a:sess-1") is None
        assert session_identity_for("codex:alpha:rollout-abc") is None
        # not a per-session channel
        for sid in ("mcp", "hermes", "file:notes.md", "api:https://x/y", ""):
            assert session_identity_for(sid) is None

    def test_claude_code_session_moved_to_another_project_stays_forgotten(
            self, store, tmp_path):
        root = tmp_path / "cc"
        p = cc_session(root, "proj-a", UUID1, f"We chose PostgreSQL. {SECRET}")
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("claude-code", ClaudeCodeAdapter(session_dir=str(root)))
        run_async(sched.sync_one("claude-code"))
        out = forget_context(store, "claude-code:")
        assert f"claude-code-session:{UUID1}" in out["suppressed_identities"]
        assert not hits(store)

        new_dir = root / "projects" / "proj-b"
        new_dir.mkdir(parents=True)
        moved = new_dir / p.name
        shutil.move(str(p), str(moved))
        bump(moved)
        store.reset_sync_cursor("claude-code")
        r = run_async(sched.sync_one("claude-code"))
        assert r["items_suppressed"] == 1
        assert not hits(store)

    def test_codex_rollout_moved_and_project_renamed_stays_forgotten(
            self, store, tmp_path):
        root = tmp_path / "codex"
        stem = f"rollout-2026-07-01T10-00-00-{UUID2}"
        p = codex_rollout(root, "01", stem, "/home/u/alpha",
                          f"We decided to use SQLite. {SECRET}")
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("codex", CodexAdapter(session_dir=str(root)))
        run_async(sched.sync_one("codex"))
        assert hits(store)
        forget_context(store, "codex:")
        assert not hits(store)

        p.unlink()
        # same runtime-assigned thread id, different dated folder, project dir
        # renamed (project comes from session_meta.cwd inside the file)
        p2 = codex_rollout(root, "02", stem, "/home/u/alpha-renamed",
                           f"We decided to use SQLite. {SECRET} more")
        bump(p2)
        store.reset_sync_cursor("codex")
        r = run_async(sched.sync_one("codex"))
        assert r["items_suppressed"] == 1
        assert not hits(store)

    def test_non_uuid_stem_is_project_scoped_and_says_so(self, store):
        pipe = IngestionPipeline(store)
        r = pipe.ingest(IngestionInput(
            source_id="claude-code:proj-a:main", content="We use Redis.",
            ingest_key="claude-code:proj-a:main"))
        out = GraphOperations(store).forget_node(r.context_node_id, reason="x")
        assert out["suppressed_identities"] == ["claude-code:proj-a:main"]
        assert any("not a UUID" in n for n in out["suppression_notice"])
        # a same-named stem in another project is a different session
        r2 = pipe.ingest(IngestionInput(
            source_id="claude-code:proj-b:main", content="We use Redis too.",
            ingest_key="claude-code:proj-b:main"))
        assert not r2.suppressed

    def test_file_watcher_rename_is_a_documented_gap(self, store, tmp_path):
        """No content-free stable identity exists for a watched file (an inode
        is reused and changes on atomic save; a content hash would not be
        content-free). The tombstone binds to the basename, and the response
        says so."""
        w = tmp_path / "watch"
        w.mkdir()
        f = w / "notes.txt"
        f.write_text(f"Meeting notes. {SECRET}\n")
        bump(f, -1000)
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("fw", FileWatcherAdapter(watch_dir=str(w)))
        run_async(sched.sync_one("fw"))
        out = forget_context(store, "file:")
        assert out["suppressed_identities"] == ["file:notes.txt"]
        assert any("renamed or moved" in n for n in out["suppression_notice"])
        # unchanged name: still suppressed
        f.write_text(f"Meeting notes. {SECRET} edited\n")
        bump(f, 5000)
        assert run_async(sched.sync_one("fw"))["items_suppressed"] == 1
        # renamed: recaptured (the gap)
        g = w / "notes-renamed.txt"
        shutil.move(str(f), str(g))
        bump(g, 9000)
        run_async(sched.sync_one("fw"))
        assert hits(store)

    def test_old_tombstone_gets_session_identity_on_open(self, db_path):
        s = GraphStore(db_path=db_path)
        s.add_ingest_suppression(f"claude-code:proj-a:{UUID1}", "source_id",
                                 "old", "me")
        s.close()
        # simulate a database written before session ids existed
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM ingest_suppressions WHERE kind = 'session_id'")
        conn.commit()
        conn.close()
        s = GraphStore(db_path=db_path)
        try:
            kinds = {r["kind"] for r in s.list_ingest_suppressions()}
            assert kinds == {"source_id", "session_id"}
            assert s.find_ingest_suppression(
                f"claude-code:other-project:{UUID1}", "") is not None
        finally:
            s.close()

    def test_restore_lifts_every_row_of_the_unit(self, store):
        pipe = IngestionPipeline(store)
        sid = f"claude-code:proj-a:{UUID1}"
        r = pipe.ingest(IngestionInput(source_id=sid, content="We use Redis.",
                                       ingest_key=sid))
        GraphOperations(store).forget_node(r.context_node_id, reason="x")
        assert {x["kind"] for x in store.list_ingest_suppressions()} == {
            "source_id", "ingest_key", "session_id"}
        assert store.lift_ingest_suppression(sid) == 3
        assert store.list_ingest_suppressions() == []
        assert not pipe.ingest(IngestionInput(
            source_id=f"claude-code:other:{UUID1}", content="We use Redis.",
            ingest_key=f"claude-code:other:{UUID1}")).suppressed


# ------------------------------------------------------------- other writers

def _openai_conv(conv_id="conv-abc", text=SECRET):
    return {"id": conv_id, "title": "Secret chat", "create_time": 1700000000,
            "update_time": 1700000100,
            "mapping": {"m1": {"id": "m1", "message": {
                "id": "m1", "author": {"role": "user"},
                "create_time": 1700000000,
                "content": {"content_type": "text", "parts": [text]}},
                "parent": None, "children": []}}}


class TestOpenAIImporterIsGated:
    def _forget_all(self, store, conv_id):
        ops = GraphOperations(store)
        for n in store.list_nodes(source_id=f"openai:conversation:{conv_id}",
                                  limit=999):
            if store.get_node(n.node_id):
                ops.forget_node(n.node_id, reason="private")

    def test_forgotten_conversation_is_not_reimported(self, db_path, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        cf = tmp_path / "conv.json"
        cf.write_text(json.dumps(_openai_conv()))
        a = OpenAIAdapter(graph_path=db_path)
        assert a.ingest_conversation(str(cf))["node_count"] > 0
        a.close()
        s = GraphStore(db_path=db_path)
        self._forget_all(s, "conv-abc")
        assert "openai:conversation:conv-abc" in [
            r["identity"] for r in s.list_ingest_suppressions()]
        s.close()
        a2 = OpenAIAdapter(graph_path=db_path)
        res = a2.ingest_conversation(str(cf))
        a2.close()
        assert res["node_count"] == 0 and res.get("suppressed") == 1
        s = GraphStore(db_path=db_path)
        try:
            assert not hits(s)
            assert s.list_ingest_suppressions()[0]["skipped_count"] == 1
        finally:
            s.close()

    def test_bulk_export_skips_and_counts_forgotten_conversations(
            self, db_path, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        bf = tmp_path / "conversations.json"
        bf.write_text(json.dumps([_openai_conv("conv-abc"),
                                  _openai_conv("conv-keep", "ordinary text")]))
        a = OpenAIAdapter(graph_path=db_path)
        a.ingest_bulk_export(str(bf))
        a.close()
        s = GraphStore(db_path=db_path)
        self._forget_all(s, "conv-abc")
        s.close()
        a2 = OpenAIAdapter(graph_path=db_path)
        res = a2.ingest_bulk_export(str(bf))
        a2.close()
        assert res["conversations_suppressed"] == 1
        s = GraphStore(db_path=db_path)
        try:
            assert not hits(s)
        finally:
            s.close()

    def test_restore_allows_reimport(self, db_path, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        cf = tmp_path / "conv.json"
        cf.write_text(json.dumps(_openai_conv()))
        a = OpenAIAdapter(graph_path=db_path)
        a.ingest_conversation(str(cf))
        a.close()
        s = GraphStore(db_path=db_path)
        self._forget_all(s, "conv-abc")
        s.lift_ingest_suppression("openai:conversation:conv-abc", actor="t")
        s.close()
        a2 = OpenAIAdapter(graph_path=db_path)
        assert a2.ingest_conversation(str(cf))["node_count"] > 0
        a2.close()


class TestSkillIngestIsGated:
    def test_forgotten_skill_is_not_reingested(self, store, tmp_path):
        from revien.skills.ingest import ingest_roots, ingest_skill_file
        sk = tmp_path / "skills" / "myskill"
        sk.mkdir(parents=True)
        md = sk / "SKILL.md"
        md.write_text(f"---\nname: myskill\ndescription: x\n---\nBody {SECRET}\n")
        node, created, _ = ingest_skill_file(store, md, "project", "p", "claude-code")
        assert created
        out = GraphOperations(store).forget_node(node.node_id, reason="private")
        assert out["suppressed_identities"], out
        node2, created2, _ = ingest_skill_file(store, md, "project", "p", "claude-code")
        assert node2 is None and not created2
        assert not store.list_nodes(node_type=NodeType.SKILL, limit=10)
        summary = ingest_roots(store, paths=[str(tmp_path / "skills")])
        assert summary["suppressed"] == 1 and summary["created"] == 0


# ---------------------------------------------------------- retention channel

class TestRetentionChannel:
    @pytest.fixture
    def client(self, db_path):
        with TestClient(create_app(db_path=db_path)) as c:
            yield c

    def test_user_reason_string_cannot_pose_as_retention(self, client, db_path):
        sid = f"claude-code:proj-a:{UUID1}"
        nid = client.post("/v1/ingest", json={
            "source_id": sid, "ingest_key": sid,
            "content": "We chose PostgreSQL."}).json()["context_node_id"]
        res = client.post(f"/v1/nodes/{nid}/forget",
                          json={"reason": "retention_expire"}).json()
        assert sid in res["suppressed_identities"]
        sups = client.get("/v1/suppressions").json()["suppressions"]
        assert sid in [s["identity"] for s in sups]
        s = GraphStore(db_path=db_path)
        try:
            assert s.count_user_forgets() == 1
        finally:
            s.close()

    def test_real_retention_still_never_suppresses_and_is_not_counted(
            self, store):
        n = Node(node_type=NodeType.FACT, label="old", content="old fact",
                 source_id=f"claude-code:p:{UUID1}",
                 last_accessed=datetime(2020, 1, 1, tzinfo=timezone.utc),
                 created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        store.add_node(n)
        GraphOperations(store).apply_retention(mode="expire", days=1)
        assert store.get_node(n.node_id) is None
        assert store.list_ingest_suppressions() == []
        assert store.count_user_forgets() == 0
        assert store.count_forget_tombstones() == 1


# --------------------------------------------------------- stdout contract

class TestNoticesGoToStderr:
    def test_pipeline_notices_leave_stdout_empty(self, store, capsys, monkeypatch):
        monkeypatch.setenv("REVIEN_INGEST_DENY", "denied-src")
        pipe = IngestionPipeline(store)
        store.add_ingest_suppression("claude-code:p:sess", "source_id", "x", "me")
        pipe.ingest(IngestionInput(source_id="claude-code:p:sess", content="hello"))
        pipe.ingest(IngestionInput(source_id="denied-src", content="hello"))
        pipe.ingest(IngestionInput(
            source_id="other",
            content="<system-reminder>recalled memory</system-reminder>"))
        cap = capsys.readouterr()
        assert cap.out == ""
        assert "ingest suppressed" in cap.err and "ingest denied" in cap.err

    def test_real_stdio_mcp_server_emits_only_json_rpc(self, db_path):
        pytest.importorskip("mcp")
        g = GraphStore(db_path=db_path)
        g.add_ingest_suppression("claude-code:p:sess", "source_id", "x", "me")
        g.close()
        env = dict(os.environ, REVIEN_SEMANTIC="0",
                   PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        p = subprocess.Popen(
            [sys.executable, "-m", "revien.cli", "mcp", "--db", db_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(REPO), env=env)

        def send(o):
            p.stdin.write(json.dumps(o) + "\n")
            p.stdin.flush()

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"}}})
            lines = [p.stdout.readline()]
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "revien_store",
                "arguments": {"content": "We chose Postgres.",
                              "source_id": "claude-code:p:sess"}}})
            lines.append(p.stdout.readline())
        finally:
            p.kill()
            err = p.stderr.read()
            p.wait()
        for line in lines:
            assert json.loads(line)["jsonrpc"] == "2.0", line
        assert json.loads(lines[1])["id"] == 2
        assert "suppressed" in err  # the notice exists, on stderr


# ------------------------------------------- forgotten content survives nowhere

def _all_rows_text(conn):
    out = {}
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        try:
            rows = conn.execute(f'SELECT * FROM "{t}"').fetchall()
        except sqlite3.Error:
            continue
        out[t] = json.dumps(rows, default=str)
    return out


NEEDLES = ["Zanzibar", "quokka", "hunter2", "bluefin"]


class TestForgottenContentSurvivesNowhere:
    def _seed(self, store):
        """A session full of the secret, plus the side tables that copy
        content: edge audit (alias), supersession candidate, usage judgment,
        embedding queue."""
        pipe = IngestionPipeline(store)
        sid = f"claude-code:proj-a:{UUID1}"
        pipe.ingest(IngestionInput(
            source_id=sid, ingest_key=sid, content_type="conversation",
            content=f"User: we chose Postgres for storage. {SECRET}. "
                    f"We decided to use Redis for cache {SECRET}."))
        # a later edit, so update-audit rows hold the secret too
        pipe.ingest(IngestionInput(
            source_id=sid, ingest_key=sid, content_type="conversation",
            content=f"User: we chose Postgres for storage. {SECRET}. "
                    f"We decided to use Redis for cache {SECRET}. Edited {SECRET}."))
        a = store.add_node(Node(node_type=NodeType.ENTITY, label="Zanzibar Holdings",
                                content="Zanzibar Holdings", source_id=sid))
        b = store.add_node(Node(node_type=NodeType.ENTITY, label="Zanzibar",
                                content="Zanzibar", source_id="other-live"))
        store.add_edge_audited(Edge(
            edge_type=EdgeType.ALIAS_OF, source_node_id=a.node_id,
            target_node_id=b.node_id,
            source_context="name_form: 'Zanzibar Holdings' / 'Zanzibar'"),
            actor="alias_pass")
        store.add_candidate(a.node_id, b.node_id, "candidate",
                            reason=f"conflict about {SECRET}",
                            trace=f"claim: {SECRET}")
        store.record_use_judgment(
            episode_key="e#1", node_id=a.node_id, used=True, reason="r",
            shared_terms=["zanzibar", "quokka"], similarity=0.9,
            query="q", mode="shadow", applied=False)
        conn = store._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS pending_embeds "
                     "(node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL)")
        conn.execute("INSERT INTO pending_embeds VALUES (?, 'now')", (a.node_id,))
        conn.commit()
        return sid

    def _forget_everything(self, store, sid):
        ops = GraphOperations(store)
        for n in store.list_nodes(limit=9999):
            if n.source_id == sid and store.get_node(n.node_id):
                ops.forget_node(n.node_id, reason="private", cascade=True)
        # the alias target belongs to another source but carries the word
        for n in store.list_nodes(limit=9999):
            if "Zanzibar" in (n.label + n.content):
                ops.forget_node(n.node_id, reason="private")

    def test_secret_is_in_no_table_file_or_export(self, store, db_path):
        sid = self._seed(store)
        assert any("Zanzibar" in v
                   for v in _all_rows_text(store._get_conn()).values())
        self._forget_everything(store, sid)

        # 1. every table, including audit before/after JSON and side tables
        for table, text in _all_rows_text(store._get_conn()).items():
            for needle in NEEDLES:
                assert needle not in text, f"{needle!r} survives in table {table}"
        # 2. the raw database bytes and the write-ahead log
        for suffix in ("", "-wal"):
            f = Path(db_path + suffix)
            if f.exists():
                blob = f.read_bytes()
                for needle in NEEDLES:
                    assert needle.encode() not in blob, \
                        f"{needle!r} survives in raw bytes of {f.name}"
        # 3. every export surface
        ops = GraphOperations(store)
        exports = {
            "export_everything": json.dumps(ops.export_everything(), default=str),
            "export_graph": store.export_graph().model_dump_json(),
        }
        store.close()
        with TestClient(create_app(db_path=db_path)) as c:
            for path in ("/v1/export", "/v1/graph"):
                exports[path] = c.get(path).text
            recall = c.post("/v1/recall", json={"query": "Postgres Redis storage"})
            exports["recall"] = recall.text
            from revien.toon import serialize_recall
            exports["toon"] = serialize_recall(recall.json())
        for name, text in exports.items():
            for needle in NEEDLES:
                assert needle not in text, f"{needle!r} survives in {name}"

    def test_cli_export_has_no_secret(self, store, db_path, tmp_path, monkeypatch):
        sid = self._seed(store)
        self._forget_everything(store, sid)
        store.close()
        out = tmp_path / "export.json"
        monkeypatch.setenv("REVIEN_HOME", str(tmp_path / "home"))
        res = CliRunner().invoke(main, ["export", "-o", str(out), "--db", db_path])
        assert res.exit_code == 0, res.output
        text = out.read_text()
        for needle in NEEDLES:
            assert needle not in text

    def test_edge_audit_rows_are_scrubbed_but_history_kept(self, store):
        a = store.add_node(Node(node_type=NodeType.ENTITY, label="Zanzibar Holdings",
                                content="Zanzibar Holdings"))
        b = store.add_node(Node(node_type=NodeType.ENTITY, label="Other",
                                content="Other"))
        e = Edge(
            edge_type=EdgeType.ALIAS_OF, source_node_id=a.node_id,
            target_node_id=b.node_id,
            source_context="name_form: 'Zanzibar Holdings' / 'Other'")
        store.add_edge_audited(e, actor="alias_pass")
        assert [r for r in store.get_all_audit() if "Zanzibar" in json.dumps(r)
                and r["node_id"] == e.edge_id], \
            "fixture must put the label into the edge audit row"
        GraphOperations(store).forget_node(a.node_id, reason="private")
        assert not [r for r in store.get_all_audit() if "Zanzibar" in json.dumps(r)]
        edge_rows = [r for r in store.get_all_audit() if r["node_id"] == e.edge_id]
        assert edge_rows and edge_rows[0]["op"]  # the fact it existed is kept

    def test_secure_delete_is_on(self, store):
        assert store._get_conn().execute("PRAGMA secure_delete").fetchone()[0] == 1

    def test_distilled_vault_note_residue_is_cleared_by_redistill(
            self, store, tmp_path):
        """DISCLOSED GAP: the forget does not touch the vault (a file outside
        the database), so the note keeps the claim until the next distill.
        The manifest hash row IS removed by the forget."""
        from revien.distill import VaultDistiller
        ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Acme",
                                  content="Acme", source_id="s"))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="k",
                                   content=f"Acme stores {SECRET} keys",
                                   source_id="s"))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                            source_node_id=ent.node_id, target_node_id=fact.node_id))
        vault = tmp_path / "vault"
        vault.mkdir()
        VaultDistiller(store, str(vault)).distill()
        GraphOperations(store).forget_node(fact.node_id, reason="private")
        manifest = store._get_conn().execute(
            "SELECT COUNT(*) FROM distill_manifest WHERE current_node_id = ?",
            (fact.node_id,)).fetchone()[0]
        assert manifest == 0
        note = vault / "Revien" / "Acme.md"
        assert "Zanzibar" in note.read_text()          # the disclosed gap
        VaultDistiller(store, str(vault)).distill()
        # the remedy: the next distill drops (or rewrites) the stale note
        assert not note.exists() or "Zanzibar" not in note.read_text()


# ------------------------------------------------------- legacy audit migration

class TestLegacyAuditScrub:
    def _old_style(self, db_path):
        """A database as an older version left it: a deleted node whose audit
        snapshots still hold its content, plus rows that must be preserved."""
        s = GraphStore(db_path=db_path)
        gone = s.add_node(Node(node_type=NodeType.FACT, label="gone",
                               content=f"deleted {SECRET}"))
        keep = s.add_node(Node(node_type=NodeType.FACT, label="keep",
                               content="a live fact"))
        other = s.add_node(Node(node_type=NodeType.ENTITY, label="Other",
                                content="x"))
        s.add_edge_audited(Edge(
            edge_type=EdgeType.ALIAS_OF, source_node_id=keep.node_id,
            target_node_id=gone.node_id,
            source_context="name_form: 'Zanzibar Co' / 'gone'"), actor="a")
        live_edge = Edge(
            edge_type=EdgeType.RELATED_TO, source_node_id=keep.node_id,
            target_node_id=other.node_id, source_context="live relation")
        s.add_edge_audited(live_edge, actor="a")
        s.add_ingest_suppression("file:x.md", "source_id", "r", "me")
        s.record_audit("tomb", "forget", actor="me",
                       after={"_tombstone": True, "original_node_id": "tomb"})
        conn = s._get_conn()
        conn.execute("DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
                     (gone.node_id, gone.node_id))      # as the old delete did
        conn.execute("DELETE FROM nodes WHERE node_id = ?", (gone.node_id,))
        conn.execute("DELETE FROM store_meta")          # not yet migrated
        conn.commit()
        s.close()
        return gone, keep, live_edge

    def test_one_time_scrub_blanks_dead_content_and_keeps_the_rest(self, db_path):
        gone, keep, live_edge = self._old_style(db_path)
        raw = sqlite3.connect(db_path)
        before = raw.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        assert "Zanzibar" in "".join(
            str(r) for r in raw.execute("SELECT * FROM audit_log"))
        raw.close()
        s = GraphStore(db_path=db_path)        # opening migrates
        try:
            audit = s.get_all_audit()
            assert len(audit) == before                      # nothing deleted
            assert "Zanzibar" not in json.dumps(audit)       # dead node + edge
            adds = {r["node_id"]: r for r in audit if r["op"] != "forget"}
            # a live node's history is untouched
            assert adds[keep.node_id]["after"]["content"] == "a live fact"
            # a live edge's history is untouched
            assert adds[live_edge.edge_id]["after"]["source_context"] == "live relation"
            # content-free rows are untouched
            assert any(r["node_id"] == "tomb" and
                       r["after"]["original_node_id"] == "tomb" for r in audit)
            assert any(r["op"] == "suppress" for r in audit)
        finally:
            s.close()
        s = GraphStore(db_path=db_path)        # second open: marker, no rescan
        try:
            assert s._get_conn().execute(
                "SELECT COUNT(*) FROM store_meta").fetchone()[0] == 1
        finally:
            s.close()


# ----------------------------------------------------------- owner decisions

class TestBlastRadiusIsNamed:
    @pytest.fixture
    def client(self, db_path):
        with TestClient(create_app(db_path=db_path)) as c:
            yield c

    def test_forget_names_session_and_the_forgotten_node_is_really_gone(
            self, client):
        sid = f"claude-code:proj-a:{UUID1}"
        client.post("/v1/ingest", json={
            "source_id": sid, "ingest_key": sid,
            "content": f"We chose PostgreSQL for storage. {SECRET}. "
                       "We decided to use Redis for caching."})
        nodes = client.get("/v1/graph").json()["nodes"]
        facts = [n for n in nodes if n["node_type"] in ("fact", "decision")
                 and "Redis" in n["content"]]
        assert facts, "extractor must produce a fact to forget"
        target = facts[0]["node_id"]
        gone_content = facts[0]["content"]
        res = client.post(f"/v1/nodes/{target}/forget", json={}).json()
        text = " ".join(res["suppression_notice"])
        assert f"claude-code-session:{UUID1}" in text
        assert "suppresses all future imports" in text
        assert "Sibling facts" in text and "remain" in text
        # The forgotten node itself is gone everywhere -- not merely "siblings
        # remain": no node with its id, no extracted node with its text, 404,
        # not in the export's nodes, not in recall, not in any audit snapshot.
        after = client.get("/v1/graph").json()["nodes"]
        assert target not in [n["node_id"] for n in after]
        assert gone_content not in [n["content"] for n in after
                                    if n["node_type"] != "context"]
        assert client.get(f"/v1/nodes/{target}").status_code == 404
        export = client.get("/v1/export").json()
        assert target not in [n["node_id"] for n in export["graph"]["nodes"]]
        for row in export["audit"]:
            if row["node_id"] == target and row["op"] != "forget":
                assert row["before"] in (None, {"_scrubbed": True})
                assert row["after"] in (None, {"_scrubbed": True})
        rec = client.post("/v1/recall", json={"query": gone_content}).json()
        assert target not in json.dumps(rec)
        # Siblings stay, and the notice is honest that the verbatim transcript
        # (context) node is one of them and still holds the original text.
        assert any(n["node_id"] != target and n["source_id"] == sid for n in after)
        assert res["context_nodes_remaining"]
        assert "verbatim context node" in text
        # forgetting that context node (cascade) removes the transcript text
        ctx = res["context_nodes_remaining"][0]
        client.post(f"/v1/nodes/{ctx}/forget", json={"cascade": True})
        assert ctx not in [n["node_id"] for n in
                           client.get("/v1/graph").json()["nodes"]]
        assert SECRET not in client.get("/v1/export").text

    def test_delete_endpoint_carries_the_notice(self, client):
        sid = f"claude-code:proj-a:{UUID1}"
        nid = client.post("/v1/ingest", json={
            "source_id": sid, "ingest_key": sid,
            "content": "We chose PostgreSQL."}).json()["context_node_id"]
        res = client.delete(f"/v1/nodes/{nid}").json()
        assert "suppresses all future imports" in " ".join(res["suppression_notice"])

    def test_shared_channel_says_deleted_but_eligible_for_recapture(self, client):
        nid = client.post("/v1/ingest", json={
            "source_id": "mcp", "content": "We chose PostgreSQL."}
        ).json()["context_node_id"]
        res = client.post(f"/v1/nodes/{nid}/forget", json={}).json()
        text = " ".join(res["suppression_notice"])
        assert "Deleted now, but eligible for recapture" in text
        assert "no stable item identity" in text
        assert res["recapture_eligible"] == [nid]
        n2 = client.post("/v1/ingest", json={
            "source_id": "mcp", "content": "We chose Redis."}
        ).json()["context_node_id"]
        res2 = client.delete(f"/v1/nodes/{n2}").json()
        assert "eligible for recapture" in " ".join(res2["suppression_notice"])

    def test_cli_suppressions_listing_names_the_blast_radius(
            self, store, db_path, monkeypatch):
        store.add_ingest_suppression("file:x.md", "source_id", "r", "me")
        store.close()
        monkeypatch.setenv("REVIEN_HOME", tempfile.mkdtemp())
        r = CliRunner().invoke(main, ["suppressions", "--db", db_path])
        assert r.exit_code == 0, r.output
        assert "WHOLE source" in r.output and "Sibling facts" in r.output

    def test_langchain_clear_documents_the_boundary(self):
        from revien.adapters.langchain_adapter import RevienMemory
        doc = RevienMemory.clear.__doc__
        assert "NO" in doc and "tombstone" in doc and "forget" in doc


class TestSuppressionsReadIsOpenLikeOtherReads:
    def test_get_is_open_but_restore_needs_the_token(self, db_path, monkeypatch):
        """Decision: reads (recall, /v1/graph, /v1/export, /v1/suppressions)
        are open; mutations need the pairing token. The test empties
        the loopback set, so every caller is treated as remote."""
        monkeypatch.setenv("REVIEN_HOME", tempfile.mkdtemp())
        s = GraphStore(db_path=db_path)
        s.add_ingest_suppression("file:x.md", "source_id", "r", "me")
        s.close()
        from revien.daemon import server as server_module
        monkeypatch.setattr(server_module, "_LOOPBACK_HOSTS", set())
        with TestClient(create_app(db_path=db_path)) as c:
            assert c.get("/v1/suppressions").status_code == 200
            # /v1/export already returns the same identities, which is why
            # gating only this route would protect nothing:
            assert "file:x.md" in c.get("/v1/export").text
            assert c.post("/v1/suppressions/restore",
                          json={"identity": "file:x.md"}).status_code in (401, 403)


# --------------------------------------------------------- daemon detection

class _Health(http.server.BaseHTTPRequestHandler):
    payload = {}

    def do_GET(self):
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _serve(payload):
    handler = type("H", (_Health,), {"payload": payload})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestResetSaysOnlyWhatItCanProve:
    OLD = {"status": "healthy", "node_count": 1, "edge_count": 0,
           "uptime_seconds": 5.0, "version": "0.3.0"}

    def _reset(self, monkeypatch, db_path, url):
        cfg = {"adapters": {"claude-code": {"type": "claude_code"}},
               "db_path": db_path}
        monkeypatch.setattr("revien.cli._load_config", lambda: cfg)
        return CliRunner().invoke(main, ["sync", "reset", "claude-code",
                                         "--db", db_path, "--daemon-url", url])

    @pytest.fixture
    def seeded(self, db_path):
        s = GraphStore(db_path=db_path)
        s.set_sync_cursor("claude-code", datetime(2026, 7, 1, tzinfo=timezone.utc))
        s.close()
        return db_path

    def test_health_reports_features_and_db_id(self, db_path):
        from revien.daemon.server import db_identity
        with TestClient(create_app(db_path=db_path)) as c:
            h = c.get("/v1/health").json()
        assert {"suppression_tombstones", "sync_reset_fence"} <= set(h["features"])
        assert h["db_id"] == db_identity(db_path)
        assert db_path not in json.dumps(h)

    def test_old_daemon_is_flagged_not_reassured(self, seeded, monkeypatch):
        srv, url = _serve(self.OLD)
        try:
            r = self._reset(monkeypatch, seeded, url)
        finally:
            srv.shutdown()
        assert r.exit_code == 0, r.output
        assert "WARNING" in r.output and "started before this version" in r.output
        assert "ignores suppressions" in r.output
        assert "cannot undo this reset" not in r.output

    def test_current_daemon_on_this_db_is_reported_fenced(self, seeded, monkeypatch):
        from revien.daemon.server import DAEMON_FEATURES, db_identity
        srv, url = _serve(dict(self.OLD, features=list(DAEMON_FEATURES),
                               db_id=db_identity(seeded)))
        try:
            r = self._reset(monkeypatch, seeded, url)
        finally:
            srv.shutdown()
        assert "is current and serves this database" in r.output
        assert "cannot undo this reset" in r.output

    def test_current_daemon_on_other_db_is_not_claimed_to_protect(
            self, seeded, monkeypatch):
        from revien.daemon.server import DAEMON_FEATURES
        srv, url = _serve(dict(self.OLD, features=list(DAEMON_FEATURES),
                               db_id="deadbeef0000"))
        try:
            r = self._reset(monkeypatch, seeded, url)
        finally:
            srv.shutdown()
        assert "different database" in r.output
        assert "cannot undo this reset" not in r.output

    def test_unreachable_daemon_says_nothing_was_checked(self, seeded, monkeypatch):
        r = self._reset(monkeypatch, seeded, "http://127.0.0.1:9")
        assert "nothing was checked" in r.output
        assert "cannot undo this reset" not in r.output
        r.output.encode("cp1252")  # stays ASCII


class TestEmbeddingIsRemoved:
    def test_forget_drops_the_vector_even_without_a_semantic_listener(
            self, db_path):
        sqlite_vec = pytest.importorskip("sqlite_vec")
        s = GraphStore(db_path=db_path)
        n = s.add_node(Node(node_type=NodeType.FACT, label="k",
                            content=f"stores {SECRET}"))
        conn = s._get_conn()
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("CREATE VIRTUAL TABLE vec_nodes USING vec0("
                     "node_id TEXT PRIMARY KEY, embedding float[4])")
        conn.execute("INSERT INTO vec_nodes(node_id, embedding) VALUES (?, ?)",
                     (n.node_id, sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4])))
        conn.commit()
        s.close()
        # a different connection with no listener and no extension loaded
        s2 = GraphStore(db_path=db_path)
        try:
            GraphOperations(s2).forget_node(n.node_id, reason="private")
            c2 = s2._get_conn()
            c2.enable_load_extension(True)
            sqlite_vec.load(c2)
            assert c2.execute("SELECT COUNT(*) FROM vec_nodes").fetchone()[0] == 0
        finally:
            s2.close()


class TestNeuralSignalLogIsPurged:
    def test_signals_for_a_forgotten_node_are_deleted(self, store, tmp_path):
        from revien.retrieval.engine import RetrievalEngine
        tdb = str(tmp_path / "training.db")
        engine = RetrievalEngine(store, training_db=tdb,
                                 model_dir=str(tmp_path / "models"))
        n = store.add_node(Node(node_type=NodeType.FACT, label="Zanzibar plan",
                                content=f"the Zanzibar plan {SECRET}"))
        engine.training_loop.log_retrieval("zanzibar", [
            {"node_id": n.node_id, "label": n.label, "node_type": "fact",
             "score": 0.9}])
        c = sqlite3.connect(tdb)
        assert c.execute("SELECT COUNT(*) FROM retrieval_signals").fetchone()[0] == 1
        GraphOperations(store).forget_node(n.node_id, reason="private")
        assert c.execute("SELECT COUNT(*) FROM retrieval_signals").fetchone()[0] == 0
        c.close()


class TestImportReplaceScrubsAudit:
    def test_replaced_graph_leaves_no_content_in_audit(self, store):
        from revien.graph.schema import Graph
        store.add_node(Node(node_type=NodeType.FACT, label="old",
                            content=f"old claim {SECRET}"))
        assert "Zanzibar" in json.dumps(store.get_all_audit())
        store.import_graph(Graph(nodes=[], edges=[]), mode="replace")
        assert store.count_nodes() == 0
        assert "Zanzibar" not in json.dumps(store.get_all_audit())


    def test_probe_never_leaves_loopback(self, seeded, monkeypatch):
        from revien.cli import _probe_daemon
        assert _probe_daemon("http://example.com:7437") is None
        assert _probe_daemon("https://127.0.0.1:7437") is None
        r = self._reset(monkeypatch, seeded, "http://192.0.2.1:7437")
        assert "nothing was checked" in r.output
