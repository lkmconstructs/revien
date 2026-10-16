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
        assert set(out["suppressed_identities"]) == {"file:notes.txt",
                                                      os.path.realpath(f)}
        assert any("is renamed" in n for n in out["suppression_notice"])
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

    def test_file_watcher_tombstone_blocks_the_file_not_every_same_name(
            self, store, tmp_path):
        """The tombstone holds the file's real path next to its base name. A
        different file with the same name is captured while the forgotten
        path exists; the forgotten file stays blocked, and a file that took
        its place after a move is blocked too (the stated trade-off)."""
        w = tmp_path / "watch"
        for d in "abcd":
            (w / d).mkdir(parents=True)
        f = w / "a" / "notes.txt"
        f.write_text(f"Meeting notes. {SECRET}\n")
        bump(f, -1000)
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("fw", FileWatcherAdapter(watch_dir=str(w)))
        run_async(sched.sync_one("fw"))
        out = forget_context(store, "file:")
        text = " ".join(out["suppression_notice"])
        assert "blocked by its path" in text and "may be this one" in text
        assert set(out["suppressed_identities"]) == {"file:notes.txt",
                                                      os.path.realpath(f)}
        other = w / "b" / "notes.txt"            # unrelated, same name
        other.write_text("A completely different shopping list.\n")
        bump(other, 9000)
        f.write_text(f"Meeting notes. {SECRET} edited\n")
        bump(f, 9000)
        r = run_async(sched.sync_one("fw"))
        assert r["items_ingested"] == 1 and r["items_suppressed"] == 1
        assert [n for n in store.list_nodes(limit=99) if "shopping" in n.content]
        assert not hits(store)
        # the forgotten file moves: the old path is gone, so the name matches
        moved = w / "c" / "notes.txt"
        shutil.move(str(f), str(moved))
        bump(moved, 12000)
        later = w / "d" / "notes.txt"            # unrelated, appears afterwards
        later.write_text("Another unrelated list.\n")
        bump(later, 12000)
        r = run_async(sched.sync_one("fw"))
        assert r["items_suppressed"] >= 1
        assert not hits(store)

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
                "SELECT COUNT(*) FROM store_meta WHERE key = ?",
                (GraphStore._LEGACY_SCRUB_MARK,)).fetchone()[0] == 1
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
        assert "blocks EVERY file with this name" in r.output
        assert "Sibling facts" not in r.output
        assert "whole session or conversation with cascade" in r.output

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
        s.database_id()  # a daemon serving this database has created it
        s.close()
        return db_path

    def test_health_reports_features_and_db_id(self, db_path):
        with TestClient(create_app(db_path=db_path)) as c:
            h = c.get("/v1/health").json()
            h2 = c.get("/v1/health").json()
        assert {"suppression_tombstones", "sync_reset_fence"} <= set(h["features"])
        s = GraphStore(db_path=db_path)
        try:
            assert h["db_id"] == s.database_id() == h2["db_id"]
        finally:
            s.close()
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
        from revien.daemon.server import DAEMON_FEATURES
        s = GraphStore(db_path=seeded)
        this_id = s.database_id()
        s.close()
        srv, url = _serve(dict(self.OLD, features=list(DAEMON_FEATURES),
                               db_id=this_id))
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


    def test_probe_egress_is_loopback_only(self, monkeypatch):
        """Observes every connection attempt instead of trusting a refused
        result: an unreachable host also returns None, so a probe that had
        lost its loopback check would still pass a return-value test. Records
        http.client connections, DNS lookups and socket connects; none may
        name anything but a loopback address. (Fails if the guard in
        _probe_daemon is removed.)"""
        import http.client
        import socket
        from revien.cli import _probe_daemon

        seen = {"http": [], "dns": [], "connect": []}

        class _Conn:
            def __init__(self, host, port=None, **kw):
                seen["http"].append(host)

            def request(self, *a, **kw):
                raise OSError("recorded, not sent")

            def close(self):
                pass

        def _dns(host, *a, **kw):
            seen["dns"].append(host)
            raise socket.gaierror("recorded")

        def _connect(self, address):
            seen["connect"].append(address)
            raise ConnectionRefusedError("recorded")

        monkeypatch.setattr(http.client, "HTTPConnection", _Conn)
        monkeypatch.setattr(socket, "getaddrinfo", _dns)
        monkeypatch.setattr(socket.socket, "connect", _connect)

        for bad in ("http://example.com:7437", "http://192.0.2.1:7437",
                    "http://127.0.0.1.evil.example:7437",
                    "http://localhost.evil.example:7437",
                    "http://user@evil.example:7437", "http://0.0.0.0:7437",
                    "https://127.0.0.1:7437", "ftp://127.0.0.1/",
                    "//127.0.0.1:7437", "127.0.0.1:7437", ""):
            assert _probe_daemon(bad) is None
        assert seen == {"http": [], "dns": [], "connect": []}, seen

        for good in ("http://127.0.0.1:7437", "http://localhost:7437",
                     "http://[::1]:7437"):
            assert _probe_daemon(good) is None  # the recorder refuses to send
        assert seen["http"] == ["127.0.0.1", "localhost", "::1"], seen

    def test_probe_never_leaves_loopback(self, seeded, monkeypatch):
        from revien.cli import _probe_daemon
        assert _probe_daemon("http://example.com:7437") is None
        assert _probe_daemon("https://127.0.0.1:7437") is None
        r = self._reset(monkeypatch, seeded, "http://192.0.2.1:7437")
        assert "nothing was checked" in r.output


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


# ======================================================================
# Review round 4: session purge, sub-agent files, query text, proposals,
# database id, vault notice, gates
# ======================================================================

def _ingest_session(store, sid, text):
    return IngestionPipeline(store).ingest(IngestionInput(
        source_id=sid, ingest_key=sid, content_type="conversation",
        content=text))


def _session_nodes(store, sid):
    return [n for n in store.list_nodes(limit=99999) if n.source_id == sid]


class TestSessionCascade:
    """What links an extracted node to its context node, and a one-call purge
    of a whole session. The pipeline never writes DERIVED_FROM; the evidence
    is asserted here so the CHANGELOG sentence stays true."""

    TEXT = ("User: We decided to use Postgres for billing. We decided to set "
            f"the vault password to {SECRET} now. I prefer dark mode.\n"
            "Assistant: ok")

    def test_pipeline_links_by_source_id_not_derived_from(self, store):
        sid = f"claude-code:proj-a:{UUID1}"
        _ingest_session(store, sid, self.TEXT)
        nodes = store.list_nodes(limit=999)
        ctx = [n for n in nodes if n.node_type == NodeType.CONTEXT][0]
        others = [n for n in nodes if n.node_type != NodeType.CONTEXT]
        assert others and all(n.source_id == sid for n in others)
        types = set()
        for n in nodes:
            for e in store.get_edges_for_node(n.node_id):
                types.add(e.edge_type)
        assert EdgeType.DERIVED_FROM not in types
        assert {EdgeType.RELATED_TO, EdgeType.MENTIONED_BY} & types
        # edges to the context node exist for the extracted claims ...
        linked = {e.source_node_id for e in store.get_edges_for_node(ctx.node_id)
                  if e.target_node_id == ctx.node_id}
        assert linked
        # ... but they are not a safe definition of "this session's data":
        # a second session's context links to an entity the FIRST created.
        sid2 = f"claude-code:proj-a:{UUID2}"
        _ingest_session(store, sid2, "User: We decided to use Postgres for "
                                     "billing. I use Postgres daily.\n"
                                     "Assistant: ok")
        ctx2 = [n for n in store.list_nodes(node_type=NodeType.CONTEXT,
                                            source_id=sid2, limit=9)][0]
        foreign = [store.get_node(e.source_node_id if e.target_node_id == ctx2.node_id
                                  else e.target_node_id)
                   for e in store.get_edges_for_node(ctx2.node_id)]
        assert any(n.source_id == sid for n in foreign), \
            "fixture must show an edge from session 2's context into session 1"

    def test_cascade_from_context_removes_the_whole_session_only(self, store):
        sid = f"claude-code:proj-a:{UUID1}"
        sid2 = f"claude-code:proj-a:{UUID2}"
        _ingest_session(store, sid, self.TEXT)
        _ingest_session(store, sid2, "User: We decided to use Postgres for "
                                     "billing. I prefer vim.\nAssistant: ok")
        keep = {n.node_id for n in _session_nodes(store, sid2)}
        mine = {n.node_id for n in _session_nodes(store, sid)}
        assert len(mine) > 2
        ctx = [n for n in _session_nodes(store, sid)
               if n.node_type == NodeType.CONTEXT][0]
        ops = GraphOperations(store)
        prev = ops.forget_preview(ctx.node_id)
        # session 2 says "We decided to use Postgres for billing" too: the
        # dedup'd node is linked from BOTH contexts, so it is kept
        shared = set(prev["kept_shared"])
        assert shared and shared <= mine
        assert set(prev["node_ids"]) == mine - shared
        assert prev["count"] == len(mine) - len(shared)
        out = ops.forget_node(ctx.node_id, cascade=True, reason="private",
                              construct_id="me")
        assert set(out["forgotten"]) == mine - shared
        assert set(out["kept_shared"]) == shared
        assert not _session_nodes(store, sid), \
            "a kept node must stop naming the forgotten session"
        assert keep | shared <= {n.node_id for n in store.list_nodes(limit=99999)}
        assert out["suppressed_identities"] == sorted(
            [f"claude-code-session:{UUID1}", sid])
        assert out["context_nodes_remaining"] == []
        # each forgotten node has its content-free tombstone row
        forgets = [r for r in store.get_all_audit() if r["op"] == "forget"]
        assert {r["node_id"] for r in forgets} >= mine - shared
        assert SECRET not in json.dumps(store.get_all_audit())
        # and the session is not recaptured
        r = _ingest_session(store, sid, self.TEXT + "\nUser: more")
        assert r.suppressed
        assert not _session_nodes(store, sid)

    def test_cascade_from_an_extracted_node_does_not_take_the_session(self, store):
        sid = f"claude-code:proj-a:{UUID1}"
        _ingest_session(store, sid, self.TEXT)
        claim = [n for n in _session_nodes(store, sid)
                 if SECRET in n.content and n.node_type != NodeType.CONTEXT][0]
        out = GraphOperations(store).forget_node(
            claim.node_id, cascade=True, reason="private")
        assert out["forgotten"] == [claim.node_id]
        assert out["context_nodes_remaining"]
        assert any("forget the context node with cascade" in n
                   for n in out["suppression_notice"])
        assert not any("whole session" in n for n in out["suppression_notice"])
        assert any(n.node_type == NodeType.CONTEXT for n in _session_nodes(store, sid))

    def test_shared_channel_context_cascade_stays_one_node(self, store):
        """source_id 'mcp' names a whole channel, not a session: a cascade
        from one of its context nodes must never sweep the channel."""
        for i in range(3):
            _ingest_session(store, "mcp", f"User: note {i} about Redis {i}.")
        ctxs = store.list_nodes(node_type=NodeType.CONTEXT, source_id="mcp", limit=9)
        before = store.count_nodes()
        out = GraphOperations(store).forget_node(
            ctxs[0].node_id, cascade=True, reason="private")
        assert out["forgotten"] == [ctxs[0].node_id]
        assert store.count_nodes() == before - 1

    def test_rest_cascade_and_preview(self, db_path):
        sid = f"claude-code:proj-a:{UUID1}"
        s = GraphStore(db_path=db_path)
        _ingest_session(s, sid, self.TEXT)
        ctx = [n for n in _session_nodes(s, sid)
               if n.node_type == NodeType.CONTEXT][0]
        total = len(_session_nodes(s, sid))
        s.close()  # one session only: nothing is shared
        with TestClient(create_app(db_path=db_path)) as c:
            p = c.get(f"/v1/nodes/{ctx.node_id}/forget/preview").json()
            assert p["count"] == total
            r = c.post(f"/v1/nodes/{ctx.node_id}/forget",
                       json={"cascade": True, "reason": "private"}).json()
            assert r["count"] == total
            assert SECRET not in c.get("/v1/export").text
        raw = Path(db_path).read_bytes()
        assert b"Zanzibar" not in raw and b"hunter2" not in raw


def _subagent_lines(session_uuid, text, sidechain):
    out = []
    for role, t in (("user", text), ("assistant", "Done: " + text)):
        out.append(json.dumps({
            "parentUuid": None, "isSidechain": sidechain, "userType": "external",
            "cwd": "/home/me/acme", "sessionId": session_uuid,
            "agentId": "a1b2c3d4", "version": "2.0.14", "type": role,
            "message": {"role": role,
                        "content": t if role == "user"
                        else [{"type": "text", "text": t}]},
            "timestamp": "2026-07-01T10:00:00.000Z"}))
    return "\n".join(out) + "\n"


class TestSubagentTranscripts:
    def _sync(self, store, root):
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("cc", ClaudeCodeAdapter(session_dir=str(root)))
        return run_async(sched.sync_one("cc"))

    def _subagent(self, root, uuid, text, sidechain, name="agent-a1b2c3d4"):
        d = root / "projects" / "-home-me-acme" / uuid / "subagents"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.jsonl"
        p.write_text(_subagent_lines(uuid, text, sidechain))
        return p

    def test_real_subagent_files_are_sidechain_and_ingest_nothing(
            self, store, tmp_path):
        """Real Claude Code sub-agent transcripts mark every entry
        isSidechain=true, which the adapter skips. The earlier probe fixture
        left the flag out; that shape is not what Claude Code writes."""
        root = tmp_path / "cc"
        self._subagent(root, UUID1, f"Task: rotate {SECRET}", sidechain=True)
        r = self._sync(store, root)
        assert r.get("items_ingested", 0) == 0
        assert not hits(store)

    def test_tombstone_covers_subagent_files_even_without_the_flag(
            self, store, tmp_path):
        root = tmp_path / "cc"
        main = cc_session(root, "-home-me-acme", UUID1, f"We rotate {SECRET}.")
        sub = self._subagent(root, UUID1, f"Task: rotate {SECRET}", sidechain=False)
        other = self._subagent(root, UUID2, "Task: write docs for Redis",
                               sidechain=False, name="agent-ffff0000")
        bump(main, -1000); bump(sub, -1000); bump(other, -1000)
        self._sync(store, root)
        # non-vacuous: before the forget, the sub-agent file IS ingested
        srcs = {n.source_id for n in store.list_nodes(limit=9999)}
        assert "claude-code:-home-me-acme:agent-a1b2c3d4" in srcs
        # metadata carries the parent UUID and nothing else from the file
        sub_ctx = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=99)
                   if n.source_id.endswith("agent-a1b2c3d4")][0]
        assert sub_ctx.metadata["parent_session_id"] == UUID1
        out = forget_context(store, f"claude-code:-home-me-acme:{UUID1}")
        assert f"claude-code-session:{UUID1}" in out["suppressed_identities"]
        for n in store.list_nodes(limit=9999):
            if n.source_id.endswith("agent-a1b2c3d4"):
                GraphOperations(store).forget_node(n.node_id, cascade=True,
                                                   reason="private")
        assert not hits(store)
        bump(sub, 5000); bump(main, 5000)
        r = self._sync(store, root)
        assert not hits(store), "sub-agent file of a forgotten session recaptured"
        assert r.get("items_suppressed", 0) >= 1
        # an unrelated session's sub-agent file is not blocked
        assert any("Redis" in n.content for n in store.list_nodes(limit=9999))

    def test_subagent_parent_id_must_be_a_uuid(self, store, tmp_path):
        root = tmp_path / "cc"
        d = root / "projects" / "-p"
        d.mkdir(parents=True)
        (d / "agent-x.jsonl").write_text(
            _subagent_lines("not-a-uuid", "hello there friend", False))
        items = run_async(ClaudeCodeAdapter(session_dir=str(root)).fetch_new_content(
            datetime(2020, 1, 1, tzinfo=timezone.utc)))
        assert items and "parent_session_id" not in items[0]["metadata"]


class TestQueryTextIsPurged:
    QUERY = "what is the Quokkazord vault key"

    def _bytes(self, path):
        out = b""
        for suffix in ("", "-wal", "-journal"):
            f = Path(str(path) + suffix)
            if f.exists():
                out += f.read_bytes()
        return out

    def test_training_db_and_receipts_hold_no_query_after_forget(
            self, store, db_path, tmp_path):
        from revien.retrieval.engine import RetrievalEngine
        tdb = str(tmp_path / "training.db")
        engine = RetrievalEngine(store, training_db=tdb,
                                 model_dir=str(tmp_path / "models"))
        n = store.add_node(Node(node_type=NodeType.FACT, label="vault key",
                                content=f"the vault key is {SECRET}"))
        m = store.add_node(Node(node_type=NodeType.FACT, label="other",
                                content="unrelated fact about Redis"))
        # one recall surfaced both; another surfaced only the unrelated node
        engine.training_loop.log_retrieval(self.QUERY, [
            {"node_id": n.node_id, "label": n.label, "node_type": "fact", "score": .9},
            {"node_id": m.node_id, "label": m.label, "node_type": "fact", "score": .5}])
        engine.training_loop.log_retrieval("Redis cache question", [
            {"node_id": m.node_id, "label": m.label, "node_type": "fact", "score": .7}])
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        store.add_recall_receipt(recall_id="r1", runtime="x", query=self.QUERY,
                                 node_ids=[n.node_id, m.node_id],
                                 issued_at=now, expires_at=now + timedelta(hours=1))
        store.add_recall_receipt(recall_id="r2", runtime="x", query="Redis cache",
                                 node_ids=[m.node_id],
                                 issued_at=now, expires_at=now + timedelta(hours=1))
        for p in (tdb, db_path):
            assert b"Quokkazord" in self._bytes(p), "fixture must plant the query"

        GraphOperations(store).forget_node(n.node_id, reason="private")

        for p in (tdb, db_path):
            assert b"Quokkazord" not in self._bytes(p), f"query text survives in {p}"
        r1 = store.get_recall_receipt("r1")
        assert r1["query"] == "" and r1["node_ids"] == [m.node_id]
        assert store.get_recall_receipt("r2")["query"] == "Redis cache"
        c = sqlite3.connect(tdb)
        try:
            rows = c.execute("SELECT query, node_id FROM retrieval_signals").fetchall()
        finally:
            c.close()
        assert rows == [("Redis cache question", m.node_id)]

    def test_training_db_sets_secure_delete_and_cleans_a_legacy_file(self, tmp_path):
        from revien.neural.training import TrainingLoop
        tdb = tmp_path / "training.db"
        # a file as an older version left it: deleted rows still in free pages
        c = sqlite3.connect(str(tdb))
        c.execute("PRAGMA secure_delete = OFF")  # some builds default to ON
        c.execute("CREATE TABLE retrieval_signals (id INTEGER PRIMARY KEY "
                  "AUTOINCREMENT, query TEXT NOT NULL, node_id TEXT NOT NULL, "
                  "node_label TEXT, node_type TEXT, score REAL, "
                  "was_used INTEGER DEFAULT 0, timestamp TEXT NOT NULL)")
        for i in range(200):
            c.execute("INSERT INTO retrieval_signals (query, node_id, timestamp) "
                      "VALUES (?, ?, 't')", (f"leftover Quokkazord {i} " * 5, f"n{i}"))
        c.commit()
        c.execute("INSERT INTO retrieval_signals (query, node_id, timestamp) "
                  "VALUES ('keeper', 'k', 't')")
        c.execute("DELETE FROM retrieval_signals WHERE id <= 200")
        c.commit()
        c.close()
        assert b"Quokkazord" in tdb.read_bytes(), "control: legacy residue exists"
        loop = TrainingLoop(db_path=str(tdb), model_dir=str(tmp_path / "m"))
        assert b"Quokkazord" not in tdb.read_bytes()
        assert loop._connect().execute("PRAGMA secure_delete").fetchone()[0] == 1


class TestSkillProposalsRespectForget:
    def _seed(self, store):
        from datetime import timedelta
        t0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
        for si in range(3):
            sid = f"claude-code:p:3f2b8c1e-5d4a-4e7b-9a10-0c6d2e8f1a4{si}"
            for k, step in enumerate(["run migrations",
                                      "deploy to staging with key Quillfeather99",
                                      "notify ops"]):
                store.add_node(Node(
                    node_type=NodeType.ACTION, label=step, content=step,
                    source_id=sid, project_key="p", session_key=f"s{si}",
                    recorded_at=t0 + timedelta(minutes=si * 10 + k)))

    def test_forgotten_proposal_is_not_recreated_and_notice_is_accurate(self, store):
        import revien.skills.proposals as P
        self._seed(store)
        assert P.propose_skills(store)["created"] == 1
        prop = [n for n in store.list_nodes(limit=999)
                if n.node_type == NodeType.SKILL][0]
        out = GraphOperations(store).forget_node(prop.node_id, reason="private")
        assert out["suppressed_identities"] == [prop.source_id]
        assert out["recapture_eligible"] == []
        text = " ".join(out["suppression_notice"])
        assert "shared channel" not in text and "eligible for recapture" not in text
        assert "not proposed again" in text
        again = P.propose_skills(store)
        assert again["created"] == 0 and again.get("suppressed") == 1
        assert not [n for n in store.list_nodes(limit=999)
                    if n.node_type == NodeType.SKILL]
        # the tombstone is counted, and restore brings the proposal back
        assert store.list_ingest_suppressions()[0]["skipped_count"] == 1
        store.lift_ingest_suppression(prop.source_id, actor="me")
        assert P.propose_skills(store)["created"] == 1


class TestDatabaseId:
    def test_random_stored_once_and_not_derived_from_the_path(self, tmp_path):
        import hashlib
        a, b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
        s = GraphStore(db_path=a)
        first = s.database_id()
        assert s.database_id() == first
        s.close()
        s = GraphStore(db_path=a)
        assert s.database_id() == first          # persisted in store_meta
        s.close()
        t = GraphStore(db_path=b)
        other = t.database_id()
        t.close()
        assert other != first
        assert len(first) >= 16
        for path in (a, str(Path(a).resolve())):
            for n in (8, 12, 16, 32, 64):
                assert first != hashlib.sha256(path.encode()).hexdigest()[:n]
                assert first != hashlib.md5(path.encode()).hexdigest()[:n]
                assert first != hashlib.sha1(path.encode()).hexdigest()[:n]
        row = sqlite3.connect(a).execute(
            "SELECT value FROM store_meta WHERE key='database_id'").fetchone()
        assert row[0] == first

    def test_health_serves_it_and_cli_reads_it_without_writing(self, tmp_path):
        from revien.cli import _read_database_id
        db = str(tmp_path / "h.db")
        with TestClient(create_app(db_path=db)) as c:
            h = c.get("/v1/health").json()
        assert h["db_id"] == _read_database_id(db)
        assert not (tmp_path / "plain.db").exists()
        assert _read_database_id(str(tmp_path / "plain.db")) is None
        assert not (tmp_path / "plain.db").exists()


class TestForgetNamesDistilledNotes:
    def test_notice_names_vault_notes_that_hold_the_claim(self, store, tmp_path):
        from revien.distill import VaultDistiller
        ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Acme",
                                  content="Acme", source_id="s"))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="k",
                                   content=f"Acme stores {SECRET} keys",
                                   source_id="s"))
        plain = store.add_node(Node(node_type=NodeType.FACT, label="p",
                                    content="Acme likes tea", source_id="t"))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                            source_node_id=ent.node_id, target_node_id=fact.node_id))
        vault = tmp_path / "vault"
        vault.mkdir()
        VaultDistiller(store, str(vault)).distill()
        out = GraphOperations(store).forget_node(fact.node_id, reason="private")
        assert out["distilled_notes_remaining"] == ["Acme"]
        text = " ".join(out["suppression_notice"])
        assert "distilled vault note" in text
        assert "next `revien distill-vault` removes it" in text
        # a node no note was rendered from says nothing about the vault
        unrelated = store.add_node(Node(node_type=NodeType.FACT, label="z",
                                        content="never distilled", source_id="u"))
        out2 = GraphOperations(store).forget_node(unrelated.node_id, reason="x")
        assert out2["distilled_notes_remaining"] == []
        assert "vault" not in " ".join(out2["suppression_notice"])
        # and the stated remedy works
        VaultDistiller(store, str(vault)).distill()
        note = vault / "Revien" / "Acme.md"
        assert not note.exists() or "Zanzibar" not in note.read_text()


class TestSingleNodeForgetIsHonest:
    """Privacy scan for a SINGLE-node forget. Decision A is pending: the
    session's verbatim context node keeps the text, and so do near-duplicate
    sibling claims. This test pins exactly that and nothing wider: the
    forgotten node's own rows are gone everywhere, and every other place the
    text still appears is a node of the same session (named in the response)
    or that node's own audit/vector rows."""

    def test_only_the_context_node_and_siblings_keep_the_text(self, store, db_path):
        sid = f"claude-code:proj-a:{UUID1}"
        _ingest_session(store, sid, TestSessionCascade.TEXT)
        store.add_candidate(
            [n for n in _session_nodes(store, sid) if SECRET in n.content][0].node_id,
            [n for n in _session_nodes(store, sid)
             if n.node_type == NodeType.CONTEXT][0].node_id,
            "candidate", reason=f"about {SECRET}", trace=SECRET)
        claim = [n for n in _session_nodes(store, sid)
                 if SECRET in n.content and n.node_type != NodeType.CONTEXT][0]
        out = GraphOperations(store).forget_node(claim.node_id, reason="private")
        survivors = {n.node_id for n in _session_nodes(store, sid)
                     if SECRET in n.content}
        ctx_ids = {n.node_id for n in _session_nodes(store, sid)
                   if n.node_type == NodeType.CONTEXT}
        assert ctx_ids & survivors, "decision A: the context node keeps the text"
        assert set(out["context_nodes_remaining"]) == ctx_ids
        assert claim.node_id not in {n.node_id for n in store.list_nodes(limit=9999)}
        for table, text in _all_rows_text(store._get_conn()).items():
            if "Zanzibar" not in text:
                continue
            rows = json.loads(text)
            for row in rows:
                if "Zanzibar" in json.dumps(row):
                    assert claim.node_id not in json.dumps(row), \
                        f"forgotten node's row survives in {table}"
                    assert any(s in json.dumps(row) for s in survivors), \
                        f"text survives in {table} outside the session's own nodes: {row!r}"


class TestEgressGateCoversTheProbe:
    def _gate(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_egress_gate", REPO / "scripts" / "gates" / "check_egress.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @pytest.mark.parametrize("name,src,want", [
        ("E1", "import http.client\n", {"http.client"}),
        ("E2 multi-name import", "import os, socket\n", {"socket"}),
        ("E3 parent package + attribute",
         "import http\ndef f():\n    return http.client.HTTPConnection('x')\n",
         {"http.client"}),
        ("E3 aliased parent",
         "import http as h\nx = h.client.HTTPSConnection('x')\n", {"http.client"}),
        ("E5 importlib literal",
         "import importlib\nimportlib.import_module('socket')\n", {"socket"}),
        ("E5 __import__ literal", "__import__('socket')\n", {"socket"}),
        ("E5 importlib dotted literal",
         "import importlib\nimportlib.import_module('http.client')\n",
         {"http.client"}),
        ("E5 non-literal", "import importlib\nimportlib.import_module(name)\n",
         {"<dynamic-import>"}),
        ("E5 imported name",
         "from importlib import import_module\nimport_module('socket')\n",
         {"socket", "<dynamic-import>"}),
        ("from-import", "from http import client\n", {"http.client"}),
        ("from xmlrpc", "from xmlrpc import client\n", {"xmlrpc.client"}),
        ("urllib3", "    import urllib3\n".lstrip(), {"urllib3"}),
        ("smtplib", "import smtplib\n", {"smtplib"}),
        ("not network", "import json, os\nimport urllib.parse\n"
                        "import importlib.util\n", set()),
    ])
    def test_gate_self_test_network_shapes(self, name, src, want):
        g = self._gate()
        assert {m for _f, m, _s in g.network_uses(src)} == want, name

    PROBE = "revien/loopback_probe.py"

    def _probe_bytes(self):
        return (REPO / self.PROBE).read_bytes()

    def test_probe_file_matches_its_pin(self):
        g = self._gate()
        assert g.pin_problem(self.PROBE, self._probe_bytes()) is None

    @pytest.mark.parametrize("name,edit", [
        ("decorator", lambda b: b.replace(
            b"def probe_daemon(", b"@(lambda f: f)\ndef probe_daemon(")),
        ("default argument", lambda b: b.replace(
            b"probe_daemon(url)",
            b"probe_daemon(url, c=http.client.HTTPSConnection('x'))")),
        ("second definition", lambda b: b + b"\ndef probe_daemon(u):\n    return 1\n"),
        ("lambda", lambda b: b + b"\nf = lambda: http.client.HTTPSConnection('x')\n"),
        ("nested function", lambda b: b.replace(
            b"    try:\n        parts", b"    def inner():\n        pass\n    try:\n        parts")),
        ("a comment", lambda b: b + b"# harmless\n"),
        ("loopback list", lambda b: b.replace(b'"::1")', b'"::1", "x")')),
    ])
    def test_any_edit_to_the_probe_file_fails_the_pin(self, name, edit):
        g = self._gate()
        real = self._probe_bytes()
        changed = edit(real)
        assert changed != real, name
        assert g.pin_problem(self.PROBE, changed), name

    def test_a_network_use_outside_the_pinned_file_is_a_violation(self):
        g = self._gate()
        src = ("def _probe_daemon():\n    import http.client\n"
               "    return http.client.HTTPConnection('h')\n")
        assert g.violations_in("revien/cli.py", src, "")
        assert g.violations_in("revien/other.py", src, "")
        # an unchanged use already on the base is not new
        assert g.violations_in("revien/other.py", src, src) == []
        assert g.violations_in(self.PROBE, src, "") == []   # judged by its pin

    def test_cli_has_no_network_client_of_its_own(self):
        g = self._gate()
        assert g.network_uses((REPO / "revien" / "cli.py").read_text()) == []
        uses = g.network_uses(self._probe_bytes().decode())
        assert {m for _f, m, _s in uses} == {"http.client"}

    def test_the_only_pinned_file_is_the_probe_and_it_has_its_test(self):
        g = self._gate()
        assert set(g.PINNED) == {self.PROBE}
        assert g.PROBE_TEST.split("::")[-1] in (
            REPO / "tests" / "test_forget_followups.py").read_text()

    def test_gate_main_runs_its_own_controls(self):
        self._gate()._controls()

    # The shapes the verifier used against the previous gate.
    @pytest.mark.parametrize("name,src", [
        ("getattr importlib",
         "import importlib\ndef f():\n    return getattr(importlib, 'import_module')('socket')\n"),
        ("getattr builtins",
         "import builtins\ndef f():\n    return getattr(builtins, '__imp' + 'ort__')('socket')\n"),
        ("exec", "def f():\n    exec('import socket')\n"),
        ("sys.modules",
         "import sys\ndef f():\n    return sys.modules['http.client'].HTTPConnection('e')\n"),
        ("alias by assignment",
         "import http\nh = http\ndef f():\n    return h.client.HTTPConnection('e')\n"),
        ("_socket", "import _socket\n"),
        ("ssl.create_connection",
         "import ssl\ndef f():\n    return ssl.create_connection(('e', 443))\n"),
        ("asyncio.open_connection",
         "import asyncio\nasync def f():\n    return await asyncio.open_connection('e', 80)\n"),
        ("subprocess", "import subprocess\nsubprocess.run(['curl', 'e'])\n"),
        ("lambda", "import http\nf = lambda: http.client.HTTPSConnection('e')\n"),
        ("second probe def",
         "def _probe_daemon():\n    import http.client\n"
         "def _probe_daemon():\n    import http.client\n"),
        ("decorator",
         "import http\n@(lambda fn: (http.client.HTTPSConnection('e'), fn)[1])\n"
         "def _probe_daemon(url):\n    pass\n"),
        ("default argument",
         "import http\ndef _probe_daemon(url, _c=http.client.HTTPSConnection('e')):\n    pass\n"),
    ])
    def test_verifier_shapes_are_reported(self, name, src):
        g = self._gate()
        assert g.violations_in("revien/cli.py", src, ""), name

    def test_every_documented_limit_really_evades_and_nothing_else_is_claimed(self):
        g = self._gate()
        assert g.EVASIONS and g.OUT_OF_SCOPE
        for name, src in g.EVASIONS:
            assert g.network_uses(src) == [], name
        # the limits are published where the user reads them
        changelog = (REPO / "CHANGELOG.md").read_text()
        for name, _src in g.EVASIONS:
            assert name in changelog, f"limit not in CHANGELOG: {name}"
        for name in g.OUT_OF_SCOPE:
            assert name in changelog, f"limit not in CHANGELOG: {name}"
