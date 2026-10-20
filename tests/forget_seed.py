"""Seed a unique secret through every capture path, for the forget tests.

Each seeder uses the REAL path (the pipeline, the REST route, the MCP tool, the
CLI, an adapter driven by the sync scheduler, an importer, a graph import,
skills, vault reconcile). It returns a Seeded naming the database, the context
node(s) the capture produced and, for file-backed paths, the source file.
"""
import asyncio
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from click.testing import CliRunner

from revien.graph.schema import NodeType
from revien.graph.store import GraphStore


def new_secret() -> str:
    return "ZQX" + uuid.uuid4().hex[:10].upper() + "UNIQ"


def sentence(secret: str) -> str:
    """Text the rule extractor turns into several node types."""
    return (f"User: remember that the parking permit is blue.\n"
            f"Assistant: Noted. We decided to use {secret} as the project "
            f"codename.")


@dataclass
class Seeded:
    name: str
    db: str
    secret: str
    ctx_ids: List[str] = field(default_factory=list)
    source_file: Optional[Path] = None
    stable: bool = True          # has a stable unit identity
    file_backed: bool = False
    extra: Dict = field(default_factory=dict)
    # Where a non-context node carries the secret, for single-node forgets.
    node_with_secret: Optional[str] = None
    # Repeat the same capture against the same database (a sync after a
    # cursor reset, a second import of the same export, ...). None for paths
    # that cannot be repeated.
    recapture: Optional[Callable[[], None]] = None


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _age(path: Path) -> None:
    """Windows' coarse clock re-reads same-tick files; age the fixture."""
    old = time.time() - 60
    os.utime(path, (old, old))


def context_ids(db: str) -> List[str]:
    store = GraphStore(db_path=db)
    try:
        return [n.node_id for n in store.list_nodes(
            node_type=NodeType.CONTEXT, limit=1000)
            if not n.node_id.startswith("tombstone:")]
    finally:
        store.close()


def node_holding(db: str, secret: str, not_type=NodeType.CONTEXT) -> Optional[str]:
    store = GraphStore(db_path=db)
    try:
        for n in store.list_nodes(limit=1000):
            if n.node_type != not_type and secret in (n.label + n.content):
                return n.node_id
    finally:
        store.close()
    return None


def holders(db: str, secret: str) -> List[str]:
    """The nodes a user could name to forget the secret: every node that holds
    it except a conversation's or a skill's whole record."""
    store = GraphStore(db_path=db)
    try:
        return [n.node_id for n in store.list_nodes(limit=1000)
                if n.node_type not in (NodeType.CONTEXT, NodeType.SKILL)
                and secret in n.label + n.content]
    finally:
        store.close()


def decision_text(db: str, secret: str) -> str:
    """The wording of the decision the seed made of the secret."""
    store = GraphStore(db_path=db)
    try:
        return [n.content for n in store.list_nodes(limit=1000)
                if n.node_type == NodeType.DECISION and secret in n.content][0]
    finally:
        store.close()


def forget_all(db: str, secret: str, ops=None, **kwargs) -> Dict:
    """Forget, one node at a time, every node that holds the secret (what a
    user does to take a wording out of a conversation). Returns the results
    merged: counts summed, the last response's leftovers."""
    own = ops is None
    store = None
    if own:
        store = GraphStore(db_path=db)
        ops = _ops(store)
    results = []
    try:
        while True:
            ids = [n.node_id for n in ops.store.list_nodes(limit=1000)
                   if n.node_type not in (NodeType.CONTEXT, NodeType.SKILL)
                   and secret in n.label + n.content]
            if not ids:
                break
            r = ops.forget_node(ids[0], **kwargs)
            results.append(r)
            if r["status"] != "forgotten":
                break
    finally:
        if own:
            store.close()
    if not results:
        return {"status": "nothing", "forgotten": [], "count": 0}
    out = dict(results[-1])
    out["forgotten"] = [i for r in results for i in r.get("forgotten", [])]
    for key in ("count", "removed_nodes", "tombstones_created"):
        out[key] = sum(r.get(key, 0) for r in results)
    out["nodes_before"] = results[0]["nodes_before"]
    out["edges_before"] = results[0]["edges_before"]
    out["cut_back"] = [i for r in results for i in r.get("cut_back", [])]
    out["bound_to"] = [b for r in results for b in r.get("bound_to", [])]
    out["not_bound"] = [b for r in results for b in r.get("not_bound", [])]
    out["steps"] = len(results)
    return out


def _ops(store):
    from revien.graph.operations import GraphOperations
    return GraphOperations(store)


def _resync(db: str, name: str, adapter) -> None:
    """A sync that reads the whole history again (the cursor is cleared)."""
    store = GraphStore(db_path=db)
    try:
        store.delete_sync_cursor(name)
    finally:
        store.close()
    _sync(db, name, adapter)


def _sync(db: str, name: str, adapter) -> None:
    from revien.daemon.scheduler import SyncScheduler
    from revien.ingestion.pipeline import IngestionPipeline
    store = GraphStore(db_path=db)
    try:
        sched = SyncScheduler(IngestionPipeline(store))
        sched.register_adapter(name, adapter)
        run_async(sched.sync_one(name))
    finally:
        store.close()


# ── shared channels (no stable identity) ──────────────────────────────

def seed_pipeline(tmp: Path, secret: str) -> Seeded:
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    db = str(tmp / "pipeline.db")
    store = GraphStore(db_path=db)
    out = IngestionPipeline(store).ingest(IngestionInput(
        source_id="app-notes", content=sentence(secret)))
    store.close()

    def again():
        st = GraphStore(db_path=db)
        IngestionPipeline(st).ingest(IngestionInput(
            source_id="app-notes", content=sentence(secret)))
        st.close()

    return Seeded("pipeline", db, secret, [out.context_node_id], stable=False,
                  recapture=again)


def seed_rest(tmp: Path, secret: str) -> Seeded:
    from fastapi.testclient import TestClient
    from revien.daemon.server import create_app
    db = str(tmp / "rest.db")
    with TestClient(create_app(db_path=db)) as client:
        r = client.post("/v1/ingest", json={
            "source_id": "rest-feed", "content": sentence(secret)})
        assert r.status_code == 200, r.text
        ctx = r.json()["context_node_id"]
    return Seeded("rest", db, secret, [ctx], stable=False)


def seed_mcp(tmp: Path, secret: str) -> Seeded:
    from revien.mcp_server import build_mcp_server, close_mcp_server
    db = str(tmp / "mcp.db")
    server = build_mcp_server(db_path=db)
    try:
        res = asyncio.run(server.call_tool("revien_store", {
            "content": sentence(secret), "source_id": "mcp"}))
        content = res[0] if isinstance(res, tuple) else res
        ctx = json.loads(content[0].text)["context_node_id"]
    finally:
        close_mcp_server(server)
    def again():
        srv = build_mcp_server(db_path=db)
        try:
            asyncio.run(srv.call_tool("revien_store", {
                "content": sentence(secret), "source_id": "mcp"}))
        finally:
            close_mcp_server(srv)

    return Seeded("mcp", db, secret, [ctx], stable=False, recapture=again)


def seed_cli(tmp: Path, secret: str) -> Seeded:
    from revien.cli import main
    db = str(tmp / "cli.db")
    res = CliRunner().invoke(main, ["ingest", sentence(secret), "--db", db])
    assert res.exit_code == 0, res.output
    return Seeded("cli", db, secret, context_ids(db), stable=False)


def seed_ollama(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.ollama_adapter import OllamaAdapter
    db = str(tmp / "ollama.db")
    a = OllamaAdapter(graph_path=db)
    try:
        a.ingest_ollama_history([
            {"role": "user", "content": f"remember that the parking permit is blue."},
            {"role": "assistant", "content": f"Noted. We decided to use {secret} as the codename."}])
    finally:
        a.close()
    return Seeded("ollama", db, secret, context_ids(db), stable=False)


def seed_ollama_exchange(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.ollama_adapter import OllamaAdapter
    db = str(tmp / "ollama2.db")
    a = OllamaAdapter(graph_path=db)
    try:
        a._ingest_exchange(f"remember that the parking permit is blue.",
                           f"Noted. We decided to use {secret} as the codename.")
    finally:
        a.close()
    return Seeded("ollama_chat", db, secret, context_ids(db), stable=False)


def seed_langchain(tmp: Path, secret: str) -> Seeded:
    from revien.adapters import RevienMemory
    db = str(tmp / "langchain.db")
    with RevienMemory(graph_path=db, session_scope="sess-lc") as mem:
        mem.save_context({"input": f"remember that the parking permit is blue."},
                         {"output": f"Noted. We decided to use {secret} as the codename."})
    return Seeded("langchain", db, secret, context_ids(db), stable=False)


def seed_hermes(tmp: Path, secret: str) -> Seeded:
    import queue
    import threading
    from revien.hermes_provider import RevienMemoryProvider
    from revien.ingestion.pipeline import IngestionPipeline
    from revien.retrieval.engine import RetrievalEngine
    from revien.semantic.index import SemanticIndex
    db = str(tmp / "hermes.db")
    store = GraphStore(db_path=db)
    prov = object.__new__(RevienMemoryProvider)
    prov._db_path = db
    prov._session_id = "hermes-sess"
    prov._store = store
    sem = SemanticIndex(store)
    prov._pipeline = IngestionPipeline(store, semantic=sem)
    prov._engine = RetrievalEngine(store, semantic=sem)
    prov._sync_queue = queue.Queue()
    prov._sync_worker = None
    prov._worker_lock = threading.Lock()
    try:
        prov._tool_store(sentence(secret))
    finally:
        store.close()
    return Seeded("hermes", db, secret, context_ids(db), stable=False)


def seed_generic_api(tmp: Path, secret: str, monkeypatch) -> Seeded:
    import httpx
    from revien.adapters.generic_api import GenericAPIAdapter
    db = str(tmp / "generic.db")

    def handler(request):
        return httpx.Response(200, json={"conversations": [
            {"content": sentence(secret),
             "timestamp": "2026-07-01T10:00:00+00:00"}]})

    real = httpx.AsyncClient

    class _Client(real):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _sync(db, "feed", GenericAPIAdapter(url="http://feed.invalid/api"))
    return Seeded("generic_api", db, secret, context_ids(db), stable=False)


# ── file-backed units ─────────────────────────────────────────────────

def _claude_lines(secret: str, session_id: Optional[str] = None):
    turns = [("user", f"remember that the parking permit is blue."),
             ("assistant", f"Noted. We decided to use {secret} as the codename.")]
    out = []
    for role, text in turns:
        entry = {"type": role,
                 "message": {"role": role,
                             "content": [{"type": "text", "text": text}]},
                 "timestamp": "2026-07-01T10:00:00Z"}
        if session_id:
            entry["sessionId"] = session_id
        out.append(json.dumps(entry))
    return out


def write_claude_session(root: Path, secret: str, project="proj-x",
                         stem="sess-1", session_id=None) -> Path:
    d = root / "projects" / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.jsonl"
    path.write_text("\n".join(_claude_lines(secret, session_id)) + "\n",
                    encoding="utf-8")
    _age(path)
    return path


def seed_claude_code(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.claude_code import ClaudeCodeAdapter
    db = str(tmp / "claude.db")
    root = tmp / "claude"
    path = write_claude_session(root, secret, session_id="uuid-claude-1")
    adapter = ClaudeCodeAdapter(session_dir=str(root / "projects"))
    _sync(db, "claude-code", adapter)
    return Seeded("claude_code", db, secret, context_ids(db), source_file=path,
                  file_backed=True, extra={"root": root},
                  recapture=lambda: _resync(db, "claude-code", adapter))


def seed_codex(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.codex import CodexAdapter
    db = str(tmp / "codex.db")
    d = tmp / "codex" / "sessions" / "2026" / "07" / "01"
    d.mkdir(parents=True)
    path = d / "rollout-2026-07-01T10-00-00-abc.jsonl"
    lines = [json.dumps({"timestamp": "2026-07-01T10:00:00Z", "type": "session_meta",
                         "payload": {"id": "0197-codex-forgotten", "cwd": "/work/proj-c"}})]
    for role, kind, text in (
            ("user", "input_text", f"remember that the parking permit is blue."),
            ("assistant", "output_text",
             f"Noted. We decided to use {secret} as the codename.")):
        lines.append(json.dumps({
            "timestamp": "2026-07-01T10:00:01Z", "type": "response_item",
            "payload": {"type": "message", "role": role,
                        "content": [{"type": kind, "text": text}]}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _age(path)
    adapter = CodexAdapter(session_dir=str(tmp / "codex" / "sessions"))
    _sync(db, "codex", adapter)
    return Seeded("codex", db, secret, context_ids(db), source_file=path,
                  file_backed=True,
                  recapture=lambda: _resync(db, "codex", adapter))


def seed_file_watcher(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.file_watcher import FileWatcherAdapter
    db = str(tmp / "fw.db")
    d = tmp / "watched"
    d.mkdir()
    path = d / "notes.md"
    path.write_text(f"Meeting notes. {sentence(secret)}\n", encoding="utf-8")
    _age(path)
    adapter = FileWatcherAdapter(watch_dir=str(d))
    _sync(db, "files", adapter)
    return Seeded("file_watcher", db, secret, context_ids(db), source_file=path,
                  file_backed=True,
                  recapture=lambda: _resync(db, "files", adapter))


def seed_obsidian(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.obsidian import ObsidianVaultAdapter
    db = str(tmp / "vault.db")
    vault = tmp / "vault"
    vault.mkdir()
    path = vault / "Atlas.md"
    path.write_text(f"# Atlas\nThe launch date is tomorrow. "
                    f"We decided to use {secret} for the demo.\n",
                    encoding="utf-8")
    _age(path)
    adapter = ObsidianVaultAdapter(vault_dir=str(vault))
    _sync(db, "vault", adapter)
    return Seeded("obsidian", db, secret, context_ids(db), source_file=path,
                  file_backed=True,
                  recapture=lambda: _resync(db, "vault", adapter))


# ── permanent units (no file) ─────────────────────────────────────────

def _chatgpt_export(path: Path, secret: str) -> None:
    conv = [{
        "id": "conv-1", "title": "Plans", "create_time": 1_750_000_000,
        "current_node": "m2",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["m1"]},
            "m1": {"id": "m1", "parent": "root", "children": ["m2"], "message": {
                "id": "m1", "author": {"role": "user"}, "create_time": 1_750_000_001,
                "content": {"content_type": "text", "parts": [
                    f"remember that the parking permit is blue."]}}},
            "m2": {"id": "m2", "parent": "m1", "children": [], "message": {
                "id": "m2", "author": {"role": "assistant"}, "create_time": 1_750_000_002,
                "content": {"content_type": "text", "parts": [
                    f"Noted. We decided to use {secret} as the codename."]}}},
        }}]
    path.write_text(json.dumps(conv), encoding="utf-8")


def seed_chatgpt_import(tmp: Path, secret: str) -> Seeded:
    from revien.importers import chatgpt
    from revien.importers.base import run_import
    from revien.ingestion.pipeline import IngestionPipeline
    db = str(tmp / "chatgpt.db")
    export = tmp / "conversations.json"
    _chatgpt_export(export, secret)
    def again():
        st = GraphStore(db_path=db)
        run_import(chatgpt.iter_units(str(export)), IngestionPipeline(st))
        st.close()

    again()
    return Seeded("chatgpt", db, secret, context_ids(db), file_backed=False,
                  extra={"export": export}, recapture=again)


def seed_claude_import(tmp: Path, secret: str) -> Seeded:
    from revien.importers import claude
    from revien.importers.base import run_import
    from revien.ingestion.pipeline import IngestionPipeline
    db = str(tmp / "claudeai.db")
    export = tmp / "claude_conversations.json"
    export.write_text(json.dumps([{
        "uuid": "cc-1", "name": "Plans", "created_at": "2026-07-01T10:00:00Z",
        "chat_messages": [
            {"uuid": "a", "sender": "human", "created_at": "2026-07-01T10:00:01Z",
             "text": f"remember that the parking permit is blue.",
             "content": [{"type": "text", "text": f"remember that the parking permit is blue."}]},
            {"uuid": "b", "sender": "assistant", "created_at": "2026-07-01T10:00:02Z",
             "text": f"Noted. We decided to use {secret} as the codename.",
             "content": [{"type": "text", "text": f"Noted. We decided to use {secret} as the codename."}]}]}]),
        encoding="utf-8")
    def again():
        st = GraphStore(db_path=db)
        run_import(claude.iter_units(str(export)), IngestionPipeline(st))
        st.close()

    again()
    return Seeded("claude_import", db, secret, context_ids(db),
                  extra={"export": export}, recapture=again)


def seed_readwise_import(tmp: Path, secret: str) -> Seeded:
    from revien.importers import readwise
    from revien.importers.base import run_import
    from revien.ingestion.pipeline import IngestionPipeline
    db = str(tmp / "readwise.db")
    export = tmp / "readwise.csv"
    export.write_text(
        "Highlight,Book Title,Book Author,Amazon Book ID,Note,Color,Tags,Location Type,Location,Highlighted at,Document tags\n"
        f"\"The parking permit is blue. We decided to use {secret}.\",Safe Book,Some Author,,,,,page,12,2026-07-01 10:00:00,\n",
        encoding="utf-8")
    def again():
        st = GraphStore(db_path=db)
        run_import(readwise.iter_units(str(export)), IngestionPipeline(st))
        st.close()

    again()
    return Seeded("readwise", db, secret, context_ids(db),
                  extra={"export": export}, recapture=again)


def seed_openai_adapter(tmp: Path, secret: str) -> Seeded:
    from revien.adapters.openai_adapter import OpenAIAdapter
    db = str(tmp / "openai.db")
    export = tmp / "oa.json"
    _chatgpt_export(export, secret)
    conv = json.loads(export.read_text())[0]
    export.write_text(json.dumps(conv), encoding="utf-8")
    def again():
        with OpenAIAdapter(graph_path=db) as a:
            a.ingest_conversation(str(export))

    again()
    return Seeded("openai_adapter", db, secret, context_ids(db),
                  extra={"export": export}, recapture=again)


def seed_import_graph(tmp: Path, secret: str) -> Seeded:
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    src = str(tmp / "graph_src.db")
    s1 = GraphStore(db_path=src)
    IngestionPipeline(s1).ingest(IngestionInput(
        source_id="elsewhere", content=sentence(secret)))
    graph = s1.export_graph()
    s1.close()
    db = str(tmp / "graph_dst.db")
    s2 = GraphStore(db_path=db)
    s2.import_graph(graph, mode="merge")
    s2.close()
    return Seeded("import_graph", db, secret, context_ids(db), stable=False)


def seed_skill(tmp: Path, secret: str) -> Seeded:
    from revien.skills.ingest import ingest_roots
    db = str(tmp / "skills.db")
    root = tmp / "skills"
    sk = root / "deploy"
    sk.mkdir(parents=True)
    path = sk / "SKILL.md"
    path.write_text(
        "---\nname: deploy\ndescription: Deploy the thing\ntriggers: [deploy]\n---\n"
        f"Use the code {secret} to unlock the release.\n", encoding="utf-8")
    def again():
        st = GraphStore(db_path=db)
        ingest_roots(st, paths=[str(root)])
        st.close()

    again()
    store = GraphStore(db_path=db)
    ids = [n.node_id for n in store.list_nodes(node_type=NodeType.SKILL, limit=10)]
    store.close()
    return Seeded("skill", db, secret, ids, source_file=path, file_backed=True,
                  recapture=again)


def seed_vault_reconcile(tmp: Path, secret: str) -> Seeded:
    from revien.distill import VaultDistiller, VaultReconciler
    from revien.graph.schema import Edge, EdgeType, Node, SourceType
    from revien.semantic.index import SemanticIndex
    db = str(tmp / "reconcile.db")
    vault = tmp / "rvault"
    vault.mkdir()
    store = GraphStore(db_path=db)
    ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Atlas Server",
                              content="Atlas Server", source_type=SourceType.EXTRACTED,
                              confidence=1.0))
    fact = store.add_node(Node(node_type=NodeType.FACT, label="ram",
                               content="Has 64GB of RAM.", source_type=SourceType.EXTRACTED,
                               confidence=1.0))
    store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=ent.node_id,
                        target_node_id=fact.node_id, weight=0.8))
    VaultDistiller(store, str(vault)).distill()
    note = vault / "Revien" / "Atlas Server.md"
    text = note.read_text(encoding="utf-8")
    marker = "Has 64GB of RAM."
    assert marker in text
    line_end = text.index("\n", text.index(marker))
    text = text[:line_end] + f"\n- The door code is {secret}." + text[line_end:]
    note.write_text(text, encoding="utf-8")
    VaultReconciler(store, str(vault), semantic=SemanticIndex(store, enabled=False)).reconcile()
    holder = None
    for n in store.list_nodes(limit=100):
        if secret in n.content:
            holder = n.node_id
    store.close()
    return Seeded("vault_reconcile", db, secret, [], stable=False,
                  node_with_secret=holder, extra={"note": note})


SEEDERS: Dict[str, Callable] = {
    "pipeline": seed_pipeline, "rest": seed_rest, "mcp": seed_mcp,
    "cli": seed_cli, "ollama": seed_ollama, "ollama_chat": seed_ollama_exchange,
    "langchain": seed_langchain, "hermes": seed_hermes,
    "claude_code": seed_claude_code, "codex": seed_codex,
    "file_watcher": seed_file_watcher, "obsidian": seed_obsidian,
    "chatgpt": seed_chatgpt_import, "claude_import": seed_claude_import,
    "readwise": seed_readwise_import, "openai_adapter": seed_openai_adapter,
    "import_graph": seed_import_graph, "skill": seed_skill,
    "vault_reconcile": seed_vault_reconcile,
}
NEEDS_MONKEYPATCH = {"generic_api": seed_generic_api}
ALL_PATHS = sorted(list(SEEDERS) + list(NEEDS_MONKEYPATCH))


def seed(name: str, tmp: Path, monkeypatch=None) -> Seeded:
    secret = new_secret()
    if name in NEEDS_MONKEYPATCH:
        s = NEEDS_MONKEYPATCH[name](tmp, secret, monkeypatch)
    else:
        s = SEEDERS[name](tmp, secret)
    if not s.ctx_ids and name != "vault_reconcile":
        s.ctx_ids = context_ids(s.db)
    s.node_with_secret = s.node_with_secret or node_holding(s.db, secret)
    return s
