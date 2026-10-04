"""External-review fixes (recall side): origin resolution, index churn on
open, node writes enqueue embeddings, scoped LangChain clear, REST caller
metadata. Each test follows the behaviour to the surface a caller sees."""

import asyncio
import json
import os
import sqlite3
import tempfile

import pytest

from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="revien-rv-")
    return os.path.join(d, "t.db")


# ── A: origin fields resolve independently ──────────────────


def test_explicit_origin_source_survives_without_runtime(db_path):
    s = GraphStore(db_path)
    p = IngestionPipeline(s)
    out = p.ingest(IngestionInput(
        source_id="mynote", content="I prefer tabs over spaces.",
        origin_source="vault", project_key="pk", session_key="sk"))
    n = s.get_node(out.context_node_id)
    assert (n.origin_runtime, n.origin_source, n.project_key, n.session_key) == (
        None, "vault", "pk", "sk")
    s.close()


def test_declared_fields_win_derived_fill_the_rest(db_path):
    s = GraphStore(db_path)
    p = IngestionPipeline(s)
    out = p.ingest(IngestionInput(
        source_id="claude-code:proj:sess", content="I prefer vim keybindings.",
        origin_source="api"))
    n = s.get_node(out.context_node_id)
    assert (n.origin_runtime, n.origin_source, n.project_key, n.session_key) == (
        "claude-code", "api", "proj", "sess")
    s.close()


def test_mcp_store_stamps_api_for_adapter_shaped_source_id(db_path):
    from revien.mcp_server import build_mcp_server
    server = build_mcp_server(db_path=db_path)
    res = asyncio.run(server.call_tool("revien_store", {
        "content": "I prefer dark mode.", "source_id": "claude-code:proj:s1"}))
    content = res[0] if isinstance(res, tuple) else res
    cid = json.loads(content[0].text)["context_node_id"]
    s = GraphStore(db_path)
    n = s.get_node(cid)
    assert n.origin_source == "api"
    assert n.origin_runtime == "claude-code"
    s.close()


# ── B: no index churn on open ───────────────────────────────


def test_reopen_does_not_drop_or_recreate_indexes(db_path):
    GraphStore(db_path).close()
    real = sqlite3.connect
    log = []

    def traced(*a, **k):
        c = real(*a, **k)
        c.set_trace_callback(log.append)
        return c

    sqlite3.connect = traced
    try:
        for _ in range(3):
            log.clear()
            s = GraphStore(db_path)
            s.count_nodes()
            s.close()
            drops = [q for q in log if "DROP INDEX" in q.upper()]
            assert drops == []
    finally:
        sqlite3.connect = real


# ── F: every node write enqueues embeddings ─────────────────


class _ParaphraseEmbedder:
    """espresso/latte land on the coffee dimension: a paraphrase hit needs a
    real vector, not a shared literal token."""
    VOCAB = ["coffee", "tax", "garden"]
    dim = 3
    is_cloud = False

    def embed(self, texts):
        out = []
        for t in texts:
            tl = (t or "").lower()
            v = [1.0 if w in tl else 0.0 for w in self.VOCAB]
            if "espresso" in tl or "latte" in tl:
                v[0] = 1.0
            out.append(v if any(v) else [0.0, 0.0, 0.1])
        return out


def _skill_root(tmp_path, desc):
    root = tmp_path / "skills" / "brew-it"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: brew-it\ndescription: {desc}\ntriggers: [brew]\n---\n"
        f"Grind the {desc.split()[-1]} beans.", encoding="utf-8")
    return root.parent


def _fresh_index(store):
    from tests.test_capture import _QueueTestIndex
    return _QueueTestIndex(store, embedder=_ParaphraseEmbedder())


def test_cli_path_skill_leaves_pending_work_and_is_semantic_hit(db_path, tmp_path):
    from revien.skills.ingest import ingest_roots
    # The CLI's `skills ingest`: a bare GraphStore, no SemanticIndex at all.
    bare = GraphStore(db_path)
    summary = ingest_roots(bare, paths=[str(_skill_root(tmp_path, "make coffee"))])
    skill_id = summary["skills"][0].node_id
    bare.close()

    store = GraphStore(db_path)           # the daemon / next process
    index = _fresh_index(store)
    assert index.pending_count() > 0
    hits = [nid for nid, _ in index.search("espresso", top_k=5)]  # drains first
    assert skill_id in hits
    assert index.pending_count() == 0
    store.close()


def test_skill_refresh_from_second_connection_requeues(db_path, tmp_path):
    from revien.skills.ingest import ingest_roots
    root = _skill_root(tmp_path, "make coffee")
    a = GraphStore(db_path)
    index = _fresh_index(a)
    ingest_roots(a, paths=[str(root)])
    assert index.drain_pending() >= 1 and index.pending_count() == 0
    # Edit on a different connection with no listener.
    b = GraphStore(db_path)
    (root / "brew-it" / "SKILL.md").write_text(
        "---\nname: brew-it\ndescription: tend the garden\ntriggers: [brew]\n---\nx\n",
        encoding="utf-8")
    ingest_roots(b, paths=[str(root)])
    b.close()
    assert index.pending_count() >= 1
    a.close()


def test_inline_embed_ingest_leaves_no_pending(db_path):
    s = GraphStore(db_path)
    index = _fresh_index(s)
    p = IngestionPipeline(s, semantic=index)
    p.ingest(IngestionInput(source_id="t", content="I drink coffee every morning."))
    assert index.pending_count() == 0
    assert len(index._vectors) >= 1
    s.close()


# ── G: scoped LangChain clear ───────────────────────────────


def _lc_memory(db, scope):
    import types
    from revien.adapters import langchain_adapter as LA
    from revien.retrieval.engine import RetrievalEngine
    st = GraphStore(db)
    m = types.SimpleNamespace(
        session_scope=scope, top_n=5, min_score=0.01, _store=st,
        _retrieval_engine=RetrievalEngine(st),
        _ingestion_pipeline=IngestionPipeline(st))
    for name in ("load_memory_variables", "save_context", "clear",
                 "_extract_query_from_inputs", "_format_retrieval_response",
                 "_create_edge_for_conversation"):
        setattr(m, name, types.MethodType(getattr(LA.RevienMemory, name), m))
    return m


def test_scoped_clear_leaves_other_session(db_path):
    a, b = _lc_memory(db_path, "session-A"), _lc_memory(db_path, "session-B")
    a.save_context({"input": "I prefer PostgreSQL for billing."}, {"output": "Noted."})
    b.save_context({"input": "My garden has tomatoes."}, {"output": "Nice."})
    a_ids = {n.node_id for n in a._store.list_nodes(session_key="session-A", limit=99)}
    b_ids = {n.node_id for n in b._store.list_nodes(session_key="session-B", limit=99)}
    assert a_ids and b_ids
    b.clear()
    left = {n.node_id for n in a._store.list_nodes(limit=999)}
    assert a_ids <= left
    assert not (b_ids & left)
    # Edges of the surviving session are intact.
    assert any(e for e in a._store.export_graph().edges
               if e.source_node_id in a_ids or e.target_node_id in a_ids)
    a._store.close(); b._store.close()


def test_unscoped_clear_still_wipes_everything(db_path):
    a = _lc_memory(db_path, "session-A")
    a.save_context({"input": "I prefer PostgreSQL for billing."}, {"output": "Noted."})
    c = _lc_memory(db_path, None)
    c.clear()
    assert c._store.count_nodes() == 0
    a._store.close(); c._store.close()


# ── H: REST caller metadata ─────────────────────────────────


def test_rest_ingest_keeps_caller_metadata_on_context_node_only(db_path, monkeypatch):
    from fastapi.testclient import TestClient
    from revien.daemon.server import create_app
    monkeypatch.setenv("REVIEN_SEMANTIC", "0")
    app = create_app(db_path)
    with TestClient(app) as c:
        r = c.post("/v1/ingest", json={
            "source_id": "api:test", "content": "I prefer PostgreSQL for the API.",
            "metadata": {"ticket": "ABC-1", "curated": "evil", "ingest_key": "x"}})
        assert r.status_code == 200
        cid = r.json()["context_node_id"]
        exported = c.get("/v1/export").json()
    nodes = {n["node_id"]: n for n in exported["graph"]["nodes"]}
    md = nodes[cid]["metadata"]
    assert md["caller"] == {"ticket": "ABC-1", "curated": "evil", "ingest_key": "x"}
    assert md.get("curated") is None          # reserved keys untouched
    assert md.get("ingest_key") != "x"
    for nid, n in nodes.items():
        if nid != cid:
            assert "caller" not in (n["metadata"] or {})
    # Round-trip: export -> import into a fresh db keeps it.
    from revien.graph.schema import Graph
    s2 = GraphStore(db_path + ".2")
    s2.import_graph(GraphStore(db_path).export_graph(), mode="replace")
    assert s2.get_node(cid).metadata["caller"]["ticket"] == "ABC-1"
    s2.close()
