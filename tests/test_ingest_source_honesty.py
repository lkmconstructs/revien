"""Fix leg: every ingest path carries an honest timestamp_source; distill gates
dates; a partial reindex never records a recipe; the successor lookup only runs
under prev and range-scans idx_nodes_session. OFFLINE (stub embedders)."""

import hashlib
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.semantic import index as sem_index
from revien.semantic.index import SEMANTIC_AVAILABLE, SemanticIndex

T0 = datetime(2023, 5, 7, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_semantic(monkeypatch):
    monkeypatch.setenv("REVIEN_SEMANTIC", "0")
    monkeypatch.delenv("REVIEN_EMBED_CONTEXT", raising=False)


def _source_by_label(db):
    conn = sqlite3.connect(db)
    try:
        return {
            r[0]: (r[1], r[2]) for r in conn.execute(
                "select label, recorded_at, "
                "json_extract(metadata,'$.recorded_at_source') "
                "from nodes where node_type='context'")
        }
    finally:
        conn.close()


# -- F1 ---------------------------------------------------------------------

def test_sync_vault_carries_adapter_timestamp_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("REVIEN_HOME", str(tmp_path / ".revien"))
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "plain.md").write_text(
        "# Plain\nThe Zanzibar kayak trip planning notes.\n", encoding="utf-8")
    (vault / "dated.md").write_text(
        "---\ndate: 2021-03-04\n---\n# Dated\nThe Yokohama ferry notes.\n",
        encoding="utf-8")
    db = str(tmp_path / "r.db")
    runner = CliRunner()
    res = runner.invoke(main, ["sync-vault", "--vault", str(vault), "--db", db, "--full"])
    assert res.exit_code == 0, res.output
    rows = _source_by_label(db)
    plain = [v for k, v in rows.items() if "Plain" in k][0]
    dated = [v for k, v in rows.items() if "Dated" in k][0]
    assert plain[1] == "mtime"
    assert dated == ("2021-03-04T00:00:00+00:00", "content")
    out = runner.invoke(main, ["recall", "Zanzibar kayak", "--db", db]).output
    assert "Date: -" in out
    assert "Date: 20" not in out


def test_cli_ingest_is_capture(tmp_path):
    db = str(tmp_path / "r.db")
    res = CliRunner().invoke(main, ["ingest", "Mara chose Aurora for billing.",
                                    "--source", "cli:test", "--db", db])
    assert res.exit_code == 0, res.output
    conn = sqlite3.connect(db)
    try:
        got = {r[0] for r in conn.execute(
            "select json_extract(metadata,'$.recorded_at_source') from nodes "
            "where node_type='context'")}
    finally:
        conn.close()
    # Manual CLI capture stamps now() as the capture time.
    assert got == {"capture"}


def test_daemon_ingest_source_field(tmp_path):
    from fastapi.testclient import TestClient
    from revien.daemon.server import create_app
    db = str(tmp_path / "d.db")
    with TestClient(create_app(db_path=db)) as client:
        a = client.post("/v1/ingest", json={
            "source_id": "t:a", "content": "Mara chose Aurora alpha.",
            "timestamp": "2024-01-02T03:04:05+00:00"})
        b = client.post("/v1/ingest", json={
            "source_id": "t:b", "content": "Mara chose Aurora beta.",
            "timestamp": "2024-01-02T03:04:05+00:00", "timestamp_source": "mtime"})
        c = client.post("/v1/ingest", json={
            "source_id": "t:c", "content": "Mara chose Aurora gamma."})
        assert a.status_code == b.status_code == c.status_code == 200
    conn = sqlite3.connect(db)
    try:
        got = dict(conn.execute(
            "select source_id, json_extract(metadata,'$.recorded_at_source') "
            "from nodes where node_type='context'"))
    finally:
        conn.close()
    assert got["t:a"] == "content"
    assert got["t:b"] == "mtime"
    # No timestamp supplied: nothing is stamped, so nothing is mislabeled.
    assert got["t:c"] is None


# -- F2 ---------------------------------------------------------------------

def _distill_text(tmp_path, source):
    from revien.distill import VaultDistiller
    store = GraphStore(db_path=str(tmp_path / "r.db"))
    pipe = IngestionPipeline(store)
    for _ in range(3):
        pipe.ingest(IngestionInput(
            source_id="claude-code:proj:s9",
            content="User: Mara decided to migrate Postgres billing to Aurora. Mara prefers Aurora.",
            timestamp=datetime(2026, 9, 30, 12, tzinfo=timezone.utc),
            timestamp_source=source, metadata={"adapter": "claude_code"}))
    vault = tmp_path / "vault"
    vault.mkdir()
    VaultDistiller(store, str(vault), min_claims=1).distill()
    store.close()
    return "\n".join(f.read_text(encoding="utf-8") for f in vault.rglob("*.md"))


def test_distill_omits_mtime_date(tmp_path):
    assert "2026-09-30" not in _distill_text(tmp_path, "mtime")


def test_distill_keeps_content_date(tmp_path):
    assert "2026-09-30" in _distill_text(tmp_path, "content")


# -- S3 ---------------------------------------------------------------------

class _BoW:
    is_cloud = False

    def __init__(self, dim, name, fail_after=None):
        self.dim, self.model_name = dim, name
        self.fail_after, self.calls = fail_after, 0

    def embed(self, texts):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("boom mid-reindex")
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


def _turn(store, text, seq):
    return store.add_node(Node(
        node_type=NodeType.CONTEXT, label=text, content=text,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        created_at=T0 + timedelta(seconds=seq), last_accessed=T0,
        recorded_at=T0 + timedelta(seconds=seq), session_key="c1:s1"))


def _meta(db):
    conn = sqlite3.connect(db)
    try:
        return dict(conn.execute("select key,value from semantic_meta"))
    finally:
        conn.close()


@pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="sqlite-vec absent")
@pytest.mark.parametrize("new_dim", [64, 32])  # same-dim swap, dim change
def test_partial_reindex_never_records_recipe(tmp_path, monkeypatch, new_dim):
    monkeypatch.setenv("REVIEN_SEMANTIC", "1")
    db = str(tmp_path / "s.db")
    store = GraphStore(db_path=db)
    idx = SemanticIndex(store, embedder=_BoW(64, "m-old"), enabled=True)
    nodes = [_turn(store, f"Speaker: fact number {k} about kiwis", k) for k in range(5)]
    idx.index_nodes([(n.node_id, n.label, n.content) for n in nodes])
    store.close()
    assert _meta(db)["embed_model"] == "m-old"

    store = GraphStore(db_path=db)
    idx = SemanticIndex(store, embedder=_BoW(new_dim, "m-new", fail_after=1), enabled=True)
    result = idx.reindex_all(batch_size=2)
    assert result["status"] == "partial"
    meta = _meta(db)
    assert meta["embed_state"] == "partial"
    assert meta["embed_model"] != "m-new"  # recipe NOT recorded
    assert any("did not finish" in w for w in idx.status()["warnings"])
    store.close()

    # Reopen: the warning persists until a full reindex succeeds.
    store = GraphStore(db_path=db)
    idx = SemanticIndex(store, embedder=_BoW(new_dim, "m-new"), enabled=True)
    assert any("did not finish" in w for w in idx.status()["warnings"])
    assert idx.warnings_note()
    final = idx.reindex_all(batch_size=2)
    assert final["status"] == "ok"
    meta = _meta(db)
    assert meta["embed_state"] == "ok"
    assert meta["embed_model"] == "m-new"
    assert meta["embed_dim"] == str(new_dim)
    assert not any("did not finish" in w for w in idx.status()["warnings"])
    store.close()


@pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="sqlite-vec absent")
def test_cli_reindex_exits_nonzero_on_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEN_SEMANTIC", "1")
    db = str(tmp_path / "s.db")
    store = GraphStore(db_path=db)
    for k in range(3):
        _turn(store, f"Speaker: fact {k}", k)
    store.close()
    emb = _BoW(8, "m-x", fail_after=0)
    monkeypatch.setattr(sem_index, "build_embedder", lambda provider=None: emb)
    res = CliRunner().invoke(main, ["reindex", "--db", db])
    assert res.exit_code != 0, res.output


# -- R4 ---------------------------------------------------------------------

def _store_with_session(tmp_path, n=6):
    store = GraphStore(db_path=str(tmp_path / "g.db"))
    nodes = [_turn(store, f"turn {k}", k) for k in range(n)]
    return store, nodes


def test_no_successor_lookup_without_listener(tmp_path):
    store, nodes = _store_with_session(tmp_path)
    seen = []
    store._get_conn().set_trace_callback(seen.append)
    store.update_node(nodes[1].node_id, content="edited")
    store.delete_node(nodes[2].node_id)
    assert not [s for s in seen if "WHERE session_key" in s], seen
    store.close()


def test_successor_lookup_runs_with_listener(tmp_path):
    store, nodes = _store_with_session(tmp_path)
    fired = []
    store.register_content_listener(
        "t", on_successor_change=lambda nid, lbl, ct: fired.append(nid))
    store.update_node(nodes[1].node_id, content="edited")
    assert fired == [nodes[2].node_id]
    store.close()


def test_index_registers_successor_listener_only_under_prev(tmp_path, monkeypatch):
    if not SEMANTIC_AVAILABLE:
        pytest.skip("sqlite-vec absent")
    monkeypatch.setenv("REVIEN_SEMANTIC", "1")
    store = GraphStore(db_path=str(tmp_path / "l.db"))
    SemanticIndex(store, embedder=_BoW(8, "m"), enabled=True)
    assert not store._has_successor_listener()
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    SemanticIndex(store, embedder=_BoW(8, "m"), enabled=True)
    assert store._has_successor_listener()
    store.close()


def test_neighbour_order_null_and_ties(tmp_path):
    store = GraphStore(db_path=str(tmp_path / "n.db"))

    def mk(label, seq, rec):
        return store.add_node(Node(
            node_type=NodeType.CONTEXT, label=label, content=label,
            source_type=SourceType.EXTRACTED, confidence=1.0,
            created_at=T0 + timedelta(seconds=seq), last_accessed=T0,
            recorded_at=rec, session_key="s"))
    order = [mk("a", 1, None), mk("b", 2, None), mk("c", 3, T0),
             mk("d", 4, T0), mk("e", 5, T0 + timedelta(days=1))]
    for i, node in enumerate(order):
        nxt = store.next_context_in_session(node)
        prv = store.previous_context_in_session(node)
        want_next = order[i + 1].node_id if i + 1 < len(order) else None
        want_prev = order[i - 1].node_id if i else None
        assert (nxt.node_id if nxt else None) == want_next
        assert (prv.node_id if prv else None) == want_prev
    store.close()


@pytest.mark.parametrize("direction", [">", "<"])
def test_neighbour_plan_uses_session_index(tmp_path, direction):
    store, _ = _store_with_session(tmp_path)
    order = "ASC" if direction == ">" else "DESC"
    plan = store._get_conn().execute(
        "EXPLAIN QUERY PLAN SELECT * FROM nodes WHERE session_key = ? AND "
        "node_type = ? AND node_id != ? AND "
        f"(recorded_at, created_at, rowid) {direction} (?, ?, ?) "
        f"ORDER BY recorded_at {order}, created_at {order}, rowid {order} LIMIT 1",
        ("s", "context", "x", "a", "b", 1)).fetchall()
    text = " ".join(str(r[-1]) for r in plan)
    assert "USING INDEX idx_nodes_session" in text
    assert "TEMP B-TREE" not in text
    store.close()


def test_old_narrow_session_index_is_upgraded(tmp_path):
    db = str(tmp_path / "o.db")
    GraphStore(db_path=db).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP INDEX idx_nodes_session")
    conn.execute("CREATE INDEX idx_nodes_session ON nodes(session_key, recorded_at)")
    conn.commit()
    conn.close()
    store = GraphStore(db_path=db)
    sql = store._get_conn().execute(
        "select sql from sqlite_master where name='idx_nodes_session'").fetchone()[0]
    assert "created_at" in sql
    store.close()
    GraphStore(db_path=db).close()  # idempotent reopen


def test_daemon_ingest_rejects_unknown_source_with_400(tmp_path):
    """An unknown timestamp_source must be refused at the API boundary (400),
    not surface as a 500 from the dataclass check."""
    from fastapi.testclient import TestClient
    from revien.daemon.server import create_app
    app = create_app(db_path=str(tmp_path / "d.db"))
    with TestClient(app) as client:
        res = client.post("/v1/ingest", json={
            "source_id": "api:test", "content": "Theo picked Fernweh-Core.",
            "timestamp_source": "bogus"})
    assert res.status_code == 400, res.text
    assert "timestamp_source" in res.text
