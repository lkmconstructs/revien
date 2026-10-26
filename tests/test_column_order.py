"""Node and edge rows are read by column name, not by position.

An older database can hold a column this store did not create, sitting
before columns a later migration appended. The clustering pass adds
``community_id`` to ``nodes`` with ALTER TABLE; on a database that clustered
before ``invalidated_at`` existed, the upgrade appends ``invalidated_at`` and
every later column after it. A positional reader then took the integer
community id for ``invalidated_at`` and every recall that built a node died.
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from revien.graph.clustering import CommunityDetector
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore

_NODES_BEFORE_INVALIDATION = (
    "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL, "
    "content TEXT, source_id TEXT, created_at TEXT NOT NULL, "
    "last_accessed TEXT NOT NULL, access_count INTEGER DEFAULT 0, "
    "metadata TEXT DEFAULT '{}', source_type TEXT DEFAULT 'inferred', "
    "confidence REAL DEFAULT 0.5, pinned INTEGER DEFAULT 0, "
    "confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '', "
    "source_context TEXT DEFAULT '', last_referenced TEXT"
)

_EDGES_BEFORE_INVALIDATION = (
    "edge_id TEXT PRIMARY KEY, edge_type TEXT NOT NULL, "
    "source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL, "
    "weight REAL DEFAULT 0.5, created_at TEXT NOT NULL, "
    "metadata TEXT DEFAULT '{}', confidence REAL DEFAULT 0.5, "
    "confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '', "
    "source_context TEXT DEFAULT ''"
)

_T0 = "2025-01-02T03:04:05+00:00"


@pytest.fixture
def clustered_old_db(tmp_path):
    """A database that clustered before invalidated_at existed, then upgraded."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE nodes ({_NODES_BEFORE_INVALIDATION})")
    conn.execute(f"CREATE TABLE edges ({_EDGES_BEFORE_INVALIDATION})")
    for nid in ("old-a", "old-b"):
        conn.execute(
            "INSERT INTO nodes (node_id, node_type, label, content, source_id, "
            "created_at, last_accessed, confidence_set_by) "
            "VALUES (?, 'entity', ?, ?, 'src-1', ?, ?, 'rule')",
            (nid, f"label {nid}", f"content {nid}", _T0, _T0),
        )
    conn.execute(
        "INSERT INTO edges (edge_id, edge_type, source_node_id, target_node_id, "
        "created_at, confidence_set_by) VALUES ('old-e', 'related_to', 'old-a', "
        "'old-b', ?, 'rule')", (_T0,),
    )
    CommunityDetector._ensure_column(conn)
    conn.execute("UPDATE nodes SET community_id = 7")
    # A column some other tool added to edges, before the alias leg's
    # invalidated_at existed.
    conn.execute("ALTER TABLE edges ADD COLUMN cluster_weight REAL DEFAULT 3.5")
    conn.commit()
    conn.close()

    store = GraphStore(db_path=path)
    yield store
    store.close()


def _columns(store, table):
    return [r[1] for r in store._get_conn().execute(
        f"PRAGMA table_info({table})").fetchall()]


def test_the_fixture_puts_a_foreign_column_before_later_ones(clustered_old_db):
    nodes = _columns(clustered_old_db, "nodes")
    assert nodes.index("community_id") < nodes.index("invalidated_at")
    assert nodes.index("community_id") < nodes.index("session_key")
    edges = _columns(clustered_old_db, "edges")
    assert edges.index("cluster_weight") < edges.index("invalidated_at")


def test_old_rows_read_back_on_every_node_reader(clustered_old_db):
    store = clustered_old_db
    node = store.get_node("old-a")
    assert node.invalidated_at is None
    assert node.confidence_set_by == "rule"
    assert node.session_key is None
    assert set(store.get_nodes_bulk(["old-a", "old-b"])) == {"old-a", "old-b"}
    assert {n.node_id for n in store.list_nodes()} == {"old-a", "old-b"}
    assert [n.node_id for n in store.search_nodes_keyword(["old-a"])] == ["old-a"]


def test_new_rows_keep_every_field(clustered_old_db):
    store = clustered_old_db
    when = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    store.add_node(Node(
        node_id="new-a", node_type=NodeType.ENTITY, label="new a",
        content="new a content", recorded_at=when, valid_until=when,
        invalidated_at=when, origin_runtime="cli", project_key="proj",
        session_key="sess", event_time_text="September 21",
    ))
    got = store.get_nodes_bulk(["new-a"])["new-a"]
    assert got.invalidated_at == when
    assert got.recorded_at == when
    assert got.valid_until == when
    assert got.origin_runtime == "cli"
    assert got.project_key == "proj"
    assert got.session_key == "sess"
    assert got.event_time_text == "September 21"


def test_edges_read_back_by_name(clustered_old_db):
    store = clustered_old_db
    old = store.get_edge("old-e")
    assert old.confidence_set_by == "rule"
    assert old.invalidated_at is None
    when = datetime(2026, 9, 21, tzinfo=timezone.utc)
    store.add_edge(Edge(
        edge_id="new-e", edge_type=EdgeType.RELATED_TO,
        source_node_id="old-a", target_node_id="old-b",
        source_context="ctx", invalidated_at=when,
    ))
    assert store.get_edge("new-e").invalidated_at == when
    assert {e.edge_id for e in store.get_edges_for_node("old-a")} == {"old-e", "new-e"}
    assert {e.edge_id for e in store.export_graph().edges} == {"old-e", "new-e"}


def test_the_readers_name_every_column_a_fresh_store_creates(tmp_path):
    """A column added to CREATE TABLE must also be added to the reader."""
    store = GraphStore(db_path=str(tmp_path / "fresh.db"))
    try:
        assert _columns(store, "nodes") == list(GraphStore._NODE_COLUMNS)
        assert _columns(store, "edges") == list(GraphStore._EDGE_COLUMNS)
    finally:
        store.close()
