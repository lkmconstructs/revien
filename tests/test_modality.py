"""Claim Sovereignty Layer — Leg 1 (modality awareness) storage tests.

Isolated: exercises ONLY the schema + store round-trip, mutability, and the
in-place migration of a pre-Leg-1 database. No ingestion/retrieval involved.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from revien.graph.schema import Modality, Node, NodeType
from revien.graph.store import GraphStore


@pytest.fixture
def store():
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    s = GraphStore(db_path=fd.name)
    yield s
    s.close()
    Path(fd.name).unlink(missing_ok=True)


def test_default_node_is_plain_text(store):
    """A node created without modality args defaults to text/answerable/unprocessed."""
    n = Node(node_type=NodeType.CONTEXT, label="hi", content="just text")
    store.add_node(n)
    got = store.get_node(n.node_id)
    assert got.source_modality is Modality.TEXT
    assert got.answerable_by_text is True
    assert got.vision_processed is False


def test_image_evidence_roundtrips(store):
    """An image turn whose answer lives in the picture: not answerable by text."""
    n = Node(
        node_type=NodeType.CONTEXT,
        label="Melanie: take a look at this",
        content="Melanie: take a look at this",
        source_modality=Modality.IMAGE,
        answerable_by_text=False,
        vision_processed=False,
    )
    store.add_node(n)
    got = store.get_node(n.node_id)
    assert got.source_modality is Modality.IMAGE
    assert got.answerable_by_text is False
    assert got.vision_processed is False


def test_vision_pass_can_flip_flags(store):
    """A later vision pass marks the image processed + now answerable by (caption) text."""
    n = Node(
        node_type=NodeType.CONTEXT, label="photo", content="photo",
        source_modality=Modality.IMAGE, answerable_by_text=False,
    )
    store.add_node(n)
    store.update_node(n.node_id, vision_processed=True, answerable_by_text=True)
    got = store.get_node(n.node_id)
    assert got.vision_processed is True
    assert got.answerable_by_text is True
    assert got.source_modality is Modality.IMAGE  # unchanged


def test_modality_is_queryable_column(store):
    """source_modality is a real column (indexed), not buried in metadata JSON."""
    store.add_node(Node(node_type=NodeType.CONTEXT, label="t", content="t"))
    store.add_node(Node(
        node_type=NodeType.CONTEXT, label="i", content="i",
        source_modality=Modality.IMAGE, answerable_by_text=False,
    ))
    conn = store._get_conn()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
    assert {"source_modality", "answerable_by_text", "vision_processed"} <= cols
    n_img = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE source_modality = 'image'"
    ).fetchone()[0]
    assert n_img == 1


# Pre-Leg-1 (confidence-era) nodes table: 17 columns, ending at invalidated_at.
_LEGACY_COLUMNS = (
    "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL, "
    "content TEXT NOT NULL, source_id TEXT DEFAULT '', created_at TEXT NOT NULL, "
    "last_accessed TEXT NOT NULL, access_count INTEGER DEFAULT 0, "
    "metadata TEXT DEFAULT '{}', source_type TEXT DEFAULT 'inferred', "
    "confidence REAL DEFAULT 0.5, pinned INTEGER DEFAULT 0, "
    "confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '', "
    "source_context TEXT DEFAULT '', last_referenced TEXT, invalidated_at TEXT"
)


def test_migration_backfills_legacy_db():
    """An old DB with no modality columns upgrades in place; old rows read as text."""
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    path = fd.name
    try:
        # Hand-build a legacy nodes table + one legacy row, no modality columns.
        conn = sqlite3.connect(path)
        conn.execute(f"CREATE TABLE nodes ({_LEGACY_COLUMNS})")
        conn.execute(
            "INSERT INTO nodes (node_id, node_type, label, content, created_at, "
            "last_accessed) VALUES ('legacy-1', 'context', 'old', 'old text', "
            "'2023-05-08T00:00:00+00:00', '2023-05-08T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        # Opening via GraphStore must migrate the table and backfill defaults.
        s = GraphStore(db_path=path)
        cols = {r[1] for r in s._get_conn().execute(
            "PRAGMA table_info(nodes)").fetchall()}
        assert {"source_modality", "answerable_by_text", "vision_processed"} <= cols

        got = s.get_node("legacy-1")
        assert got is not None
        assert got.source_modality is Modality.TEXT
        assert got.answerable_by_text is True
        assert got.vision_processed is False
        s.close()
    finally:
        Path(path).unlink(missing_ok=True)
