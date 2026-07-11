"""Claim Sovereignty Layer — Leg 2 (ingestion-time temporal resolution) tests.

Slice 1 here: the temporal SCHEMA + store round-trip, recorded_at propagation
through the pipeline, and the in-place migration of a pre-Leg-2 database. The
resolver itself (relative->absolute, fuzzy->range) is exercised in
test_temporal_resolver.py.
"""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from revien.graph.schema import Node, NodeType, TemporalGranularity
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

MAY8 = datetime(2023, 5, 8, tzinfo=timezone.utc)


@pytest.fixture
def store():
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    s = GraphStore(db_path=fd.name)
    yield s
    s.close()
    Path(fd.name).unlink(missing_ok=True)


def test_temporal_fields_default_none(store):
    n = Node(node_type=NodeType.CONTEXT, label="x", content="x")
    store.add_node(n)
    got = store.get_node(n.node_id)
    assert got.recorded_at is None
    assert got.event_time_start is None
    assert got.event_time_end is None
    assert got.event_time_granularity is None
    assert got.event_time_confidence is None
    assert got.event_time_text == ""


def test_event_time_range_roundtrips(store):
    """A YEAR-granular event is stored as a [start, end] range, not a single day."""
    y0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    y1 = datetime(2022, 12, 31, tzinfo=timezone.utc)
    n = Node(
        node_type=NodeType.CONTEXT, label="paint", content="painted a sunrise",
        recorded_at=MAY8,
        event_time_start=y0, event_time_end=y1,
        event_time_granularity=TemporalGranularity.YEAR,
        event_time_confidence=0.6, event_time_text="last year",
    )
    store.add_node(n)
    got = store.get_node(n.node_id)
    assert got.recorded_at == MAY8
    assert got.event_time_start == y0
    assert got.event_time_end == y1
    assert got.event_time_granularity is TemporalGranularity.YEAR
    assert got.event_time_confidence == 0.6
    assert got.event_time_text == "last year"
    # The honesty property: the range spans the whole year, not a guessed instant.
    assert (got.event_time_end - got.event_time_start).days >= 364


def test_pipeline_propagates_recorded_at(store):
    """IngestionInput.timestamp now reaches the node as recorded_at (was dropped)."""
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(
        source_id="s1",
        content="Melanie: we use PostgreSQL.",
        timestamp=MAY8,
    ))
    ctx = store.get_node(out.context_node_id)
    assert ctx.recorded_at == MAY8
    # recorded_at is distinct from created_at (when the row was written).
    assert ctx.created_at != MAY8
    # Every node from the unit shares the recorded_at anchor.
    others = [n for n in store.list_nodes(limit=10**9)]
    assert all(n.recorded_at == MAY8 for n in others)


def test_pipeline_recorded_at_none_when_no_timestamp(store):
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(source_id="s2", content="A: hello there"))
    assert store.get_node(out.context_node_id).recorded_at is None


# Pre-Leg-2 (Leg-1 era) nodes table: 20 columns, ending at vision_processed.
_PRE_TEMPORAL_COLUMNS = (
    "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL, "
    "content TEXT NOT NULL, source_id TEXT DEFAULT '', created_at TEXT NOT NULL, "
    "last_accessed TEXT NOT NULL, access_count INTEGER DEFAULT 0, "
    "metadata TEXT DEFAULT '{}', source_type TEXT DEFAULT 'inferred', "
    "confidence REAL DEFAULT 0.5, pinned INTEGER DEFAULT 0, "
    "confidence_set_at TEXT, confidence_set_by TEXT DEFAULT '', "
    "source_context TEXT DEFAULT '', last_referenced TEXT, invalidated_at TEXT, "
    "source_modality TEXT DEFAULT 'text', answerable_by_text INTEGER DEFAULT 1, "
    "vision_processed INTEGER DEFAULT 0"
)


def test_migration_backfills_pre_temporal_db():
    """A Leg-1-era DB upgrades in place; old rows read with no recorded/event time."""
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    path = fd.name
    try:
        conn = sqlite3.connect(path)
        conn.execute(f"CREATE TABLE nodes ({_PRE_TEMPORAL_COLUMNS})")
        conn.execute(
            "INSERT INTO nodes (node_id, node_type, label, content, created_at, "
            "last_accessed) VALUES ('leg1-1', 'context', 'old', 'old', "
            "'2023-05-08T00:00:00+00:00', '2023-05-08T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        s = GraphStore(db_path=path)
        cols = {r[1] for r in s._get_conn().execute(
            "PRAGMA table_info(nodes)").fetchall()}
        assert {"recorded_at", "event_time_start", "event_time_end",
                "event_time_granularity", "event_time_confidence",
                "event_time_text"} <= cols

        got = s.get_node("leg1-1")
        assert got is not None
        assert got.recorded_at is None
        assert got.event_time_start is None
        assert got.event_time_granularity is None
        assert got.event_time_text == ""
        s.close()
    finally:
        Path(path).unlink(missing_ok=True)
