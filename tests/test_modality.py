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
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.modality import (
    RETRIEVAL_FAILURE,
    UNAVAILABLE_MODALITY,
    answer_available_in_text,
    classify_miss,
)


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


# ── L1 part 2: pipeline propagation ───────────────────────────────────────────

def test_pipeline_defaults_to_text(store):
    """A caller that says nothing about modality produces plain text nodes."""
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(
        source_id="s1", content="Alice: our backend uses PostgreSQL and Python."))
    ctx = store.get_node(out.context_node_id)
    assert ctx.source_modality is Modality.TEXT
    assert ctx.answerable_by_text is True


def test_pipeline_propagates_mixed_to_context_only(store):
    """answerable_by_text=False rides onto the verbatim turn, NOT extracted nodes.

    Extracted entity/fact nodes were pulled from the text, so text answered for
    them; only the turn-as-a-whole has answer content potentially locked in the
    unread image.
    """
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(
        source_id="s2",
        content="Melanie: take a look at this. Our backend uses PostgreSQL.",
        source_modality=Modality.MIXED,
        answerable_by_text=False,
        vision_processed=False,
    ))
    ctx = store.get_node(out.context_node_id)
    assert ctx.source_modality is Modality.MIXED
    assert ctx.answerable_by_text is False

    others = [n for n in store.list_nodes(limit=10**9)
              if n.node_id != out.context_node_id]
    assert others, "expected at least one extracted node"
    # Provenance modality rides onto extracted nodes...
    assert all(n.source_modality is Modality.MIXED for n in others)
    # ...but they stay answerable by text (they came FROM the text).
    assert all(n.answerable_by_text is True for n in others)


# ── L1 part 2: miss classification ────────────────────────────────────────────

def test_classify_miss_unavailable_modality():
    """All gold nodes are unread images -> the miss is unavoidable, not retrieval."""
    img = Node(node_type=NodeType.CONTEXT, label="p", content="p",
               source_modality=Modality.MIXED, answerable_by_text=False,
               vision_processed=False)
    assert classify_miss([img]) == UNAVAILABLE_MODALITY
    assert answer_available_in_text(img) is False


def test_classify_miss_retrieval_failure_when_any_text_available():
    """If even one gold node is answerable by text, the miss is a retrieval bug."""
    txt = Node(node_type=NodeType.CONTEXT, label="t", content="t")
    img = Node(node_type=NodeType.CONTEXT, label="p", content="p",
               source_modality=Modality.MIXED, answerable_by_text=False)
    assert classify_miss([txt, img]) == RETRIEVAL_FAILURE
    assert classify_miss([txt]) == RETRIEVAL_FAILURE


def test_classify_miss_vision_processed_counts_as_available():
    """A processed image (caption now in text) is no longer a modality excuse."""
    vp = Node(node_type=NodeType.CONTEXT, label="p", content="p",
              source_modality=Modality.MIXED, answerable_by_text=True,
              vision_processed=True)
    assert answer_available_in_text(vp) is True
    assert classify_miss([vp]) == RETRIEVAL_FAILURE


def test_classify_miss_no_gold_is_retrieval_failure():
    """No gold nodes -> cannot blame modality; conservative default."""
    assert classify_miss([]) == RETRIEVAL_FAILURE


# ── L1 part 2: bench tags image turns at ingest ───────────────────────────────

def test_bench_ingest_tags_image_turn_mixed(store):
    """ingest_conversation marks an image turn MIXED/not-answerable, text turn TEXT."""
    from revien_bench.loader import Conversation, Turn
    from revien_bench.ingest_locomo import ingest_conversation

    conv = Conversation(conv_id="c1", speaker_a="A", speaker_b="B")
    conv.session_dates = {1: "8 May, 2023"}
    conv.turns = [
        Turn(dia_id="D1:1", speaker="A", text="Hi there, good to chat",
             session=1, session_date="8 May, 2023", has_image=False),
        Turn(dia_id="D1:2", speaker="B", text="take a look at this",
             session=1, session_date="8 May, 2023",
             blip_caption="a sunrise painting", has_image=True),
    ]
    ingest_conversation(conv, store, use_blip_caption=False)

    nodes = store.list_nodes(limit=10**9)
    def ctx_for(dia):
        hits = [n for n in nodes if n.node_type == NodeType.CONTEXT
                and (n.metadata or {}).get("dia_id") == dia]
        assert hits, f"no context node for {dia}"
        return hits[0]

    img = ctx_for("D1:2")
    assert img.source_modality is Modality.MIXED
    assert img.answerable_by_text is False
    assert classify_miss([img]) == UNAVAILABLE_MODALITY

    txt = ctx_for("D1:1")
    assert txt.source_modality is Modality.TEXT
    assert txt.answerable_by_text is True
