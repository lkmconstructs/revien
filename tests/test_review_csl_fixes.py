"""External-review fixes (truth maintenance): favourite shortcut, ingestion
chronology, validity-window integrity, idempotent decay. Behaviour is checked
at the surface a caller sees (ingest -> governance outcome -> recall)."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from revien.consolidate import ConsolidationReport, Consolidator
from revien.fact_change import detect_change
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.semantic.index import SemanticIndex


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("REVIEN_CSL", "1")
    for k in ("REVIEN_SENSITIVITY_BACKEND", "REVIEN_TENSION_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    d = tempfile.mkdtemp(prefix="revien-csl-")
    s = GraphStore(os.path.join(d, "t.db"))
    yield s
    s.close()


def _pipe(store):
    return IngestionPipeline(store, semantic=SemanticIndex(store, enabled=False))


def _ctx(store, needle):
    return [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=99)
            if needle in n.content]


# -- C: favourite shortcut ---------------------------------------------


@pytest.mark.parametrize("new", [
    "I moved to Boston now.",
    "Alice switched jobs, she is at Google now.",
])
def test_unrelated_transition_does_not_retire_favourite(store, new):
    assert detect_change("My favorite food is pizza.", new) == (None, False)
    p = _pipe(store)
    p.ingest(IngestionInput(source_id="u", content="My favorite food is pizza."))
    out = p.ingest(IngestionInput(source_id="u", content=new))
    assert out.governance == []
    assert all(n.invalidated_at is None for n in _ctx(store, "pizza"))


def test_genuine_favourite_change_still_supersedes(store):
    assert detect_change("My favorite food is pizza.",
                         "My favourite food is sushi now.") == ("favourite", True)
    p = _pipe(store)
    p.ingest(IngestionInput(source_id="u", content="My favorite food is pizza."))
    out = p.ingest(IngestionInput(source_id="u", content="My favourite food is sushi now."))
    assert [g.action for g in out.governance] == ["auto_supersede"]
    assert all(n.invalidated_at is not None for n in _ctx(store, "pizza"))


def test_same_favourite_restated_is_not_a_change():
    assert detect_change("My favorite food is pizza.",
                         "My favorite food is pizza now.") == ("favourite", False)


@pytest.mark.parametrize("old,new", [
    ("The Office is hands down my favourite comfort show.",
     "My comfort show used to be The Office but it's Parks and Rec now."),
    ("Blue has always been my favourite colour.",
     "I've completely gone off blue, it's forest green for me these days."),
    ("My favourite coffee order is a flat white.",
     "I'm an oat milk latte person now, that's my go-to."),
])
def test_favourite_change_with_category_named_on_one_side(old, new):
    # The first cut of C sent these to the general path; the fact-update
    # bench fell from 8/68 to 3/68. A shared topic or a favourite-shaped new
    # claim is the link a bare "now" lacks.
    assert detect_change(old, new) == ("favourite", True)


# -- D: chronology before supersession ---------------------------------


def _color(store, text, when):
    return _pipe(store).ingest(IngestionInput(
        source_id="u", content=text, timestamp=when))


JUNE = datetime(2026, 6, 1, tzinfo=timezone.utc)
JAN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_late_history_does_not_retire_current_claim(store):
    _color(store, "My favorite color is green.", JUNE)
    out = _color(store, "My favorite color is blue now.", JAN)
    assert [g.action for g in out.governance] == ["auto_supersede"]
    green, = _ctx(store, "green")
    blue, = _ctx(store, "blue")
    assert green.invalidated_at is None
    assert green.valid_from is None and green.valid_until is None
    assert blue.invalidated_at is not None
    assert blue.valid_until == JUNE
    assert blue.valid_from is None or blue.valid_from <= blue.valid_until


def test_in_order_claims_still_update(store):
    _color(store, "My favorite color is green.", JAN)
    _color(store, "My favorite color is blue now.", JUNE)
    green, = _ctx(store, "green")
    blue, = _ctx(store, "blue")
    assert green.invalidated_at is not None
    assert green.valid_from == JAN and green.valid_until == JUNE
    assert blue.invalidated_at is None and blue.valid_from == JUNE


def test_equal_timestamps_supersede_without_a_window(store):
    _color(store, "My favorite color is green.", JUNE)
    _color(store, "My favorite color is blue now.", JUNE)
    green, = _ctx(store, "green")
    assert green.invalidated_at is not None
    assert green.valid_from is None and green.valid_until is None


def test_store_rejects_inverted_window(store):
    n = store.add_node(Node(node_type=NodeType.FACT, label="x", content="y",
                            valid_from=JAN))
    with pytest.raises(ValueError, match="inverted validity window"):
        store.set_node_validity(n.node_id, valid_until=JAN - timedelta(days=1))
    assert store.get_node(n.node_id).valid_until is None
    with pytest.raises(ValueError, match="inverted validity window"):
        store.add_node(Node(node_type=NodeType.FACT, label="x2", content="y2",
                            valid_from=JUNE, valid_until=JAN))


# -- E: idempotent decay -----------------------------------------------


def _aged_inferred(store, weeks=10, confidence=0.5):
    old = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    return store.add_node(Node(
        node_type=NodeType.FACT, label="inferred thing", content="x",
        source_type=SourceType.INFERRED, confidence=confidence,
        created_at=old, last_referenced=old))


def test_repeated_decay_passes_charge_once(store):
    n = _aged_inferred(store)
    c = Consolidator(store, GraphOperations(store))
    seen = []
    for _ in range(3):
        c._decay_pass(ConsolidationReport())
        seen.append(round(store.get_node(n.node_id).confidence, 3))
    assert seen == [0.4, 0.4, 0.4]
    assert "confidence_decayed_at" in store.get_node(n.node_id).metadata


def test_decay_charges_only_the_new_interval(store):
    n = _aged_inferred(store)
    ops = GraphOperations(store)
    ops._apply_decay(store.get_node(n.node_id))
    # Two more weeks pass: wind the marker back two weeks.
    node = store.get_node(n.node_id)
    marker = datetime.fromisoformat(node.metadata["confidence_decayed_at"])
    store.update_node(n.node_id, _audit_op=None, metadata={
        **node.metadata,
        "confidence_decayed_at": (marker - timedelta(weeks=2)).isoformat()})
    ops._apply_decay(store.get_node(n.node_id))
    assert store.get_node(n.node_id).confidence == pytest.approx(0.38, abs=0.002)


def test_decay_read_path_does_not_double_charge(store):
    n = _aged_inferred(store)
    ops = GraphOperations(store)
    ops._apply_decay(store.get_node(n.node_id))
    node = store.get_node(n.node_id)
    assert ops._compute_decayed_confidence(node) == pytest.approx(node.confidence)


def test_pinned_and_non_inferred_nodes_do_not_decay(store):
    old = datetime.now(timezone.utc) - timedelta(weeks=10)
    pinned = store.add_node(Node(node_type=NodeType.FACT, label="p", content="p",
                                 source_type=SourceType.INFERRED, confidence=0.5,
                                 pinned=True, created_at=old, last_referenced=old))
    extracted = store.add_node(Node(node_type=NodeType.FACT, label="e", content="e",
                                    source_type=SourceType.EXTRACTED, confidence=0.5,
                                    created_at=old, last_referenced=old))
    Consolidator(store, GraphOperations(store))._decay_pass(ConsolidationReport())
    assert store.get_node(pinned.node_id).confidence == 0.5
    assert store.get_node(extracted.node_id).confidence == 0.5
