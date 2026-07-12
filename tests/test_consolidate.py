"""
B3.1 — Dream-mode consolidation: decay persists for the right nodes and only
those, orphans are reported-not-deleted, invalidation is opt-in + soft, and
the report is never silent about what happened.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from revien.consolidate import Consolidator, REPORT_ITEM_CAP
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


def _stale(store, label, source_type=SourceType.INFERRED, pinned=False,
           weeks_old=10, confidence=0.8):
    """An old node whose last_referenced is weeks in the past (decay-eligible
    when INFERRED and un-pinned)."""
    old = datetime.now(timezone.utc) - timedelta(weeks=weeks_old)
    return store.add_node(Node(
        node_type=NodeType.FACT, label=label, content=f"fact about {label}",
        source_type=source_type, confidence=confidence, pinned=pinned,
        last_referenced=old,
    ))


class TestDecayPass:
    def test_inferred_stale_node_decays_and_is_reported(self, store):
        n = _stale(store, "stale-inferred")
        report = Consolidator(store).run(recluster=False)
        after = store.get_node(n.node_id)
        assert after.confidence < 0.8
        assert report.nodes_decayed == 1
        assert report.decayed_sample[0]["node_id"] == n.node_id
        assert report.decayed_sample[0]["before"] > report.decayed_sample[0]["after"]

    def test_pinned_and_extracted_are_immune(self, store):
        pinned = _stale(store, "pinned", pinned=True)
        extracted = _stale(store, "extracted", source_type=SourceType.EXTRACTED)
        report = Consolidator(store).run(recluster=False)
        assert store.get_node(pinned.node_id).confidence == 0.8
        assert store.get_node(extracted.node_id).confidence == 0.8
        assert report.nodes_decayed == 0

    def test_decay_is_audited(self, store):
        n = _stale(store, "audited")
        Consolidator(store).run(recluster=False)
        ops = [r["op"] for r in store.get_all_audit() if r["node_id"] == n.node_id]
        assert "decay" in ops

    def test_decay_pass_can_be_disabled(self, store):
        n = _stale(store, "untouched")
        report = Consolidator(store).run(decay=False, recluster=False)
        assert store.get_node(n.node_id).confidence == 0.8
        assert report.decay_ran is False


class TestOrphanPass:
    def _orphan(self, store, label="lonely", node_type=NodeType.ENTITY):
        return store.add_node(Node(
            node_type=node_type, label=label, content=f"about {label}",
        ))

    def test_orphans_reported_not_touched_by_default(self, store):
        orphan = self._orphan(store)
        report = Consolidator(store).run(recluster=False)
        assert report.orphans_found == 1
        assert report.orphan_sample[0]["node_id"] == orphan.node_id
        assert report.orphans_invalidated == 0
        assert store.get_node(orphan.node_id).invalidated_at is None

    def test_invalidate_orphans_is_soft_and_audited(self, store):
        orphan = self._orphan(store)
        report = Consolidator(store).run(
            recluster=False, invalidate_orphans=True)
        after = store.get_node(orphan.node_id)
        assert report.orphans_invalidated == 1
        assert after is not None            # soft: the node still exists
        assert after.invalidated_at is not None

    def test_connected_nodes_are_not_orphans(self, store):
        a = self._orphan(store, "a")
        b = self._orphan(store, "b")
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=a.node_id, target_node_id=b.node_id, weight=0.5,
        ))
        report = Consolidator(store).run(recluster=False)
        assert report.orphans_found == 0

    def test_context_nodes_exempt(self, store):
        # A verbatim turn stands alone by design — never an orphan finding.
        self._orphan(store, "bare turn", node_type=NodeType.CONTEXT)
        report = Consolidator(store).run(recluster=False)
        assert report.orphans_found == 0

    def test_already_invalidated_not_recounted(self, store):
        orphan = self._orphan(store)
        GraphOperations(store).invalidate_node(
            orphan.node_id, reason="test", construct_id="test")
        report = Consolidator(store).run(recluster=False)
        assert report.orphans_found == 0


class TestReport:
    def test_report_serializes_with_all_sections(self, store):
        _stale(store, "one")
        d = Consolidator(store).run(recluster=False).to_dict()
        for key in ("started_at", "duration_ms", "decay", "recluster",
                    "reindex", "orphans"):
            assert key in d
        assert d["decay"]["ran"] is True
        assert d["reindex"]["ran"] is False  # backfill stays opt-in

    def test_sample_capped(self, store):
        for i in range(REPORT_ITEM_CAP + 10):
            store.add_node(Node(
                node_type=NodeType.ENTITY, label=f"o{i}", content=f"c{i}",
            ))
        report = Consolidator(store).run(recluster=False)
        assert report.orphans_found == REPORT_ITEM_CAP + 10
        assert len(report.orphan_sample) == REPORT_ITEM_CAP


class TestStoreOrphanHelper:
    def test_sql_helper_matches_edge_reality(self, store):
        a = store.add_node(Node(node_type=NodeType.ENTITY, label="a", content="a"))
        b = store.add_node(Node(node_type=NodeType.ENTITY, label="b", content="b"))
        c = store.add_node(Node(node_type=NodeType.ENTITY, label="c", content="c"))
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=a.node_id, target_node_id=b.node_id, weight=0.5,
        ))
        assert store.list_orphan_node_ids() == [c.node_id]
