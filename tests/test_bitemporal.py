"""
B2 — Bi-temporal validity: supersession closes/opens windows, recall(as_of=)
answers "what was true AT this time?".

The headline scenario: "I live in Boston" (said in January) superseded by
"I live in Portland" (said in June). A July query returns Portland; an
as_of=March query returns Boston — the superseded fact, recovered through
its validity window.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.supersession_ingest import ClaimGovernor
from revien.retrieval.engine import RetrievalEngine


def _dt(month, day=1):
    return datetime(2026, month, day, tzinfo=timezone.utc)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


class TestValidityStorage:
    def test_fresh_db_has_columns_and_roundtrip(self, store):
        node = store.add_node(Node(
            node_type=NodeType.FACT, label="addr", content="lives in Boston",
            valid_from=_dt(1), valid_until=_dt(6),
        ))
        got = store.get_node(node.node_id)
        assert got.valid_from == _dt(1)
        assert got.valid_until == _dt(6)

    def test_set_node_validity_partial_and_audited(self, store):
        node = store.add_node(Node(
            node_type=NodeType.FACT, label="addr", content="lives in Boston",
        ))
        assert store.set_node_validity(
            node.node_id, valid_until=_dt(6), actor="test") is True
        got = store.get_node(node.node_id)
        assert got.valid_until == _dt(6)
        assert got.valid_from is None
        ops = [r["op"] for r in store.get_all_audit() if r["node_id"] == node.node_id]
        assert "validity" in ops

    def test_set_bound_never_overwritten(self, store):
        node = store.add_node(Node(
            node_type=NodeType.FACT, label="addr", content="lives in Boston",
            valid_until=_dt(6),
        ))
        # A second supersession racing the first must not move the recorded
        # transition — the write is refused, not applied.
        assert store.set_node_validity(node.node_id, valid_until=_dt(9)) is False
        assert store.get_node(node.node_id).valid_until == _dt(6)

    def test_unknown_node_returns_false(self, store):
        assert store.set_node_validity("nope", valid_until=_dt(6)) is False


class TestSupersessionClosesWindow:
    def test_supersede_closes_old_opens_new_at_content_time(self, store):
        gov = ClaimGovernor(store, GraphOperations(store))
        # The favourite dimension is the one ordinary fact path the gate
        # auto-supersedes offline (no recognizer needed).
        old = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="fav1",
            content="My favourite color is blue.", recorded_at=_dt(1),
        ))
        new = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="fav2",
            content="My favourite color is green.", recorded_at=_dt(6),
        ))
        outcomes = gov.govern(new)
        assert [o.action for o in outcomes] == ["auto_supersede"]

        old_after = store.get_node(old.node_id)
        new_after = store.get_node(new.node_id)
        assert old_after.invalidated_at is not None
        # Window closed at the NEW claim's content time, opened from the old
        # claim's own content time:
        assert old_after.valid_until == _dt(6)
        assert old_after.valid_from == _dt(1)
        # New claim's validity opens at the same transition instant:
        assert new_after.valid_from == _dt(6)
        assert new_after.valid_until is None


class TestAsOfRecall:
    def _superseded_pair(self, store):
        """Boston (Jan, superseded in June) -> Portland (June, live)."""
        from revien.graph.schema import Edge, EdgeType
        anchor = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Home",
            content="where she lives",
        ))
        boston = store.add_node(Node(
            node_type=NodeType.FACT, label="lives in Boston",
            content="She lives in Boston.", recorded_at=_dt(1),
            valid_from=_dt(1), valid_until=_dt(6),
        ))
        portland = store.add_node(Node(
            node_type=NodeType.FACT, label="lives in Portland",
            content="She lives in Portland.", recorded_at=_dt(6),
            valid_from=_dt(6),
        ))
        GraphOperations(store).invalidate_node(
            boston.node_id, reason="superseded", construct_id="test")
        for t in (boston, portland):
            store.add_edge(Edge(
                edge_type=EdgeType.RELATED_TO,
                source_node_id=anchor.node_id, target_node_id=t.node_id,
                weight=0.5,
            ))
        return boston, portland

    def test_no_as_of_returns_only_live_fact(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        boston, portland = self._superseded_pair(store)
        engine = RetrievalEngine(store)
        ids = {r.node_id for r in engine.recall("where does she live home",
                                                top_n=10).results}
        assert portland.node_id in ids
        assert boston.node_id not in ids

    def test_as_of_march_recovers_superseded_fact(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        boston, portland = self._superseded_pair(store)
        engine = RetrievalEngine(store)
        ids = {r.node_id for r in engine.recall(
            "where does she live home", top_n=10, as_of=_dt(3)).results}
        # March: Boston was true (window covers it, despite invalidation);
        # Portland was NOT yet true (valid_from June).
        assert boston.node_id in ids
        assert portland.node_id not in ids

    def test_as_of_july_returns_current_fact(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        boston, portland = self._superseded_pair(store)
        engine = RetrievalEngine(store)
        ids = {r.node_id for r in engine.recall(
            "where does she live home", top_n=10, as_of=_dt(7)).results}
        assert portland.node_id in ids
        assert boston.node_id not in ids

    def test_invalidated_without_window_stays_hidden(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        from revien.graph.schema import Edge, EdgeType
        anchor = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Home", content="where she lives",
        ))
        ghost = store.add_node(Node(
            node_type=NodeType.FACT, label="lives in Denver",
            content="She lives in Denver.",
        ))
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=anchor.node_id, target_node_id=ghost.node_id,
            weight=0.5,
        ))
        GraphOperations(store).invalidate_node(
            ghost.node_id, reason="wrong", construct_id="test")
        engine = RetrievalEngine(store)
        ids = {r.node_id for r in engine.recall(
            "where does she live home", top_n=10, as_of=_dt(3)).results}
        # Invalidated with NO window: validity unknown, not historical.
        assert ghost.node_id not in ids
