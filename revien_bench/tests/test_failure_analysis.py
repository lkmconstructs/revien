"""
Tests for revien_bench.failure_analysis — the per-query miss taxonomy.

Every cause class gets a synthetic scenario: never_extracted, outranked,
filtered_out, no_anchors, walk_depth_miss, disconnected. Plus the aggregate
fold and a live engine.recall(debug=True) diagnostics smoke test.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex

from revien_bench import failure_analysis as FA


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _node(store, label, dia=None, node_type=NodeType.FACT):
    now = datetime.now(timezone.utc)
    md = {"dia_id": dia} if dia else {}
    n = Node(
        node_type=node_type,
        label=label,
        content=f"content about {label}",
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        created_at=now,
        last_accessed=now,
        metadata=md,
    )
    return store.add_node(n)


def _edge(store, a, b):
    store.add_edge(Edge(
        edge_type=EdgeType.RELATED_TO,
        source_node_id=a.node_id,
        target_node_id=b.node_id,
        weight=0.5,
    ))


class TestBuildDiaMap:
    def test_maps_only_tagged_nodes(self, store):
        a = _node(store, "alpha", dia="D1:1")
        b = _node(store, "beta", dia="D1:1")
        _node(store, "untagged")
        m = FA.build_dia_map(store)
        assert set(m) == {"D1:1"}
        assert set(m["D1:1"]) == {a.node_id, b.node_id}


class TestClassifyMisses:
    def test_hit_is_not_classified(self, store):
        _node(store, "alpha", dia="D1:1")
        out = FA.classify_misses(
            store, {"anchors": {"all": []}, "scores": {}, "filtered": {}},
            gold={"D1:1"}, retrieved=["D1:1"], dia_map=FA.build_dia_map(store),
        )
        assert out == {}

    def test_never_extracted(self, store):
        out = FA.classify_misses(
            store, {"anchors": {"all": []}, "scores": {}, "filtered": {}},
            gold={"D9:9"}, retrieved=[], dia_map={},
        )
        assert out["D9:9"]["cause"] == "never_extracted"

    def test_outranked_with_rank_detail(self, store):
        gold_node = _node(store, "gold", dia="D1:2")
        diag = {
            "anchors": {"all": ["x"]},
            # Three competitors scored above the gold node.
            "scores": {gold_node.node_id: 0.10, "a": 0.9, "b": 0.8, "c": 0.5},
            "filtered": {},
        }
        out = FA.classify_misses(
            store, diag, gold={"D1:2"}, retrieved=[],
            dia_map=FA.build_dia_map(store),
        )
        assert out["D1:2"]["cause"] == "outranked"
        assert out["D1:2"]["best_rank"] == 4
        assert out["D1:2"]["best_score"] == 0.10

    def test_outranked_wins_over_filtered_when_any_node_scored(self, store):
        """A dia_id often maps to several nodes (context + extracts). The most
        favorable node decides the class."""
        n1 = _node(store, "scored", dia="D1:3")
        n2 = _node(store, "filtered", dia="D1:3")
        diag = {
            "anchors": {"all": ["x"]},
            "scores": {n1.node_id: 0.2},
            "filtered": {n2.node_id: "invalidated"},
        }
        out = FA.classify_misses(
            store, diag, gold={"D1:3"}, retrieved=[],
            dia_map=FA.build_dia_map(store),
        )
        assert out["D1:3"]["cause"] == "outranked"

    def test_filtered_out(self, store):
        n = _node(store, "hidden", dia="D1:4")
        diag = {
            "anchors": {"all": ["x"]},
            "scores": {},
            "filtered": {n.node_id: "invalidated"},
        }
        out = FA.classify_misses(
            store, diag, gold={"D1:4"}, retrieved=[],
            dia_map=FA.build_dia_map(store),
        )
        assert out["D1:4"]["cause"] == "filtered_out"
        assert out["D1:4"]["reasons"] == ["invalidated"]

    def test_no_anchors(self, store):
        _node(store, "exists", dia="D1:5")
        diag = {"anchors": {"all": []}, "scores": {}, "filtered": {}}
        out = FA.classify_misses(
            store, diag, gold={"D1:5"}, retrieved=[],
            dia_map=FA.build_dia_map(store),
        )
        assert out["D1:5"]["cause"] == "no_anchors"

    def test_walk_depth_vs_disconnected(self, store):
        # Chain: anchor -> b -> c -> d -> far_gold (distance 4 > max_depth 3,
        # but reachable at DEEP_DEPTH). island_gold has no edges at all.
        anchor = _node(store, "anchor")
        b = _node(store, "b")
        c = _node(store, "c")
        d = _node(store, "d")
        far_gold = _node(store, "far", dia="D1:6")
        island_gold = _node(store, "island", dia="D1:7")
        _edge(store, anchor, b)
        _edge(store, b, c)
        _edge(store, c, d)
        _edge(store, d, far_gold)

        diag = {
            "anchors": {"all": [anchor.node_id]},
            # Walked frontier at max_depth 3 never reached either gold node.
            "scores": {},
            "filtered": {},
        }
        out = FA.classify_misses(
            store, diag, gold={"D1:6", "D1:7"}, retrieved=[],
            dia_map=FA.build_dia_map(store),
        )
        assert out["D1:6"]["cause"] == "walk_depth_miss"
        assert out["D1:7"]["cause"] == "disconnected"
        assert island_gold.node_id  # (used: distinct from far_gold)


class TestAggregate:
    def test_folds_causes_and_tolerates_old_rows(self):
        rows = [
            {  # classified row: two misses
                "category_name": "multi_hop",
                "gold_evidence": ["D1:1", "D1:2"],
                "gold_miss_causes": {
                    "D1:1": {"cause": "outranked", "best_rank": 12, "best_score": 0.1},
                    "D1:2": {"cause": "never_extracted"},
                },
            },
            {  # classified row: clean hit
                "category_name": "single_hop",
                "gold_evidence": ["D1:3"],
                "gold_miss_causes": {},
            },
            {  # resumed row from a pre-taxonomy checkpoint
                "category_name": "single_hop",
                "gold_evidence": ["D1:4"],
            },
            {  # adversarial row without gold evidence — ignored entirely
                "category_name": "adversarial",
                "gold_evidence": [],
            },
        ]
        agg = FA.aggregate_failures(rows)
        assert agg["gold_items_missed"] == 2
        assert agg["by_cause"] == {"outranked": 1, "never_extracted": 1}
        assert agg["by_category"]["multi_hop"]["outranked"] == 1
        assert agg["rows_classified"] == 2
        assert agg["rows_unclassified"] == 1
        assert agg["outranked_detail"]["median_best_rank"] == 12


class TestEngineDiagnostics:
    """recall(debug=True) must expose what the classifier needs; debug=False
    must stay diagnostics-free (zero behavior change on the default path)."""

    def test_debug_diagnostics_shape(self, store):
        a = _node(store, "postgres", dia="D1:1")
        b = _node(store, "hetzner", dia="D1:2")
        _edge(store, a, b)
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))

        resp = eng.recall("postgres", debug=True)
        d = resp.diagnostics
        assert d is not None
        assert set(d["anchors"]) == {"entity", "keyword", "semantic", "all"}
        assert d["anchors"]["all"], "query should have anchored on the postgres node"
        assert a.node_id in d["node_distances"]
        assert a.node_id in d["scores"]
        assert d["max_depth"] == 3

    def test_default_recall_has_no_diagnostics(self, store):
        _node(store, "postgres", dia="D1:1")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.recall("postgres").diagnostics is None
