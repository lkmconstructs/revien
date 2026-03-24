"""
Test 1: Graph CRUD
Create nodes of every type. Create edges of every type. Update nodes.
Delete nodes and verify edges are cleaned up. Export graph to JSON and
reimport. Verify graph integrity after reimport.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations


@pytest.fixture
def store():
    """Create a temporary GraphStore for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def ops(store):
    return GraphOperations(store)


# ── Node Creation (every type) ────────────────────────────

class TestNodeCreation:
    def test_create_entity_node(self, store):
        node = Node(node_type=NodeType.ENTITY, label="Alice", content="Project lead")
        result = store.add_node(node)
        assert result.node_id == node.node_id
        assert result.node_type == NodeType.ENTITY

    def test_create_topic_node(self, store):
        node = Node(node_type=NodeType.TOPIC, label="pricing strategy", content="Discussion about tier pricing")
        result = store.add_node(node)
        assert result.node_type == NodeType.TOPIC

    def test_create_decision_node(self, store):
        node = Node(node_type=NodeType.DECISION, label="Enterprise tier at $499/mo", content="We decided on $499/month with 20% annual discount")
        result = store.add_node(node)
        assert result.node_type == NodeType.DECISION

    def test_create_fact_node(self, store):
        node = Node(node_type=NodeType.FACT, label="PostgreSQL for DB", content="Database layer uses PostgreSQL, not MySQL")
        result = store.add_node(node)
        assert result.node_type == NodeType.FACT

    def test_create_preference_node(self, store):
        node = Node(node_type=NodeType.PREFERENCE, label="Prefers Python", content="Prefers Python over JavaScript for backend")
        result = store.add_node(node)
        assert result.node_type == NodeType.PREFERENCE

    def test_create_event_node(self, store):
        node = Node(node_type=NodeType.EVENT, label="Deployed PatternWall v4.3", content="Deployed PatternWall v4.3 on Feb 16")
        result = store.add_node(node)
        assert result.node_type == NodeType.EVENT

    def test_create_context_node(self, store):
        node = Node(node_type=NodeType.CONTEXT, label="Pricing session", content="Session where we scoped enterprise pricing")
        result = store.add_node(node)
        assert result.node_type == NodeType.CONTEXT

    def test_all_node_types_covered(self):
        """Verify we test every NodeType."""
        assert len(NodeType) == 7


# ── Edge Creation (every type) ────────────────────────────

class TestEdgeCreation:
    def _make_two_nodes(self, store):
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="Node A", content="A"))
        n2 = store.add_node(Node(node_type=NodeType.ENTITY, label="Node B", content="B"))
        return n1, n2

    def test_create_related_to_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.RELATED_TO, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.RELATED_TO

    def test_create_decided_in_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.DECIDED_IN, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.DECIDED_IN

    def test_create_mentioned_by_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.MENTIONED_BY, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.MENTIONED_BY

    def test_create_depends_on_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.DEPENDS_ON, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.DEPENDS_ON

    def test_create_followed_by_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.FOLLOWED_BY, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.FOLLOWED_BY

    def test_create_contradicts_edge(self, store):
        n1, n2 = self._make_two_nodes(store)
        edge = Edge(edge_type=EdgeType.CONTRADICTS, source_node_id=n1.node_id, target_node_id=n2.node_id)
        result = store.add_edge(edge)
        assert result.edge_type == EdgeType.CONTRADICTS

    def test_all_edge_types_covered(self):
        assert len(EdgeType) == 6


# ── Update Operations ─────────────────────────────────────

class TestNodeUpdate:
    def test_update_label(self, store):
        node = store.add_node(Node(node_type=NodeType.FACT, label="Old label", content="Content"))
        updated = store.update_node(node.node_id, label="New label")
        assert updated.label == "New label"
        assert updated.content == "Content"  # unchanged

    def test_update_content(self, store):
        node = store.add_node(Node(node_type=NodeType.FACT, label="Label", content="Old content"))
        updated = store.update_node(node.node_id, content="New content")
        assert updated.content == "New content"

    def test_update_access_count(self, store):
        node = store.add_node(Node(node_type=NodeType.ENTITY, label="Test", content="Test"))
        updated = store.update_node(node.node_id, access_count=5)
        assert updated.access_count == 5

    def test_update_metadata(self, store):
        node = store.add_node(Node(node_type=NodeType.ENTITY, label="Test", content="Test"))
        updated = store.update_node(node.node_id, metadata={"tool": "claude-code"})
        assert updated.metadata == {"tool": "claude-code"}

    def test_update_nonexistent_returns_none(self, store):
        result = store.update_node("nonexistent-id", label="Nope")
        assert result is None


# ── Delete Operations ─────────────────────────────────────

class TestNodeDeletion:
    def test_delete_node(self, store):
        node = store.add_node(Node(node_type=NodeType.ENTITY, label="Doomed", content="Gone soon"))
        assert store.delete_node(node.node_id) is True
        assert store.get_node(node.node_id) is None

    def test_delete_cleans_up_edges(self, store):
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        n2 = store.add_node(Node(node_type=NodeType.ENTITY, label="B", content="B"))
        edge = store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=n1.node_id,
            target_node_id=n2.node_id,
        ))
        store.delete_node(n1.node_id)
        # Edge should be gone
        assert store.get_edge(edge.edge_id) is None
        # n2 should have no edges
        assert len(store.get_edges_for_node(n2.node_id)) == 0

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_node("fake-id") is False

    def test_no_orphaned_edges_after_deletion(self, store):
        """Create a hub node with multiple edges, delete it, verify zero orphans."""
        hub = store.add_node(Node(node_type=NodeType.TOPIC, label="Hub", content="Hub"))
        spokes = []
        for i in range(5):
            n = store.add_node(Node(node_type=NodeType.ENTITY, label=f"Spoke {i}", content=f"S{i}"))
            spokes.append(n)
            store.add_edge(Edge(
                edge_type=EdgeType.RELATED_TO,
                source_node_id=hub.node_id,
                target_node_id=n.node_id,
            ))
        assert store.count_edges() == 5
        store.delete_node(hub.node_id)
        assert store.count_edges() == 0


# ── Export/Import ─────────────────────────────────────────

class TestExportImport:
    def test_export_produces_valid_graph(self, store):
        store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        store.add_node(Node(node_type=NodeType.TOPIC, label="B", content="B"))
        graph = store.export_graph()
        assert len(graph.nodes) == 2
        assert graph.version == "1.0"

    def test_export_import_roundtrip(self, store):
        # Build a graph
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="Alice", content="Project lead"))
        n2 = store.add_node(Node(node_type=NodeType.DECISION, label="$499/mo", content="Enterprise pricing"))
        n3 = store.add_node(Node(node_type=NodeType.CONTEXT, label="Session 1", content="Pricing discussion"))
        store.add_edge(Edge(edge_type=EdgeType.DECIDED_IN, source_node_id=n2.node_id, target_node_id=n3.node_id, weight=0.9))
        store.add_edge(Edge(edge_type=EdgeType.MENTIONED_BY, source_node_id=n2.node_id, target_node_id=n1.node_id, weight=0.7))

        # Export
        graph = store.export_graph()
        graph_json = graph.model_dump_json()

        # Reimport into fresh store
        fd, path2 = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store2 = GraphStore(db_path=path2)
        try:
            reimported = Graph.model_validate_json(graph_json)
            store2.import_graph(reimported)

            # Verify integrity
            assert store2.count_nodes() == 3
            assert store2.count_edges() == 2

            # Verify specific node data survived
            alice = store2.get_node(n1.node_id)
            assert alice is not None
            assert alice.label == "Alice"
            assert alice.node_type == NodeType.ENTITY

            decision = store2.get_node(n2.node_id)
            assert decision is not None
            assert decision.label == "$499/mo"
            assert decision.content == "Enterprise pricing"

            # Verify edges survived
            edges = store2.get_edges_for_node(n2.node_id)
            assert len(edges) == 2
            edge_types = {e.edge_type for e in edges}
            assert EdgeType.DECIDED_IN in edge_types
            assert EdgeType.MENTIONED_BY in edge_types
        finally:
            store2.close()
            os.unlink(path2)

    def test_import_clears_existing_data(self, store):
        store.add_node(Node(node_type=NodeType.ENTITY, label="Old", content="Old"))
        assert store.count_nodes() == 1

        new_graph = Graph(
            nodes=[Node(node_type=NodeType.FACT, label="New", content="New")],
            edges=[],
        )
        store.import_graph(new_graph, clear_existing=True)
        assert store.count_nodes() == 1
        nodes = store.list_nodes()
        assert nodes[0].label == "New"


# ── Operations Layer ──────────────────────────────────────

class TestGraphOperations:
    def test_find_node_by_label(self, store, ops):
        store.add_node(Node(node_type=NodeType.ENTITY, label="Alice", content="Project lead"))
        store.add_node(Node(node_type=NodeType.ENTITY, label="Bob", content="Engineer"))
        result = ops.find_node_by_label("alice")  # case-insensitive
        assert result is not None
        assert result.label == "Alice"

    def test_find_node_fuzzy(self, store, ops):
        store.add_node(Node(node_type=NodeType.ENTITY, label="PostgreSQL", content="Database"))
        results = ops.find_nodes_by_label_fuzzy("Postgres")  # distance ~3
        assert len(results) >= 1

    def test_touch_node_increments(self, store, ops):
        node = store.add_node(Node(node_type=NodeType.ENTITY, label="Test", content="Test"))
        assert node.access_count == 0
        touched = ops.touch_node(node.node_id)
        assert touched.access_count == 1
        touched2 = ops.touch_node(node.node_id)
        assert touched2.access_count == 2

    def test_connect_nodes(self, store, ops):
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        n2 = store.add_node(Node(node_type=NodeType.TOPIC, label="B", content="B"))
        edge = ops.connect_nodes(n1.node_id, n2.node_id, EdgeType.RELATED_TO, weight=0.8)
        assert edge.weight == 0.8
        assert store.count_edges() == 1

    def test_get_node_with_edges(self, store, ops):
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="Center", content="Hub"))
        n2 = store.add_node(Node(node_type=NodeType.TOPIC, label="Spoke", content="Spoke"))
        ops.connect_nodes(n1.node_id, n2.node_id, EdgeType.RELATED_TO)
        result = ops.get_node_with_edges(n1.node_id)
        assert result is not None
        assert len(result["connected_nodes"]) == 1
        assert result["connected_nodes"][0]["label"] == "Spoke"

    def test_get_subgraph(self, store, ops):
        # Chain: A -> B -> C -> D
        a = store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        b = store.add_node(Node(node_type=NodeType.ENTITY, label="B", content="B"))
        c = store.add_node(Node(node_type=NodeType.ENTITY, label="C", content="C"))
        d = store.add_node(Node(node_type=NodeType.ENTITY, label="D", content="D"))
        ops.connect_nodes(a.node_id, b.node_id, EdgeType.FOLLOWED_BY)
        ops.connect_nodes(b.node_id, c.node_id, EdgeType.FOLLOWED_BY)
        ops.connect_nodes(c.node_id, d.node_id, EdgeType.FOLLOWED_BY)

        # Depth 1 from A should get A, B
        sub = ops.get_subgraph(a.node_id, max_depth=1)
        labels = {n.label for n in sub.nodes}
        assert "A" in labels
        assert "B" in labels

        # Depth 2 from A should get A, B, C
        sub2 = ops.get_subgraph(a.node_id, max_depth=2)
        labels2 = {n.label for n in sub2.nodes}
        assert "A" in labels2
        assert "B" in labels2
        assert "C" in labels2


# ── Listing and Counting ──────────────────────────────────

class TestListingAndCounting:
    def test_list_nodes_by_type(self, store):
        store.add_node(Node(node_type=NodeType.ENTITY, label="E1", content="E1"))
        store.add_node(Node(node_type=NodeType.ENTITY, label="E2", content="E2"))
        store.add_node(Node(node_type=NodeType.TOPIC, label="T1", content="T1"))
        entities = store.list_nodes(node_type=NodeType.ENTITY)
        assert len(entities) == 2
        topics = store.list_nodes(node_type=NodeType.TOPIC)
        assert len(topics) == 1

    def test_count_nodes(self, store):
        assert store.count_nodes() == 0
        store.add_node(Node(node_type=NodeType.ENTITY, label="X", content="X"))
        assert store.count_nodes() == 1

    def test_count_edges(self, store):
        assert store.count_edges() == 0
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        n2 = store.add_node(Node(node_type=NodeType.ENTITY, label="B", content="B"))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=n1.node_id, target_node_id=n2.node_id))
        assert store.count_edges() == 1

    def test_get_neighbors(self, store):
        n1 = store.add_node(Node(node_type=NodeType.ENTITY, label="A", content="A"))
        n2 = store.add_node(Node(node_type=NodeType.ENTITY, label="B", content="B"))
        n3 = store.add_node(Node(node_type=NodeType.ENTITY, label="C", content="C"))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=n1.node_id, target_node_id=n2.node_id))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=n1.node_id, target_node_id=n3.node_id))
        neighbors = store.get_neighbors(n1.node_id)
        assert len(neighbors) == 2
        assert n2.node_id in neighbors
        assert n3.node_id in neighbors
