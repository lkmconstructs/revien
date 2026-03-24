"""
Revien Graph Operations — Higher-level operations on the graph store.
Convenience layer over raw CRUD.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .schema import Edge, EdgeType, Graph, Node, NodeType
from .store import GraphStore


class GraphOperations:
    """High-level graph operations built on GraphStore."""

    def __init__(self, store: GraphStore):
        self.store = store

    def find_node_by_label(
        self, label: str, node_type: Optional[NodeType] = None
    ) -> Optional[Node]:
        """Find a node by exact label match, optionally filtered by type."""
        nodes = self.store.list_nodes(node_type=node_type, limit=999999)
        for node in nodes:
            if node.label.lower() == label.lower():
                return node
        return None

    def find_nodes_by_label_fuzzy(
        self, label: str, max_distance: int = 3
    ) -> List[Node]:
        """Find nodes with labels within Levenshtein distance."""
        nodes = self.store.list_nodes(limit=999999)
        matches = []
        for node in nodes:
            dist = _levenshtein(node.label.lower(), label.lower())
            if dist < max_distance:
                matches.append(node)
        return matches

    def connect_nodes(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 0.5,
    ) -> Edge:
        """Create an edge between two existing nodes."""
        edge = Edge(
            edge_type=edge_type,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=weight,
        )
        return self.store.add_edge(edge)

    def touch_node(self, node_id: str) -> Optional[Node]:
        """Increment access_count and update last_accessed."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        return self.store.update_node(
            node_id,
            access_count=node.access_count + 1,
            last_accessed=datetime.now(timezone.utc),
        )

    def get_node_with_edges(self, node_id: str) -> Optional[Dict]:
        """Get a node with all its edges and connected node summaries."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        edges = self.store.get_edges_for_node(node_id)
        connected = []
        for edge in edges:
            other_id = (
                edge.target_node_id
                if edge.source_node_id == node_id
                else edge.source_node_id
            )
            other_node = self.store.get_node(other_id)
            if other_node:
                connected.append(
                    {
                        "node_id": other_node.node_id,
                        "node_type": other_node.node_type.value,
                        "label": other_node.label,
                        "edge_type": edge.edge_type.value,
                        "edge_weight": edge.weight,
                    }
                )
        return {
            "node": node.model_dump(),
            "edges": [e.model_dump() for e in edges],
            "connected_nodes": connected,
        }

    def get_subgraph(self, node_id: str, max_depth: int = 2) -> Graph:
        """Extract a subgraph around a node up to max_depth hops."""
        visited_nodes = set()
        visited_edges = set()
        frontier = {node_id}

        for _ in range(max_depth + 1):
            next_frontier = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                edges = self.store.get_edges_for_node(nid)
                for edge in edges:
                    visited_edges.add(edge.edge_id)
                    other_id = (
                        edge.target_node_id
                        if edge.source_node_id == nid
                        else edge.source_node_id
                    )
                    if other_id not in visited_nodes:
                        next_frontier.add(other_id)
            frontier = next_frontier

        nodes = []
        for nid in visited_nodes:
            node = self.store.get_node(nid)
            if node:
                nodes.append(node)

        edges = []
        for eid in visited_edges:
            edge = self.store.get_edge(eid)
            if edge:
                edges.append(edge)

        return Graph(nodes=nodes, edges=edges)


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
