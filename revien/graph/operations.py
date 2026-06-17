"""
Revien Graph Operations — Higher-level operations on the graph store.
Convenience layer over raw CRUD.
Confidence layer: confidence tagging, reinforcement, decay, propagation
(2-hop bounded).
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .schema import Edge, EdgeType, Graph, Node, NodeType, SourceType
from .store import GraphStore


class GraphOperations:
    """High-level graph operations built on GraphStore."""

    # Decay demotes but never deletes: INFERRED nodes decay toward this floor,
    # never to zero. Only explicit correction (correct_node) sets confidence to
    # 0.0. Aged memories stay retrievable but rank low — "store everything,
    # compact nothing."
    DECAY_FLOOR = 0.15

    def __init__(self, store: GraphStore):
        self.store = store

    # ── Confidence Layer: Scoring, Decay, Reinforcement ───────

    def _apply_decay(self, node: Node) -> Node:
        """Apply lazy decay to INFERRED nodes not referenced recently.

        Decay rate: -0.01 per week since last_referenced (or created_at).
        Decay DEMOTES but never DELETES: confidence floors at DECAY_FLOOR, so
        aged memories rank low but stay retrievable. Only explicit correction
        (correct_node) drives confidence to 0.0. Pinned and non-INFERRED nodes
        are immune. Persists only when the change is meaningful (> 0.005).
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node

        now = datetime.now(timezone.utc)
        reference_time = node.last_referenced or node.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        days_since = (now - reference_time).days
        weeks_since = days_since / 7.0
        decay_amount = weeks_since * 0.01

        # Demote, don't delete: floor at DECAY_FLOOR; min() guard never raises
        # a node already sitting below the floor.
        new_confidence = min(node.confidence, max(self.DECAY_FLOOR, node.confidence - decay_amount))
        if node.confidence - new_confidence > 0.005:
            return self.store.update_node(
                node.node_id,
                confidence=new_confidence,
                confidence_set_at=now,
                source_context="lazy_decay",
            ) or node
        return node

    def reinforce_node(self, node_id: str, construct_id: str = "") -> Optional[Node]:
        """Reinforce a node's confidence after successful use.

        Increment confidence by +0.05 (capped at 1.0). Update last_referenced
        timestamp and audit fields.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc)
        new_confidence = min(1.0, node.confidence + 0.05)
        return self.store.update_node(
            node_id,
            confidence=new_confidence,
            confidence_set_at=now,
            confidence_set_by=construct_id,
            source_context="reinforced_after_use",
            last_referenced=now,
            metadata={**node.metadata, "_reinforced_by": construct_id},
        )

    def correct_node(
        self, node_id: str, correction_context: str = "", construct_id: str = ""
    ) -> Optional[Node]:
        """Mark a node as CORRECTED due to conflicting information.

        Set source_type to CORRECTED and confidence to 0.0.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc)
        return self.store.update_node(
            node_id,
            source_type=SourceType.CORRECTED,
            confidence=0.0,
            confidence_set_at=now,
            confidence_set_by=construct_id,
            source_context=correction_context,
            metadata={
                **node.metadata,
                "_corrected_by": construct_id,
                "_corrected_at": now.isoformat(),
            },
        )

    def propagate_confidence(
        self, source_node_id: str, max_hops: int = 2
    ) -> Dict[str, float]:
        """Propagate confidence from a source node to connected nodes.

        Confidence decays per hop and is capped at 2 hops max (bounded
        propagation). Returns dict of {node_id: new_confidence} for the nodes
        whose confidence would be raised by propagation.
        """
        if max_hops > 2:
            max_hops = 2

        source = self.store.get_node(source_node_id)
        if source is None:
            return {}

        updates: Dict[str, float] = {}
        visited = {source_node_id}
        frontier = [(source_node_id, source.confidence, 0)]  # (node_id, confidence, depth)

        while frontier:
            current_id, current_conf, depth = frontier.pop(0)
            if depth >= max_hops:
                continue

            edges = self.store.get_edges_for_node(current_id)
            for edge in edges:
                neighbor_id = (
                    edge.target_node_id
                    if edge.source_node_id == current_id
                    else edge.source_node_id
                )

                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # Propagated confidence decays by 0.1 per hop.
                propagated = current_conf * (1.0 - (0.1 * (depth + 1)))
                propagated = max(0.0, min(1.0, propagated))

                neighbor = self.store.get_node(neighbor_id)
                if neighbor and propagated > neighbor.confidence:
                    updates[neighbor_id] = propagated
                    frontier.append((neighbor_id, propagated, depth + 1))

        # Apply updates (mark propagated; confidence raise tracked in updates dict).
        for node_id in updates:
            node = self.store.get_node(node_id)
            if node is not None:
                self.store.update_node(
                    node_id,
                    metadata={**node.metadata, "_propagated": True},
                )

        return updates

    def retrieve_with_confidence(
        self,
        node_type: Optional[NodeType] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict], List[float]]:
        """Retrieve nodes with confidence scoring.

        Applies lazy decay, then returns (nodes_as_dicts, confidence_scores)
        sorted by confidence descending.
        """
        raw_nodes = self.store.list_nodes(
            node_type=node_type, source_id=source_id, limit=limit, offset=offset
        )

        nodes_with_scores = []
        for node in raw_nodes:
            decayed_node = self._apply_decay(node)
            score = decayed_node.confidence
            nodes_with_scores.append((decayed_node.model_dump(), score))

        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        nodes = [n for n, _ in nodes_with_scores]
        scores = [s for _, s in nodes_with_scores]
        return nodes, scores

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
        self, label: str, max_distance: int = 5, min_ratio: float = 0.75
    ) -> List[Node]:
        """Find nodes with similar labels using both Levenshtein distance
        and ratio-based matching. Ratio-based matching handles length
        differences better (e.g., 'PostgreSQL' vs 'Postgres')."""
        from difflib import SequenceMatcher
        nodes = self.store.list_nodes(limit=999999)
        matches = []
        for node in nodes:
            label_lower = label.lower()
            node_lower = node.label.lower()
            dist = _levenshtein(node_lower, label_lower)
            ratio = SequenceMatcher(None, node_lower, label_lower).ratio()
            if dist < max_distance or ratio >= min_ratio:
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
