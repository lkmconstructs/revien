"""
Revien Deduplication — Prevents duplicate nodes from polluting the graph.
Uses exact label match + fuzzy match (Levenshtein distance + ratio-based similarity).
"""

from typing import List, Optional, Tuple

from revien.graph.schema import Edge, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations


class Deduplicator:
    """
    Deduplication engine for ingestion.
    Before creating a node, checks if a semantically equivalent node exists.
    If found, increments the existing node's access_count and returns it
    instead of creating a duplicate.
    """

    def __init__(self, store: GraphStore, ops: GraphOperations):
        self.store = store
        self.ops = ops

    def deduplicate_node(self, candidate: Node) -> Tuple[Node, bool]:
        """
        Check if a semantically equivalent node exists.

        Returns:
            (node, is_new): The node to use and whether it was newly created.
            If a duplicate exists, returns the existing node (with incremented
            access_count) and is_new=False.
        """
        # Context nodes are never deduplicated — each session is unique
        if candidate.node_type == NodeType.CONTEXT:
            stored = self.store.add_node(candidate)
            return stored, True

        # 1. Exact label match (case-insensitive), same type
        existing = self._find_exact_match(candidate)
        if existing:
            self.ops.touch_node(existing.node_id)
            return existing, False

        # 2. Fuzzy match (Levenshtein < 3), same type
        existing = self._find_fuzzy_match(candidate)
        if existing:
            self.ops.touch_node(existing.node_id)
            return existing, False

        # 3. No match — create new node
        stored = self.store.add_node(candidate)
        return stored, True

    def deduplicate_nodes(
        self, candidates: List[Node]
    ) -> List[Tuple[Node, bool]]:
        """Deduplicate a batch of candidate nodes."""
        results = []
        for candidate in candidates:
            results.append(self.deduplicate_node(candidate))
        return results

    def _find_exact_match(self, candidate: Node) -> Optional[Node]:
        """Find a node with the exact same label and type."""
        return self.ops.find_node_by_label(
            candidate.label, node_type=candidate.node_type
        )

    def _find_fuzzy_match(self, candidate: Node) -> Optional[Node]:
        """Find a similar node using Levenshtein distance and ratio matching."""
        matches = self.ops.find_nodes_by_label_fuzzy(
            candidate.label, max_distance=5, min_ratio=0.75
        )

        # Filter to same type
        for match in matches:
            if match.node_type == candidate.node_type:
                return match
        return None
