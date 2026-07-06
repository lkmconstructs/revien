"""
Revien Graph Walker — BFS traversal from anchor nodes with distance tracking.
Collects all reachable nodes within max_depth hops.

Perf shape (OPEN 2): ONE traversal computes both distances and paths, and
neighbors are fetched one round-trip per BFS LEVEL (store.get_neighbors_bulk)
instead of one per visited node. The engine used to run walk() AND
walk_with_paths() back to back — two full traversals, one SELECT per node
each — per recall.
"""

from collections import deque
from typing import Dict, List, Tuple

from revien.graph.store import GraphStore


class GraphWalker:
    """
    Walks the graph from anchor nodes using BFS.
    Tracks shortest distance to each discovered node.
    """

    def __init__(self, store: GraphStore, max_depth: int = 3):
        self.store = store
        self.max_depth = max_depth

    def walk_full(
        self, anchor_ids: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Level-synchronous BFS from all anchors at once.

        Returns:
            (distances, paths):
              distances: node_id -> shortest graph distance from any anchor
                         (0 = anchor itself)
              paths:     node_id -> node_ids from its nearest anchor to it
        """
        distances: Dict[str, int] = {}
        paths: Dict[str, List[str]] = {}

        # Seed with anchors that actually exist (single bulk existence check).
        existing = self.store.get_nodes_bulk(dict.fromkeys(anchor_ids))
        frontier: List[str] = []
        for anchor_id in anchor_ids:
            if anchor_id in existing and anchor_id not in distances:
                distances[anchor_id] = 0
                paths[anchor_id] = [anchor_id]
                frontier.append(anchor_id)

        depth = 0
        while frontier and depth < self.max_depth:
            neighbor_map = self.store.get_neighbors_bulk(frontier)
            next_frontier: List[str] = []
            next_dist = depth + 1
            for current_id in frontier:
                current_path = paths[current_id]
                for neighbor_id in neighbor_map.get(current_id, ()):
                    if neighbor_id not in distances:
                        distances[neighbor_id] = next_dist
                        paths[neighbor_id] = current_path + [neighbor_id]
                        next_frontier.append(neighbor_id)
            frontier = next_frontier
            depth = next_dist

        return distances, paths

    def walk(self, anchor_ids: List[str]) -> Dict[str, int]:
        """
        BFS walk from anchor nodes.

        Args:
            anchor_ids: Starting node IDs for traversal

        Returns:
            Dict mapping node_id -> shortest graph distance from any anchor.
            Distance 0 = anchor node itself.
        """
        distances, _ = self.walk_full(anchor_ids)
        return distances

    def walk_with_paths(
        self, anchor_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        BFS walk that also tracks the path from anchor to each node.
        Useful for explaining why a node was retrieved.

        Returns:
            Dict mapping node_id -> list of node_ids from anchor to this node.
        """
        _, paths = self.walk_full(anchor_ids)
        return paths
