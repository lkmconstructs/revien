"""
Revien Graph Walker — BFS traversal from anchor nodes with distance tracking.
Collects all reachable nodes within max_depth hops.
"""

from collections import deque
from typing import Dict, List, Set

from revien.graph.store import GraphStore


class GraphWalker:
    """
    Walks the graph from anchor nodes using BFS.
    Tracks shortest distance to each discovered node.
    """

    def __init__(self, store: GraphStore, max_depth: int = 3):
        self.store = store
        self.max_depth = max_depth

    def walk(self, anchor_ids: List[str]) -> Dict[str, int]:
        """
        BFS walk from anchor nodes.

        Args:
            anchor_ids: Starting node IDs for traversal

        Returns:
            Dict mapping node_id -> shortest graph distance from any anchor.
            Distance 0 = anchor node itself.
        """
        # node_id -> shortest distance from any anchor
        visited: Dict[str, int] = {}

        # BFS queue: (node_id, current_distance)
        queue: deque = deque()

        # Seed with anchor nodes at distance 0
        for anchor_id in anchor_ids:
            # Verify anchor exists
            if self.store.get_node(anchor_id) is not None:
                if anchor_id not in visited:
                    visited[anchor_id] = 0
                    queue.append((anchor_id, 0))

        # BFS traversal
        while queue:
            current_id, current_dist = queue.popleft()

            # Don't explore beyond max depth
            if current_dist >= self.max_depth:
                continue

            # Get all neighbors
            neighbor_ids = self.store.get_neighbors(current_id)
            next_dist = current_dist + 1

            for neighbor_id in neighbor_ids:
                # Only visit if we haven't found a shorter path
                if neighbor_id not in visited:
                    visited[neighbor_id] = next_dist
                    queue.append((neighbor_id, next_dist))

        return visited

    def walk_with_paths(
        self, anchor_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        BFS walk that also tracks the path from anchor to each node.
        Useful for explaining why a node was retrieved.

        Returns:
            Dict mapping node_id -> list of node_ids from anchor to this node.
        """
        paths: Dict[str, List[str]] = {}
        queue: deque = deque()

        for anchor_id in anchor_ids:
            if self.store.get_node(anchor_id) is not None:
                if anchor_id not in paths:
                    paths[anchor_id] = [anchor_id]
                    queue.append((anchor_id, [anchor_id], 0))

        while queue:
            current_id, current_path, current_dist = queue.popleft()

            if current_dist >= self.max_depth:
                continue

            neighbor_ids = self.store.get_neighbors(current_id)
            next_dist = current_dist + 1

            for neighbor_id in neighbor_ids:
                if neighbor_id not in paths:
                    new_path = current_path + [neighbor_id]
                    paths[neighbor_id] = new_path
                    queue.append((neighbor_id, new_path, next_dist))

        return paths
