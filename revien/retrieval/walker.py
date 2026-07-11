"""
Revien Graph Walker — BFS traversal from anchor nodes with distance tracking.
Collects all reachable nodes within max_depth hops.

Perf shape (OPEN 2): ONE traversal computes distances, paths, and path
strengths, and neighbors are fetched one round-trip per BFS LEVEL
(store.get_neighbors_weighted_bulk) instead of one per visited node. The
engine used to run walk() AND walk_with_paths() back to back — two full
traversals, one SELECT per node each — per recall.

Path strength (A1, weighted walk): edges carry weight (author-drawn vault
edges 0.8, extractor co-occurrence guesses 0.3) and mark_used() reinforcement,
but BFS treated every edge identically. The walk now tracks, per node, the
best product of edge strengths along its shortest-hop path — anchors are 1.0,
each hop multiplies by that edge's strength. Hops stay the walk's frontier
metric (a weak edge still DISCOVERS a node — ranking, not reachability, is
where strength bites); among same-level parents the strongest path wins.
"""

from typing import Dict, List, Tuple

from revien.graph.store import GraphStore


class GraphWalker:
    """
    Walks the graph from anchor nodes using BFS.
    Tracks shortest distance and strongest same-length path to each node.
    """

    def __init__(
        self,
        store: GraphStore,
        max_depth: int = 3,
        use_edge_confidence: bool = False,
    ):
        self.store = store
        self.max_depth = max_depth
        # When True, edge strength = weight * edge confidence; default weight
        # only. Env-gated at the engine (REVIEN_EDGE_CONFIDENCE_IN_WALK).
        self.use_edge_confidence = use_edge_confidence

    def walk_full(
        self, anchor_ids: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, float]]:
        """
        Level-synchronous BFS from all anchors at once.

        Returns:
            (distances, paths, strengths):
              distances: node_id -> shortest graph distance from any anchor
                         (0 = anchor itself)
              paths:     node_id -> node_ids from its nearest anchor to it
              strengths: node_id -> best product of edge strengths along a
                         shortest-hop path (anchors = 1.0)
        """
        distances: Dict[str, int] = {}
        paths: Dict[str, List[str]] = {}
        strengths: Dict[str, float] = {}

        # Seed with anchors that actually exist (single bulk existence check).
        existing = self.store.get_nodes_bulk(dict.fromkeys(anchor_ids))
        frontier: List[str] = []
        for anchor_id in anchor_ids:
            if anchor_id in existing and anchor_id not in distances:
                distances[anchor_id] = 0
                paths[anchor_id] = [anchor_id]
                strengths[anchor_id] = 1.0
                frontier.append(anchor_id)

        depth = 0
        while frontier and depth < self.max_depth:
            neighbor_map = self.store.get_neighbors_weighted_bulk(
                frontier, use_confidence=self.use_edge_confidence
            )
            next_frontier: List[str] = []
            next_dist = depth + 1
            for current_id in frontier:
                current_path = paths[current_id]
                current_strength = strengths[current_id]
                for neighbor_id, edge_strength in neighbor_map.get(current_id, ()):
                    path_strength = current_strength * edge_strength
                    if neighbor_id not in distances:
                        distances[neighbor_id] = next_dist
                        paths[neighbor_id] = current_path + [neighbor_id]
                        strengths[neighbor_id] = path_strength
                        next_frontier.append(neighbor_id)
                    elif (
                        distances[neighbor_id] == next_dist
                        and path_strength > strengths[neighbor_id]
                    ):
                        # Discovered this level via a weaker parent: same hop
                        # count, stronger path. Upgrade strength AND the
                        # explain-path so they tell the same story. Children
                        # expand next level and read the upgraded value.
                        strengths[neighbor_id] = path_strength
                        paths[neighbor_id] = current_path + [neighbor_id]
            frontier = next_frontier
            depth = next_dist

        return distances, paths, strengths

    def walk(self, anchor_ids: List[str]) -> Dict[str, int]:
        """
        BFS walk from anchor nodes.

        Args:
            anchor_ids: Starting node IDs for traversal

        Returns:
            Dict mapping node_id -> shortest graph distance from any anchor.
            Distance 0 = anchor node itself.
        """
        distances, _, _ = self.walk_full(anchor_ids)
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
        _, paths, _ = self.walk_full(anchor_ids)
        return paths
