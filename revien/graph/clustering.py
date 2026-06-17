"""
Revien Community Detection — community clustering over the memory graph.

Assigns every node to a community. Identifies centroid nodes per community.
Used by the retrieval engine for community-first routing: queries hit
community centroids first, then search within the best-matching communities.

Architecture:
- Loads graph from SQLite into networkx (lightweight for <10K nodes)
- Runs community detection via a pluggable backend:
    * "louvain" (default) — networkx.algorithms.community.louvain_communities
      (networkx is a light, pure-Python base dependency)
    * "leiden" (opt-in) — leidenalg + python-igraph, installed via
      `pip install revien[leiden]`. If the compiled deps are missing,
      logs a warning and falls back to Louvain.
- Computes per-community centroids (highest weighted degree)
- Persists community_id on each node in SQLite
- Tracks community metadata (size, centroid, top labels) in memory
- Reclustering is debounced — runs after N ingests or on demand

Backend selection (in priority order):
1. explicit `backend=` argument to CommunityDetector(...)
2. REVIEN_CLUSTER_BACKEND environment variable
3. default: "louvain"
"""

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import networkx as nx
from networkx.algorithms.community import louvain_communities

logger = logging.getLogger("clustering")

# Valid backend identifiers
BACKEND_LOUVAIN = "louvain"
BACKEND_LEIDEN = "leiden"


@dataclass
class Community:
    """Summary of a detected community."""
    community_id: int
    size: int
    centroid_id: str  # node with highest degree centrality
    centroid_label: str
    top_labels: List[str]  # up to 5 most connected node labels
    node_ids: Set[str] = field(default_factory=set, repr=False)


def _resolve_backend(backend: Optional[str]) -> str:
    """
    Resolve the clustering backend identifier.
    Precedence: explicit arg > REVIEN_CLUSTER_BACKEND env > louvain default.
    """
    choice = (backend or os.environ.get("REVIEN_CLUSTER_BACKEND") or BACKEND_LOUVAIN).lower()
    if choice not in (BACKEND_LOUVAIN, BACKEND_LEIDEN):
        logger.warning(
            "[CLUSTER] Unknown backend %r, falling back to %s", choice, BACKEND_LOUVAIN
        )
        return BACKEND_LOUVAIN
    return choice


def _louvain_partition(G: "nx.Graph") -> List[Set[str]]:
    """Run Louvain via networkx. Returns a list of node-id sets."""
    communities_sets = louvain_communities(
        G,
        weight="weight",
        resolution=1.0,
        seed=42,
    )
    return [set(s) for s in communities_sets]


def _leiden_partition(G: "nx.Graph") -> List[Set[str]]:
    """
    Run Leiden via leidenalg + python-igraph (opt-in extra).
    Raises ImportError if the optional deps are not installed — the caller
    is responsible for catching it and falling back to Louvain.
    """
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore

    # Build an igraph Graph mirroring the networkx graph, preserving weights.
    node_list = list(G.nodes())
    index_of = {nid: i for i, nid in enumerate(node_list)}

    edges = []
    weights = []
    for src, tgt, data in G.edges(data=True):
        edges.append((index_of[src], index_of[tgt]))
        weights.append(data.get("weight", 0.5))

    ig_graph = ig.Graph(n=len(node_list), edges=edges)
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=1.0,
        seed=42,
    )

    result: List[Set[str]] = []
    for members in partition:
        result.append({node_list[i] for i in members})
    return result


class CommunityDetector:
    """
    Community detection over the Revien graph.

    Default backend is Louvain (networkx, a light base dependency). Leiden is
    available as an opt-in backend (`pip install revien[leiden]`); if its
    compiled deps are missing the detector logs a warning and falls back to
    Louvain transparently.

    Usage:
        detector = CommunityDetector(db_path)                  # louvain
        detector = CommunityDetector(db_path, backend="leiden")  # opt-in leiden
        detector.run()  # cluster the graph
        detector.get_community(node_id)  # lookup
        detector.get_communities_for_anchors([id1, id2])  # routing
    """

    # Recluster after this many ingests
    RECLUSTER_INTERVAL = 50

    def __init__(self, db_path: str, backend: Optional[str] = None):
        self.db_path = db_path
        self.backend = _resolve_backend(backend)
        self._communities: Dict[int, Community] = {}
        self._node_to_community: Dict[str, int] = {}
        self._ingest_count_since_cluster = 0
        self._last_cluster_node_count = 0

    def _partition(self, G: "nx.Graph") -> List[Set[str]]:
        """
        Partition the graph using the selected backend.
        Leiden falls back to Louvain if its optional deps are unavailable.
        """
        if self.backend == BACKEND_LEIDEN:
            try:
                logger.info("[CLUSTER] Using Leiden backend")
                return _leiden_partition(G)
            except ImportError:
                logger.warning(
                    "[CLUSTER] Leiden backend requested but leidenalg/igraph not "
                    "installed (pip install revien[leiden]); falling back to Louvain"
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "[CLUSTER] Leiden partition failed (%s); falling back to Louvain", e
                )
        return _louvain_partition(G)

    def run(self) -> Dict[int, Community]:
        """
        Run community detection on the full graph.
        Persists community_id to each node in SQLite.
        Returns dict of community_id -> Community.
        """
        conn = sqlite3.connect(self.db_path)

        # Ensure community_id column exists
        self._ensure_column(conn)

        # Load graph into networkx
        G = self._load_graph(conn)

        if G.number_of_nodes() < 3:
            logger.info("[CLUSTER] Graph too small (%d nodes), skipping", G.number_of_nodes())
            conn.close()
            return {}

        # Partition via the selected backend (Louvain default / Leiden opt-in)
        try:
            communities_sets = self._partition(G)
        except Exception as e:
            logger.error("[CLUSTER] Community detection failed: %s", e)
            conn.close()
            return {}

        # Build community metadata
        self._communities.clear()
        self._node_to_community.clear()

        # Get labels for all nodes
        label_map = {}
        rows = conn.execute("SELECT node_id, label FROM nodes").fetchall()
        for row in rows:
            label_map[row[0]] = row[1]

        # Process each community
        updates = []  # (community_id, node_id) pairs for batch update
        for cid, node_set in enumerate(communities_sets):
            node_ids = set(node_set)

            # Find centroid: highest degree within this community subgraph
            subgraph = G.subgraph(node_ids)
            degree_dict = dict(subgraph.degree(weight="weight"))

            if not degree_dict:
                continue

            centroid_id = max(degree_dict, key=degree_dict.get)
            centroid_label = label_map.get(centroid_id, "unknown")

            # Top labels by degree
            sorted_by_degree = sorted(
                degree_dict.items(), key=lambda x: x[1], reverse=True
            )
            top_labels = [
                label_map.get(nid, "?") for nid, _ in sorted_by_degree[:5]
            ]

            community = Community(
                community_id=cid,
                size=len(node_ids),
                centroid_id=centroid_id,
                centroid_label=centroid_label,
                top_labels=top_labels,
                node_ids=node_ids,
            )
            self._communities[cid] = community

            for nid in node_ids:
                self._node_to_community[nid] = cid
                updates.append((cid, nid))

        # Batch update community_id in SQLite
        conn.executemany(
            "UPDATE nodes SET community_id = ? WHERE node_id = ?",
            updates,
        )
        conn.commit()
        self._last_cluster_node_count = G.number_of_nodes()
        self._ingest_count_since_cluster = 0

        logger.info(
            "[CLUSTER] Detected %d communities across %d nodes, %d edges (backend=%s)",
            len(self._communities), G.number_of_nodes(), G.number_of_edges(), self.backend,
        )

        conn.close()
        return self._communities

    def load_from_db(self) -> bool:
        """
        Load existing community assignments from SQLite without reclustering.
        Returns True if communities were found.
        """
        conn = sqlite3.connect(self.db_path)
        self._ensure_column(conn)

        rows = conn.execute(
            "SELECT node_id, community_id, label FROM nodes WHERE community_id IS NOT NULL"
        ).fetchall()

        if not rows:
            conn.close()
            return False

        self._communities.clear()
        self._node_to_community.clear()

        # Group nodes by community
        community_nodes: Dict[int, List[tuple]] = {}
        for node_id, cid, label in rows:
            self._node_to_community[node_id] = cid
            if cid not in community_nodes:
                community_nodes[cid] = []
            community_nodes[cid].append((node_id, label))

        # Build community objects (without full degree info — just size + first node as centroid)
        for cid, nodes in community_nodes.items():
            self._communities[cid] = Community(
                community_id=cid,
                size=len(nodes),
                centroid_id=nodes[0][0],
                centroid_label=nodes[0][1],
                top_labels=[n[1] for n in nodes[:5]],
                node_ids={n[0] for n in nodes},
            )

        conn.close()
        logger.info("[CLUSTER] Loaded %d communities from DB", len(self._communities))
        return True

    def notify_ingest(self) -> bool:
        """
        Called after each ingest. Returns True if reclustering should happen.
        The caller decides whether to actually run() — keeps ingestion non-blocking.
        """
        self._ingest_count_since_cluster += 1
        return self._ingest_count_since_cluster >= self.RECLUSTER_INTERVAL

    def get_community(self, node_id: str) -> Optional[int]:
        """Get community_id for a node, or None if unclustered."""
        return self._node_to_community.get(node_id)

    def get_communities_for_anchors(self, anchor_ids: List[str]) -> List[int]:
        """
        Given a set of anchor node IDs, return the distinct communities they belong to.
        Used by retrieval engine for community-first routing.
        """
        communities = set()
        for nid in anchor_ids:
            cid = self._node_to_community.get(nid)
            if cid is not None:
                communities.add(cid)
        return sorted(communities)

    def get_community_node_ids(self, community_id: int) -> Set[str]:
        """Get all node IDs in a community."""
        community = self._communities.get(community_id)
        if community:
            return community.node_ids
        return set()

    def get_all_communities(self) -> List[Dict]:
        """Return summary of all communities for API response."""
        return [
            {
                "community_id": c.community_id,
                "size": c.size,
                "centroid_id": c.centroid_id,
                "centroid_label": c.centroid_label,
                "top_labels": c.top_labels,
            }
            for c in sorted(self._communities.values(), key=lambda c: c.size, reverse=True)
        ]

    @property
    def community_count(self) -> int:
        return len(self._communities)

    @property
    def is_clustered(self) -> bool:
        return len(self._communities) > 0

    def _load_graph(self, conn: sqlite3.Connection) -> nx.Graph:
        """Load the full graph from SQLite into networkx."""
        G = nx.Graph()

        # Load nodes
        rows = conn.execute("SELECT node_id FROM nodes").fetchall()
        for row in rows:
            G.add_node(row[0])

        # Load edges with weights
        rows = conn.execute(
            "SELECT source_node_id, target_node_id, weight FROM edges"
        ).fetchall()
        for src, tgt, weight in rows:
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, weight=weight or 0.5)

        return G

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection) -> None:
        """Add community_id column if it doesn't exist."""
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}
            if "community_id" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN community_id INTEGER")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_community ON nodes(community_id)")
                conn.commit()
                logger.info("[CLUSTER] Added community_id column to nodes table")
        except sqlite3.OperationalError as e:
            logger.warning("[CLUSTER] Column migration note: %s", e)
