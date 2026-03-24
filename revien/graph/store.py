"""
Revien Graph Store — SQLite-backed persistent graph storage.
Every node, every edge, every relationship — stored, never compacted.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import Edge, EdgeType, Graph, Node, NodeType


class GraphStore:
    """SQLite-backed graph store. Thread-safe via check_same_thread=False."""

    def __init__(self, db_path: str = "revien.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                content TEXT NOT NULL,
                source_id TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Node CRUD ─────────────────────────────────────────

    def add_node(self, node: Node) -> Node:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO nodes
               (node_id, node_type, label, content, source_id,
                created_at, last_accessed, access_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.node_id,
                node.node_type.value,
                node.label,
                node.content,
                node.source_id,
                node.created_at.isoformat(),
                node.last_accessed.isoformat(),
                node.access_count,
                json.dumps(node.metadata),
            ),
        )
        conn.commit()
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, node_id: str, **kwargs) -> Optional[Node]:
        """Update specific fields on a node. Returns updated node or None."""
        existing = self.get_node(node_id)
        if existing is None:
            return None

        updates = {}
        for field in ("label", "content", "source_id", "access_count", "metadata"):
            if field in kwargs:
                updates[field] = kwargs[field]
        if "last_accessed" in kwargs:
            updates["last_accessed"] = kwargs["last_accessed"]

        if not updates:
            return existing

        conn = self._get_conn()
        set_clauses = []
        values = []
        for key, val in updates.items():
            set_clauses.append(f"{key} = ?")
            if key == "metadata":
                values.append(json.dumps(val))
            elif key == "last_accessed":
                values.append(val.isoformat() if isinstance(val, datetime) else val)
            else:
                values.append(val)
        values.append(node_id)

        conn.execute(
            f"UPDATE nodes SET {', '.join(set_clauses)} WHERE node_id = ?",
            values,
        )
        conn.commit()
        return self.get_node(node_id)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges. Returns True if deleted."""
        conn = self._get_conn()
        # Edges cascade via FK, but be explicit for safety
        conn.execute(
            "DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
            (node_id, node_id),
        )
        cursor = conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
        conn.commit()
        return cursor.rowcount > 0

    def list_nodes(
        self,
        node_type: Optional[NodeType] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Node]:
        conn = self._get_conn()
        query = "SELECT * FROM nodes WHERE 1=1"
        params: list = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type.value)
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_nodes(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    # ── Edge CRUD ─────────────────────────────────────────

    def add_edge(self, edge: Edge) -> Edge:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO edges
               (edge_id, edge_type, source_node_id, target_node_id,
                weight, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.edge_id,
                edge.edge_type.value,
                edge.source_node_id,
                edge.target_node_id,
                edge.weight,
                edge.created_at.isoformat(),
                json.dumps(edge.metadata),
            ),
        )
        conn.commit()
        return edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM edges WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node (as source or target)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM edges WHERE source_node_id = ? OR target_node_id = ?",
            (node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get IDs of all nodes directly connected to the given node."""
        edges = self.get_edges_for_node(node_id)
        neighbors = set()
        for e in edges:
            if e.source_node_id == node_id:
                neighbors.add(e.target_node_id)
            else:
                neighbors.add(e.source_node_id)
        return list(neighbors)

    def delete_edge(self, edge_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM edges WHERE edge_id = ?", (edge_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count_edges(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # ── Graph Export/Import ───────────────────────────────

    def export_graph(self) -> Graph:
        nodes = self.list_nodes(limit=999999)
        conn = self._get_conn()
        edge_rows = conn.execute("SELECT * FROM edges").fetchall()
        edges = [self._row_to_edge(r) for r in edge_rows]
        return Graph(nodes=nodes, edges=edges)

    def import_graph(self, graph: Graph, clear_existing: bool = True) -> None:
        """Import a full graph. Optionally clear existing data first."""
        conn = self._get_conn()
        if clear_existing:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.commit()
        for node in graph.nodes:
            self.add_node(node)
        for edge in graph.edges:
            self.add_edge(edge)

    # ── Internal Helpers ──────────────────────────────────

    def _row_to_node(self, row: tuple) -> Node:
        return Node(
            node_id=row[0],
            node_type=NodeType(row[1]),
            label=row[2],
            content=row[3],
            source_id=row[4],
            created_at=datetime.fromisoformat(row[5]),
            last_accessed=datetime.fromisoformat(row[6]),
            access_count=row[7],
            metadata=json.loads(row[8]),
        )

    def _row_to_edge(self, row: tuple) -> Edge:
        return Edge(
            edge_id=row[0],
            edge_type=EdgeType(row[1]),
            source_node_id=row[2],
            target_node_id=row[3],
            weight=row[4],
            created_at=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]),
        )
