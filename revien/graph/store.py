"""
Revien Graph Store — SQLite-backed persistent graph storage.
Every node, every edge, every relationship — stored, never compacted.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import Edge, EdgeType, Graph, Node, NodeType, SourceType


class GraphStore:
    """SQLite-backed graph store. Thread-safe via check_same_thread=False."""

    def __init__(self, db_path: str = "revien.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create tables if they don't exist, with confidence-layer columns.

        Fresh databases get the confidence columns directly in the CREATE TABLE
        statements. Existing databases are upgraded by
        ``_migrate_add_confidence_columns`` (see also
        ``revien/graph/migrations/001_confidence_layer.py`` for a standalone
        migration runner).
        """
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
                metadata TEXT DEFAULT '{}',
                source_type TEXT DEFAULT 'inferred',
                confidence REAL DEFAULT 0.5,
                pinned INTEGER DEFAULT 0,
                confidence_set_at TEXT,
                confidence_set_by TEXT DEFAULT '',
                source_context TEXT DEFAULT '',
                last_referenced TEXT
            );

            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                confidence REAL DEFAULT 0.5,
                confidence_set_at TEXT,
                confidence_set_by TEXT DEFAULT '',
                source_context TEXT DEFAULT '',
                FOREIGN KEY (source_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target_node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
            CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence);
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        # Upgrade pre-confidence databases in place (backwards compatibility).
        self._migrate_add_confidence_columns(conn)

    def _migrate_add_confidence_columns(self, conn: sqlite3.Connection) -> None:
        """Add confidence-layer columns to existing nodes/edges tables.

        No-op for fresh databases (columns already created above). For
        databases created before the confidence layer, this ALTERs the tables
        and lets the column defaults backfill old rows
        (source_type='inferred', confidence=0.5, pinned=0).
        """
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}

            now_iso = datetime.now(timezone.utc).isoformat()

            if "source_type" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN source_type TEXT DEFAULT 'inferred'")
            if "confidence" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN confidence REAL DEFAULT 0.5")
            if "pinned" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN pinned INTEGER DEFAULT 0")
            if "confidence_set_at" not in columns:
                conn.execute(f"ALTER TABLE nodes ADD COLUMN confidence_set_at TEXT DEFAULT '{now_iso}'")
            if "confidence_set_by" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN confidence_set_by TEXT DEFAULT ''")
            if "source_context" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN source_context TEXT DEFAULT ''")
            if "last_referenced" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN last_referenced TEXT")

            cursor = conn.execute("PRAGMA table_info(edges)")
            edge_columns = {row[1] for row in cursor.fetchall()}

            if "confidence" not in edge_columns:
                conn.execute("ALTER TABLE edges ADD COLUMN confidence REAL DEFAULT 0.5")
            if "confidence_set_at" not in edge_columns:
                conn.execute(f"ALTER TABLE edges ADD COLUMN confidence_set_at TEXT DEFAULT '{now_iso}'")
            if "confidence_set_by" not in edge_columns:
                conn.execute("ALTER TABLE edges ADD COLUMN confidence_set_by TEXT DEFAULT ''")
            if "source_context" not in edge_columns:
                conn.execute("ALTER TABLE edges ADD COLUMN source_context TEXT DEFAULT ''")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence)")

            conn.commit()
        except sqlite3.OperationalError as e:
            # Column already exists or other migration issue — log but don't fail.
            print(f"Migration note: {e}")

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
                created_at, last_accessed, access_count, metadata,
                source_type, confidence, pinned, confidence_set_at,
                confidence_set_by, source_context, last_referenced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                node.source_type.value,
                node.confidence,
                1 if node.pinned else 0,
                node.confidence_set_at.isoformat(),
                node.confidence_set_by,
                node.source_context,
                node.last_referenced.isoformat() if node.last_referenced else None,
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
        allowed_fields = (
            "label", "content", "source_id", "access_count", "metadata",
            "last_accessed", "confidence", "pinned", "source_type",
            "confidence_set_at", "confidence_set_by", "source_context",
            "last_referenced",
        )
        for field in allowed_fields:
            if field in kwargs:
                updates[field] = kwargs[field]

        if not updates:
            return existing

        conn = self._get_conn()
        set_clauses = []
        values = []
        datetime_fields = ("last_accessed", "confidence_set_at", "last_referenced")
        for key, val in updates.items():
            set_clauses.append(f"{key} = ?")
            if key == "metadata":
                values.append(json.dumps(val))
            elif key in datetime_fields:
                values.append(val.isoformat() if isinstance(val, datetime) else val)
            elif key == "source_type":
                values.append(val.value if hasattr(val, "value") else val)
            elif key == "pinned":
                values.append(1 if val else 0)
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
                weight, created_at, metadata,
                confidence, confidence_set_at, confidence_set_by, source_context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.edge_id,
                edge.edge_type.value,
                edge.source_node_id,
                edge.target_node_id,
                edge.weight,
                edge.created_at.isoformat(),
                json.dumps(edge.metadata),
                edge.confidence,
                edge.confidence_set_at.isoformat(),
                edge.confidence_set_by,
                edge.source_context,
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

    def update_edge_weight(self, edge_id: str, weight: float) -> bool:
        """Update an edge's weight. Clamps to [0.0, 1.0]. Returns True if updated."""
        weight = max(0.0, min(1.0, weight))
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE edges SET weight = ? WHERE edge_id = ?",
            (weight, edge_id),
        )
        conn.commit()
        return cursor.rowcount > 0

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
            # Confidence Layer (columns 9+)
            source_type=SourceType(row[9]) if len(row) > 9 and row[9] else SourceType.INFERRED,
            confidence=float(row[10]) if len(row) > 10 and row[10] is not None else 0.5,
            pinned=bool(row[11]) if len(row) > 11 else False,
            confidence_set_at=datetime.fromisoformat(row[12]) if len(row) > 12 and row[12] else datetime.now(timezone.utc),
            confidence_set_by=row[13] if len(row) > 13 and row[13] else "",
            source_context=row[14] if len(row) > 14 and row[14] else "",
            last_referenced=datetime.fromisoformat(row[15]) if len(row) > 15 and row[15] else None,
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
            # Confidence Layer (columns 7+)
            confidence=float(row[7]) if len(row) > 7 and row[7] is not None else 0.5,
            confidence_set_at=datetime.fromisoformat(row[8]) if len(row) > 8 and row[8] else datetime.now(timezone.utc),
            confidence_set_by=row[9] if len(row) > 9 and row[9] else "",
            source_context=row[10] if len(row) > 10 and row[10] else "",
        )
