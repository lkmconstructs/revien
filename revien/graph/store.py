"""
Revien Graph Store — SQLite-backed persistent graph storage.
Every node, every edge, every relationship — stored, never compacted.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    Edge, EdgeType, Graph, Modality, Node, NodeType, SourceType, TemporalGranularity,
)


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
                last_referenced TEXT,
                invalidated_at TEXT,
                source_modality TEXT DEFAULT 'text',
                answerable_by_text INTEGER DEFAULT 1,
                vision_processed INTEGER DEFAULT 0,
                recorded_at TEXT,
                event_time_start TEXT,
                event_time_end TEXT,
                event_time_granularity TEXT,
                event_time_confidence REAL,
                event_time_text TEXT DEFAULT ''
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

            -- Provenance Layer (leg 6a): append-only audit trail. Never updated,
            -- never deleted. before_json/after_json capture node state snapshots
            -- around each recorded op. No FK to nodes(node_id) on purpose — the
            -- trail must survive even if a node row is later removed.
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                op TEXT NOT NULL,
                actor TEXT DEFAULT '',
                ts TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_source ON nodes(source_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
            CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence);
            CREATE INDEX IF NOT EXISTS idx_audit_node ON audit_log(node_id);
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        # Upgrade pre-confidence databases in place (backwards compatibility).
        self._migrate_add_confidence_columns(conn)
        # Claim Sovereignty Layer (Leg 1): add modality columns to older DBs.
        self._migrate_add_modality_columns(conn)
        # Claim Sovereignty Layer (Leg 2): add temporal columns to older DBs.
        self._migrate_add_temporal_columns(conn)

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
            # Provenance Layer (leg 6a): soft-invalidation marker. Nullable,
            # backfills NULL on existing rows (every old node is live).
            if "invalidated_at" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN invalidated_at TEXT")

            # Provenance Layer (leg 6a): append-only audit trail for pre-6a DBs.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    op TEXT NOT NULL,
                    actor TEXT DEFAULT '',
                    ts TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_node ON audit_log(node_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)")

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

    def _migrate_add_modality_columns(self, conn: sqlite3.Connection) -> None:
        """Add Claim Sovereignty Layer (Leg 1) modality columns to existing DBs.

        Idempotent and no-op for fresh databases (columns already created in
        ``_ensure_db``). For older databases this ALTERs the nodes table; the
        column defaults backfill every existing row as a plain text node
        (source_modality='text', answerable_by_text=1, vision_processed=0), so
        recall/scoring stay byte-identical until something deliberately tags a
        node with a non-text modality.
        """
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}

            if "source_modality" not in columns:
                conn.execute(
                    "ALTER TABLE nodes ADD COLUMN source_modality TEXT DEFAULT 'text'"
                )
            if "answerable_by_text" not in columns:
                conn.execute(
                    "ALTER TABLE nodes ADD COLUMN answerable_by_text INTEGER DEFAULT 1"
                )
            if "vision_processed" not in columns:
                conn.execute(
                    "ALTER TABLE nodes ADD COLUMN vision_processed INTEGER DEFAULT 0"
                )

            # Index the answerability flag — the classification path filters on it
            # (unavailable_modality vs retrieval_failure) and it is low-cardinality.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_modality ON nodes(source_modality)"
            )

            conn.commit()
        except sqlite3.OperationalError as e:
            # Column already exists or other migration issue — log but don't fail.
            print(f"Migration note: {e}")

    def _migrate_add_temporal_columns(self, conn: sqlite3.Connection) -> None:
        """Add Claim Sovereignty Layer (Leg 2) temporal columns to existing DBs.

        Idempotent and no-op for fresh databases. All columns are nullable and
        backfill NULL on existing rows (recorded_at unknown, no event resolved),
        so recall/scoring stay byte-identical until ingestion populates them.
        event_time is stored as a [start, end] RANGE plus a granularity label so a
        fuzzy expression is recorded honestly, never as a false-precise instant.
        """
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}

            if "recorded_at" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN recorded_at TEXT")
            if "event_time_start" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN event_time_start TEXT")
            if "event_time_end" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN event_time_end TEXT")
            if "event_time_granularity" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN event_time_granularity TEXT")
            if "event_time_confidence" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN event_time_confidence REAL")
            if "event_time_text" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN event_time_text TEXT DEFAULT ''")

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

    # ── Provenance Layer (leg 6a): append-only audit ──────

    def record_audit(
        self,
        node_id: str,
        op: str,
        actor: str = "",
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Append one row to the audit_log. Defensive by design.

        Provenance is append-only: this only ever INSERTs. An audit-write
        failure must NEVER break the underlying op, so any error here is
        swallowed (the operation that triggered it has already committed).
        """
        try:
            conn = self._get_conn()
            stamp = (ts or datetime.now(timezone.utc)).isoformat()
            conn.execute(
                """INSERT INTO audit_log (node_id, op, actor, ts, before_json, after_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    op,
                    actor or "",
                    stamp,
                    json.dumps(before) if before is not None else None,
                    json.dumps(after) if after is not None else None,
                ),
            )
            conn.commit()
        except Exception:
            # Never let provenance bookkeeping break the underlying op.
            pass

    @staticmethod
    def _audit_snapshot(node: Optional["Node"]) -> Optional[dict]:
        """Compact, JSON-safe snapshot of a node for audit before/after fields."""
        if node is None:
            return None
        try:
            return node.model_dump(mode="json")
        except Exception:
            return None

    def _record_node_audit(
        self,
        node_id: str,
        op: str,
        actor: str = "",
        before_node: Optional["Node"] = None,
        after_node: Optional["Node"] = None,
    ) -> None:
        """Snapshot the given nodes and append an audit row — fully guarded.

        Snapshotting AND the write both happen inside one try/except so an
        audit-write failure (snapshot or INSERT) can never break the op that
        triggered it. Provenance is best-effort and append-only.
        """
        try:
            self.record_audit(
                node_id, op, actor=actor,
                before=self._audit_snapshot(before_node),
                after=self._audit_snapshot(after_node),
            )
        except Exception:
            pass

    def get_node_audit(self, node_id: str) -> list[dict]:
        """Full audit history for one node, chronological (oldest first)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, node_id, op, actor, ts, before_json, after_json
               FROM audit_log WHERE node_id = ? ORDER BY id ASC""",
            (node_id,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def get_recent_audit(self, limit: int = 50) -> list[dict]:
        """Most recent audit entries across all nodes (newest first)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, node_id, op, actor, ts, before_json, after_json
               FROM audit_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def get_all_audit(self) -> list[dict]:
        """Entire audit log, chronological (oldest first). For full export."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, node_id, op, actor, ts, before_json, after_json
               FROM audit_log ORDER BY id ASC""",
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    @staticmethod
    def _row_to_audit(row: tuple) -> dict:
        return {
            "id": row[0],
            "node_id": row[1],
            "op": row[2],
            "actor": row[3] or "",
            "ts": row[4],
            "before": json.loads(row[5]) if row[5] else None,
            "after": json.loads(row[6]) if row[6] else None,
        }

    # ── Node CRUD ─────────────────────────────────────────

    def add_node(self, node: Node) -> Node:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO nodes
               (node_id, node_type, label, content, source_id,
                created_at, last_accessed, access_count, metadata,
                source_type, confidence, pinned, confidence_set_at,
                confidence_set_by, source_context, last_referenced,
                invalidated_at, source_modality, answerable_by_text,
                vision_processed, recorded_at, event_time_start,
                event_time_end, event_time_granularity, event_time_confidence,
                event_time_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
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
                node.invalidated_at.isoformat() if node.invalidated_at else None,
                node.source_modality.value,
                1 if node.answerable_by_text else 0,
                1 if node.vision_processed else 0,
                node.recorded_at.isoformat() if node.recorded_at else None,
                node.event_time_start.isoformat() if node.event_time_start else None,
                node.event_time_end.isoformat() if node.event_time_end else None,
                node.event_time_granularity.value if node.event_time_granularity else None,
                node.event_time_confidence,
                node.event_time_text,
            ),
        )
        conn.commit()
        # Provenance hook: record node creation (append-only, never fatal).
        self._record_node_audit(
            node.node_id, "create",
            actor=node.confidence_set_by,
            before_node=None, after_node=node,
        )
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(
        self, node_id: str, _audit_op: Optional[str] = "update",
        _audit_actor: str = "", **kwargs
    ) -> Optional[Node]:
        """Update specific fields on a node. Returns updated node or None.

        Provenance (leg 6a): records an audit entry (default op ``update``)
        capturing before/after snapshots. Callers in the operations/engine layer
        that record their own semantic op (reinforce/correct/decay/invalidate/
        access) pass ``_audit_op=None`` to suppress this generic entry and avoid
        double-logging.
        """
        existing = self.get_node(node_id)
        if existing is None:
            return None

        updates = {}
        allowed_fields = (
            "label", "content", "source_id", "access_count", "metadata",
            "last_accessed", "confidence", "pinned", "source_type",
            "confidence_set_at", "confidence_set_by", "source_context",
            "last_referenced", "invalidated_at",
            # Claim Sovereignty Layer (Leg 1): modality fields are mutable so a
            # later vision pass can flip vision_processed / answerable_by_text.
            "source_modality", "answerable_by_text", "vision_processed",
            # Claim Sovereignty Layer (Leg 2): temporal fields, populated by the
            # resolver at ingest and revisable by a later verify pass.
            "recorded_at", "event_time_start", "event_time_end",
            "event_time_granularity", "event_time_confidence", "event_time_text",
        )
        for field in allowed_fields:
            if field in kwargs:
                updates[field] = kwargs[field]

        if not updates:
            return existing

        conn = self._get_conn()
        set_clauses = []
        values = []
        datetime_fields = (
            "last_accessed", "confidence_set_at", "last_referenced", "invalidated_at",
            "recorded_at", "event_time_start", "event_time_end",
        )
        for key, val in updates.items():
            set_clauses.append(f"{key} = ?")
            if key == "metadata":
                values.append(json.dumps(val))
            elif key in datetime_fields:
                values.append(val.isoformat() if isinstance(val, datetime) else val)
            elif key in ("source_type", "source_modality", "event_time_granularity"):
                values.append(val.value if hasattr(val, "value") else val)
            elif key in ("pinned", "answerable_by_text", "vision_processed"):
                values.append(1 if val else 0)
            else:
                values.append(val)
        values.append(node_id)

        conn.execute(
            f"UPDATE nodes SET {', '.join(set_clauses)} WHERE node_id = ?",
            values,
        )
        conn.commit()
        updated = self.get_node(node_id)
        # Provenance hook: generic update entry unless a semantic caller
        # (reinforce/correct/decay/invalidate/access) suppresses it.
        if _audit_op:
            self._record_node_audit(
                node_id, _audit_op, actor=_audit_actor,
                before_node=existing, after_node=updated,
            )
        return updated

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
            # Provenance Layer (column 16): soft-invalidation marker.
            invalidated_at=datetime.fromisoformat(row[16]) if len(row) > 16 and row[16] else None,
            # Claim Sovereignty Layer (columns 17-19): modality awareness. Guards
            # default to a plain text node so rows from a pre-Leg-1 DB read back
            # unchanged even before the in-place migration runs.
            source_modality=Modality(row[17]) if len(row) > 17 and row[17] else Modality.TEXT,
            answerable_by_text=bool(row[18]) if len(row) > 18 and row[18] is not None else True,
            vision_processed=bool(row[19]) if len(row) > 19 and row[19] is not None else False,
            # Claim Sovereignty Layer (columns 20-25): temporal resolution. All
            # nullable — a pre-Leg-2 row reads back with no recorded/event time.
            recorded_at=datetime.fromisoformat(row[20]) if len(row) > 20 and row[20] else None,
            event_time_start=datetime.fromisoformat(row[21]) if len(row) > 21 and row[21] else None,
            event_time_end=datetime.fromisoformat(row[22]) if len(row) > 22 and row[22] else None,
            event_time_granularity=TemporalGranularity(row[23]) if len(row) > 23 and row[23] else None,
            event_time_confidence=float(row[24]) if len(row) > 24 and row[24] is not None else None,
            event_time_text=row[25] if len(row) > 25 and row[25] else "",
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
