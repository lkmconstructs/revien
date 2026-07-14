"""
Revien Graph Store — SQLite-backed persistent graph storage.
Every node, every edge, every relationship — stored, never compacted.
"""

import functools
import json
import sqlite3
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    Edge, EdgeType, Graph, Modality, Node, NodeType, SourceType, TemporalGranularity,
)


class ImportRefusedError(ValueError):
    """import_graph(mode="refuse") hit a non-empty database. Nothing changed."""


class ImportValidationError(ValueError):
    """The import payload failed prevalidation. Nothing changed."""


def _locked(method):
    """Hold the store lock for the FULL call.

    One shared connection (check_same_thread=False) means every overlapping
    caller — the Hermes worker thread, the sync dream job in APScheduler's
    threadpool, main-thread recalls — races the same cursor. The lock is
    reentrant so audited mutations can call record_audit internally and
    content listeners can re-enter store methods without deadlocking.
    Coarse on purpose: one lock, whole method, no cleverness.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class GraphStore:
    """SQLite-backed graph store. Thread-safe: one shared connection
    (check_same_thread=False) serialized by a reentrant lock."""

    def __init__(self, db_path: str = "revien.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # Serializes ALL access to the shared connection (see _locked).
        self._lock = threading.RLock()
        # transaction() nesting depth. >0 means per-method commits are
        # deferred to the outermost transaction's single commit.
        self._txn_depth = 0
        # Listener fires queued while a transaction is open — flushed after
        # the outer commit (a listener must never see, or commit, uncommitted
        # state on the shared connection).
        self._deferred_fires: list = []
        # Content listeners: {name: (on_content_change, on_delete)}. Keyed so a
        # layer (e.g. the semantic index) registering twice replaces itself
        # instead of firing twice. Listener errors never break store writes.
        self._content_listeners: dict = {}
        self._ensure_db()

    def register_content_listener(
        self, name: str, on_content_change=None, on_delete=None
    ) -> None:
        """Register callbacks fired after a node's label/content changes or a
        node is deleted. on_content_change(node_id, label, content);
        on_delete(node_id). Re-registering the same name replaces the prior
        callbacks. This is how the semantic index keeps embeddings in sync
        with node edits (stale-vector bug: edits used to require a manual
        /v1/reindex before vector search saw the new content — now they queue
        a re-embed drained by the next search). Listeners fire while the
        store lock is held and MUST stay cheap: queue writes and small SQL,
        never model inference."""
        self._content_listeners[name] = (on_content_change, on_delete)

    def _fire_content_change(self, node: "Node") -> None:
        # Inside a transaction: defer until after the outer commit. Listeners
        # (the semantic index) write and commit on the SAME connection — an
        # inline fire would flush a half-done transaction.
        if self._txn_depth > 0:
            self._deferred_fires.append(("change", node))
            return
        for on_change, _ in list(self._content_listeners.values()):
            if on_change is None:
                continue
            try:
                on_change(node.node_id, node.label, node.content)
            except Exception:  # noqa: BLE001 - listeners must never break writes
                pass

    def _fire_delete(self, node_id: str) -> None:
        if self._txn_depth > 0:
            self._deferred_fires.append(("delete", node_id))
            return
        for _, on_delete in list(self._content_listeners.values()):
            if on_delete is None:
                continue
            try:
                on_delete(node_id)
            except Exception:  # noqa: BLE001 - listeners must never break writes
                pass

    def _flush_deferred_fires(self) -> None:
        """Fire listener callbacks queued during a transaction (post-commit)."""
        pending, self._deferred_fires = self._deferred_fires, []
        for kind, payload in pending:
            if kind == "change":
                self._fire_content_change(payload)
            else:
                self._fire_delete(payload)

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
                event_time_text TEXT DEFAULT '',
                valid_from TEXT,
                valid_until TEXT
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

            -- Claim Sovereignty Layer (Leg 3 wiring): supersession candidate queue.
            -- When the gate routes a scoped contradiction to CANDIDATE / VERSION_LOCKED
            -- (protected, sensitive, ambiguous, iron-grip, ...), NEITHER claim is
            -- mutated — the pair is parked here for human adjudication. The queue is
            -- the "preserve + surface" half of the policy; auto-supersede is the only
            -- path that mutates data, and only after the full gate clears.
            CREATE TABLE IF NOT EXISTS supersession_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                existing_node_id TEXT NOT NULL,
                new_node_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT DEFAULT '',
                trace TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                resolution TEXT
            );

            -- Distill manifest (editable-vault leg): the last-rendered snapshot
            -- per distilled note. This is the ONLY safe referent for
            -- deletion-as-rejection — a line's absence means "the user deleted
            -- it" only relative to what was last SHOWN, which the live graph
            -- cannot reconstruct (it may hold claims rendered after the user
            -- last saw the note). anchor_node_id is the immutable join key that
            -- matches the file's <!--rv:ID--> comment; current_node_id follows
            -- corrections so file and manifest never desync after an edit.
            CREATE TABLE IF NOT EXISTS distill_manifest (
                note_stem TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                anchor_node_id TEXT NOT NULL,
                current_node_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                section TEXT DEFAULT '',
                PRIMARY KEY (note_stem, anchor_node_id)
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
            CREATE INDEX IF NOT EXISTS idx_candidates_unresolved
                ON supersession_candidates(resolved_at);
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        self._commit(conn)

        # Upgrade pre-confidence databases in place (backwards compatibility).
        self._migrate_add_confidence_columns(conn)
        # Claim Sovereignty Layer (Leg 1): add modality columns to older DBs.
        self._migrate_add_modality_columns(conn)
        # Claim Sovereignty Layer (Leg 2): add temporal columns to older DBs.
        self._migrate_add_temporal_columns(conn)
        self._migrate_add_validity_columns(conn)

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

            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column/table/index already
            # there). Anything else — locked db, disk, corruption — must be
            # loud at startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

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

            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column/table/index already
            # there). Anything else — locked db, disk, corruption — must be
            # loud at startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

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

            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column/table/index already
            # there). Anything else — locked db, disk, corruption — must be
            # loud at startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    def _migrate_add_validity_columns(self, conn: sqlite3.Connection) -> None:
        """Add bi-temporal validity columns (B2) to existing DBs.

        Idempotent and no-op for fresh databases. Both columns are nullable
        and backfill NULL (validity unbounded/unknown), so recall stays
        byte-identical until a supersession closes a window or a caller
        passes as_of."""
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}

            if "valid_from" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN valid_from TEXT")
            if "valid_until" not in columns:
                conn.execute("ALTER TABLE nodes ADD COLUMN valid_until TEXT")

            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column/table/index already
            # there). Anything else — locked db, disk, corruption — must be
            # loud at startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    @_locked
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Transactions ──────────────────────────────────────

    @contextmanager
    def transaction(self):
        """Everything inside lands in ONE commit; any failure rolls back ALL
        of it. Holds the store lock for the whole span, so no other thread can
        interleave (or commit) mid-transaction on the shared connection.

        Nests: an inner transaction() rides the outer one — per-method commits
        (_commit) are no-ops until the outermost level commits. Listener fires
        queued inside are flushed only after that commit, and discarded on
        rollback (a listener must never act on state that never landed).

        NO SAVEPOINTS: nesting is depth-counting only. Only the OUTERMOST
        level's commit/rollback is real — if a caller catches an exception
        from a nested block and carries on, the nested block's partial
        statements are still sitting in the outer transaction and will land
        with its commit. Don't catch-and-continue across a failed nested
        transaction; let it propagate to the outermost level.
        """
        with self._lock:
            conn = self._get_conn()
            if self._txn_depth > 0:
                # Nested: defer to the outermost transaction's commit/rollback.
                self._txn_depth += 1
                try:
                    yield conn
                finally:
                    self._txn_depth -= 1
                return
            self._txn_depth = 1
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                self._deferred_fires = []
                raise
            finally:
                self._txn_depth = 0
            self._flush_deferred_fires()

    def _commit(self, conn: sqlite3.Connection) -> None:
        """Commit — unless a transaction() is open, whose single commit at the
        outermost level is the only one that counts."""
        if self._txn_depth == 0:
            conn.commit()

    # ── Provenance Layer (leg 6a): append-only audit ──────

    @_locked
    def record_audit(
        self,
        node_id: str,
        op: str,
        actor: str = "",
        before: Optional[dict] = None,
        after: Optional[dict] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Append one row to the audit_log. LOUD by design.

        Provenance is append-only: this only ever INSERTs. A failed write
        RAISES — a mutation the trail can't account for is worse than a
        surfaced error (the old swallow-everything shape was silent
        provenance loss). Inside a transaction() the row rides the caller's
        single commit, so mutation and audit land — or roll back — together.
        """
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
        self._commit(conn)

    @staticmethod
    def _audit_snapshot(node: Optional["Node"]) -> Optional[dict]:
        """Compact, JSON-safe snapshot of a node for audit before/after fields."""
        if node is None:
            return None
        try:
            return node.model_dump(mode="json")
        except Exception:
            return None

    @_locked
    def _record_node_audit(
        self,
        node_id: str,
        op: str,
        actor: str = "",
        before_node: Optional["Node"] = None,
        after_node: Optional["Node"] = None,
    ) -> None:
        """Snapshot the given nodes and append an audit row.

        Two contracts, keyed on whether a transaction() is open:

        * Paired with a mutation (inside a transaction): an audit failure
          RAISES, rolling the mutation back with it. Provenance is a required
          invariant — no stand-alone mutations without their audit row.
        * Stand-alone annotation (access marks, normalized-merge notes): the
          old best-effort contract holds — an audit hiccup must not break the
          recall/dedup path it decorates, so the failure is swallowed.
        """
        if self._txn_depth > 0:
            self.record_audit(
                node_id, op, actor=actor,
                before=self._audit_snapshot(before_node),
                after=self._audit_snapshot(after_node),
            )
            return
        try:
            self.record_audit(
                node_id, op, actor=actor,
                before=self._audit_snapshot(before_node),
                after=self._audit_snapshot(after_node),
            )
        except Exception:
            pass

    @_locked
    def get_node_audit(self, node_id: str) -> list[dict]:
        """Full audit history for one node, chronological (oldest first)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, node_id, op, actor, ts, before_json, after_json
               FROM audit_log WHERE node_id = ? ORDER BY id ASC""",
            (node_id,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    @_locked
    def get_recent_audit(self, limit: int = 50) -> list[dict]:
        """Most recent audit entries across all nodes (newest first)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, node_id, op, actor, ts, before_json, after_json
               FROM audit_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    @_locked
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

    @_locked
    def add_node(self, node: Node) -> Node:
        # Mutation + its audit row land in ONE commit (see transaction()):
        # no crash window where a node exists the trail can't account for.
        with self.transaction() as conn:
            self._add_node_stmt(conn, node)
            # Provenance hook: record node creation. Inside the transaction an
            # audit failure rolls the insert back — provenance is required.
            self._record_node_audit(
                node.node_id, "create",
                actor=node.confidence_set_by,
                before_node=None, after_node=node,
            )
        return node

    @staticmethod
    def _add_node_stmt(conn: sqlite3.Connection, node: Node) -> None:
        """The bare INSERT for one node — no commit, no audit. Callers own
        the transaction (add_node wraps it; import_graph batches many)."""
        conn.execute(
            """INSERT INTO nodes
               (node_id, node_type, label, content, source_id,
                created_at, last_accessed, access_count, metadata,
                source_type, confidence, pinned, confidence_set_at,
                confidence_set_by, source_context, last_referenced,
                invalidated_at, source_modality, answerable_by_text,
                vision_processed, recorded_at, event_time_start,
                event_time_end, event_time_granularity, event_time_confidence,
                event_time_text, valid_from, valid_until)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                node.valid_from.isoformat() if node.valid_from else None,
                node.valid_until.isoformat() if node.valid_until else None,
            ),
        )

    @_locked
    def get_node(self, node_id: str) -> Optional[Node]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    # SQLite's default parameter limit is 999; chunk IN() queries below it.
    _IN_CHUNK = 500

    @_locked
    def get_nodes_bulk(self, node_ids) -> dict:
        """Fetch many nodes in chunked IN() queries: {node_id: Node}. The
        recall path scores hundreds of walked nodes per query — one SELECT
        per node was a measured latency driver (OPEN 2)."""
        ids = [i for i in node_ids]
        out: dict = {}
        if not ids:
            return out
        conn = self._get_conn()
        for i in range(0, len(ids), self._IN_CHUNK):
            chunk = ids[i:i + self._IN_CHUNK]
            rows = conn.execute(
                f"SELECT * FROM nodes WHERE node_id IN "
                f"({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            for row in rows:
                node = self._row_to_node(row)
                out[node.node_id] = node
        return out

    @_locked
    def get_neighbors_bulk(self, node_ids) -> dict:
        """Neighbor ids for many nodes at once: {node_id: [neighbor_ids]}.
        Lets the BFS walker do one round-trip per LEVEL instead of one per
        visited node. Every requested id gets a key (possibly empty list)."""
        ids = [i for i in node_ids]
        out: dict = {nid: [] for nid in ids}
        if not ids:
            return out
        conn = self._get_conn()
        for i in range(0, len(ids), self._IN_CHUNK):
            chunk = ids[i:i + self._IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT source_node_id, target_node_id FROM edges "
                f"WHERE source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders})",
                chunk + chunk,
            ).fetchall()
            wanted = set(chunk)
            for src, tgt in rows:
                if src in wanted:
                    out[src].append(tgt)
                if tgt in wanted:
                    out[tgt].append(src)
        # De-duplicate while keeping it a plain list per node.
        return {nid: list(dict.fromkeys(neigh)) for nid, neigh in out.items()}

    @_locked
    def get_neighbors_weighted_bulk(
        self, node_ids, use_confidence: bool = False
    ) -> dict:
        """Neighbors WITH edge strength for many nodes at once:
        {node_id: [(neighbor_id, strength), ...]}.

        strength is the edge weight (0-1), optionally multiplied by the
        edge-level confidence when use_confidence=True. Parallel edges between
        the same pair keep the MAX strength — the strongest relationship is
        the one the walk should credit. Same one-round-trip-per-BFS-level
        shape as get_neighbors_bulk; the extra columns ride the same rows.
        """
        ids = [i for i in node_ids]
        out: dict = {nid: {} for nid in ids}
        if not ids:
            return {}
        conn = self._get_conn()
        for i in range(0, len(ids), self._IN_CHUNK):
            chunk = ids[i:i + self._IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT source_node_id, target_node_id, weight, confidence "
                f"FROM edges "
                f"WHERE source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders})",
                chunk + chunk,
            ).fetchall()
            wanted = set(chunk)
            for src, tgt, weight, confidence in rows:
                strength = weight if weight is not None else 0.5
                if use_confidence:
                    strength *= confidence if confidence is not None else 0.5
                if src in wanted:
                    prev = out[src].get(tgt)
                    if prev is None or strength > prev:
                        out[src][tgt] = strength
                if tgt in wanted:
                    prev = out[tgt].get(src)
                    if prev is None or strength > prev:
                        out[tgt][src] = strength
        return {nid: list(neigh.items()) for nid, neigh in out.items()}

    # ── Distill manifest (editable-vault leg) ──────────────────────────
    @_locked
    def replace_distill_manifest(self, note_stem: str, rows: list) -> None:
        """Replace all manifest rows for a note (full regen at distill time).
        Each row is a dict: entity_id, anchor_node_id, current_node_id,
        content_hash, section."""
        conn = self._get_conn()
        conn.execute("DELETE FROM distill_manifest WHERE note_stem = ?", (note_stem,))
        for r in rows:
            conn.execute(
                """INSERT INTO distill_manifest
                   (note_stem, entity_id, anchor_node_id, current_node_id,
                    content_hash, section)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (note_stem, r["entity_id"], r["anchor_node_id"],
                 r["current_node_id"], r["content_hash"], r.get("section", "")),
            )
        self._commit(conn)

    @_locked
    def get_distill_manifest(self, note_stem: str) -> list:
        """All manifest rows for one note, as dicts."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT entity_id, anchor_node_id, current_node_id, content_hash, section
               FROM distill_manifest WHERE note_stem = ?""",
            (note_stem,),
        ).fetchall()
        return [
            {"entity_id": r[0], "anchor_node_id": r[1], "current_node_id": r[2],
             "content_hash": r[3], "section": r[4]}
            for r in rows
        ]

    @_locked
    def list_manifest_note_stems(self) -> list:
        """Every note stem that has a manifest (the reconcilable notes)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT note_stem FROM distill_manifest"
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def update_manifest_row(
        self, note_stem: str, anchor_node_id: str,
        current_node_id: str, content_hash: str,
    ) -> None:
        """After absorbing an edit: point the anchor at its correction and
        record the new text hash, so the next reconcile sees no phantom diff.
        The anchor_node_id (file join key) is never changed."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE distill_manifest SET current_node_id = ?, content_hash = ?
               WHERE note_stem = ? AND anchor_node_id = ?""",
            (current_node_id, content_hash, note_stem, anchor_node_id),
        )
        self._commit(conn)

    @_locked
    def add_manifest_row(
        self, note_stem: str, entity_id: str, anchor_node_id: str,
        current_node_id: str, content_hash: str, section: str = "",
    ) -> None:
        """Record a user-added claim so a later re-add of the same line is a
        no-op, and it round-trips like any other line."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO distill_manifest
               (note_stem, entity_id, anchor_node_id, current_node_id,
                content_hash, section)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (note_stem, entity_id, anchor_node_id, current_node_id, content_hash, section),
        )
        self._commit(conn)

    @_locked
    def delete_manifest_row(self, note_stem: str, anchor_node_id: str) -> None:
        """After absorbing a deletion: drop the row so the forgotten claim is
        no longer expected in the note."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM distill_manifest WHERE note_stem = ? AND anchor_node_id = ?",
            (note_stem, anchor_node_id),
        )
        self._commit(conn)

    @_locked
    def manifest_refs_elsewhere(self, node_id: str, exclude_stem: str) -> bool:
        """True if this node is rendered into a DIFFERENT note's manifest — so
        deleting its line from one note must not soft-invalidate it globally."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM distill_manifest WHERE current_node_id = ? "
            "AND note_stem != ? LIMIT 1",
            (node_id, exclude_stem),
        ).fetchone()
        return row is not None

    @_locked
    def search_nodes_keyword(
        self, keywords, limit: int = 10, exclude_context: bool = True
    ) -> list["Node"]:
        """SQL-side keyword search over label+content (case-insensitive
        substring, OR across keywords), newest first. Replaces the recall
        fallback's list_nodes(limit=999999)-then-scan-in-Python, which was
        the single biggest recall latency driver (OPEN 2). Semantics match
        the old Python scan: any-keyword substring hit qualifies."""
        kws = [k for k in keywords if k]
        if not kws:
            return []
        conn = self._get_conn()
        conditions = " OR ".join(
            ["instr(lower(label || ' ' || content), ?) > 0"] * len(kws)
        )
        query = f"SELECT * FROM nodes WHERE ({conditions})"
        if exclude_context:
            query += " AND node_type != 'context'"
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = conn.execute(query, [k.lower() for k in kws] + [limit]).fetchall()
        return [self._row_to_node(r) for r in rows]

    @_locked
    def update_node(
        self, node_id: str, _audit_op: Optional[str] = "update",
        _audit_actor: str = "", **kwargs
    ) -> Optional[Node]:
        """Update specific fields on a node. Returns updated node or None.

        Provenance (leg 6a): records an audit entry (default op ``update``)
        capturing before/after snapshots, in the SAME commit as the UPDATE —
        an audit failure rolls the mutation back. Callers in the operations/
        engine layer that record their own semantic op (reinforce/correct/
        decay/invalidate/access) pass ``_audit_op=None`` to suppress this
        generic entry and avoid double-logging.
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

        # Mutation + audit in ONE commit; audit failure rolls the UPDATE back.
        with self.transaction():
            conn.execute(
                f"UPDATE nodes SET {', '.join(set_clauses)} WHERE node_id = ?",
                values,
            )
            updated = self.get_node(node_id)
            # Provenance hook: generic update entry unless a semantic caller
            # (reinforce/correct/decay/invalidate/access) suppresses it.
            if _audit_op:
                self._record_node_audit(
                    node_id, _audit_op, actor=_audit_actor,
                    before_node=existing, after_node=updated,
                )
        # Content listeners: fire only when the embedded text actually changed,
        # so metadata/access-count updates (the hot path) never trigger a
        # re-embed.
        if updated is not None and (
            ("label" in updates and updated.label != existing.label)
            or ("content" in updates and updated.content != existing.content)
        ):
            self._fire_content_change(updated)
        return updated

    @_locked
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges. Returns True if deleted."""
        with self.transaction() as conn:
            # Edges cascade via FK, but be explicit for safety
            conn.execute(
                "DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
                (node_id, node_id),
            )
            cursor = conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
            deleted = cursor.rowcount > 0
        if deleted:
            self._fire_delete(node_id)
        return deleted

    @_locked
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

    @_locked
    def count_nodes(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    # ── Supersession candidate queue (CSL Leg 3 wiring) ───────────────────────
    @_locked
    def add_candidate(
        self, existing_node_id: str, new_node_id: str, action: str,
        reason: str = "", trace: str = "",
    ) -> int:
        """Park a contradiction the gate routed to review. Mutates no node."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO supersession_candidates
               (existing_node_id, new_node_id, action, reason, trace, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (existing_node_id, new_node_id, action, reason, trace, now),
        )
        self._commit(conn)
        return cur.lastrowid

    @_locked
    def list_candidates(self, unresolved_only: bool = True) -> list[dict]:
        conn = self._get_conn()
        q = ("SELECT id, existing_node_id, new_node_id, action, reason, trace, "
             "created_at, resolved_at, resolution FROM supersession_candidates")
        if unresolved_only:
            q += " WHERE resolved_at IS NULL"
        q += " ORDER BY id ASC"
        cols = ["id", "existing_node_id", "new_node_id", "action", "reason",
                "trace", "created_at", "resolved_at", "resolution"]
        return [dict(zip(cols, r)) for r in conn.execute(q).fetchall()]

    @_locked
    def resolve_candidate(self, candidate_id: int, resolution: str) -> None:
        """Mark a queued candidate adjudicated (resolution is free-text/audit)."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE supersession_candidates SET resolved_at = ?, resolution = ? WHERE id = ?",
            (now, resolution, candidate_id),
        )
        self._commit(conn)

    @_locked
    def count_candidates(self, unresolved_only: bool = True) -> int:
        conn = self._get_conn()
        q = "SELECT COUNT(*) FROM supersession_candidates"
        if unresolved_only:
            q += " WHERE resolved_at IS NULL"
        return conn.execute(q).fetchone()[0]

    @_locked
    def set_node_validity(
        self, node_id: str, valid_from=None, valid_until=None,
        actor: str = "",
    ) -> bool:
        """Set a node's bi-temporal validity bounds (B2). Partial update:
        only non-None arguments are written, and an ALREADY-SET bound is
        never silently overwritten (a recorded transition is a fact — moving
        it needs a deliberate audit story, not a second supersession racing
        the first). Audited. Returns False for unknown node or no-op."""
        node = self.get_node(node_id)
        if node is None:
            return False
        updates, params = [], []
        if valid_from is not None and node.valid_from is None:
            updates.append("valid_from = ?")
            params.append(valid_from.isoformat())
        if valid_until is not None and node.valid_until is None:
            updates.append("valid_until = ?")
            params.append(valid_until.isoformat())
        if not updates:
            return False
        # Mutation + audit in ONE commit; audit failure rolls the UPDATE back.
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE nodes SET {', '.join(updates)} WHERE node_id = ?",
                params + [node_id],
            )
            after = self.get_node(node_id)
            self._record_node_audit(
                node_id, "validity", actor=actor,
                before_node=node, after_node=after,
            )
        return True

    @_locked
    def list_orphan_node_ids(self) -> list[str]:
        """Node ids with NO edges in either direction (B3 dream pass).

        One set-difference query instead of a per-node edge lookup — the
        consolidation sweep runs this over the whole graph."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT node_id FROM nodes WHERE node_id NOT IN "
            "(SELECT source_node_id FROM edges) "
            "AND node_id NOT IN (SELECT target_node_id FROM edges)"
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def list_tension_pairs(self, live_only: bool = True) -> list[dict]:
        """The tensions view (B1): every CONFLICTS_WITH pair with both claims.

        live_only (default) keeps pairs where BOTH claims are un-invalidated —
        a superseded claim's tension is history, not a live pull. Pass False
        to include those for lineage/audit reading.
        """
        conn = self._get_conn()
        q = (
            "SELECT e.edge_id, e.source_context, e.created_at, e.weight, "
            "       e.confidence_set_by, "
            "       a.node_id, a.label, a.content, a.invalidated_at, "
            "       b.node_id, b.label, b.content, b.invalidated_at "
            "FROM edges e "
            "JOIN nodes a ON a.node_id = e.source_node_id "
            "JOIN nodes b ON b.node_id = e.target_node_id "
            "WHERE e.edge_type = 'conflicts_with' "
            "ORDER BY e.created_at ASC"
        )
        out = []
        for r in conn.execute(q).fetchall():
            if live_only and (r[8] is not None or r[12] is not None):
                continue
            out.append({
                "edge_id": r[0],
                "source_context": r[1],
                "created_at": r[2],
                "weight": r[3],
                "set_by": r[4],
                "a": {"node_id": r[5], "label": r[6], "content": r[7],
                      "invalidated_at": r[8]},
                "b": {"node_id": r[9], "label": r[10], "content": r[11],
                      "invalidated_at": r[12]},
            })
        return out

    # ── Edge CRUD ─────────────────────────────────────────

    @_locked
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
        self._commit(conn)
        return edge

    @_locked
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM edges WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    @_locked
    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node (as source or target)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM edges WHERE source_node_id = ? OR target_node_id = ?",
            (node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @_locked
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

    @_locked
    def update_edge_weight(self, edge_id: str, weight: float) -> bool:
        """Update an edge's weight. Clamps to [0.0, 1.0]. Returns True if updated."""
        weight = max(0.0, min(1.0, weight))
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE edges SET weight = ? WHERE edge_id = ?",
            (weight, edge_id),
        )
        self._commit(conn)
        return cursor.rowcount > 0

    @_locked
    def delete_edge(self, edge_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM edges WHERE edge_id = ?", (edge_id,))
        self._commit(conn)
        return cursor.rowcount > 0

    @_locked
    def count_edges(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # ── Graph Export/Import ───────────────────────────────

    @_locked
    def export_graph(self) -> Graph:
        nodes = self.list_nodes(limit=999999)
        conn = self._get_conn()
        edge_rows = conn.execute("SELECT * FROM edges").fetchall()
        edges = [self._row_to_edge(r) for r in edge_rows]
        return Graph(nodes=nodes, edges=edges)

    @_locked
    def import_graph(
        self,
        graph: Graph,
        mode: str = "refuse",
        clear_existing: Optional[bool] = None,
        semantic=None,
    ) -> dict:
        """Import a graph — prevalidated, transactional, honest about outcome.

        mode:
          * "refuse" (default) — a non-empty database raises ImportRefusedError
            untouched. An import must never silently eat an existing graph.
          * "merge" — records whose IDs already exist are skipped, NEVER
            mutated; only valid new nodes/edges are added.
          * "replace" — delete + insert in ONE transaction. Any failure
            mid-import rolls back to the original graph intact (the old shape
            committed the wipe FIRST, so a malformed edge left the db erased
            and half-replaced).

        Prevalidation runs before any write: duplicate IDs inside the payload
        and edges referencing nodes absent from (payload ∪ existing-on-merge)
        raise ImportValidationError with the db untouched.

        Audits for imported nodes ride the same single transaction as the
        rows themselves.

        semantic: optional SemanticIndex to bring in sync after the commit —
        stale vectors dropped (replace) and imported nodes QUEUED for embed
        (drained at the next search / idle sweep; no model inference runs
        under the store lock). Reported in the returned dict as
        ``semantic_rebuild`` ("queued (N nodes pending embed)", or "skipped
        (disabled)" when no enabled index is available — honestly, never
        silently).

        clear_existing is DEPRECATED: True maps to mode="replace", False to
        mode="merge". New callers pass mode.
        """
        if clear_existing is not None:
            warnings.warn(
                "import_graph(clear_existing=...) is deprecated; use "
                "mode='replace' (was clear_existing=True) or mode='merge' "
                "(was clear_existing=False).",
                DeprecationWarning,
                stacklevel=3,  # past the _locked wrapper, to the caller
            )
            mode = "replace" if clear_existing else "merge"
        if mode not in ("refuse", "merge", "replace"):
            raise ValueError(
                f"import_graph mode must be 'refuse', 'merge' or 'replace', got {mode!r}"
            )

        conn = self._get_conn()

        # ── Prevalidate: any violation = error, db untouched ──
        payload_node_ids = [n.node_id for n in graph.nodes]
        payload_node_set = set(payload_node_ids)
        if len(payload_node_set) != len(payload_node_ids):
            dupes = sorted({i for i in payload_node_ids if payload_node_ids.count(i) > 1})
            raise ImportValidationError(
                f"duplicate node IDs in import payload: {dupes[:5]}"
            )
        payload_edge_ids = [e.edge_id for e in graph.edges]
        if len(set(payload_edge_ids)) != len(payload_edge_ids):
            dupes = sorted({i for i in payload_edge_ids if payload_edge_ids.count(i) > 1})
            raise ImportValidationError(
                f"duplicate edge IDs in import payload: {dupes[:5]}"
            )

        # Edge endpoints must resolve: payload nodes always count; on merge,
        # nodes already in the database count too (an imported edge may attach
        # to existing memory).
        known = set(payload_node_set)
        if mode == "merge":
            known |= self._existing_ids("nodes", "node_id")
        dangling = sorted(
            {ep for e in graph.edges
             for ep in (e.source_node_id, e.target_node_id)
             if ep not in known}
        )
        if dangling:
            raise ImportValidationError(
                f"edges reference unknown node IDs: {dangling[:5]}"
            )

        existing_nodes = self.count_nodes()
        existing_edges = self.count_edges()
        if mode == "refuse" and (existing_nodes or existing_edges):
            raise ImportRefusedError(
                f"database is not empty ({existing_nodes} nodes, "
                f"{existing_edges} edges); use mode='merge' or mode='replace'"
            )

        result = {
            "nodes_added": 0, "edges_added": 0,
            "skipped_nodes": 0, "skipped_edges": 0,
            "removed_nodes": 0, "removed_edges": 0,
        }
        removed_node_ids: list = []

        if mode == "merge":
            existing_node_ids = self._existing_ids("nodes", "node_id")
            existing_edge_ids = self._existing_ids("edges", "edge_id")
            new_nodes = [n for n in graph.nodes if n.node_id not in existing_node_ids]
            new_edges = [e for e in graph.edges if e.edge_id not in existing_edge_ids]
            result["skipped_nodes"] = len(graph.nodes) - len(new_nodes)
            result["skipped_edges"] = len(graph.edges) - len(new_edges)
        else:
            new_nodes, new_edges = list(graph.nodes), list(graph.edges)

        # ── One transaction: wipe (replace only) + inserts + their audits ──
        with self.transaction():
            if mode == "replace":
                removed_node_ids = [
                    r[0] for r in conn.execute("SELECT node_id FROM nodes").fetchall()
                ]
                result["removed_nodes"] = existing_nodes
                result["removed_edges"] = existing_edges
                # Direct SQL, not delete_node: the wipe and the inserts must
                # be one atomic unit, and per-node listener churn is pointless
                # when the semantic index is rebuilt wholesale below.
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
            for node in new_nodes:
                self.add_node(node)  # rides this transaction; audit included
                result["nodes_added"] += 1
            for edge in new_edges:
                self.add_edge(edge)  # rides this transaction
                result["edges_added"] += 1

        # ── Semantic index, post-commit ──
        result["semantic_rebuild"] = self._semantic_after_import(
            semantic, mode, new_nodes, removed_node_ids,
        )
        return result

    def _existing_ids(self, table: str, column: str) -> set:
        """All ids currently in a table. Import-scale helper (one full read)."""
        conn = self._get_conn()
        return {r[0] for r in conn.execute(f"SELECT {column} FROM {table}").fetchall()}

    def _semantic_after_import(
        self, semantic, mode: str, new_nodes: list, removed_node_ids: list,
    ) -> str:
        """Queue the optional semantic index to catch up with what landed.

        Cheap SQL only — NO embedding runs here. import_graph holds the store
        lock, and model inference under it (a cold fastembed load hangs for
        minutes) would stall every store consumer — and freeze the daemon's
        event loop on the async import endpoint. replace: drop the wiped
        nodes' vectors, queue every imported node; merge/refuse: queue the
        new nodes. The pending queue drains at the next search or the idle
        sweep — model warm, lock released. No enabled index -> reported as
        skipped; a layer that broke mid-operation -> reported as failed.
        Never silent.
        """
        if semantic is None or not getattr(semantic, "is_enabled", False):
            return "skipped (disabled)"
        if mode == "replace":
            for nid in removed_node_ids:
                semantic.remove_node(nid)
        queued = semantic.defer_nodes(
            [(n.node_id, n.label, n.content) for n in new_nodes]
        )
        if not semantic.is_enabled:
            # remove_node/defer_nodes self-disabled the layer mid-operation —
            # say so; "queued 0" would read as a clean no-op.
            reason = getattr(semantic, "_broken_reason", None) or "unknown"
            return f"failed (semantic layer self-disabled: {reason})"
        return f"queued ({queued} nodes pending embed)"

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
            # Bi-temporal validity (B2, columns 26-27): nullable — a pre-B2
            # row reads back with an unbounded window.
            valid_from=datetime.fromisoformat(row[26]) if len(row) > 26 and row[26] else None,
            valid_until=datetime.fromisoformat(row[27]) if len(row) > 27 and row[27] else None,
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
