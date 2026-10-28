"""
Revien Graph Store — SQLite-backed persistent graph storage.
Every node, every edge, every relationship — stored, never compacted.
"""

import functools
import json
import sqlite3
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from revien import extensions as _extensions

from .forget_store import ForgetStoreMixin
from .origin import derive_origin, derive_recorded_at_source
from .schema import (
    LABEL_CUT_KEY, Edge, EdgeType, Graph, Modality, Node, NodeType, SourceType,
    TemporalGranularity,
)


def _next_day(day: str) -> str:
    """'YYYY-MM-DD' -> the following calendar day, same format."""
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


class StoreBusy(sqlite3.OperationalError):
    """Another connection held the write lock for the whole of a bounded
    wait (see GraphStore.transaction)."""


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


class GraphStore(ForgetStoreMixin):
    """SQLite-backed graph store. Thread-safe: one shared connection
    (check_same_thread=False) serialized by a reentrant lock."""

    def __init__(self, db_path: str = "revien.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # Terminal-close flag: close() sets it and _get_conn refuses to
        # reopen. Without it a post-close call silently resurrected the
        # connection and re-locked the db file — the unlink guarantee has to
        # be structural, not "hope nobody calls us after close".
        self._closed = False
        # Serializes ALL access to the shared connection (see _locked).
        self._lock = threading.RLock()
        # Held by a capture's write section and by a forget, so a forget and
        # an ingest of the same unit never interleave inside one process.
        self._capture_lock = threading.RLock()
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
        # Extension types (revien/extensions.py) must be registered before
        # any row is read, so every surface that opens a store gets them. A
        # configured extension that fails raises here, before a connection
        # is opened.
        _extensions.load_extensions()
        # A database holding a type nobody registered is refused before this
        # store has a connection: no WAL switch, DDL or migration.
        self._refuse_unknown_types()
        self._ensure_db()

    def register_content_listener(
        self, name: str, on_content_change=None, on_delete=None,
        on_successor_change=None,
    ) -> None:
        """Register callbacks fired after a node's label/content changes or a
        node is deleted. on_content_change(node_id, label, content);
        on_delete(node_id). Re-registering the same name replaces the prior
        callbacks. This is how the semantic index keeps embeddings in sync
        with node edits (stale-vector bug: edits used to require a manual
        /v1/reindex before vector search saw the new content — now they queue
        a re-embed drained by the next search). Listeners fire while the
        store lock is held and MUST stay cheap: queue writes and small SQL,
        never model inference.

        on_successor_change(node_id, label, content), optional: fired for the
        NEXT CONTEXT node in the same session whenever a CONTEXT node's
        content changes or the node is deleted. A layer that embeds a turn
        together with its predecessor (REVIEN_EMBED_CONTEXT=prev) needs this:
        the successor's stored vector still carries the old/deleted text."""
        self._content_listeners[name] = (
            on_content_change, on_delete, on_successor_change)

    def _fire_content_change(self, node: "Node") -> None:
        # Inside a transaction: defer until after the outer commit. Listeners
        # (the semantic index) write and commit on the SAME connection — an
        # inline fire would flush a half-done transaction.
        if self._txn_depth > 0:
            self._deferred_fires.append(("change", node))
            return
        for on_change, _d, _s in list(self._content_listeners.values()):
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
        for _c, on_delete, _s in list(self._content_listeners.values()):
            if on_delete is None:
                continue
            try:
                on_delete(node_id)
            except Exception:  # noqa: BLE001 - listeners must never break writes
                pass

    def _fire_successor_change(self, succ: "Node") -> None:
        if self._txn_depth > 0:
            self._deferred_fires.append(("successor", succ))
            return
        for _c, _d, on_succ in list(self._content_listeners.values()):
            if on_succ is None:
                continue
            try:
                on_succ(succ.node_id, succ.label, succ.content)
            except Exception:  # noqa: BLE001 - listeners must never break writes
                pass

    def _flush_deferred_fires(self) -> None:
        """Fire listener callbacks queued during a transaction (post-commit)."""
        pending, self._deferred_fires = self._deferred_fires, []
        for kind, payload in pending:
            if kind == "change":
                self._fire_content_change(payload)
            elif kind == "successor":
                self._fire_successor_change(payload)
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
                valid_until TEXT,
                origin_runtime TEXT,
                origin_source TEXT,
                project_key TEXT,
                session_key TEXT
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
                invalidated_at TEXT,
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

            -- Adapter-sync cursors (repair leg R3): the persisted per-adapter
            -- sync position. cursor is the ISO-8601 instant captured BEFORE
            -- the adapter's fetch ran (stamp-before-fetch), written only after
            -- a fully successful sync. No row = never synced: the scheduler
            -- starts that adapter at EPOCH so pre-daemon history is ingested.
            -- Memory-only cursors reset to now() on every daemon restart and
            -- silently skipped everything from the offline window.
            CREATE TABLE IF NOT EXISTS sync_cursors (
                adapter_name TEXT PRIMARY KEY,
                cursor TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
            CREATE INDEX IF NOT EXISTS idx_audit_node ON audit_log(node_id);
            CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
            CREATE INDEX IF NOT EXISTS idx_candidates_unresolved
                ON supersession_candidates(resolved_at);
        """)
        # idx_nodes_confidence / idx_edges_confidence are NOT created here
        # either: a database from before the confidence layer (v0.2.x) has no
        # confidence column, and CREATE INDEX on it failed the whole open
        # ("no such column: confidence"). _migrate_add_confidence_columns
        # creates both after the ALTERs, for fresh and legacy databases alike.
        # idx_nodes_origin_* are NOT created here: on a legacy DB whose
        # "nodes" table predates the origin columns, CREATE TABLE IF NOT
        # EXISTS is a no-op (the table already exists) but CREATE INDEX ...
        # ON nodes(origin_runtime) would still fail — no such column — since
        # the ALTER hasn't run yet. Same reason idx_nodes_modality lives only
        # in its own migration guard. Created in _migrate_add_origin_columns
        # below, after the ALTERs, for both fresh and legacy databases.
        conn.execute("PRAGMA foreign_keys = ON")
        self._commit(conn)

        # Upgrade pre-confidence databases in place (backwards compatibility).
        self._migrate_add_confidence_columns(conn)
        # Claim Sovereignty Layer (Leg 1): add modality columns to older DBs.
        self._migrate_add_modality_columns(conn)
        # Claim Sovereignty Layer (Leg 2): add temporal columns to older DBs.
        self._migrate_add_temporal_columns(conn)
        self._migrate_add_validity_columns(conn)
        self._migrate_add_edge_invalidation_column(conn)
        # Origin Layer (WS0): add origin columns to older DBs and backfill
        # from source_id wherever derive_origin recognizes it.
        self._migrate_add_origin_columns(conn)
        # recorded_at_source backfill (user_version 4); needs origin columns.
        self._migrate_backfill_recorded_at_source(conn)
        # Usage layer (user_version 5): ALL of its DDL lives in one guard.
        self._migrate_add_usage_layer(conn)
        # Forget layer (user_version 6): ownership, unit registry,
        # suppression records. Its DDL lives in graph/forget_store.py.
        self._migrate_add_forget_layer(conn)
        # Exact lane indexes (user_version 7): label lookup + event-time range.
        self._migrate_add_exact_lane_indexes(conn)

    _EXACT_LANE_INDEXES = (
        ("idx_nodes_label_lower", "nodes(lower(label))"),
        ("idx_nodes_event_time", "nodes(event_time_start, event_time_end)"),
        ("idx_nodes_recorded_at", "nodes(recorded_at)"),
    )

    def _migrate_add_exact_lane_indexes(self, conn: sqlite3.Connection) -> None:
        """Migration 007: three indexes for the opt-in exact lane (REVIEN_EXACT).

        ``idx_nodes_label_lower`` serves case-insensitive exact label lookup
        (``WHERE lower(label) IN (...)``); ``idx_nodes_event_time`` serves the
        event-date range query; ``idx_nodes_recorded_at`` serves the
        said-at (recorded_at) range query. Indexes only: no row is read or rewritten.
        Idempotent and cheap on every connect: once ``PRAGMA user_version``
        is >= 7 AND all three indexes exist it returns after one catalogue
        read (a marker never overrides the schema: an index dropped
        out-of-band is recreated). Upgrade-safe from v6 and older: the
        temporal columns it indexes are added by the guards that run before
        this one."""
        try:
            have = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name IN (?,?,?)",
                [n for n, _ in self._EXACT_LANE_INDEXES])}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= 7 and len(have) == len(self._EXACT_LANE_INDEXES):
                return
            for name, target in self._EXACT_LANE_INDEXES:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {target}")
            if version < 7:
                conn.execute("PRAGMA user_version = 7")
            self._commit(conn)
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    @staticmethod
    def _distinct_values(conn: sqlite3.Connection, table: str, column: str) -> set:
        """Every distinct value of an indexed column, in about one index
        probe per value instead of a scan of every row (a "loose index scan":
        the next value is the smallest one above the last)."""
        rows = conn.execute(
            f"WITH RECURSIVE d(v) AS (SELECT MIN({column}) FROM {table} "
            f"UNION ALL SELECT (SELECT MIN({column}) FROM {table} "
            f"WHERE {column} > d.v) FROM d WHERE d.v IS NOT NULL) "
            "SELECT v FROM d WHERE v IS NOT NULL")
        return {r[0] for r in rows}

    def _refuse_unknown_types(self) -> None:
        """Run ``_check_stored_types`` on a brief connection of its own,
        opened with no pragma and closed before this store opens its
        connection. That is what makes the refusal precede everything the
        store's own open does: the switch to WAL (which rewrites the header of a
        database that is not in WAL yet), the DDL, and every migration.
        A file that does not exist yet has nothing to check."""
        if not Path(self.db_path).is_file():
            return
        probe = sqlite3.connect(self.db_path)
        try:
            self._check_stored_types(probe)
        finally:
            probe.close()

    def _check_stored_types(self, conn: sqlite3.Connection) -> None:
        """Refuse to open a database whose node or edge types are not all
        known: built in, or registered by a loaded extension.

        The alternatives are worse. Reading such a row as some other type
        changes what it means; passing it through as an "unknown" type lets
        recall, dedup, supersession and forget act on a row nobody can
        classify; skipping it makes memory vanish without a word. So the open
        fails with ``UnknownTypeError``, naming each type, how many rows hold
        it and what to do, having read nothing but the two type columns. The
        check probes the two type indexes once per distinct type; a database
        that does not have the tables yet has nothing to check.
        """
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('nodes', 'edges')")}
        unknown = {"nodes": {}, "edges": {}}
        for table, column, known in (
                ("nodes", "node_type", set(_extensions.all_node_type_values())),
                ("edges", "edge_type", set(_extensions.all_edge_type_values()))):
            if table not in have:
                continue
            for value in self._distinct_values(conn, table, column) - known:
                unknown[table][value] = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",
                    (value,)).fetchone()[0]
        if unknown["nodes"] or unknown["edges"]:
            _extensions.check_known_types(
                unknown["nodes"], unknown["edges"],
                where=f"the database {self.db_path}")

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

    def _migrate_add_edge_invalidation_column(self, conn: sqlite3.Connection) -> None:
        """Add the edge soft-invalidation column (alias leg) to existing DBs.

        Idempotent and no-op for fresh databases. Nullable, backfills NULL on
        every existing row (every old edge is live), so edge reads and the
        graph walk stay byte-identical until something explicitly reverses an
        edge (today: `revien aliases --remove`)."""
        try:
            cursor = conn.execute("PRAGMA table_info(edges)")
            columns = {row[1] for row in cursor.fetchall()}
            if "invalidated_at" not in columns:
                conn.execute("ALTER TABLE edges ADD COLUMN invalidated_at TEXT")
            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column already there).
            # Anything else — locked db, disk, corruption — must be loud at
            # startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    def _migrate_backfill_recorded_at_source(self, conn: sqlite3.Connection) -> None:
        """Label pre-existing dated rows with metadata["recorded_at_source"]
        derived from their origin (derive_recorded_at_source). Rows that derive
        None stay unlabeled (unknown, rendered undated). One transaction;
        idempotent (only rows lacking the key are touched); skipped once
        ``PRAGMA user_version`` >= 4."""
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 4:
            return
        rows = conn.execute(
            "SELECT node_id, metadata, origin_runtime, origin_source FROM nodes "
            "WHERE recorded_at IS NOT NULL"
        ).fetchall()
        for node_id, raw, runtime, source in rows:
            try:
                meta = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                continue
            if not isinstance(meta, dict) or "recorded_at_source" in meta:
                continue
            derived = derive_recorded_at_source(runtime, source)
            if derived is None:
                continue
            meta["recorded_at_source"] = derived
            conn.execute("UPDATE nodes SET metadata = ? WHERE node_id = ?",
                         (json.dumps(meta), node_id))
        conn.execute("PRAGMA user_version = 4")
        self._commit(conn)

    # Migration 005 — usage layer. This is the ONLY place the usage layer's
    # DDL lives; _ensure_db does not create these objects inline. Fresh and
    # existing databases take the same path.
    _USAGE_LAYER_DDL = """
        -- Usage ledger (N1, metacognition tier 1): one row per judgment of
        -- whether an answer actually drew on a memory it was handed by
        -- recall. episode_key = "{session source_id}#{tool_use_id}". The
        -- UNIQUE pair makes a rescan of a grown session a no-op, and the
        -- INSERT OR IGNORE claim is what decides who may apply, so two
        -- processes can never both count one use. Abstentions are not
        -- judgments and are never written here. applied = 1 only when
        -- mark_used ran (mode "apply"). shared_terms holds the memory's own
        -- words, so delete_node removes a node's rows.
        CREATE TABLE IF NOT EXISTS use_judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_key TEXT NOT NULL,
            node_id TEXT NOT NULL,
            used INTEGER NOT NULL,
            reason TEXT DEFAULT '',
            shared_terms TEXT DEFAULT '[]',
            similarity REAL,
            query TEXT DEFAULT '',
            mode TEXT NOT NULL,
            applied INTEGER NOT NULL DEFAULT 0,
            judged_at TEXT NOT NULL,
            UNIQUE (episode_key, node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_use_judgments_node
            ON use_judgments(node_id);

        -- Recall receipts (N1, any runtime): a recall made with a receipt
        -- gets a random recall_id the client hands back with its answer
        -- (POST /v1/usage/observe). Ids only: node_ids is a JSON list, no
        -- memory content; query is kept so words the client merely repeated
        -- can be discounted. status: open -> redeemed, or open -> expired
        -- after 24h unjudged. An expired row IS the flag: it stays until a
        -- person dismisses it, so no recall vanishes unseen.
        CREATE TABLE IF NOT EXISTS recall_receipts (
            recall_id TEXT PRIMARY KEY,
            runtime TEXT NOT NULL DEFAULT '',
            query TEXT NOT NULL DEFAULT '',
            node_ids TEXT NOT NULL DEFAULT '[]',
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            closed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_recall_receipts_status
            ON recall_receipts(status, expires_at);

        -- Per-database key/value facts. Today: the running daemons' heartbeats
        -- (daemon_heartbeat:<pid> -> features, pid, time). No path, no
        -- content.
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """

    def _migrate_add_usage_layer(self, conn: sqlite3.Connection) -> None:
        """Migration 005: usage ledger, recall receipts, store_meta and the
        ``sync_cursors.generation`` fence column.

        Idempotent and cheap on every connect: once ``PRAGMA user_version``
        is >= 5 AND the schema in front of us really has every object, it
        returns after a handful of catalogue reads. A marker never overrides
        the schema (a table dropped out-of-band is recreated). Existing rows
        are untouched: ``generation`` backfills 0 on every cursor row.
        """
        try:
            have = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('use_judgments','recall_receipts','store_meta',"
                "'idx_use_judgments_node','idx_recall_receipts_status')")}
            cursor_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(sync_cursors)")}
            complete = len(have) == 5 and "generation" in cursor_cols
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= 5 and complete:
                return
            conn.executescript(self._USAGE_LAYER_DDL)
            if "generation" not in cursor_cols:
                conn.execute(
                    "ALTER TABLE sync_cursors ADD COLUMN generation "
                    "INTEGER NOT NULL DEFAULT 0")
            if version < 5:
                conn.execute("PRAGMA user_version = 5")
            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (another opener got there
            # first). Anything else must be loud at startup.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    def _migrate_add_origin_columns(self, conn: sqlite3.Connection) -> None:
        """Add Origin Layer (WS0) columns to existing DBs, then backfill.

        Idempotent and no-op for fresh databases (columns already created in
        ``_ensure_db``). All four columns are nullable TEXT and backfill NULL
        on ALTER for old rows — then the backfill pass below fills in every
        row whose source_id derive_origin recognizes. Runs on every connect
        (same as the other migration guards) UNTIL ``PRAGMA user_version``
        reaches 3: a full ``SELECT ... WHERE origin_runtime IS NULL`` table
        scan on every single connect doesn't scale on a large graph once the
        backfill has already done its job, so once a backfill pass completes
        cleanly, user_version is bumped to 3 (only if it was lower) and every
        later connect skips the scan outright. A DB already at user_version
        >= 3 is trusted — new rows are stamped at ingest time, not by this
        scan.
        """
        try:
            cursor = conn.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}

            columns_added = False
            for col in ("origin_runtime", "origin_source", "project_key", "session_key"):
                if col not in columns:
                    conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} TEXT")
                    columns_added = True

            # G6: composite indexes — not single-column. list_nodes(
            # origin_runtime=..., limit=N) and list_nodes(project_key=...,
            # limit=N) both filter AND ``ORDER BY created_at DESC`` in the
            # same query — a single-column index on origin_runtime/
            # project_key answers the WHERE but leaves SQLite to sort the
            # matching rows itself ("USE TEMP B-TREE FOR ORDER BY" in
            # EXPLAIN QUERY PLAN), which is the expensive part at scale.
            # (origin_runtime, created_at DESC) / (project_key, created_at
            # DESC) let the index satisfy both the filter AND the order in
            # one pass. Old single-column index names are dropped first so
            # an upgraded DB converges on the composite instead of
            # carrying a redundant single-column index alongside it
            # forever. idx_nodes_origin_source stays single-column — it
            # isn't ordered by created_at anywhere in this codebase.
            # created_at is always present: it's a NOT NULL column in this
            # store's CREATE TABLE nodes, so a nodes table missing it isn't
            # a state this store can produce.
            # Drop only when the stored definition is not the composite
            # (an unconditional drop+create ran on every open). Both the old
            # single-column form and the composite share the name, so look
            # at the stored SQL.
            for _name in ("idx_nodes_origin_runtime", "idx_nodes_project"):
                _row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' "
                    "AND name=?", (_name,)).fetchone()
                if _row and "created_at" not in (_row[0] or ""):
                    conn.execute(f"DROP INDEX {_name}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_origin_runtime "
                "ON nodes(origin_runtime, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_origin_source ON nodes(origin_source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_project "
                "ON nodes(project_key, created_at DESC)"
            )
            # previous_context_in_session: session_key equality, then order.
            # recorded_at exists by now (temporal migration runs first).
            # Older DBs carry (session_key, recorded_at); neighbour lookups
            # need created_at in the key too. Drop the narrow one (idempotent).
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name='idx_nodes_session'").fetchone()
            if row and "created_at" not in (row[0] or ""):
                conn.execute("DROP INDEX idx_nodes_session")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_session "
                "ON nodes(session_key, recorded_at, created_at)"
            )
            self._commit(conn)

            user_version = conn.execute("PRAGMA user_version").fetchone()[0]
            # A column that had to be ALTERed back in just now means this
            # connection is genuinely pre-origin-layer no matter what
            # user_version claims (e.g. a column dropped out-of-band after
            # the marker was set) — never trust the marker over the schema
            # actually in front of us.
            if user_version >= 3 and not columns_added:
                return

            # Backfill: any row with a source_id but no derived origin yet.
            # Rows whose source_id derive_origin doesn't recognize stay NULL
            # (honest "unknown", never a guess) — they're rescanned on every
            # connect UNTIL user_version reaches 3 below, which self-corrects
            # if a future adapter convention is added before then.
            rows = conn.execute(
                "SELECT node_id, source_id FROM nodes "
                "WHERE origin_runtime IS NULL AND source_id != ''"
            ).fetchall()
            for node_id, source_id in rows:
                origin = derive_origin(source_id)
                if origin.runtime is None:
                    continue
                conn.execute(
                    "UPDATE nodes SET origin_runtime = ?, origin_source = ?, "
                    "project_key = ?, session_key = ? WHERE node_id = ?",
                    (origin.runtime, origin.source, origin.project, origin.session, node_id),
                )
            conn.execute("PRAGMA user_version = 3")
            self._commit(conn)
        except sqlite3.OperationalError as e:
            # Swallow ONLY the idempotency race (column/index already there).
            # Anything else — locked db, disk, corruption — must be loud at
            # startup, not a "Migration note" that hides it.
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    def _get_conn(self) -> sqlite3.Connection:
        if self._closed:
            # Closed is closed. Reopening here would re-lock the db file
            # after the owner already released it — on Windows the unlink
            # the close existed FOR then fails. Late callers (a straggling
            # scheduler job, an adapter used after its `with` block, an MCP
            # call racing shutdown) get a loud error their own error
            # handling absorbs, not a silent resurrection.
            raise RuntimeError(
                f"GraphStore is closed (db: {self.db_path}). "
                "Construct a new GraphStore to reopen the database."
            )
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Freed pages are overwritten with zeros, so deleted text does
            # not linger in the file (docs/FORGET_SPEC.md §4).
            self._conn.execute("PRAGMA secure_delete = ON")
        return self._conn

    @_locked
    def close(self) -> None:
        """Close the connection and mark the store terminally closed.

        Idempotent (a second close is a no-op) and thread-safe: it takes the
        store lock, so a close racing another thread's method waits for that
        method to finish instead of yanking the connection mid-statement.
        After close, any use raises RuntimeError — see _get_conn.
        """
        self._closed = True
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Transactions ──────────────────────────────────────

    @contextmanager
    def transaction(self, immediate: bool = False, wait: float = 0.0):
        """Everything inside lands in ONE commit; any failure rolls back ALL
        of it. Holds the store lock for the whole span, so no other thread can
        interleave (or commit) mid-transaction on the shared connection.

        ``immediate=True`` opens it with BEGIN IMMEDIATE (see below).
        ``wait``: seconds to keep trying (with backoff) after SQLite's own
        busy timeout has run out, before raising StoreBusy.

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
                if immediate:
                    # Take SQLite's write lock NOW. Two connections (a daemon
                    # and a CLI forget) then take turns for the whole span, so
                    # a check made inside it cannot be overtaken by a write
                    # from the other one.
                    conn.commit()
                    deadline = time.monotonic() + max(0.0, wait)
                    delay = 0.05
                    while True:
                        try:
                            conn.execute("BEGIN IMMEDIATE")
                            break
                        except sqlite3.OperationalError as e:
                            if "locked" not in str(e).lower() \
                                    and "busy" not in str(e).lower():
                                raise
                            if time.monotonic() >= deadline:
                                if wait > 0:
                                    raise StoreBusy(str(e)) from e
                                raise
                            time.sleep(delay)
                            delay = min(delay * 2, 1.0)
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
        before = self._cut_secret_metadata(conn, before)
        after = self._cut_secret_metadata(conn, after)
        actor, = self._cut_secret_strings(conn, actor or "")
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
    def add_node(self, node: Node, owner: Optional[tuple] = None) -> Node:
        """Insert a node (and its audit row) in one commit.

        ``owner`` is ``(unit_id, label_ok, wording_ok)``: the writer's record
        of who wrote this node, stored in the same commit as the node
        (docs/FORGET_SPEC.md §3). A writer with no unit passes nothing and
        the node has no ownership row, i.e. a forget of another node
        never removes it."""
        self._check_validity_window(node.valid_from, node.valid_until)
        # Mutation + its audit row land in ONE commit (see transaction()):
        # no crash window where a node exists the trail can't account for.
        # IMMEDIATE: the denylist is read (in _add_node_stmt) with the write
        # lock already held, so a forget in another process cannot slip in
        # between the read and the insert.
        with self.transaction(immediate=True) as conn:
            self._add_node_stmt(conn, node)
            self._enqueue_embed_stmt(conn, node.node_id)
            if owner is not None:
                unit_id, label_ok, wording_ok = owner
                self._record_provenance_stmt(
                    conn, node.node_id, unit_id, label_ok, wording_ok)
            # Provenance hook: record node creation. Inside the transaction an
            # audit failure rolls the insert back — provenance is required.
            # The create row records who wrote the node: the confidence setter
            # if it named one, else the capture (unit) that vouched for it. A
            # forget uses this to tell a label truncated by the capture that
            # wrote it from a record that merely looks like one.
            self._record_node_audit(
                node.node_id, "create",
                actor=node.confidence_set_by or (
                    f"capture:{owner[0]}" if owner is not None else ""),
                before_node=None, after_node=node,
            )
        return node

    # The deferred-embedding queue is owned by SemanticIndex.PENDING_TABLE
    # (same name, same shape). The store enqueues into it in the SAME
    # transaction as the node write, so every route that creates or edits a
    # node — pipeline, skills, a bare CLI store, a second connection — leaves
    # durable pending index work for whichever process runs the next drain,
    # with or without a live SemanticIndex listener on this instance. Plain
    # SQL, no extension, no embedder.
    _PENDING_EMBED_TABLE = "pending_embeds"

    @classmethod
    def _enqueue_embed_stmt(cls, conn: sqlite3.Connection, node_id: str) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {cls._PENDING_EMBED_TABLE} "
            f"(node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL)"
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {cls._PENDING_EMBED_TABLE} "
            f"(node_id, queued_at) VALUES (?, ?)",
            (node_id, datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def _discard_pending_stmt(cls, conn: sqlite3.Connection, node_id=None) -> None:
        """Drop queue row(s) of a deleted node (all rows when node_id is None);
        a table that was never created means nothing was ever queued."""
        try:
            if node_id is None:
                conn.execute(f"DELETE FROM {cls._PENDING_EMBED_TABLE}")
            else:
                conn.execute(
                    f"DELETE FROM {cls._PENDING_EMBED_TABLE} WHERE node_id = ?",
                    (node_id,))
        except sqlite3.OperationalError:
            pass

    @_locked
    def clear_pending_embeds(self, node_ids) -> None:
        """Drop queue rows for nodes a caller just embedded inline."""
        ids = [(i,) for i in node_ids]
        if not ids:
            return
        conn = self._get_conn()
        try:
            conn.executemany(
                f"DELETE FROM {self._PENDING_EMBED_TABLE} WHERE node_id = ?", ids)
            self._commit(conn)
        except sqlite3.OperationalError:  # table never created: nothing queued
            pass

    @staticmethod
    def _add_node_stmt(conn: sqlite3.Connection, node: Node) -> None:
        """The bare INSERT for one node — no commit, no audit. Callers own
        the transaction (add_node wraps it; import_graph batches many).
        A secret a ``forget --wording`` named is cut out of the node first
        (the node object itself, so its audit row quotes the cut text)."""
        GraphStore._cut_derived(conn, node)
        (node.label, node.content, node.source_context,
         node.event_time_text) = GraphStore._cut_secret_strings(
            conn, node.label, node.content, node.source_context,
            node.event_time_text)
        node.metadata = GraphStore._cut_secret_metadata(conn, node.metadata)
        # identity columns: the same cut as everywhere they are stored
        (node.source_id, node.origin_source, node.session_key,
         node.project_key, node.confidence_set_by) = GraphStore._cut_secret_strings(
            conn, node.source_id, node.origin_source, node.session_key,
            node.project_key, node.confidence_set_by)
        conn.execute(
            """INSERT INTO nodes
               (node_id, node_type, label, content, source_id,
                created_at, last_accessed, access_count, metadata,
                source_type, confidence, pinned, confidence_set_at,
                confidence_set_by, source_context, last_referenced,
                invalidated_at, source_modality, answerable_by_text,
                vision_processed, recorded_at, event_time_start,
                event_time_end, event_time_granularity, event_time_confidence,
                event_time_text, valid_from, valid_until,
                origin_runtime, origin_source, project_key, session_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                node.origin_runtime,
                node.origin_source,
                node.project_key,
                node.session_key,
            ),
        )

    @_locked
    def get_node(self, node_id: str) -> Optional[Node]:
        conn = self._get_conn()
        row = conn.execute(
            f"{self._SELECT_NODES} WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    # SQLite's default parameter limit is 999; chunk IN() queries below it.
    _IN_CHUNK = 500

    def _context_neighbour(self, node: Node, forward: bool) -> Optional[Node]:
        """Shared body of previous/next_context_in_session. The order key is
        (recorded_at, created_at, rowid) with NULL recorded_at sorting first.
        Written as plain column comparisons so SQLite range-scans
        idx_nodes_session (session_key, recorded_at, created_at); a COALESCE
        or a computed ORDER BY would force a scan of the whole session."""
        if not node.session_key:
            return None
        conn = self._get_conn()
        rid = conn.execute(
            "SELECT rowid FROM nodes WHERE node_id = ?", (node.node_id,)
        ).fetchone()
        rid = rid[0] if rid else None
        if rid is None:
            return None
        op, direction = (">", "ASC") if forward else ("<", "DESC")
        base = (f"{self._SELECT_NODES} WHERE session_key = ? AND node_type = ? "
                "AND node_id != ? AND ")
        order = (f" ORDER BY recorded_at {direction}, created_at {direction}, "
                 f"rowid {direction} LIMIT 1")
        head = (node.session_key, NodeType.CONTEXT.value, node.node_id)
        created = node.created_at.isoformat()
        rec = node.recorded_at.isoformat() if node.recorded_at else None
        # Candidate groups in order of proximity. A dated node's neighbour is
        # first among dated rows past it; NULL-dated rows sort before every
        # dated one, so going backward they are the fallback group, going
        # forward from a NULL-dated node the dated rows are.
        queries = []
        if rec is not None:
            queries.append((base + f"(recorded_at, created_at, rowid) {op} (?, ?, ?)"
                            + order, head + (rec, created, rid)))
            if not forward:
                queries.append((base + "recorded_at IS NULL" + order, head))
        else:
            queries.append((base + "recorded_at IS NULL AND "
                            f"(created_at, rowid) {op} (?, ?)" + order,
                            head + (created, rid)))
            if forward:
                queries.append((base + "recorded_at IS NOT NULL" + order, head))
        for sql, args in queries:
            row = conn.execute(sql, args).fetchone()
            if row:
                return self._row_to_node(row)
        return None

    @_locked
    def previous_context_in_session(self, node: Node) -> Optional[Node]:
        """The CONTEXT node immediately before ``node`` in its session: same
        non-null session_key, greatest (recorded_at, created_at) strictly
        before the node's. Ties (a session whose turns share a coarse
        recorded_at AND created_at) fall back to insertion order (rowid).
        None when the node has no session_key or is the session's first."""
        return self._context_neighbour(node, forward=False)

    @_locked
    def next_context_in_session(self, node: Node) -> Optional[Node]:
        """Mirror of previous_context_in_session: the CONTEXT node
        immediately AFTER ``node`` in its session (same ordering key,
        reversed). None when the node has no session_key or is the last."""
        return self._context_neighbour(node, forward=True)

    def _has_successor_listener(self) -> bool:
        return any(v[2] is not None for v in self._content_listeners.values())

    def _notify_successor(self, node: Node) -> None:
        """Tell listeners the CONTEXT node after ``node`` has a stale
        predecessor (node edited, or about to vanish)."""
        if node.node_type != NodeType.CONTEXT or not self._has_successor_listener():
            return
        succ = self.next_context_in_session(node)
        if succ is not None:
            self._fire_successor_change(succ)

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
                f"{self._SELECT_NODES} WHERE node_id IN "
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
        visited node. Every requested id gets a key (possibly empty list).

        Excludes soft-invalidated edges (alias leg): an invalidated edge
        (e.g. a reversed ALIAS_OF) must never route a walk — the row stays
        for audit/history, but "the walk still crosses it" would make
        --remove a lie. Blast radius today is nil (edge invalidation is new
        this leg, nothing pre-existing relied on walking a dead edge); any
        FUTURE invalidation caller inherits this exclusion, which is the
        correct default for every edge type, not just ALIAS_OF.
        """
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
                f"WHERE (source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders})) "
                f"AND invalidated_at IS NULL",
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

        Excludes soft-invalidated edges — see get_neighbors_bulk's docstring;
        same ruling, same rationale, for every edge type.
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
                f"WHERE (source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders})) "
                f"AND invalidated_at IS NULL",
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
        with self.transaction(immediate=True) as conn:
            note_stem, = self._cut_secret_strings(conn, note_stem)
            conn.execute("DELETE FROM distill_manifest WHERE note_stem = ?", (note_stem,))
            for r in rows:
                section, = self._cut_secret_strings(conn, r.get("section", ""))
                conn.execute(
                    """INSERT INTO distill_manifest
                       (note_stem, entity_id, anchor_node_id, current_node_id,
                        content_hash, section)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (note_stem, r["entity_id"], r["anchor_node_id"],
                     r["current_node_id"], r["content_hash"], section),
                )

    @_locked
    def get_distill_manifest(self, note_stem: str) -> list:
        """All manifest rows for one note, as dicts."""
        conn = self._get_conn()
        note_stem, = self._cut_secret_strings(conn, note_stem)
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
        with self.transaction(immediate=True) as conn:
            note_stem, = self._cut_secret_strings(conn, note_stem)
            conn.execute(
                """UPDATE distill_manifest SET current_node_id = ?, content_hash = ?
                   WHERE note_stem = ? AND anchor_node_id = ?""",
                (current_node_id, content_hash, note_stem, anchor_node_id),
            )

    @_locked
    def add_manifest_row(
        self, note_stem: str, entity_id: str, anchor_node_id: str,
        current_node_id: str, content_hash: str, section: str = "",
    ) -> None:
        """Record a user-added claim so a later re-add of the same line is a
        no-op, and it round-trips like any other line."""
        with self.transaction(immediate=True) as conn:
            note_stem, section = self._cut_secret_strings(conn, note_stem, section)
            conn.execute(
                """INSERT OR REPLACE INTO distill_manifest
                   (note_stem, entity_id, anchor_node_id, current_node_id,
                    content_hash, section)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (note_stem, entity_id, anchor_node_id, current_node_id, content_hash, section),
            )

    @_locked
    def delete_manifest_row(self, note_stem: str, anchor_node_id: str) -> None:
        """After absorbing a deletion: drop the row so the forgotten claim is
        no longer expected in the note."""
        conn = self._get_conn()
        note_stem, = self._cut_secret_strings(conn, note_stem)
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

    # ── Adapter-sync cursors (repair leg R3) ───────────────────────────
    @_locked
    def get_sync_cursor(self, adapter_name: str) -> Optional[datetime]:
        """Persisted sync position for one adapter. None = never synced (or
        reset) — the scheduler treats that as EPOCH (full history)."""
        return self.get_sync_state(adapter_name)[0]

    @_locked
    def get_sync_state(self, adapter_name: str) -> tuple:
        """(cursor or None, generation). No row = (None, 0). A reset row
        (blank cursor) = (None, generation). Read this BEFORE fetching and
        pass the generation to ``set_sync_cursor_if_generation``."""
        adapter_name = self._cut_ident(adapter_name)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT cursor, generation FROM sync_cursors WHERE adapter_name = ?",
            (adapter_name,),
        ).fetchone()
        if row is None:
            return (None, 0)
        cursor = datetime.fromisoformat(row[0]) if row[0] else None
        return (cursor, int(row[1] or 0))

    @_locked
    def get_sync_generation(self, adapter_name: str) -> int:
        return self.get_sync_state(adapter_name)[1]

    @_locked
    def set_sync_cursor(self, adapter_name: str, cursor: datetime) -> None:
        """Persist an adapter's sync position (upsert, unfenced; keeps the
        row's generation). For callers that cannot be reset (usage observer).
        The adapter scheduler uses ``set_sync_cursor_if_generation``."""
        with self.transaction(immediate=True):
            adapter_name = self._cut_ident(adapter_name)
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO sync_cursors (adapter_name, cursor, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(adapter_name) DO UPDATE SET
                       cursor = excluded.cursor, updated_at = excluded.updated_at""",
                (adapter_name, cursor.isoformat(),
                 datetime.now(timezone.utc).isoformat()),
            )
            self._commit(conn)

    @_locked
    def set_sync_cursor_if_generation(
        self, adapter_name: str, cursor: datetime, expected_generation: int,
    ) -> bool:
        """Compare-and-set: persist the cursor only if the row's generation
        still equals ``expected_generation`` (the one read before the fetch).
        Returns False, writing nothing, when a reset happened in between.
        Atomic in SQL, so it also fences a reset from another process.
        Absent row is generation 0."""
        with self.transaction(immediate=True):
            adapter_name = self._cut_ident(adapter_name)
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                "UPDATE sync_cursors SET cursor = ?, updated_at = ? "
                "WHERE adapter_name = ? AND generation = ?",
                (cursor.isoformat(), now, adapter_name, expected_generation),
            )
            ok = cur.rowcount > 0
            if not ok and expected_generation == 0:
                # No row yet? INSERT OR IGNORE loses to any row that exists.
                ins = conn.execute(
                    "INSERT OR IGNORE INTO sync_cursors "
                    "(adapter_name, cursor, updated_at, generation) "
                    "VALUES (?, ?, ?, 0)",
                    (adapter_name, cursor.isoformat(), now),
                )
                ok = ins.rowcount > 0
            self._commit(conn)
            return ok

    @_locked
    def list_sync_cursor_names(self) -> List[str]:
        """Names of all adapters with a sync_cursors row (recorded state),
        including rows left blank by a reset."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT adapter_name FROM sync_cursors ORDER BY adapter_name"
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def reset_sync_cursor(self, adapter_name: str) -> Optional[bool]:
        """Acknowledged reset: blank the cursor (next sync starts at EPOCH)
        and BUMP the generation, so any sync already in flight fails its
        compare-and-set and cannot write its older position back. Returns
        None if the adapter has no row (nothing recorded, nothing changed),
        else True if it held a real position, False if already blank."""
        with self.transaction(immediate=True):
            adapter_name = self._cut_ident(adapter_name)
            conn = self._get_conn()
            row = conn.execute(
                "SELECT cursor FROM sync_cursors WHERE adapter_name = ?",
                (adapter_name,),
            ).fetchone()
            if row is None:
                return None
            had_position = bool(row[0])
            conn.execute(
                "UPDATE sync_cursors SET cursor = '', generation = generation + 1, "
                "updated_at = ? WHERE adapter_name = ?",
                (datetime.now(timezone.utc).isoformat(), adapter_name),
            )
            self._commit(conn)
            return had_position

    @_locked
    def delete_sync_cursor(self, adapter_name: str) -> bool:
        """Remove one adapter's sync row outright. Not fenced: a stale writer
        can re-create the row. ``reset_sync_cursor`` is the fenced reset;
        this remains for housekeeping. Returns True if a row existed."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM sync_cursors WHERE adapter_name = ?",
            (adapter_name,),
        )
        deleted = cur.rowcount > 0
        self._commit(conn)
        return deleted

    # ── store_meta (per-database key/value facts) ─────────

    @_locked
    def set_meta(self, key: str, value: str) -> None:
        """Upsert one ``store_meta`` row (table created by migration 005)."""
        with self.transaction(immediate=True) as conn:
            value, = self._cut_secret_strings(conn, value)
            conn.execute(
                "INSERT INTO store_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value))

    @_locked
    def get_meta(self, key: str) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    @_locked
    def delete_meta(self, key: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM store_meta WHERE key = ?", (key,))
        self._commit(conn)
        return cur.rowcount > 0

    @_locked
    def list_meta(self, prefix: str) -> Dict[str, str]:
        """Every ``store_meta`` row whose key starts with ``prefix``."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM store_meta WHERE substr(key, 1, ?) = ?",
            (len(prefix), prefix)).fetchall()
        return {k: v for k, v in rows}

    # ── Usage ledger (N1) ─────────────────────────────────

    @_locked
    def has_use_judgment(self, episode_key: str, node_id: str) -> bool:
        """True when this (episode, node) pair has already been judged."""
        conn = self._get_conn()
        episode_key, = self._cut_secret_strings(conn, episode_key)
        row = conn.execute(
            "SELECT 1 FROM use_judgments WHERE episode_key = ? AND node_id = ?",
            (episode_key, node_id),
        ).fetchone()
        return row is not None

    @_locked
    def record_use_judgment(
        self,
        *,
        episode_key: str,
        node_id: str,
        used: bool,
        reason: str,
        shared_terms: List[str],
        similarity: Optional[float],
        query: str,
        mode: str,
        applied: bool,
    ) -> bool:
        """Claim one judgment: INSERT OR IGNORE on the (episode_key, node_id)
        pair. Returns True only for the caller whose insert landed — only that
        caller may apply it, so a judgment is counted at most once even across
        processes. Rows are never updated."""
        with self.transaction(immediate=True) as conn:
            reason, query, episode_key = self._cut_secret_strings(
                conn, reason, query, episode_key)
            terms = [self._cut_secret_strings(conn, t)[0] for t in shared_terms]
            cursor = conn.execute(
                """INSERT OR IGNORE INTO use_judgments
                   (episode_key, node_id, used, reason, shared_terms, similarity,
                    query, mode, applied, judged_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (episode_key, node_id, int(bool(used)), reason,
                 json.dumps(terms), similarity, query, mode,
                 int(bool(applied)), datetime.now(timezone.utc).isoformat()),
            )
        return cursor.rowcount == 1

    @_locked
    def get_use_judgment(self, episode_key: str, node_id: str) -> Optional[Dict]:
        """The ledger row for one (episode, memory) pair, or None. ``applied``
        is 0 (not applied), 2 (an apply is in progress) or 1 (applied)."""
        conn = self._get_conn()
        episode_key, = self._cut_secret_strings(conn, episode_key)
        r = conn.execute(
            "SELECT used, mode, applied, reason, shared_terms, similarity "
            "FROM use_judgments WHERE episode_key = ? AND node_id = ?",
            (episode_key, node_id)).fetchone()
        if r is None:
            return None
        return {"used": bool(r[0]), "mode": r[1], "applied": int(r[2]),
                "reason": r[3] or "", "shared_terms": json.loads(r[4] or "[]"),
                "similarity": r[5]}

    @_locked
    def claim_use_application(self, episode_key: str, node_id: str) -> bool:
        """Take the right to apply one used judgment: applied 0 -> 2 in a
        single UPDATE, so of any number of concurrent callers exactly one
        gets True."""
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE use_judgments SET applied = 2 WHERE episode_key = ? "
            "AND node_id = ? AND used = 1 AND applied = 0",
            (episode_key, node_id))
        self._commit(conn)
        return cur.rowcount == 1

    @_locked
    def finish_use_application(self, episode_key: str, node_id: str,
                               ok: bool) -> None:
        """After mark_used: 2 -> 1 on success, 2 -> 0 (claimable again) on
        failure."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE use_judgments SET applied = ? WHERE episode_key = ? "
            "AND node_id = ? AND applied = 2",
            (1 if ok else 0, episode_key, node_id))
        self._commit(conn)

    @_locked
    def list_use_judgments(
        self, limit: int = 50, used: Optional[bool] = None
    ) -> List[Dict]:
        """Most recent judgments first. ``used`` filters to used/unused rows."""
        conn = self._get_conn()
        sql = ("SELECT episode_key, node_id, used, reason, shared_terms, "
               "similarity, query, mode, applied, judged_at FROM use_judgments")
        params: list = []
        if used is not None:
            sql += " WHERE used = ?"
            params.append(int(used))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "episode_key": r[0],
                "node_id": r[1],
                "used": bool(r[2]),
                "reason": r[3] or "",
                "shared_terms": json.loads(r[4] or "[]"),
                "similarity": r[5],
                "query": r[6] or "",
                "mode": r[7],
                "applied": r[8] == 1,
                # 2 = an apply was claimed and never finished: mark_used may
                # or may not have run, so it is neither applied nor not.
                "apply_state": {1: "applied", 2: "unknown"}.get(r[8], "not_applied"),
                "judged_at": r[9],
            }
            for r in rows
        ]

    # ── Recall receipts (N1, any runtime) ─────────────────

    @staticmethod
    def _receipt_row(r) -> Dict:
        return {
            "recall_id": r[0], "runtime": r[1] or "", "query": r[2] or "",
            "node_ids": json.loads(r[3] or "[]"), "issued_at": r[4],
            "expires_at": r[5], "status": r[6], "closed_at": r[7],
        }

    _RECEIPT_COLUMNS = ("recall_id, runtime, query, node_ids, issued_at, "
                        "expires_at, status, closed_at")

    @_locked
    def add_recall_receipt(
        self, *, recall_id: str, runtime: str, query: str,
        node_ids: List[str], issued_at: datetime, expires_at: datetime,
    ) -> None:
        """Record one issued receipt (status open)."""
        with self.transaction(immediate=True) as conn:
            query, runtime = self._cut_secret_strings(conn, query, runtime)
            conn.execute(
                f"INSERT INTO recall_receipts ({self._RECEIPT_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, 'open', NULL)",
                (recall_id, runtime, query, json.dumps(list(node_ids)),
                 issued_at.isoformat(), expires_at.isoformat()),
            )

    @_locked
    def get_recall_receipt(self, recall_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT {self._RECEIPT_COLUMNS} FROM recall_receipts "
            "WHERE recall_id = ?", (recall_id,),
        ).fetchone()
        return self._receipt_row(row) if row else None

    @_locked
    def close_recall_receipt(self, recall_id: str, status: str) -> bool:
        """Move an OPEN receipt to ``status`` (redeemed). True only for the
        caller whose update landed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE recall_receipts SET status = ?, closed_at = ? "
            "WHERE recall_id = ? AND status = 'open'",
            (status, datetime.now(timezone.utc).isoformat(), recall_id),
        )
        self._commit(conn)
        return cursor.rowcount == 1

    @_locked
    def expire_recall_receipts(self, now: Optional[datetime] = None) -> int:
        """Open receipts past their expiry become flags (status expired).
        Never deleted. Returns how many were flagged now."""
        now = now or datetime.now(timezone.utc)
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE recall_receipts SET status = 'expired', closed_at = ? "
            "WHERE status = 'open' AND expires_at < ?",
            (now.isoformat(), now.isoformat()),
        )
        self._commit(conn)
        return cursor.rowcount

    @_locked
    def list_receipt_flags(self, include_dismissed: bool = False) -> List[Dict]:
        """Receipts that expired unjudged, newest first."""
        conn = self._get_conn()
        statuses = ("expired", "dismissed") if include_dismissed else ("expired",)
        marks = ", ".join("?" for _ in statuses)
        rows = conn.execute(
            f"SELECT {self._RECEIPT_COLUMNS} FROM recall_receipts "
            f"WHERE status IN ({marks}) ORDER BY expires_at DESC", statuses,
        ).fetchall()
        return [self._receipt_row(r) for r in rows]

    @_locked
    def dismiss_receipt_flag(self, recall_id: str) -> bool:
        """A person has seen an expired-unjudged flag. Kept, marked dismissed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE recall_receipts SET status = 'dismissed' "
            "WHERE recall_id = ? AND status = 'expired'", (recall_id,),
        )
        self._commit(conn)
        return cursor.rowcount == 1

    @_locked
    def use_judgment_counts(self) -> Dict[str, int]:
        """Totals over the whole ledger: judged, used, unused, applied, and
        apply_unknown (a separate bucket, not part of applied)."""
        conn = self._get_conn()
        judged, used, applied, unknown = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(used), 0), COALESCE(SUM(applied = 1), 0), "
            "COALESCE(SUM(applied = 2), 0) FROM use_judgments"
        ).fetchone()
        return {
            "judged": int(judged),
            "used": int(used),
            "unused": int(judged) - int(used),
            "applied": int(applied),
            # claimed for apply and never finished (in progress, or the
            # process died): mark_used may have run. Not counted as applied.
            "apply_unknown": int(unknown),
        }

    @_locked
    def find_context_node_by_ingest_key(
        self, ingest_key: str, node_type: NodeType = NodeType.CONTEXT
    ) -> Optional[Node]:
        """The node stamped with this ingest key (stable identity for a
        re-ingestable unit, e.g. one adapter session file). Defaults to
        CONTEXT (its original, still the common case); ``node_type=SKILL``
        is the same lookup scoped to skill ingest (revien/skills/ingest.py)
        instead of a separate near-duplicate. metadata is a JSON TEXT
        column: a LIKE prefilter on the JSON-encoded key narrows the scan,
        then the parsed metadata confirms the exact key (substring hits
        never false-positive). Unindexed on purpose — one lookup per keyed
        ingest at current scale; add a real index if keyed re-ingest gets hot.

        OLDEST WINS: rows scan created_at ASC, so if a concurrent same-key
        first-ingest race ever stamped twins, every later ingest refreshes
        the OLDEST one deterministically and the newer twin is left orphaned
        (never refreshed, never removed). Low exposure — the race window is
        one first-ever ingest — stated so nobody discovers it the hard way."""
        conn = self._get_conn()
        # json.dumps gives the exact serialized form ("quoted, escaped"), so
        # the prefilter matches how the key sits inside the metadata JSON.
        needle = json.dumps(ingest_key)
        rows = conn.execute(
            f"{self._SELECT_NODES} WHERE node_type = ? AND metadata LIKE ? "
            "ORDER BY created_at ASC",
            (node_type.value, f"%{needle}%"),
        ).fetchall()
        for row in rows:
            node = self._row_to_node(row)
            if (node.metadata or {}).get("ingest_key") == ingest_key:
                return node
        return None

    # ── Exact lane lookups (REVIEN_EXACT) ─────────────────────────────────
    # Narrow projections, no Node construction: the lane needs ids, labels,
    # types and event ranges, not whole rows. Label equality, the event-time
    # and recorded_at ranges, node ids and neighbours are indexed reads
    # (idx_nodes_label_lower, idx_nodes_event_time, idx_nodes_recorded_at,
    # the node_id primary key, the edge indexes); exact_label_contains is one
    # pass over the label column (bounded by the lane: exact.MAX_CONTAINED_CODES).

    @staticmethod
    def _exact_scope_sql(origin_runtime, include_context, live_only):
        sql, params = "", []
        if not include_context:
            sql += " AND node_type != 'context'"
        if live_only:
            sql += " AND invalidated_at IS NULL"
        if origin_runtime is not None:
            runtimes = (
                [origin_runtime] if isinstance(origin_runtime, str)
                else list(origin_runtime)
            )
            sql += f" AND origin_runtime IN ({', '.join('?' * len(runtimes))})"
            params.extend(runtimes)
        return sql, params

    @_locked
    def exact_label_matches(
        self, labels_lower, origin_runtime=None, include_context=False,
        live_only=True, limit: int = 500,
    ) -> list:
        """[(node_id, lower(label), node_type)] for nodes whose label equals
        one of ``labels_lower`` (SQLite lower(): ASCII case folding). Index
        lookup, not a scan."""
        labels = sorted({l for l in labels_lower if l})
        out: list = []
        if not labels:
            return out
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        conn = self._get_conn()
        for i in range(0, len(labels), 400):
            chunk = labels[i:i + 400]
            rows = conn.execute(
                "SELECT node_id, lower(label), node_type FROM nodes "
                f"WHERE lower(label) IN ({','.join('?' * len(chunk))})"
                f"{scope} ORDER BY node_id LIMIT ?",
                chunk + sparams + [limit],
            ).fetchall()
            out.extend((r[0], r[1], r[2]) for r in rows)
        return out

    @_locked
    def exact_label_contains(
        self, needles_lower, origin_runtime=None, include_context=False,
        live_only=True, limit: int = 400,
    ) -> list:
        """[(node_id, label, node_type)] whose label contains ANY of
        ``needles_lower`` as a substring (the caller verifies token
        boundaries). One pass over the label column for all needles, never
        content; used for code-like tokens only."""
        needles = [n for n in needles_lower if n]
        if not needles:
            return []
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        cond = " OR ".join(["instr(lower(label), ?) > 0"] * len(needles))
        rows = self._get_conn().execute(
            "SELECT node_id, label, node_type FROM nodes "
            f"WHERE ({cond}){scope} ORDER BY node_id LIMIT ?",
            needles + sparams + [limit],
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    @_locked
    def exact_node_ids(
        self, node_ids, origin_runtime=None, include_context=False,
        live_only=True,
    ) -> list:
        """[(node_id, label, node_type)] for ids that exist (primary key)."""
        ids = sorted({i for i in node_ids if i})
        if not ids:
            return []
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        rows = self._get_conn().execute(
            "SELECT node_id, label, node_type FROM nodes "
            f"WHERE node_id IN ({','.join('?' * len(ids))}){scope}",
            ids + sparams,
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    @_locked
    def exact_neighbors(
        self, node_ids, origin_runtime=None, include_context=False,
        live_only=True, limit: int = 1000,
    ) -> dict:
        """{node_id: [neighbor ids]} one hop through LIVE edges, either
        direction, for each requested id. Two indexed reads (idx_edges_source /
        idx_edges_target, then the node primary key for the scope check).
        An alias edge (ALIAS_OF) is not a neighbour: it joins two names of the
        SAME thing. A neighbour must be in scope (live, not context unless
        asked, origin filter). The node itself is never its own neighbour.
        ``limit`` bounds the edge rows read per direction, per call."""
        ids = sorted({i for i in node_ids if i})
        out: dict = {i: [] for i in ids}
        if not ids:
            return out
        conn = self._get_conn()
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT source_node_id, target_node_id FROM edges "
            f"WHERE source_node_id IN ({marks}) AND invalidated_at IS NULL "
            f"AND edge_type != 'alias_of' ORDER BY edge_id LIMIT ?",
            ids + [limit],
        ).fetchall()
        rows += [(b, a) for a, b in conn.execute(
            f"SELECT source_node_id, target_node_id FROM edges "
            f"WHERE target_node_id IN ({marks}) AND invalidated_at IS NULL "
            f"AND edge_type != 'alias_of' ORDER BY edge_id LIMIT ?",
            ids + [limit],
        ).fetchall()]
        pairs: dict = {}
        for ent, other in rows:
            if other != ent:
                pairs.setdefault(ent, set()).add(other)
        wanted = sorted({o for s in pairs.values() for o in s})
        if not wanted:
            return out
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        in_scope: set = set()
        for i in range(0, len(wanted), self._IN_CHUNK):
            chunk = wanted[i:i + self._IN_CHUNK]
            in_scope.update(r[0] for r in conn.execute(
                f"SELECT node_id FROM nodes WHERE node_id IN "
                f"({','.join('?' * len(chunk))}){scope}",
                chunk + sparams,
            ).fetchall())
        for ent, others in pairs.items():
            out[ent] = sorted(o for o in others if o in in_scope)
        return out

    @_locked
    def event_time_year_spans(self) -> list:
        """Distinct (start_year, end_year) pairs over dated nodes, from the
        covering event-time index. Lets a year-less date ("9/21") expand to
        the years the graph actually has."""
        rows = self._get_conn().execute(
            "SELECT DISTINCT substr(event_time_start, 1, 4), "
            "substr(COALESCE(event_time_end, event_time_start), 1, 4) "
            "FROM nodes WHERE event_time_start IS NOT NULL"
        ).fetchall()
        out = []
        for a, b in rows:
            try:
                out.append((int(a), int(b)))
            except (TypeError, ValueError):
                continue
        return out

    @_locked
    def recorded_year_list(self) -> list:
        """Distinct years of recorded_at (index scan), for year-less dates."""
        out = []
        for (y,) in self._get_conn().execute(
                "SELECT DISTINCT substr(recorded_at, 1, 4) FROM nodes "
                "WHERE recorded_at IS NOT NULL"):
            try:
                out.append(int(y))
            except (TypeError, ValueError):
                continue
        return out

    @_locked
    def exact_recorded_matches(
        self, day_ranges, origin_runtime=None, include_context=False,
        live_only=True, limit: int = 400,
    ) -> list:
        """[(node_id, node_type, recorded_at)] whose recorded_at (WHEN THE
        CONTENT WAS SAID, in the speaker's own offset: the first ten characters
        are the local calendar day) falls in an inclusive ('YYYY-MM-DD',
        'YYYY-MM-DD') range, and ONLY where metadata.recorded_at_source is
        content, capture or import (the sources the renderers trust, see
        revien/dates.py). A file mtime, an unlabeled row, and created_at
        (ingest time) are never matched."""
        ranges = list(day_ranges)
        if not ranges:
            return []
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        conn = self._get_conn()
        seen: dict = {}
        for i in range(0, len(ranges), 100):
            chunk = ranges[i:i + 100]
            terms, params = [], []
            for lo, hi in chunk:
                terms.append("(recorded_at >= ? AND recorded_at < ?)")
                params.extend([lo, _next_day(hi)])
            rows = conn.execute(
                "SELECT node_id, node_type, recorded_at FROM nodes "
                f"WHERE recorded_at IS NOT NULL AND ({' OR '.join(terms)}) "
                "AND json_extract(metadata, '$.recorded_at_source') "
                f"IN ('content', 'capture', 'import'){scope} "
                "ORDER BY node_id LIMIT ?",
                params + sparams + [limit],
            ).fetchall()
            for r in rows:
                seen[r[0]] = (r[0], r[1], r[2])
        return list(seen.values())

    @_locked
    def exact_event_matches(
        self, day_ranges, origin_runtime=None, include_context=False,
        live_only=True, limit: int = 400,
    ) -> list:
        """[(node_id, node_type, start, end, granularity)] for dated nodes
        whose [event_time_start, event_time_end] overlaps any inclusive
        ('YYYY-MM-DD', 'YYYY-MM-DD') range in ``day_ranges``. Comparison is on
        the stored ISO text (the speaker's own calendar day, see
        engine._iso_utc); a node with no event_time_start is never returned.
        Content text is never consulted."""
        ranges = list(day_ranges)
        if not ranges:
            return []
        scope, sparams = self._exact_scope_sql(
            origin_runtime, include_context, live_only)
        conn = self._get_conn()
        seen: dict = {}
        for i in range(0, len(ranges), 100):
            chunk = ranges[i:i + 100]
            terms, params = [], []
            for lo, hi in chunk:
                # [start, end] overlaps day range [lo, hi]  <=>
                #   start < day-after-hi  AND  end >= lo   (text order)
                terms.append(
                    "(event_time_start < ? AND "
                    "COALESCE(event_time_end, event_time_start) >= ?)")
                params.extend([_next_day(hi), lo])
            rows = conn.execute(
                "SELECT node_id, node_type, event_time_start, event_time_end, "
                "event_time_granularity FROM nodes "
                f"WHERE event_time_start IS NOT NULL AND ({' OR '.join(terms)})"
                f"{scope} ORDER BY node_id LIMIT ?",
                params + sparams + [limit],
            ).fetchall()
            for r in rows:
                seen[r[0]] = (r[0], r[1], r[2], r[3], r[4])
        return list(seen.values())

    @_locked
    def search_nodes_keyword(
        self, keywords, limit: int = 10, exclude_context: bool = True,
        origin_runtime: Optional[Union[str, List[str]]] = None,
        project_key: Optional[str] = None,
    ) -> list["Node"]:
        """SQL-side keyword search over label+content (case-insensitive
        substring, OR across keywords), newest first. Replaces the recall
        fallback's list_nodes(limit=999999)-then-scan-in-Python, which was
        the single biggest recall latency driver (OPEN 2). Semantics match
        the old Python scan: any-keyword substring hit qualifies.

        Origin Layer (WS0 Leg B): origin_runtime/project_key are the SAME
        SQL prefilter as list_nodes — applied in the WHERE clause here, not
        as a post-fetch scan, so a filtered recall's keyword-anchor path
        never even considers another runtime's rows as candidates."""
        kws = [k for k in keywords if k]
        if not kws:
            return []
        conn = self._get_conn()
        conditions = " OR ".join(
            ["instr(lower(label || ' ' || content), ?) > 0"] * len(kws)
        )
        query = f"{self._SELECT_NODES} WHERE ({conditions})"
        params: list = [k.lower() for k in kws]
        if exclude_context:
            query += " AND node_type != 'context'"
        if origin_runtime is not None:
            # G4: ``is not None``, never a truthy check — origin_runtime=[]
            # is a PRESENT-but-empty filter (fail closed, matches nothing
            # via "IN ()", valid SQL), and a truthy check would silently
            # treat that the same as "no filter" (origin_runtime=None).
            runtimes = (
                [origin_runtime] if isinstance(origin_runtime, str) else list(origin_runtime)
            )
            placeholders = ", ".join("?" * len(runtimes))
            query += f" AND origin_runtime IN ({placeholders})"
            params.extend(runtimes)
        if project_key:
            query += " AND project_key = ?"
            params.append(project_key)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    @staticmethod
    def _named_secrets(conn) -> list:
        """The keyed entries of every secret string a ``forget --wording``
        named, read on the connection that is about to write. Read inside the
        write transaction (BEGIN IMMEDIATE), so a forget committed by another
        process is either seen here or waits for this write and cuts it."""
        from revien.forget import gate as forget_gate

        return forget_gate.literals_on(conn)

    @classmethod
    def _cut_secret_strings(cls, conn, *values):
        """The values with the named secrets cut out, in order. Strings and
        JSON text only; anything else is returned as it came."""
        from revien.forget import gate as forget_gate

        entries = cls._named_secrets(conn)
        if not entries:
            return values
        out = []
        for v in values:
            if isinstance(v, str) and v:
                v, _ = forget_gate.cut_literals(v, entries)
            out.append(v)
        return tuple(out)

    def _cut_ident(self, value):
        """An identity string with the secrets a forget --wording named cut
        out (the same cut the forget made in the stored identities)."""
        if not isinstance(value, str) or not value:
            return value
        return self._cut_secret_strings(self._get_conn(), value)[0]

    @classmethod
    def _cut_derived(cls, conn, node) -> None:
        """A label, a preview or a context line derived from the content
        (cut off where the text was truncated) keeps the piece of a named
        secret that came before the cut. Before the content itself is cut:
        measure each derived string against the occurrence in the content,
        and cut the piece that stops exactly inside it. The secret is read off
        the content (the entries hold only keyed hashes)."""
        from revien.forget import gate as forget_gate
        from revien.forget import keys as forget_keys
        from revien.forget import scrub

        entries = cls._named_secrets(conn)
        content = node.content
        if not entries or not isinstance(content, str) or not content:
            return
        spans = forget_keys.find_literals(content, entries)
        if not spans:
            return

        groups = {}
        for s0, e0 in spans:
            groups.setdefault(scrub.canonical(content[s0:e0]), []).append((s0, e0))

        def fix(text):
            if not isinstance(text, str) or not text:
                return text
            for part, group in groups.items():
                piece = scrub.aligned_edge_fragment(text, content, group, part)
                if piece:
                    return scrub.cut_edge_fragment(text, part, piece)[0]
            return text

        node.label = fix(node.label)
        node.source_context = fix(node.source_context)
        node.event_time_text = fix(node.event_time_text)
        if isinstance(node.metadata, dict):
            def walk(v):
                if isinstance(v, str):
                    return fix(v)
                if isinstance(v, list):
                    return [walk(x) for x in v]
                if isinstance(v, dict):
                    return {k: walk(x) for k, x in v.items()}
                return v
            node.metadata = walk(node.metadata)

    @classmethod
    def _cut_secret_metadata(cls, conn, meta):
        from revien.forget import gate as forget_gate

        entries = cls._named_secrets(conn)
        if not entries or not isinstance(meta, dict):
            return meta
        raw, n = forget_gate.cut_literals_json(json.dumps(meta), entries)
        return json.loads(raw) if n else meta

    def _without_named_secrets(self, kwargs: dict, conn=None) -> dict:
        """An edit (PUT /v1/nodes, a correction, a supersession) cannot write a
        secret a ``forget --wording`` named back into a node."""
        conn = conn or self._get_conn()
        out = dict(kwargs)
        for key in ("label", "content", "source_context", "event_time_text",
                    "source_id", "confidence_set_by", "origin_source",
                    "session_key", "project_key"):
            if isinstance(out.get(key), str):
                out[key], = self._cut_secret_strings(conn, out[key])
        if isinstance(out.get("metadata"), dict):
            out["metadata"] = self._cut_secret_metadata(conn, out["metadata"])
        return out

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
        # The write lock is taken BEFORE the denylist is read, so a forget in
        # another process cannot land between the check and the write.
        text_edit = any(k in kwargs for k in (
            "label", "content", "source_context", "event_time_text", "metadata",
            "source_id", "confidence_set_by", "origin_source", "session_key",
            "project_key"))
        with self.transaction(immediate=text_edit) as conn:
            existing = self.get_node(node_id)
            if existing is None:
                return None

            if text_edit:
                kwargs = self._without_named_secrets(kwargs, conn)

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

            # The cut-label mark (LABEL_CUT_KEY) says the content is the whole
            # name the label was cut from. An edit cannot add or change it,
            # and an edit of the content drops it: the content is no longer
            # that name.
            had = (existing.metadata or {}).get(LABEL_CUT_KEY)
            md = updates.get("metadata", existing.metadata) or {}
            if isinstance(md, dict) and LABEL_CUT_KEY in md and (
                    md[LABEL_CUT_KEY] != had
                    or updates.get("content", existing.content) != existing.content):
                updates["metadata"] = {
                    k: v for k, v in md.items() if k != LABEL_CUT_KEY}

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
            if True:                       # (one block with the guard above)
                conn.execute(
                    f"UPDATE nodes SET {', '.join(set_clauses)} WHERE node_id = ?",
                    values,
                )
                updated = self.get_node(node_id)
                if updated is not None and (
                    updated.label != existing.label
                    or updated.content != existing.content
                ):
                    self._enqueue_embed_stmt(conn, node_id)
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
            self._notify_successor(updated)
        return updated

    @_locked
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges. Returns True if deleted."""
        # The successor must be found while the row still exists; its stored
        # vector (prev-context embeds) carries the text being forgotten.
        # Only when a layer actually listens for successor changes (the
        # semantic index registers one only under REVIEN_EMBED_CONTEXT=prev):
        # otherwise this is a SELECT + get_node on every delete for nothing.
        successor = None
        if self._has_successor_listener():
            doomed = self.get_node(node_id)
            if doomed is not None and doomed.node_type == NodeType.CONTEXT:
                successor = self.next_context_in_session(doomed)
        with self.transaction() as conn:
            # Edges cascade via FK, but be explicit for safety
            conn.execute(
                "DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
                (node_id, node_id),
            )
            # Usage judgments keep the memory's own words (shared_terms), so
            # they go with it — right-to-forget runs through here, and its
            # promise is that the content is actually gone.
            conn.execute("DELETE FROM use_judgments WHERE node_id = ?", (node_id,))
            # Receipts hold only ids, but a forgotten node's id should not
            # linger in an open receipt or a flag either.
            for recall_id, raw in conn.execute(
                "SELECT recall_id, node_ids FROM recall_receipts "
                "WHERE node_ids LIKE ?", (f'%"{node_id}"%',),
            ).fetchall():
                remaining = [n for n in json.loads(raw or "[]") if n != node_id]
                conn.execute(
                    "UPDATE recall_receipts SET node_ids = ? WHERE recall_id = ?",
                    (json.dumps(remaining), recall_id),
                )
            cursor = conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
            deleted = cursor.rowcount > 0
            self._discard_pending_stmt(conn, node_id)
            # Everything else that points at the node by id goes with it:
            # ownership, queue entries naming it, the distill manifest rows
            # that render it. None of them can outlive the node it describes.
            conn.execute(
                "DELETE FROM node_provenance WHERE node_id = ?", (node_id,))
            conn.execute(
                "DELETE FROM supersession_candidates "
                "WHERE existing_node_id = ? OR new_node_id = ?",
                (node_id, node_id))
            conn.execute(
                "DELETE FROM distill_manifest WHERE anchor_node_id = ? "
                "OR current_node_id = ? OR entity_id = ?",
                (node_id, node_id, node_id))
        if deleted:
            self._fire_delete(node_id)
            if successor is not None:
                self._fire_successor_change(successor)
        return deleted

    @_locked
    def list_nodes(
        self,
        node_type: Optional[NodeType] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        origin_runtime: Optional[Union[str, List[str]]] = None,
        project_key: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> list[Node]:
        """List nodes, newest first. Origin Layer (WS0 Leg B): origin_runtime
        and project_key are SQL WHERE prefilters, not a post-fetch scan — a
        single runtime string ('claude-code') or a list of runtimes
        (['claude-code', 'codex']) both compile to a bound-parameter IN
        clause, never string interpolation. A node whose origin_runtime is
        NULL (unknown provenance) never matches either filter — filtered
        recall never silently includes "unknown" as a fourth runtime."""
        conn = self._get_conn()
        query = f"{self._SELECT_NODES} WHERE 1=1"
        params: list = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type.value)
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if origin_runtime is not None:
            # G4: ``is not None``, never a truthy check — see the same note
            # in search_nodes_keyword above. origin_runtime=[] must fail
            # closed (match nothing), not silently unfilter.
            runtimes = (
                [origin_runtime] if isinstance(origin_runtime, str) else list(origin_runtime)
            )
            placeholders = ", ".join("?" * len(runtimes))
            query += f" AND origin_runtime IN ({placeholders})"
            params.extend(runtimes)
        if project_key:
            query += " AND project_key = ?"
            params.append(project_key)
        if session_key:
            query += " AND session_key = ?"
            params.append(session_key)
        # G7: created_at ties (same-millisecond ingest, common in bulk
        # backfills/tests) previously left paging order unstable across
        # calls — node_id is a deterministic tiebreaker so offset paging
        # never skips or repeats a row.
        query += " ORDER BY created_at DESC, node_id LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    @_locked
    def list_draft_proposed_skills(self, limit: int = 1000, offset: int = 0) -> list[Node]:
        """SKILL nodes whose metadata LIKELY has status="proposed" AND
        draft=true — an SQL LIKE prefilter (G9); callers still parse+confirm
        metadata exactly, a substring hit is never trusted as the whole truth."""
        conn = self._get_conn()
        rows = conn.execute(
            f"{self._SELECT_NODES} WHERE node_type = ? "
            "AND metadata LIKE '%\"status\": \"proposed\"%' "
            "AND metadata LIKE '%\"draft\": true%' "
            "ORDER BY created_at DESC, node_id LIMIT ? OFFSET ?",
            (NodeType.SKILL.value, limit, offset),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    @_locked
    def count_nodes(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    @_locked
    def count_nodes_by_recorded_at_source(self) -> dict:
        """Node counts per metadata recorded_at_source. Two honest buckets
        instead of one "unknown": "undated" (recorded_at NULL) and
        "unlabeled" (recorded_at set, no source recorded)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT CASE WHEN recorded_at IS NULL THEN 'undated' ELSE "
            "COALESCE(json_extract(metadata, '$.recorded_at_source'), 'unlabeled') END, "
            "COUNT(*) FROM nodes GROUP BY 1 ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {src: count for src, count in rows}

    @_locked
    def count_nodes_by_origin_runtime(self) -> dict:
        """Per-runtime node counts (WS0 Leg B) — feeds `revien status`'s
        per-runtime table. NULL origin_runtime (unrecognized/pre-origin-layer
        source_id) groups under the string key "unknown" so its count isn't
        silently dropped from the total."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT COALESCE(origin_runtime, 'unknown'), COUNT(*) "
            "FROM nodes GROUP BY 1 "
            "ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {runtime: count for runtime, count in rows}

    # ── Supersession candidate queue (CSL Leg 3 wiring) ───────────────────────
    @_locked
    def add_candidate(
        self, existing_node_id: str, new_node_id: str, action: str,
        reason: str = "", trace: str = "",
    ) -> int:
        """Park a contradiction the gate routed to review. Mutates no node."""
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction(immediate=True) as conn:
            reason, trace = self._cut_secret_strings(conn, reason, trace)
            cur = conn.execute(
                """INSERT INTO supersession_candidates
                   (existing_node_id, new_node_id, action, reason, trace, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (existing_node_id, new_node_id, action, reason, trace, now),
            )
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
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction(immediate=True) as conn:
            resolution, = self._cut_secret_strings(conn, resolution)
            conn.execute(
                "UPDATE supersession_candidates SET resolved_at = ?, resolution = ? WHERE id = ?",
                (now, resolution, candidate_id),
            )

    @_locked
    def count_candidates(self, unresolved_only: bool = True) -> int:
        conn = self._get_conn()
        q = "SELECT COUNT(*) FROM supersession_candidates"
        if unresolved_only:
            q += " WHERE resolved_at IS NULL"
        return conn.execute(q).fetchone()[0]

    @staticmethod
    def _check_validity_window(valid_from, valid_until) -> None:
        """Raise ValueError when valid_from is after valid_until. Equal bounds
        are an empty window, not an inverted one. Naive datetimes read as UTC."""
        if valid_from is None or valid_until is None:
            return
        vf = valid_from if valid_from.tzinfo else valid_from.replace(tzinfo=timezone.utc)
        vu = valid_until if valid_until.tzinfo else valid_until.replace(tzinfo=timezone.utc)
        if vf > vu:
            raise ValueError(
                f"inverted validity window: valid_from {vf.isoformat()} is "
                f"after valid_until {vu.isoformat()}")

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
        # Never persist an inverted window. The bound that would stand after
        # this call (an already-set bound wins) must not put from after until.
        self._check_validity_window(
            node.valid_from or valid_from, node.valid_until or valid_until)
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

    @_locked
    def list_alias_pairs(self, live_only: bool = True) -> list[dict]:
        """The alias view (alias leg): every ALIAS_OF pair with both labels
        and the evidence that drew the edge. Mirrors list_tension_pairs.

        live_only (default) excludes reversed (soft-invalidated) alias edges
        — a removed alias is history, not a live recall-expansion path. Pass
        False for lineage/audit reading (the CLI's ``--all``).
        """
        conn = self._get_conn()
        q = (
            "SELECT e.edge_id, e.source_context, e.created_at, e.weight, "
            "       e.confidence, e.confidence_set_by, e.metadata, e.invalidated_at, "
            "       a.node_id, a.label, a.node_type, a.invalidated_at, "
            "       b.node_id, b.label, b.node_type, b.invalidated_at "
            "FROM edges e "
            "JOIN nodes a ON a.node_id = e.source_node_id "
            "JOIN nodes b ON b.node_id = e.target_node_id "
            "WHERE e.edge_type = 'alias_of' "
            "ORDER BY e.created_at ASC"
        )
        out = []
        for r in conn.execute(q).fetchall():
            if live_only and r[7] is not None:
                continue
            meta = json.loads(r[6]) if r[6] else {}
            out.append({
                "edge_id": r[0],
                "source_context": r[1],
                "created_at": r[2],
                "weight": r[3],
                "confidence": r[4],
                "set_by": r[5],
                "method": meta.get("method"),
                "embedding_sim": meta.get("embedding_sim"),
                "cooccurrence": meta.get("cooccurrence"),
                "invalidated_at": r[7],
                "a": {"node_id": r[8], "label": r[9], "node_type": r[10],
                      "invalidated_at": r[11]},
                "b": {"node_id": r[12], "label": r[13], "node_type": r[14],
                      "invalidated_at": r[15]},
            })
        return out

    # ── Edge CRUD ─────────────────────────────────────────

    @staticmethod
    def _add_edge_stmt(conn: sqlite3.Connection, edge: Edge) -> None:
        """The bare INSERT for one edge — no commit, no audit. Callers own
        the transaction (add_edge and add_edge_audited both wrap it).
        A secret a ``forget --wording`` named is cut out first."""
        edge.source_context, edge.confidence_set_by = GraphStore._cut_secret_strings(
            conn, edge.source_context, edge.confidence_set_by)
        edge.metadata = GraphStore._cut_secret_metadata(conn, edge.metadata)
        conn.execute(
            """INSERT INTO edges
               (edge_id, edge_type, source_node_id, target_node_id,
                weight, created_at, metadata,
                confidence, confidence_set_at, confidence_set_by, source_context,
                invalidated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                edge.invalidated_at.isoformat() if edge.invalidated_at else None,
            ),
        )

    @_locked
    def add_edge(self, edge: Edge) -> Edge:
        """Unaudited edge creation — UNCHANGED behavior for its many existing
        callers (mention-linking, connect_nodes, tombstone re-pointing, the
        /v1/edges endpoint, import_graph). See add_edge_audited for the
        provenance-bearing path (alias leg)."""
        with self.transaction(immediate=True) as conn:
            self._add_edge_stmt(conn, edge)
        return edge

    @_locked
    def add_edge_audited(self, edge: Edge, actor: str = "") -> Edge:
        """Audited edge creation (alias leg): the INSERT and its 'create'
        audit row land in ONE commit, mirroring add_node exactly — an audit
        failure rolls the edge creation back rather than leaving an edge the
        trail can't account for. Use this for edges whose provenance matters
        (ALIAS_OF today); add_edge above is UNCHANGED for every other caller.
        """
        with self.transaction(immediate=True) as conn:
            self._add_edge_stmt(conn, edge)
            self._record_edge_audit(
                edge.edge_id, "create", actor=actor,
                before_edge=None, after_edge=edge,
            )
        return edge

    @_locked
    def _record_edge_audit(
        self,
        edge_id: str,
        op: str,
        actor: str = "",
        before_edge: Optional["Edge"] = None,
        after_edge: Optional["Edge"] = None,
    ) -> None:
        """Edge analogue of _record_node_audit — same audit_log table, same
        transaction-depth contract (required/raises inside a transaction,
        best-effort/swallowed standalone). There is no dedicated edge_audit
        table: audit_log's subject-id column carries a node_id OR an edge_id
        interchangeably (both are opaque UUID strings; the ``op`` and the
        before/after snapshot shape disambiguate which)."""
        if self._txn_depth > 0:
            self.record_audit(
                edge_id, op, actor=actor,
                before=self._audit_snapshot(before_edge),
                after=self._audit_snapshot(after_edge),
            )
            return
        try:
            self.record_audit(
                edge_id, op, actor=actor,
                before=self._audit_snapshot(before_edge),
                after=self._audit_snapshot(after_edge),
            )
        except Exception:
            pass

    def get_edge_audit(self, edge_id: str) -> list[dict]:
        """Edge audit history — same audit_log table as get_node_audit, keyed
        by edge_id instead of node_id (see _record_edge_audit)."""
        return self.get_node_audit(edge_id)

    @_locked
    def update_edge(
        self, edge_id: str, _audit_op: Optional[str] = "update",
        _audit_actor: str = "", **kwargs
    ) -> Optional[Edge]:
        """Update specific fields on an edge. Mirrors update_node: mutation +
        audit land in ONE commit, an audit failure rolls the UPDATE back.
        Fields are confined to what provenance/invalidation needs today —
        the existing (unaudited) update_edge_weight keeps serving weight
        reinforcement callers, untouched.
        """
        existing = self.get_edge(edge_id)
        if existing is None:
            return None

        allowed_fields = (
            "metadata", "confidence", "confidence_set_at",
            "confidence_set_by", "source_context", "invalidated_at",
        )
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return existing

        conn = self._get_conn()
        set_clauses = []
        values = []
        datetime_fields = ("confidence_set_at", "invalidated_at")
        for key, val in updates.items():
            set_clauses.append(f"{key} = ?")
            if key == "metadata":
                values.append(json.dumps(val))
            elif key in datetime_fields:
                values.append(val.isoformat() if isinstance(val, datetime) else val)
            else:
                values.append(val)
        values.append(edge_id)

        with self.transaction(immediate=True):
            # the denylist is read with the write lock held
            for i, key in enumerate(updates):
                if key in ("source_context", "confidence_set_by"):
                    values[i], = self._cut_secret_strings(conn, values[i])
                elif key == "metadata":
                    values[i] = json.dumps(self._cut_secret_metadata(conn, updates[key]))
            conn.execute(
                f"UPDATE edges SET {', '.join(set_clauses)} WHERE edge_id = ?",
                values,
            )
            updated = self.get_edge(edge_id)
            if _audit_op:
                self._record_edge_audit(
                    edge_id, _audit_op, actor=_audit_actor,
                    before_edge=existing, after_edge=updated,
                )
        return updated

    @_locked
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        conn = self._get_conn()
        row = conn.execute(
            f"{self._SELECT_EDGES} WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    @_locked
    def get_edges_for_node(self, node_id: str) -> list[Edge]:
        """Get all edges connected to a node (as source or target)."""
        conn = self._get_conn()
        rows = conn.execute(
            f"{self._SELECT_EDGES} WHERE source_node_id = ? OR target_node_id = ?",
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
        edge_rows = conn.execute(self._SELECT_EDGES).fetchall()
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

        # ── One transaction: gate + wipe (replace only) + inserts + audits ──
        # The suppression gate (docs/FORGET_SPEC.md §6) decides INSIDE the
        # write transaction, after it has taken SQLite's write lock (BEGIN
        # IMMEDIATE): a forget from another connection either committed
        # before this point, and is seen here, or waits until this import
        # has committed, and then removes what it wrote.
        from revien.forget import gate as forget_gate
        from revien.forget import scrub as forget_scrub
        owners: dict = {}
        # A session's record names the id inside the file (and the file) its
        # other nodes came from: the id is the identity, the file name is not.
        session_ids: dict = {}
        for n in graph.nodes:
            md = n.metadata if isinstance(n.metadata, dict) else {}
            caller = md.get("caller") if isinstance(md.get("caller"), dict) else {}
            inner = caller.get("session_id")
            if (n.node_type.value == "context" and n.origin_runtime == "claude-code"
                    and n.session_key and isinstance(inner, str) and inner.strip()):
                session_ids[(n.origin_runtime, n.session_key)] = (
                    inner.strip(), caller.get("path"))
        with self.transaction(immediate=True):
            kept_nodes = []
            dropped_ids = set()
            for node in new_nodes:
                unit_id, label_ok, wording_ok, likely = self.ownership_for_fields(
                    node.node_type.value, node.origin_runtime, node.session_key,
                    node.source_id, node.metadata)
                md = node.metadata if isinstance(node.metadata, dict) else {}
                path = md.get("path") if isinstance(md.get("path"), str) else None
                named = session_ids.get((node.origin_runtime, node.session_key))
                if named is not None and likely:
                    canonical = f"session:{node.origin_runtime}:{named[0]}"
                    # the ownership row names the session by the id inside
                    # the file, not by the file name
                    unit_id = unit_id.replace(likely, canonical)
                    likely = canonical
                    if path is None and isinstance(named[1], str):
                        path = named[1]
                ref = forget_gate.UnitRef(likely, path)
                # A node of a forgotten unit, or whose whole text was
                # forgotten, is not brought back. Its label alone decides
                # nothing: a label that equals a forgotten label must not
                # cost the node its other sentences.
                if forget_gate.check(self, ref, node.content) is not None:
                    dropped_ids.add(node.node_id)
                    continue
                # The standing denylist: sentences forgotten from this origin
                # are left out of an imported node; a node
                # that was only a forgotten sentence is not imported.
                cleaned, left_out = forget_gate.strip_forgotten(
                    self, ref, node.content)
                if left_out:
                    if node.node_type.value not in ("context", "skill") \
                            or not cleaned.strip():
                        dropped_ids.add(node.node_id)
                        continue
                    node.content = cleaned
                new_label, label_out = forget_gate.strip_forgotten(
                    self, ref, node.label)
                if label_out:
                    first = next(iter(forget_scrub.iter_sentences(node.content)), None)
                    node.label = (new_label.strip()
                                  or (first[2] if first else "") or "[removed]")[:200]
                owners[node.node_id] = (unit_id, label_ok, wording_ok)
                kept_nodes.append(node)
            new_nodes = kept_nodes
            kept_edges = [e for e in new_edges
                          if e.source_node_id not in dropped_ids
                          and e.target_node_id not in dropped_ids]
            result["skipped_suppressed"] = len(dropped_ids)
            result["skipped_suppressed_edges"] = len(new_edges) - len(kept_edges)
            new_edges = kept_edges

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
                # Same promise as delete_node: judgments carry the removed
                # memories' own words, receipts their ids.
                conn.execute("DELETE FROM use_judgments")
                conn.execute("UPDATE recall_receipts SET node_ids = '[]'")
                conn.execute("DELETE FROM node_provenance")
                self._discard_pending_stmt(conn)
            for node in new_nodes:
                # rides this transaction; audit included. The graph file
                # carries no record of who wrote a node, so ownership is
                # proven only where an ingest key proves it (see
                # ownership_for_fields); the rest is marked unproven.
                self.add_node(node, owner=owners[node.node_id])
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

    # Readers name their columns rather than SELECT *, so _row_to_node and
    # _row_to_edge get this order whatever the table's physical order is. An
    # older DB can hold a column added outside this store (clustering's
    # community_id) ahead of columns a later migration appended.
    _NODE_COLUMNS = (
        "node_id", "node_type", "label", "content", "source_id",
        "created_at", "last_accessed", "access_count", "metadata",
        "source_type", "confidence", "pinned", "confidence_set_at",
        "confidence_set_by", "source_context", "last_referenced",
        "invalidated_at", "source_modality", "answerable_by_text",
        "vision_processed", "recorded_at", "event_time_start",
        "event_time_end", "event_time_granularity", "event_time_confidence",
        "event_time_text", "valid_from", "valid_until",
        "origin_runtime", "origin_source", "project_key", "session_key",
    )
    _EDGE_COLUMNS = (
        "edge_id", "edge_type", "source_node_id", "target_node_id",
        "weight", "created_at", "metadata",
        "confidence", "confidence_set_at", "confidence_set_by", "source_context",
        "invalidated_at",
    )
    _SELECT_NODES = f"SELECT {', '.join(_NODE_COLUMNS)} FROM nodes"
    _SELECT_EDGES = f"SELECT {', '.join(_EDGE_COLUMNS)} FROM edges"

    def _row_to_node(self, row: tuple) -> Node:
        try:
            node_type = NodeType(row[1])
        except ValueError:
            # A type nobody registered, written after this store opened (the
            # open-time check cannot see rows that do not exist yet).
            raise _extensions.unknown_row_error("node", row[1], row[0]) from None
        return Node(
            node_id=row[0],
            node_type=node_type,
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
            # Origin Layer (columns 28-31): nullable — a pre-WS0 row reads
            # back with no origin at all until the backfill pass runs.
            origin_runtime=row[28] if len(row) > 28 and row[28] else None,
            origin_source=row[29] if len(row) > 29 and row[29] else None,
            project_key=row[30] if len(row) > 30 and row[30] else None,
            session_key=row[31] if len(row) > 31 and row[31] else None,
        )

    def _row_to_edge(self, row: tuple) -> Edge:
        try:
            edge_type = EdgeType(row[1])
        except ValueError:
            raise _extensions.unknown_row_error("edge", row[1], row[0]) from None
        return Edge(
            edge_id=row[0],
            edge_type=edge_type,
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
            # Provenance Layer, edge half (column 11, alias leg): soft-
            # invalidation marker. Absent/NULL on every pre-alias-leg row.
            invalidated_at=datetime.fromisoformat(row[11]) if len(row) > 11 and row[11] else None,
        )
