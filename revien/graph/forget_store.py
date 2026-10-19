"""Storage for the forget layer (migration 006).

Ownership (``node_provenance``), the unit registry (``capture_units``) and the
suppression records (``suppressions``) live here, as a mixin on GraphStore.
This module imports nothing from store.py, so there is no cycle.

Nothing in this file decides what to forget. It records who wrote what at
write time, answers questions about it, and keeps the content-free suppression
records. The decisions are in revien/forget/.
"""

import functools
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .units import UNPROVEN_PREFIX, node_likely_unit


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ForgetStoreMixin:
    # Migration 006: the ONLY place this layer's DDL lives.
    _FORGET_LAYER_DDL = """
        -- Who wrote (or merged into) which node, recorded AT WRITE TIME by
        -- every writer. label_ok: the unit vouched for the node's name.
        -- wording_ok: the unit vouched for its wording. A unit_id starting
        -- 'unproven:' marks a node whose ownership could not be proven (it
        -- predates this table): it is kept, listed, and never deleted by a
        -- cascade. Nothing here is ever inferred from edges or text.
        CREATE TABLE IF NOT EXISTS node_provenance (
            node_id TEXT NOT NULL,
            unit_id TEXT NOT NULL,
            label_ok INTEGER NOT NULL DEFAULT 1,
            wording_ok INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (node_id, unit_id)
        );
        CREATE INDEX IF NOT EXISTS idx_node_provenance_unit
            ON node_provenance(unit_id);

        -- The units with a stable identity that have been captured. path and
        -- path_keys are the file identity (real path, inode) for a
        -- file-backed unit; no content.
        CREATE TABLE IF NOT EXISTS capture_units (
            unit_id TEXT PRIMARY KEY,
            path TEXT,
            path_keys TEXT NOT NULL DEFAULT '[]',
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );

        -- Suppression records. Content-free: identities, keyed fingerprints
        -- (HMAC under the key in the Revien home) and times. kind 'file':
        -- suppressed until the file changes. kind 'wording': no source is
        -- suppressed, only the forgotten sentences (a standing denylist).
        -- kind 'permanent': no file
        -- behind it, suppressed for good. status 'lifted' keeps the row as
        -- the record that a restore happened.
        -- scope 'origin': the sentences are kept out of captures that come
        -- from the forgotten origin (its unit id, its file identity, or a
        -- copy of it recognised by signature_fps, the keyed fingerprints of
        -- the origin's own sentences). scope 'global': the exact wording a
        -- forget --everywhere targeted, kept out of every source.
        CREATE TABLE IF NOT EXISTS suppressions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT,
            path_keys TEXT NOT NULL DEFAULT '[]',
            content_fps TEXT NOT NULL DEFAULT '[]',
            wording_fps TEXT NOT NULL DEFAULT '[]',
            sentence_fps TEXT NOT NULL DEFAULT '[]',
            short_fps TEXT NOT NULL DEFAULT '[]',
            signature_fps TEXT NOT NULL DEFAULT '[]',
            scope TEXT NOT NULL DEFAULT 'origin',
            file_fp TEXT,
            created_at TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            lifted_at TEXT,
            lifted_by TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_suppressions_unit
            ON suppressions(unit_id, status);
    """

    _FORGET_OBJECTS = (
        "node_provenance", "capture_units", "suppressions",
        "idx_node_provenance_unit", "idx_suppressions_unit")

    def _migrate_add_forget_layer(self, conn: sqlite3.Connection) -> None:
        """Migration 006: ownership, unit registry, suppression records.

        Idempotent and cheap on every connect. The backfill of existing nodes
        runs once (user_version < 6) and NEVER guesses: a context or skill
        node stamped with an ingest key is provably its unit's own record;
        every other existing node gets an ``unproven:`` marker, which keeps
        it and lists it. Nothing is deleted or rewritten."""
        try:
            have = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE name IN (%s)" % ",".join(
                    "?" * len(self._FORGET_OBJECTS)), self._FORGET_OBJECTS)}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= 6 and len(have) == len(self._FORGET_OBJECTS):
                self._ensure_suppression_columns(conn)
                return
            conn.executescript(self._FORGET_LAYER_DDL)
            self._ensure_suppression_columns(conn)
            if version < 6:
                self._backfill_provenance(conn)
                conn.execute("PRAGMA user_version = 6")
            self._commit(conn)
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e) or "already exists" in str(e):
                print(f"Migration note: {e}")
            else:
                raise

    @staticmethod
    def _ensure_suppression_columns(conn: sqlite3.Connection) -> None:
        """A suppressions table made before these columns existed gets them."""
        have = {r[1] for r in conn.execute("PRAGMA table_info(suppressions)")}
        for col, ddl in (
                ("sentence_fps", "TEXT NOT NULL DEFAULT '[]'"),
                ("short_fps", "TEXT NOT NULL DEFAULT '[]'"),
                ("signature_fps", "TEXT NOT NULL DEFAULT '[]'"),
                ("scope", "TEXT NOT NULL DEFAULT 'origin'")):
            if col not in have:
                conn.execute(f"ALTER TABLE suppressions ADD COLUMN {col} {ddl}")
                conn.commit()

    @staticmethod
    def ownership_for_fields(node_type: str, runtime: Optional[str],
                             session: Optional[str], source_id: Optional[str],
                             meta: Optional[dict]):
        """(unit_id, label_ok, wording_ok, likely_unit) for a node that
        arrives with no record of who wrote it (the upgrade backfill, a graph
        import). A context or skill node stamped with an ingest key is
        provably its unit's own record. Anything else is unproven: the
        returned unit_id is the ``unproven:`` marker, with the likely unit
        (or empty) after it."""
        meta = meta if isinstance(meta, dict) else {}
        ikey = meta.get("ingest_key")
        ikey = ikey if isinstance(ikey, str) and ikey else None
        likely = node_likely_unit(runtime, session, source_id, ikey)
        if node_type in ("context", "skill") and ikey and likely:
            return likely, 1, 1, likely
        return UNPROVEN_PREFIX + (likely or ""), 0, 0, likely

    def _backfill_provenance(self, conn: sqlite3.Connection) -> None:
        now = _now()
        rows = conn.execute(
            "SELECT node_id, node_type, origin_runtime, session_key, "
            "source_id, metadata FROM nodes WHERE node_id NOT IN "
            "(SELECT node_id FROM node_provenance) "
            "AND node_id NOT LIKE 'tombstone:%'").fetchall()
        for node_id, ntype, runtime, session, source_id, raw in rows:
            try:
                meta = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            unit_id, label_ok, wording_ok, likely = self.ownership_for_fields(
                ntype, runtime, session, source_id, meta)
            conn.execute(
                "INSERT OR IGNORE INTO node_provenance "
                "(node_id, unit_id, label_ok, wording_ok) VALUES (?,?,?,?)",
                (node_id, unit_id, label_ok, wording_ok))
            if label_ok:
                caller = meta.get("caller")
                path = None
                if isinstance(caller, dict) and isinstance(caller.get("path"), str):
                    path = caller["path"]
                elif isinstance(meta.get("path"), str):
                    path = meta["path"]
                conn.execute(
                    "INSERT OR IGNORE INTO capture_units "
                    "(unit_id, path, path_keys, first_seen, last_seen) "
                    "VALUES (?,?,?,?,?)",
                    (unit_id, path, "[]", now, now))

    # ── ownership ─────────────────────────────────────────

    @staticmethod
    def _record_provenance_stmt(
        conn: sqlite3.Connection, node_id: str, unit_id: str,
        label_ok: int = 1, wording_ok: int = 1, proves: bool = False,
    ) -> None:
        """Upsert one ownership row. Repeats only ever upgrade (MAX): a unit
        that vouched once keeps having vouched. ``proves`` drops this node's
        ``unproven:`` markers (used only when the write itself proves the
        node is the unit's own record, e.g. a keyed refresh)."""
        conn.execute(
            "INSERT INTO node_provenance (node_id, unit_id, label_ok, wording_ok) "
            "VALUES (?,?,?,?) ON CONFLICT(node_id, unit_id) DO UPDATE SET "
            "label_ok = MAX(label_ok, excluded.label_ok), "
            "wording_ok = MAX(wording_ok, excluded.wording_ok)",
            (node_id, unit_id, 1 if label_ok else 0, 1 if wording_ok else 0))
        if proves:
            conn.execute(
                "DELETE FROM node_provenance WHERE node_id = ? AND "
                "substr(unit_id, 1, ?) = ?",
                (node_id, len(UNPROVEN_PREFIX), UNPROVEN_PREFIX))

    @_locked
    def record_provenance(
        self, node_id: str, unit_id: str, label_ok: bool = True,
        wording_ok: bool = True, proves: bool = False,
    ) -> None:
        conn = self._get_conn()
        self._record_provenance_stmt(
            conn, node_id, unit_id, label_ok, wording_ok, proves)
        self._commit(conn)

    @_locked
    def get_provenance(self, node_id: str) -> List[Dict]:
        conn = self._get_conn()
        return [
            {"unit_id": u, "label_ok": bool(l), "wording_ok": bool(w)}
            for u, l, w in conn.execute(
                "SELECT unit_id, label_ok, wording_ok FROM node_provenance "
                "WHERE node_id = ? ORDER BY unit_id", (node_id,))]

    @_locked
    def get_provenance_bulk(self, node_ids: Iterable[str]) -> Dict[str, List[Dict]]:
        ids = list(node_ids)
        out: Dict[str, List[Dict]] = {i: [] for i in ids}
        conn = self._get_conn()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for n, u, l, w in conn.execute(
                    "SELECT node_id, unit_id, label_ok, wording_ok FROM "
                    f"node_provenance WHERE node_id IN ({marks})", chunk):
                out[n].append({"unit_id": u, "label_ok": bool(l),
                               "wording_ok": bool(w)})
        return out

    @_locked
    def nodes_of_unit(self, unit_id: str) -> List[Dict]:
        conn = self._get_conn()
        return [
            {"node_id": n, "label_ok": bool(l), "wording_ok": bool(w)}
            for n, l, w in conn.execute(
                "SELECT node_id, label_ok, wording_ok FROM node_provenance "
                "WHERE unit_id = ?", (unit_id,))]

    @_locked
    def delete_provenance(self, node_id: str,
                          unit_ids: Optional[Iterable[str]] = None) -> int:
        """Drop a node's ownership rows (all of them, or only these units')."""
        conn = self._get_conn()
        if unit_ids is None:
            cur = conn.execute(
                "DELETE FROM node_provenance WHERE node_id = ?", (node_id,))
        else:
            units = list(unit_ids)
            if not units:
                return 0
            cur = conn.execute(
                "DELETE FROM node_provenance WHERE node_id = ? AND unit_id IN (%s)"
                % ",".join("?" * len(units)), [node_id, *units])
        self._commit(conn)
        return cur.rowcount

    # ── unit registry ─────────────────────────────────────

    @_locked
    def register_unit(self, unit_id: str, path: Optional[str] = None,
                      path_keys: Optional[List[str]] = None) -> None:
        """Note that a stable unit was captured (and where its file is).
        Keys accumulate: a file seen at two paths keeps both."""
        conn = self._get_conn()
        now = _now()
        row = conn.execute(
            "SELECT path_keys FROM capture_units WHERE unit_id = ?",
            (unit_id,)).fetchone()
        keys = list(path_keys or [])
        if row is not None:
            try:
                keys = sorted(set(json.loads(row[0] or "[]")) | set(keys))
            except ValueError:
                pass
            conn.execute(
                "UPDATE capture_units SET last_seen = ?, "
                "path = COALESCE(?, path), path_keys = ? WHERE unit_id = ?",
                (now, path, json.dumps(keys), unit_id))
        else:
            conn.execute(
                "INSERT INTO capture_units (unit_id, path, path_keys, "
                "first_seen, last_seen) VALUES (?,?,?,?,?)",
                (unit_id, path, json.dumps(sorted(set(keys))), now, now))
        self._commit(conn)

    @_locked
    def get_unit(self, unit_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT unit_id, path, path_keys, first_seen, last_seen "
            "FROM capture_units WHERE unit_id = ?", (unit_id,)).fetchone()
        if row is None:
            return None
        return {"unit_id": row[0], "path": row[1],
                "path_keys": json.loads(row[2] or "[]"),
                "first_seen": row[3], "last_seen": row[4]}

    # ── suppression records ───────────────────────────────

    @staticmethod
    def _row_to_suppression(r) -> Dict:
        return {
            "id": r[0], "unit_id": r[1], "kind": r[2], "path": r[3],
            "path_keys": json.loads(r[4] or "[]"),
            "content_fps": json.loads(r[5] or "[]"),
            "wording_fps": json.loads(r[6] or "[]"),
            "file_fp": r[7], "created_at": r[8], "actor": r[9],
            "status": r[10], "lifted_at": r[11], "lifted_by": r[12],
            "sentence_fps": json.loads(r[13] or "[]"),
            "short_fps": json.loads(r[14] or "[]"),
            "signature_fps": json.loads(r[15] or "[]"),
            "scope": r[16] or "origin",
        }

    _SUPPRESSION_COLS = (
        "id, unit_id, kind, path, path_keys, content_fps, wording_fps, "
        "file_fp, created_at, actor, status, lifted_at, lifted_by, sentence_fps, short_fps, "
        "signature_fps, scope")

    @_locked
    def add_suppression(
        self, unit_id: str, kind: str, path: Optional[str],
        path_keys: List[str], content_fps: List[str],
        wording_fps: List[str], file_fp: Optional[str], actor: str,
        sentence_fps: Optional[List[str]] = None,
        short_fps: Optional[List[str]] = None,
        signature_fps: Optional[List[str]] = None,
        scope: str = "origin",
    ) -> int:
        """Write (or extend) the active record for a unit. A second forget of
        the same unit merges its fingerprints into the one record."""
        if kind not in ("file", "permanent", "wording"):
            raise ValueError(
                f"suppression kind must be file, permanent or wording, got {kind!r}")
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT {self._SUPPRESSION_COLS} FROM suppressions "
            "WHERE unit_id = ? AND status = 'active'", (unit_id,)).fetchone()
        if row is not None:
            cur = self._row_to_suppression(row)
            conn.execute(
                "UPDATE suppressions SET kind = ?, path = COALESCE(?, path), "
                "path_keys = ?, content_fps = ?, wording_fps = ?, "
                "sentence_fps = ?, short_fps = ?, signature_fps = ?, "
                "file_fp = COALESCE(?, file_fp) WHERE id = ?",
                (kind, path,
                 json.dumps(sorted(set(cur["path_keys"]) | set(path_keys))),
                 json.dumps(sorted(set(cur["content_fps"]) | set(content_fps))),
                 json.dumps(sorted(set(cur["wording_fps"]) | set(wording_fps))),
                 json.dumps(sorted(set(cur["sentence_fps"]) | set(sentence_fps or []))),
                 json.dumps(sorted(set(cur["short_fps"]) | set(short_fps or []))),
                 json.dumps(sorted(set(cur["signature_fps"]) | set(signature_fps or []))),
                 file_fp, cur["id"]))
            self._commit(conn)
            return cur["id"]
        cur = conn.execute(
            "INSERT INTO suppressions (unit_id, kind, path, path_keys, "
            "content_fps, wording_fps, sentence_fps, short_fps, signature_fps, "
            "scope, file_fp, created_at, actor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (unit_id, kind, path, json.dumps(sorted(set(path_keys))),
             json.dumps(sorted(set(content_fps))),
             json.dumps(sorted(set(wording_fps))),
             json.dumps(sorted(set(sentence_fps or []))),
             json.dumps(sorted(set(short_fps or []))),
             json.dumps(sorted(set(signature_fps or []))), scope,
             file_fp, _now(), actor))
        self._commit(conn)
        return cur.lastrowid

    @_locked
    def list_suppressions(self, status: Optional[str] = "active",
                          include_wording: bool = False) -> List[Dict]:
        """Suppression records. The standing-denylist entries (kind
        'wording') are left out unless asked for: they are not sources, and
        a restore reaches them through the units they cover."""
        conn = self._get_conn()
        sql = f"SELECT {self._SUPPRESSION_COLS} FROM suppressions"
        where, params = [], []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if not include_wording:
            where.append("kind != 'wording'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
        return [self._row_to_suppression(r) for r in rows]

    @_locked
    def count_suppressions(self, status: str = "active") -> int:
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM suppressions WHERE status = ?",
            (status,)).fetchone()[0]

    @_locked
    def lift_suppression(self, suppression_id: int, by: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE suppressions SET status = 'lifted', lifted_at = ?, "
            "lifted_by = ? WHERE id = ? AND status = 'active'",
            (_now(), by, suppression_id))
        self._commit(conn)
        return cur.rowcount > 0

    # ── storage hygiene ───────────────────────────────────

    @_locked
    def scrub_storage(self) -> Dict:
        """After a forget: make freed pages overwrite with zeros, truncate the
        WAL, and rebuild the file so the freelist holds nothing old.

        Returns what actually happened, never what was hoped for. Refuses to
        run inside a transaction (VACUUM cannot)."""
        info = {"secure_delete": False, "wal_truncated": False,
                "vacuumed": False, "note": ""}
        if self._txn_depth > 0:
            info["note"] = "skipped: a transaction is open"
            return info
        conn = self._get_conn()
        conn.commit()
        conn.execute("PRAGMA secure_delete = ON")
        info["secure_delete"] = conn.execute(
            "PRAGMA secure_delete").fetchone()[0] in (1, 2)
        notes = []

        def checkpoint() -> bool:
            try:
                row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                return bool(row) and row[0] == 0
            except sqlite3.Error as e:
                notes.append(f"checkpoint: {e}")
                return False

        checkpoint()
        try:
            conn.execute("VACUUM")
            info["vacuumed"] = True
        except sqlite3.OperationalError as e:
            # A vec0 table needs its module loaded to be rebuilt.
            try:
                self._load_vec_extension(conn)
                conn.execute("VACUUM")
                info["vacuumed"] = True
            except Exception as e2:  # noqa: BLE001
                notes.append(f"vacuum: {e2 or e}")
        info["wal_truncated"] = checkpoint()
        info["note"] = "; ".join(notes)
        return info

    @staticmethod
    def _load_vec_extension(conn: sqlite3.Connection) -> None:
        import sqlite_vec  # noqa: PLC0415 - optional
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
