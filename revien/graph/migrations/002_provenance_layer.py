"""
Migration 002 — Provenance Layer (leg 6a).

Brings an EXISTING ``revien.db`` up to the provenance core:

    nodes:
        invalidated_at  TEXT (nullable) -> added, backfills NULL on old rows
                                            (every existing node is LIVE).

    audit_log (new table, append-only):
        id           INTEGER PRIMARY KEY AUTOINCREMENT
        node_id      TEXT NOT NULL
        op           TEXT NOT NULL        -- create|reinforce|correct|decay|access|invalidate|update
        actor        TEXT DEFAULT ''
        ts           TEXT NOT NULL
        before_json  TEXT                 -- node snapshot before the op (nullable)
        after_json   TEXT                 -- node snapshot after the op (nullable)

Design rule (leg 6a): nothing auto-deletes; provenance is append-only. This
migration is purely additive — it never drops, rewrites, or removes data.

Backfill rationale: ``invalidated_at`` is NULL for all pre-6a rows because no
node was ever marked stale before this layer existed. NULL == live, which is
exactly the desired default. No audit rows are synthesized for historical ops
(there is no reliable record of them); the trail begins at migration time.

Idempotent: the column add and table create are guarded, so re-running is safe.

Usage:
    python -m revien.graph.migrations.002_provenance_layer [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(db_path: str = "revien.db") -> dict:
    """Run the provenance-layer migration against ``db_path``.

    Returns a summary dict:
        {"invalidated_at_added": bool, "audit_log_created": bool}
    """
    conn = sqlite3.connect(db_path)
    try:
        # ── nodes.invalidated_at (nullable; old rows stay NULL = live) ──
        node_cols = _columns(conn, "nodes")
        invalidated_added = False
        if "invalidated_at" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN invalidated_at TEXT")
            invalidated_added = True

        # ── audit_log (append-only provenance trail) ─────────
        existed = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
        ).fetchone()
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

        conn.commit()
        return {
            "invalidated_at_added": invalidated_added,
            "audit_log_created": existed is None,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(
        f"Migration 002 complete on {target}: "
        f"invalidated_at_added={summary['invalidated_at_added']}, "
        f"audit_log_created={summary['audit_log_created']}."
    )
