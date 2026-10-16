"""
Migration 005 - usage layer.

Brings an EXISTING ``revien.db`` (created by v0.3.0, or by any build up to
migration 004) up to the usage layer:

    use_judgments       table + idx_use_judgments_node         -> created
    recall_receipts     table + idx_recall_receipts_status     -> created
    store_meta          table (key/value: database id, daemon heartbeat)
                                                                -> created
    sync_cursors.generation  INTEGER NOT NULL DEFAULT 0         -> added

Existing rows are never rewritten: every existing sync cursor gets
generation 0, and the new tables start empty. ``PRAGMA user_version`` is
raised to 5 (only if it was lower).

Idempotent: every object is created only if missing, so a second run changes
nothing. This is the only place the usage layer's DDL lives; the work is
GraphStore._migrate_add_usage_layer (the one place the work lives, as for
003 and 004), run as part of the normal connect-time chain.

Usage:
    python -m revien.graph.migrations.005_usage_layer [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys

from revien.graph.store import GraphStore

_OBJECTS = ("use_judgments", "recall_receipts", "store_meta")


def _present(conn: sqlite3.Connection) -> set:
    found = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?,?)",
        _OBJECTS)}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sync_cursors)")}
    if "generation" in cols:
        found.add("sync_cursors.generation")
    return found


def migrate(db_path: str = "revien.db") -> dict:
    """Run the usage-layer migration against ``db_path``.

    user_version is lowered below 5 first so the connect-time guard cannot
    skip a deliberately invoked run; the guard re-raises it to 5. Returns
    {"created": [objects added by this run]}.
    """
    conn = sqlite3.connect(db_path)
    try:
        before = _present(conn)
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 5:
            conn.execute("PRAGMA user_version = 4")
            conn.commit()
    finally:
        conn.close()

    GraphStore(db_path=db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        after = _present(conn)
    finally:
        conn.close()
    return {"created": sorted(after - before)}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(f"Migration 005 complete on {target}: created={summary['created']}.")
