"""
Migration 007 - exact lane indexes.

Brings an EXISTING ``revien.db`` (any version up to migration 006) up to the
indexes the opt-in exact lane (REVIEN_EXACT) reads through:

    idx_nodes_label_lower  ON nodes(lower(label))
    idx_nodes_event_time   ON nodes(event_time_start, event_time_end)
    idx_nodes_recorded_at  ON nodes(recorded_at)

Indexes only. No row is read, rewritten or deleted, and recall does not change
unless REVIEN_EXACT is on. ``PRAGMA user_version`` is raised to 7 (only if it
was lower). Idempotent: an index that exists is left alone, and a second run
changes nothing. The DDL lives in GraphStore and runs as part of the normal
connect-time chain, exactly as for 003-006.

Usage:
    python -m revien.graph.migrations.007_exact_lane_indexes [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys

from revien.graph.store import GraphStore

_INDEXES = ("idx_nodes_label_lower", "idx_nodes_event_time", "idx_nodes_recorded_at")


def _present(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name IN (?,?,?)",
        _INDEXES)}


def migrate(db_path: str = "revien.db") -> dict:
    """Run the exact-lane index migration against ``db_path``.

    user_version is lowered below 7 first so the connect-time guard cannot
    skip a deliberately invoked run; the guard re-raises it to 7. Returns
    {"created": [indexes added by this run]}."""
    conn = sqlite3.connect(db_path)
    try:
        before = _present(conn)
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 7:
            conn.execute("PRAGMA user_version = 6")
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
    print(f"Migration 007 complete on {target}: created={summary['created']}.")
