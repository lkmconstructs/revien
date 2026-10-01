"""
Migration 004 — recorded_at_source backfill.

Rows ingested before commit 4829c75 carry no metadata["recorded_at_source"],
so their recorded_at (a file mtime for Claude Code / Codex / file-watcher /
Obsidian rows) could be rendered as "when it was said". This labels every row
that has a recorded_at and no source, using derive_recorded_at_source
(revien/graph/origin.py) over the row's origin columns. Rows that derive None
stay unlabeled and render undated.

Idempotent: only rows lacking the key are touched; a second run reports 0.

Usage:
    python -m revien.graph.migrations.004_recorded_at_source [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys

from revien.graph.store import GraphStore


def _labeled_count(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM nodes "
            "WHERE json_extract(metadata, '$.recorded_at_source') IS NOT NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0  # no nodes table yet


def migrate(db_path: str = "revien.db") -> dict:
    """Thin wrapper over GraphStore._migrate_backfill_recorded_at_source (the
    one place the work lives). user_version is lowered below 4 first so the
    connect-time guard cannot skip a deliberately invoked run; the guard
    re-raises it to 4. Returns {"backfilled": int}."""
    conn = sqlite3.connect(db_path)
    try:
        before = _labeled_count(conn)
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 4:
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
    finally:
        conn.close()

    GraphStore(db_path=db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        after = _labeled_count(conn)
    finally:
        conn.close()
    return {"backfilled": after - before}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(f"Migration 004 complete on {target}: backfilled={summary['backfilled']}.")
