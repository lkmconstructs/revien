"""
Migration 003 — Origin Layer (WS0).

Brings an EXISTING ``revien.db`` up to the origin core:

    nodes:
        origin_runtime  TEXT (nullable) -> added
        origin_source   TEXT (nullable) -> added
        project_key     TEXT (nullable) -> added
        session_key     TEXT (nullable) -> added

Backfill: every existing row whose source_id derive_origin (see
revien/graph/origin.py) recognizes gets its origin columns filled in from
that source_id. Rows with an unrecognized or empty source_id stay NULL — an
honest "unknown", never a guess.

Idempotent: the column adds and index creates are guarded, and the backfill
only touches rows still at origin_runtime IS NULL, so a second run reports
0 backfilled.

Usage:
    python -m revien.graph.migrations.003_origin_layer [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys

from revien.graph.origin import derive_origin


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(db_path: str = "revien.db") -> dict:
    """Run the origin-layer migration against ``db_path``.

    Returns a summary dict:
        {"columns_added": [...], "backfilled": int}
    """
    conn = sqlite3.connect(db_path)
    try:
        node_cols = _columns(conn, "nodes")
        columns_added = []
        for col in ("origin_runtime", "origin_source", "project_key", "session_key"):
            if col not in node_cols:
                conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} TEXT")
                columns_added.append(col)

        # G6: composite indexes — see the matching comment in
        # store.py's _migrate_add_origin_columns for why (list_nodes
        # filters AND orders by created_at DESC in the same query; a
        # single-column index can't satisfy both, so SQLite falls back to
        # a temp B-tree sort). Old single-column index names are dropped
        # first so a DB that already ran the pre-composite version of this
        # migration converges on the composite instead of keeping both.
        # ``created_at`` is guarded (re-read after the ALTERs above,
        # never assumed) because a degenerate/hand-built nodes table
        # (no created_at column at all) can't back a created_at-ordered
        # index — that table falls back to a single-column index instead
        # of the migration raising on it.
        has_created_at = "created_at" in _columns(conn, "nodes")
        conn.execute("DROP INDEX IF EXISTS idx_nodes_origin_runtime")
        conn.execute("DROP INDEX IF EXISTS idx_nodes_project")
        if has_created_at:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_origin_runtime "
                "ON nodes(origin_runtime, created_at DESC)"
            )
        else:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_origin_runtime "
                "ON nodes(origin_runtime)"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_origin_source ON nodes(origin_source)"
        )
        if has_created_at:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_project "
                "ON nodes(project_key, created_at DESC)"
            )
        else:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_project ON nodes(project_key)"
            )
        conn.commit()

        rows = conn.execute(
            "SELECT node_id, source_id FROM nodes "
            "WHERE origin_runtime IS NULL AND source_id != ''"
        ).fetchall()
        backfilled = 0
        for node_id, source_id in rows:
            origin = derive_origin(source_id)
            if origin.runtime is None:
                continue
            conn.execute(
                "UPDATE nodes SET origin_runtime = ?, origin_source = ?, "
                "project_key = ?, session_key = ? WHERE node_id = ?",
                (origin.runtime, origin.source, origin.project, origin.session, node_id),
            )
            backfilled += 1
        # Mirrors store.py's _migrate_add_origin_columns marker: once this
        # backfill has run, later connects (via GraphStore) skip the
        # NULL-scan outright. Only raise it, never lower it.
        current_version = conn.execute("PRAGMA user_version").fetchone()[0]
        if current_version < 3:
            conn.execute("PRAGMA user_version = 3")
        conn.commit()

        return {"columns_added": columns_added, "backfilled": backfilled}
    finally:
        conn.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(
        f"Migration 003 complete on {target}: "
        f"columns_added={summary['columns_added']}, backfilled={summary['backfilled']}."
    )
