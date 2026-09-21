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

from revien.graph.store import GraphStore


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _origin_backfilled_count(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE origin_runtime IS NOT NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0  # no nodes table yet — fresh/nonexistent db


def migrate(db_path: str = "revien.db") -> dict:
    """Run the origin-layer migration against ``db_path``.

    Thin wrapper: the ALTER/index/backfill work lives in exactly one place,
    ``GraphStore._migrate_add_origin_columns`` — opening (and closing) a
    ``GraphStore`` against ``db_path`` runs it as part of the normal
    connect-time migration chain. This function only observes the
    before/after state to report the same summary shape callers already
    depend on.

    ``PRAGMA user_version`` is forced to 0 first: the connect-time guard in
    ``_migrate_add_origin_columns`` skips its backfill scan once the db is
    already marked version >= 3 (a per-connection performance optimization —
    most opens have nothing left to backfill). A deliberately-invoked
    standalone migration has no such performance concern and must always
    retry — e.g. a row NULLed out-of-band after the marker was set — so it
    forces the scan every time. The guard re-raises the marker to 3 once
    the (possibly no-op) backfill pass completes, restoring it.

    Returns a summary dict:
        {"columns_added": [...], "backfilled": int}
    """
    conn = sqlite3.connect(db_path)
    try:
        before_cols = _columns(conn, "nodes")
        before_backfilled = _origin_backfilled_count(conn)
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    finally:
        conn.close()

    store = GraphStore(db_path=db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        after_cols = _columns(conn, "nodes")
        after_backfilled = _origin_backfilled_count(conn)
    finally:
        conn.close()

    origin_cols = ("origin_runtime", "origin_source", "project_key", "session_key")
    columns_added = [c for c in origin_cols if c not in before_cols and c in after_cols]
    return {
        "columns_added": columns_added,
        "backfilled": after_backfilled - before_backfilled,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(
        f"Migration 003 complete on {target}: "
        f"columns_added={summary['columns_added']}, backfilled={summary['backfilled']}."
    )
