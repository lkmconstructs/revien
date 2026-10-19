"""
Migration 006 - forget layer.

Brings an EXISTING ``revien.db`` (created by v0.2.x, v0.3.0, or by any build up
to migration 005) up to the forget layer (docs/FORGET_SPEC.md):

    node_provenance   table + idx_node_provenance_unit  -> created
    capture_units     table                              -> created
    suppressions      table + idx_suppressions_unit      -> created

and backfills ownership for the nodes already there WITHOUT GUESSING:

* a context or skill node stamped with an ingest key is provably its unit's
  own record (it is never merged into) and gets a proven ownership row;
* every other node gets an ``unproven:`` marker row. It is kept, it is listed
  when a forget touches its likely unit, and it is never deleted on a guess.

Nothing is deleted or rewritten. ``PRAGMA user_version`` is raised to 6 (only
if it was lower). The keyed-fingerprint key is a file in the Revien home
(``$REVIEN_HOME`` or ``~/.revien/forget.key``), created here owner-only if it
does not exist; it is never written to the database.

Idempotent: every object is created only if missing and the backfill only
touches nodes with no ownership row, so a second run changes nothing. The DDL
and the backfill live in GraphStore (graph/forget_store.py), run as part of the
normal connect-time chain, exactly as for 003-005.

Usage:
    python -m revien.graph.migrations.006_forget_layer [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys

from revien.graph.store import GraphStore

_OBJECTS = ("node_provenance", "capture_units", "suppressions")


def _present(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?,?)",
        _OBJECTS)}


def _provenance_rows(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM node_provenance").fetchone()[0]
    except sqlite3.Error:
        return 0


def migrate(db_path: str = "revien.db") -> dict:
    """Run the forget-layer migration against ``db_path``.

    user_version is lowered below 6 first so the connect-time guard cannot
    skip a deliberately invoked run; the guard re-raises it to 6. Returns
    {"created": [tables added by this run], "backfilled": rows added}."""
    conn = sqlite3.connect(db_path)
    try:
        before = _present(conn)
        rows_before = _provenance_rows(conn)
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 6:
            conn.execute("PRAGMA user_version = 5")
            conn.commit()
    finally:
        conn.close()

    GraphStore(db_path=db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        after = _present(conn)
        added = _provenance_rows(conn) - rows_before
    finally:
        conn.close()

    # The key for keyed fingerprints. Made here so it exists before the first
    # forget; a missing key is also made on first use.
    from revien.forget.keys import load_key
    load_key(create=True)
    return {"created": sorted(after - before), "backfilled": added}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(f"Migration 006 complete on {target}: created={summary['created']}, "
          f"ownership rows backfilled={summary['backfilled']}.")
