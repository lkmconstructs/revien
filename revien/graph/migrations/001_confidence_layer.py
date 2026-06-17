"""
Migration 001 — Confidence Layer.

Adds the confidence-layer columns to an EXISTING ``revien.db`` and backfills
pre-confidence rows so old data behaves as ground truth:

    nodes:
        source_type        TEXT     -> backfilled to 'extracted'
        confidence         REAL     -> backfilled to 1.0
        pinned             INTEGER  -> backfilled to 0 (False)
        confidence_set_at  TEXT     -> backfilled to migration time
        confidence_set_by  TEXT     -> backfilled to '' (then 'migration_001')
        source_context     TEXT     -> backfilled to 'migration_001_backfill'
        last_referenced    TEXT     -> left NULL

    edges:
        confidence         REAL     -> backfilled to 1.0
        confidence_set_at  TEXT     -> backfilled to migration time
        confidence_set_by  TEXT     -> backfilled to '' (then 'migration_001')
        source_context     TEXT     -> backfilled to 'migration_001_backfill'

Backfill rationale: rows that existed before the confidence layer were written
without any inference step — they are treated as EXTRACTED ground truth with
confidence 1.0 and pinned=False, matching the leg-1 spec.

Idempotent: existing columns are skipped, and backfill only touches rows whose
confidence-layer fields are still NULL, so re-running is safe.

Usage:
    python -m revien.graph.migrations.001_confidence_layer [path/to/revien.db]

Defaults to ``revien.db`` in the current working directory.
"""

import sqlite3
import sys
from datetime import datetime, timezone


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate(db_path: str = "revien.db") -> dict:
    """Run the confidence-layer migration against ``db_path``.

    Returns a summary dict: {"nodes_backfilled": int, "edges_backfilled": int}.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        # ── Add node columns if missing ──────────────────────
        node_cols = _columns(conn, "nodes")
        if "source_type" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN source_type TEXT")
        if "confidence" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN confidence REAL")
        if "pinned" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN pinned INTEGER")
        if "confidence_set_at" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN confidence_set_at TEXT")
        if "confidence_set_by" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN confidence_set_by TEXT")
        if "source_context" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN source_context TEXT")
        if "last_referenced" not in node_cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN last_referenced TEXT")

        # ── Add edge columns if missing ──────────────────────
        edge_cols = _columns(conn, "edges")
        if "confidence" not in edge_cols:
            conn.execute("ALTER TABLE edges ADD COLUMN confidence REAL")
        if "confidence_set_at" not in edge_cols:
            conn.execute("ALTER TABLE edges ADD COLUMN confidence_set_at TEXT")
        if "confidence_set_by" not in edge_cols:
            conn.execute("ALTER TABLE edges ADD COLUMN confidence_set_by TEXT")
        if "source_context" not in edge_cols:
            conn.execute("ALTER TABLE edges ADD COLUMN source_context TEXT")

        # ── Backfill old rows (only NULL fields) ─────────────
        # Old rows = ground truth: source_type=EXTRACTED, confidence=1.0,
        # pinned=False.
        node_cur = conn.execute(
            """
            UPDATE nodes
            SET source_type       = COALESCE(source_type, 'extracted'),
                confidence        = COALESCE(confidence, 1.0),
                pinned            = COALESCE(pinned, 0),
                confidence_set_at = COALESCE(confidence_set_at, ?),
                confidence_set_by = COALESCE(NULLIF(confidence_set_by, ''), 'migration_001'),
                source_context    = COALESCE(NULLIF(source_context, ''), 'migration_001_backfill')
            WHERE source_type IS NULL
               OR confidence IS NULL
               OR pinned IS NULL
               OR confidence_set_at IS NULL
            """,
            (now_iso,),
        )
        nodes_backfilled = node_cur.rowcount

        edge_cur = conn.execute(
            """
            UPDATE edges
            SET confidence        = COALESCE(confidence, 1.0),
                confidence_set_at = COALESCE(confidence_set_at, ?),
                confidence_set_by = COALESCE(NULLIF(confidence_set_by, ''), 'migration_001'),
                source_context    = COALESCE(NULLIF(source_context, ''), 'migration_001_backfill')
            WHERE confidence IS NULL
               OR confidence_set_at IS NULL
            """,
            (now_iso,),
        )
        edges_backfilled = edge_cur.rowcount

        # ── Confidence indexes ───────────────────────────────
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence)")

        conn.commit()
        return {
            "nodes_backfilled": nodes_backfilled,
            "edges_backfilled": edges_backfilled,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "revien.db"
    summary = migrate(target)
    print(
        f"Migration 001 complete on {target}: "
        f"{summary['nodes_backfilled']} node rows, "
        f"{summary['edges_backfilled']} edge rows backfilled."
    )
