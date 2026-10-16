"""G2: legacy db (no origin columns) opens, backfills every known source_id
convention, and is idempotent.

CHECK: python scripts/gates/check_origin_backfill.py
EXPECT: origin backfill verification passed
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.origin import derive_origin  # noqa: E402
from revien.graph.schema import Node, NodeType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402

# Every documented source_id convention, plus adversarial variants.
CASES = [
    "claude-code:fernweh:sess-1",
    "codex:fernweh:sess-2",
    "hermes",
    "openai:conversation:c-9",
    "vault:notes/a.md#slug",
    "file:x.txt",
    "api:https://x/y",
    "ollama_history",
    "ollama_chat",
    "langchain",
    # adversarial
    "claude-code:proj:with:colons:sess",
    "codex::",
    "vault:a#b#c",
    "openai:conversation:",
    "",
]


def fail(msg):
    print(f"check_origin_backfill: {msg}", file=sys.stderr)
    sys.exit(1)


def null_and_reset(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE nodes SET origin_runtime=NULL, origin_source=NULL, "
        "project_key=NULL, session_key=NULL"
    )
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()


def dump(db_path):
    conn = sqlite3.connect(db_path)
    rows = {
        row[0]: row[1:]
        for row in conn.execute(
            "SELECT source_id, origin_runtime, origin_source, project_key, "
            "session_key FROM nodes"
        ).fetchall()
    }
    conn.close()
    return rows


def main():
    tmpdir = tempfile.mkdtemp()
    base_path = os.path.join(tmpdir, "base.db")

    store = GraphStore(db_path=base_path)
    recorded = []  # (source_id, node_id)
    for src in CASES:
        n = store.add_node(Node(
            node_type=NodeType.CONTEXT,
            label=f"legacy: {src or '(empty)'}",
            content="legacy content",
            source_id=src,
        ))
        recorded.append((src, n.node_id))
    store.close()

    null_and_reset(base_path)

    # Two independent copies: one for the standalone migration, one for the
    # GraphStore auto-migrate (_ensure_db) path.
    standalone_path = os.path.join(tmpdir, "standalone.db")
    autopath = os.path.join(tmpdir, "auto.db")
    shutil.copy(base_path, standalone_path)
    shutil.copy(base_path, autopath)

    # ── standalone migration 003 ────────────────────────────────────────
    import importlib
    migration_mod = importlib.import_module("revien.graph.migrations.003_origin_layer")
    summary1 = migration_mod.migrate(standalone_path)

    expected_recognized = sum(1 for src, _ in recorded if derive_origin(src).runtime is not None)
    if summary1["backfilled"] != expected_recognized:
        fail(
            f"standalone migrate() backfilled {summary1['backfilled']}, "
            f"expected {expected_recognized}"
        )

    # Verify every recognizable row derived correctly.
    standalone_dump = dump(standalone_path)
    for src, _node_id in recorded:
        expected = derive_origin(src)
        got = standalone_dump[src]
        if tuple(got) != tuple(expected):
            fail(f"source_id={src!r}: expected {tuple(expected)}, got {tuple(got)}")

    # Idempotent: second run backfills 0.
    summary1b = migration_mod.migrate(standalone_path)
    if summary1b["backfilled"] != 0:
        fail(f"second migrate() run backfilled {summary1b['backfilled']}, expected 0")

    # user_version == 4 after (chain ends at the recorded_at_source backfill).
    conn = sqlite3.connect(standalone_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    if version != 4:
        fail(f"PRAGMA user_version after migrate() == {version}, expected 4")

    # ── GraphStore auto-migrate path (_ensure_db) ───────────────────────
    store2 = GraphStore(db_path=autopath)
    store2.close()
    auto_dump = dump(autopath)

    if auto_dump != standalone_dump:
        diff = {
            k: (standalone_dump[k], auto_dump[k])
            for k in standalone_dump
            if standalone_dump[k] != auto_dump[k]
        }
        fail(f"standalone migration and GraphStore auto-migrate disagree: {diff}")

    conn = sqlite3.connect(autopath)
    auto_version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    if auto_version != 4:
        fail(f"GraphStore auto-migrate PRAGMA user_version == {auto_version}, expected 4")

    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    print("origin backfill verification passed")


if __name__ == "__main__":
    main()
