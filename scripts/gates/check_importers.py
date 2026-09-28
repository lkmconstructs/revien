"""G13: ChatGPT, Claude, and Readwise imports go through the pipeline,
honor the deny list, stamp historical recorded_at and import origin, and
are idempotent on re-run.

CHECK: python scripts/gates/check_importers.py
EXPECT: importer verification passed
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.schema import EdgeType, NodeType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.ingestion.pipeline import IngestionPipeline  # noqa: E402
from revien.importers import chatgpt as chatgpt_importer  # noqa: E402
from revien.importers import claude as claude_importer  # noqa: E402
from revien.importers import readwise as readwise_importer  # noqa: E402
from revien.importers.base import run_import  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def fail(msg):
    print(f"check_importers: {msg}", file=sys.stderr)
    sys.exit(1)


def zip_json(path: Path, data) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(data))


# ── ChatGPT export: 2 conversations, epoch create_time in 2024 ─────────────

def chatgpt_conversation(conv_id: str, title: str, epoch: float, reply: str):
    return {
        "id": conv_id,
        "title": title,
        "create_time": epoch,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["u1"], "message": None},
            "u1": {
                "id": "u1", "parent": "root", "children": ["a1"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Mara asks a question."]},
                    "create_time": epoch,
                },
            },
            "a1": {
                "id": "a1", "parent": "u1", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [reply]},
                    "create_time": epoch + 10.0,
                },
            },
        },
        "current_node": "a1",
    }


CHATGPT_EPOCH_1 = 1704067200.0  # 2024-01-01T00:00:00Z
CHATGPT_EPOCH_2 = 1719792000.0  # 2024-07-01T00:00:00Z


def build_chatgpt_zip(tmpdir: Path) -> Path:
    zpath = tmpdir / "chatgpt.zip"
    zip_json(zpath, [
        chatgpt_conversation("conv-mara-1", "Fernweh-Core planning", CHATGPT_EPOCH_1,
                              "Theo and Mara settle on the rollout plan."),
        chatgpt_conversation("conv-mara-2", "Fernweh-Core follow-up", CHATGPT_EPOCH_2,
                              "Sam confirms the follow-up steps."),
    ])
    return zpath


# ── Claude export: 1 conversation, content-blocks-only message, ISO 'Z' ────

def build_claude_zip(tmpdir: Path) -> Path:
    zpath = tmpdir / "claude.zip"
    conv = {
        "uuid": "conv-theo-1",
        "name": "Sam's onboarding notes",
        "created_at": "2024-03-05T12:00:00Z",
        "chat_messages": [
            {
                "uuid": "m1", "sender": "human",
                "text": "What did we decide about Fernweh-Core's rollout?",
                "created_at": "2024-03-05T12:00:00Z",
            },
            {
                # content-blocks-only: no usable top-level `text`.
                "uuid": "m2", "sender": "assistant", "text": "",
                "content": [{"type": "text", "text": "We staged it behind a flag for Sam's team."}],
                "created_at": "2024-03-05T12:00:10Z",
            },
        ],
    }
    zip_json(zpath, [conv])
    return zpath


# ── Readwise CSV: 2 highlights ──────────────────────────────────────────────

def build_readwise_csv(tmpdir: Path) -> Path:
    import csv

    fieldnames = [
        "Highlight", "Book Title", "Book Author", "Note", "Tags",
        "Location", "Highlighted at", "Document tags",
    ]
    rows = [
        {
            "Highlight": "Mara keeps the architecture immutable on purpose.",
            "Book Title": "Fernweh-Core Field Notes",
            "Book Author": "Theo",
            "Note": "Worth re-reading.",
            "Tags": "#architecture, #memory",
            "Location": "142",
            "Highlighted at": "2024-05-04 10:32:00",  # naive
            "Document tags": "",
        },
        {
            # Optional columns missing (Note, Author, Tags, Location, Document tags).
            "Highlight": "Sovereignty of Choice matters most in the sanctum.",
            "Book Title": "Fernweh-Core Field Notes",
            "Book Author": "",
            "Note": "",
            "Tags": "",
            "Location": "",
            "Highlighted at": "2024-06-01 09:00:00",  # naive
            "Document tags": "",
        },
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_path = tmpdir / "readwise.csv"
    csv_path.write_text(buf.getvalue(), encoding="utf-8")
    return csv_path


def fresh_store(tmpdir: Path, name: str):
    db_path = str(tmpdir / name)
    return GraphStore(db_path=db_path), db_path


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="revien-gate13-"))

    chatgpt_zip = build_chatgpt_zip(tmpdir)
    claude_zip = build_claude_zip(tmpdir)
    readwise_csv = build_readwise_csv(tmpdir)

    # ── main run: origin, recorded_at, CONTEXT nodes, declared edge ────────
    store, db_path = fresh_store(tmpdir, "main.db")
    pipeline = IngestionPipeline(store)

    chatgpt_units = list(chatgpt_importer.iter_units(str(chatgpt_zip)))
    claude_units = list(claude_importer.iter_units(str(claude_zip)))
    readwise_units = list(readwise_importer.iter_units(str(readwise_csv)))

    if len(chatgpt_units) != 2:
        fail(f"expected 2 chatgpt units, got {len(chatgpt_units)}")
    if len(claude_units) != 1:
        fail(f"expected 1 claude unit, got {len(claude_units)}")
    if len(readwise_units) != 2:
        fail(f"expected 2 readwise units, got {len(readwise_units)}")

    all_the_units = chatgpt_units + claude_units + readwise_units
    report1 = run_import(all_the_units, pipeline, dry_run=False)

    if report1.units_seen != 5:
        fail(f"units_seen == {report1.units_seen}, expected 5")
    if report1.units_ingested != 5:
        fail(f"units_ingested == {report1.units_ingested}, expected 5 on first import")

    expected_runtimes = {"chatgpt", "claude", "readwise"}
    all_nodes = store.list_nodes(limit=10000)
    if not all_nodes:
        fail("no nodes were created by the import")

    now = datetime.now(timezone.utc)
    for node in all_nodes:
        if node.origin_source != "import":
            fail(f"node {node.node_id} ({node.label!r}) origin_source == "
                 f"{node.origin_source!r}, expected 'import'")
        if node.origin_runtime not in expected_runtimes:
            fail(f"node {node.node_id} origin_runtime == {node.origin_runtime!r}, "
                 f"expected one of {expected_runtimes}")
        if node.recorded_at is None:
            fail(f"node {node.node_id} has no recorded_at")
        if node.recorded_at.year != 2024:
            fail(f"node {node.node_id} recorded_at.year == {node.recorded_at.year}, "
                 "expected 2024 (fixture historical time)")
        if node.recorded_at >= now:
            fail(f"node {node.node_id} recorded_at is not in the past: {node.recorded_at}")

    # CONTEXT nodes exist per unit (5 units -> 5 CONTEXT nodes minimum).
    ctx_nodes = [n for n in all_nodes if n.node_type == NodeType.CONTEXT]
    if len(ctx_nodes) < 5:
        fail(f"expected at least 5 CONTEXT nodes (one per unit), got {len(ctx_nodes)}")

    # Readwise unit has a RELATED_TO/declared edge to an ENTITY labeled with
    # the book title.
    entities = store.list_nodes(node_type=NodeType.ENTITY, limit=1000)
    book = next((n for n in entities if n.label == "Fernweh-Core Field Notes"), None)
    if book is None:
        fail("expected an ENTITY node labeled 'Fernweh-Core Field Notes'")

    readwise_ctx = [n for n in ctx_nodes if n.source_id.startswith("readwise:")]
    if not readwise_ctx:
        fail("no readwise CONTEXT node found")
    found_edge = False
    for ctx in readwise_ctx:
        for edge in store.get_edges_for_node(ctx.node_id):
            if edge.edge_type == EdgeType.RELATED_TO and edge.target_node_id == book.node_id:
                found_edge = True
    if not found_edge:
        fail("no RELATED_TO edge from a readwise CONTEXT node to the book ENTITY")

    nodes_after_first = store.count_nodes()
    edges_after_first = store.count_edges()

    # ── re-run all three imports: 0 nodes, 0 edges, units_unchanged == seen ──
    chatgpt_units2 = list(chatgpt_importer.iter_units(str(chatgpt_zip)))
    claude_units2 = list(claude_importer.iter_units(str(claude_zip)))
    readwise_units2 = list(readwise_importer.iter_units(str(readwise_csv)))
    report2 = run_import(
        chatgpt_units2 + claude_units2 + readwise_units2, pipeline, dry_run=False,
    )

    if report2.units_seen != 5:
        fail(f"second run units_seen == {report2.units_seen}, expected 5")
    if report2.units_unchanged != report2.units_seen:
        fail(f"second run units_unchanged == {report2.units_unchanged}, "
             f"expected == units_seen ({report2.units_seen})")
    if report2.nodes_created != 0:
        fail(f"second run nodes_created == {report2.nodes_created}, expected 0")
    if report2.edges_created != 0:
        fail(f"second run edges_created == {report2.edges_created}, expected 0")
    if store.count_nodes() != nodes_after_first:
        fail("re-running imports changed the node count")
    if store.count_edges() != edges_after_first:
        fail("re-running imports changed the edge count")

    store.close()

    # ── deny list: positive control first (without deny it is ingested) ────
    control_store, _ = fresh_store(tmpdir, "control.db")
    control_pipeline = IngestionPipeline(control_store)
    chatgpt_units_control = list(chatgpt_importer.iter_units(str(chatgpt_zip)))
    denied_source_id = chatgpt_units_control[0].source_id
    control_report = run_import(chatgpt_units_control, control_pipeline, dry_run=False)
    if control_report.units_denied != 0:
        fail("positive control: without REVIEN_INGEST_DENY set, a unit was denied")
    control_nodes = control_store.list_nodes(source_id=denied_source_id, limit=100)
    if not control_nodes:
        fail("positive control: the unit was NOT ingested without the deny list set "
             "(deny-list detector would be meaningless)")
    control_store.close()

    # Now with REVIEN_INGEST_DENY set to that source_id, a fresh import
    # reports units_denied == 1 and no node carries that source_id.
    os.environ["REVIEN_INGEST_DENY"] = denied_source_id
    try:
        deny_store, _ = fresh_store(tmpdir, "deny.db")
        deny_pipeline = IngestionPipeline(deny_store)
        chatgpt_units_deny = list(chatgpt_importer.iter_units(str(chatgpt_zip)))
        deny_report = run_import(chatgpt_units_deny, deny_pipeline, dry_run=False)
        if deny_report.units_denied != 1:
            fail(f"units_denied == {deny_report.units_denied}, expected 1")
        deny_nodes = deny_store.list_nodes(source_id=denied_source_id, limit=100)
        if deny_nodes:
            fail(f"a node still carries the denied source_id {denied_source_id!r}")
        deny_store.close()
    finally:
        del os.environ["REVIEN_INGEST_DENY"]

    # ── static check: no importer file imports a network client ────────────
    forbidden = ("import httpx", "import requests", "urllib.request", "import socket", "aiohttp")

    def scan_text(text: str):
        return [f for f in forbidden if f in text]

    # positive control: scanner detects a synthetic "+import httpx" string.
    control_hits = scan_text("some diff line\n+import httpx\nmore text")
    if "import httpx" not in control_hits:
        fail("positive control: scanner failed to detect a synthetic '+import httpx' line")

    importers_dir = REPO_ROOT / "revien" / "importers"
    violations = []
    for py_file in sorted(importers_dir.glob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        hits = scan_text(text)
        if hits:
            violations.append((py_file.name, hits))
    if violations:
        fail(f"revien/importers/ files import a network client: {violations}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("importer verification passed")


if __name__ == "__main__":
    main()
