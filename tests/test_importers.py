"""
Importers leg (WS1, v0.4b): ChatGPT / Claude.ai / Readwise exports through
the ingestion pipeline via revien/importers/.

Covers: ChatGPT branch selection (current_node wins, abandoned branches
never ingested), system-message and non-string-part skipping, epoch->UTC
recorded_at, origin fields on every produced node, a Claude export message
carrying only content-blocks (no top-level text), Readwise tag parsing +
missing optional columns + the declared-link entity edge + a naive
timestamp, idempotent re-import (second run is all units_unchanged), the
REVIEN_INGEST_DENY gate, --dry-run writing nothing, a CliRunner smoke pass
for each of the three commands, and a zero-egress import check.

Fictional names only (Mara/Theo/Sam, Fernweh-Core).
"""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.graph.schema import EdgeType, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionPipeline
from revien.importers import chatgpt as chatgpt_importer
from revien.importers import claude as claude_importer
from revien.importers import readwise as readwise_importer
from revien.importers.base import run_import


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def pipeline(store):
    return IngestionPipeline(store)


def _zip_json(tmp_path: Path, name: str, data, nested: bool = False) -> Path:
    """Build a .zip export containing conversations.json (optionally one
    folder deep, to cover open_export's "any depth" search)."""
    zpath = tmp_path / name
    member = "chatgpt-export/conversations.json" if nested else "conversations.json"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(member, json.dumps(data))
    return zpath


# ── ChatGPT export fixture: an edited branch ─────────────

def _chatgpt_conversation():
    """root -> system(skip) -> u1("Hello") -> {a_old (ABANDONED),
    a_new (current) -> u2 (non-string parts only, dropped) -> a2}.
    current_node = a2, so the abandoned a_old branch must never appear."""
    return {
        "id": "conv-mara-1",
        "title": "Fernweh-Core planning",
        "create_time": 1700000000.0,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["sys"], "message": None},
            "sys": {
                "id": "sys", "parent": "root", "children": ["u1"],
                "message": {
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": [""]},
                    "create_time": 1700000001.0,
                },
            },
            "u1": {
                "id": "u1", "parent": "sys", "children": ["a_old", "a_new"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello, Mara here."]},
                    "create_time": 1700000010.0,
                },
            },
            "a_old": {
                "id": "a_old", "parent": "u1", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Old reply — abandoned branch."]},
                    "create_time": 1700000020.0,
                },
            },
            "a_new": {
                "id": "a_new", "parent": "u1", "children": ["u2"],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["New reply — the current branch."]},
                    "create_time": 1700000030.0,
                },
            },
            "u2": {
                "id": "u2", "parent": "a_new", "children": ["a2"],
                "message": {
                    "author": {"role": "user"},
                    # No string parts at all (e.g. an image asset pointer) —
                    # the whole message must be dropped, not stringified.
                    "content": {"content_type": "multimodal_text",
                                "parts": [{"content_type": "image_asset_pointer"}]},
                    "create_time": 1700000040.0,
                },
            },
            "a2": {
                "id": "a2", "parent": "u2", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["You're welcome, Theo will see this too."]},
                    "create_time": 1700000050.0,
                },
            },
        },
        "current_node": "a2",
    }


class TestChatGPTImporter:
    def test_branch_selection_and_skips(self, tmp_path):
        zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()])
        units = list(chatgpt_importer.iter_units(str(zpath)))
        assert len(units) == 1
        unit = units[0]

        assert "New reply" in unit.content
        assert "Old reply" not in unit.content, "abandoned branch must not be ingested"
        assert "image_asset_pointer" not in unit.content
        # System message is never carried into the transcript content.
        assert unit.content.count("Hello, Mara here.") == 1

    def test_epoch_to_utc_timestamp_and_origin(self, tmp_path, store, pipeline):
        zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()])
        units = list(chatgpt_importer.iter_units(str(zpath)))
        report = run_import(units, pipeline, dry_run=False)
        assert report.units_ingested == 1
        assert report.units_seen == 1

        nodes = store.list_nodes(source_id="chatgpt:conversation:conv-mara-1", limit=100)
        assert nodes, "expected at least the context node"
        for node in nodes:
            assert node.origin_runtime == "chatgpt"
            assert node.origin_source == "import"
            assert node.session_key == "conv-mara-1"
            assert node.recorded_at is not None

        ctx = next(n for n in nodes if n.node_type == NodeType.CONTEXT)
        # Earliest KEPT message is u1 at epoch 1700000010.0 (sys is skipped).
        assert ctx.recorded_at.timestamp() == 1700000010.0

    def test_nested_zip_member_found(self, tmp_path, store, pipeline):
        zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()], nested=True)
        units = list(chatgpt_importer.iter_units(str(zpath)))
        assert len(units) == 1

    def test_bare_json_path(self, tmp_path):
        jpath = tmp_path / "conversations.json"
        jpath.write_text(json.dumps([_chatgpt_conversation()]), encoding="utf-8")
        units = list(chatgpt_importer.iter_units(str(jpath)))
        assert len(units) == 1


# ── Claude.ai export ──────────────────────────────────────

def _claude_conversation():
    return {
        "uuid": "conv-theo-1",
        "name": "Sam's onboarding notes",
        "created_at": "2026-01-05T12:00:00Z",
        "chat_messages": [
            {
                "uuid": "m1", "sender": "human",
                "text": "What did we decide about Fernweh-Core's rollout?",
                "created_at": "2026-01-05T12:00:00Z",
            },
            {
                # No top-level `text` — only a text content block.
                "uuid": "m2", "sender": "assistant", "text": "",
                "content": [{"type": "text", "text": "We staged it behind a flag for Sam's team."}],
                "created_at": "2026-01-05T12:00:10Z",
            },
        ],
    }


class TestClaudeImporter:
    def test_content_blocks_only_message(self, tmp_path):
        zpath = _zip_json(tmp_path, "claude.zip", [_claude_conversation()])
        units = list(claude_importer.iter_units(str(zpath)))
        assert len(units) == 1
        assert "staged it behind a flag" in units[0].content
        assert "Fernweh-Core's rollout" in units[0].content

    def test_origin_and_timestamp(self, store, pipeline, tmp_path):
        zpath = _zip_json(tmp_path, "claude.zip", [_claude_conversation()])
        units = list(claude_importer.iter_units(str(zpath)))
        report = run_import(units, pipeline, dry_run=False)
        assert report.units_ingested == 1

        nodes = store.list_nodes(source_id="claude:conversation:conv-theo-1", limit=100)
        for node in nodes:
            assert node.origin_runtime == "claude"
            assert node.origin_source == "import"
            assert node.session_key == "conv-theo-1"


# ── Readwise CSV export ───────────────────────────────────

def _readwise_csv(rows, fieldnames=None):
    fieldnames = fieldnames or [
        "Highlight", "Book Title", "Book Author", "Note", "Color", "Tags",
        "Location Type", "Location", "Highlighted at", "Document tags",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


class TestReadwiseImporter:
    def test_missing_optional_columns_and_tags(self, tmp_path):
        # Only Highlight + Book Title + Document tags — everything else
        # (Note, Color, Tags, Location Type, Highlighted at) absent.
        text = _readwise_csv(
            [{"Highlight": "Sovereignty of Choice matters.", "Book Title": "Fernweh-Core Field Notes",
              "Document tags": "#memory, #sovereignty"}],
            fieldnames=["Highlight", "Book Title", "Document tags"],
        )
        csv_path = tmp_path / "readwise.csv"
        csv_path.write_text(text, encoding="utf-8")

        units = list(readwise_importer.iter_units(str(csv_path)))
        assert len(units) == 1
        unit = units[0]
        assert unit.metadata["tags"] == ["memory", "sovereignty"]
        assert unit.metadata["author"] is None
        assert unit.links == ["Fernweh-Core Field Notes"]

    def test_naive_timestamp_and_entity_edge(self, tmp_path, store, pipeline):
        text = _readwise_csv([
            {
                "Highlight": "Mara keeps the architecture immutable on purpose.",
                "Book Title": "Fernweh-Core Field Notes",
                "Book Author": "Theo",
                "Note": "Worth re-reading.",
                "Location": "142",
                "Highlighted at": "2021-05-04 10:32:00",  # naive, no offset
                "Tags": "#architecture",
            },
        ])
        csv_path = tmp_path / "readwise.csv"
        csv_path.write_text(text, encoding="utf-8")

        units = list(readwise_importer.iter_units(str(csv_path)))
        assert len(units) == 1
        unit = units[0]
        assert unit.timestamp is not None
        assert unit.timestamp.tzinfo is not None
        assert "Worth re-reading." in unit.content

        report = run_import(units, pipeline, dry_run=False)
        assert report.units_ingested == 1

        entities = store.list_nodes(node_type=NodeType.ENTITY, limit=100)
        book = next((n for n in entities if n.label == "Fernweh-Core Field Notes"), None)
        assert book is not None, "declared link should create/find the book entity"

        ctx_nodes = store.list_nodes(node_type=NodeType.CONTEXT, limit=100)
        ctx = next(n for n in ctx_nodes if n.source_id.startswith("readwise:"))
        edges = store.get_edges_for_node(ctx.node_id)
        assert any(
            e.edge_type == EdgeType.RELATED_TO and e.target_node_id == book.node_id
            for e in edges
        )

    def test_row_without_highlight_is_dropped(self, tmp_path):
        text = _readwise_csv([{"Highlight": "", "Book Title": "Empty Row Book"}])
        csv_path = tmp_path / "readwise.csv"
        csv_path.write_text(text, encoding="utf-8")
        units = list(readwise_importer.iter_units(str(csv_path)))
        assert units == []


# ── Idempotency (all three importers) ────────────────────

@pytest.mark.parametrize("build", [
    lambda tmp_path: (chatgpt_importer, _zip_json(tmp_path, "c.zip", [_chatgpt_conversation()])),
    lambda tmp_path: (claude_importer, _zip_json(tmp_path, "c.zip", [_claude_conversation()])),
    lambda tmp_path: (readwise_importer, tmp_path / "r.csv"),
], ids=["chatgpt", "claude", "readwise"])
def test_second_run_is_unchanged(build, tmp_path, store, pipeline):
    module, path = build(tmp_path)
    if module is readwise_importer:
        path.write_text(_readwise_csv([{
            "Highlight": "Consent Is Law.", "Book Title": "Fernweh-Core Field Notes",
            "Highlighted at": "2021-05-04 10:32:00",
        }]), encoding="utf-8")

    units_first = list(module.iter_units(str(path)))
    report1 = run_import(units_first, pipeline, dry_run=False)
    assert report1.units_ingested == report1.units_seen
    assert report1.units_unchanged == 0

    nodes_after_first = store.count_nodes()

    units_second = list(module.iter_units(str(path)))
    report2 = run_import(units_second, pipeline, dry_run=False)
    assert report2.units_seen == report1.units_seen
    assert report2.units_unchanged == report2.units_seen
    assert report2.units_ingested == 0
    assert store.count_nodes() == nodes_after_first


# ── Deny list ─────────────────────────────────────────────

def test_deny_list_blocks_one_unit(tmp_path, store, pipeline, monkeypatch):
    zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()])
    units = list(chatgpt_importer.iter_units(str(zpath)))
    denied_source_id = units[0].source_id

    monkeypatch.setenv("REVIEN_INGEST_DENY", denied_source_id)
    report = run_import(units, pipeline, dry_run=False)
    assert report.units_denied == 1
    assert report.units_ingested == 0

    nodes = store.list_nodes(source_id=denied_source_id, limit=100)
    assert nodes == []


# ── Dry run ───────────────────────────────────────────────

def test_dry_run_writes_nothing(tmp_path, store, pipeline):
    zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()])
    units = list(chatgpt_importer.iter_units(str(zpath)))

    before = store.count_nodes()
    report = run_import(units, pipeline=None, dry_run=True)
    assert report.units_ingested == 1
    assert store.count_nodes() == before == 0


# ── CLI smoke ─────────────────────────────────────────────

class TestCliSmoke:
    def test_import_chatgpt(self, tmp_path):
        zpath = _zip_json(tmp_path, "chatgpt.zip", [_chatgpt_conversation()])
        db_path = str(tmp_path / "revien.db")
        runner = CliRunner()
        result = runner.invoke(main, ["import-chatgpt", str(zpath), "--db", db_path])
        assert result.exit_code == 0, result.output
        assert "seen=1" in result.output
        assert "ingested=1" in result.output

    def test_import_chatgpt_dry_run(self, tmp_path):
        jpath = tmp_path / "conversations.json"
        jpath.write_text(json.dumps([_chatgpt_conversation()]), encoding="utf-8")
        db_path = str(tmp_path / "revien.db")
        runner = CliRunner()
        result = runner.invoke(
            main, ["import-chatgpt", str(jpath), "--db", db_path, "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "dry run: nothing written" in result.output
        assert not Path(db_path).exists(), "a dry run must never create the db file"

    def test_import_claude(self, tmp_path):
        zpath = _zip_json(tmp_path, "claude.zip", [_claude_conversation()])
        db_path = str(tmp_path / "revien.db")
        runner = CliRunner()
        result = runner.invoke(main, ["import-claude", str(zpath), "--db", db_path])
        assert result.exit_code == 0, result.output
        assert "seen=1" in result.output

    def test_import_readwise(self, tmp_path):
        csv_path = tmp_path / "readwise.csv"
        csv_path.write_text(_readwise_csv([{
            "Highlight": "Edge Over Theater.", "Book Title": "Fernweh-Core Field Notes",
            "Highlighted at": "2021-05-04 10:32:00",
        }]), encoding="utf-8")
        db_path = str(tmp_path / "revien.db")
        runner = CliRunner()
        result = runner.invoke(main, ["import-readwise", str(csv_path), "--db", db_path])
        assert result.exit_code == 0, result.output
        assert "seen=1" in result.output
        assert "ingested=1" in result.output


# ── F2: refresh vs unchanged ──────────────────────────────

def test_edited_conversation_is_ingested_not_unchanged(tmp_path, store, pipeline):
    """A re-import of an EDITED conversation is a keyed REFRESH: it must
    count as units_ingested (nodes created or refreshed), never
    units_unchanged — even when re-extraction happens to add 0 brand-new
    nodes/edges. Only a byte-identical re-import is unchanged."""
    conv = _chatgpt_conversation()
    zpath1 = _zip_json(tmp_path, "c1.zip", [conv])
    units1 = list(chatgpt_importer.iter_units(str(zpath1)))
    report1 = run_import(units1, pipeline, dry_run=False)
    assert report1.units_ingested == 1
    assert report1.units_unchanged == 0

    # Edit one message's text (same conv id -> same source_id/ingest_key).
    conv["mapping"]["a2"]["message"]["content"]["parts"] = [
        "You're welcome, Theo will see this too. EDITED."
    ]
    zpath2 = _zip_json(tmp_path, "c2.zip", [conv])
    units2 = list(chatgpt_importer.iter_units(str(zpath2)))
    report2 = run_import(units2, pipeline, dry_run=False)
    assert report2.units_ingested == 1, "an edited re-import must be ingested, not unchanged"
    assert report2.units_unchanged == 0

    # Re-running the EDITED content again (byte-identical this time) IS
    # unchanged.
    units3 = list(chatgpt_importer.iter_units(str(zpath2)))
    report3 = run_import(units3, pipeline, dry_run=False)
    assert report3.units_unchanged == 1
    assert report3.units_ingested == 0


# ── F3: a malformed conversation mid-batch is logged, not fatal ──────────

def test_malformed_middle_conversation_is_logged_not_fatal(tmp_path, store, pipeline):
    good1 = _chatgpt_conversation()
    good1["id"] = "conv-good-1"
    good2 = _chatgpt_conversation()
    good2["id"] = "conv-good-2"
    bad = {"id": "conv-bad", "title": "Bad", "create_time": 1.0,
           "mapping": {"a": "not-a-dict"}, "current_node": "a"}

    zpath = _zip_json(tmp_path, "mixed.zip", [good1, bad, good2])
    units = list(chatgpt_importer.iter_units(str(zpath)))
    assert len(units) == 3

    report = run_import(units, pipeline, dry_run=False)
    assert report.units_seen == 3
    assert report.units_ingested == 2
    assert len(report.errors) == 1
    assert report.errors[0][0] == "chatgpt:conversation:error:1"


def test_non_array_top_level_json_raises_value_error(tmp_path):
    jpath = tmp_path / "conversations.json"
    jpath.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    with pytest.raises(ValueError):
        list(chatgpt_importer.iter_units(str(jpath)))


def test_chatgpt_cli_partial_import_exits_zero(tmp_path):
    """F11: a partial import (2 ingested, 1 error) exits 0 — only a total
    failure (errors > 0 AND nothing ingested/unchanged) is non-zero."""
    good1 = _chatgpt_conversation()
    good1["id"] = "conv-good-1"
    good2 = _chatgpt_conversation()
    good2["id"] = "conv-good-2"
    bad = {"id": "conv-bad", "title": "Bad", "create_time": 1.0,
           "mapping": {"a": "not-a-dict"}, "current_node": "a"}
    zpath = _zip_json(tmp_path, "mixed.zip", [good1, bad, good2])
    db_path = str(tmp_path / "revien.db")
    runner = CliRunner()
    result = runner.invoke(main, ["import-chatgpt", str(zpath), "--db", db_path])
    assert result.exit_code == 0, result.output
    assert "errors=1" in result.output
    assert "ingested=2" in result.output


# ── F4: readwise digest disambiguates identical-text highlights ──────────

def test_readwise_distinct_highlights_same_text_get_distinct_source_ids(tmp_path):
    text = _readwise_csv([
        {"Highlight": "The map is not the territory.", "Book Title": "Fernweh Atlas",
         "Note": "first note"},
        {"Highlight": "The map is not the territory.", "Book Title": "Fernweh Atlas",
         "Note": "second note, different"},
    ])
    csv_path = tmp_path / "readwise.csv"
    csv_path.write_text(text, encoding="utf-8")
    units = list(readwise_importer.iter_units(str(csv_path)))
    assert len(units) == 2
    assert units[0].source_id != units[1].source_id


# ── readwise digest must be order-stable (not row-index-keyed) ───────────

def test_readwise_digest_order_stable_across_prepended_row(tmp_path, store, pipeline):
    """The source_id digest must depend only on highlight+location+note,
    never on row position. If Readwise (or a re-export) prepends a new row
    ahead of previously-imported highlights, those highlights' row_index
    shifts by one — a row-index-keyed digest would change their source_id
    and cause a full re-ingest (and eventually unbounded duplication)
    instead of recognizing them as unchanged."""
    row_a = {"Highlight": "Truth Before Self.", "Book Title": "Fernweh Atlas", "Location": "10"}
    row_b = {"Highlight": "Consent Is Law.", "Book Title": "Fernweh Atlas", "Location": "20"}
    row_c = {"Highlight": "Memory Is Sacred.", "Book Title": "Fernweh Atlas", "Location": "5"}

    csv_path = tmp_path / "readwise.csv"
    csv_path.write_text(_readwise_csv([row_a, row_b]), encoding="utf-8")

    units_first = list(readwise_importer.iter_units(str(csv_path)))
    report1 = run_import(units_first, pipeline, dry_run=False)
    assert report1.units_ingested == 2
    nodes_after_first = store.count_nodes()

    # Prepend a third, brand-new row ahead of a and b — their row_index
    # shifts from (0, 1) to (1, 2).
    csv_path.write_text(_readwise_csv([row_c, row_a, row_b]), encoding="utf-8")
    units_second = list(readwise_importer.iter_units(str(csv_path)))
    report2 = run_import(units_second, pipeline, dry_run=False)

    assert report2.units_ingested == 1, "only the new row should be ingested"
    assert report2.units_unchanged == 2, "the shifted-but-identical rows must be recognized unchanged"

    # Node count grew by exactly the new unit's nodes: one new CONTEXT node
    # for "Memory Is Sacred." plus its extracted ENTITY node (the shared
    # book ENTITY node already existed from the first import, so it's not
    # re-created) — never the 3+ nodes a full re-ingest of all three rows
    # would produce.
    nodes_added = store.count_nodes() - nodes_after_first
    ctx_nodes = store.list_nodes(node_type=NodeType.CONTEXT, limit=100)
    new_ctx = [n for n in ctx_nodes if n.content.startswith("Memory Is Sacred")]
    assert len(new_ctx) == 1
    assert nodes_added == 2


# ── F10: multiple conversations.json members in one zip ──────────────────

def test_open_export_picks_shallowest_of_multiple_members(tmp_path, capsys):
    from revien.importers.base import open_export

    zpath = tmp_path / "double.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("nested/conversations.json", json.dumps([{"deep": True}]))
        zf.writestr("conversations.json", json.dumps([{"shallow": True}]))
    raw = open_export(str(zpath))
    data = json.loads(raw)
    assert data == [{"shallow": True}]
    err = capsys.readouterr().err
    assert "multiple conversations.json" in err
    assert "nested/conversations.json" in err


def test_no_network_imports():
    importers_dir = Path(__file__).resolve().parent.parent / "revien" / "importers"
    forbidden = ("httpx", "requests", "urllib", "socket")
    for py_file in importers_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in text, f"{py_file.name} imports {name}"
            assert f"from {name}" not in text, f"{py_file.name} imports from {name}"


def test_no_network_imports_runtime(tmp_path):
    """Source-text grep (above) only catches a direct `import httpx` line in
    the importers themselves — it would miss a transitive pull-in through a
    module an importer imports (e.g. importers/base.py's slugify reaching
    into revien.adapters.obsidian, whose PACKAGE __init__ loads
    generic_api.py/ollama_adapter.py, which import httpx). Run a full
    Readwise import through the CLI, in a subprocess so this test's own
    already-imported modules can't mask the result, and assert httpx never
    lands in sys.modules.

    REVIEN_SEMANTIC/REVIEN_RERANK are pinned off explicitly (not just
    inherited from tests/conftest.py's autouse fixture, which wouldn't
    reach a subprocess anyway if pytest's own env didn't happen to carry
    it): the semantic/rerank layers pull in fastembed -> huggingface_hub
    -> httpx on first use, which is a real, unrelated, already-accepted
    dependency chain for that opt-in layer — not the regression this test
    is targeting (revien.adapters' __init__ loading httpx just to reach a
    slug helper)."""
    csv_path = tmp_path / "readwise.csv"
    csv_path.write_text(_readwise_csv([{
        "Highlight": "Architecture Is Immutable.", "Book Title": "Fernweh Atlas",
        "Highlighted at": "2021-05-04 10:32:00",
    }]), encoding="utf-8")
    db_path = tmp_path / "revien.db"

    script = f"""
import sys
from click.testing import CliRunner
from revien.cli import main

runner = CliRunner()
result = runner.invoke(main, ["import-readwise", {str(csv_path)!r}, "--db", {str(db_path)!r}])
assert result.exit_code == 0, result.output
assert "httpx" not in sys.modules, sorted(sys.modules.keys())
assert "revien.adapters" not in sys.modules, sorted(sys.modules.keys())
print("OK")
"""
    env = dict(os.environ, REVIEN_SEMANTIC="0", REVIEN_RERANK="0")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
