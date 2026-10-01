"""recorded_at_source backfill (user_version 4) and the tightened render rule.

A memory whose date is unknown or a file mtime is never shown as "when said".
"""

import importlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from revien.adapters.langchain_adapter import RevienMemory
from revien.adapters.ollama_adapter import OllamaAdapter
from revien.dates import DATE_NOTE, said_date
from revien.graph.origin import derive_recorded_at_source
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.hermes_provider import RevienMemoryProvider
from revien.ingestion.pipeline import IngestionInput
from revien_bench.answerers import RetrievedContext, _format_context
from datetime import datetime, timezone

ISO = "2023-05-07T00:00:00+00:00"
SAID = datetime(2023, 5, 7, tzinfo=timezone.utc)

TABLE = [
    ("claude-code", "live", "mtime"),
    ("codex", "live", "mtime"),
    ("file", "watch", "mtime"),
    ("obsidian", "vault", "mtime"),
    ("hermes", "live", "capture"),
    ("ollama", "live", "capture"),
    ("langchain", "live", "capture"),
    ("api", "api", "capture"),
    ("chatgpt", "import", "content"),
    ("claude", "import", "content"),
    ("readwise", "import", "content"),
    ("openai", "import", "content"),
    (None, "import", "content"),
    ("openai", "live", None),
    (None, None, None),
    (None, "live", None),
    ("mystery", None, None),
]


@pytest.mark.parametrize("runtime,source,expected", TABLE)
def test_derive_table(runtime, source, expected):
    assert derive_recorded_at_source(runtime, source) == expected


def _mk(store, node_id, runtime, source, dated=True):
    store.add_node(Node(
        node_id=node_id, node_type=NodeType.FACT, label=node_id,
        content=node_id, source_id=node_id,
        recorded_at=SAID if dated else None,
        origin_runtime=runtime, origin_source=source,
        metadata={"recorded_at_source": "content", "keep": 1},
    ))


def _legacy_db(tmp_path):
    path = str(tmp_path / "legacy.db")
    store = GraphStore(db_path=path)
    for i, (runtime, source, _) in enumerate(TABLE):
        _mk(store, f"n{i}", runtime, source)
    _mk(store, "undated", "claude-code", "live", dated=False)
    store.close()
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE nodes SET metadata = json_remove(metadata, '$.recorded_at_source')")
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()
    return path


def _sources(path):
    conn = sqlite3.connect(path)
    try:
        out = {nid: json.loads(m).get("recorded_at_source")
               for nid, m in conn.execute("SELECT node_id, metadata FROM nodes")}
        keep = {json.loads(m).get("keep")
                for (m,) in conn.execute("SELECT metadata FROM nodes")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    return out, keep, version


def test_backfill_on_legacy_db_then_idempotent(tmp_path):
    path = _legacy_db(tmp_path)
    GraphStore(db_path=path).close()
    first, keep, version = _sources(path)
    assert version == 4
    assert keep == {1}  # other metadata untouched
    for i, (_, _, expected) in enumerate(TABLE):
        assert first[f"n{i}"] == expected
    assert first["undated"] is None  # recorded_at NULL: left alone
    GraphStore(db_path=path).close()
    assert _sources(path)[0] == first


def test_backfill_never_overwrites_existing_source(tmp_path):
    path = str(tmp_path / "keep.db")
    store = GraphStore(db_path=path)
    _mk(store, "a", "claude-code", "live")  # carries source "content"
    store.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()
    GraphStore(db_path=path).close()
    assert _sources(path)[0]["a"] == "content"


def test_migration_004_matches_in_store_guard(tmp_path):
    mod = importlib.import_module("revien.graph.migrations.004_recorded_at_source")
    a = _legacy_db(tmp_path)
    (tmp_path / "b").mkdir()
    b = _legacy_db(tmp_path / "b")
    GraphStore(db_path=a).close()
    first = mod.migrate(b)
    assert first["backfilled"] == sum(1 for t in TABLE if t[2])
    assert _sources(a)[0] == _sources(b)[0]
    assert _sources(b)[2] == 4
    assert mod.migrate(b)["backfilled"] == 0


def test_said_date_requires_dated_source():
    for src in ("content", "capture", "import"):
        assert said_date(ISO, src) == "2023-05-07"
    for src in (None, "mtime", "bogus"):
        assert said_date(ISO, src) is None


def _resp(source, with_key=True):
    r = SimpleNamespace(
        node_id="n-1", node_type="fact", label="Billing",
        content="Mara moved billing to Postgres.", score=0.8,
        score_breakdown={}, path=[], recorded_at=ISO, recorded_at_source=source)
    return SimpleNamespace(results=[r], retrieval_time_ms=1.0)


@pytest.mark.parametrize("source", [None, "mtime"])
def test_no_date_no_note_on_every_surface(source, tmp_path):
    hermes = RevienMemoryProvider._format_context(_resp(source))
    lang = RevienMemory._format_retrieval_response(object(), _resp(source))
    for text in (hermes, lang):
        assert "2023-05-07" not in text and DATE_NOTE not in text
        assert "Mara moved billing to Postgres." in text

    ctx = RetrievedContext(query="q",
        contents=["Mara moved billing to Postgres."], dates=[ISO],
        date_sources=[source])
    block = _format_context(ctx)
    assert "2023" not in block and "May" not in block

    adapter = OllamaAdapter(graph_path=str(tmp_path / "o.db"))
    adapter.pipeline.ingest(IngestionInput(
        source_id="s1", timestamp=SAID,
        timestamp_source=source or "mtime",
        content="User: Mara moved the Postgres billing migration.\n"
                "Assistant: Noted, Postgres billing migration."))
    if source is None:
        adapter.store.close()
        conn = sqlite3.connect(str(tmp_path / "o.db"))
        conn.execute("UPDATE nodes SET metadata = json_remove(metadata, '$.recorded_at_source')")
        conn.commit()
        conn.close()
        adapter = OllamaAdapter(graph_path=str(tmp_path / "o.db"))
    out = adapter.get_context_for_prompt("Postgres billing migration")
    assert "2023-05-07" not in out and DATE_NOTE not in out
    assert "Postgres" in out


def test_content_row_renders_date_and_note():
    hermes = RevienMemoryProvider._format_context(_resp("content"))
    lang = RevienMemory._format_retrieval_response(object(), _resp("content"))
    for text in (hermes, lang):
        assert "[2023-05-07] Mara moved billing to Postgres." in text
        assert DATE_NOTE in text
    ctx = RetrievedContext(query="q",
        contents=["Mara moved billing to Postgres."], dates=[ISO],
        date_sources=["content"])
    assert "7 May 2023" in _format_context(ctx)


def test_status_counts_by_recorded_at_source(tmp_path):
    from click.testing import CliRunner
    from revien.cli import main
    path = _legacy_db(tmp_path)
    out = CliRunner().invoke(main, ["status", "--db", path]).output
    assert "Nodes by recorded_at_source:" in out
    assert "  mtime: 4 (date not shown)" in out
    assert "unknown" not in out.split("Nodes by recorded_at_source:")[1]
