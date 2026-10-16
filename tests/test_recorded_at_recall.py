"""Recall results carry recorded_at — when the content was SAID.

A consuming model can only resolve "yesterday" / "next month" if every
retrieved memory says when it was said. recorded_at is the node's
recorded_at (from IngestionInput.timestamp), never created_at (ingest time).
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from revien.adapters.langchain_adapter import RevienMemory
from revien.adapters.ollama_adapter import OllamaAdapter
from revien.daemon.server import create_app
from revien.graph.store import GraphStore
from revien.hermes_provider import RevienMemoryProvider
from revien.ingestion.fence import fence_content
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.mcp_server import build_mcp_server
from revien.retrieval.engine import RetrievalEngine
from revien.toon import parse_recall, serialize_recall
from tests.test_mcp import _call

TEXT = "User: Mara moved the Postgres billing migration.\nAssistant: Noted, Postgres billing migration."
QUERY = "Postgres billing migration"
SAID = datetime(2023, 5, 7, tzinfo=timezone.utc)
ISO = "2023-05-07T00:00:00+00:00"


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except (PermissionError, OSError):  # pragma: no cover - Windows WAL race
        pass


def _seed(path, timestamp):
    store = GraphStore(db_path=path)
    IngestionPipeline(store).ingest(
        IngestionInput(source_id="s1", content=TEXT, timestamp=timestamp)
    )
    return store


class TestEngine:
    def test_result_carries_iso_date(self, db_path):
        store = _seed(db_path, SAID)
        results = RetrievalEngine(store).recall(QUERY, top_n=10).results
        store.close()
        assert results
        dated = [r.recorded_at for r in results if r.recorded_at]
        assert dated and set(dated) == {ISO}

    def test_none_when_absent_never_created_at(self, db_path):
        store = _seed(db_path, None)
        results = RetrievalEngine(store).recall(QUERY, top_n=10).results
        store.close()
        assert results
        assert all(r.recorded_at is None for r in results)


class TestSurfaces:
    def test_daemon_shape(self, db_path):
        with TestClient(create_app(db_path=db_path)) as c:
            c.post("/v1/ingest", json={
                "source_id": "s1", "content": TEXT, "timestamp": ISO})
            data = c.post("/v1/recall", json={"query": QUERY}).json()
        assert data["results"]
        assert all("recorded_at" in r for r in data["results"])
        assert ISO in {r["recorded_at"] for r in data["results"]}

    def test_mcp_shape(self, db_path):
        server = build_mcp_server(db_path=db_path)
        _call(server, "revien_store", {"content": TEXT})
        out = _call(server, "revien_recall", {"query": QUERY})
        assert out["results"]
        assert all("recorded_at" in r for r in out["results"])
        # revien_store has no timestamp parameter: the store call is the
        # capture moment, so it is stamped and labelled capture.
        assert all(r["recorded_at"] for r in out["results"])
        assert {r["recorded_at_source"] for r in out["results"]} == {"capture"}


def _payload(recorded_at):
    return {
        "query": "q",
        "results": [{
            "node_id": "n-1", "node_type": "fact", "label": "L",
            "content": "c", "score": 0.5,
            "score_breakdown": {"recency": 0.9}, "path": ["n-1"],
            "origin_runtime": None, "origin_source": None,
            "project_key": None, "recorded_at": recorded_at,
        }],
        "nodes_examined": 1, "retrieval_time_ms": 1.0,
        "semantic_active": False, "semantic_note": None,
        "skill_proposals": [],
    }


class TestToon:
    @pytest.mark.parametrize("value", [ISO, None])
    def test_round_trip(self, value):
        payload = _payload(value)
        back = parse_recall(serialize_recall(payload))
        assert back == payload
        assert back["results"][0]["recorded_at"] == value


def _resp(recorded_at, content="Mara moved billing to Postgres.", source="content"):
    r = SimpleNamespace(
        node_id="n-1", node_type="fact", label="Billing", content=content,
        score=0.8, score_breakdown={}, path=[], recorded_at=recorded_at,
        recorded_at_source=source,
    )
    return SimpleNamespace(results=[r], retrieval_time_ms=1.0)


class TestPromptRendering:
    def test_hermes(self):
        fmt = RevienMemoryProvider._format_context
        assert fmt(_resp(ISO)).splitlines() == [
            "## Relevant memory (Revien)",
            TestDateNote.NOTE,
            "- [2023-05-07] Mara moved billing to Postgres.",
        ]
        assert "- Mara moved billing to Postgres." in fmt(_resp(None)).splitlines()

    def test_langchain(self):
        fmt = RevienMemory._format_retrieval_response
        dated = fmt(object(), _resp(ISO))
        assert "\n[2023-05-07] Mara moved billing to Postgres.\n" in dated
        assert dated.startswith("## Relevant Context (from 1 nodes)")
        undated = fmt(object(), _resp(None))
        assert "\n\nMara moved billing to Postgres.\n" in undated
        assert "[20" not in undated

    def test_ollama(self, db_path):
        adapter = OllamaAdapter(graph_path=db_path)
        adapter.pipeline.ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=SAID))
        ctx = adapter.get_context_for_prompt(QUERY)
        assert ctx.startswith("[Revien Memory Context]")
        assert ctx.endswith("[End Memory Context]")
        assert any(l.startswith("- [2023-05-07] [Score: ")
                   for l in ctx.splitlines())

    def test_ollama_age_never_from_ingest_time(self, db_path):
        # said a year ago, ingested just now: no "N days ago" from created_at
        said = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=365)
        adapter = OllamaAdapter(graph_path=db_path)
        adapter.pipeline.ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=said))
        ctx = adapter.get_context_for_prompt(QUERY)
        lines = [l for l in ctx.splitlines() if l.startswith("- ")]
        assert lines
        assert all(l.startswith(f"- [{said.date().isoformat()}] [Score: ")
                   for l in lines)
        assert "ago" not in ctx

    def test_ollama_undated_has_no_age(self, db_path):
        adapter = OllamaAdapter(graph_path=db_path)
        adapter.pipeline.ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=None))
        ctx = adapter.get_context_for_prompt(QUERY)
        lines = [l for l in ctx.splitlines() if l.startswith("- ")]
        assert lines
        assert all(l.startswith("- [Score: ") for l in lines)
        assert "ago" not in ctx and "unknown time" not in ctx


class TestDateNote:
    NOTE = ("(dates in brackets are when each memory was said; "
            "resolve 'yesterday' etc. against them)")

    def test_hermes(self):
        fmt = RevienMemoryProvider._format_context
        assert fmt(_resp(ISO)).splitlines()[1] == self.NOTE
        assert self.NOTE not in fmt(_resp(None))

    def test_langchain(self):
        fmt = RevienMemory._format_retrieval_response
        dated = fmt(object(), _resp(ISO)).splitlines()
        assert self.NOTE in dated[1:3]
        assert self.NOTE not in fmt(object(), _resp(None))

    def test_ollama(self, db_path):
        adapter = OllamaAdapter(graph_path=db_path)
        adapter.pipeline.ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=SAID))
        dated = adapter.get_context_for_prompt(QUERY).splitlines()
        assert dated[0] == "[Revien Memory Context]"
        assert self.NOTE in dated[1:4]
        assert dated[-1] == "[End Memory Context]"

    def test_ollama_undated(self, db_path):
        adapter = OllamaAdapter(graph_path=db_path)
        adapter.pipeline.ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=None))
        assert self.NOTE not in adapter.get_context_for_prompt(QUERY)

    def test_fence_strips_note(self):
        for ctx in (
            RevienMemoryProvider._format_context(_resp(ISO)),
            RevienMemory._format_retrieval_response(object(), _resp(ISO)),
        ):
            out = fence_content("User: hi\n" + ctx + "\nAssistant: ok").content
            assert "dates in brackets" not in out and "Postgres" not in out


class TestNoAgeFromIngestTime:
    """Every rendering surface: recorded_at a year ago, created_at now."""

    def _old(self):
        return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=365)

    def test_hermes_and_langchain_show_date_not_age(self):
        said = self._old().isoformat()
        hermes = RevienMemoryProvider._format_context(_resp(said))
        assert said[:10] in hermes and "ago" not in hermes
        undated = RevienMemoryProvider._format_context(_resp(None))
        assert "ago" not in undated and "[20" not in undated

    def test_cli_recall_and_tensions_no_relative_age(self, db_path):
        from click.testing import CliRunner
        from revien.cli import main
        store = _seed(db_path, self._old())
        store.close()
        out = CliRunner().invoke(main, ["recall", QUERY, "--db", db_path])
        assert "ago" not in out.output
        out = CliRunner().invoke(main, ["tensions", "--db", db_path])
        assert "ago" not in out.output

    def test_daemon_and_mcp_carry_no_age(self, db_path):
        _seed(db_path, self._old()).close()
        with TestClient(create_app(db_path=db_path)) as client:
            body = client.post("/v1/recall", json={"query": QUERY}).text
        assert "ago" not in body.lower()
        server = build_mcp_server(db_path=db_path)
        out = _call(server, "revien_recall", {"query": QUERY})
        assert "ago" not in str(out).lower()


class TestFenceStillMatches:
    def test_dated_hermes_and_ollama_lines_stripped(self):
        text = (
            "User: hi\n"
            "[Revien Memory Context]\n"
            "The following context is retrieved:\n"
            "- [2023-05-07] [Score: 91%] (2 days ago) DB: PostgreSQL.\n"
            "\n[End Memory Context]\n"
            "## Relevant memory (Revien)\n"
            "- [2023-05-07] Hermes dated memory line\n"
            "Assistant: ok"
        )
        out = fence_content(text).content
        assert "PostgreSQL" not in out
        assert "Hermes dated memory line" not in out
        assert "User: hi" in out and "Assistant: ok" in out

    def test_dated_langchain_block_stripped(self):
        text = (
            "User: hi\n"
            "## Relevant Context (from 1 nodes)\n\n### Result 1: L\n"
            "Type: fact | Score: 0.800\n\n[2023-05-07] dated langchain body\n\n"
            "[Retrieved in 1.00ms]\n"
        )
        out = fence_content(text).content
        assert "dated langchain body" not in out
        assert "User: hi" in out
