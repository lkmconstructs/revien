"""A consuming model is never told something false about a memory's date.

B2: whole-session adapters stamp content time (earliest message timestamp),
    never the file mtime; the source rides to every render surface; a date
    that is only an mtime is neither shown nor announced.
S6: the speaker's own calendar day survives (offset preserved, not UTC'd).
S7: TOON from an older daemon (no recorded_at columns) still parses.
S8: hermes / MCP store pass a capture timestamp when the caller gives none.
N9: the fence strips only the exact emitted note line.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.codex import CodexAdapter
from revien.adapters.langchain_adapter import RevienMemory
from revien.adapters.ollama_adapter import OllamaAdapter
from revien.cli import main
from revien.dates import DATE_NOTE, said_date
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.store import GraphStore
from revien.hermes_provider import RevienMemoryProvider
from revien.ingestion.fence import fence_content
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.mcp_server import build_mcp_server
from revien.retrieval.engine import RetrievalEngine
from revien.toon import parse_recall, serialize_recall
from tests.test_mcp import _call

QUERY = "Postgres billing migration"
TEXT = ("User: Mara moved the Postgres billing migration.\n"
        "Assistant: Noted, Postgres billing migration.")


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:  # pragma: no cover - Windows WAL race
        pass


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _recall(path):
    store = GraphStore(db_path=path)
    try:
        return RetrievalEngine(store).recall(QUERY, top_n=5)
    finally:
        store.close()


def _write_jsonl(path, lines, mtime):
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))


MTIME = datetime(2023, 5, 3, 12, 0, tzinfo=timezone.utc)


def _claude_msgs(stamp1, stamp2):
    def m(t, who, text, ts):
        d = {"type": t, "content": text}
        if ts:
            d["timestamp"] = ts
        return d
    return [
        m("user", "u", "Mara moved the Postgres billing migration.", stamp1),
        m("assistant", "a", "Noted, Postgres billing migration.", stamp2),
    ]


def _ingest_via_scheduler(adapter, path):
    store = GraphStore(db_path=path)
    try:
        sched = SyncScheduler(IngestionPipeline(store))
        sched.register_adapter("a", adapter)
        _run(sched.sync_all())
    finally:
        store.close()


# ── B2 ────────────────────────────────────────────────────────────────

class TestSessionAdaptersStampContentTime:
    def test_claude_code_uses_earliest_message_timestamp_not_mtime(self, tmp_path, db_path):
        f = tmp_path / "proj" / "sess.jsonl"
        f.parent.mkdir()
        # out of order on purpose: the EARLIEST wins
        _write_jsonl(f, _claude_msgs("2023-05-02T10:00:00Z", "2023-05-01T09:00:00Z"), MTIME)
        adapter = ClaudeCodeAdapter(session_dir=str(tmp_path))
        items = _run(adapter.fetch_new_content(datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert len(items) == 1
        assert items[0]["timestamp"].startswith("2023-05-01T09:00:00")
        assert items[0]["timestamp_source"] == "content"

        _ingest_via_scheduler(adapter, db_path)
        resp = _recall(db_path)
        assert resp.results
        assert {r.recorded_at_source for r in resp.results} == {"content"}
        block = RevienMemoryProvider._format_context(resp)
        assert "[2023-05-01]" in block
        assert "2023-05-03" not in block
        assert DATE_NOTE in block

    def test_claude_code_without_message_timestamps_falls_back_to_mtime_and_hides_it(
            self, tmp_path, db_path):
        f = tmp_path / "proj" / "sess.jsonl"
        f.parent.mkdir()
        _write_jsonl(f, _claude_msgs(None, None), MTIME)
        adapter = ClaudeCodeAdapter(session_dir=str(tmp_path))
        items = _run(adapter.fetch_new_content(datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert items[0]["timestamp_source"] == "mtime"
        assert items[0]["timestamp"].startswith("2023-05-03")

        _ingest_via_scheduler(adapter, db_path)
        resp = _recall(db_path)
        assert resp.results and {r.recorded_at_source for r in resp.results} == {"mtime"}
        # every render surface: no date, no note
        hermes = RevienMemoryProvider._format_context(resp)
        langchain = RevienMemory._format_retrieval_response(object(), resp)
        a = OllamaAdapter(graph_path=db_path)
        ollama = a.get_context_for_prompt(QUERY)
        for text in (hermes, langchain, ollama):
            assert "2023-05" not in text
            assert "dates in brackets" not in text
        cli = CliRunner().invoke(main, ["recall", QUERY, "--db", db_path])
        assert "Date: 2023" not in cli.output

    def test_codex_uses_earliest_message_timestamp_not_mtime(self, tmp_path):
        f = tmp_path / "2023" / "05" / "rollout-x.jsonl"
        f.parent.mkdir(parents=True)

        def item(role, text, ts):
            return {"timestamp": ts, "type": "response_item",
                    "payload": {"type": "message", "role": role,
                                "content": [{"type": "input_text" if role == "user"
                                             else "output_text", "text": text}]}}
        _write_jsonl(f, [
            {"timestamp": "2023-04-30T00:00:00Z", "type": "session_meta",
             "payload": {"cwd": "/work/billing"}},
            item("user", "Mara moved the Postgres billing migration.", "2023-05-01T08:00:00Z"),
            item("assistant", "Noted.", "2023-05-01T08:00:05Z"),
        ], MTIME)
        items = _run(CodexAdapter(session_dir=str(tmp_path)).fetch_new_content(
            datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert items[0]["timestamp"].startswith("2023-05-01T08:00:00")
        assert items[0]["timestamp_source"] == "content"

    def test_codex_without_message_timestamps_is_mtime(self, tmp_path):
        f = tmp_path / "rollout-y.jsonl"
        _write_jsonl(f, [{"type": "response_item",
                          "payload": {"type": "message", "role": "user",
                                      "content": [{"type": "input_text", "text": "hi there"}]}}],
                     MTIME)
        items = _run(CodexAdapter(session_dir=str(tmp_path)).fetch_new_content(
            datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert items[0]["timestamp_source"] == "mtime"

    def test_file_watcher_is_mtime(self, tmp_path):
        from revien.adapters.file_watcher import FileWatcherAdapter
        (tmp_path / "n.md").write_text("Some note about Postgres billing.", encoding="utf-8")
        items = _run(FileWatcherAdapter(watch_dir=str(tmp_path)).fetch_new_content(
            datetime(2000, 1, 1, tzinfo=timezone.utc)))
        assert items and items[0]["timestamp_source"] == "mtime"


class TestKeyedRefreshKeepsEarlierForMtime:
    def _ingest(self, pipe, content, ts, source):
        return pipe.ingest(IngestionInput(
            source_id="s:k", content=content, ingest_key="s:k",
            timestamp=ts, timestamp_source=source))

    def test_mtime_refresh_never_moves_recorded_at_forward(self, db_path):
        store = GraphStore(db_path=db_path)
        pipe = IngestionPipeline(store)
        t1 = datetime(2023, 5, 3, tzinfo=timezone.utc)
        t2 = t1 + timedelta(days=30)
        out = self._ingest(pipe, TEXT, t1, "mtime")
        self._ingest(pipe, TEXT + "\nUser: and more Postgres billing.", t2, "mtime")
        ctx = store.get_node(out.context_node_id)
        assert ctx.recorded_at == t1
        assert (ctx.metadata or {}).get("recorded_at_source") == "mtime"
        store.close()

    def test_content_refresh_still_takes_the_new_value(self, db_path):
        store = GraphStore(db_path=db_path)
        pipe = IngestionPipeline(store)
        t1 = datetime(2023, 5, 3, tzinfo=timezone.utc)
        t2 = t1 + timedelta(days=30)
        out = self._ingest(pipe, TEXT, t1, "content")
        self._ingest(pipe, TEXT + "\nUser: and more Postgres billing.", t2, "content")
        assert store.get_node(out.context_node_id).recorded_at == t2
        store.close()

    def test_real_content_time_replaces_a_legacy_mtime_stamp(self, db_path):
        store = GraphStore(db_path=db_path)
        pipe = IngestionPipeline(store)
        mt = datetime(2023, 5, 3, tzinfo=timezone.utc)
        said = datetime(2023, 5, 1, tzinfo=timezone.utc)
        out = self._ingest(pipe, TEXT, mt, "mtime")
        self._ingest(pipe, TEXT + "\nUser: more Postgres billing.", said, "content")
        ctx = store.get_node(out.context_node_id)
        assert ctx.recorded_at == said
        assert ctx.metadata["recorded_at_source"] == "content"
        store.close()

    def test_unknown_source_is_refused(self):
        with pytest.raises(ValueError):
            IngestionInput(source_id="s", content="x", timestamp_source="whenever")


class TestMtimeNeverRendered:
    def test_said_date_gate(self):
        iso = "2023-05-07T21:30:00-04:00"
        assert said_date(iso, "content") == "2023-05-07"
        assert said_date(iso, "capture") == "2023-05-07"
        assert said_date(iso, "import") == "2023-05-07"
        assert said_date(iso, None) is None  # unknown source: never dated
        assert said_date(iso, "mtime") is None
        assert said_date(None, "content") is None

    def test_bench_reader_context_skips_mtime_dates(self):
        from revien_bench import answerers as A
        ctx = A.RetrievedContext(
            query="q", contents=["alpha", "beta"], labels=["", ""],
            dates=["2023-05-07T00:00:00+00:00", "2023-05-08T00:00:00+00:00"],
            date_sources=["content", "mtime"])
        out = A._format_context(ctx)
        assert "[7 May 2023] alpha" in out
        assert "8 May 2023" not in out


# ── S6 ────────────────────────────────────────────────────────────────

class TestSpeakersCalendarDay:
    EDT = timezone(timedelta(hours=-4))
    WHEN = datetime(2023, 5, 7, 21, 30, tzinfo=timezone(timedelta(hours=-4)))

    def _seed(self, path):
        store = GraphStore(db_path=path)
        IngestionPipeline(store).ingest(
            IngestionInput(source_id="s1", content=TEXT, timestamp=self.WHEN))
        store.close()

    def test_every_surface_keeps_the_day(self, db_path):
        self._seed(db_path)
        resp = _recall(db_path)
        assert {r.recorded_at for r in resp.results} == {"2023-05-07T21:30:00-04:00"}

        with TestClient(create_app(db_path=db_path)) as c:
            data = c.post("/v1/recall", json={"query": QUERY}).json()
        assert {r["recorded_at"][:10] for r in data["results"]} == {"2023-05-07"}

        mcp = _call(build_mcp_server(db_path=db_path), "revien_recall", {"query": QUERY})
        assert {r["recorded_at"][:10] for r in mcp["results"]} == {"2023-05-07"}

        back = parse_recall(serialize_recall(data))
        assert {r["recorded_at"][:10] for r in back["results"]} == {"2023-05-07"}

        cli = CliRunner().invoke(main, ["recall", QUERY, "--db", db_path])
        assert "Date: 2023-05-07" in cli.output and "2023-05-08" not in cli.output

        assert "[2023-05-07]" in RevienMemoryProvider._format_context(resp)
        assert "[2023-05-07]" in RevienMemory._format_retrieval_response(object(), resp)
        assert "[2023-05-07]" in OllamaAdapter(graph_path=db_path).get_context_for_prompt(QUERY)

        from revien_bench.answerers import _reader_date
        assert _reader_date(resp.results[0].recorded_at) == "7 May 2023"

    def test_naive_is_utc(self, db_path):
        store = GraphStore(db_path=db_path)
        IngestionPipeline(store).ingest(IngestionInput(
            source_id="s1", content=TEXT, timestamp=datetime(2023, 5, 7, 22, 30)))
        store.close()
        resp = _recall(db_path)
        assert {r.recorded_at for r in resp.results} == {"2023-05-07T22:30:00+00:00"}


# ── S7 ────────────────────────────────────────────────────────────────

def test_toon_from_older_daemon_still_parses():
    old = (
        "query: q\n"
        "results[1]{node_id,node_type,label,content,score,score_breakdown.recency,"
        "origin_runtime,origin_source,project_key}:\n"
        "  n-1,fact,L,c,0.5,0.9,null,null,null\n"
        "paths[1]:\n  - [1]: n-1\n"
        "nodes_examined: 1\nretrieval_time_ms: 1.0\nsemantic_active: false\n"
        "semantic_note: null\nskill_proposals: []\n"
    )
    back = parse_recall(old)
    row = back["results"][0]
    assert row["recorded_at"] is None and row["recorded_at_source"] is None
    assert row["node_id"] == "n-1"


def test_toon_roundtrips_recorded_at_source(db_path):
    store = GraphStore(db_path=db_path)
    IngestionPipeline(store).ingest(IngestionInput(
        source_id="s1", content=TEXT,
        timestamp=datetime(2023, 5, 7, tzinfo=timezone.utc), timestamp_source="import"))
    store.close()
    with TestClient(create_app(db_path=db_path)) as c:
        data = c.post("/v1/recall", json={"query": QUERY}).json()
    assert {r["recorded_at_source"] for r in data["results"]} == {"import"}
    assert parse_recall(serialize_recall(data)) == data


# ── S8 ────────────────────────────────────────────────────────────────

class TestCaptureTimestamps:
    def test_mcp_store_stamps_capture_now(self, db_path):
        before = datetime.now(timezone.utc) - timedelta(seconds=2)
        server = build_mcp_server(db_path=db_path)
        _call(server, "revien_store", {"content": TEXT})
        out = _call(server, "revien_recall", {"query": QUERY})
        assert out["results"]
        for r in out["results"]:
            assert r["recorded_at_source"] == "capture"
            assert datetime.fromisoformat(r["recorded_at"]) >= before

    def test_hermes_tool_store_and_recall_carry_date_and_source(self, db_path):
        store = GraphStore(db_path=db_path)
        prov = RevienMemoryProvider.__new__(RevienMemoryProvider)
        prov._pipeline = IngestionPipeline(store)
        prov._engine = RetrievalEngine(store)
        prov._store = store
        prov._session_id = "sess"
        prov._tool_store(TEXT)
        out = prov._tool_recall(QUERY)
        assert out["results"]
        assert all(r["recorded_at"] and r["recorded_at_source"] == "capture"
                   for r in out["results"])
        store.close()


# ── N9 ────────────────────────────────────────────────────────────────

class TestFenceExactNote:
    def _resp(self, *items):
        rs = [SimpleNamespace(node_id=f"n{i}", node_type="fact", label=f"L{i}", content=c,
                              score=0.8, score_breakdown={}, path=[], recorded_at=d,
                              recorded_at_source="content")
              for i, (c, d) in enumerate(items)]
        return SimpleNamespace(results=rs, retrieval_time_ms=1.0)

    def test_emitted_block_is_stripped_note_and_all(self):
        block = RevienMemoryProvider._format_context(
            self._resp(("SECRETA one", "2023-05-07T00:00:00+00:00")))
        assert DATE_NOTE in block
        out = fence_content(f"User: hello\n{block}\nAssistant: KEEP").content
        assert "SECRETA" not in out and "dates in brackets" not in out
        assert "User: hello" in out and "KEEP" in out

    def test_user_line_that_only_resembles_the_note_survives(self):
        block = RevienMemoryProvider._format_context(
            self._resp(("SECRETA one", "2023-05-07T00:00:00+00:00")))
        mine = "(dates in brackets matter to me, says the user)"
        out = fence_content(f"{block}\n{mine}\nAssistant: x").content
        assert mine in out
        assert "SECRETA" not in out
