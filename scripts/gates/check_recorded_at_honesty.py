"""G16: a consuming model is never shown a date for a memory unless the date
is when it was said.

CHECK: python scripts/gates/check_recorded_at_honesty.py
EXPECT: recorded_at honesty verification passed
"""
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from click.testing import CliRunner  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from revien.adapters.claude_code import ClaudeCodeAdapter  # noqa: E402
from revien.adapters.langchain_adapter import RevienMemory  # noqa: E402
from revien.adapters.ollama_adapter import OllamaAdapter  # noqa: E402
from revien.cli import main  # noqa: E402
from revien.daemon.scheduler import SyncScheduler  # noqa: E402
from revien.daemon.server import create_app  # noqa: E402
from revien.dates import DATE_NOTE  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.hermes_provider import RevienMemoryProvider  # noqa: E402
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline  # noqa: E402
from revien.retrieval.engine import RetrievalEngine  # noqa: E402
from revien.toon import parse_recall, serialize_recall  # noqa: E402

QUERY = "Postgres billing migration"
TEXT = ("User: Mara moved the Postgres billing migration.\n"
        "Assistant: Noted, Postgres billing migration.")
MTIME = datetime(2023, 5, 3, 12, 0, tzinfo=timezone.utc)
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def fail(msg):
    print(f"check_recorded_at_honesty: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def recall(path, query=QUERY):
    store = GraphStore(db_path=path)
    try:
        return RetrievalEngine(store).recall(query, top_n=5)
    finally:
        store.close()


def jsonl(path, stamp1, stamp2):
    def m(kind, text, ts):
        d = {"type": kind, "content": text}
        if ts:
            d["timestamp"] = ts
        return d
    rows = [m("user", "Mara moved the Postgres billing migration.", stamp1),
            m("assistant", "Noted, Postgres billing migration.", stamp2)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ts = MTIME.timestamp()
    os.utime(path, (ts, ts))


def ingest_session_dir(session_dir, db):
    adapter = ClaudeCodeAdapter(session_dir=str(session_dir))
    items = run(adapter.fetch_new_content(EPOCH))
    check(len(items) == 1, f"adapter returned {len(items)} items, expected 1")
    store = GraphStore(db_path=db)
    try:
        sched = SyncScheduler(IngestionPipeline(store))
        sched.register_adapter("a", adapter)
        run(sched.sync_all())
    finally:
        store.close()
    return items[0]


def surfaces(resp, db, query=QUERY):
    """Every text surface a consuming model (or the terminal) sees."""
    return {
        "hermes": RevienMemoryProvider._format_context(resp),
        "langchain": RevienMemory._format_retrieval_response(object(), resp),
        "ollama": OllamaAdapter(graph_path=db).get_context_for_prompt(query),
        "cli": CliRunner().invoke(main, ["recall", query, "--db", db]).output,
    }


def main_check():
    tmp = tempfile.mkdtemp()
    try:
        # 1. Content-dated session: earliest message time wins over mtime.
        d1 = Path(tmp) / "dated"
        jsonl(d1 / "proj" / "sess.jsonl", "2023-05-02T10:00:00Z", "2023-05-01T09:00:00Z")
        db1 = os.path.join(tmp, "dated.db")
        item = ingest_session_dir(d1, db1)
        check(item["timestamp"].startswith("2023-05-01T09:00:00"),
              f"adapter timestamp {item['timestamp']!r} is not the earliest message time")
        check(item["timestamp_source"] == "content", "dated fixture not marked content")
        resp = recall(db1)
        check(resp.results, "dated fixture: recall returned nothing")
        check({r.recorded_at_source for r in resp.results} == {"content"},
              "dated fixture: recorded_at_source is not content on every result")
        check(all(r.recorded_at.startswith("2023-05-01") for r in resp.results),
              "dated fixture: recorded_at is not 2023-05-01 (mtime leaked?)")
        conn = sqlite3.connect(db1)
        stored = [row[0] for row in conn.execute(
            "SELECT recorded_at FROM nodes WHERE recorded_at IS NOT NULL")]
        conn.close()
        check(stored and all(s.startswith("2023-05-01") for s in stored),
              f"stored recorded_at values not all 2023-05-01: {sorted(set(stored))}")
        # positive control: a content row shows both date and note
        pos = surfaces(resp, db1)
        for name in ("hermes", "langchain", "ollama"):
            check("[2023-05-01]" in pos[name], f"positive control: {name} lacks the date")
            check("2023-05-03" not in pos[name], f"{name} shows the mtime")
            check(DATE_NOTE in pos[name], f"positive control: {name} lacks the date note")
        check("Date: 2023-05-01" in pos["cli"], "positive control: cli lacks the date")

        # 2. Undated session: falls back to mtime, which is never shown.
        d2 = Path(tmp) / "undated"
        jsonl(d2 / "proj" / "sess.jsonl", None, None)
        db2 = os.path.join(tmp, "undated.db")
        item = ingest_session_dir(d2, db2)
        check(item["timestamp_source"] == "mtime", "undated fixture not marked mtime")
        resp = recall(db2)
        check(resp.results and {r.recorded_at_source for r in resp.results} == {"mtime"},
              "undated fixture: source is not mtime on every result")
        for name, text in surfaces(resp, db2).items():
            check("2023-05" not in text, f"mtime row: {name} shows a date")
            check("dates in brackets" not in text and DATE_NOTE not in text,
                  f"mtime row: {name} shows the date note")
            if name != "cli":
                check("Postgres" in text, f"mtime row: {name} lost the memory itself")

        # 3. The speaker's own evening: -04:00 21:30 renders the 7th, not the 8th.
        db3 = os.path.join(tmp, "edt.db")
        when = datetime(2023, 5, 7, 21, 30, tzinfo=timezone(timedelta(hours=-4)))
        store = GraphStore(db_path=db3)
        IngestionPipeline(store).ingest(IngestionInput(
            source_id="s1", content=TEXT, timestamp=when, timestamp_source="capture"))
        store.close()
        resp = recall(db3)
        check({r.recorded_at for r in resp.results} == {"2023-05-07T21:30:00-04:00"},
              "offset not preserved in recorded_at")
        for name, text in surfaces(resp, db3).items():
            check("2023-05-07" in text, f"evening row: {name} lacks the speaker's day")
            check("2023-05-08" not in text, f"evening row: {name} shows the UTC day")

        # 4. Legacy backfill: strip the key, reset user_version, reopen.
        db4 = os.path.join(tmp, "legacy.db")
        store = GraphStore(db_path=db4)
        pipe = IngestionPipeline(store)
        said = datetime(2023, 5, 7, tzinfo=timezone.utc)
        pipe.ingest(IngestionInput(
            source_id="openai:conversation:c-9", timestamp=said, timestamp_source="import",
            content=TEXT))
        pipe.ingest(IngestionInput(
            source_id="mystery-source", timestamp=said, timestamp_source="import",
            content="User: Tobias planned the Kubernetes ingress rollout.\n"
                    "Assistant: Noted, Kubernetes ingress rollout."))
        store.close()
        conn = sqlite3.connect(db4)
        conn.execute("UPDATE nodes SET metadata = json_remove(metadata, '$.recorded_at_source')")
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
        conn.close()
        GraphStore(db_path=db4).close()  # reopen runs the backfill
        known = recall(db4, QUERY)
        check(known.results and {r.recorded_at_source for r in known.results} == {"content"},
              "backfill: openai-origin row did not get source content from its origin")
        check("[2023-05-07]" in RevienMemoryProvider._format_context(known),
              "backfill: a derived-content row lost its date")
        unknown = recall(db4, "Kubernetes ingress rollout")
        unk_rows = [r for r in unknown.results if "Kubernetes" in r.content]
        check(unk_rows and all(r.recorded_at_source is None for r in unk_rows),
              "backfill: unknown-origin row was given a source")
        text = RevienMemoryProvider._format_context(unknown)
        check("2023-05-07" not in text and DATE_NOTE not in text,
              "backfill: unknown-origin row renders a date or note")
        check("Kubernetes" in text, "backfill: unknown-origin row lost its content")

        # 5. TOON round-trip carries recorded_at and recorded_at_source.
        with TestClient(create_app(db_path=db1)) as c:
            data = c.post("/v1/recall", json={"query": QUERY}).json()
        check(data["results"] and {r["recorded_at_source"] for r in data["results"]} == {"content"},
              "daemon recall lacks recorded_at_source")
        check(all(r["recorded_at"].startswith("2023-05-01") for r in data["results"]),
              "daemon recall recorded_at wrong")
        check(parse_recall(serialize_recall(data)) == data, "TOON round-trip is lossy")

        # 6. parse_recall accepts a row from an older daemon lacking both columns.
        old = (
            "query: q\n"
            "results[1]{node_id,node_type,label,content,score,score_breakdown.recency,"
            "origin_runtime,origin_source,project_key}:\n"
            "  n-1,fact,L,c,0.5,0.9,null,null,null\n"
            "paths[1]:\n  - [1]: n-1\n"
            "nodes_examined: 1\nretrieval_time_ms: 1.0\nsemantic_active: false\n"
            "semantic_note: null\nskill_proposals: []\n"
        )
        row = parse_recall(old)["results"][0]
        check(row["node_id"] == "n-1" and row["recorded_at"] is None
              and row["recorded_at_source"] is None, "old-format TOON row not accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        main_check()
    except SystemExit:
        raise
    except Exception as exc:  # any crash is a failed gate
        fail(f"{type(exc).__name__}: {exc}")
    print("recorded_at honesty verification passed")
