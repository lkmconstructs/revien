"""
Adapter synchronization repair tests (repair leg R3).

Pins the four sync-cursor defects and the ingest-key idempotency contract:

  a) First-ever sync starts at EPOCH — pre-daemon history is ingested (the
     old register_adapter stamped now(), skipping everything before start).
  b) Cursors are persisted in SQLite (sync_cursors table) — a daemon restart
     resumes from the last successful sync instead of resetting to now().
  c) Stamp-before-fetch — the new cursor is captured BEFORE the fetch and
     persisted only on success, so content landing mid-sync is caught by the
     next window instead of skipped forever. Failed syncs leave the cursor.
  d) Immediate first sync — the interval job fires once at scheduler start
     (inside the R2 lifespan), not after the first 6h window.
  e) ingest_key — re-ingesting the same unit refreshes the ONE existing
     context node (no-op on unchanged content, in-place update on grown
     content) instead of stacking duplicate whole-session context nodes.
     No key = today's append behavior, unchanged.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from revien.adapters.base import RevienAdapter
from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.codex import CodexAdapter
from revien.daemon.scheduler import EPOCH, SyncScheduler
from revien.daemon.server import create_app
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


SESSION_V1 = (
    "User: We decided to use PostgreSQL for the backend database.\n"
    "Assistant: Noted. PostgreSQL for the backend it is."
)
SESSION_V2 = (
    SESSION_V1
    + "\nUser: Also, we decided to use Redis for the caching layer.\n"
    "Assistant: Confirmed. Redis handles caching."
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _item(content, ts, key=None, source="fake:unit:one"):
    d = {
        "content": content,
        "content_type": "conversation",
        "timestamp": ts.isoformat(),
        "metadata": {"adapter": "fake"},
        "source_id": source,
    }
    if key is not None:
        d["ingest_key"] = key
    return d


class FakeAdapter(RevienAdapter):
    """Adapter over an in-memory item list. fetch_new_content filters by the
    item timestamp (> since), mirroring the real adapters' mtime filter.
    ``on_fetch`` runs AFTER the returned snapshot is taken and BEFORE the
    return — the hook models content landing mid-fetch (the stamp race)."""

    def __init__(self, items=None, healthy=True):
        self.items = list(items or [])
        self.healthy = healthy
        self.seen_since = []
        self.on_fetch = None

    async def health_check(self):
        return self.healthy

    async def fetch_new_content(self, since):
        self.seen_since.append(since)
        snapshot = [
            dict(it) for it in self.items
            if datetime.fromisoformat(it["timestamp"]) > since
        ]
        if self.on_fetch is not None:
            self.on_fetch()
        return snapshot


class AlwaysFetchAdapter(FakeAdapter):
    """Ignores `since` — models the whole-file mtime re-fetch the session
    adapters do: every sync returns the full current unit again."""

    async def fetch_new_content(self, since):
        self.seen_since.append(since)
        return [dict(it) for it in self.items]


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(db_path):
    s = GraphStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def pipeline(store):
    return IngestionPipeline(store)


# ── Persistent cursors (defects a + b) ─────────────────────


class TestPersistentCursor:
    def test_first_ever_sync_starts_at_epoch_and_ingests_history(
        self, store, pipeline
    ):
        """No persisted cursor = first sync EVER: since is EPOCH, so content
        that existed long before the daemon started is ingested."""
        ancient = datetime(2019, 6, 1, tzinfo=timezone.utc)
        adapter = FakeAdapter([_item("Pre-daemon history: we chose FastAPI.", ancient)])
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        result = run_async(scheduler.sync_one("fake"))

        assert result["status"] == "ok"
        assert result["items_ingested"] == 1
        assert adapter.seen_since[0] == EPOCH
        # Cursor persisted after the successful sync.
        cursor = store.get_sync_cursor("fake")
        assert cursor is not None
        assert cursor > ancient

    def test_registration_does_not_stamp_a_cursor(self, store, pipeline):
        """The old defect: register_adapter set the cursor to now(). It must
        not — the position is store state, absent until the first sync."""
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", FakeAdapter())
        assert store.get_sync_cursor("fake") is None

    def test_restart_resumes_from_persisted_cursor(self, store, pipeline):
        """A NEW scheduler over the same store (daemon restart) syncs from
        the persisted cursor — not from EPOCH, not from restart-time now()."""
        adapter = FakeAdapter(
            [_item("First window content.", datetime(2020, 1, 1, tzinfo=timezone.utc))]
        )
        first = SyncScheduler(pipeline=pipeline)
        first.register_adapter("fake", adapter)
        run_async(first.sync_one("fake"))
        persisted = store.get_sync_cursor("fake")
        assert persisted is not None

        restarted = SyncScheduler(pipeline=pipeline)
        restarted.register_adapter("fake", adapter)
        run_async(restarted.sync_one("fake"))

        assert adapter.seen_since[1] == persisted
        assert adapter.seen_since[1] != EPOCH

    def test_cursor_survives_store_reopen(self, db_path):
        """The sync_cursors table is real persistence: close the db, reopen
        it (fresh GraphStore over the same file), read the same cursor."""
        stamp = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
        s1 = GraphStore(db_path=db_path)
        s1.set_sync_cursor("claude_code", stamp)
        s1.close()

        s2 = GraphStore(db_path=db_path)
        try:
            assert s2.get_sync_cursor("claude_code") == stamp
        finally:
            s2.close()

    def test_set_sync_cursor_upserts(self, store):
        a = datetime(2026, 1, 1, tzinfo=timezone.utc)
        b = datetime(2026, 2, 2, tzinfo=timezone.utc)
        store.set_sync_cursor("x", a)
        store.set_sync_cursor("x", b)
        assert store.get_sync_cursor("x") == b

    def test_unregister_keeps_cursor(self, store, pipeline):
        """Unregistering must not lose the position — re-registering resumes
        instead of re-ingesting all history."""
        adapter = FakeAdapter(
            [_item("Content before unregister.", datetime(2020, 1, 1, tzinfo=timezone.utc))]
        )
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)
        run_async(scheduler.sync_one("fake"))
        persisted = store.get_sync_cursor("fake")

        scheduler.unregister_adapter("fake")
        assert store.get_sync_cursor("fake") == persisted


# ── Stamp-before-fetch (defect c) ──────────────────────────


class TestStampBeforeFetch:
    def test_content_landing_mid_fetch_is_caught_by_next_sync(
        self, store, pipeline
    ):
        """Content arriving between fetch and cursor stamp used to be skipped
        forever (cursor was post-fetch now()). With stamp-before-fetch the
        cursor is the PRE-fetch instant, so the next window catches it."""
        adapter = FakeAdapter(
            [_item("Alpha: content from before the sync.",
                   datetime(2020, 1, 1, tzinfo=timezone.utc))]
        )

        def land_mid_fetch():
            # Wall clock is already past this sync's pre-fetch stamp; +1s
            # guards against clock granularity making the two equal.
            adapter.items.append(_item(
                "Bravo: landed between fetch and stamp.",
                datetime.now(timezone.utc) + timedelta(seconds=1),
                source="fake:unit:two",
            ))
            adapter.on_fetch = None  # inject exactly once

        adapter.on_fetch = land_mid_fetch
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        first = run_async(scheduler.sync_one("fake"))
        assert first["items_ingested"] == 1  # Alpha only; Bravo landed mid-fetch

        second = run_async(scheduler.sync_one("fake"))
        assert second["items_ingested"] == 1, (
            "content that landed between fetch and stamp must be caught by "
            "the next sync window, not skipped forever"
        )

    def test_unhealthy_sync_leaves_cursor_untouched(self, store, pipeline):
        adapter = FakeAdapter(
            [_item("Never lost.", datetime(2020, 1, 1, tzinfo=timezone.utc))],
            healthy=False,
        )
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        result = run_async(scheduler.sync_one("fake"))
        assert result["status"] == "unhealthy"
        assert store.get_sync_cursor("fake") is None

        # Recovery: nothing was lost — the item still syncs from EPOCH.
        adapter.healthy = True
        result = run_async(scheduler.sync_one("fake"))
        assert result["items_ingested"] == 1

    def test_fetch_error_leaves_cursor_untouched(self, store, pipeline):
        adapter = FakeAdapter(
            [_item("Still here.", datetime(2020, 1, 1, tzinfo=timezone.utc))]
        )

        def boom():
            raise RuntimeError("fetch exploded")

        adapter.on_fetch = boom
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        results = run_async(scheduler.sync_all())
        assert results["fake"]["status"] == "error"
        assert store.get_sync_cursor("fake") is None

    def test_ingest_failure_leaves_cursor_untouched(
        self, store, pipeline, monkeypatch
    ):
        """The cursor is persisted only AFTER ingest completes — a mid-ingest
        failure must not advance it (those items would be lost)."""
        adapter = FakeAdapter(
            [_item("Doomed this round.", datetime(2020, 1, 1, tzinfo=timezone.utc))]
        )
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        def broken_ingest(_input):
            raise RuntimeError("ingest exploded")

        monkeypatch.setattr(pipeline, "ingest", broken_ingest)
        results = run_async(scheduler.sync_all())
        assert results["fake"]["status"] == "error"
        assert store.get_sync_cursor("fake") is None

    def test_empty_fetch_still_advances_cursor(self, store, pipeline):
        """A successful sync with nothing new IS a success — the window
        advances so the next sync doesn't rescan the same span."""
        adapter = FakeAdapter([])
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)
        result = run_async(scheduler.sync_one("fake"))
        assert result["items_ingested"] == 0
        assert store.get_sync_cursor("fake") is not None


# ── Immediate first sync (defect d) ────────────────────────


class TestImmediateFirstSync:
    def test_startup_sync_fires_within_client_context(self, db_path):
        """The daemon's shape (R2): scheduler attached to app.state, started
        by the lifespan inside the running loop. The auto-sync job must fire
        once AT startup — content is in the graph within seconds, without a
        manual /v1/sync and without waiting out the 6h interval."""
        app = create_app(db_path=db_path)
        scheduler = SyncScheduler(pipeline=app.state.pipeline)
        adapter = FakeAdapter([_item(
            "Startup sync content: the daemon syncs at boot.",
            datetime(2021, 1, 1, tzinfo=timezone.utc),
        )])
        scheduler.register_adapter("fake", adapter)
        app.state.scheduler = scheduler

        with TestClient(app):
            deadline = time.time() + 20
            while time.time() < deadline:
                if app.state.store.count_nodes() > 0:
                    break
                time.sleep(0.05)
            assert app.state.store.count_nodes() > 0, (
                "the startup kick must sync registered adapters within "
                "seconds of the lifespan starting the scheduler"
            )
            assert adapter.seen_since, "fetch_new_content never ran at startup"


# ── Stable ingestion keys (defect e) ───────────────────────


class TestIngestKey:
    KEY = "fake:project:session-001"

    def test_first_keyed_ingest_stamps_key_and_hash(self, store, pipeline):
        out = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        ctx = store.get_node(out.context_node_id)
        assert ctx.metadata["ingest_key"] == self.KEY
        assert ctx.metadata["ingest_content_hash"] == hashlib.sha256(
            SESSION_V1.encode("utf-8")
        ).hexdigest()

    def test_same_content_twice_is_noop(self, store, pipeline):
        out1 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        out2 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        assert out2.context_node_id == out1.context_node_id
        assert out2.nodes_created == 0
        assert out2.nodes_deduplicated == 0
        assert out2.edges_created == 0
        assert out2.total_nodes_in_graph == out1.total_nodes_in_graph
        assert out2.total_edges_in_graph == out1.total_edges_in_graph
        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 1

    def test_grown_session_refreshes_same_context_node(self, store, pipeline):
        out1 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        out2 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V2, ingest_key=self.KEY,
        ))
        # SAME node, refreshed in place — never a second context node.
        assert out2.context_node_id == out1.context_node_id
        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 1
        ctx = store.get_node(out1.context_node_id)
        assert ctx.content == SESSION_V2
        assert ctx.metadata["ingest_content_hash"] == hashlib.sha256(
            SESSION_V2.encode("utf-8")
        ).hexdigest()

    def test_refresh_dedups_extracted_nodes_and_edges(self, store, pipeline):
        pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        out2 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V2, ingest_key=self.KEY,
        ))
        # Nodes from the unchanged prefix merged into their existing selves.
        assert out2.nodes_deduplicated > 0
        entity_labels = [
            n.label.lower()
            for n in store.list_nodes(node_type=NodeType.ENTITY, limit=999)
        ]
        assert len(entity_labels) == len(set(entity_labels)), (
            f"duplicate entities after refresh: {entity_labels}"
        )
        # Only-missing edges: no (source, target, type) pair appears twice.
        ctx_id = out2.context_node_id
        edges = store.get_edges_for_node(ctx_id)
        seen = [(e.source_node_id, e.target_node_id, e.edge_type) for e in edges]
        assert len(seen) == len(set(seen)), "refresh created duplicate edges"

    def test_refresh_is_audited(self, store, pipeline):
        out1 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V2, ingest_key=self.KEY,
        ))
        ops = [a["op"] for a in store.get_node_audit(out1.context_node_id)]
        assert "ingest_refresh" in ops

    def test_no_key_keeps_append_behavior(self, store, pipeline):
        """Default path pinned: without a key, re-ingesting the same content
        appends a NEW context node every time — today's behavior."""
        out1 = pipeline.ingest(IngestionInput(
            source_id="no-key-source", content=SESSION_V1,
        ))
        out2 = pipeline.ingest(IngestionInput(
            source_id="no-key-source", content=SESSION_V1,
        ))
        assert out2.context_node_id != out1.context_node_id
        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 2
        for ctx in contexts:
            assert "ingest_key" not in (ctx.metadata or {})

    def test_mid_refresh_failure_rolls_back_atomically(
        self, store, pipeline, monkeypatch
    ):
        """A failure inside _refresh_keyed's transaction must leave the unit
        EXACTLY as it was — old content, old hash, old node/edge counts — and
        the next refresh must succeed cleanly."""
        out1 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V1, ingest_key=self.KEY,
        ))
        nodes_before = store.count_nodes()
        edges_before = store.count_edges()
        v1_hash = hashlib.sha256(SESSION_V1.encode("utf-8")).hexdigest()

        def boom(edge):
            raise RuntimeError("edge write exploded")

        monkeypatch.setattr(store, "add_edge", boom)
        with pytest.raises(RuntimeError, match="edge write exploded"):
            pipeline.ingest(IngestionInput(
                source_id=self.KEY, content=SESSION_V2, ingest_key=self.KEY,
            ))
        monkeypatch.undo()

        ctx = store.get_node(out1.context_node_id)
        assert ctx.content == SESSION_V1, "rolled-back refresh mutated content"
        assert ctx.metadata["ingest_content_hash"] == v1_hash
        assert store.count_nodes() == nodes_before
        assert store.count_edges() == edges_before

        # Recovery: the same refresh succeeds once the failure clears.
        out2 = pipeline.ingest(IngestionInput(
            source_id=self.KEY, content=SESSION_V2, ingest_key=self.KEY,
        ))
        assert out2.context_node_id == out1.context_node_id
        assert store.get_node(out1.context_node_id).content == SESSION_V2

    def test_keys_do_not_cross_units(self, store, pipeline):
        """Two different keys are two different units — each gets its own
        context node and refreshes independently."""
        out_a = pipeline.ingest(IngestionInput(
            source_id="unit-a", content=SESSION_V1, ingest_key="key:a",
        ))
        out_b = pipeline.ingest(IngestionInput(
            source_id="unit-b", content=SESSION_V1, ingest_key="key:b",
        ))
        assert out_a.context_node_id != out_b.context_node_id
        assert store.find_context_node_by_ingest_key("key:a").node_id == out_a.context_node_id
        assert store.find_context_node_by_ingest_key("key:b").node_id == out_b.context_node_id


class TestPreR3Adoption:
    """C1: live daemons carry months of pre-R3 stacked whole-session context
    duplicates (unkeyed, same stable source_id). The first keyed ingest must
    ADOPT the newest one — never add duplicate N+1."""

    SRC = "claude-code:proj:sess-legacy"
    V_A = "User: We decided to use PostgreSQL for the backend."
    V_B = V_A + "\nAssistant: Noted. PostgreSQL confirmed."
    V_C = V_B + "\nUser: We also decided to add Redis caching."
    V_D = V_C + "\nAssistant: Redis caching confirmed."

    def _stack_pre_r3_duplicates(self, pipeline):
        """Three unkeyed ingests of the same growing session — exactly what
        pre-R3 syncs left behind. Tiny sleeps keep created_at ordering
        deterministic on coarse clocks."""
        outs = []
        for content in (self.V_A, self.V_B, self.V_C):
            outs.append(pipeline.ingest(IngestionInput(
                source_id=self.SRC, content=content,
            )))
            time.sleep(0.005)
        return outs

    def test_keyed_ingest_adopts_newest_unkeyed_duplicate(
        self, store, pipeline
    ):
        outs = self._stack_pre_r3_duplicates(pipeline)
        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 3  # the pre-R3 stack
        newest_id = outs[-1].context_node_id

        # First post-upgrade sync: session grew again, now keyed.
        out = pipeline.ingest(IngestionInput(
            source_id=self.SRC, content=self.V_D, ingest_key=self.SRC,
        ))

        # Adopted the NEWEST duplicate (most complete), refreshed in place —
        # total context count unchanged (no fourth copy).
        assert out.context_node_id == newest_id
        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 3
        adopted = store.get_node(newest_id)
        assert adopted.content == self.V_D
        assert adopted.metadata["ingest_key"] == self.SRC

        # The older two duplicates are untouched (known limitation: they
        # remain in the graph until a consolidate-pass cleanup exists).
        for old_out, old_content in zip(outs[:2], (self.V_A, self.V_B)):
            old = store.get_node(old_out.context_node_id)
            assert old.content == old_content
            assert "ingest_key" not in (old.metadata or {})

        # Follow-up syncs hit the keyed path directly on the adopted node.
        again = pipeline.ingest(IngestionInput(
            source_id=self.SRC, content=self.V_D, ingest_key=self.SRC,
        ))
        assert again.context_node_id == newest_id
        assert again.nodes_created == 0

    def test_adoption_with_unchanged_content_stamps_without_refresh(
        self, store, pipeline
    ):
        """Upgrade with a session that hasn't changed since the last pre-R3
        sync: adopt = stamp key + hash on the newest duplicate, zero new
        nodes/edges, and the stamp is audited."""
        outs = self._stack_pre_r3_duplicates(pipeline)
        newest_id = outs[-1].context_node_id
        nodes_before = store.count_nodes()
        edges_before = store.count_edges()

        out = pipeline.ingest(IngestionInput(
            source_id=self.SRC, content=self.V_C, ingest_key=self.SRC,
        ))

        assert out.context_node_id == newest_id
        assert out.nodes_created == 0
        assert out.edges_created == 0
        assert store.count_nodes() == nodes_before
        assert store.count_edges() == edges_before
        adopted = store.get_node(newest_id)
        assert adopted.content == self.V_C  # untouched
        assert adopted.metadata["ingest_key"] == self.SRC
        assert adopted.metadata["ingest_content_hash"] == hashlib.sha256(
            self.V_C.encode("utf-8")
        ).hexdigest()
        assert "ingest_adopt" in [
            a["op"] for a in store.get_node_audit(newest_id)
        ]

    def test_adoption_never_steals_nodes_owned_by_another_key(
        self, store, pipeline
    ):
        """A context node stamped with a DIFFERENT key belongs to that key —
        a new key over the same source_id appends its own node."""
        out_a = pipeline.ingest(IngestionInput(
            source_id="shared:source", content=SESSION_V1, ingest_key="key:a",
        ))
        out_b = pipeline.ingest(IngestionInput(
            source_id="shared:source", content=SESSION_V2, ingest_key="key:b",
        ))
        assert out_b.context_node_id != out_a.context_node_id
        assert store.get_node(out_a.context_node_id).content == SESSION_V1


class TestSchedulerKeyedResync:
    def test_repeated_whole_unit_syncs_keep_one_context_node(
        self, store, pipeline
    ):
        """The real defect-e shape end to end: the adapter re-fetches the
        WHOLE session on every sync (mtime bump), the key makes re-ingest
        refresh instead of duplicate."""
        key = "fake:proj:sess-1"
        adapter = AlwaysFetchAdapter([_item(
            SESSION_V1, datetime(2026, 7, 1, tzinfo=timezone.utc),
            key=key, source=key,
        )])
        scheduler = SyncScheduler(pipeline=pipeline)
        scheduler.register_adapter("fake", adapter)

        run_async(scheduler.sync_one("fake"))
        # Session grew; adapter re-fetches the whole thing next sync.
        adapter.items = [_item(
            SESSION_V2, datetime(2026, 7, 2, tzinfo=timezone.utc),
            key=key, source=key,
        )]
        run_async(scheduler.sync_one("fake"))
        run_async(scheduler.sync_one("fake"))  # unchanged third pass: no-op

        contexts = store.list_nodes(node_type=NodeType.CONTEXT, limit=999)
        assert len(contexts) == 1, (
            f"expected ONE refreshed context node, got {len(contexts)}"
        )
        assert contexts[0].content == SESSION_V2


# ── Adapters emit the key (claude_code + codex) ────────────


class TestAdapterIngestKeys:
    def test_claude_code_emits_ingest_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "projects" / "proj-x"
            project_dir.mkdir(parents=True)
            session = project_dir / "session-abc.jsonl"
            with open(session, "w") as f:
                f.write(json.dumps({
                    "type": "human", "content": "We decided to ship on Friday.",
                    "timestamp": "2026-07-01T10:00:00Z",
                }) + "\n")
                f.write(json.dumps({
                    "type": "assistant", "content": "Friday ship confirmed.",
                    "timestamp": "2026-07-01T10:00:05Z",
                }) + "\n")

            adapter = ClaudeCodeAdapter(session_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) == 1
            assert results[0]["ingest_key"] == results[0]["source_id"]
            assert results[0]["source_id"] == "claude-code:proj-x:session-abc"

    def test_codex_emits_ingest_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            day_dir = Path(tmpdir) / "2026" / "07" / "01"
            day_dir.mkdir(parents=True)
            rollout = day_dir / "rollout-2026-07-01T10-00-00-abc.jsonl"
            with open(rollout, "w") as f:
                f.write(json.dumps({
                    "timestamp": "2026-07-01T10:00:00Z", "type": "session_meta",
                    "payload": {"cwd": "/home/user/proj-y"},
                }) + "\n")
                f.write(json.dumps({
                    "timestamp": "2026-07-01T10:00:01Z", "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text",
                                     "text": "We decided to use SQLite."}],
                    },
                }) + "\n")

            adapter = CodexAdapter(session_dir=tmpdir)
            results = run_async(adapter.fetch_new_content(
                datetime(2020, 1, 1, tzinfo=timezone.utc)
            ))
            assert len(results) == 1
            assert results[0]["ingest_key"] == results[0]["source_id"]
            assert results[0]["source_id"].startswith("codex:proj-y:rollout-")
