"""Migration 005 (usage layer): use_judgments, recall_receipts, store_meta
and sync_cursors.generation.

Existing databases are real ones: tests/fixtures/*.sql are dumps of databases
made by running origin/main (03217f4, migrations 001-004 applied) and Revien
0.3.0 (the v0.3.0 tag and commit 7757085). "main-used" was built by main's
real flows: the ChatGPT, Claude.ai and Readwise importers, a Claude Code
adapter sync through the scheduler, an mcp-source ingest and a recall.
"""
import importlib
import inspect
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from revien.graph.store import GraphStore

FIX = Path(__file__).parent / "fixtures"
FIXTURES = {
    "main-03217f4": "db_main_03217f4.sql",
    "main-used-03217f4": "db_main_used_03217f4.sql",
    "v0.3.0-tag": "db_v030_tag_7bf8738.sql",
    "v0.3.0-7757085": "db_v030_7757085.sql",
}
NEW_TABLES = {"use_judgments", "recall_receipts", "store_meta"}
NEW_INDEXES = {"idx_use_judgments_node", "idx_recall_receipts_status"}
T = datetime(2026, 7, 1, tzinfo=timezone.utc)


def migration():
    return importlib.import_module("revien.graph.migrations.005_usage_layer")


def made_by(name, tmp_path):
    path = tmp_path / f"{name}.db"
    conn = sqlite3.connect(path)
    conn.executescript((FIX / FIXTURES[name]).read_text())
    conn.commit()
    conn.close()
    return str(path)


def catalogue(path):
    conn = sqlite3.connect(path)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        uv = conn.execute("PRAGMA user_version").fetchone()[0]
        cols = {t: [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
                for t in tables if not t.startswith("sqlite_")}
        return tables, indexes, uv, cols
    finally:
        conn.close()


def data_of(path):
    conn = sqlite3.connect(path)
    try:
        return {
            "nodes": conn.execute(
                "SELECT node_id, node_type, label, content, source_id "
                "FROM nodes ORDER BY node_id").fetchall(),
            "edges": conn.execute(
                "SELECT edge_id, source_node_id, target_node_id "
                "FROM edges ORDER BY edge_id").fetchall(),
            "audit": conn.execute(
                "SELECT node_id, op FROM audit_log ORDER BY id").fetchall(),
        }
    finally:
        conn.close()


class TestFixturesAreWhatTheyClaim:
    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_no_usage_layer_yet(self, name, tmp_path):
        tables, _idx, uv, cols = catalogue(made_by(name, tmp_path))
        assert not tables & NEW_TABLES
        assert "generation" not in cols.get("sync_cursors", [])
        assert uv == (4 if name.startswith("main") else 0)


class TestUpgrade:
    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_upgrade_adds_the_layer_and_keeps_every_row(self, name, tmp_path):
        path = made_by(name, tmp_path)
        before = data_of(path)
        s = GraphStore(db_path=path)
        s.close()
        tables, indexes, uv, cols = catalogue(path)
        assert uv == 5
        assert NEW_TABLES <= tables
        assert NEW_INDEXES <= indexes
        assert "generation" in cols["sync_cursors"]
        after = data_of(path)
        assert after["nodes"] == before["nodes"]
        assert after["edges"] == before["edges"]
        assert after["audit"] == before["audit"]

    def test_existing_cursor_survives_with_generation_zero(self, tmp_path):
        path = made_by("main-03217f4", tmp_path)
        s = GraphStore(db_path=path)
        try:
            assert s.get_sync_state("claude-code") == (T, 0)
            assert s.reset_sync_cursor("claude-code") is True
            assert s.get_sync_state("claude-code") == (None, 1)
        finally:
            s.close()

    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_upgraded_store_works_end_to_end(self, name, tmp_path):
        path = made_by(name, tmp_path)
        s = GraphStore(db_path=path)
        try:
            nodes = s.list_nodes(limit=10)
            assert len(nodes) >= 3
            node = nodes[0]
            assert s.record_use_judgment(
                episode_key="e#1", node_id=node.node_id, used=True, reason="r",
                shared_terms=["x"], similarity=0.5, query="q", mode="shadow",
                applied=False)
            assert s.use_judgment_counts()["judged"] == 1
            from datetime import timedelta
            s.add_recall_receipt(
                recall_id="r1", runtime="api", query="q",
                node_ids=[node.node_id], issued_at=T, expires_at=T + timedelta(hours=24))
            assert s.get_recall_receipt("r1")["node_ids"] == [node.node_id]
            s.set_meta("k", "v")
            assert s.get_meta("k") == "v"
            # privacy: deleting the node removes its judgments and scrubs its id
            assert s.delete_node(node.node_id)
            assert s.use_judgment_counts()["judged"] == 0
            assert s.get_recall_receipt("r1")["node_ids"] == []
        finally:
            s.close()

    def test_origin_backfill_of_003_still_runs_on_a_v030_database(self, tmp_path):
        path = made_by("v0.3.0-tag", tmp_path)
        GraphStore(db_path=path).close()
        conn = sqlite3.connect(path)
        rows = dict(conn.execute(
            "SELECT source_id, origin_runtime FROM nodes").fetchall())
        conn.close()
        assert rows["claude-code:proj-x:sess-1"] == "claude-code"
        assert rows["mcp"] is None


class TestIdempotent:
    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_a_second_open_changes_nothing(self, name, tmp_path):
        path = made_by(name, tmp_path)
        GraphStore(db_path=path).close()
        first = (catalogue(path), data_of(path))
        GraphStore(db_path=path).close()
        assert (catalogue(path), data_of(path)) == first

    def test_standalone_migrate_reports_what_it_created_then_nothing(
            self, tmp_path):
        path = made_by("main-03217f4", tmp_path)
        first = migration().migrate(path)
        assert set(first["created"]) == NEW_TABLES | {"sync_cursors.generation"}
        assert migration().migrate(path)["created"] == []
        assert catalogue(path)[2] == 5

    def test_standalone_equals_the_connect_time_guard(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = made_by("v0.3.0-tag", tmp_path / "a")
        b = made_by("v0.3.0-tag", tmp_path / "b")
        GraphStore(db_path=a).close()
        migration().migrate(b)
        assert catalogue(a) == catalogue(b)

    def test_a_marker_never_overrides_the_schema(self, tmp_path):
        path = made_by("main-03217f4", tmp_path)
        GraphStore(db_path=path).close()
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE recall_receipts")
        conn.execute("DROP TABLE sync_cursors")
        conn.execute("CREATE TABLE sync_cursors (adapter_name TEXT PRIMARY KEY, "
                     "cursor TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.commit()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        conn.close()
        GraphStore(db_path=path).close()
        tables, indexes, _uv, cols = catalogue(path)
        assert "recall_receipts" in tables and "idx_recall_receipts_status" in indexes
        assert "generation" in cols["sync_cursors"]

    def test_existing_usage_rows_are_kept_when_reopened(self, tmp_path):
        path = str(tmp_path / "r.db")
        s = GraphStore(db_path=path)
        s.set_meta("keep", "me")
        s.set_sync_cursor("a", T)
        s.reset_sync_cursor("a")
        s.close()
        s = GraphStore(db_path=path)
        try:
            assert s.get_meta("keep") == "me"
            assert s.get_sync_state("a") == (None, 1)
        finally:
            s.close()


class TestFreshDatabase:
    def test_a_fresh_database_has_the_layer_at_version_5(self, tmp_path):
        path = str(tmp_path / "fresh.db")
        GraphStore(db_path=path).close()
        tables, indexes, uv, cols = catalogue(path)
        assert uv == 5
        assert NEW_TABLES <= tables and NEW_INDEXES <= indexes
        assert "generation" in cols["sync_cursors"]

    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_upgraded_and_fresh_databases_have_the_same_usage_schema(
            self, name, tmp_path):
        fresh = str(tmp_path / "fresh.db")
        GraphStore(db_path=fresh).close()
        old = made_by(name, tmp_path)
        GraphStore(db_path=old).close()
        ft, fi, _fv, fc = catalogue(fresh)
        ot, oi, _ov, oc = catalogue(old)
        for t in NEW_TABLES:
            assert fc[t] == oc[t]
        assert fc["sync_cursors"] == oc["sync_cursors"]
        assert NEW_INDEXES <= fi and NEW_INDEXES <= oi


class TestConvention:
    def test_ensure_db_creates_none_of_the_usage_layer_inline(self):
        src = inspect.getsource(GraphStore._ensure_db)
        for word in ("use_judgments", "recall_receipts", "store_meta",
                     "generation"):
            assert word not in src, f"{word} must live in migration 005 only"
        assert "_migrate_add_usage_layer" in src

    def test_the_migration_file_follows_the_numbered_convention(self):
        mod = migration()
        assert callable(mod.migrate)
        assert (Path(mod.__file__).name) == "005_usage_layer.py"


class TestMainDatabaseThatWasUsed:
    """A database created AND USED by origin/main (migration 004 already
    applied), then opened by this branch."""
    NAME = "main-used-03217f4"

    def test_it_really_holds_importer_and_adapter_data(self, tmp_path):
        path = made_by(self.NAME, tmp_path)
        conn = sqlite3.connect(path)
        runtimes = {r[0] for r in conn.execute(
            "SELECT DISTINCT origin_runtime FROM nodes")}
        labelled = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE "
            "json_extract(metadata,'$.recorded_at_source') IS NOT NULL").fetchone()[0]
        conn.close()
        assert {"chatgpt", "claude", "readwise", "claude-code"} <= runtimes
        assert labelled > 0
        assert catalogue(path)[2] == 4

    def test_004_is_not_rerun_and_origin_and_dates_are_untouched(self, tmp_path):
        path = made_by(self.NAME, tmp_path)
        def snap():
            c = sqlite3.connect(path)
            try:
                return c.execute(
                    "SELECT node_id, origin_runtime, origin_source, project_key, "
                    "session_key, recorded_at, metadata FROM nodes ORDER BY node_id"
                ).fetchall()
            finally:
                c.close()
        before = snap()
        GraphStore(db_path=path).close()
        assert snap() == before
        assert catalogue(path)[2] == 5

    def test_recall_sync_and_reset_work_on_the_upgraded_database(self, tmp_path):
        import asyncio
        from revien.adapters.claude_code import ClaudeCodeAdapter
        from revien.daemon.scheduler import SyncScheduler
        from revien.ingestion.pipeline import IngestionPipeline
        from revien.retrieval.engine import RetrievalEngine

        path = made_by(self.NAME, tmp_path)
        s = GraphStore(db_path=path)
        try:
            assert RetrievalEngine(s).recall("PostgreSQL billing").results
            cursor, gen = s.get_sync_state("claude-code")
            assert cursor is not None and gen == 0
            (tmp_path / "none").mkdir()
            sched = SyncScheduler(pipeline=IngestionPipeline(s))
            sched.register_adapter("claude-code", ClaudeCodeAdapter(
                session_dir=str(tmp_path / "none")))
            res = asyncio.new_event_loop().run_until_complete(
                sched.sync_one("claude-code"))
            assert res["status"] == "ok"      # fenced write at generation 0
            assert s.reset_sync_cursor("claude-code") is True
            assert s.get_sync_state("claude-code") == (None, 1)
        finally:
            s.close()
