"""A forgotten watched file blocks that file, not every file with its name."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.adapters.file_watcher import FileWatcherAdapter
from revien.cli import main
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Graph, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionPipeline

TXT = {"projA": "We decided projA stores the Trevalon77 key in env.\n",
       "projB": "We decided projB keeps the Brindlemoor88 token in vault.\n",
       "projC": "We decided projC uses Rust for the CLI.\n"}


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


def sync(db, watch):
    s = GraphStore(db_path=db)
    try:
        sch = SyncScheduler(pipeline=IngestionPipeline(s))
        sch.register_adapter("files", FileWatcherAdapter(
            watch_dir=str(watch), recursive=True))
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(sch.sync_one("files"))
        finally:
            loop.close()
    finally:
        s.close()


def bump(p, secs=100):
    st = p.stat()
    os.utime(p, (st.st_atime + secs, st.st_mtime + secs))


def write(watch, proj, secs=0):
    f = watch / proj / "README.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(TXT[proj])
    bump(f, secs)
    return f


def captured(db, word):
    s = GraphStore(db_path=db)
    try:
        return any(word in (n.content or "") for n in s.list_nodes(limit=999))
    finally:
        s.close()


def forget_file(db, word):
    s = GraphStore(db_path=db)
    try:
        ctx = [n for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=99)
               if word in n.content][0]
        return GraphOperations(s).forget_node(
            ctx.node_id, cascade=True, reason="private")
    finally:
        s.close()


def cli(*args):
    return CliRunner().invoke(main, list(args))


class TestAForgottenFileBlocksOnlyThatFile:
    def test_same_name_in_another_folder_is_still_captured(self, db_path, tmp_path):
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        sync(db_path, w)
        out = forget_file(db_path, "Trevalon77")
        assert os.path.realpath(a) in out["suppressed_identities"]
        write(w, "projC", 5000)
        r = sync(db_path, w)
        assert r["items_ingested"] == 1 and not r.get("items_suppressed")
        assert captured(db_path, "projC")
        bump(a, 9000)
        r = sync(db_path, w)
        assert r["items_suppressed"] == 1
        assert not captured(db_path, "Trevalon77")

    def test_the_path_is_listed_and_the_footer_is_true(self, db_path, tmp_path):
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        sync(db_path, w)
        out = forget_file(db_path, "Trevalon77")
        listing = cli("suppressions", "--db", db_path).output
        assert os.path.realpath(a) in listing and "[file_path]" in listing
        assert "Sibling facts" not in listing
        assert "Name-wide" not in listing
        assert "Sibling facts" not in " ".join(out["suppression_notice"])
        assert "blocked by its path" in " ".join(out["suppression_notice"])

    def test_restore_of_a_path_lifts_only_that_file(self, db_path, tmp_path):
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        b = write(w, "projB", -1000)
        sync(db_path, w)
        forget_file(db_path, "Trevalon77")
        forget_file(db_path, "Brindlemoor88")
        r = cli("restore", str(a), "--db", db_path)
        assert r.exit_code == 0, r.output
        names = [x["identity"] for x in GraphStore(db_path=db_path)
                 .list_ingest_suppressions()]
        assert os.path.realpath(a) not in names
        assert os.path.realpath(b) in names
        assert "file:README.md" in names, "B is still forgotten by name too"
        bump(a, 9000)
        bump(b, 9000)
        r = sync(db_path, w)
        assert r["items_ingested"] == 1 and r["items_suppressed"] == 1
        assert captured(db_path, "Trevalon77")
        assert not captured(db_path, "Brindlemoor88")
        # the last path of a name takes the name with it
        assert cli("restore", str(b), "--db", db_path).exit_code == 0
        assert GraphStore(db_path=db_path).list_ingest_suppressions() == []

    def test_restore_by_name_says_which_files_it_lifts(self, db_path, tmp_path):
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        sync(db_path, w)
        forget_file(db_path, "Trevalon77")
        r = cli("restore", "file:README.md", "--db", db_path)
        assert r.exit_code == 0
        assert os.path.realpath(a) in r.output
        assert GraphStore(db_path=db_path).list_ingest_suppressions() == []

    def test_rest_restore_reports_what_it_lifted(self, db_path, tmp_path):
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        sync(db_path, w)
        forget_file(db_path, "Trevalon77")
        app = create_app(db_path=db_path)
        with TestClient(app) as c:
            r = c.post("/v1/suppressions/restore",
                       json={"identity": str(a)})
            assert r.status_code == 200, r.text
            assert os.path.realpath(a) in r.json()["lifted_identities"]
            assert c.get("/v1/suppressions").json()["suppressions"] == []

    def test_a_moved_file_and_a_later_same_name_file_are_blocked(
            self, db_path, tmp_path):
        """The bound: once the forgotten path is gone, the name decides."""
        w = tmp_path / "w"
        a = write(w, "projA", -1000)
        sync(db_path, w)
        forget_file(db_path, "Trevalon77")
        moved = w / "elsewhere" / "README.md"
        moved.parent.mkdir()
        a.rename(moved)
        bump(moved, 9000)
        write(w, "projC", 9000)
        r = sync(db_path, w)
        assert r["items_suppressed"] == 2 and not r.get("items_ingested")
        assert not captured(db_path, "Trevalon77")

    def test_a_tombstone_without_a_recorded_path_stays_name_wide_and_says_so(
            self, db_path, tmp_path):
        s = GraphStore(db_path=db_path)
        s.add_ingest_suppression("file:README.md", "source_id", "old", "me")
        s.close()
        w = tmp_path / "w"
        write(w, "projC", 5000)
        r = sync(db_path, w)
        assert r["items_suppressed"] == 1
        assert "Name-wide" in cli("suppressions", "--db", db_path).output

    def test_an_imported_unrelated_file_passes_and_the_forgotten_one_does_not(
            self, db_path, tmp_path):
        w = tmp_path / "w"
        write(w, "projA", -1000)
        write(w, "projC", -1000)
        src = str(tmp_path / "src.db")
        sync(src, w)
        s = GraphStore(db_path=src)
        graph = s.export_graph()
        s.close()
        g = GraphStore(db_path=db_path)
        try:
            a_ctx = [n for n in graph.nodes if n.node_type == NodeType.CONTEXT
                     and "Trevalon77" in n.content][0]
            g.add_ingest_suppression(
                os.path.realpath(a_ctx.metadata["source_path"]), "file_path",
                "p", "me")
            g.add_ingest_suppression("file:README.md", "source_id", "p", "me")
            res = g.import_graph(Graph(**graph.model_dump(mode="json")),
                                 mode="merge")
            ctx = {n.node_id: n for n in g.list_nodes(
                node_type=NodeType.CONTEXT, limit=99)}
            assert any("projC" in n.content for n in ctx.values())
            assert not any("Trevalon77" in n.content for n in ctx.values())
            assert res["suppressed_nodes"] >= 1
        finally:
            g.close()
