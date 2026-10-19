"""Restore (docs/FORGET_SPEC.md §7): one path at a time, bulk only with an
explicit pattern that is listed first, always audited."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.forget import restore as rs
from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore
from tests import forget_seed as fs


def two_watched_files(tmp_path, monkeypatch):
    """Two forgotten files with the SAME name in different folders."""
    from revien.adapters.file_watcher import FileWatcherAdapter
    db = str(tmp_path / "r.db")
    for folder, word in (("projA", "ALPHAWORD"), ("projB", "BRAVOWORD")):
        d = tmp_path / "watched" / folder
        d.mkdir(parents=True)
        p = d / "notes.md"
        p.write_text(f"Plan notes. We decided to use {word} for the gate.\n")
        fs._age(p)
    fs._sync(db, "files", FileWatcherAdapter(watch_dir=str(tmp_path / "watched")))
    store = GraphStore(db_path=db)
    ops = GraphOperations(store)
    for n in store.list_nodes(limit=100):
        if n.node_type.value == "context":
            ops.forget_node(n.node_id, cascade=True)
    store.close()
    return db, tmp_path / "watched"


def active(db):
    s = GraphStore(db_path=db)
    try:
        return s.list_suppressions("active")
    finally:
        s.close()


def run(db, *args, input=None):
    return CliRunner().invoke(main, ["restore", *args, "--db", db], input=input)


class TestRestore:
    def test_a_path_lifts_that_one_file_only(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        assert len(active(db)) == 2
        r = run(db, str(root / "projA" / "notes.md"))
        assert r.exit_code == 0, r.output
        left = active(db)
        assert len(left) == 1 and "projB" in left[0]["path"]

    def test_a_bare_name_never_fans_out(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        r = run(db, "notes.md")
        assert r.exit_code == 1
        assert "name, not a path" in r.output
        assert "projA" in r.output and "projB" in r.output       # candidates listed
        assert len(active(db)) == 2                              # nothing lifted

    def test_a_pattern_lists_what_it_lifts_before_lifting(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        # declined at the prompt: the list is shown, nothing is lifted
        r = run(db, "projA/**", input="n\n")
        assert r.exit_code == 1
        assert "Will lift 1 suppression(s)" in r.output and "projA" in r.output
        assert "projB" not in r.output
        assert len(active(db)) == 2
        r = run(db, "projA/**", input="y\n")
        assert r.exit_code == 0, r.output
        assert len(active(db)) == 1

    def test_the_pattern_can_be_confirmed_up_front(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        r = run(db, "*/proj*/notes.md", "--yes")
        assert r.exit_code == 0 and active(db) == []

    def test_a_pattern_of_only_wildcards_is_refused(self, tmp_path, monkeypatch):
        db, _ = two_watched_files(tmp_path, monkeypatch)
        for pat in ("*", "**", "?*"):
            r = run(db, pat, "--yes")
            assert r.exit_code == 1 and "narrower pattern" in r.output
        assert len(active(db)) == 2

    def test_every_restore_is_audited_and_content_free(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        run(db, str(root / "projA" / "notes.md"))
        s = GraphStore(db_path=db)
        try:
            rows = [a for a in s.get_all_audit() if a["op"] == "restore"]
            assert len(rows) == 1 and rows[0]["actor"] == "cli:restore"
            assert "ALPHAWORD" not in json.dumps(rows) and "BRAVOWORD" not in json.dumps(rows)
            lifted = s.list_suppressions("lifted")
            assert len(lifted) == 1 and lifted[0]["lifted_by"] == "cli:restore"
        finally:
            s.close()

    def test_a_restored_file_is_captured_again(self, tmp_path, monkeypatch):
        from revien.adapters.file_watcher import FileWatcherAdapter
        db, root = two_watched_files(tmp_path, monkeypatch)
        run(db, str(root / "projA" / "notes.md"))
        fs._resync(db, "files", FileWatcherAdapter(watch_dir=str(root)))
        s = GraphStore(db_path=db)
        try:
            text = " ".join(n.content for n in s.list_nodes(limit=100))
        finally:
            s.close()
        assert "ALPHAWORD" in text and "BRAVOWORD" not in text

    def test_list_and_nothing_to_restore(self, tmp_path, monkeypatch):
        db, root = two_watched_files(tmp_path, monkeypatch)
        r = run(db, "--list")
        assert "2 active suppression(s)" in r.output and "until the file changes" in r.output
        none = run(db, str(tmp_path / "never" / "forgotten.md"))
        assert none.exit_code == 1 and "No active suppression" in none.output

    def test_restore_by_unit_id(self, tmp_path, monkeypatch):
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(seeded.ctx_ids[0], cascade=True)
        s.close()
        r = run(seeded.db, "item:chatgpt:conversation:conv-1")
        assert r.exit_code == 0, r.output
        assert active(seeded.db) == []

    def test_no_rest_or_mcp_route_can_lift_a_suppression(self, tmp_path):
        """Only an explicit restore (CLI) or a CLI import override lifts one."""
        import inspect

        import revien.daemon.server as srv
        import revien.mcp_server as mcp
        from revien.daemon.server import create_app
        app = create_app(db_path=str(tmp_path / "x.db"))
        paths = [getattr(r, "path", "") for r in app.routes]
        assert not [p for p in paths if "restore" in p or "suppress" in p]
        for mod in (srv, mcp):
            src = inspect.getsource(mod)
            assert "lift_suppression" not in src
            assert "override_suppression" not in src
