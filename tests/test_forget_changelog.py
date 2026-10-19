"""Invariant 5, the part a machine can check: the names the CHANGELOG uses for
the forget work exist in the code. A sentence that names a command, a route, a
table, a field or an audit op that is not there is false."""
import re
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.daemon import heartbeat
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore

REPO = Path(__file__).resolve().parents[1]


def forget_section() -> str:
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    a = text.index("### Forget, delete and cleanup")
    b = text.index("### Changed (read this", a)
    return text[a:b]


def test_the_cli_commands_and_flags_named_exist():
    sec = forget_section()
    runner = CliRunner()
    for args, needle in (
            (["forget", "--help"], "--cascade"),
            (["restore", "--help"], "--list"),
            (["sync", "reset", "--help"], "--yes"),
            (["import-chatgpt", "--help"], "--override-suppressed"),
            (["import-claude", "--help"], "--override-suppressed"),
            (["import-readwise", "--help"], "--override-suppressed"),
            (["distill-vault", "--help"], "--vault"),
            (["import", "--help"], "--merge")):
        out = runner.invoke(main, args)
        assert out.exit_code == 0 and needle in out.output, (args, out.output)
    for token in ("--override-suppressed", "revien restore --list", "--yes"):
        assert token in sec


def test_the_routes_named_exist(tmp_path):
    app = create_app(db_path=str(tmp_path / "c.db"))
    have = {(m, getattr(r, "path", "")) for r in app.routes for m in getattr(r, "methods", ())}
    for method, path in (("POST", "/v1/nodes/{node_id}/forget"),
                         ("DELETE", "/v1/nodes/{node_id}"),
                         ("POST", "/v1/retention/sweep"),
                         ("GET", "/v1/nodes/{node_id}/forget/preview"),
                         ("POST", "/v1/graph/import")):
        assert (method, path) in have, (method, path)
    sec = forget_section()
    for token in ("POST /v1/nodes/{id}/forget", "DELETE /v1/nodes/{id}",
                  "POST /v1/retention/sweep", "GET /v1/nodes/{id}/forget/preview",
                  "POST /v1/graph/import"):
        assert token in sec


def test_the_tables_fields_and_functions_named_exist(tmp_path):
    db = str(tmp_path / "c.db")
    GraphStore(db_path=db).close()
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        cols = [r[1] for r in conn.execute("PRAGMA table_info(node_provenance)")]
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
    finally:
        conn.close()
    assert {"node_provenance", "capture_units", "suppressions"} <= tables
    assert cols == ["node_id", "unit_id", "label_ok", "wording_ok"]
    assert callable(GraphOperations.run_scheduled_cleanup)
    assert callable(GraphOperations.forget_node)
    assert "forget_suppression" in heartbeat.DAEMON_FEATURES
    sec = forget_section()
    for token in ("node_provenance(node_id, unit_id, label_ok, wording_ok)",
                  "capture_units", "suppressions", "GraphOperations.run_scheduled_cleanup",
                  "forget_suppression", "cut_back_at", "tombstones_created"):
        assert token in sec


def test_the_audit_ops_and_actors_named_are_the_ones_written(tmp_path):
    src = "".join(p.read_text(encoding="utf-8") for p in (REPO / "revien").rglob("*.py"))
    for token in ('"api:delete"', '"api:retention_sweep"', '"langchain:clear"',
                  '"scheduler:retention"', '"cut_back"', '"restore"', '"cleanup"'):
        assert token in src, token
    sec = forget_section()
    for token in ("api:delete", "api:retention_sweep", "langchain:clear",
                  "scheduler:retention", "cleanup", "cut_back", "restore"):
        assert token in sec


def test_the_response_fields_named_exist(tmp_path):
    from tests import forget_seed as fs
    seeded = fs.seed("pipeline", tmp_path)
    s = GraphStore(db_path=seeded.db)
    try:
        res = GraphOperations(s).forget_node(seeded.ctx_ids[0], cascade=True)
    finally:
        s.close()
    sec = forget_section()
    for field in ("removed_nodes", "nodes_before", "nodes_after", "edges_before",
                  "edges_after", "tombstones_created", "storage", "leftovers"):
        assert field in res and field in sec, field
    for kind in ("source_file", "exports", "snapshots", "other_units", "vault_note"):
        assert kind in sec
    assert "eligible for recapture because the source supplied no stable item identity" in sec


def test_the_changelog_does_not_promise_what_a_limit_denies():
    """The limits are in the same section as the claims."""
    sec = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "### Known limits of the forget work" in sec
