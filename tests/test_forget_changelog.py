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
from tests import forget_seed as fs

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
            (["forget", "--help"], "--everywhere"),
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
    # the cascade flag is gone from the verb, and the text does not offer it
    assert "--cascade" not in runner.invoke(main, ["forget", "--help"]).output
    assert "`--cascade` are removed" in sec


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
        res = GraphOperations(s).forget_node(fs.holders(seeded.db, seeded.secret)[0])
    finally:
        s.close()
    sec = forget_section()
    for field in ("removed_nodes", "nodes_before", "nodes_after", "edges_before",
                  "edges_after", "tombstones_created", "storage", "leftovers"):
        assert field in res and field in sec, field
    for kind in ("source_file", "exports", "snapshots", "other_units", "vault_note"):
        assert kind in sec
    assert "`not_bound`" in sec and res["not_bound"]
    assert "shared channel" in res["not_bound"][0]["note"]


def test_the_changelog_does_not_promise_what_a_limit_denies():
    """The limits are in the same section as the claims."""
    sec = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "### Known limits of the forget work" in sec


def test_the_retrain_story_matches_the_code(tmp_path):
    """The CHANGELOG says what retrains the model; every part of it is checked."""
    from revien.neural.training import TrainingLoop
    sec = forget_section()
    assert "nothing retrains it automatically" not in sec
    assert "there is no scheduled or nightly job" in sec
    assert TrainingLoop.INITIAL_THRESHOLD == 100 and TrainingLoop.RETRAIN_INTERVAL == 500
    assert "100 retrieval signals" in sec and "every 500 more" in sec
    for method in ("mark_stale", "is_stale"):
        assert callable(getattr(TrainingLoop, method))
    assert "`model_stale`" in sec and "POST /v1/training/run" in sec
    app = create_app(db_path=str(tmp_path / "t.db"))
    have = {(m, getattr(r, "path", "")) for r in app.routes for m in getattr(r, "methods", ())}
    assert ("POST", "/v1/training/run") in have
    # no scheduler job calls the trainer
    for rel in ("daemon/scheduler.py", "daemon/daemon.py", "consolidate.py"):
        src = (REPO / "revien" / rel).read_text(encoding="utf-8")
        assert "force_train" not in src and "training_loop.train" not in src


def test_the_origin_scope_and_everywhere_fields_named_exist(tmp_path):
    from tests import forget_seed as fs
    seeded = fs.seed("pipeline", tmp_path)
    s = GraphStore(db_path=seeded.db)
    try:
        res = GraphOperations(s).forget_node(
            fs.holders(seeded.db, seeded.secret)[0], everywhere=True)
    finally:
        s.close()
    sec = forget_section()
    for field in ("reentry", "cut_in_place", "still_held_by", "model", "model_stale",
                  "fallback", "identity_note", "bound_to", "not_bound"):
        assert field in sec, field
    for field in ("reentry", "cut_in_place", "still_held_by", "model", "model_stale"):
        assert field in res, field
    assert set(res["reentry"]) >= {"scope", "stops", "does_not_stop"}
    assert set(res["model"]) >= {"neural_ranking", "retrained", "stale", "note"}
    out = CliRunner().invoke(main, ["forget", "--help"])
    assert "--everywhere" in out.output and "forget --everywhere" in sec
    out = CliRunner().invoke(main, ["restore", "--help"])
    assert "--list" in out.output and "revien restore forget:<id>" in sec


def test_there_is_no_copy_rule_in_the_code_or_the_text():
    """A capture with another identity lands as written: nothing in the code
    or the text recognises a copy by its content."""
    from revien.forget import engine, gate
    for name in ("COPY_OF_ORIGIN", "COPY_OF_CAPTURE", "COPY_MIN_SHARED", "is_copy_of",
                 "long_fingerprints", "known_detail", "strip_forgotten_report"):
        assert not hasattr(gate, name), name
    assert not hasattr(engine, "COPY_RULE")
    assert not hasattr(engine.ForgetEngine, "_held_sentences")
    sec = forget_section()
    spec = (REPO / "docs" / "FORGET_SPEC.md").read_text(encoding="utf-8")
    for text in (sec, spec):
        assert "80%" not in text and "copy rule" not in text.lower()
        assert "held what was forgotten" not in text
        assert "lands as written" in text
        assert "a copy of the source under a different identity" in text or \
            "a copy of this source under a different identity" in text


def test_the_known_limit_about_concurrent_forgets_is_the_true_one():
    sec = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "is not serialised with a daemon" not in sec
    assert "take turns on SQLite's write lock" in sec


def test_the_source_classes_named_are_the_ones_in_the_code():
    """(1) an id that travels with a copy, (2) path and inode, (3) content
    alone: the adapters and the gate say the same."""
    codex = (REPO / "revien" / "adapters" / "codex.py").read_text(encoding="utf-8")
    assert '"session_key": jsonl_file.stem' in codex            # the unit id stays the file name
    assert 'meta_extra["session_id"] = session_id' in codex and '"session_meta"' in codex
    cc = (REPO / "revien" / "adapters" / "claude_code.py").read_text(encoding="utf-8")
    assert 'meta_extra["session_id"] = session_id' in cc and "sessionId" in cc
    gate_src = (REPO / "revien" / "forget" / "gate.py").read_text(encoding="utf-8")
    assert 'runtime == "claude-code"' in gate_src         # only Claude Code's sessionId replaces the unit id
    assert 'sid=sid' in gate_src and 'f"sid:codex:' in gate_src
    sec = forget_section()
    assert "a Codex rollout by the `id` in its `session_meta` line" in sec
    assert "a Codex rollout file (its unit is the file name" not in sec


def test_sentences_that_were_false_are_gone_and_the_true_ones_are_there():
    sec = forget_section()
    spec = (REPO / "docs" / "FORGET_SPEC.md").read_text(encoding="utf-8")
    for text in (sec, spec):
        assert "no source is suppressed on a guess" not in text
        assert "Nothing is deleted, cut back to its label or suppressed because of it" not in text
        assert "at most 8 nodes" not in text
        assert "suppresses future imports" not in text
        assert "forgets the whole unit" not in text
    # what is claimed now (each is exercised in tests/test_forget_narrow.py,
    # test_forget_round3.py and test_forget_round4.py)
    for phrase in (
            "Forget is for one specific thing",
            "To remove a whole conversation, use delete; to keep a specific detail out, forget that detail.",
            "never the content string Revien gives it",
            "A capture that does not have that identity lands as written",
            "a node written by several sources keeps all of its owners",
            "has no single origin",
            "Every node whose wording was cut is listed with its source in `cut_in_place`",
            "no surviving node holding what was forgotten",
            "Every such node is named under `other_units` and `still_held_by`",
            "it keeps trying for up to 25 seconds, then answers `busy` (HTTP 503)",
            "the unit id stays the file name"):
        assert phrase in sec, phrase


def test_the_numbers_in_the_changelog_are_the_code_s():
    from revien.forget.engine import ForgetEngine
    sec = forget_section()
    assert ForgetEngine.BUSY_WAIT_SECONDS == 25.0 and "up to 25 seconds" in sec
    assert ForgetEngine.DISTINCTIVE_MAX_ELSEWHERE == 2 and "at most 2 nodes another live unit vouches for" in sec
