"""Round 8 of `forget --wording`: a response never echoes the string, a
stdin string can still be confirmed, every store write that persists text is
guarded, the guard and the write share one transaction, a lost forget.key is
loud, and a string that sits inside longer words is asked about first."""
import io
import json
import logging
import sqlite3
import threading
import time

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.forget import keys
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import _free_port, all_text, cc, forget_wording, leaks, new_key


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r8.db")


def _cfg(monkeypatch, db):
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})


def _cli(db, *args, input=None):
    return CliRunner().invoke(
        main, ["forget", *args, "--db", db, "--daemon-url", f"http://127.0.0.1:{_free_port()}"],
        input=input)


# ── 3. no response echoes the submitted value ────────────────────────────

def test_a_malformed_body_never_echoes_what_was_sent(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    bodies = [
        {"json": {"wording": [key]}},
        {"json": {"wording": {"k": key}}},
        {"json": {"wording": 12345, "preview": key}},
        {"json": {"wordng": key}},                       # misspelled field
        {"json": {"wording": key, "preview": [key]}},
        {"json": [key]},
        {"json": key},
        {"content": ('{"wording": "' + key).encode(),
         "headers": {"content-type": "application/json"}},    # broken JSON
        {"content": key.encode(), "headers": {"content-type": "text/plain"}},
        {"data": {"wording": key}},                      # a form, not JSON
    ]
    with TestClient(create_app(db_path=db)) as client:
        for kw in bodies:
            r = client.post("/v1/forget/wording", **kw)
            assert r.status_code == 422, (kw, r.status_code, r.text)
            assert key not in r.text, kw
            assert key not in json.dumps(dict(r.headers)), kw
            assert r.json()["reason"] == "invalid_request"
        # and other routes keep FastAPI's own answer
        other = client.post("/v1/ingest", json={"nope": 1})
        assert other.status_code == 422 and "detail" in other.json()
    assert leaks(db, "__none__") == {}


# ── 4. a string on stdin can still be confirmed ──────────────────────────

def test_stdin_string_is_confirmed_on_the_terminal_not_by_yes(db, monkeypatch):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    _cfg(monkeypatch, db)
    monkeypatch.setattr("revien.cli._open_tty", lambda: io.StringIO("n\n"))
    no = _cli(db, "--wording", "-", input=key + "\n")
    assert no.exit_code == 1 and "Nothing was changed" in no.output
    assert key not in no.output and "Occurrences found: " in no.output   # blast radius shown
    assert leaks(db, key)                                                 # nothing was cut
    monkeypatch.setattr("revien.cli._open_tty", lambda: io.StringIO("y\n"))
    yes = _cli(db, "--wording", "-", input=key + "\n")
    assert yes.exit_code == 0, yes.output
    assert key not in yes.output
    assert leaks(db, key) == {}


def test_stdin_string_with_no_terminal_changes_nothing_and_says_how_to_go_on(db, monkeypatch):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    _cfg(monkeypatch, db)
    monkeypatch.setattr("revien.cli._open_tty", lambda: None)
    out = _cli(db, "--wording", "-", input=key + "\n")
    assert out.exit_code == 1 and "--yes" in out.output and "no terminal" in out.output
    assert "Occurrences found" in out.output and key not in out.output
    assert out.output.count("Nothing was changed.") == 1
    assert leaks(db, key)


# ── 5. every write that persists text is guarded ─────────────────────────

def _forgotten(db, key):
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    assert forget_wording(db, key)["status"] == "forgotten"


def test_the_edge_routes_cannot_write_the_string_back(db):
    key = new_key()
    _forgotten(db, key)
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="alpha note", content="alpha note text"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="beta note", content="beta note text"))
    s.close()
    with TestClient(create_app(db_path=db)) as client:
        r = client.post("/v1/edges", json={
            "source_node_id": a.node_id, "target_node_id": b.node_id,
            "edge_type": "related_to", "metadata": {"why": f"because {key} said so"},
            "source_context": f"copied from {key}"})
        assert r.status_code in (200, 201), r.text
        assert key not in r.text
    s = GraphStore(db_path=db)
    e = Edge(source_node_id=a.node_id, target_node_id=b.node_id,
             edge_type=EdgeType.RELATED_TO, metadata={"k": key}, source_context=key)
    s.add_edge(e)
    s.add_edge_audited(Edge(source_node_id=b.node_id, target_node_id=a.node_id,
                            edge_type=EdgeType.RELATED_TO, metadata={"k": key}), actor=key)
    s.update_edge(e.edge_id, metadata={"k2": f"x {key} y"}, source_context=key)
    s.close()
    assert leaks(db, key) == {}


def test_a_new_node_cannot_carry_the_string_in_its_metadata(db):
    key = new_key()
    _forgotten(db, key)
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="note", content="plain",
                    metadata={"k": key, "nested": [f"x {key}"]}))
    s.close()
    assert leaks(db, key) == {}


def test_candidates_receipts_judgments_audit_and_node_edits_cannot_carry_it(db):
    from datetime import datetime, timedelta, timezone
    key = new_key()
    _forgotten(db, key)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="alpha note", content="alpha"))
    m = s.add_node(Node(node_type=NodeType.FACT, label="beta note", content="beta"))
    cid = s.add_candidate(n.node_id, m.node_id, "review", reason=f"r {key}", trace=f"t {key}")
    s.resolve_candidate(cid, f"resolved by {key}")
    now = datetime.now(timezone.utc)
    s.add_recall_receipt(recall_id="rc1", runtime="x", query=f"what is {key}",
                         node_ids=[n.node_id], issued_at=now, expires_at=now + timedelta(hours=1))
    s.record_use_judgment(episode_key="ep1", node_id=n.node_id, used=True,
                          reason=f"used {key}", shared_terms=[key, "alpha"],
                          similarity=0.5, query=f"query {key}", mode="m", applied=True)
    s.record_audit(n.node_id, "custom", actor=key, before={"k": key}, after={"k": [key]})
    s.update_node(n.node_id, label=f"relabel {key}", content=f"c {key}",
                  metadata={"m": key}, source_context=key)
    s.close()
    assert leaks(db, key) == {}


def test_a_recall_query_logged_for_training_carries_no_named_secret(db, monkeypatch):
    from revien.retrieval.engine import RetrievalEngine
    from revien.semantic.index import SemanticIndex
    monkeypatch.setenv("REVIEN_NEURAL", "1")
    key = new_key()
    _forgotten(db, key)
    s = GraphStore(db_path=db)
    eng = RetrievalEngine(s, semantic=SemanticIndex(s, enabled=False))
    logged = []

    class Spy:
        def log_retrieval(self, **kw):
            logged.append(kw)

        def mark_used(self, *a):
            pass

    eng.training_loop = Spy()
    s.add_node(Node(node_type=NodeType.FACT, label="venue note", content="the venue is booked"))
    eng.recall(f"venue question about {key}")
    s.close()
    assert logged and all(key not in json.dumps(kw, default=str) for kw in logged)


# ── 6. the guard and the write share one IMMEDIATE transaction ───────────

@pytest.mark.parametrize("path", ["add_node", "update_node", "add_edge"])
def test_a_forget_landing_between_the_guard_and_the_write_cannot_leave_a_row(
        db, monkeypatch, path):
    """A write holds the database write lock BEFORE it reads the denylist, so
    a forget from another connection either is seen by the write or waits for
    it and cuts what it wrote. The delay is injected inside the write."""
    key = new_key()
    cc(db, "s1", "User: The venue is booked for the offsite.")
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="alpha note", content="alpha"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="beta note", content="beta"))
    # the forget starts from a record of the string existing already? No:
    # the secret is named WHILE the write is in flight
    delay = {"stmt": "_add_node_stmt" if path != "add_edge" else "_add_edge_stmt"}
    orig = getattr(GraphStore, delay["stmt"])

    def slow(conn, obj):
        return orig(conn, obj)

    if path == "add_node" or path == "add_edge":
        # the delay sits AFTER the denylist was read and BEFORE the insert:
        # on the current code that is the last guard call of the statement, on
        # the code before it the guard that ran ahead of the transaction
        orig_meta = GraphStore._cut_secret_metadata.__func__ \
            if hasattr(GraphStore, "_cut_secret_metadata") else None
        if orig_meta is not None:
            waited = []

            def meta_then_wait(cls, conn, meta):
                out = orig_meta(cls, conn, meta)
                if not waited:
                    waited.append(1)
                    time.sleep(1.0)
                return out
            monkeypatch.setattr(GraphStore, "_cut_secret_metadata", classmethod(meta_then_wait))
        orig_guard_old = GraphStore._without_named_secrets

        waited_old = []

        def old_guard_then_wait(self, kwargs, *a):
            out = orig_guard_old(self, kwargs, *a)
            if not waited_old:
                waited_old.append(1)
                time.sleep(1.0)
            return out

        monkeypatch.setattr(GraphStore, "_without_named_secrets", old_guard_then_wait)
    else:
        pass
    done = {}

    def forget():
        time.sleep(0.3)
        other = GraphStore(db_path=db)
        try:
            done["r"] = GraphOperations(other).forget_wording(key)
        finally:
            other.close()

    t = threading.Thread(target=forget)
    t.start()
    if path == "add_node":
        s.add_node(Node(node_type=NodeType.FACT, label=f"x {key}", content=f"we use {key}"))
    elif path == "add_edge":
        s.add_edge(Edge(source_node_id=a.node_id, target_node_id=b.node_id,
                        edge_type=EdgeType.RELATED_TO, metadata={"k": key}))
    else:
        orig_guard = GraphStore._without_named_secrets

        def guard_then_wait(self, kwargs, conn=None):
            out = orig_guard(self, kwargs, conn)
            time.sleep(1.0)               # the denylist was read; the write comes after
            return out

        monkeypatch.setattr(GraphStore, "_without_named_secrets", guard_then_wait)
        s.update_node(a.node_id, content=f"we use {key}")
    t.join()
    s.close()
    assert done["r"]["status"] == "forgotten"
    assert leaks(db, key) == {}


# ── 7. a lost or replaced forget.key is loud ─────────────────────────────

def _replace_key():
    keys.key_path().write_text("ab" * 32)
    keys._cache.clear()


def test_a_capture_after_the_key_changed_says_so_and_the_daemon_reports_it(db, caplog):
    key = new_key()
    _forgotten(db, key)
    with TestClient(create_app(db_path=db)) as client:
        assert client.get("/v1/health").json()["forget_key"] == "ok"
    _replace_key()
    from revien.ingestion import pipeline
    pipeline._KEY_WARNED.clear()
    caplog.set_level(logging.ERROR)
    s = GraphStore(db_path=db)
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    out = IngestionPipeline(s).ingest(IngestionInput(
        source_id="claude-code:proj:s9", origin_runtime="claude-code", session_key="s9",
        content="User: Plain words about the venue."))
    s.close()
    assert out.warnings and "no longer apply" in out.warnings[0]
    assert "forget.key mismatch" in caplog.text and key not in caplog.text
    caplog.clear()
    with TestClient(create_app(db_path=db)) as client:
        assert "forget.key mismatch" in caplog.text        # said at daemon start
        assert client.get("/v1/health").json()["forget_key"] == "mismatch"
        r = client.post("/v1/ingest", json={"source_id": "x:y", "content": "User: some plain words."})
        assert r.status_code == 200 and r.json()["warnings"]


def test_a_lost_key_is_not_silently_made_anew_by_looking(db):
    key = new_key()
    _forgotten(db, key)
    keys.key_path().unlink()
    keys._cache.clear()
    s = GraphStore(db_path=db)
    from revien.forget import gate
    problem = gate.key_problem(s)
    s.close()
    assert problem and problem["missing"] and problem["literals"] == 1
    assert not keys.key_path().exists()


def test_no_forget_says_gone_everywhere_while_entries_are_orphaned(db):
    key, other = new_key(), new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.\n"
                 f"User: We decided to use {other} as the staging key.")
    forget_wording(db, key)
    _replace_key()
    second = forget_wording(db, other)
    assert second["scope"] != "gone everywhere"
    assert "gone everywhere" not in second["message"].lower()
    left = [l for l in second["leftovers"] if l["kind"] == "forget_key"]
    assert left and left[0]["literals"] == 1 and "no longer apply" in left[0]["note"]


def test_a_node_forget_does_not_say_gone_everywhere_while_entries_are_orphaned(db):
    key, other = new_key(), new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    cc(db, "s2", f"User: We decided to use {other} as the staging key.")
    forget_wording(db, key)
    s = GraphStore(db_path=db)
    fact = [n for n in s.list_nodes(limit=50)
            if n.node_type == NodeType.FACT and other in n.content][0]
    s.close()
    _replace_key()
    s = GraphStore(db_path=db)
    res = GraphOperations(s).forget_node(fact.node_id)
    s.close()
    assert res["scope"] == "forgotten from Revien"
    assert any(l["kind"] == "forget_key" for l in res["leftovers"])
    assert "no longer apply" in res["message"]


# ── 8. a string inside longer words is asked about ───────────────────────

def test_a_string_inside_longer_words_is_counted_and_refused_until_allowed(db, monkeypatch):
    cc(db, "s1", "User: The secretary keeps the secret list. Ask the secretary.")
    pre = forget_wording(db, "secret", dry_run=True)
    assert 0 < pre["inside_longer_words"] < pre["would_cut_occurrences"]
    assert "inside a longer word" in pre["blast_radius"]
    refused = forget_wording(db, "secret")
    assert refused["status"] == "refused" and refused["reason"] == "inside_longer_words"
    assert "secretary" in all_text(db)                                # nothing was cut
    with TestClient(create_app(db_path=db)) as client:
        r = client.post("/v1/forget/wording", json={"wording": "secret"})
        assert r.status_code == 400 and r.json()["detail"]["reason"] == "inside_longer_words"
    _cfg(monkeypatch, db)
    cli = _cli(db, "--wording", "secret", "--yes")
    assert cli.exit_code == 1 and "Name a longer string, or run again" in cli.output
    assert "secretary" in all_text(db)
    ok = _cli(db, "--wording", "secret", "--yes", "--include-longer-words")
    assert ok.exit_code == 0, ok.output
    body = all_text(db)
    assert "ary" in body and "secret" not in body


def test_rest_cuts_inside_longer_words_only_when_told_to(db):
    cc(db, "s1", "User: The secretary keeps the secret list. Ask the secretary.")
    with TestClient(create_app(db_path=db)) as client:
        pre = client.post("/v1/forget/wording", json={"wording": "secret", "preview": True})
        assert pre.json()["inside_longer_words"] > 0
        go = client.post("/v1/forget/wording",
                         json={"wording": "secret", "allow_inside_words": True})
        assert go.status_code == 200 and go.json()["status"] == "forgotten"
    assert "secretary" not in all_text(db) and "ary" in all_text(db)


def test_a_string_that_is_a_whole_word_is_not_asked_about(db):
    key = new_key()
    cc(db, "s1", f"User: use {key} please.")
    assert forget_wording(db, key, dry_run=True)["inside_longer_words"] == 0
    assert forget_wording(db, key)["status"] == "forgotten"


# ── 9. two guards that had no test ───────────────────────────────────────

def test_the_raw_database_file_is_searched_and_reported(db, monkeypatch):
    """When the rewrite does not clear the file, the answer says so."""
    key = new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    monkeypatch.setattr(GraphStore, "scrub_storage",
                        lambda self: {"vacuumed": True, "wal_truncated": True})
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    left = [l for l in res["leftovers"] if l["kind"] == "storage"]
    assert left and "was still found in the" in left[0]["note"]


def test_a_wrapped_copy_pasted_after_the_forget_is_kept_out_by_the_capture_gate(db):
    from revien.forget import gate
    key = new_key()
    cc(db, "s1", "User: unrelated words.")
    forget_wording(db, key)
    half = len(key) // 2
    text = f"User: the key for the deploy pipeline is {key[:half]}\n  {key[half:]} and it is live."
    s = GraphStore(db_path=db)
    cut, n = gate.strip_literals(s, text)
    s.close()
    assert n == 1 and key[:half] not in cut and key[half:] not in cut
