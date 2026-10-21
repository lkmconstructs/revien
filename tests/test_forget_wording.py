"""`revien forget --wording`: forget a literal secret (docs/FORGET_SPEC.md §2a).

Each acceptance row of the PR #6 fix brief is a test on a throwaway database.
Every row ends at zero rows and zero raw bytes: the secret is looked for in
EVERY table (every column, JSON decoded), in the database file, its -wal and
-shm, in the vectors, in training.db and in the snapshots, not through any
helper that knows where it ought to be.
"""
import json
import socket
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.forget import gate, keys, scrub
from revien.forget.engine import ForgetEngine
from revien.forget.wording import MIN_WORDING
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests import forget_seed as fs


def new_key() -> str:
    return "ghp_" + uuid.uuid4().hex[:18] + uuid.uuid4().hex[:18]


# ── scanning everything ──────────────────────────────────────────────────

def _cells(conn):
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND sql NOT LIKE 'CREATE VIRTUAL%'")]
    for t in tables:
        if t.startswith("sqlite_") or t.startswith("vec_nodes_"):
            continue
        try:
            for row in conn.execute(f'SELECT * FROM "{t}"'):
                for v in row:
                    yield t, v
        except sqlite3.Error:
            continue


def _flatten(v):
    if isinstance(v, (bytes, bytearray)):
        yield bytes(v).decode("utf-8", "ignore")
    elif isinstance(v, str):
        yield v
        try:
            inner = json.loads(v)
        except ValueError:
            return
        if isinstance(inner, (dict, list)):
            yield from _walk(inner)


def _walk(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for x in o.values():
            yield from _walk(x)
    elif isinstance(o, list):
        for x in o:
            yield from _walk(x)


def leaks(db: str, *pieces: str) -> dict:
    """{where: n} for every place any piece (or the pieces joined) is found:
    every table, and the raw db / -wal / -shm bytes."""
    found = {}
    conn = sqlite3.connect(db)
    try:
        joined = "".join(pieces)
        for table, v in _cells(conn):
            for text in _flatten(v):
                flat = scrub.canonical(text)
                if any(p in text or p in flat for p in pieces) or joined in flat:
                    found[table] = found.get(table, 0) + 1
                    break
    finally:
        conn.close()
    for suffix in ("", "-wal", "-shm"):
        p = Path(db + suffix)
        if p.exists():
            blob = p.read_bytes()
            if any(x.encode() in blob for x in pieces):
                found["raw" + (suffix or "-db")] = 1
    return found


def cc(db, session, text, runtime="claude-code"):
    s = GraphStore(db_path=db)
    try:
        return IngestionPipeline(s).ingest(IngestionInput(
            source_id=f"{runtime}:proj:{session}", ingest_key=f"{runtime}:proj:{session}",
            origin_runtime=runtime, session_key=session, content=text))
    finally:
        s.close()


def forget_wording(db, secret, **kw):
    s = GraphStore(db_path=db)
    try:
        return GraphOperations(s).forget_wording(secret, **kw)
    finally:
        s.close()


def node_with(db, secret, ntype=None):
    s = GraphStore(db_path=db)
    try:
        for n in s.list_nodes(limit=500):
            if secret in n.label + n.content and (ntype is None or n.node_type == ntype):
                return n
    finally:
        s.close()


def all_text(db):
    s = GraphStore(db_path=db)
    try:
        return " ".join(n.label + " " + n.content for n in s.list_nodes(limit=1000))
    finally:
        s.close()


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "w.db")


# ── q4: pasted again in a new sentence ───────────────────────────────────

@pytest.mark.parametrize("before", ["nothing", "plain", "everywhere"])
@pytest.mark.parametrize("where", ["same_session", "other_session"])
def test_q4_a_secret_pasted_again_stays_out(db, before, where):
    key = new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.\nUser: The venue is booked.")
    if before != "nothing":
        s = GraphStore(db_path=db)
        fact = [n for n in s.list_nodes(limit=100) if key in n.content
                and n.node_type != NodeType.CONTEXT][0]
        GraphOperations(s).forget_node(fact.node_id, everywhere=(before == "everywhere"))
        s.close()
    res = forget_wording(db, key)
    assert res["status"] == "forgotten", res
    assert leaks(db, key) == {}
    again = f"User: Remember that the CI uses {key} for deploys, please."
    cc(db, "s1" if where == "same_session" else "s2",
       f"User: The venue is booked.\n{again}" if where == "same_session" else again)
    assert leaks(db, key) == {}, "the same bytes came back"


# ── q7/q8: only in a context node, no fact extracted ─────────────────────

def test_q7_q8_a_secret_only_in_a_context_node(db):
    key = new_key()
    cc(db, "s1", "\n".join(f"{key}" for _ in range(4)) + "\nplain closing line here")
    assert node_with(db, key, NodeType.CONTEXT) is not None
    assert all(n.node_type == NodeType.CONTEXT for n in [node_with(db, key)])
    res = forget_wording(db, key)
    assert res["status"] == "forgotten" and res["occurrences_cut"] >= 4
    assert leaks(db, key) == {}
    assert "plain closing line here" in all_text(db)       # only the string went


# ── a2 / a3 / a9: history the node-level verb never reaches ──────────────

def test_a2_a_node_deleted_before_the_forget(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    s = GraphStore(db_path=db)
    ops = GraphOperations(s)
    holders = [n.node_id for n in s.list_nodes(limit=100) if key in n.label + n.content]
    assert holders
    for nid in holders:
        assert ops.delete_node(nid)
    s.close()
    assert leaks(db, key), "the audit trail keeps what the deleted nodes held"
    res = forget_wording(db, key)
    assert res["status"] == "forgotten"
    assert leaks(db, key) == {}


def test_a3_a_node_edited_away_before_the_forget(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    s = GraphStore(db_path=db)
    for n in s.list_nodes(limit=100):
        if key in n.label + n.content:
            s.update_node(n.node_id, content="[edited]", label="[edited]")
    s.close()
    assert leaks(db, key), "the create and update audit rows keep it"
    forget_wording(db, key)
    assert leaks(db, key) == {}


def test_a9_a_key_line_removed_from_the_session_and_resynced(tmp_path):
    from tests.test_forget_narrow import sync, write_session
    key = new_key()
    root = tmp_path / "claude"
    db = str(tmp_path / "a9.db")
    write_session(root, "p", "s1", "uuid-A9", [
        ("user", "The venue is booked for the whole of June."),
        ("user", f"My deploy key is {key}.")])
    sync(db, root)
    write_session(root, "p", "s1", "uuid-A9", [
        ("user", "The venue is booked for the whole of June."),
        ("user", "A later remark about the parking permits.")])
    sync(db, root)
    assert leaks(db, key), "the refreshed-away line is still in the audit rows"
    res = forget_wording(db, key)
    assert res["status"] == "forgotten", res
    assert leaks(db, key) == {}


# ── B: the three shapes a sentence cut leaves behind ─────────────────────

@pytest.mark.parametrize("shape", ["bearer_own_line", "url_token", "wrapped"])
def test_b_key_shapes(db, shape):
    key = new_key()
    half = len(key) // 2
    text = {
        "bearer_own_line": f"User: The header was\nAuthorization:\nBearer {key}\nand that failed.",
        "url_token": f"User: call https://api.example.com/v1/data?token={key}&x=1 and retry.",
        "wrapped": f"User: the key is {key[:half]}\n  {key[half:]} and it is live.",
    }[shape]
    cc(db, "s1", text + "\nUser: The venue is booked.")
    res = forget_wording(db, key)
    assert res["status"] == "forgotten", res
    pieces = (key, key[:half], key[half:]) if shape == "wrapped" else (key,)
    assert leaks(db, *pieces) == {}
    body = all_text(db)
    assert "The venue is booked." in body
    if shape == "url_token":
        assert "https://api.example.com/v1/data?token=&x=1" in body   # the string, not the sentence


# ── C2: two sources, --everywhere first ──────────────────────────────────

def test_c2_a_copy_in_an_owned_conversation_is_out_or_listed(db):
    key = new_key()
    cc(db, "sA", f"User: We decided to use {key} as the deploy key.\nUser: The venue is booked.")
    cc(db, "sB", f"User: The CI job also reads {key} from the vault.\nUser: Lunch is at noon.")
    s = GraphStore(db_path=db)
    fact = [n for n in s.list_nodes(limit=100) if key in n.content
            and n.node_type != NodeType.CONTEXT
            and "deploy key" in n.content][0]
    res = GraphOperations(s).forget_node(fact.node_id, everywhere=True)
    s.close()
    held = leaks(db, key)
    if held:
        assert res["scope"] != "gone everywhere", res["message"]
        assert res["still_held_by"], "still_held_by: [] while a copy still exists"
    else:
        assert res["scope"] in ("gone everywhere", "forgotten from Revien")


# ── q5: a hung or broken daemon at forget time ───────────────────────────

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Listener:
    """hang: accepts and never answers. http500: answers 500."""
    def __init__(self, mode):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(5)
        self.url = f"http://127.0.0.1:{self.srv.getsockname()[1]}"
        self.mode, self.conns = mode, []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            self.conns.append(c)
            if self.mode == "http500":
                try:
                    c.recv(4096)
                    c.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
                except OSError:
                    pass

    def close(self):
        self.srv.close()
        for c in self.conns:
            c.close()


@pytest.mark.real_probe
@pytest.mark.parametrize("mode", ["hang", "http500"])
@pytest.mark.parametrize("verb", ["wording", "node"])
def test_q5_a_hung_or_broken_daemon_refuses_the_forget(db, monkeypatch, mode, verb):
    key = new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.\nUser: The venue is booked.")
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
    srv = _Listener(mode)
    try:
        if verb == "wording":
            args = ["forget", "--wording", key, "--yes", "--db", db, "--daemon-url", srv.url]
        else:
            fact = node_with(db, key, NodeType.FACT)
            args = ["forget", fact.node_id, "--yes", "--db", db, "--daemon-url", srv.url]
        out = CliRunner().invoke(main, args)
    finally:
        srv.close()
    assert out.exit_code == 1, out.output
    assert "Nothing was changed." in out.output and "did not give a current Revien health answer" in out.output
    assert key not in out.output
    assert leaks(db, key), "a refused forget changed nothing"


@pytest.mark.real_probe
def test_q5_nothing_listening_still_proceeds(db, monkeypatch):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
    out = CliRunner().invoke(main, ["forget", "--wording", key, "--yes", "--db", db,
                                    "--daemon-url", f"http://127.0.0.1:{_free_port()}"])
    assert out.exit_code == 0, out.output
    assert leaks(db, key) == {}


# ── the string is never recorded anywhere new ────────────────────────────

def test_the_forget_records_the_string_nowhere(db, monkeypatch, caplog):
    import logging
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
    caplog.set_level(logging.DEBUG)
    runner = CliRunner()
    pre = runner.invoke(main, ["forget", "--wording", "-", "--preview", "--db", db,
                               "--daemon-url", f"http://127.0.0.1:{_free_port()}"], input=key + "\n")
    out = runner.invoke(main, ["forget", "--wording", "-", "--yes", "--db", db,
                               "--daemon-url", f"http://127.0.0.1:{_free_port()}"], input=key + "\n")
    js = runner.invoke(main, ["forget", "--wording", "-", "--yes", "--json-output", "--db", db,
                              "--daemon-url", f"http://127.0.0.1:{_free_port()}"], input=key + "\n")
    for result in (pre, out, js):
        assert result.exit_code == 0, result.output
        assert key not in result.output
    assert key not in caplog.text
    assert leaks(db, key) == {}                    # no suppression row, audit row, or log
    s = GraphStore(db_path=db)
    try:
        rec = [r for r in s.list_suppressions("active") if r["literal_fps"]]
    finally:
        s.close()
    assert rec and rec[0]["scope"] == "global"
    entry = rec[0]["literal_fps"][0]
    assert entry.split(":")[0] == str(len(key))
    assert key not in json.dumps(rec)


def test_a_string_shorter_than_the_floor_is_refused_and_not_echoed(db, monkeypatch):
    cc(db, "s1", "User: the cat sat on the mat today.")
    short = "cat"
    assert len(short) < MIN_WORDING
    res = forget_wording(db, short)
    assert res["status"] == "refused" and res["reason"] == "wording_too_short"
    assert short not in res["message"]
    assert "the cat sat on the mat today." in all_text(db)       # nothing cut
    assert forget_wording(db, "   \n  ")["status"] == "refused"
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
    out = CliRunner().invoke(main, ["forget", "--wording", short, "--yes", "--db", db,
                                    "--daemon-url", f"http://127.0.0.1:{_free_port()}"])
    assert out.exit_code == 1 and "at least" in out.output and short not in out.output


def test_the_preview_counts_without_changing_anything(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    before = leaks(db, key)
    res = forget_wording(db, key, dry_run=True)
    assert res["status"] == "preview" and res["would_cut_occurrences"] >= 2
    assert "nodes" in res["would_cut_in"] and key not in json.dumps(res)
    assert leaks(db, key) == before


# ── the boundary choice: raw characters, not whole words ─────────────────

def test_raw_characters_are_cut_inside_a_token_and_case_is_exact(db):
    key = new_key()
    cc(db, "s1", f"User: see prefix{key}suffix and also {key.upper()} in a header.")
    forget_wording(db, key)
    body = all_text(db)
    assert key not in body
    assert "prefixsuffix" in body                     # no word boundary needed
    assert key.upper() in body                        # a different string: not guessed
    # (a named-node forget keeps whole-word matching: see test_forget_review.py)


# ── the verb reaches the other places a secret hides ─────────────────────

def test_vectors_receipts_queries_training_and_snapshots_are_clean(tmp_path):
    from tests import forget_surfaces as surf
    from revien.watch import snapshot_db
    key = new_key()
    db = str(tmp_path / "all.db")
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.\nUser: The venue is booked.")
    fact = node_with(db, key, NodeType.FACT)
    planted = surf.plant(db, key, tmp_path, [fact.node_id, node_with(db, key, NodeType.CONTEXT).node_id])
    snaps = Path(db + ".snapshots")
    plain = snapshot_db(db, use_gzip=False, out_dir=snaps)
    packed = snapshot_db(db, use_gzip=True, out_dir=snaps)
    s = GraphStore(db_path=db)
    ops = GraphOperations(s)
    ops.training_dbs = (planted["training_db"],)
    res = ops.forget_wording(key)
    s.close()
    assert res["status"] == "forgotten"
    assert leaks(db, key) == {}
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    for nid in planted["ids"]:
        assert conn.execute("SELECT count(*) FROM vec_nodes WHERE node_id = ?", (nid,)).fetchone()[0] == 0
    conn.close()
    assert key.encode() not in open(planted["training_db"], "rb").read()
    import gzip
    assert key.encode() not in Path(plain).read_bytes()
    assert key.encode() not in gzip.decompress(Path(packed).read_bytes())
    assert res["snapshots"]["scrubbed"] == 2


def test_cut_nodes_are_re_embedded_from_the_cut_text(tmp_path):
    from tests.forget_surfaces import TinyEmbedder
    from revien.semantic.index import SemanticIndex
    key = new_key()
    db = str(tmp_path / "emb.db")
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    ctx = [n for n in s.list_nodes(limit=50) if n.node_type == NodeType.CONTEXT][0]
    SemanticIndex(s, embedder=TinyEmbedder(), enabled=True).index_node(ctx.node_id, ctx.label, ctx.content)
    res = GraphOperations(s).forget_wording(key)
    pending = s._get_conn().execute(
        "SELECT count(*) FROM pending_embeds WHERE node_id = ?", (ctx.node_id,)).fetchone()[0]
    vec = s._get_conn().execute(
        "SELECT count(*) FROM vec_nodes WHERE node_id = ?", (ctx.node_id,)).fetchone()[0]
    s.close()
    assert res["status"] == "forgotten"
    assert vec == 0 and pending == 1          # the old vector is gone; a new one is queued from the cut text
    assert key not in all_text(db)


def test_the_audit_scrub_reaches_rows_of_nodes_that_no_longer_exist(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    s = GraphStore(db_path=db)
    gone = []
    for n in s.list_nodes(limit=50):
        if key in n.label + n.content:
            gone.append(n.node_id)
            s.delete_node(n.node_id)
    orphans = s._get_conn().execute(
        "SELECT count(*) FROM audit_log WHERE node_id IN (%s) AND (before_json LIKE ? OR after_json LIKE ?)"
        % ",".join("?" * len(gone)), (*gone, f"%{key}%", f"%{key}%")).fetchone()[0]
    s.close()
    assert orphans >= 1
    forget_wording(db, key)
    s = GraphStore(db_path=db)
    after = s._get_conn().execute(
        "SELECT count(*) FROM audit_log WHERE before_json LIKE ? OR after_json LIKE ?",
        (f"%{key}%", f"%{key}%")).fetchone()[0]
    s.close()
    assert after == 0


# ── every capture path keeps the string out ──────────────────────────────

def test_an_edit_cannot_put_the_string_back(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    forget_wording(db, key)
    with TestClient(create_app(db_path=db)) as client:
        n = client.get("/v1/nodes").json()
        nid = (n["nodes"] if isinstance(n, dict) else n)[0]["node_id"]
        r = client.put(f"/v1/nodes/{nid}", json={"content": f"pasted {key} again", "label": key + " label"})
        assert r.status_code == 200
        assert key not in r.text
    assert leaks(db, key) == {}


def test_a_new_node_the_importer_the_adapter_and_the_graph_import_cannot_carry_it(tmp_path):
    key = new_key()
    db = str(tmp_path / "paths.db")
    cc(db, "s1", f"User: My deploy key is {key}.")
    s = GraphStore(db_path=db)
    graph = s.export_graph()
    s.close()
    forget_wording(db, key)
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="direct", content=f"added directly: {key} ok"))
    result = s.import_graph(graph, mode="merge")
    s.close()
    from revien.adapters.openai_adapter import OpenAIAdapter
    export = tmp_path / "oa.json"
    fs._chatgpt_export(export, key)
    export.write_text(json.dumps(json.loads(export.read_text())[0]))
    with OpenAIAdapter(graph_path=db) as a:
        a.ingest_conversation(str(export))
    from revien.importers import chatgpt
    from revien.importers.base import run_import
    exp = tmp_path / "ch.json"
    fs._chatgpt_export(exp, key)
    s = GraphStore(db_path=db)
    run_import(chatgpt.iter_units(str(exp)), IngestionPipeline(s), override_suppression=True)
    s.close()
    assert leaks(db, key) == {}, "a capture path let it back"


# ── forget.key: lost or replaced, said out loud ──────────────────────────

def test_a_replaced_forget_key_is_disclosed_at_the_next_forget(db, tmp_path):
    key, other = new_key(), new_key()
    cc(db, "s1", f"User: My deploy key is {key}. Also {other} is another one.")
    first = forget_wording(db, key)
    assert not first.get("warnings")
    kp = keys.key_path()
    kp.write_text("ab" * 32)                  # forget.key lost and replaced
    keys._cache.clear()
    second = forget_wording(db, other)
    assert second["warnings"] and "forget.key" in second["warnings"][0]
    assert "forget those again" in second["message"]
    assert keys.key_check() == GraphStore(db_path=db).get_meta("forget_key_check")


# ── restore lifts a secret's record, and nothing else does ───────────────

def test_restore_lifts_the_record_and_the_string_may_land_again(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    forget_wording(db, key)
    s = GraphStore(db_path=db)
    rec = [r for r in s.list_suppressions("active") if r["literal_fps"]][0]
    s.close()
    out = CliRunner().invoke(main, ["restore", "--list", "--db", db])
    assert "secret string" in out.output and key not in out.output
    r = CliRunner().invoke(main, ["restore", rec["unit_id"], "--db", db])
    assert r.exit_code == 0, r.output
    cc(db, "s2", f"User: the key {key} again.")
    assert leaks(db, key)


# ── REST ─────────────────────────────────────────────────────────────────

def test_rest_forgets_a_string_and_previews(db):
    key = new_key()
    cc(db, "s1", f"User: My deploy key is {key}.")
    with TestClient(create_app(db_path=db)) as client:
        pre = client.post("/v1/forget/wording", json={"wording": key, "preview": True})
        assert pre.status_code == 200 and pre.json()["status"] == "preview"
        assert key not in pre.text
        short = client.post("/v1/forget/wording", json={"wording": "abc"})
        assert short.status_code == 400 and "abc" not in short.text
        r = client.post("/v1/forget/wording", json={"wording": key})
        assert r.status_code == 200 and r.json()["status"] == "forgotten"
        assert key not in r.text
    assert leaks(db, key) == {}


def test_the_delete_command_exists_and_writes_no_forget_record(db, monkeypatch):
    cc(db, "s1", "User: Plain words about the venue today.")
    s = GraphStore(db_path=db)
    ctx = [n for n in s.list_nodes(limit=50) if n.node_type == NodeType.CONTEXT][0]
    s.close()
    monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
    out = CliRunner().invoke(main, ["delete", ctx.node_id, "--yes", "--db", db])
    assert out.exit_code == 0, out.output
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(ctx.node_id) is None
        assert s.list_suppressions(None) == []
        assert [a["op"] for a in s.get_node_audit(ctx.node_id) if a["op"] == "delete"] == ["delete"]
    finally:
        s.close()


def test_a_skill_file_and_a_retyped_note_claim_cannot_carry_it(tmp_path):
    from revien.distill import VaultReconciler
    from revien.semantic.index import SemanticIndex
    from revien.skills.ingest import ingest_roots
    key = new_key()
    db = str(tmp_path / "sk.db")
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    forget_wording(db, key)
    root = tmp_path / "skills" / "deploy"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: Deploy the thing\ntriggers: [deploy]\n---\n"
        f"Export TOKEN={key} before you run the release.\n", encoding="utf-8")
    s = GraphStore(db_path=db)
    ingest_roots(s, paths=[str(tmp_path / "skills")])
    ent = s.add_node(Node(node_type=NodeType.ENTITY, label="Atlas", content="Atlas"))
    rec = VaultReconciler(s, str(tmp_path), semantic=SemanticIndex(s, enabled=False))
    assert rec._new_claim_node(ent.node_id, f"Atlas uses {key} for deploys.", None) is None
    s.close()
    assert leaks(db, key) == {}
    assert "release" in all_text(db)           # the rest of the skill was captured


def test_a_capture_in_flight_when_the_forget_lands_cannot_write_the_string(tmp_path):
    """The gate is asked again inside the write transaction: a forget that lands
    while the extractor runs is not written over (new record and refresh)."""
    from revien.ingestion.extractor import RuleBasedExtractor
    key = new_key()
    db = str(tmp_path / "race.db")
    store = GraphStore(db_path=db)
    try:
        class Forgetting(RuleBasedExtractor):
            fired = False

            def extract(self, content, source_id=""):
                out = super().extract(content, source_id)
                if not Forgetting.fired and key in content:
                    Forgetting.fired = True
                    other = GraphStore(db_path=db)
                    try:
                        GraphOperations(other).forget_wording(key)
                    finally:
                        other.close()
                return out

        mk = lambda text: IngestionInput(
            source_id="claude-code:proj:r", ingest_key="claude-code:proj:r",
            origin_runtime="claude-code", session_key="r", content=text)
        pipe = IngestionPipeline(store)
        pipe.ingest(mk("User: The venue is booked."))
        IngestionPipeline(store, extractor=Forgetting()).ingest(
            mk(f"User: The venue is booked.\nUser: We decided to use {key} as the deploy key."))
    finally:
        store.close()
    assert Forgetting.fired
    assert leaks(db, key) == {}
