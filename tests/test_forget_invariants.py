"""Forget, delete and cleanup: the invariants of docs/FORGET_SPEC.md §9.

Invariant 1 runs once per capture path: a unique secret is seeded through the
real path, a forget runs, and the secret must be absent from every surface of
§4 except the leftovers the response names.
"""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from revien.daemon.server import create_app
from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore
from tests import forget_seed as fs
from tests import forget_surfaces as surf

ALL = [p for p in fs.ALL_PATHS]


def _forget(seeded, planted=None, cascade=True, keep_open=False):
    store = GraphStore(db_path=seeded.db)
    try:
        ops = GraphOperations(store)
        if planted:
            ops.training_dbs = (planted["training_db"],)
        handle = seeded.ctx_ids[0] if seeded.ctx_ids else seeded.node_with_secret
        result = ops.forget_node(handle, cascade=cascade)
    except BaseException:
        store.close()
        raise
    if keep_open:
        return result, store
    store.close()
    return result


@pytest.mark.parametrize("path", ALL)
def test_secret_is_gone_from_every_surface_after_a_forget(path, tmp_path, monkeypatch):
    """Invariant 1."""
    if path == "langchain":
        from revien.adapters.langchain_adapter import LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("langchain-core is not installed")
    seeded = fs.seed(path, tmp_path, monkeypatch)
    secret = seeded.secret
    handle_ids = list(seeded.ctx_ids) or [seeded.node_with_secret]
    assert handle_ids[0], f"{path}: the capture produced nothing to forget"

    # the secret really is in the store, and in the raw file, before
    before = surf.raw_bytes(seeded.db)
    assert secret.encode() in b"".join(before.values())

    planted = surf.plant(seeded.db, secret, tmp_path, handle_ids)
    # the store stays open while the raw files are read: a file that is only
    # clean because closing the connection checkpointed it proves nothing
    result, open_store = _forget(seeded, planted, keep_open=True)
    assert result["status"] == "forgotten", result

    if path == "import_graph":
        # A graph file carries no record of who wrote a node, so every node it
        # brings is unproven (spec §3): kept and LISTED, untouched. What still
        # holds the secret must be exactly what the response lists.
        open_store.close()
        s2 = GraphStore(db_path=seeded.db)
        try:
            holding = {n.node_id for n in s2.list_nodes(limit=1000)
                       if secret in n.label + n.content}
        finally:
            s2.close()
        assert holding and holding <= set(result["unproven"]), (holding, result["unproven"])
        assert result["scope"] == "forgotten from Revien"
        return

    # every table, decoded (JSON escapes non-ASCII and line breaks)
    conn = sqlite3.connect(seeded.db)
    try:
        leftover = scrub.find_residue(conn, [secret])
    finally:
        conn.close()
    assert leftover == {}, f"{path}: secret still in {leftover}"

    # the raw file, the WAL and the freelist
    for name, blob in surf.raw_bytes(seeded.db).items():
        assert secret.encode() not in blob, f"{path}: secret in raw {name}"
    open_store.close()

    # vec_nodes and training.db
    conn = sqlite3.connect(seeded.db)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        for nid in planted["ids"]:
            assert conn.execute(
                "SELECT count(*) FROM vec_nodes WHERE node_id = ?",
                (nid,)).fetchone()[0] == 0
    finally:
        conn.close()
    assert secret.encode() not in open(planted["training_db"], "rb").read()
    tconn = sqlite3.connect(planted["training_db"])
    try:
        assert tconn.execute(
            "SELECT count(*) FROM retrieval_signals").fetchone()[0] == 0
    finally:
        tconn.close()

    # the response tells the truth about the database
    store = GraphStore(db_path=seeded.db)
    try:
        assert result["nodes_after"] == store.count_nodes()
        assert result["edges_after"] == store.count_edges()
        assert (result["nodes_before"] - result["removed_nodes"]
                + result["tombstones_created"]) == result["nodes_after"]
        assert result["count"] == result["removed_nodes"] == len(result["forgotten"])
        for nid in result["forgotten"]:
            assert store.get_node(nid) is None
    finally:
        store.close()

    # what is left is named, and only what may be left
    kinds = {l["kind"] for l in result["leftovers"]}
    assert kinds <= {"source_file", "exports", "snapshots", "vault_note",
                     "other_units"}, result["leftovers"]
    if seeded.file_backed:
        assert "source_file" in kinds
        assert str(seeded.source_file.resolve()).casefold() in [
            l["path"].casefold() for l in result["leftovers"]
            if l["kind"] == "source_file"]
    if seeded.stable:
        assert result["suppressed"], result
    elif path == "import_graph":
        # a graph file carries no record of who wrote a node: ownership is
        # unproven, so nothing can be suppressed, and the response says so
        assert not result["suppressed"]
        assert "suppresses nothing" in result["blast_radius"]
    else:
        assert not result["suppressed"]
        assert "eligible for recapture because the source supplied no stable item identity" in json.dumps(result)
    # a distilled note rendered the node (planted), so this is "forgotten
    # from Revien", never "gone everywhere", and it names the manual step
    notes = [l for l in result["leftovers"] if l["kind"] == "vault_note"]
    assert "Some Note" in [n["note"] for n in notes]
    assert notes[0]["fix"] == "revien distill-vault"
    assert result["scope"] == "forgotten from Revien"
    assert "revien distill-vault" in result["message"]


@pytest.mark.parametrize("path", ["pipeline", "claude_code", "chatgpt", "obsidian"])
def test_no_api_surface_shows_the_secret_after_a_forget(path, tmp_path, monkeypatch):
    """Invariant 1, the API surfaces: export, graph, nodes, audit, recall and
    TOON output, and `revien export`."""
    from click.testing import CliRunner
    from revien.cli import main
    seeded = fs.seed(path, tmp_path, monkeypatch)
    _forget(seeded)
    out = tmp_path / "out.json"
    res = CliRunner().invoke(main, ["export", str(out), "--db", seeded.db])
    assert res.exit_code == 0, res.output
    assert seeded.secret not in out.read_text()
    with TestClient(create_app(db_path=seeded.db)) as client:
        for url in ("/v1/export", "/v1/graph", "/v1/nodes", "/v1/audit/recent",
                    "/v1/communities", "/v1/tensions"):
            r = client.get(url)
            assert r.status_code == 200, (url, r.text)
            assert seeded.secret not in r.text, url
        for fmt in ("json", "toon"):
            r = client.post("/v1/recall", json={
                "query": "safe code codename", "format": fmt, "include_context": True})
            assert seeded.secret not in r.text, fmt


def test_community_labels_held_in_memory_do_not_keep_a_forgotten_label(tmp_path):
    """/v1/communities reads labels the community detector keeps in memory."""
    db = str(tmp_path / "comm.db")
    secret = fs.new_secret()
    with TestClient(create_app(db_path=db)) as client:
        ids = []
        for i in range(4):
            text = (f"User: We decided to use postgres for service{i}. "
                    f"Our team lead Alice Smith owns service{i}.")
            if i == 0:
                text += f' The vault codename is "{secret}".'
            r = client.post("/v1/ingest", json={"source_id": f"feed-{i}", "content": text})
            ids.append(r.json()["context_node_id"])
        client.post("/v1/cluster")
        assert secret in client.get("/v1/communities").text   # the test sees it first
        r = client.post(f"/v1/nodes/{ids[0]}/forget", json={"cascade": True})
        assert r.status_code == 200, r.text
        assert secret not in client.get("/v1/communities").text
        assert secret not in client.get("/v1/graph").text


def test_a_trained_model_on_disk_is_named_not_hidden(tmp_path, monkeypatch):
    models = Path.home() / ".revien" / "models"
    models.mkdir(parents=True)
    (models / "vectorizer.pkl").write_bytes(b"model")
    seeded = fs.seed("pipeline", tmp_path, monkeypatch)
    result = _forget(seeded)
    assert [l["kind"] for l in result["leftovers"]] == ["neural_model"]
    assert result["scope"] == "forgotten from Revien"
    assert result["model_stale"] is True
    assert "POST /v1/training/run" in result["leftovers"][0]["note"]
    assert "nothing retrains it" in result["leftovers"][0]["note"]


def test_gone_everywhere_is_said_only_when_it_is_true(tmp_path, monkeypatch):
    seeded = fs.seed("pipeline", tmp_path, monkeypatch)
    result = _forget(seeded)
    assert result["leftovers"] == [] and result["remaining"] == {}
    assert result["scope"] == "gone everywhere"
    assert result["message"].startswith("Gone everywhere.")
    seeded = fs.seed("claude_code", tmp_path / "b", monkeypatch) if (tmp_path / "b").mkdir() is None else None
    result = _forget(seeded)
    assert result["leftovers"] and result["scope"] == "forgotten from Revien"
    assert result["message"].startswith("Forgotten from Revien.")
