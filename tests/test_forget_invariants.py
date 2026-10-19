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


def _forget(seeded, planted=None, cascade=True):
    store = GraphStore(db_path=seeded.db)
    try:
        ops = GraphOperations(store)
        if planted:
            ops.training_dbs = (planted["training_db"],)
        handle = seeded.ctx_ids[0] if seeded.ctx_ids else seeded.node_with_secret
        return ops.forget_node(handle, cascade=cascade)
    finally:
        store.close()


@pytest.mark.parametrize("path", ALL)
def test_secret_is_gone_from_every_surface_after_a_forget(path, tmp_path, monkeypatch):
    """Invariant 1."""
    if path == "langchain":
        pytest.importorskip("langchain_core")
    seeded = fs.seed(path, tmp_path, monkeypatch)
    secret = seeded.secret
    handle_ids = list(seeded.ctx_ids) or [seeded.node_with_secret]
    assert handle_ids[0], f"{path}: the capture produced nothing to forget"

    # the secret really is in the store, and in the raw file, before
    before = surf.raw_bytes(seeded.db)
    assert secret.encode() in b"".join(before.values())

    planted = surf.plant(seeded.db, secret, tmp_path, handle_ids)
    result = _forget(seeded, planted)
    assert result["status"] == "forgotten", result

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
