"""Round 10: a piece of the string is reported from every surface (history,
snapshots), and a derived field is cut only where it stops exactly inside the
occurrence cut from the same node's content."""
from pathlib import Path

import pytest

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.watch import snapshot_db
from tests.test_forget_wording import cc, forget_wording, leaks, new_key


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r10.db")


def _pieces(res):
    left = [l for l in res["leftovers"] if l["kind"] == "partial_copy"]
    return left[0]["pieces"] if left else []


def test_a_renamed_node_leaves_its_history_and_snapshots_in_the_report(db):
    key = new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    other = s.add_node(Node(node_type=NodeType.FACT, label=f"{key[:16]} budget",
                            content="an unrelated budget note"))
    snaps = Path(db + ".snapshots")
    snapshot_db(db, use_gzip=False, out_dir=snaps)
    snapshot_db(db, use_gzip=True, out_dir=snaps)
    s.update_node(other.node_id, label="renamed budget note")
    s.close()
    res = forget_wording(db, key)
    assert res["status"] == "forgotten"
    pieces = _pieces(res)
    surfaces = {p["surface"] for p in pieces}
    assert "audit_log" in surfaces
    assert any(x.startswith("snapshot ") and x.endswith(".db node") for x in surfaces)
    assert any(x.startswith("snapshot ") and x.endswith(".db.gz node") for x in surfaces)
    assert {p["characters"] for p in pieces} == {16}
    assert sum(1 for p in pieces if p["surface"] == "audit_log"
               and p["node_id"] == other.node_id) >= 2        # its two audit rows
    assert res["scope"] != "gone everywhere"
    assert key[:16] not in res["message"] + str(res["leftovers"])


@pytest.mark.parametrize("secret,label", [
    ("password123456", "please reset the password"),
    ("hunter22xy", "reset the hunter22"),
    ("staging-credentials-77x1", "rotate the staging-credentials"),
])
def test_a_label_that_merely_ends_like_the_key_is_not_cut(db, secret, label):
    cc(db, "s1", f"User: We decided to use {secret} as the project codename.")
    s = GraphStore(db_path=db)
    own = s.add_node(Node(node_type=NodeType.FACT, label=label, content=label + " for everyone"))
    same = s.add_node(Node(node_type=NodeType.FACT, label=label,
                           content=f"{label} today; the old one was {secret}"))
    s.close()
    res = forget_wording(db, secret)
    assert res["status"] == "forgotten"
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(own.node_id).label == label                 # a node with no occurrence
        assert s.get_node(same.node_id).label == label                # the piece is not where the cut was
    finally:
        s.close()
    assert leaks(db, secret) == {}
    assert res["scope"] != "gone everywhere"                           # and the pieces are reported
    assert {(p["node_id"], p["field"]) for p in _pieces(res)} >= {(own.node_id, "label")}


def test_every_derived_field_stops_inside_the_cut_and_goes_with_it(db):
    key = new_key()
    s = GraphStore(db_path=db)
    line = f"the deploy token is {key[:12]}"
    s.add_node(Node(node_type=NodeType.FACT, label="deploy note",
                    content=f"the deploy token is {key} ok",
                    source_context=line, event_time_text=line))
    s.close()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    res = forget_wording(db, key)
    assert res["status"] == "forgotten"
    s = GraphStore(db_path=db)
    try:
        for n in s.list_nodes(limit=50):
            assert key[:12] not in n.source_context and key[:12] not in n.event_time_text
    finally:
        s.close()
    assert leaks(db, key[:12]) == {}


def test_a_piece_below_the_floor_is_neither_cut_nor_reported(db):
    secret = "Zq7-pass-wd9"
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="pass note",
                    content=f"the pass is {secret} ok", source_context="the pass is Zq7-pa"))
    s.close()
    res = forget_wording(db, secret)
    s = GraphStore(db_path=db)
    try:
        assert [n.source_context for n in s.list_nodes(limit=10)] == ["the pass is Zq7-pa"]
    finally:
        s.close()
    assert not _pieces(res)


def test_a_piece_at_the_end_of_another_nodes_content_is_reported(db):
    key = new_key()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    other = s.add_node(Node(node_type=NodeType.FACT, label="notes",
                            content=f"notes about {key[:12]}"))
    s.close()
    res = forget_wording(db, key)
    got = {(p["node_id"], p["field"]): p["characters"] for p in _pieces(res)}
    assert got[(other.node_id, "content")] == 12
