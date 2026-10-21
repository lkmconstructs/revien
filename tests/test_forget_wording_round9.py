"""Round 9: the half of a wrapped key that a label kept goes with the key; a
piece in a node the key was never cut from is reported, not cut; the wrap
rule is pinned exactly as it behaves."""
from pathlib import Path

import pytest

from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.watch import snapshot_db
from tests.test_forget_wording import cc, forget_wording, leaks, new_key


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r9.db")


def _wrapped(db, key, session="s1"):
    half = len(key) // 2
    cc(db, session, f"User: the key for the deploy pipeline is {key[:half]}\n"
                    f"  {key[half:]} and it is live.\nUser: The venue is booked.")
    return half


def test_b_wrapped_ends_at_zero_raw_bytes_in_label_audit_and_snapshots(db, tmp_path):
    key = new_key()
    half = _wrapped(db, key)
    snaps = Path(db + ".snapshots")
    plain = snapshot_db(db, use_gzip=False, out_dir=snaps)
    gz = snapshot_db(db, use_gzip=True, out_dir=snaps)
    s = GraphStore(db_path=db)
    assert any(key[:half] in n.label for n in s.list_nodes(limit=100))     # the label kept it
    s.close()
    res = forget_wording(db, key)
    assert res["status"] == "forgotten", res
    assert leaks(db, key, key[:half], key[half:]) == {}
    s = GraphStore(db_path=db)
    try:
        for n in s.list_nodes(limit=100):
            assert key[:half] not in n.label + n.content
            for a in s.get_node_audit(n.node_id):
                assert key[:half] not in str(a)
    finally:
        s.close()
    import gzip
    for path in (plain, gz):
        raw = Path(path).read_bytes()
        if str(path).endswith(".gz"):
            raw = gzip.decompress(raw)
        assert key[:half].encode() not in raw and key[half:].encode() not in raw
    assert not [l for l in res["leftovers"] if l["kind"] == "partial_copy"]


def test_a_piece_in_a_node_the_key_was_never_cut_from_is_reported_not_cut(db):
    key = new_key()
    half = _wrapped(db, key)
    s = GraphStore(db_path=db)
    starts = s.add_node(Node(node_type=NodeType.FACT, label=f"{key[:8]} budget notes",
                             content="independent text one"))
    ends = s.add_node(Node(node_type=NodeType.FACT, label=f"budget notes {key[:12]}",
                           content="independent text two"))
    s.close()
    res = forget_wording(db, key)
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(starts.node_id).label == f"{key[:8]} budget notes"
        assert s.get_node(ends.node_id).label == f"budget notes {key[:12]}"
    finally:
        s.close()
    pieces = [l for l in res["leftovers"] if l["kind"] == "partial_copy"][0]["pieces"]
    got = {(p["node_id"], p["field"]): p["characters"] for p in pieces}
    assert got[(starts.node_id, "label")] == 8
    assert got[(ends.node_id, "label")] == 12
    assert res["scope"] != "gone everywhere"
    text = res["message"] + str(res["leftovers"])
    assert key[:8] not in text                          # where and how long, never the characters


def test_the_wrap_rule_is_exactly_a_break_after_a_line_of_twenty_or_more():
    lit = scrub.Literal("tokenabc")
    short = "token\n   abc"
    assert scrub.cut_text(short, [lit]) == (short, 0)                 # a short line: two texts
    long_line = "User: The value is token\n   abc"                    # 24 characters before the break
    assert len("User: The value is token") >= scrub.MIN_WRAPPED_LINE
    assert scrub.cut_text(long_line, [lit]) == ("User: The value is ", 1)
    ends = scrub.cut_text("the report is due by the end\nofmonth", [scrub.Literal("endofmonth")])
    assert ends == ("the report is due by the ", 1) and scrub.MIN_WRAPPED_LINE == 20
    short_ends = "by the end\nofmonth"                                # 10 characters: not a wrap
    assert scrub.cut_text(short_ends, [scrub.Literal("endofmonth")]) == (short_ends, 0)
