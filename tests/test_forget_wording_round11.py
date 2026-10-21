"""Round 11: the piece report covers every table, says when a snapshot could
not be read, leaves Revien's own actor strings out, and each of its field
scans has a test of its own."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import cc, forget_wording, new_key
from tests.test_forget_wording_round10 import _pieces


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r11.db")


def _seed(db, key):
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="alpha note", content="alpha"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="beta note", content="beta"))
    return s, a, b


def _got(res):
    return {(p["surface"], p["field"]): p["characters"] for p in _pieces(res)}


def test_the_report_reaches_edges_candidates_judgments_and_receipts(db):
    key = new_key()
    s, a, b = _seed(db, key)
    piece = key[:12]
    s.add_edge(Edge(source_node_id=a.node_id, target_node_id=b.node_id,
                    edge_type=EdgeType.RELATED_TO, metadata={"why": f"x {piece}"},
                    source_context=f"{piece} ctx"))
    s.add_candidate(a.node_id, b.node_id, "review", reason=f"{piece} reason", trace="t")
    now = datetime.now(timezone.utc)
    s.add_recall_receipt(recall_id="rc1", runtime="x", query=f"{piece} question",
                         node_ids=[a.node_id], issued_at=now, expires_at=now + timedelta(hours=1))
    s.record_use_judgment(episode_key="ep1", node_id=a.node_id, used=True, reason="r",
                          shared_terms=[], similarity=0.5, query=f"{piece} asked",
                          mode="m", applied=True)
    s.close()
    got = _got(forget_wording(db, key))
    for want in (("edges", "source_context"), ("edges", "metadata"),
                 ("supersession_candidates", "reason"), ("recall_receipts", "query"),
                 ("use_judgments", "query")):
        assert got.get(want) == 12, (want, got)


def test_an_unreadable_snapshot_is_a_leftover_and_not_gone_everywhere(db):
    key = new_key()
    s, a, b = _seed(db, key)
    s.close()
    snaps = Path(db + ".snapshots")
    snaps.mkdir()
    bad = snaps / "broken.db"
    bad.write_bytes(b"this is not a database" * 50)
    res = forget_wording(db, key)
    left = [l for l in res["leftovers"] if l["kind"] == "snapshot_unreadable"]
    assert left and str(bad) in left[0]["note"] and "could not be read" in left[0]["note"]
    assert res["scope"] != "gone everywhere"


@pytest.mark.parametrize("damage", ["truncated", "bad_deflate"])
def test_a_damaged_gzip_snapshot_is_a_leftover_not_a_crash(db, damage):
    import gzip
    key = new_key()
    s, a, b = _seed(db, key)
    s.close()
    snaps = Path(db + ".snapshots")
    snaps.mkdir()
    whole = gzip.compress(Path(db).read_bytes())
    bad = snaps / "half.db.gz"
    if damage == "truncated":  # a kill mid-write leaves the file cut short
        bad.write_bytes(whole[: len(whole) // 2])
    else:  # valid gzip header, garbage deflate stream
        bad.write_bytes(whole[:10] + b"\xff" * 200)
    res = forget_wording(db, key)
    left = [l for l in res["leftovers"] if l["kind"] == "snapshot_unreadable"]
    assert left and str(bad) in left[0]["note"]
    assert res["scope"] != "gone everywhere"


def test_revien_own_actor_is_not_reported_as_a_piece(db):
    secret = "api:forget-secret-9z8y7x"
    cc(db, "s1", f"User: We decided to use {secret} as the project codename.")
    res = forget_wording(db, secret, actor="api:forget")
    assert not [p for p in _pieces(res) if p["field"] == "actor"], _pieces(res)


def test_a_merge_actor_is_content_and_is_scanned(db):
    key = new_key()
    s, a, b = _seed(db, key)
    s.record_audit(a.node_id, "normalized_merge", actor=f"{key[:12]} budget")
    s.close()
    got = _got(forget_wording(db, key))
    assert got.get(("audit_log", "actor")) == 12


def test_event_time_and_metadata_are_scanned(db):
    key = new_key()
    s, a, b = _seed(db, key)
    other = s.add_node(Node(node_type=NodeType.FACT, label="plain", content="plain text",
                            event_time_text=f"around {key[:12]}",
                            metadata={"k": f"note {key[:10]}"}))
    s.close()
    res = forget_wording(db, key)
    got = {(p["node_id"], p["field"]): p["characters"] for p in _pieces(res)
           if p.get("node_id") == other.node_id}
    assert got[(other.node_id, "event_time_text")] == 12
    assert got[(other.node_id, "metadata")] == 10


def test_a_label_that_starts_like_the_end_of_the_key_elsewhere_is_not_cut(db):
    key = new_key()
    tail = key[-12:]
    s = GraphStore(db_path=db)
    node = s.add_node(Node(node_type=NodeType.FACT, label=f"{tail} budget",
                           content=f"alpha {key} x {tail} budget y"))
    s.close()
    res = forget_wording(db, key)
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(node.node_id).label == f"{tail} budget"        # not where the cut was
    finally:
        s.close()
    assert (node.node_id, "label") in {(p.get("node_id"), p["field"]) for p in _pieces(res)}
