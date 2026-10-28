"""Second independent verification card for forget --wording, rebuilt as tests.

Every case ends at zero across EVERY table, the raw db / -wal / -shm and both
kinds of `revien watch` snapshot (.db and .db.gz, tables and raw bytes), or at
a leftover the card allows (named in the test). Keys are generated, never
printed; no assertion message contains one.
"""
import gzip
import json
import random
import sqlite3
import tempfile
from pathlib import Path

import pytest

from revien.adapters.openai_adapter import OpenAIAdapter
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.watch import snapshot_db
from tests.test_forget_wording import _cells, _flatten, cc, forget_wording, leaks


def fake_key(n=40, prefix="ghp_", seed=None):
    rnd = random.Random(seed)
    return prefix + "".join(rnd.choice("abcdef0123456789") for _ in range(n - len(prefix)))


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "verify2.db")


def snapshots(db):
    snaps = Path(db + ".snapshots")
    return (snapshot_db(db, use_gzip=False, out_dir=snaps),
            snapshot_db(db, use_gzip=True, out_dir=snaps))


def everywhere(db, *pieces):
    """leaks() plus every snapshot: tables (every column) and raw bytes."""
    found = dict(leaks(db, *pieces))
    snaps = Path(db + ".snapshots")
    for f in sorted(snaps.iterdir()) if snaps.is_dir() else []:
        raw = f.read_bytes()
        if f.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        if any(p.encode() in raw for p in pieces):
            found[f"{f.name}:raw"] = 1
        if not (f.name.endswith(".db") or f.name.endswith(".db.gz")):
            continue                                  # a -wal / -shm sidecar: raw only
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            path.write_bytes(raw)
            conn = sqlite3.connect(str(path))
            try:
                for table, v in _cells(conn):
                    if any(p in t for t in _flatten(v) for p in pieces):
                        found[f"{f.name}:{table}"] = 1
            finally:
                conn.close()
    return found


def pieces_reported(res):
    left = [l for l in res["leftovers"] if l["kind"] == "partial_copy"]
    return left[0]["pieces"] if left else []


def conversation(path, conv_id, text, role="user", title="Chat"):
    conv = {"id": conv_id, "title": title, "create_time": 1.7e9, "update_time": 1.7e9,
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["m1"]},
                "m1": {"id": "m1", "parent": "root", "children": [], "message": {
                    "id": "m1-" + conv_id, "author": {"role": role}, "create_time": 1.7e9,
                    "content": {"content_type": "text", "parts": [text]}}}}}
    Path(path).write_text(json.dumps(conv))
    return str(path)


def adapter_ingest(db, path):
    with OpenAIAdapter(graph_path=db) as a:
        a.ingest_conversation(path)


def gone(res):
    assert res["status"] == "forgotten", res
    assert res["scope"] == "gone everywhere", (res["scope"], res["leftovers"], res["remaining"])


# ── cases 1-8 ─────────────────────────────────────────────────────────────

def test_case1_a_wrapped_key_with_snapshots_is_gone(db):
    key = fake_key(seed=1)
    cc(db, "s1", f"User: the deploy pipeline key is {key[:22]}\n  {key[22:]} and it is live.")
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key, key[:22], key[22:]) == {}


def test_case2_a_label_truncated_by_the_adapter_is_gone(db, tmp_path):
    key = fake_key(200, "tok_", seed=2)
    adapter_ingest(db, conversation(tmp_path / "c.json", "c2",
                                    f"GET https://api.example.com/v1/data?token={key}"))
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key[:60], key[-60:]) == {}


def test_case3_a_piece_in_a_deleted_nodes_audit_row_is_cut_when_it_aligns_and_reported_when_not(db):
    key = fake_key(seed=3)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    aligned = s.add_node(Node(node_type=NodeType.FACT, label=f"note {key[:20]}",
                              content=f"note {key} here"))
    # (the store guard cuts these at write time once the key is named; here the
    # key is not yet named, so both nodes are written as they would have been)
    loose = s.add_node(Node(node_type=NodeType.FACT, label=f"{key[:20]} budget",
                            content="an unrelated budget note"))
    s.delete_node(aligned.node_id)
    s.delete_node(loose.node_id)
    s.close()
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    got = pieces_reported(res)
    # the aligned node's audit rows are cut with it; the loose one is reported
    assert not any(p.get("node_id") == aligned.node_id for p in got), got
    assert any(p.get("node_id") == loose.node_id and p["surface"] == "audit_log"
               for p in got), got
    assert key[:20] not in json.dumps(res["leftovers"])


def test_case4_content_edited_away_leaves_a_reported_piece(db):
    key = fake_key(seed=4)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label=f"{key[:20]} budget",
                        content=f"budget note {key}"))
    s.update_node(n.node_id, content="edited away")
    s.close()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    assert any(p.get("node_id") == n.node_id and p["field"] == "label"
               for p in pieces_reported(res))


def test_case5_a_short_line_wrap_is_reported_under_the_documented_limit(db):
    key = fake_key(seed=5)
    cc(db, "s1", f"User: key\n{key[:15]}\n{key[15:]} done and the line is short.")
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"          # two short lines are two texts (CHANGELOG)


def test_case6_a_piece_below_the_floor_is_by_design_left_alone(db):
    key = fake_key(seed=6)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label=f"{key[-7:]} budget", content="budget"))
    s.close()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    forget_wording(db, key)
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(n.node_id).label == f"{key[-7:]} budget"
    finally:
        s.close()


def test_case7_a_metadata_preview_that_aligns_is_cut(db):
    key = fake_key(seed=7)
    content = f"first words {key} and more words after it"
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="note", content=content,
                    metadata={"preview": content[:32]}))
    s.close()
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key, key[:20]) == {}


def test_case8_a_pre_derived_label_cannot_bring_a_piece_back(db):
    key = fake_key(seed=8)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    forget_wording(db, key)
    content = f"we use {key} for deploys"
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label=content[:19], content=content))
    s.close()
    assert everywhere(db, key, key[:11]) == {}


# ── extras ────────────────────────────────────────────────────────────────

def _bearer(seed=11):
    return fake_key(320, "eyJFAKE.", seed=seed)


def test_E1_an_adapter_label_with_an_ellipsis_is_cut_with_its_audit_and_snapshots(db, tmp_path):
    key = _bearer()
    adapter_ingest(db, conversation(tmp_path / "c1.json", "c1", f"Authorization: Bearer {key}"))
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key, key[:60], key[-60:]) == {}


def test_E1b_the_same_message_imported_again_lands_nothing(db, tmp_path):
    key = _bearer(12)
    adapter_ingest(db, conversation(tmp_path / "c1.json", "c1", f"Authorization: Bearer {key}"))
    gone(forget_wording(db, key))
    adapter_ingest(db, conversation(tmp_path / "c2.json", "c2", f"Authorization: Bearer {key}",
                                    title=f"Bearer {key}"))
    assert everywhere(db, key, key[:60], key[:40]) == {}


def _edge_with(db, by, ctx=""):
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="alpha note", content="alpha"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="beta note", content="beta"))
    e = Edge(source_node_id=a.node_id, target_node_id=b.node_id,
             edge_type=EdgeType.RELATED_TO, confidence_set_by=by, source_context=ctx)
    s.add_edge(e)
    return s, a, b, e


def test_E2_E2c_a_piece_in_confidence_set_by_is_reported(db):
    key = fake_key(seed=21)
    s, a, b, e = _edge_with(db, f"agent:{key[:20]}")
    s.close()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    assert any(p["field"] == "confidence_set_by" for p in pieces_reported(res))


def test_E2b_a_piece_in_a_non_merge_audit_actor_is_reported(db):
    key = fake_key(seed=22)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="n", content="n note"))
    s.record_audit(n.node_id, "skill_accept", actor=key[:20])
    s.close()
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    assert any(p["field"] == "actor" and p["surface"] == "audit_log" for p in pieces_reported(res))


def test_E3_a_key_in_confidence_set_by_is_cut(db):
    key = fake_key(seed=23)
    s, a, b, e = _edge_with(db, key)
    s.close()
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key) == {}


def test_E4_snapshots_of_a_deleted_edge_lose_the_key_too(db):
    key = fake_key(seed=24)
    s, a, b, e = _edge_with(db, key)
    snapshots(db)
    s.delete_edge(e.edge_id)
    s.close()
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key) == {}


def test_E5_a_key_in_source_id_is_renamed_everywhere_and_reported(db):
    key = fake_key(seed=25)
    url = f"https://example.com/page?token={key}"
    cc(db, "s0", "User: unrelated words about the venue.")
    s = GraphStore(db_path=db)
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    # (written before the key is named: the pipeline cut does not apply yet)
    s.add_node(Node(node_type=NodeType.FACT, label="page note", content=f"page {key}",
                    source_id=url), owner=(url, 1, 1))
    s.close()
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key) == {}
    assert res["renamed"].get("nodes.source_id") == 1
    assert res["renamed"].get("node_provenance.unit_id") == 1
    assert all(key not in i for i in res["renamed_ids"])


def test_E5b_a_new_capture_whose_source_id_carries_the_key_lands_nothing(db):
    key = fake_key(seed=26)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    gone(forget_wording(db, key))
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    s = GraphStore(db_path=db)
    IngestionPipeline(s).ingest(IngestionInput(
        source_id=f"https://example.com/page?token={key}",
        content="User: The venue is booked for the offsite on Friday."))
    s.add_node(Node(node_type=NodeType.FACT, label="n", content="n",
                    source_id=f"https://x.example/?token={key}", session_key=key,
                    project_key=key, origin_source=key, confidence_set_by=key),
               owner=(f"unit-{key}", 1, 1))
    s.register_unit(f"unit-{key}", f"/tmp/{key}", [f"unit:{key}"])
    a = s.add_node(Node(node_type=NodeType.FACT, label="a", content="a note"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="b", content="b note"))
    s.add_edge(Edge(source_node_id=a.node_id, target_node_id=b.node_id,
                    edge_type=EdgeType.RELATED_TO, confidence_set_by=key))
    s.close()
    assert everywhere(db, key) == {}


def test_E6_snapshots_of_a_deleted_node_lose_a_key_in_source_id(db):
    key = fake_key(seed=27)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="page", content="page",
                        source_id=f"https://example.com/?token={key}"))
    snapshots(db)
    s.delete_node(n.node_id)
    s.close()
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key) == {}


def test_E7_a_key_found_only_in_a_snapshot_is_cut_and_counts(db):
    key = fake_key(seed=28)
    cc(db, "s1", "User: unrelated words about the venue.")
    snaps = Path(db + ".snapshots")
    snapshot_db(db, use_gzip=False, out_dir=snaps)
    snap = next(snaps.glob("*.db"))
    conn = sqlite3.connect(str(snap))
    conn.execute("UPDATE nodes SET label = ? WHERE rowid = (SELECT min(rowid) FROM nodes)",
                 (f"leak {key}",))
    conn.commit()
    conn.close()
    res = forget_wording(db, key)
    gone(res)                                          # not "nothing found": the snapshot held it
    assert everywhere(db, key) == {}


def test_CRLF_a_key_wrapped_with_crlf_and_a_tab_leaves_no_half_in_a_derived_fact(db):
    key = fake_key(seed=29)
    cc(db, "s1", f"User: We decided to use {key[:20]}\r\n\t{key[20:]} as the deploy key.\r\n"
                 "User: The venue is booked.")
    snapshots(db)
    res = forget_wording(db, key)
    gone(res)
    assert everywhere(db, key, key[:20], key[20:]) == {}
    # and the same paste again, after the forget, lands neither half
    cc(db, "s2", f"User: We decided to use {key[:20]}\r\n\t{key[20:]} as the deploy key.")
    assert everywhere(db, key, key[:20], key[20:]) == {}


def test_E8_a_snapshot_that_still_holds_the_key_after_the_scrub_is_a_leftover(db, monkeypatch):
    key = fake_key(seed=30)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    snapshots(db)
    from revien.forget.engine import ForgetEngine
    monkeypatch.setattr(ForgetEngine, "_scrub_snapshot_wording", lambda self, p, lit: [])
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    hits = [l for l in res["leftovers"] if l["kind"] == "snapshot_residue"]
    wheres = {h["where"] for h in hits}
    assert any(w.startswith("raw bytes") for w in wheres) and any(w.startswith("table ") for w in wheres)
    assert key not in json.dumps(res)


def test_E9_an_identity_that_cannot_be_renamed_is_a_leftover_never_gone_everywhere(db):
    key = fake_key(seed=31)
    s = GraphStore(db_path=db)
    conn = s._get_conn()
    # two units that differ only by the key: the cut form of one is the other
    for uid in (f"unit-{key}", "unit-"):
        conn.execute("INSERT INTO capture_units (unit_id, path, path_keys, first_seen, last_seen)"
                     " VALUES (?, NULL, '[]', 'x', 'x')", (uid,))
    conn.commit()
    s.close()
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    assert any(l["kind"] == "identity_unrenamed" for l in res["leftovers"])
    assert "capture_units" in res["remaining"]
    assert key not in json.dumps(res)
