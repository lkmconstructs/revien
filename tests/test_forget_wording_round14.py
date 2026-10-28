"""Round 14: a record two sessions vouch for is judged against the captured
text of both; the last write paths re-cut a named string; a failing test never
prints a key."""
import json

import pytest

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import cc, forget_wording
from tests.test_forget_wording_verify2 import everywhere, fake_key, pieces_reported


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r14.db")


def _labels(db):
    s = GraphStore(db_path=db)
    try:
        return {(n.node_type.value, n.label) for n in s.list_nodes(limit=100)}
    finally:
        s.close()


def test_a_decision_two_sessions_vouch_for_is_not_cut_for_a_longer_key_in_one(db):
    cc(db, "A", "User: We decided to rotate the staging-credentials.")
    cc(db, "B", "User: We decided to rotate the staging-credentials-77x1 today.")
    before = _labels(db)
    assert ("decision", "rotate the staging-credentials") in before
    res = forget_wording(db, "staging-credentials-77x1")
    assert ("decision", "rotate the staging-credentials") in _labels(db)
    assert res["scope"] != "gone everywhere"


def _two_sessions(db, a_text):
    """A decision vouched for by two units; only unit B's captured text holds
    the key. Unit A's text holds the same words standing alone."""
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session A", content=a_text),
               owner=("unitA", 1, 1))
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content="User: We decided to rotate the staging-credentials-77x1 today."),
               owner=("unitB", 1, 1))
    # (its own words do not repeat the stretch: only the sessions' texts do)
    d = s.add_node(Node(node_type=NodeType.DECISION, label="rotate the staging-credentials",
                        content="A rotation decision was recorded."),
                   owner=("unitB", 1, 1))            # written by B's capture ...
    s.record_provenance(d.node_id, "unitA")          # ... and later vouched for by A
    s.close()
    return d.node_id


@pytest.mark.parametrize("a_text", [
    "User: We decided to rotate the staging-credentials.",
    "User: We decided to ROTATE THE Staging-Credentials.",          # differs only by case
])
def test_the_captured_text_of_every_vouching_unit_is_evidence(db, a_text):
    did = _two_sessions(db, a_text)
    res = forget_wording(db, "staging-credentials-77x1")
    s = GraphStore(db_path=db)
    try:
        label = s.get_node(did).label
    finally:
        s.close()
    ok = label == "rotate the staging-credentials"
    assert ok, "the decision was cut although its words also stand alone"
    assert res["scope"] != "gone everywhere"
    assert pieces_reported(res)


def test_a_decision_vouched_for_by_one_unit_is_still_cut_when_it_stops_inside(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content="User: We decided to rotate the staging-credentials-77x1 today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label="rotate the staging-credentials",
                        content="We decided to rotate the staging-credentials-77x1 today."),
                   owner=("unitB", 1, 1))
    s.close()
    forget_wording(db, "staging-credentials-77x1")
    s = GraphStore(db_path=db)
    try:
        assert not s.get_node(d.node_id).label.endswith("credentials")
    finally:
        s.close()


# ── the last write paths ─────────────────────────────────────────────────

def _named(db, seed):
    key = fake_key(seed=seed)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    assert forget_wording(db, key)["scope"] == "gone everywhere"
    return key


def test_update_edge_cannot_put_the_key_in_confidence_set_by(db):
    key = _named(db, 61)
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="a", content="a note"))
    b = s.add_node(Node(node_type=NodeType.FACT, label="b", content="b note"))
    e = Edge(source_node_id=a.node_id, target_node_id=b.node_id, edge_type=EdgeType.RELATED_TO)
    s.add_edge(e)
    s.update_edge(e.edge_id, confidence_set_by=f"agent:{key}")
    s.close()
    clean = everywhere(db, key) == {}
    assert clean


def test_the_manifest_cannot_carry_the_key_and_is_still_found(db):
    key = _named(db, 62)
    s = GraphStore(db_path=db)
    stem = f"note-{key}"
    s.replace_distill_manifest(stem, [{"entity_id": "e1", "anchor_node_id": "n1",
                                       "current_node_id": "n1", "content_hash": "h",
                                       "section": f"sec {key}"}])
    s.add_manifest_row(stem, "e1", "n2", "n2", "h2", f"sec {key}")
    s.update_manifest_row(stem, "n1", "n1", "h3")
    rows = s.get_distill_manifest(stem)
    found = len(rows) == 2
    s.close()
    assert found
    clean = everywhere(db, key) == {}
    assert clean


def test_lifting_a_record_cannot_put_the_key_in_lifted_by(db):
    key = _named(db, 63)
    s = GraphStore(db_path=db)
    rec = s.list_suppressions("active")[0]
    s.lift_suppression(rec["id"], f"cli:restore:{key}")
    s.close()
    clean = everywhere(db, key) == {}
    assert clean


def test_a_cursor_name_is_cut_inside_the_write_transaction(db, monkeypatch):
    """The denylist is read with the write lock held: the cut happens after
    the transaction opened, not before it."""
    key = _named(db, 64)
    s = GraphStore(db_path=db)
    seen = []
    orig = GraphStore._cut_ident

    def spy(self, value):
        seen.append(self._txn_depth)
        return orig(self, value)

    monkeypatch.setattr(GraphStore, "_cut_ident", spy)
    from datetime import datetime, timezone
    s.set_sync_cursor(f"adapter-{key}", datetime.now(timezone.utc))
    s.reset_sync_cursor(f"adapter-{key}")
    _c, gen = s.get_sync_state(f"adapter-{key}")
    s.set_sync_cursor_if_generation(f"adapter-{key}", datetime.now(timezone.utc), gen)
    s.close()
    writes_in_txn = seen[:2] + seen[3:]
    assert all(d > 0 for d in writes_in_txn), seen


# ── a collision never moves nodes.source_id onto the other source's name ─

def test_a_colliding_source_id_is_kept_and_listed(db):
    key = fake_key(seed=65)
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="a", content="a note",
                        source_id=f"unit-{key}"), owner=(f"unit-{key}", 1, 1))
    b = s.add_node(Node(node_type=NodeType.FACT, label="b", content="b note",
                        source_id="unit-"), owner=("unit-", 1, 1))
    s.close()
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute("UPDATE nodes SET source_id = ? WHERE node_id = ?", (f"unit-{key}", a.node_id))
    conn.execute("INSERT OR IGNORE INTO capture_units (unit_id, path, path_keys, first_seen, last_seen)"
                 " VALUES ('unit-', NULL, '[]', 'x', 'x')")
    conn.commit()
    conn.close()
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    listed = any(l["kind"] == "identity_unrenamed" and "nodes.source_id" in l["columns"]
                 for l in res["leftovers"])
    assert listed
    conn = sqlite3.connect(db)
    try:
        src = dict(conn.execute("SELECT node_id, source_id FROM nodes"))
    finally:
        conn.close()
    assert src[a.node_id] != src[b.node_id]
