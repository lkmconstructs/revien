"""Round 13: a derived stretch is cut only when every place it stands in the
capture is a cut occurrence; every identity writer re-cuts a named string; the
inventory sees any column type that can hold text; collisions never merge."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.watch import snapshot_db
from tests.test_forget_wording import cc, forget_wording
from tests.test_forget_wording_ash2 import everywhere, fake_key, pieces_reported, snapshots


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r13.db")


def _labels(db):
    s = GraphStore(db_path=db)
    try:
        return {(n.node_type.value, n.label) for n in s.list_nodes(limit=100)}
    finally:
        s.close()


# ── ambiguity: a stretch that also stands alone is reported, not cut ─────

def test_a_decision_whose_words_also_stand_alone_survives_a_longer_key(db):
    cc(db, "s1", "User: We decided to rotate the staging-credentials.\n"
                 "User: We decided to rotate the staging-credentials-77x1 key.")
    res = forget_wording(db, "staging-credentials-77x1")
    assert ("decision", "rotate the staging-credentials") in _labels(db)
    assert res["scope"] != "gone everywhere"
    assert pieces_reported(res)


def test_facts_and_decisions_from_another_sentence_keep_their_words(db):
    cc(db, "s1", "User: We decided to use the deploy-token-prod.\n"
                 "User: We decided to use the deploy-token-prod-9f3k2 for builds.")
    res = forget_wording(db, "deploy-token-prod-9f3k2")
    labels = _labels(db)
    assert ("decision", "use the deploy-token-prod") in labels
    assert ("fact", "the deploy-token-prod") in labels
    assert res["scope"] != "gone everywhere"


def test_a_decision_is_not_cut_for_a_password_that_extends_its_words(db):
    cc(db, "s1", "User: We decided to reset the password.\n"
                 "User: We decided to reset the password123456 now.")
    res = forget_wording(db, "password123456")
    assert ("decision", "reset the password") in _labels(db)
    assert res["scope"] != "gone everywhere"            # and it said so


def test_a_label_inside_one_node_that_also_stands_alone_is_not_cut(db):
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="reset the password",
                        content="reset the password for the mailbox, and later reset the password123456"))
    s.close()
    res = forget_wording(db, "password123456")
    s = GraphStore(db_path=db)
    try:
        assert s.get_node(n.node_id).label == "reset the password"
    finally:
        s.close()
    assert res["scope"] != "gone everywhere"


# ── re-entry: every writer re-cuts a named string ────────────────────────

def _named(db, seed):
    key = fake_key(seed=seed)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    assert forget_wording(db, key)["scope"] == "gone everywhere"
    return key


def test_update_node_cannot_put_the_key_in_source_id_or_confidence_set_by(db):
    key = _named(db, 41)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="n", content="n note"))
    s.update_node(n.node_id, source_id=f"https://x.example/?token={key}",
                  confidence_set_by=f"agent:{key}")
    s.close()
    assert everywhere(db, key) == {}


def test_a_receipt_runtime_cannot_carry_the_key(db):
    key = _named(db, 42)
    s = GraphStore(db_path=db)
    now = datetime.now(timezone.utc)
    s.add_recall_receipt(recall_id="rc9", runtime=f"rt-{key}", query="q", node_ids=[],
                         issued_at=now, expires_at=now + timedelta(hours=1))
    s.close()
    assert everywhere(db, key) == {}


def test_store_meta_cannot_carry_the_key(db):
    key = _named(db, 43)
    s = GraphStore(db_path=db)
    s.set_meta("some_setting", f"value {key}")
    s.close()
    assert everywhere(db, key) == {}


def test_a_sync_cursor_adapter_name_cannot_carry_the_key_and_stays_one_name(db):
    key = _named(db, 44)
    s = GraphStore(db_path=db)
    name = f"adapter-{key}"
    s.set_sync_cursor(name, datetime.now(timezone.utc))
    cursor, gen = s.get_sync_state(name)
    assert cursor is not None
    assert s.set_sync_cursor_if_generation(name, datetime.now(timezone.utc), gen)
    s.close()
    assert everywhere(db, key) == {}


def test_a_judgment_episode_key_cannot_carry_the_key_and_is_still_found(db):
    key = _named(db, 45)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="n", content="n note"))
    ep = f"https://x.example/?token={key}#tool_1"
    s.record_use_judgment(episode_key=ep, node_id=n.node_id, used=True, reason="r",
                          shared_terms=[], similarity=0.5, query="q", mode="m", applied=True)
    found = s.has_use_judgment(ep, n.node_id) and s.get_use_judgment(ep, n.node_id) is not None
    s.close()
    assert found, "the judgment was not found again by its episode key"
    clean = everywhere(db, key) == {}
    assert clean


def test_an_episode_key_forgotten_with_its_row_is_renamed_not_left(db):
    key = fake_key(seed=46)
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label="n", content="n note"))
    s.record_use_judgment(episode_key=f"sess:{key}#t", node_id=n.node_id, used=True,
                          reason="r", shared_terms=[], similarity=0.5, query="q",
                          mode="m", applied=True)
    s.close()
    res = forget_wording(db, key)
    assert res["scope"] == "gone everywhere" and res["renamed"].get("use_judgments.episode_key") == 1
    assert everywhere(db, key) == {}


# ── the pipeline strips identities, or a resync duplicates the fact ─────

def test_a_resync_of_a_source_whose_id_carries_the_key_adds_no_nodes(db):
    key = fake_key(seed=47)
    url = f"https://example.com/page?token={key}"
    text = f"User: We decided to use {key} as the deploy key.\nUser: The venue is booked."

    def sync():
        s = GraphStore(db_path=db)
        try:
            IngestionPipeline(s).ingest(IngestionInput(
                source_id=url, ingest_key=url, content=text, session_key=url))
            return s.count_nodes()
        finally:
            s.close()

    sync()
    res = forget_wording(db, key)
    assert res["status"] == "forgotten"
    s = GraphStore(db_path=db)
    before = s.count_nodes()
    s.close()
    after = sync()
    assert after == before, (before, after)
    assert everywhere(db, key) == {}


# ── inventory: BLOB and untyped columns are searched and cut ────────────

def test_a_blob_or_untyped_column_holding_the_key_is_cut(db):
    key = fake_key(seed=48)
    s = GraphStore(db_path=db)
    conn = s._get_conn()
    conn.execute("CREATE TABLE extra_store (blobby BLOB, loose, js JSON)")
    conn.execute("INSERT INTO extra_store VALUES (?, ?, ?)",
                 (f"blob {key}".encode(), f"loose {key}", json.dumps({"k": f"x {key}"})))
    conn.commit()
    s.close()
    res = forget_wording(db, key)
    assert res["scope"] == "gone everywhere" and res["cut_in"].get("extra_store") == 1
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT blobby, loose, js FROM extra_store").fetchone()
    finally:
        conn.close()
    assert key.encode() not in bytes(row[0]) and key not in row[1] and key not in row[2]


# ── raw pieces: named as a piece, never as the string ───────────────────

def test_a_piece_of_the_key_in_the_database_file_is_called_a_piece(db):
    key = fake_key(seed=49)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="middle",
                    content=f"see {key[:18]}-notes and more words after it"))
    s.close()
    res = forget_wording(db, key)
    note = " ".join(l["note"] for l in res["leftovers"] if l["kind"] == "storage")
    assert "a piece of the string (not the whole of it)" in note
    assert "characters of it" in note and "the string was still found" not in note
    assert res["scope"] != "gone everywhere"


def test_a_piece_in_the_middle_of_a_snapshot_cell_is_found_by_its_bytes(db):
    key = fake_key(seed=50)
    cc(db, "s1", f"User: We decided to use {key} as the deploy key.")
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.FACT, label="middle",
                    content=f"see {key[:18]}-notes and more words after it"))
    s.close()
    snapshots(db)
    res = forget_wording(db, key)
    hits = [l for l in res["leftovers"] if l["kind"] == "snapshot_residue"]
    assert hits and any("characters of it" in h["where"] for h in hits)
    assert key not in json.dumps(res)


# ── a rename that collides never merges two sources ─────────────────────

def test_a_colliding_unit_rename_keeps_ownership_apart_and_is_listed(db):
    key = fake_key(seed=51)
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.FACT, label="a", content="a note"),
                    owner=(f"unit-{key}", 1, 1))
    b = s.add_node(Node(node_type=NodeType.FACT, label="b", content="b note"),
                    owner=("unit-", 1, 1))
    s.register_unit(f"unit-{key}")
    s.close()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE node_provenance SET unit_id = ? WHERE node_id = ?",
                 (f"unit-{key}", a.node_id))
    conn.execute("INSERT OR IGNORE INTO capture_units (unit_id, path, path_keys, first_seen, last_seen)"
                 " VALUES ('unit-', NULL, '[]', 'x', 'x')")
    conn.commit()
    conn.close()
    res = forget_wording(db, key)
    assert res["scope"] != "gone everywhere"
    assert any(l["kind"] == "identity_unrenamed" and "node_provenance.unit_id" in l["columns"]
               for l in res["leftovers"])
    conn = sqlite3.connect(db)
    try:
        units = dict(conn.execute("SELECT node_id, unit_id FROM node_provenance"))
    finally:
        conn.close()
    assert units[a.node_id] != units[b.node_id]            # still two sources
