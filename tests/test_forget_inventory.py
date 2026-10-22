"""The text-column inventory is read from the schema; a column the schema
gains must be looked at by a person (included, or excluded with a reason)."""
import sqlite3

from revien.forget import inventory
from revien.graph.store import GraphStore

#: Every TEXT-like column that is scanned, cut and guarded. A new column in
#: the schema fails test_every_text_column_has_been_classified until it is
#: added here (it is included at run time either way) or to EXCLUDED with the
#: reason it can hold nothing a person typed.
INCLUDED = {
    ("audit_log", "actor"), ("audit_log", "before_json"), ("audit_log", "after_json"),
    ("capture_units", "unit_id"), ("capture_units", "path"), ("capture_units", "path_keys"),
    ("distill_manifest", "note_stem"), ("distill_manifest", "section"),
    ("edges", "metadata"), ("edges", "confidence_set_by"), ("edges", "source_context"),
    ("node_provenance", "unit_id"),
    ("nodes", "label"), ("nodes", "content"), ("nodes", "source_id"),
    ("nodes", "metadata"), ("nodes", "confidence_set_by"), ("nodes", "source_context"),
    ("nodes", "event_time_text"), ("nodes", "origin_source"), ("nodes", "project_key"),
    ("nodes", "session_key"),
    ("recall_receipts", "runtime"), ("recall_receipts", "query"),
    ("store_meta", "value"),
    ("supersession_candidates", "reason"), ("supersession_candidates", "trace"),
    ("supersession_candidates", "resolution"),
    ("suppressions", "path_keys"), ("suppressions", "actor"), ("suppressions", "lifted_by"),
    ("sync_cursors", "adapter_name"), ("sync_cursors", "cursor"),
    ("use_judgments", "episode_key"), ("use_judgments", "reason"), ("use_judgments", "shared_terms"), ("use_judgments", "query"),
}


def _store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "inv.db"))
    conn = s._get_conn()
    s._enqueue_embed_stmt(conn, "x")           # the lazily made queue table
    conn.commit()
    return s


def test_every_text_column_has_been_classified(tmp_path):
    s = _store(tmp_path)
    try:
        found = set(inventory.all_text_columns(s._get_conn()))
    finally:
        s.close()
    unknown = found - INCLUDED - set(inventory.EXCLUDED)
    assert not unknown, f"classify these columns (include them or exclude with a reason): {sorted(unknown)}"


def test_no_classification_is_stale_and_every_exclusion_has_a_reason(tmp_path):
    s = _store(tmp_path)
    try:
        found = set(inventory.all_text_columns(s._get_conn()))
    finally:
        s.close()
    assert not (INCLUDED - found), sorted(INCLUDED - found)
    stale = {k for k in inventory.EXCLUDED if k not in found}
    assert not stale, sorted(stale)
    assert all(isinstance(r, str) and len(r) > 8 for r in inventory.EXCLUDED.values())
    assert not (INCLUDED & set(inventory.EXCLUDED))


def test_the_inventory_is_what_the_schema_says_not_a_list(tmp_path):
    s = _store(tmp_path)
    conn = s._get_conn()
    try:
        conn.execute("CREATE TABLE brand_new (id INTEGER PRIMARY KEY, note TEXT, n INTEGER)")
        cols = inventory.columns(conn)
        assert cols["brand_new"] == ["note"]                 # included until a person decides
        for want in (("nodes", "source_id"), ("nodes", "origin_source"),
                     ("nodes", "confidence_set_by"), ("nodes", "session_key"),
                     ("nodes", "project_key"), ("edges", "confidence_set_by"),
                     ("capture_units", "unit_id"), ("node_provenance", "unit_id"),
                     ("sync_cursors", "cursor"), ("audit_log", "actor")):
            assert want[1] in cols[want[0]], want
    finally:
        s.close()


def test_own_actor_names_are_matched_exactly():
    assert inventory.is_own("api:forget") and not inventory.is_own("api:forget-secret-9z8y")
    assert not inventory.is_own("api:forget ")


def test_a_column_of_any_type_that_can_hold_text_is_seen(tmp_path):
    s = _store(tmp_path)
    conn = s._get_conn()
    try:
        conn.execute("CREATE TABLE dummy (payload JSON, at DATETIME, flag BOOLEAN, "
                     "raw BLOB, free, n INTEGER, w REAL, big BIGINT)")
        seen = set(inventory.all_text_columns(conn))
        for col in ("payload", "at", "flag", "raw", "free"):
            assert ("dummy", col) in seen, col
        assert ("dummy", "n") not in seen and ("dummy", "w") not in seen
        assert ("dummy", "big") not in seen
        assert set(inventory.columns(conn)["dummy"]) == {"payload", "at", "flag", "raw", "free"}
        # and the classification test would fail on it
        unknown = seen - INCLUDED - set(inventory.EXCLUDED)
        assert ("dummy", "payload") in unknown
    finally:
        s.close()
