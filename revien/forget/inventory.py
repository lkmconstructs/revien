"""One inventory of the text columns of the store.

``forget --wording`` has to find a secret wherever the store keeps text, and
the capture gate has to keep it out of the same places. A hand-picked list of
columns is how a copy ends up where nothing looks, so the inventory is read
from the schema: every TEXT (or BLOB, or untyped) column of every table, in
the live database and in each snapshot of it. The cut, the residue check
behind "gone everywhere", the piece report, the snapshot scrub and the
capture gate all use it.

A column is left out only by an explicit entry in ``EXCLUDED`` with the
reason it cannot hold anything a person typed. A column the schema gains and
nobody has reviewed is included until someone does (and
``tests/test_forget_inventory.py`` fails until it is classified).
"""
import sqlite3
from typing import Dict, List, Tuple

#: Revien's own exact constant actor and op strings. A cell that equals one
#: of them is never a person's text, so the piece report and the residue
#: check skip it by exact equality (not by the kind of audit row).
OWN_ACTORS = frozenset({
    "", "forget", "api:forget", "cli:forget", "cli:restore", "cli:delete",
    "api:delete", "api:retention_sweep", "scheduler:retention",
    "langchain:clear", "csl_supersession", "csl_tension_gate", "alias_pass",
    "migration_001", "user", "system", "rule_based", "llm",
})

_ID = "an id Revien generates"
_TIME = "a time Revien writes from a datetime"
_ENUM = "a value from a fixed set of names"
_HASH = "a keyed digest or hash, never the text"
_TYPE = ("a built-in type name or one an extension registered in code "
         "(lowercase snake_case, checked at registration); no other value "
         "is written, and a database holding any other is refused at open")

#: (table, column) -> why it can hold nothing a person typed.
EXCLUDED: Dict[Tuple[str, str], str] = {
    ("audit_log", "node_id"): _ID,
    ("audit_log", "op"): "an operation name chosen by the code",
    ("audit_log", "ts"): _TIME,
    ("capture_units", "first_seen"): _TIME, ("capture_units", "last_seen"): _TIME,
    ("distill_manifest", "entity_id"): _ID,
    ("distill_manifest", "anchor_node_id"): _ID,
    ("distill_manifest", "current_node_id"): _ID,
    ("distill_manifest", "content_hash"): _HASH,
    ("edges", "edge_id"): _ID, ("edges", "edge_type"): _TYPE,
    ("edges", "source_node_id"): _ID, ("edges", "target_node_id"): _ID,
    ("edges", "created_at"): _TIME, ("edges", "confidence_set_at"): _TIME,
    ("edges", "invalidated_at"): _TIME,
    ("node_provenance", "node_id"): _ID,
    ("nodes", "node_id"): _ID, ("nodes", "node_type"): _TYPE,
    ("nodes", "created_at"): _TIME, ("nodes", "last_accessed"): _TIME,
    ("nodes", "source_type"): _ENUM, ("nodes", "confidence_set_at"): _TIME,
    ("nodes", "last_referenced"): _TIME, ("nodes", "invalidated_at"): _TIME,
    ("nodes", "source_modality"): _ENUM, ("nodes", "recorded_at"): _TIME,
    ("nodes", "event_time_start"): _TIME, ("nodes", "event_time_end"): _TIME,
    ("nodes", "event_time_granularity"): _ENUM, ("nodes", "valid_from"): _TIME,
    ("nodes", "valid_until"): _TIME,
    ("nodes", "origin_runtime"): "a runtime name from a fixed set",
    ("recall_receipts", "recall_id"): _ID,
    ("recall_receipts", "node_ids"): "a list of node ids",
    ("recall_receipts", "issued_at"): _TIME, ("recall_receipts", "expires_at"): _TIME,
    ("recall_receipts", "status"): _ENUM, ("recall_receipts", "closed_at"): _TIME,
    ("store_meta", "key"): "a setting name chosen by the code",
    ("supersession_candidates", "existing_node_id"): _ID,
    ("supersession_candidates", "new_node_id"): _ID,
    ("supersession_candidates", "action"): _ENUM,
    ("supersession_candidates", "created_at"): _TIME,
    ("supersession_candidates", "resolved_at"): _TIME,
    ("suppressions", "unit_id"): "the record's own id, forget:<uuid>",
    ("suppressions", "wording_fps"): _HASH, ("suppressions", "sentence_fps"): _HASH,
    ("suppressions", "short_fps"): _HASH, ("suppressions", "window_fps"): _HASH,
    ("suppressions", "literal_fps"): _HASH, ("suppressions", "scope"): _ENUM,
    ("suppressions", "created_at"): _TIME, ("suppressions", "status"): _ENUM,
    ("suppressions", "lifted_at"): _TIME, ("suppressions", "key_check"): _HASH,
    ("sync_cursors", "updated_at"): _TIME,
    ("use_judgments", "node_id"): _ID,
    ("use_judgments", "mode"): _ENUM, ("use_judgments", "judged_at"): _TIME,
    ("pending_embeds", "node_id"): _ID, ("pending_embeds", "queued_at"): _TIME,
}

#: Columns that name a source, a unit or a project. Cutting a secret out of
#: one changes an identity, so the same cut is made in every one of them and
#: the rename is reported (ids only).
IDENTITY = frozenset({
    ("nodes", "source_id"), ("nodes", "origin_source"), ("nodes", "session_key"),
    ("nodes", "project_key"), ("capture_units", "unit_id"),
    ("capture_units", "path"), ("capture_units", "path_keys"),
    ("node_provenance", "unit_id"), ("suppressions", "path_keys"),
    ("sync_cursors", "adapter_name"), ("sync_cursors", "cursor"),
    ("distill_manifest", "note_stem"), ("distill_manifest", "section"),
    ("use_judgments", "episode_key"),
})

_SKIP_PREFIXES = ("sqlite_", "vec_nodes")


def text_bearing(declared_type) -> bool:
    """Can a column of this declared type hold text? SQLite decides by the
    type's affinity (https://sqlite.org/datatype3.html §3.1): INTEGER and REAL
    columns keep numbers; TEXT, BLOB (and no type) and NUMERIC columns ("JSON",
    "DATETIME", "BOOLEAN", "DECIMAL", ...) will keep a string that is not a
    number as text. So everything except INTEGER and REAL affinity counts."""
    t = (declared_type or "").upper()
    if "INT" in t:
        return False
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return True
    if "BLOB" in t or t == "":
        return True
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return False
    return True                    # NUMERIC affinity can still hold a string


def columns(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """{table: [text column, ...]} read from the schema of ``conn``."""
    out: Dict[str, List[str]] = {}
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error:
        return out
    for name, sql in tables:
        if name.startswith(_SKIP_PREFIXES) or (sql and "VIRTUAL" in sql.upper()):
            continue
        try:
            info = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        except sqlite3.Error:
            continue
        cols = [r[1] for r in info
                if text_bearing(r[2]) and (name, r[1]) not in EXCLUDED]
        if cols:
            out[name] = cols
    return out


def all_text_columns(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    """Every (table, column) that is TEXT-like, excluded or not (for the
    schema test)."""
    out = []
    for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall():
        if name.startswith(_SKIP_PREFIXES) or (sql and "VIRTUAL" in sql.upper()):
            continue
        for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall():
            if text_bearing(r[2]):
                out.append((name, r[1]))
    return out


def is_own(value) -> bool:
    return isinstance(value, str) and value in OWN_ACTORS
