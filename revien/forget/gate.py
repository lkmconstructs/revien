"""The suppression gate: every capture path asks it before writing.

A forget leaves a content-free record (docs/FORGET_SPEC.md §6). This module
answers one question for a capture about to be written: does a record say this
must not come back?

What a record can match, and what each match means:

* the unit's stable id (a session id, a conversation id, a real file path)
* the file itself, by real path or inode (so a symlink, a hard link, the same
  file moved, or a case change all land on the same record)
* the text itself, by keyed fingerprint (so a re-timestamped or copied file
  with identical text does not come back)

A *permanent* record (a unit with no file behind it) blocks every match. A
*file* record blocks a match until the file changes, and "changed" is only
ever decided from the file on disk: the gate reads the file named by the
capture, and only a path that exists and is the record's own file can lift it.
A made-up path, or a path to some other file, proves nothing, so a name-level
match with no such evidence stays blocked.

A capture that arrives through a shared channel (mcp, hermes, ollama,
langchain, an api feed, the CLI) has no stable identity. The gate says so
(``UnitRef.stable`` is false); it only ever blocks such a capture when its text
fingerprint equals a forgotten text, and never claims more than that.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from revien.forget.keys import file_fingerprint, fingerprint
from revien.graph.units import (
    CAPTURE_PREFIX, identity_keys, is_real_file, item_unit, path_unit_key,
    real_path, session_unit)

# Units whose identity is exactly one file: the file's bytes decide "changed".
FILE_LEVEL = ("session:", "file:", "skill:")


@dataclass
class UnitRef:
    unit_id: Optional[str]          # stable identity, or None (shared channel)
    path: Optional[str] = None      # the file it came from, as reported
    synthetic: str = field(default_factory=lambda: f"{CAPTURE_PREFIX}{uuid.uuid4()}")

    @property
    def stable(self) -> bool:
        return self.unit_id is not None

    @property
    def owner(self) -> str:
        """The unit id written to node_provenance."""
        return self.unit_id or self.synthetic


@dataclass
class Block:
    record: Dict
    reason: str

    @property
    def note(self) -> str:
        return (f"suppressed ({self.reason}); a forget on "
                f"{self.record['unit_id']} said this must not come back. "
                "Use 'revien restore' to lift it.")


def unit_for_input(input_data, origin) -> UnitRef:
    """The unit a pipeline input belongs to. ``origin`` is the resolved
    (runtime, source, project, session) tuple."""
    runtime, _source, _project, session = origin
    meta = input_data.metadata if isinstance(input_data.metadata, dict) else {}
    path = meta.get("path") if isinstance(meta.get("path"), str) else None
    sid = input_data.source_id or ""
    ikey = (input_data.ingest_key or "").strip() or None
    if runtime in ("claude-code", "codex") and session:
        # A sub-agent file carries its parent's sessionId: it is that session.
        sid_override = meta.get("session_id")
        if isinstance(sid_override, str) and sid_override.strip() \
                and runtime == "claude-code":
            session = sid_override.strip()
        return UnitRef(session_unit(runtime, session), path)
    if runtime == "obsidian" and sid.startswith("vault:"):
        return UnitRef(sid, path)
    if runtime == "file" and path:
        return UnitRef("file:" + path_unit_key(path), path)
    if ikey and ikey.startswith("skill:"):
        return UnitRef(ikey, ikey[len("skill:"):])
    if ikey:
        return UnitRef(item_unit(ikey), None)
    group = getattr(input_data, "capture_id", None)
    if isinstance(group, str) and group.startswith(CAPTURE_PREFIX):
        return UnitRef(None, path, synthetic=group)
    return UnitRef(None, path)


def register(store, unit: UnitRef) -> None:
    """Note a stable unit in the registry, with its file identity when the
    reported path is a real file. Shared-channel captures are not registered."""
    if not unit.stable:
        return
    keys: List[str] = []
    path = None
    if unit.path and is_real_file(unit.path):
        path = real_path(unit.path)
        keys = identity_keys(unit.path)
    store.register_unit(unit.unit_id, path, keys)


def check(store, unit: UnitRef, content: str) -> Optional[Block]:
    """The first suppression record that forbids this capture, or None."""
    records = store.list_suppressions("active")
    if not records:
        return None
    fp = fingerprint(content) if content and content.strip() else None
    verified = unit.path if (unit.path and is_real_file(unit.path)) else None
    keys = set(identity_keys(verified)) if verified else set()
    for rec in records:
        file_level = rec["unit_id"].startswith(FILE_LEVEL)
        by_unit = unit.stable and rec["unit_id"] == unit.unit_id
        by_key = file_level and bool(keys & set(rec["path_keys"]))
        by_fp = fp is not None and (
            fp in rec["content_fps"] or fp in rec["wording_fps"])
        if not (by_unit or by_key or by_fp):
            continue
        if rec["kind"] == "permanent":
            return Block(rec, "no file behind it, so suppressed for good")
        same_file = bool(keys & set(rec["path_keys"]))
        if not same_file:
            # Matched by name (or by text) with no on-disk evidence about the
            # recorded file: a name-level suppression is never lifted by a
            # caller's say-so, and a path that does not exist proves nothing.
            return Block(rec, "no evidence the file changed")
        if by_fp:
            return Block(rec, "the same text was forgotten")
        if file_level:
            current = file_fingerprint(real_path(verified))
            if current is None or current == rec["file_fp"]:
                return Block(rec, "the file has not changed")
            continue  # the file changed: new evidence
        # A vault section: its own text decides, not the whole note.
        continue
    return None
