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
    #: An id read from inside the source that travels with a copy of it
    #: (a Codex rollout's session id), as ``sid:<runtime>:<id>``. It is an
    #: extra key for the unit, never its id.
    sid: Optional[str] = None

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
        sid = None
        if runtime == "codex":
            inner = meta.get("session_id")
            if isinstance(inner, str) and inner.strip():
                sid = f"sid:codex:{inner.strip()}"
        return UnitRef(session_unit(runtime, session), path, sid=sid)
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
    if unit.sid:
        keys = keys + [unit.sid]
    store.register_unit(unit.unit_id, path, keys)


# How much of a forgotten source a capture has to carry to count as a COPY of
# it: at least COPY_OF_ORIGIN of the source's own long sentences (a copy has
# all of them, give or take an edit), and those have to be at least
# COPY_OF_CAPTURE of the capture's own long sentences (a long conversation
# that merely quotes a short one is not a copy of it), and at least
# COPY_MIN_SHARED sentences in all, so that one or two shared sentences (a
# line of boilerplate) never make two different sources look like copies.
COPY_OF_ORIGIN = 0.8
COPY_OF_CAPTURE = 0.5
COPY_MIN_SHARED = 2


def _noun(unit_id: Optional[str]) -> str:
    """What to call the thing that has to change, by unit type."""
    uid = unit_id or ""
    if uid.startswith("session:"):
        return "session file"
    if uid.startswith("vault:"):
        return "note"
    if uid.startswith("skill:"):
        return "skill file"
    return "file"


def long_fingerprints(text: str):
    """Keyed fingerprints of the sentences in ``text`` that are long enough
    (``scrub.MIN_SENTENCE``) to be matched anywhere."""
    from revien.forget import scrub
    out = set()
    for _l, _r, body in scrub.iter_sentences(text or ""):
        key = scrub.sentence_key(body)
        if len(key) >= scrub.MIN_SENTENCE:
            out.add(fingerprint(key))
    return out


def is_copy_of(signature, capture_fps) -> bool:
    """True when a capture is a copy of a forgotten origin: it carries at
    least ``COPY_OF_ORIGIN`` of the origin's signature (the keyed fingerprints
    of all of the origin's own long sentences), and that is at least
    ``COPY_OF_CAPTURE`` of the capture's own long sentences, and at least
    ``COPY_MIN_SHARED`` sentences in all. A copy that was renamed, moved or
    had a line appended still qualifies; a different source that shares a
    sentence or two does not."""
    sig = set(signature)
    if not sig or not capture_fps:
        return False
    shared = len(sig & set(capture_fps))
    return (shared >= COPY_MIN_SHARED and shared / len(sig) >= COPY_OF_ORIGIN
            and shared / len(capture_fps) >= COPY_OF_CAPTURE)


IDENTITY, CONTENT = "identity", "content"


def _origin_match(rec: Dict, unit: UnitRef, keys: set, capture_fps) -> Optional[str]:
    """How an origin-scoped record speaks about THIS capture, if it does:
    ``IDENTITY`` when the capture comes from the forgotten origin itself (the
    same unit id, which for a session is the id inside the file; the same file
    by real path or inode; an id that travels with a copy), ``CONTENT`` when it
    only looks like a copy of it (see is_copy_of), or None. A different source
    that merely holds one of the same sentences is not covered."""
    if rec["unit_id"].startswith("forget:"):
        covered = set(rec["path_keys"])
        if unit.stable and f"unit:{unit.unit_id}" in covered:
            return IDENTITY
        if keys and keys & covered:
            return IDENTITY
        if is_copy_of(rec.get("signature_fps") or [], capture_fps):
            return CONTENT
        return None
    if unit.stable and rec["unit_id"] == unit.unit_id:
        return IDENTITY
    if rec["unit_id"].startswith(FILE_LEVEL) and keys \
            and keys & set(rec["path_keys"]):
        return IDENTITY
    return None


def _origin_applies(rec: Dict, unit: UnitRef, keys: set, capture_fps) -> bool:
    return _origin_match(rec, unit, keys, capture_fps) is not None


def _capture_keys(unit: UnitRef) -> set:
    """Every key that names where this capture comes from: its file (real
    path, inode) and the id inside it, when it has one."""
    verified = unit.path if (unit.path and is_real_file(unit.path)) else None
    keys = set(identity_keys(verified)) if verified else set()
    if unit.sid:
        keys.add(unit.sid)
    return keys


def known_detail(store, unit: UnitRef, content: str,
                 match_content: Optional[str] = None):
    """(long fps, short fps, content_only, sources) of the sentences a forget
    left out of captures from this origin.

    A capture that comes from the forgotten origin itself (identity) loses
    every sentence the forget took out. A capture only recognised as a copy by
    its content loses NOTHING but the sentences that held what was forgotten
    (``held_fps``): the rest of the document, and every short fragment, lands.
    ``content_only`` is the set of long fps that only a content match asked
    for; ``sources`` names the origins matched by content. ``match_content``
    is the whole text the capture belongs to, when ``content`` is only a part
    of it (one message of a conversation)."""
    records = [r for r in store.list_suppressions("active", include_wording=True)
               if r.get("sentence_fps") or r.get("short_fps")]
    if not records:
        return set(), set(), set(), []
    keys = _capture_keys(unit)
    capture_fps = None
    long_fps, short_fps = set(), set()
    identity_long, content_long = set(), set()
    sources: List[str] = []
    for rec in records:
        if rec.get("scope") == "global":
            long_fps |= set(rec["sentence_fps"])
            identity_long |= set(rec["sentence_fps"])
            continue
        if capture_fps is None:
            capture_fps = long_fingerprints(
                match_content if match_content is not None else content)
        how = _origin_match(rec, unit, keys, capture_fps)
        if how == IDENTITY:
            long_fps |= set(rec["sentence_fps"])
            short_fps |= set(rec["short_fps"])
            identity_long |= set(rec["sentence_fps"])
        elif how == CONTENT:
            held = set(rec.get("held_fps") or [])
            long_fps |= held
            content_long |= held
            sources += [k[len("unit:"):] for k in rec["path_keys"]
                        if k.startswith("unit:")]
    return long_fps, short_fps, content_long - identity_long, sources


def known(store, unit: UnitRef, content: str, match_content: Optional[str] = None):
    """(long fingerprints, short fingerprints): see known_detail."""
    long_fps, short_fps, _c, _s = known_detail(store, unit, content, match_content)
    return long_fps, short_fps


def _forgotten_test(long_fps, short_fps):
    from revien.forget import scrub

    def forgotten(key: str) -> bool:
        fp = fingerprint(key)
        return fp in (long_fps if len(key) >= scrub.MIN_SENTENCE else short_fps)
    return forgotten


def check(store, unit: UnitRef, content: str) -> Optional[Block]:
    """The first suppression record that forbids this capture, or None."""
    records = store.list_suppressions("active", include_wording=True)
    if not records:
        return None
    fp = fingerprint(content) if content and content.strip() else None
    verified = unit.path if (unit.path and is_real_file(unit.path)) else None
    keys = set(identity_keys(verified)) if verified else set()
    named = keys | ({unit.sid} if unit.sid else set())
    capture_fps = None
    for rec in records:
        file_level = rec["unit_id"].startswith(FILE_LEVEL)
        by_unit = unit.stable and rec["unit_id"] == unit.unit_id
        by_key = file_level and bool(named & set(rec["path_keys"]))
        by_text = fp is not None and fp in rec["wording_fps"]       # a forgotten node's own text
        by_whole = fp is not None and fp in rec["content_fps"]      # the forgotten record, whole
        # A capture that is the forgotten record's whole text but comes from
        # no identity of the origin is only a copy by content: it is not
        # refused, the sentences that held what was forgotten are cut (see
        # strip_forgotten).
        by_fp = by_text or (by_whole and (by_unit or by_key))
        if rec["kind"] == "wording":
            # A standing denylist entry, not a suppressed source: it blocks a
            # capture only when the whole text is a forgotten text, and then
            # only a capture that is the forgotten origin's (or a copy of it),
            # or one with no identity of its own to tell apart.
            if not by_fp:
                continue
            if rec.get("scope") == "global" or not unit.stable:
                return Block(rec, "the same text was forgotten")
            if capture_fps is None:
                capture_fps = long_fingerprints(content)
            if _origin_applies(rec, unit, named, capture_fps):
                return Block(rec, "the same text was forgotten")
            continue
        if not (by_unit or by_key or by_fp):
            continue
        noun = _noun(rec["unit_id"])
        if rec["kind"] == "permanent":
            return Block(rec, "no file behind it, so suppressed for good")
        same_file = bool(keys & set(rec["path_keys"]))
        if not same_file:
            # Matched by name (or by text) with no on-disk evidence about the
            # recorded file: a name-level suppression is never lifted by a
            # caller's say-so, and a path that does not exist proves nothing.
            return Block(rec, f"no evidence the {noun} changed")
        if by_fp:
            return Block(rec, "the same text was forgotten")
        if file_level:
            current = file_fingerprint(real_path(verified))
            if current is None or current == rec["file_fp"]:
                return Block(rec, f"the {noun} has not changed")
            continue  # the file changed: new evidence
        # A vault section: its own text decides, not the whole note.
        continue
    return None


def strip_forgotten_report(store, unit: UnitRef, content: str,
                           match_content: Optional[str] = None):
    """Leave forgotten sentences out of a capture. THE STANDING DENYLIST.

    Forget is a rule, not a one-time subtraction, and it is bound to the
    origin that was forgotten, not to the world:

    * a capture from the forgotten origin itself (its unit id, which for a
      session is the id inside the file; its file by real path or inode)
      loses every sentence the forget took out;
    * a capture only recognised as a COPY of it by content (see is_copy_of)
      loses nothing but the sentences that held what was forgotten
      (``held_fps``); the rest of the document lands, and so does every
      boilerplate sentence;
    * a ``forget --everywhere`` also keeps the exact wording it named out of
      every source.

    Another source that merely holds the same sentence keeps its own copy.
    Returns (content, how many sentences were left out, report); the report
    is None when nothing was, else says which lines went (never what they
    said) and whether the match was by identity or by content alone."""
    from revien.forget import scrub

    if not content:
        return content, 0, None
    long_fps, short_fps, content_only, sources = known_detail(
        store, unit, content, match_content)
    if not long_fps and not short_fps:
        return content, 0, None
    forgotten = _forgotten_test(long_fps, short_fps)
    lines, by_content = [], 0
    for start, _end, body in scrub.iter_sentences(content):
        key = scrub.sentence_key(body)
        if len(key) >= scrub.MIN_SHORT and forgotten(key):
            lines.append(content.count("\n", 0, start) + 1)
            if len(key) >= scrub.MIN_SENTENCE and fingerprint(key) in content_only:
                by_content += 1
    new_text, n = scrub.strip_sentences(content, forgotten, min_len=scrub.MIN_SHORT)
    if not n:
        return new_text, 0, None
    report = {
        "left_out": n, "lines": sorted(set(lines)),
        "matched_by": ("content" if by_content == n
                       else "identity" if by_content == 0 else "both"),
        "sources": sorted(set(sources)) if by_content else [],
    }
    return new_text, n, report


def strip_forgotten(store, unit: UnitRef, content: str,
                    match_content: Optional[str] = None):
    """(content without the forgotten sentences, how many): see
    strip_forgotten_report."""
    text, n, _report = strip_forgotten_report(store, unit, content, match_content)
    return text, n


def enforce_after_write(store, unit: UnitRef, content: str, new_ids, merged_ids=(),
                        existing_ids=()):
    """The last word, said inside one write transaction.

    A capture checks the gate before it extracts, and again before it writes,
    but a forget can still land between that check and the commit (it can come
    from another process, another connection). So after a capture has written,
    this runs under BEGIN IMMEDIATE, which makes it take turns with a forget's
    own write transaction: either the forget's denylist is visible here and
    the capture is undone, or the forget has not committed yet and its removal
    pass, which runs after the denylist is committed, finds what was written.
    Either order, a forgotten sentence cannot stay. Returns the Block when the
    whole capture was undone, else None. Nodes whose words were cut are
    re-queued for embedding. ``existing_ids`` are records the capture
    refreshed in place (a keyed context node): their forgotten sentences are
    cut, but a record that was there before the capture is never deleted by
    undoing it."""
    from revien.forget import scrub

    with store.transaction(immediate=True) as conn:
        block = check(store, unit, content)
        if block is not None:
            for nid in new_ids:
                store.delete_node(nid)
                # A capture that is undone leaves no trace: the creation
                # rows quote what was just refused.
                conn.execute("DELETE FROM audit_log WHERE node_id = ?", (nid,))
            for nid in merged_ids:
                store.delete_provenance(nid, [unit.owner])
            return block
        long_fps, short_fps = known(store, unit, content)
        if not long_fps and not short_fps:
            return None
        forgotten = _forgotten_test(long_fps, short_fps)

        fresh = set(new_ids)
        for nid in list(new_ids) + [i for i in existing_ids if i not in fresh]:
            row = conn.execute(
                "SELECT node_type, label, content FROM nodes WHERE node_id = ?",
                (nid,)).fetchone()
            if row is None:
                continue
            ntype, label, text = row
            if ntype in ("context", "skill"):
                new_text, n = scrub.strip_sentences(text, forgotten, min_len=scrub.MIN_SHORT)
                if n:
                    new_label, _ = scrub.strip_sentences(label, forgotten, min_len=scrub.MIN_SHORT)
                    if not new_label.strip():
                        first = next(iter(scrub.iter_sentences(new_text)), None)
                        new_label = (first[2] if first else "[removed]")[:200]
                    conn.execute(
                        "UPDATE nodes SET content = ?, label = ? WHERE node_id = ?",
                        (new_text, new_label[:200], nid))
                    store._enqueue_embed_stmt(conn, nid)
            elif nid in fresh and text \
                    and len(scrub.sentence_key(text)) >= scrub.MIN_SHORT \
                    and forgotten(scrub.sentence_key(text)):
                store.delete_node(nid)
                conn.execute("DELETE FROM audit_log WHERE node_id = ?", (nid,))
    return None
