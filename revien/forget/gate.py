"""The denylist gate: every capture path asks it before writing.

A forget of one specific thing (a fact, an entity, a decision, a preference)
leaves a content-free record (docs/FORGET_SPEC.md §6): keyed fingerprints of
the sentences that held its wording, bound to the identity of the source it
was forgotten from. This module answers, for a capture about to be written,
which of those sentences it must leave out.

A record binds to a capture only by IDENTITY:

* the unit's id (for a Claude Code session the id inside the file, sub-agent
  files included; for a conversation import the conversation id; for a Codex
  rollout its file name, plus the ``id`` in its session_meta line as one more
  key; for a Readwise highlight its id);
* the file itself, by real path or inode (so a symlink, a hard link, the same
  file moved or a case change land on the same record).

A capture whose identity does not match lands as written: Revien does not
guess from content that it is a copy. The one exception is a record made by
``forget --everywhere``, whose exact wording is kept out of every source (a
capture that is exactly that wording is refused; a sentence of it is left out).
"""

import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from revien.forget.keys import fingerprint
from revien.graph.units import (
    CAPTURE_PREFIX, identity_keys, is_real_file, item_unit, path_unit_key,
    real_path, session_unit)


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
        return (f"refused ({self.reason}); a forget said this wording must "
                "not come back. Use "
                f"'revien restore {self.record['unit_id']}' to lift it.")


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


def _capture_keys(unit: UnitRef) -> set:
    """Every key that names where this capture comes from: its file (real
    path, inode) and the id inside it, when it has one."""
    verified = unit.path if (unit.path and is_real_file(unit.path)) else None
    keys = set(identity_keys(verified)) if verified else set()
    if unit.sid:
        keys.add(unit.sid)
    return keys


def binds(rec: Dict, unit: UnitRef, keys: set) -> bool:
    """Does a record speak about THIS capture? Only if it comes from the
    source the wording was forgotten from: the same unit id, or the same file
    by real path or inode (or the id inside it). A different source that
    happens to hold one of the same sentences is not covered, and neither is a
    copy under a different identity."""
    covered = set(rec["path_keys"])
    if unit.stable and f"unit:{unit.unit_id}" in covered:
        return True
    return bool(keys and keys & covered)


# The wording that was forgotten is also kept as keyed fingerprints of its
# form (case folded, one space between words) and its length in words. A
# sentence is checked by hashing every run of that many words in it, so a NEW
# sentence that holds the wording as whole words is recognised too, without
# the wording itself being stored anywhere.
_WORD = re.compile(r"\w+", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _form(text: str) -> str:
    """The wording from its first word to its last, case folded, with one
    space between words ("Brigitte Vandermeer." -> "brigitte vandermeer")."""
    text = _SPACE.sub(" ", unicodedata.normalize("NFC", text).strip())
    toks = list(_WORD.finditer(text))
    if toks:
        text = text[toks[0].start():toks[-1].end()]
    return text.casefold()


def window_fps_for(wordings: Iterable[str]) -> List[str]:
    """``"<words>:<keyed fingerprint>"`` for each wording."""
    out = set()
    for w in wordings:
        form = _form(w or "")
        n = len(_WORD.findall(form))
        if n:
            out.add(f"{n}:{fingerprint(form)}")
    return sorted(out)


def _split_windows(entries: Iterable[str]) -> Dict[int, set]:
    wanted: Dict[int, set] = {}
    for e in entries:
        n, _, fp = e.partition(":")
        if n.isdigit() and fp:
            wanted.setdefault(int(n), set()).add(fp)
    return wanted


def holds_wording(text: str, wanted: Dict[int, set]) -> bool:
    """Does ``text`` contain, as whole words, one of the forgotten wordings?"""
    if not wanted:
        return False
    toks = list(_WORD.finditer(text))
    for n, fps in wanted.items():
        for i in range(len(toks) - n + 1):
            if fingerprint(_form(text[toks[i].start():toks[i + n - 1].end()])) in fps:
                return True
    return False


def known(store, unit: UnitRef):
    """(long fingerprints, short fingerprints, wording fingerprints, wording
    windows) of what the forgets that bind to this capture's source (and
    every ``--everywhere``) left out. By identity only: the capture's text
    plays no part in which records apply."""
    records = [r for r in store.list_suppressions("active")
               if r.get("sentence_fps") or r.get("short_fps")
               or r.get("wording_fps") or r.get("window_fps")]
    if not records:
        return set(), set(), set(), {}
    keys = _capture_keys(unit)
    long_fps, short_fps, wording, windows = set(), set(), set(), set()
    for rec in records:
        if rec.get("scope") == "global":
            wording |= set(rec["wording_fps"])
            windows |= set(rec["window_fps"])
        elif binds(rec, unit, keys):
            long_fps |= set(rec["sentence_fps"])
            short_fps |= set(rec["short_fps"])
            wording |= set(rec["wording_fps"])
            windows |= set(rec["window_fps"])
    return long_fps, short_fps, wording, _split_windows(windows)


def _forgotten_test(long_fps, short_fps, windows):
    from revien.forget import scrub

    def forgotten(key: str) -> bool:
        fp = fingerprint(key)
        if fp in (long_fps if len(key) >= scrub.MIN_SENTENCE else short_fps):
            return True
        return holds_wording(key, windows)
    return forgotten


def _json_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_strings(v)


def _scrub_audit_rows(conn, node_id: str, forgotten) -> None:
    """The audit snapshots a capture wrote for a record it then had to cut
    keep the sentence it left out: drop the snapshot bodies that hold one."""
    import json

    from revien.forget import scrub

    for rid, before, after in conn.execute(
            "SELECT id, before_json, after_json FROM audit_log WHERE node_id = ?",
            (node_id,)).fetchall():
        hit = False
        for raw in (before, after):
            if not raw:
                continue
            try:
                texts = list(_json_strings(json.loads(raw)))
            except ValueError:
                texts = [raw]
            for text in texts:
                if any(len(scrub.sentence_key(body)) >= scrub.MIN_SHORT
                       and forgotten(scrub.sentence_key(body))
                       for _l, _r, body in scrub.iter_sentences(text)):
                    hit = True
                    break
            if hit:
                break
        if hit:
            conn.execute("UPDATE audit_log SET before_json = NULL, "
                         "after_json = NULL WHERE id = ?", (rid,))


def check(store, unit: UnitRef, content: str) -> Optional[Block]:
    """The record that refuses this capture whole, or None. Only a
    ``forget --everywhere`` can: when the capture IS the exact wording it
    named. Everything else is a matter of leaving sentences out
    (strip_forgotten), never of refusing a source."""
    if not content or not content.strip():
        return None
    records = [r for r in store.list_suppressions("active")
               if r.get("scope") == "global" and r.get("wording_fps")]
    if not records:
        return None
    fp = fingerprint(content)
    for rec in records:
        if fp in rec["wording_fps"]:
            return Block(rec, "the same text was forgotten")
    return None


def strip_forgotten(store, unit: UnitRef, content: str):
    """Leave forgotten sentences out of a capture. THE DENYLIST.

    Forget is a rule, not a one-time subtraction, and it is bound to the
    origin the wording was forgotten from: a sentence that held the wording
    (at least ``scrub.MIN_SHORT`` characters) is left out of a capture that
    comes from that origin (its unit id, its file by real path or inode, the id
    inside it), and out of every source only when ``forget --everywhere`` named
    that exact wording. Another source that holds the same sentence, and a copy
    of the origin under a different identity, land as written.
    Returns (content, how many sentences were left out)."""
    from revien.forget import scrub

    if not content:
        return content, 0
    long_fps, short_fps, _wording, windows = known(store, unit)
    if not long_fps and not short_fps and not windows:
        return content, 0
    return scrub.strip_sentences(
        content, _forgotten_test(long_fps, short_fps, windows),
        min_len=scrub.MIN_SHORT)


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
        long_fps, short_fps, wording, windows = known(store, unit)
        if not long_fps and not short_fps and not wording and not windows:
            return None
        forgotten = _forgotten_test(long_fps, short_fps, windows)

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
                    _scrub_audit_rows(conn, nid, forgotten)
            elif nid in fresh and (
                    (text and len(scrub.sentence_key(text)) >= scrub.MIN_SHORT
                     and forgotten(scrub.sentence_key(text)))
                    or (label and fingerprint(label.strip()) in wording)
                    or (text and fingerprint(text.strip()) in wording)):
                # a node the capture made of what was forgotten: a fact or
                # decision that is a forgotten sentence, an entity whose name
                # is the forgotten wording
                store.delete_node(nid)
                conn.execute("DELETE FROM audit_log WHERE node_id = ?", (nid,))
    return None
