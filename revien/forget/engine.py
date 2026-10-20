"""The forget verb (docs/FORGET_SPEC.md §1-§6).

Forget is privacy for ONE specific thing: a fact, an entity, a decision, a
preference, or a named piece of wording. It removes the node, cuts that
wording out of the stored conversation of the source it came from, scrubs every
copy the database keeps, leaves a content-free record so the same wording does
not come back from that source (named by its identity: a session id, a
conversation id, a real path or inode), and says exactly what it did and what
it does not cover.

It does not forget a whole conversation: a context node is refused (use
delete to remove a conversation, or forget the detail in it). It is the only
code path that writes a ``forget`` audit entry; the verb is recorded by being
here, never by a reason string a caller passes.

Who owns what comes from ``node_provenance``, written when each node was
written. Nothing here infers ownership from edges or text. A node whose
ownership is unproven is kept and listed; it is deleted only when the user
names it.
"""

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from revien.forget import gate, scrub
from revien.forget.keys import fingerprints
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import StoreBusy
from revien.graph.units import (
    describe, identity_keys, is_file_backed, is_real_file, is_shared_channel,
    is_unproven)

TOMBSTONE_PREFIX = "tombstone:"
VERB = "forget"
WHOLE_CONVERSATION = ("To remove a whole conversation or file, use delete; to "
                      "keep a specific detail out, forget that detail.")
NO_IDENTITY_NOTE = ("this capture came through a shared channel, which gives "
                    "it no identity of its own, so nothing is kept out of a "
                    "later capture of it; use --everywhere to keep this "
                    "wording out of every source")

# Labels inside the forgotten sentence are cut from the unit's conversation
# too, but only when long enough to be identifying wording.
_MIN_LABEL_CUT = 4


def is_whole_record(node: Node) -> bool:
    """A conversation's or a file's stored text as a whole: the context node,
    or a skill. Not one thing to forget."""
    md = node.metadata or {}
    if md.get("_tombstone") or md.get("origin") == "engine":
        return False
    return node.node_type in (NodeType.CONTEXT, NodeType.SKILL)


def is_tombstone(node_id: str) -> bool:
    return node_id.startswith(TOMBSTONE_PREFIX)


def is_holder(node: Node) -> bool:
    """A node that holds a unit's words verbatim: the session's context node,
    a skill, or a node an importer marked as the verbatim message."""
    md = node.metadata or {}
    if md.get("_tombstone"):
        return False
    if node.node_type == NodeType.SKILL and md.get("origin") == "engine":
        return False            # a proposal is the engine's summary, not a file
    return node.node_type in (NodeType.CONTEXT, NodeType.SKILL) or bool(md.get("verbatim"))


def _word_in(needle: str, haystack: str) -> bool:
    return bool(re.search(
        r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack, re.IGNORECASE))


def stated(node: Node) -> List[str]:
    """The exact wording of a node: what would be said in a conversation. An
    entity or a topic is its label; the content string the extractor gives it
    ('Entity: Brigitte Vandermeer') is Revien's own, never said by anyone."""
    if node.node_type in (NodeType.ENTITY, NodeType.TOPIC):
        return [node.label]
    return [node.content]


def node_wording(node: Node) -> List[str]:
    """Every string a node's wording is looked for by: the exact wording, and
    the label of a node whose label is a shorter form of it."""
    out = stated(node)
    if node.node_type not in (NodeType.ENTITY, NodeType.TOPIC):
        out.append(node.label)
    return out


@dataclass
class Plan:
    target: Optional[Node] = None
    units: List[str] = field(default_factory=list)          # forgotten units
    remove: List[Node] = field(default_factory=list)
    cut_back: List[Node] = field(default_factory=list)
    keep_vouched: List[Dict] = field(default_factory=list)  # {node, by}
    unproven: List[Node] = field(default_factory=list)      # kept, listed, untouched
    legacy_cut: List[Node] = field(default_factory=list)    # pre-provenance shared nodes, cut back (§5)
    everywhere: bool = False
    # forget --everywhere: the EXACT wording that was targeted (the target
    # node's own content, never a label, never a whole session's sentences),
    # and the nodes of other sources it is cut out of, in place.
    everywhere_strings: List[str] = field(default_factory=list)
    everywhere_cut: List[Node] = field(default_factory=list)
    co_owners: List[str] = field(default_factory=list)      # --everywhere: several sources wrote the target; none is suppressed
    likely_unit: bool = False                               # unit known from the node's source key, ownership unproven
    holders: List[Node] = field(default_factory=list)       # wording cut out
    strings: List[str] = field(default_factory=list)
    # Everything the forgotten text said, whole: where the words that only it
    # had are taken from (see scrub.distinctive_tokens).
    token_sources: List[str] = field(default_factory=list)
    refusal: Optional[Dict] = None


class ForgetEngine:
    #: How long a forget keeps waiting for a capture that holds the write lock
    #: (after SQLite's own timeout), before it refuses cleanly.
    BUSY_WAIT_SECONDS = 25.0

    def __init__(self, store, semantic=None, training_dbs: Iterable[str] = (),
                 clustering=None, retrain=None, retrain_reason=None):
        self.store = store
        self.semantic = semantic
        self.training_dbs = [str(p) for p in training_dbs]
        self.clustering = clustering
        #: Retrains the ranking model from the (scrubbed) retrieval signals;
        #: set only when neural ranking is on. Returns True when it ran.
        self.retrain = retrain
        #: Says why the last retrain produced no model.
        self.retrain_reason = retrain_reason

    # ── planning (read only) ──────────────────────────────

    def _members(self, unit_id: str) -> Dict[str, Dict]:
        return {m["node_id"]: m for m in self.store.nodes_of_unit(unit_id)}

    def _classify(self, rows: List[Dict], forgotten: Set[str]):
        """What happens to a node that the forgotten units wrote:
        'unproven' (kept), 'keep' (another live unit vouched in full),
        'cut_back' (another live unit vouched for the name only), 'remove'."""
        if not rows or any(is_unproven(r["unit_id"]) for r in rows):
            return "unproven", []
        live = [r for r in rows if r["unit_id"] not in forgotten]
        full = [r["unit_id"] for r in live if r["label_ok"] and r["wording_ok"]]
        if full:
            return "keep", full
        name = [r["unit_id"] for r in live if r["label_ok"]]
        if name:
            return "cut_back", name
        return "remove", []

    def _carries(self, node: Node, primary: List[str]) -> bool:
        """True if the node's own words contain the forgotten wording, or its
        label is a whole word of it. Whole words only: 'API' is not in
        'rapid'."""
        if scrub.contains(f"{node.label}\n{node.content}", primary):
            return True
        label = (node.label or "").strip()
        if len(label) >= _MIN_LABEL_CUT:
            return any(_word_in(label, w) for w in primary)
        return False

    def _likely_unit(self, node: Node) -> Optional[str]:
        from revien.graph.units import node_likely_unit
        # the unit its unproven marker names was recorded with the node (a
        # graph import records the id inside the session file there)
        for r in self.store.get_provenance(node.node_id):
            if r["unit_id"].startswith("unproven:") and len(r["unit_id"]) > 9:
                return r["unit_id"][len("unproven:"):]
        md = node.metadata if isinstance(node.metadata, dict) else {}
        return node_likely_unit(node.origin_runtime, node.session_key,
                                node.source_id, md.get("ingest_key"))

    def _nodes_containing(self, primary: List[str]) -> List[Node]:
        out: Dict[str, Node] = {}
        conn = self.store._get_conn()
        with self.store._lock:
            # SQLite's own lower() folds ASCII only ('Élodie' stays 'Élodie'),
            # so the prefilter lowers in Python, as the match itself does.
            conn.create_function("ulower", 1, lambda t: (t or "").lower(),
                                 deterministic=True)
            for w in primary:
                for (nid,) in conn.execute(
                        "SELECT node_id FROM nodes WHERE instr(ulower(label || "
                        "char(10) || content), ?) > 0", (w.lower(),)):
                    if nid in out or is_tombstone(nid):
                        continue
                    n = self.store.get_node(nid)
                    if n is not None and scrub.contains(
                            f"{n.label}\n{n.content}", primary):
                        out[nid] = n
        return list(out.values())

    def plan(self, node_id: str, everywhere: bool = False) -> Plan:
        store = self.store
        plan = Plan(everywhere=everywhere)
        target = store.get_node(node_id)
        plan.target = target
        if target is None:
            return plan
        if is_whole_record(target):
            # A conversation (or a skill file) is not one thing to forget.
            plan.refusal = {"reason": "whole_conversation",
                            "message": WHOLE_CONVERSATION}
            return plan
        rows = store.get_provenance(node_id)
        proven = [r for r in rows if not is_unproven(r["unit_id"])]
        forgotten = [r["unit_id"] for r in proven if r["wording_ok"]]
        plan.units = sorted(set(forgotten))
        primary = scrub.clean_strings(node_wording(target))
        if everywhere and len(plan.units) > 1:
            # Several sources wrote it: there is no single origin to bind the
            # wording to, so it is kept out of every source instead.
            plan.co_owners = plan.units
            plan.units = []
        if not plan.units and not plan.co_owners:
            # Ownership of this node is unproven, but the source it came from
            # may still be known (a session id, a real path in its source
            # key). Bind to that identity whenever it is known.
            likely = self._likely_unit(target)
            if likely:
                plan.units = [likely]
                plan.likely_unit = True
        if everywhere:
            # Only the wording the user named: this node's own words.
            plan.everywhere_strings = scrub.clean_strings(stated(target))
        fset = set(plan.units)

        remove: Dict[str, Node] = {target.node_id: target}
        cut_back: Dict[str, Node] = {}
        keep: List[Dict] = []
        unproven: Dict[str, Node] = {}

        # The target itself. (With --everywhere it is the node the user
        # named, and it goes whatever else vouched for it.)
        verdict, by = ("remove", []) if everywhere else self._classify(rows, fset)
        if verdict == "keep":
            keep.append({"node": target, "by": by})
            del remove[target.node_id]
        elif verdict == "cut_back":
            cut_back[target.node_id] = target
            del remove[target.node_id]

        for unit_id in plan.units:
            members = self._members(unit_id)
            for nid in members:
                if nid == target.node_id:
                    continue
                n = store.get_node(nid)
                if n is None or is_tombstone(nid):
                    continue
                # The unit's stored conversation is cut, not removed; any
                # other node is removed only if it carries the wording.
                if is_holder(n) or not self._carries(n, primary):
                    continue
                v, by = self._classify(store.get_provenance(nid), fset)
                if v == "unproven":
                    unproven[nid] = n
                elif v == "keep":
                    keep.append({"node": n, "by": by})
                elif v == "cut_back":
                    if plan.everywhere_strings:
                        # --everywhere never cuts a node back to its label:
                        # the exact sentence is cut out of it in place below
                        continue
                    cut_back[nid] = n
                else:
                    remove[nid] = n

        # Ownership that could not be proven: kept and LISTED (spec §3), never
        # edited. The one exception is §5: a description saved before
        # provenance existed, on a non-context node whose likely unit is being
        # forgotten, is treated as unproven wording and cut back (logged). A
        # context node, a hand-added node or another unit's history is never
        # touched on a guess.
        legacy_cut: Dict[str, Node] = {}
        for unit_id in plan.units:
            for nid in self._unproven_nodes_for(unit_id):
                if nid in remove or nid in cut_back or nid == target.node_id:
                    continue
                n = store.get_node(nid)
                if n is None:
                    continue
                if is_holder(n):
                    unproven[nid] = n
                elif not plan.everywhere_strings:
                    legacy_cut[nid] = n
        for n in self._legacy_carriers(primary):
            if n.node_id not in remove and n.node_id != target.node_id \
                    and n.node_id not in cut_back and n.node_id not in legacy_cut:
                unproven[n.node_id] = n
        # Every string that gets cut: the target's wording and the wording of
        # what is removed or cut back.
        strings: List[str] = list(primary)
        for n in list(remove.values()) + list(cut_back.values()):
            if n is not target and n.node_type not in (
                    NodeType.ENTITY, NodeType.TOPIC):
                strings.append(n.content)
        plan.strings = scrub.clean_strings(strings)
        plan.token_sources = [target.label, target.content] + [
            t for n in list(remove.values()) + list(cut_back.values())
            for t in (n.label, n.content)]

        # Where the wording is cut: the forgotten units' stored conversation.
        holders: Dict[str, Node] = {}
        had_holder = False
        for unit_id in plan.units:
            for nid in self._members(unit_id):
                if nid in remove or nid == target.node_id:
                    continue
                n = store.get_node(nid)
                if n is None or not is_holder(n):
                    continue
                had_holder = True
                if scrub.contains(n.content, plan.strings) or \
                        scrub.contains(n.label, plan.strings):
                    holders[nid] = n

        # forget --everywhere: the targeted wording, cut in place from every
        # other node that holds it. Never a deletion, never a cut-back to the
        # label, never a refusal of the source it sits in.
        everywhere_cut: Dict[str, Node] = {}
        if plan.everywhere_strings:
            handled = (set(remove) | set(cut_back) | set(holders)
                       | set(legacy_cut) | {target.node_id})
            for n in self._nodes_containing(plan.everywhere_strings):
                if n.node_id in handled or is_tombstone(n.node_id):
                    continue
                everywhere_cut[n.node_id] = n
                unproven.pop(n.node_id, None)

        # The wording must be found verbatim where the unit stored it.
        if (plan.units and had_holder and not is_holder(target)
                and not any(scrub.contains(h.content, primary)
                            or scrub.contains(h.label, primary)
                            for h in holders.values())):
            first = None
            for unit_id in plan.units:
                for nid in self._members(unit_id):
                    n = store.get_node(nid)
                    if n is not None and is_holder(n):
                        first = n
                        break
                if first:
                    break
            plan.refusal = {
                "reason": "wording_not_found",
                "context_node_id": first.node_id if first else None,
                "message": (
                    "Refused: the forgotten wording could not be found word "
                    f"for word in the stored conversation (context node "
                    f"{first.node_id if first else '?'}), so it cannot be "
                    "cut out - the model may have paraphrased it. Nothing "
                    "was changed. " + WHOLE_CONVERSATION),
            }

        plan.remove = list(remove.values())
        plan.cut_back = list(cut_back.values())
        plan.keep_vouched = keep
        plan.unproven = [n for n in unproven.values() if n.node_id not in holders]
        plan.legacy_cut = list(legacy_cut.values())
        plan.holders = list(holders.values())
        plan.everywhere_cut = list(everywhere_cut.values())
        return plan

    def _unproven_nodes_for(self, unit_id: str) -> List[str]:
        conn = self.store._get_conn()
        with self.store._lock:
            return [r[0] for r in conn.execute(
                "SELECT node_id FROM node_provenance WHERE unit_id = ?",
                ("unproven:" + unit_id,))]

    def _legacy_carriers(self, primary: List[str]) -> List[Node]:
        """Nodes with no proven owner whose own words contain the forgotten
        wording (whole words). They are LISTED, not edited: their owner is
        unknown, and they may be another session's own words."""
        out: Dict[str, Node] = {}
        conn = self.store._get_conn()
        with self.store._lock:
            rows = conn.execute(
                "SELECT n.node_id FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM "
                "node_provenance p WHERE p.node_id = n.node_id AND "
                "substr(p.unit_id, 1, 9) != 'unproven:')").fetchall()
        for (nid,) in rows:
            if is_tombstone(nid):
                continue
            n = self.store.get_node(nid)
            if n is None:
                continue
            if scrub.contains(f"{n.label}\n{n.content}", primary):
                out[nid] = n
                continue
            # a node whose own words (12+ characters) are part of the
            # forgotten wording: listed, so the user can see it holds it
            for text in (n.content, n.label):
                t = (text or "").strip()
                if len(t) >= 12 and any(scrub.contains(w, [t]) for w in primary):
                    out[nid] = n
                    break
        return list(out.values())

    # ── the verb ──────────────────────────────────────────

    def forget(self, node_id: str, actor: str = "forget",
               dry_run: bool = False, everywhere: bool = False) -> Dict:
        """Forget in two phases so the order of a racing capture cannot
        matter. Phase 1 commits the standing denylist (the forget record:
        keyed sentence fingerprints and the identity of the origin) under
        BEGIN IMMEDIATE. Phase 2 plans the
        removal AFTER that commit, so it sees anything a capture wrote before
        the denylist existed; a capture that writes after it is stopped by the
        denylist (gate.enforce_after_write)."""
        with self.store._capture_lock:
            if is_tombstone(node_id):
                return {"status": "skipped", "verb": VERB, "node_id": node_id,
                        "forgotten": [], "count": 0, "reason": "is_tombstone"}
            plan = self.plan(node_id, everywhere)
            if plan.target is None:
                return {"status": "not_found", "verb": VERB,
                        "node_id": node_id, "forgotten": [], "count": 0}
            if plan.refusal is not None:
                return {"status": "refused", "verb": VERB, "node_id": node_id,
                        "forgotten": [], "count": 0, **plan.refusal}
            if dry_run:
                return self._summary(plan, preview=True)
            records, bound, not_bound = self._denylist_records(plan)
            try:
                fallback = self._record_denylist(records, actor)
            except StoreBusy:
                return self._busy(node_id, committed=False)
            plan = self.plan(node_id, everywhere)
            if plan.target is None:       # removed by a racing forget
                return {"status": "not_found", "verb": VERB,
                        "node_id": node_id, "forgotten": [], "count": 0}
            try:
                return self._execute(plan, actor, bound, not_bound, fallback)
            except StoreBusy:
                return self._busy(node_id, committed=True)

    @staticmethod
    def _busy(node_id: str, committed: bool) -> Dict:
        if committed:
            message = ("Revien is busy writing, so the removal did not run. "
                       "The forgotten wording is already kept out of new "
                       "captures, but nothing has been removed yet. Run the "
                       "same forget again in a moment.")
        else:
            message = ("Revien is busy writing; nothing was changed. Try "
                       "again in a moment.")
        return {"status": "busy", "verb": VERB, "node_id": node_id,
                "forgotten": [], "count": 0, "reason": "database_busy",
                "denylist_written": committed, "message": message}

    # ── execution ─────────────────────────────────────────

    def _manifest_notes(self, ids: Iterable[str]) -> List[str]:
        ids = list(ids)
        if not ids:
            return []
        conn = self.store._get_conn()
        marks = ",".join("?" * len(ids))
        with self.store._lock:
            rows = conn.execute(
                "SELECT DISTINCT note_stem FROM distill_manifest WHERE "
                f"anchor_node_id IN ({marks}) OR current_node_id IN ({marks})",
                ids + ids).fetchall()
        return sorted(r[0] for r in rows)

    def _forgotten_sentences(self, text: str, plan: Plan, whole: bool) -> List[str]:
        out = []
        for _l, _r, body in scrub.iter_sentences(text):
            key = scrub.sentence_key(body)
            if len(key) < scrub.MIN_SHORT:
                continue
            if whole or scrub.contains(key, plan.strings) or any(
                    key.casefold() in w.casefold() for w in plan.strings):
                out.append(key)
        return out

    def _holder_texts(self, unit_id: str) -> List[str]:
        out = []
        for nid in self._members(unit_id):
            n = self.store.get_node(nid)
            if n is not None and is_holder(n):
                out.append(n.content)
        return out

    def _forgotten_keys(self, plan: Plan) -> List[str]:
        """Every sentence this forget takes out (before anything is cut): the
        sentences of what is removed, and the sentences that hold the wording
        in the stored conversation."""
        gone = plan.remove + plan.cut_back + plan.legacy_cut
        keys: List[str] = []
        for n in gone + [plan.target]:
            for text in self._said(n):
                keys += self._forgotten_sentences(text, plan, True)
        for h in plan.holders:
            keys += self._forgotten_sentences(h.content, plan, False)
        for unit_id in plan.units:
            for t in self._holder_texts(unit_id):
                keys += self._forgotten_sentences(t, plan, False)
        return list(dict.fromkeys(keys))

    @staticmethod
    def _said(node: Node) -> List[str]:
        """The text of a node that someone could have said: nothing for an
        entity or a topic, whose content is a string Revien made up."""
        if node.node_type in (NodeType.ENTITY, NodeType.TOPIC):
            return [node.label]
        return [node.content]

    @staticmethod
    def _split_fps(keys: Iterable[str]):
        long_fps = fingerprints(k for k in keys if len(k) >= scrub.MIN_SENTENCE)
        short_fps = fingerprints(k for k in keys if len(k) < scrub.MIN_SENTENCE)
        return long_fps, short_fps

    def _denylist_records(self, plan: Plan):
        """(the forget records to write, the sources they bind to, the
        sources they cannot bind to), built BEFORE anything is cut so the
        fingerprints are of the text as it was.

        The origin record keeps the forgotten sentences out of captures that
        come from the origin the node was forgotten from: it carries the unit
        ids and file identities of that origin, and nothing else binds to it.
        A source with no identity of its own (a shared channel) cannot be
        bound. The global record exists only for ``forget --everywhere`` and
        holds only the sentences of the exact wording that was targeted."""
        store = self.store
        gone = plan.remove + plan.cut_back + plan.legacy_cut
        nodes_text = [t for n in gone + [plan.target] for t in self._said(n)]
        wording = list(nodes_text)
        if plan.target.node_type not in (NodeType.ENTITY, NodeType.TOPIC):
            wording += [n.label for n in gone + [plan.target]]
        keys = self._forgotten_keys(plan)
        bound: List[Dict] = []
        not_bound: List[Dict] = []
        covered: List[str] = []
        for unit_id in plan.units:
            if is_shared_channel(unit_id):
                not_bound.append({"unit_id": unit_id, "note": NO_IDENTITY_NOTE})
                continue
            info = store.get_unit(unit_id) or {}
            path = info.get("path")
            pkeys = list(info.get("path_keys") or [])
            if path and is_real_file(path):
                pkeys = sorted(set(pkeys) | set(identity_keys(path)))
            covered += [f"unit:{unit_id}"] + pkeys
            bound.append({"unit_id": unit_id})
        records: List[Dict] = []
        if covered:
            long_fps, short_fps = self._split_fps(keys)
            records.append({
                "unit_id": f"forget:{uuid.uuid4()}", "scope": "origin",
                "path_keys": sorted(set(covered)),
                "wording_fps": fingerprints(wording),
                # (only the wording itself: a fact's extractor label is a
                # fragment of it, not something the user named)
                "window_fps": gate.window_fps_for(
                    scrub.clean_strings(stated(plan.target))),
                "sentence_fps": long_fps, "short_fps": short_fps})
        if plan.everywhere_strings:
            # (the wording is matched in any sentence by its windows, so no
            # sentence is listed: nothing of the text is kept)
            records.append({
                "unit_id": f"forget:{uuid.uuid4()}", "scope": "global",
                "path_keys": [],
                "wording_fps": fingerprints(plan.everywhere_strings),
                "window_fps": gate.window_fps_for(plan.everywhere_strings),
                "sentence_fps": [], "short_fps": []})
        return records, bound, not_bound

    def _record_denylist(self, records, actor):
        """Phase 1: commit the forget records (the standing denylist) under
        BEGIN IMMEDIATE, visible to every other connection at once."""
        store = self.store
        now = datetime.now(timezone.utc)
        short_total = sum(len(r["short_fps"]) for r in records)
        with store.transaction(immediate=True, wait=self.BUSY_WAIT_SECONDS):
            for r in records:
                sid = store.add_suppression(
                    r["unit_id"], r["path_keys"], r["wording_fps"],
                    r["sentence_fps"], r["short_fps"], actor, scope=r["scope"],
                    window_fps=r["window_fps"])
                store.record_audit(f"suppression:{sid}", "suppress", actor=actor,
                                   after={"scope": r["scope"], "verb": VERB},
                                   ts=now)
        if not short_total:
            return None
        return {
            "fragments": short_total,
            "note": (f"{short_total} forgotten fragment(s) are shorter than "
                     f"{scrub.MIN_SENTENCE} characters. A short text match "
                     "would hit unrelated sentences, so they are kept out "
                     "of the source they came from only, by that source's "
                     "identity and their keyed fingerprint, never matched on "
                     "their own in another source.")}

    def _execute(self, plan: Plan, actor: str, bound, not_bound,
                 fallback) -> Dict:
        store = self.store
        target = plan.target
        now = datetime.now(timezone.utc)
        strings = plan.strings
        remove_ids = [n.node_id for n in plan.remove]
        cut_ids = [n.node_id for n in plan.cut_back]
        legacy_ids = [n.node_id for n in plan.legacy_cut]
        holder_ids = [n.node_id for n in plan.holders]
        keep_ids = [k["node"].node_id for k in plan.keep_vouched]
        place_ids = [n.node_id for n in plan.everywhere_cut]
        forgotten_units = set(plan.units)
        changed_ids = list(dict.fromkeys(cut_ids + legacy_ids + holder_ids))
        # what this forget takes out, read before anything is cut
        forgotten_keys = self._forgotten_keys(plan)
        identifying = self._identifying_tokens(plan, forgotten_keys)
        strict_tokens = identifying[0]

        before = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        manifest_notes = self._manifest_notes(remove_ids + cut_ids + legacy_ids)
        stats = {
            "audit_rows_scrubbed": 0, "judgments_removed": 0,
            "judgments_queries_cleared": 0, "receipts_cleared": 0,
            "candidates_removed": 0, "candidates_scrubbed": 0,
            "manifest_rows_removed": 0, "pending_embeds_removed": 0,
            "wording_cuts": 0,
        }
        wording_cut: List[Dict] = []
        cut_in_place: List[Dict] = []
        placed_labels: List[str] = []
        tombstones = 0
        removed_done: List[str] = []

        with store.transaction(immediate=True, wait=self.BUSY_WAIT_SECONDS) as conn:
            # What the recall surfaces remember about these nodes, read
            # BEFORE the nodes go (deleting a node strips its id).
            affected = set(remove_ids + cut_ids + changed_ids + place_ids)
            episodes = self._episodes_for(conn, affected)
            receipt_ids = self._receipts_for(conn, affected)

            # Lineage: surviving children of a removed node point at a
            # content-free marker, as before.
            tombstones += self._repoint_children(remove_ids, now)

            # Cut the wording out of the stored conversation.
            for nid in holder_ids:
                info = self._cut_node(conn, nid, strings, cut_back=False)
                if info["cuts"]:
                    stats["wording_cuts"] += info["cuts"]
                    wording_cut.append({"context_node_id": nid,
                                        "occurrences": info["cuts"]})
                    if plan.everywhere:
                        cut_in_place.append({
                            "node_id": nid, "occurrences": info["cuts"],
                            "units": self._owners_of(nid), "origin": True})
                    store.record_audit(nid, "forget_cut", actor=actor,
                                       after={"verb": VERB}, ts=now)
            # forget --everywhere: the exact targeted wording, cut in place
            # out of the nodes of other sources. The sentence goes; the rest
            # of the node stays, and the node is never deleted, cut back to
            # its label or suppressed.
            for nid in place_ids:
                info = self._cut_node(conn, nid, plan.everywhere_strings,
                                      cut_back=False, keep_node=True)
                if info.get("replaced_label"):
                    placed_labels.append(info["replaced_label"])
                if info["cuts"]:
                    stats["wording_cuts"] += info["cuts"]
                    cut_in_place.append({
                        "node_id": nid, "occurrences": info["cuts"],
                        "units": self._owners_of(nid)})
                    store.record_audit(nid, "forget_cut", actor=actor,
                                       after={"verb": VERB, "everywhere": True},
                                       ts=now)
            # Shared nodes whose wording only the forgotten unit supplied, and
            # (spec §5) descriptions saved before provenance existed.
            for nid in cut_ids:
                # (its name is what another unit vouched for: it stays)
                label = next((n.label for n in plan.cut_back if n.node_id == nid), "")
                own = [x for x in strings
                       if x.casefold() != (label or "").strip().casefold()]
                self._cut_node(conn, nid, own, cut_back=True)
                store.delete_provenance(nid, forgotten_units)
                store.record_audit(nid, "cut_back", actor=actor,
                                   after={"verb": VERB}, ts=now)
            for nid in legacy_ids:
                if self._cut_node(conn, nid, strings, cut_back=True)["changed"]:
                    store.record_audit(nid, "cut_back", actor=actor,
                                       after={"verb": VERB, "legacy": True}, ts=now)
            for nid in keep_ids:
                store.delete_provenance(nid, forgotten_units)

            # The forgotten nodes themselves.
            for nid in remove_ids:
                store.record_audit(
                    nid, VERB, actor=actor, before=None,
                    after={"_tombstone": True, "original_node_id": nid,
                           "deleted_at": now.isoformat(), "verb": VERB},
                    ts=now)
                if store.delete_node(nid):
                    removed_done.append(nid)

            # Everything else that quotes them. The words only the forgotten
            # text had identify an echo of it (a query typed before the
            # forget, a candidate's reason). Whole words only.
            echo = scrub.distinctive_tokens(conn, plan.token_sources,
                                            set(holder_ids + cut_ids))
            all_strings = scrub.clean_strings(list(strings) + echo)
            # The side fields of the nodes that were cut (when it was said,
            # where it came from, caller metadata) quote single words, not
            # whole sentences, so they are cut with the words only the
            # forgotten text had.
            for nid in holder_ids + cut_ids:
                self._cut_aux(conn, nid, all_strings)
            stats["audit_rows_scrubbed"] = self._scrub_audit(
                conn, set(remove_ids), set(changed_ids), all_strings)
            if place_ids:
                # Other sources' history is theirs: only the exact wording
                # that was targeted is cut from it.
                stats["audit_rows_scrubbed"] += self._scrub_audit(
                    conn, set(), set(place_ids),
                    scrub.clean_strings(plan.everywhere_strings + placed_labels))
            q = self._scrub_queries(conn, episodes, receipt_ids, all_strings)
            stats.update(q)
            stats["candidates_scrubbed"] = self._scrub_candidates(conn, all_strings)
        strings = all_strings
        # ── committed ──

        gone = removed_done
        problems = self._purge_vectors(gone + changed_ids + place_ids)
        training = self._scrub_training(
            gone + cut_ids + legacy_ids + place_ids + holder_ids,
            list(strings) + sorted(strict_tokens))
        model = self._model_after_forget(now)
        snapshots = self._scrub_snapshots(
            remove_ids, set(changed_ids), strings, episodes, receipt_ids,
            place=(set(place_ids), plan.everywhere_strings, placed_labels))
        if self.clustering is not None:
            # The detector keeps community labels in memory. A graph too small
            # to cluster returns before it rebuilds, so drop them first.
            try:
                self.clustering._communities.clear()
                self.clustering._node_to_community.clear()
                self.clustering.run()
            except Exception:  # noqa: BLE001 - never fail a forget on this
                pass
        storage = store.scrub_storage()
        after = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        # What is still in the database: the cut strings, and every sentence
        # this forget took out, whole words only. Revien's own actor and op
        # strings are not content.
        witnesses = scrub.clean_strings(
            list(strings) + sorted(strict_tokens)
            + [k for k in forgotten_keys if len(k) >= scrub.MIN_SENTENCE])
        with store._lock:
            residue = scrub.find_residue(store._get_conn(), witnesses,
                                         ignore=scrub.NOT_CONTENT_TABLES)
        held = self._held_elsewhere(
            plan, forgotten_keys, identifying,
            touched=cut_ids + legacy_ids + keep_ids + place_ids)

        return self._finish(
            plan, before, after, tombstones, removed_done, stats, wording_cut,
            bound, not_bound, legacy_ids, manifest_notes,
            problems, training, storage, residue, snapshots, held, fallback,
            cut_in_place, model)

    def _owners_of(self, node_id: str) -> List[str]:
        rows = self.store.get_provenance(node_id)
        units = sorted({r["unit_id"] for r in rows if not is_unproven(r["unit_id"])})
        if units:
            return units
        return ["(ownership unproven)" if rows else "(no recorded source)"]

    # ── execution helpers ─────────────────────────────────

    def _episodes_for(self, conn, ids: Set[str]) -> Set[str]:
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        return {r[0] for r in conn.execute(
            f"SELECT DISTINCT episode_key FROM use_judgments WHERE node_id IN ({marks})",
            list(ids))}

    def _receipts_for(self, conn, ids: Set[str]) -> Set[str]:
        out = set()
        for recall_id, raw in conn.execute(
                "SELECT recall_id, node_ids FROM recall_receipts"):
            try:
                listed = set(json.loads(raw or "[]"))
            except ValueError:
                continue
            if listed & ids:
                out.add(recall_id)
        return out

    _MERGE_OPS = ("normalized_merge", "semantic_merge")

    def _scrub_audit(self, conn, removed: Set[str], touched: Set[str],
                     strings: List[str]) -> int:
        """Audit history. The snapshots of a removed node are dropped; the
        snapshots of a node the forget CUT are cut the same way. Every other
        node's history is its owner's and is left alone, except the actor of a
        merge entry, which records the merged candidate's label (and so may be
        the forgotten wording)."""
        n = 0
        rows = conn.execute(
            "SELECT id, node_id, op, actor, before_json, after_json "
            "FROM audit_log").fetchall()
        for rid, node_id, op, actor, bj, aj in rows:
            if node_id in removed and op != VERB:
                if bj is not None or aj is not None or actor:
                    conn.execute(
                        "UPDATE audit_log SET before_json = NULL, "
                        "after_json = NULL, actor = '' WHERE id = ?", (rid,))
                    n += 1
                continue
            if node_id in touched:
                nb, cb = scrub.cut_json_text(bj, strings) if bj else (bj, 0)
                na, ca = scrub.cut_json_text(aj, strings) if aj else (aj, 0)
            else:
                nb, cb, na, ca = bj, 0, aj, 0
            if node_id in touched or op in self._MERGE_OPS:
                nact, cact = scrub.cut_text(actor or "", strings)
            else:
                nact, cact = actor, 0
            if cb or ca or cact:
                conn.execute(
                    "UPDATE audit_log SET before_json = ?, after_json = ?, "
                    "actor = ? WHERE id = ?", (nb, na, nact, rid))
                n += 1
        return n

    def _repoint_children(self, remove_ids: List[str], now: datetime) -> int:
        """Children of a removed node keep a lineage edge to a content-free
        marker, not to a deleted row. A child that is itself removed has no
        lineage to keep."""
        store = self.store
        doomed = set(remove_ids)
        made = 0
        for nid in remove_ids:
            kids = []
            for e in store.get_edges_for_node(nid):
                if (e.edge_type == EdgeType.DERIVED_FROM
                        and e.target_node_id == nid
                        and e.source_node_id not in doomed):
                    kids.append(e)
            if not kids:
                continue
            tid = f"{TOMBSTONE_PREFIX}{nid}"
            if store.get_node(tid) is None:
                store.add_node(Node(
                    node_id=tid, node_type=NodeType.CONTEXT,
                    label="[forgotten node]", content="", pinned=True,
                    source_type=SourceType.DERIVED, confidence=0.0,
                    source_context="tombstone",
                    metadata={"_tombstone": True, "_original_node_id": nid,
                              "_deleted_at": now.isoformat(), "_verb": VERB}))
                made += 1
            for e in kids:
                store.add_edge(Edge(
                    edge_type=EdgeType.DERIVED_FROM,
                    source_node_id=e.source_node_id, target_node_id=tid,
                    weight=e.weight,
                    source_context="lineage_repointed_to_tombstone"))
                store.delete_edge(e.edge_id)
        return made

    def _cut_node(self, conn, node_id: str, strings: List[str],
                  cut_back: bool, keep_node: bool = False) -> Dict:
        """Cut the wording out of one node's stored text, with raw SQL so no
        audit snapshot of the old text is written. ``cut_back`` also reduces
        its description to its label (and clears an engine proposal's steps).
        Queues a re-embed; the old vector is purged after the commit."""
        row = conn.execute(
            "SELECT label, content, event_time_text, source_context, metadata, "
            "node_type FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return {"cuts": 0, "changed": False}
        label, content, ett, sctx, meta_raw, ntype = row
        cuts = 0
        new_label, c = scrub.cut_text(label, strings); cuts += c
        new_content, c = scrub.cut_text(content, strings); cuts += c
        new_ett, c = scrub.cut_text(ett or "", strings); cuts += c
        new_sctx, c = scrub.cut_text(sctx or "", strings); cuts += c
        new_meta, c = scrub.cut_json_text(meta_raw or "{}", strings); cuts += c
        try:
            meta = json.loads(new_meta) if new_meta else {}
        except ValueError:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if cut_back:
            new_content = new_label if new_label.strip() else "[removed]"
            if not new_label.strip():
                new_label = "[removed]"
            meta["cut_back_at"] = datetime.now(timezone.utc).isoformat()
            if ntype == "skill" and meta.get("origin") == "engine":
                for key in ("steps", "draft"):
                    if key in meta:
                        meta[key] = [] if isinstance(meta[key], list) else ""
        elif not cuts:
            return {"cuts": 0, "changed": False}
        if not new_label.strip():
            new_label = "[removed]"
        replaced_label = None
        if keep_node:
            lab = new_label.strip()
            if len(lab) >= scrub.MIN_SHORT and any(
                    lab.casefold() in w.casefold() for w in strings):
                replaced_label = new_label
                new_label = "[removed]"      # the label was a piece of the cut wording
        if keep_node and not new_content.strip():
            # the node was only the cut sentence: it stays, as a marker
            new_content = "[removed]"
        conn.execute(
            "UPDATE nodes SET label = ?, content = ?, event_time_text = ?, "
            "source_context = ?, metadata = ? WHERE node_id = ?",
            (new_label[:200], new_content, new_ett, new_sctx,
             json.dumps(meta), node_id))
        self.store._enqueue_embed_stmt(conn, node_id)
        return {"cuts": cuts, "changed": True, "replaced_label": replaced_label}

    def _cut_aux(self, conn, node_id: str, strings: List[str]) -> None:
        """Cut event_time_text, source_context and metadata only."""
        row = conn.execute(
            "SELECT event_time_text, source_context, metadata FROM nodes "
            "WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return
        ett, sctx, meta = row
        n_ett, a = scrub.cut_text(ett or "", strings)
        n_sctx, b = scrub.cut_text(sctx or "", strings)
        n_meta, c = scrub.cut_json_text(meta or "{}", strings)
        if a or b or c:
            conn.execute(
                "UPDATE nodes SET event_time_text = ?, source_context = ?, "
                "metadata = ? WHERE node_id = ?",
                (n_ett, n_sctx, n_meta, node_id))

    def _scrub_queries(self, conn, episodes, receipt_ids, strings) -> Dict:
        out = {"judgments_removed": 0, "judgments_queries_cleared": 0,
               "receipts_cleared": 0}
        pat_hits = lambda text: scrub.contains(text or "", strings)  # noqa: E731
        for rid, ep, query, terms in conn.execute(
                "SELECT id, episode_key, query, shared_terms FROM use_judgments"
        ).fetchall():
            nt, ct = scrub.cut_json_text(terms or "[]", strings)
            if ep in episodes or pat_hits(query):
                conn.execute(
                    "UPDATE use_judgments SET query = '', shared_terms = ? "
                    "WHERE id = ?", (nt, rid))
                out["judgments_queries_cleared"] += 1
            elif ct:
                conn.execute("UPDATE use_judgments SET shared_terms = ? "
                             "WHERE id = ?", (nt, rid))
        for recall_id, query in conn.execute(
                "SELECT recall_id, query FROM recall_receipts").fetchall():
            if recall_id in receipt_ids or pat_hits(query):
                if query:
                    conn.execute(
                        "UPDATE recall_receipts SET query = '' WHERE recall_id = ?",
                        (recall_id,))
                    out["receipts_cleared"] += 1
        return out

    def _scrub_candidates(self, conn, strings) -> int:
        n = 0
        for cid, reason, trace in conn.execute(
                "SELECT id, reason, trace FROM supersession_candidates").fetchall():
            nr, c1 = scrub.cut_text(reason or "", strings)
            nt, c2 = scrub.cut_json_text(trace or "", strings)
            if c1 or c2:
                conn.execute(
                    "UPDATE supersession_candidates SET reason = ?, trace = ? "
                    "WHERE id = ?", (nr, nt, cid))
                n += 1
        return n

    def _purge_vectors(self, ids: List[str]) -> Optional[str]:
        """Delete the embeddings of removed or edited nodes whether or not
        the semantic layer is switched on in this process. Returns a problem
        description, or None."""
        ids = list(dict.fromkeys(ids))
        store = self.store
        with store._lock:
            conn = store._get_conn()
            if conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'vec_nodes'"
            ).fetchone() is None:
                return None
            try:
                store._load_vec_extension(conn)
                conn.executemany("DELETE FROM vec_nodes WHERE node_id = ?",
                                 [(i,) for i in ids])
                conn.commit()
            except Exception as e:  # noqa: BLE001
                return f"vector rows could not be removed: {e}"
        return None

    def _scrub_training(self, ids: List[str], strings: List[str]) -> Dict:
        """Rows of training.db about these nodes, or quoting the wording."""
        out = {"checked": 0, "rows_removed": 0, "problems": []}
        for path in self.training_dbs:
            if not os.path.exists(path):
                continue
            out["checked"] += 1
            try:
                conn = sqlite3.connect(path)
                try:
                    conn.execute("PRAGMA secure_delete = ON")
                    n = 0
                    if ids:
                        marks = ",".join("?" * len(ids))
                        n += conn.execute(
                            f"DELETE FROM retrieval_signals WHERE node_id IN ({marks})",
                            ids).rowcount
                    for rid, query, label in conn.execute(
                            "SELECT id, query, node_label FROM retrieval_signals"
                    ).fetchall():
                        if scrub.contains(query, strings) or \
                                scrub.contains(label or "", strings):
                            n += conn.execute(
                                "DELETE FROM retrieval_signals WHERE id = ?",
                                (rid,)).rowcount
                    conn.commit()
                    conn.execute("VACUUM")
                    out["rows_removed"] += n
                finally:
                    conn.close()
            except sqlite3.Error as e:
                out["problems"].append(f"{path}: {e}")
        return out

    # ── reporting ─────────────────────────────────────────

    def _scrub_snapshots(self, removed_ids, touched_ids, strings, episodes,
                         receipt_ids=frozenset(), place=(frozenset(), (), ())) -> Dict:
        """`revien watch` backups are copies of the raw store, so the forget
        reaches them too, with the same treatment as the live database: the
        removed nodes go (and their vectors), the cut ones are cut, the
        recall queries and receipts of the same recalls are cleared, and the
        file is rewritten. A snapshot (or part of one) that cannot be opened
        is named, not skipped silently."""
        import gzip
        import shutil
        import tempfile

        out = {"scrubbed": 0, "unreadable": []}
        snap = Path(str(self.store.db_path) + ".snapshots")
        if not snap.is_dir():
            return out
        for f in sorted(snap.iterdir()):
            name = f.name
            if not (name.endswith(".db") or name.endswith(".db.gz")):
                continue
            try:
                if name.endswith(".db.gz"):
                    with tempfile.TemporaryDirectory() as tmp:
                        raw = Path(tmp) / "snap.db"
                        with gzip.open(f, "rb") as src, open(raw, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        partial = self._scrub_db_copy(
                            raw, removed_ids, touched_ids, strings, episodes,
                            receipt_ids, place)
                        packed = Path(tmp) / "snap.db.gz"
                        with open(raw, "rb") as src, gzip.open(packed, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        shutil.copyfile(packed, f)
                else:
                    partial = self._scrub_db_copy(
                        f, removed_ids, touched_ids, strings, episodes,
                        receipt_ids, place)
                out["scrubbed"] += 1
                for note in partial:
                    out["unreadable"].append(f"{name}: {note}")
            except Exception as e:  # noqa: BLE001
                out["unreadable"].append(f"{name}: {e}")
        return out

    def _scrub_db_copy(self, path, removed_ids, touched_ids, strings,
                       episodes=frozenset(), receipt_ids=frozenset(),
                       place=(frozenset(), (), ())) -> List[str]:
        """Scrub one snapshot file in place. Returns notes about anything in
        it that could not be scrubbed."""
        place_ids, place_strings, placed_labels = place
        audit_strings = scrub.clean_strings(list(place_strings) + list(placed_labels))
        notes: List[str] = []
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("PRAGMA secure_delete = ON")
            ids = list(removed_ids)
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                marks = ",".join("?" * len(chunk))
                for sql in (
                    f"DELETE FROM edges WHERE source_node_id IN ({marks}) OR target_node_id IN ({marks})",
                    f"DELETE FROM nodes WHERE node_id IN ({marks})",
                    f"DELETE FROM node_provenance WHERE node_id IN ({marks})",
                    f"DELETE FROM use_judgments WHERE node_id IN ({marks})",
                    f"DELETE FROM supersession_candidates WHERE existing_node_id IN ({marks}) OR new_node_id IN ({marks})",
                    f"DELETE FROM distill_manifest WHERE anchor_node_id IN ({marks}) OR current_node_id IN ({marks})",
                    f"DELETE FROM pending_embeds WHERE node_id IN ({marks})",
                    f"UPDATE audit_log SET before_json = NULL, after_json = NULL, actor = '' WHERE node_id IN ({marks}) AND op != 'forget'",
                ):
                    params = chunk * sql.count("(" + marks + ")")
                    try:
                        conn.execute(sql, params)
                    except sqlite3.OperationalError:
                        pass   # a snapshot from before that table existed
            for nid in touched_ids:
                row = conn.execute(
                    "SELECT label, content, event_time_text, source_context, metadata "
                    "FROM nodes WHERE node_id = ?", (nid,)).fetchone()
                if row:
                    cut = [scrub.cut_text(row[0], strings)[0], scrub.cut_text(row[1], strings)[0],
                           scrub.cut_text(row[2] or "", strings)[0],
                           scrub.cut_text(row[3] or "", strings)[0],
                           scrub.cut_json_text(row[4] or "{}", strings)[0]]
                    conn.execute(
                        "UPDATE nodes SET label=?, content=?, event_time_text=?, "
                        "source_context=?, metadata=? WHERE node_id=?", (*cut, nid))
                for rid, bj, aj in conn.execute(
                        "SELECT id, before_json, after_json FROM audit_log WHERE node_id = ?",
                        (nid,)).fetchall():
                    conn.execute("UPDATE audit_log SET before_json=?, after_json=? WHERE id=?",
                                 (scrub.cut_json_text(bj, strings)[0] if bj else bj,
                                  scrub.cut_json_text(aj, strings)[0] if aj else aj, rid))
            # nodes of other sources the exact targeted wording was cut from
            for nid in place_ids:
                row = conn.execute(
                    "SELECT label, content, event_time_text, source_context, metadata "
                    "FROM nodes WHERE node_id = ?", (nid,)).fetchone()
                if row:
                    new_content = scrub.cut_text(row[1], place_strings)[0]
                    if not new_content.strip():
                        new_content = "[removed]"
                    new_label = scrub.cut_text(row[0], place_strings)[0] or "[removed]"
                    lab = new_label.strip()
                    if len(lab) >= scrub.MIN_SHORT and any(
                            lab.casefold() in w.casefold() for w in place_strings):
                        new_label = "[removed]"
                    cut = [new_label, new_content,
                           scrub.cut_text(row[2] or "", place_strings)[0],
                           scrub.cut_text(row[3] or "", place_strings)[0],
                           scrub.cut_json_text(row[4] or "{}", place_strings)[0]]
                    conn.execute(
                        "UPDATE nodes SET label=?, content=?, event_time_text=?, "
                        "source_context=?, metadata=? WHERE node_id=?", (*cut, nid))
                for rid, bj, aj in conn.execute(
                        "SELECT id, before_json, after_json FROM audit_log WHERE node_id = ?",
                        (nid,)).fetchall():
                    conn.execute("UPDATE audit_log SET before_json=?, after_json=? WHERE id=?",
                                 (scrub.cut_json_text(bj, audit_strings)[0] if bj else bj,
                                  scrub.cut_json_text(aj, audit_strings)[0] if aj else aj, rid))
            # recall queries: the same ones the live database clears (the
            # recalls that surfaced a forgotten node), and any that quote it
            try:
                for rid_col, ep, text, terms in conn.execute(
                        "SELECT rowid, episode_key, query, shared_terms FROM use_judgments"
                ).fetchall():
                    nt = scrub.cut_json_text(terms or "[]", strings)[0]
                    if ep in episodes or scrub.contains(text or "", strings):
                        conn.execute(
                            "UPDATE use_judgments SET query = '', shared_terms = ? "
                            "WHERE rowid = ?", (nt, rid_col))
                    elif nt != (terms or "[]"):
                        conn.execute("UPDATE use_judgments SET shared_terms = ? "
                                     "WHERE rowid = ?", (nt, rid_col))
            except sqlite3.OperationalError:
                pass
            try:
                for recall_id, text, raw in conn.execute(
                        "SELECT recall_id, query, node_ids FROM recall_receipts"
                ).fetchall():
                    if recall_id in receipt_ids or scrub.contains(text or "", strings):
                        conn.execute(
                            "UPDATE recall_receipts SET query = '' WHERE recall_id = ?",
                            (recall_id,))
                    try:
                        listed = json.loads(raw or "[]")
                    except ValueError:
                        continue
                    kept = [i for i in listed if i not in removed_ids]
                    if kept != listed:
                        conn.execute(
                            "UPDATE recall_receipts SET node_ids = ? WHERE recall_id = ?",
                            (json.dumps(kept), recall_id))
            except sqlite3.OperationalError:
                pass
            try:
                for cid, reason, trace in conn.execute(
                        "SELECT id, reason, trace FROM supersession_candidates").fetchall():
                    conn.execute(
                        "UPDATE supersession_candidates SET reason=?, trace=? WHERE id=?",
                        (scrub.cut_text(reason or "", strings)[0],
                         scrub.cut_json_text(trace or "", strings)[0], cid))
            except sqlite3.OperationalError:
                pass
            # the vectors of the removed and the cut nodes
            vec_ids = list(dict.fromkeys(
                list(removed_ids) + list(touched_ids) + list(place_ids)))
            has_vec = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'vec_nodes'"
            ).fetchone() is not None
            vec_ok = False
            if has_vec:
                try:
                    self.store._load_vec_extension(conn)
                    vec_ok = True
                except Exception as e:  # noqa: BLE001
                    if vec_ids:
                        notes.append(
                            "its vector index could not be opened to remove "
                            f"the vectors of the forgotten nodes ({e})")
            if vec_ok and vec_ids:
                conn.executemany("DELETE FROM vec_nodes WHERE node_id = ?",
                                 [(i,) for i in vec_ids])
            conn.commit()
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError as e:
                if not (has_vec and not vec_ok):
                    notes.append(f"could not be rewritten ({e})")
        finally:
            conn.close()
        return notes

    def _model_after_forget(self, now: datetime) -> Dict:
        """The trained ranking model cannot unlearn a sentence: it was fitted
        on queries and node labels. Mark it stale where it lives (the training
        database, read by the retrain path) and, when neural ranking is on,
        retrain it now from the scrubbed signals. Says what happened."""
        marked = 0
        for path in self.training_dbs:
            if not os.path.exists(path):
                continue
            try:
                conn = sqlite3.connect(path)
                try:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS training_state "
                        "(key TEXT PRIMARY KEY, value TEXT)")
                    conn.execute(
                        "INSERT OR REPLACE INTO training_state (key, value) "
                        "VALUES ('model_stale', ?)", (now.isoformat(),))
                    conn.commit()
                    marked += 1
                finally:
                    conn.close()
            except sqlite3.Error:
                continue
        retrained, note = False, ""
        if self.retrain is not None:
            try:
                retrained = bool(self.retrain())
            except Exception as e:  # noqa: BLE001 - never fail a forget on this
                note = f"the retrain could not run: {type(e).__name__}: {e}"
            if not retrained and not note:
                reason = None
                if self.retrain_reason is not None:
                    try:
                        reason = self.retrain_reason()
                    except Exception:  # noqa: BLE001
                        reason = None
                note = "the retrain could not run: " + (
                    reason or "it reported no reason")
        return {"neural_ranking": self.retrain is not None,
                "retrained": retrained, "stale": not retrained,
                "marked": marked, "note": note}

    #: A word of the forgotten text is "distinctive" when at most this many
    #: nodes that another live unit vouches for use it (a word the rest of the
    #: graph uses freely identifies nothing).
    DISTINCTIVE_MAX_ELSEWHERE = 2

    def _identifying_tokens(self, plan: Plan, forgotten_keys: List[str]):
        """(strict, loose): words of the forgotten text that point back at it.
        ``strict``: no node another live unit vouches for uses the word, so any
        hit anywhere is the forgotten text. ``loose``: such nodes use it, but
        only DISTINCTIVE_MAX_ELSEWHERE or fewer; a node nobody vouches for
        that holds it (a hand-added note, a node whose owner is unproven) is
        still holding the forgotten text. There is no cap on how many such
        nodes there are. Read before anything is cut."""
        texts = [plan.target.content, plan.target.label] + list(forgotten_keys)
        for n in plan.remove + plan.cut_back + plan.legacy_cut:
            texts += [n.content, n.label]
        candidates: Set[str] = set()
        for t in texts:
            candidates |= scrub.tokens_of(t)
        if not candidates:
            return set(), set()
        forgotten_units = set(plan.units)
        conn = self.store._get_conn()
        with self.store._lock:
            vouched_elsewhere = set()
            for nid, unit in conn.execute(
                    "SELECT node_id, unit_id FROM node_provenance"):
                if not is_unproven(unit) and unit not in forgotten_units:
                    vouched_elsewhere.add(nid)
            elsewhere: Dict[str, int] = {}
            for nid, label, content in conn.execute(
                    "SELECT node_id, label, content FROM nodes"):
                if is_tombstone(nid) or nid not in vouched_elsewhere:
                    continue
                for tok in scrub.tokens_of(f"{label}\n{content}") & candidates:
                    elsewhere[tok] = elsewhere.get(tok, 0) + 1
        strict = {t for t in candidates if elsewhere.get(t, 0) == 0}
        loose = {t for t in candidates
                 if elsewhere.get(t, 0) <= self.DISTINCTIVE_MAX_ELSEWHERE}
        return strict, loose

    def _held_elsewhere(self, plan: Plan, forgotten_keys: List[str],
                        identifying=(frozenset(), frozenset()),
                        touched: Iterable[str] = ()) -> List[Dict]:
        """Nodes that STILL carry what this forget took out, and whose they
        are. A node counts when it holds, as whole words,
          * a sentence this forget took out (any owner: another source's own
            copy, a kept shared node, a hand-added note);
          * a distinctive word of the forgotten text, if nobody vouches for
            the node (unowned, or its owner is unproven);
          * for a node this forget cut back, cut in place or kept: a label or description of
            20 or more characters that is a piece of the forgotten text, which
            is the claim itself and not just a name.
        Plain forget is per source, so these are named, not touched;
        `forget --everywhere` cuts the exact wording out of them."""
        strict, loose = identifying
        touched = set(touched)
        sentences = [k for k in forgotten_keys if len(k) >= scrub.MIN_SENTENCE]
        if plan.target is not None and not is_holder(plan.target):
            sentences += scrub.clean_strings([plan.target.content])
        sentence_pat = scrub.compile_strings(scrub.clean_strings(sentences))
        token_pat = scrub.compile_strings(scrub.clean_strings(sorted(loose)))
        whole = list(forgotten_keys)
        if plan.target is not None:
            whole.append(plan.target.content)
        found: Dict[str, Node] = {}
        conn = self.store._get_conn()
        with self.store._lock:
            rows = conn.execute("SELECT node_id, label, content FROM nodes").fetchall()
            vouched = {}
            for nid, unit in conn.execute("SELECT node_id, unit_id FROM node_provenance"):
                if not is_unproven(unit):
                    vouched[nid] = True
        for nid, label, content in rows:
            if is_tombstone(nid):
                continue
            text = scrub.nfc(f"{label}\n{content}")
            if sentence_pat is not None and sentence_pat.search(text):
                found[nid] = None
                continue
            if token_pat is not None and token_pat.search(text) \
                    and nid not in vouched:
                found[nid] = None
                continue
            if nid in touched:
                for piece in (label, content):
                    t = (piece or "").strip()
                    if len(t) >= scrub.MIN_SENTENCE and any(
                            scrub.contains(w, [t]) for w in whole):
                        found[nid] = None
                        break
        # nodes kept and listed because their owner is unproven and what they
        # say is part of the forgotten text (a fragment of a forgotten
        # sentence, which a whole-sentence match would not see)
        for n in plan.unproven:
            if self.store.get_node(n.node_id) is None:
                continue
            for text in (n.content, n.label):
                t = (text or "").strip()
                if len(t) >= 12 and any(scrub.contains(w, [t]) for w in whole):
                    found.setdefault(n.node_id, None)
                    break
        return [{"node_id": nid, "units": self._owners_of(nid)} for nid in found]

    def _leftovers(self, plan, manifest_notes, problems, training, storage,
                   snapshots, held, model) -> List[Dict]:
        store = self.store
        left: List[Dict] = []
        for unit_id in plan.units:
            info = store.get_unit(unit_id) or {}
            if is_file_backed(unit_id) and info.get("path"):
                left.append({
                    "kind": "source_file", "unit_id": unit_id,
                    "path": info["path"],
                    "note": "your own file still contains it; Revien does "
                            "not edit your files"})
        for stem in manifest_notes:
            left.append({
                "kind": "vault_note", "note": stem,
                "note_text": f"the distilled note '{stem}' in your vault "
                             "still shows it",
                "fix": "revien distill-vault",
            })
        if store.get_meta("exports_written"):
            left.append({
                "kind": "exports",
                "note": "an export written to disk before this forget may "
                        "still contain it"})
        for u in snapshots.get("unreadable", []):
            left.append({
                "kind": "snapshots",
                "note": f"a `revien watch` backup could not be fully scrubbed "
                        f"and may still contain it ({u})"})
        for h in held:
            if h["units"] in (["(ownership unproven)"], ["(no recorded source)"]):
                note = ("this node holds part of the forgotten text, or words "
                        "only it used, and its owner is not proven, so Revien "
                        "left it untouched; delete it by id if it is yours")
            else:
                note = ("this node still holds the wording and belongs to "
                        + ", ".join(h["units"]) + "; forget it there too, or "
                        "use `forget --everywhere`")
            left.append({"kind": "other_units", "node_id": h["node_id"],
                         "units": h["units"], "note": note})
        models = Path(os.path.expanduser("~")) / ".revien" / "models"
        if model["stale"] and models.is_dir() and any(models.iterdir()):
            left.append({
                "kind": "neural_model", "path": str(models),
                "note": "the trained ranking model cannot unlearn and is "
                        "marked stale: it may still hold words from the "
                        "forgotten text until it is retrained. "
                        + ("Neural ranking is on, and " + model["note"] + ". "
                           "It retries on its own as retrieval signals "
                           "accumulate; to retrain now run POST /v1/training/run"
                           if model["neural_ranking"] else
                           "Neural ranking is off, so nothing retrains it: run "
                           "POST /v1/training/run")})
        if problems:
            left.append({"kind": "vectors", "note": problems})
        for p in training.get("problems", []):
            left.append({"kind": "training_db", "note": p})
        if not storage.get("vacuumed") or not storage.get("wal_truncated"):
            left.append({
                "kind": "storage",
                "note": "the database file could not be fully rewritten or "
                        "the write-ahead log truncated: " +
                        (storage.get("note") or "another connection is open")})
        return left

    # ── what a forget does and does not stop coming back ──────

    @staticmethod
    def _identity_class(unit_id: str) -> str:
        """How a source is recognised when its text comes back, from the way
        the code names it (revien/graph/units.py, revien/forget/gate.py)."""
        if unit_id.startswith("session:claude-code:"):
            return ("by its sessionId when the file carries one (a renamed, "
                    "moved or re-saved copy is the same session, and so is a "
                    "sub-agent file with its sessionId); a file with no "
                    "sessionId is known by its file name")
        if unit_id.startswith("session:codex:"):
            return ("by the id in the rollout's session_meta when it carries "
                    "one (a renamed or moved copy of the file keeps it); a "
                    "rollout without one is known by its file name")
        if unit_id.startswith("item:"):
            return ("by the id its source gave it (a conversation or item id "
                    "that travels with the export)")
        if unit_id.startswith(("file:", "skill:", "vault:")):
            return "by its real path and inode"
        return "it has no identity of its own (a shared channel)"

    DIFFERENT_IDENTITY = ("a copy of this source under a different identity "
                          "(another id, another path) is not recognised")

    def _reentry(self, plan: Plan, fallback, bound=None, not_bound=None) -> Dict:
        stops: List[str] = []
        not_stops: List[str] = []
        units = [b["unit_id"] for b in bound] if bound is not None else \
            [u for u in (plan.units or []) if not is_shared_channel(u)]
        for u in units:
            stops.append(f"{describe(u)}: this wording is left out of a later "
                         f"capture of it, {self._identity_class(u)}")
        if plan.everywhere:
            stops.append("the exact forgotten wording, in every source: the "
                         "sentences that held it are left out of any capture, "
                         "and a capture that is exactly that wording is "
                         "refused")
        else:
            not_stops.append(self.DIFFERENT_IDENTITY + "; use --everywhere to "
                             "keep this wording out of every source")
            not_stops.append("the same sentence arriving from another source "
                             "(another session, file or channel): that source "
                             "keeps its own copy, and still_held_by names it")
        if not units and not plan.everywhere:
            not_stops.append("no source with an identity was recorded for "
                             "this node, so nothing is kept out of a later "
                             "capture")
        not_stops.append("the rest of the source: only the sentences that "
                         "held this wording are left out; everything else in "
                         "it is captured as usual")
        if plan.everywhere:
            not_stops.append("a rewording of it: only the sentences that held "
                             "it, word for word, are recognised")
        out = {"scope": "everywhere" if plan.everywhere else "this source",
               "stops": stops, "does_not_stop": not_stops}
        if fallback:
            out["short_fragments"] = fallback["note"]
        return out

    def _finish(self, plan, before, after, tombstones, removed, stats,
                wording_cut, bound, not_bound, legacy_ids,
                manifest_notes, problems, training, storage, residue,
                snapshots, held, fallback, cut_in_place, model) -> Dict:
        target = plan.target
        leftovers = self._leftovers(
            plan, manifest_notes, problems, training, storage, snapshots, held,
            model)
        remaining = dict(residue)
        # "Gone everywhere" is printed only when it is true: nothing left in
        # Revien's own tables and nothing named as left behind.
        scope = "gone everywhere" if not leftovers and not remaining \
            else "forgotten from Revien"
        radius = self._blast_radius(plan, bound, not_bound, removed)
        result = {
            "status": "forgotten", "verb": VERB, "node_id": target.node_id,
            "everywhere": plan.everywhere,
            "forgotten": list(removed),
            "count": len(removed),
            "removed_nodes": len(removed),
            "tombstones_created": tombstones,
            "nodes_before": before["nodes"], "nodes_after": after["nodes"],
            "edges_before": before["edges"], "edges_after": after["edges"],
            "cut_back": [n.node_id for n in plan.cut_back] + list(legacy_ids),
            "kept": [{"node_id": k["node"].node_id, "by": k["by"],
                      "reason": "another live unit vouched for the name and "
                                "the wording"} for k in plan.keep_vouched],
            "unproven": [n.node_id for n in plan.unproven],
            "still_held_by": sorted({u for h in held for u in h["units"]}),
            "wording_cut": wording_cut,
            "cut_in_place": cut_in_place,
            "bound_to": bound,
            "not_bound": not_bound,
            "blast_radius": radius,
            "details": stats,
            "training_db": training,
            "snapshots": snapshots,
            "storage": storage,
            "remaining": remaining,
            "leftovers": leftovers,
            "scope": scope,
            "model": model,
            "model_stale": model["stale"],
            "reentry": self._reentry(plan, fallback, bound, not_bound),
        }
        if plan.likely_unit:
            result["identity_note"] = (
                "ownership of this node is unproven; the source was "
                "identified from its source key, so the wording is kept out "
                "of that source")
        elif plan.co_owners:
            result["identity_note"] = (
                "this node was written by several sources ("
                + ", ".join(plan.co_owners) + "), so it has no single origin: "
                "the exact wording is cut out of each of them in place and "
                "kept out of every source")
        elif plan.everywhere and not plan.units:
            result["identity_note"] = (
                "no source is known for this node; the exact wording is cut "
                "out of every node that holds it and kept out of every "
                "source")
        if fallback:
            result["fallback"] = fallback
        result["message"] = self._message(result)
        return result

    def _blast_radius(self, plan, bound, not_bound, removed) -> str:
        n = len(removed)
        count = "one node" if n == 1 else f"{n} nodes"
        parts = [f"This forgets {count}"]
        if bound:
            names = ", ".join(describe(b["unit_id"]) for b in bound)
            parts.append("and keeps its wording out of later captures of "
                         f"{names}")
        elif not_bound:
            parts.append("and cannot keep its wording out of a later capture: "
                         + NO_IDENTITY_NOTE)
        elif plan.everywhere:
            parts.append("and keeps its exact wording out of every source")
        else:
            parts.append("and keeps nothing out of a later capture: no "
                         "source with an identity was recorded for it")
        return " ".join(parts) + "."

    def _message(self, r: Dict) -> str:
        lines = [f"{r['scope'][0].upper()}{r['scope'][1:]}. {r['blast_radius']}"]
        if r["cut_back"]:
            lines.append(f"{len(r['cut_back'])} shared node(s) cut back to "
                         "their label.")
        if r["kept"]:
            lines.append(f"{len(r['kept'])} node(s) kept because another "
                         "unit vouched for them.")
        if r["unproven"]:
            lines.append(f"{len(r['unproven'])} node(s) with unproven "
                         "ownership were kept and listed, untouched.")
        for w in r["wording_cut"]:
            lines.append(f"Cut {w['occurrences']} occurrence(s) from context "
                         f"node {w['context_node_id']}.")
        if r["cut_in_place"]:
            lines.append(
                f"--everywhere cut the exact wording in place out of "
                f"{len(r['cut_in_place'])} node(s): "
                + "; ".join(f"{c['node_id']} ({', '.join(c['units'])})"
                            for c in r["cut_in_place"]) + ".")
        re_ = r["reentry"]
        lines.append(f"Scope: {re_['scope']}. Stops coming back: "
                     + (" | ".join(re_["stops"]) or "nothing") + ". Does not stop: "
                     + " | ".join(re_["does_not_stop"]) + ".")
        if r.get("fallback"):
            lines.append(r["fallback"]["note"])
        if r.get("identity_note"):
            note = r["identity_note"]
            lines.append(note[0].upper() + note[1:] + ".")
        for l in r["leftovers"]:
            if l["kind"] == "vault_note":
                lines.append(f"Forgotten from Revien only: {l['note_text']}. "
                             f"Run '{l['fix']}' to remove it from the note.")
            else:
                lines.append(f"Left in place: {l['note']}.")
        if r["remaining"]:
            lines.append("Still found in: " + ", ".join(sorted(r["remaining"])) + ".")
        return " ".join(lines)

    def _summary(self, plan: Plan, preview: bool) -> Dict:
        bound = [{"unit_id": u} for u in plan.units if not is_shared_channel(u)]
        not_bound = [{"unit_id": u, "note": NO_IDENTITY_NOTE}
                     for u in plan.units if is_shared_channel(u)]
        return {
            "status": "preview", "verb": VERB, "node_id": plan.target.node_id,
            "everywhere": plan.everywhere,
            "would_remove": [n.node_id for n in plan.remove],
            "would_cut_back": [n.node_id for n in plan.cut_back + plan.legacy_cut],
            "would_keep": [{"node_id": k["node"].node_id, "by": k["by"]}
                           for k in plan.keep_vouched],
            "unproven": [n.node_id for n in plan.unproven],
            "would_cut_wording_from": [h.node_id for h in plan.holders],
            "would_cut_in_place": [n.node_id for n in plan.everywhere_cut],
            "bound_to": bound, "not_bound": not_bound,
            "reentry": self._reentry(plan, None, bound, not_bound),
            "count": len(plan.remove), "forgotten": [],
            "blast_radius": self._blast_radius(
                plan, bound, not_bound, [n.node_id for n in plan.remove]).replace(
                    "forgets", "would forget"),
        }
