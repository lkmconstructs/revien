"""The forget verb (docs/FORGET_SPEC.md §1-§6).

Forget is privacy: the user wants this content gone and not coming back. It
removes the node, cuts the forgotten wording out of the session's stored
conversation, scrubs every copy the database keeps, leaves a content-free
suppression record so the source does not bring it back, and says exactly what
it did. It is the only code path that writes a ``forget`` audit entry; the
verb is recorded by being here, never by a reason string a caller passes.

Who owns what comes from ``node_provenance``, written when each node was
written. Nothing here infers ownership from edges or text. A node whose
ownership is unproven is kept and listed; it is deleted only when the user
names it.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from revien.forget import scrub
from revien.forget.keys import file_fingerprint, fingerprints
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.units import (
    CAPTURE_PREFIX, describe, identity_keys, is_file_backed, is_real_file,
    is_shared_channel, is_unproven, real_path)

TOMBSTONE_PREFIX = "tombstone:"
VERB = "forget"
RECAPTURE_NOTE = ("eligible for recapture because the source supplied no "
                  "stable item identity")

# Labels inside the forgotten sentence are cut from the unit's conversation
# too, but only when long enough to be identifying wording.
_MIN_LABEL_CUT = 4


def is_tombstone(node_id: str) -> bool:
    return node_id.startswith(TOMBSTONE_PREFIX)


def is_holder(node: Node) -> bool:
    """A node that holds a unit's words verbatim: the session's context node,
    a skill, or a node an importer marked as the verbatim message."""
    md = node.metadata or {}
    if md.get("_tombstone"):
        return False
    return node.node_type in (NodeType.CONTEXT, NodeType.SKILL) or bool(md.get("verbatim"))


def _word_in(needle: str, haystack: str) -> bool:
    return bool(re.search(
        r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack, re.IGNORECASE))


@dataclass
class Plan:
    target: Optional[Node] = None
    cascade: bool = False
    cascade_note: str = ""
    units: List[str] = field(default_factory=list)          # forgotten units
    remove: List[Node] = field(default_factory=list)
    cut_back: List[Node] = field(default_factory=list)
    keep_vouched: List[Dict] = field(default_factory=list)  # {node, by}
    unproven: List[Node] = field(default_factory=list)      # kept, scrubbed
    holders: List[Node] = field(default_factory=list)       # wording cut out
    strings: List[str] = field(default_factory=list)
    refusal: Optional[Dict] = None


class ForgetEngine:
    def __init__(self, store, semantic=None, training_dbs: Iterable[str] = (),
                 clustering=None):
        self.store = store
        self.semantic = semantic
        self.training_dbs = [str(p) for p in training_dbs]
        self.clustering = clustering

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
        label = (node.label or "").strip()
        for w in primary:
            if w.casefold() in (node.content or "").casefold() \
                    or w.casefold() in label.casefold():
                return True
            if len(label) >= _MIN_LABEL_CUT and _word_in(label, w):
                return True
        return False

    def plan(self, node_id: str, cascade: bool = False) -> Plan:
        store = self.store
        plan = Plan(cascade=cascade)
        target = store.get_node(node_id)
        plan.target = target
        if target is None:
            return plan
        rows = store.get_provenance(node_id)
        proven = [r for r in rows if not is_unproven(r["unit_id"])]
        holder = is_holder(target)
        forgotten = [r["unit_id"] for r in proven
                     if r["wording_ok"] or holder]
        plan.units = sorted(set(forgotten))
        fset = set(plan.units)

        primary = scrub.clean_strings([target.content, target.label])
        if cascade and not holder:
            plan.cascade_note = ("cascade applies to a unit's context node; "
                                 "this node was forgotten on its own")
            plan.cascade = False
        unit_cascade = plan.cascade and holder

        remove: Dict[str, Node] = {target.node_id: target}
        cut_back: Dict[str, Node] = {}
        keep: List[Dict] = []
        unproven: Dict[str, Node] = {}

        # The target itself.
        verdict, by = self._classify(rows, fset)
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
                if not unit_cascade and not self._carries(n, primary):
                    continue
                v, by = self._classify(store.get_provenance(nid), fset)
                if v == "unproven":
                    unproven[nid] = n
                elif v == "keep":
                    keep.append({"node": n, "by": by})
                elif v == "cut_back":
                    cut_back[nid] = n
                else:
                    remove[nid] = n

        # Ownership that could not be proven: kept and listed. The likely
        # unit is a hint for listing and scrubbing, never a reason to delete.
        for unit_id in plan.units:
            for nid in self._unproven_nodes_for(unit_id):
                if nid in remove or nid in cut_back or nid == target.node_id:
                    continue
                n = store.get_node(nid)
                if n is not None:
                    unproven[nid] = n
        if not plan.units or not proven:
            for n in self._legacy_carriers(primary):
                if n.node_id not in remove and n.node_id != target.node_id \
                        and n.node_id not in cut_back:
                    unproven[n.node_id] = n

        # Every string that gets cut: the target's wording, the wording of
        # what is removed or cut back, and the labels of removed nodes that
        # are inside the forgotten sentence.
        strings: List[str] = list(primary)
        for n in list(remove.values()) + list(cut_back.values()):
            strings.append(n.content)
            if n.node_id != target.node_id and n.node_id in remove \
                    and any(_word_in(n.label.strip(), w) for w in primary
                            if len(n.label.strip()) >= _MIN_LABEL_CUT):
                strings.append(n.label)
        if holder or unit_cascade:
            # The unit's own words: handled by removal, not by cutting.
            strings = [s for s in strings if s != target.content] \
                if unit_cascade else strings
        plan.strings = scrub.clean_strings(strings)

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
        for n in unproven.values():
            if is_holder(n) and (scrub.contains(n.content, plan.strings)):
                holders.setdefault(n.node_id, n)

        # The wording must be found verbatim where the unit stored it.
        if (not holder and plan.units and had_holder
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
                    "was changed. To remove it, forget the whole "
                    "conversation: forget that context node with cascade."),
            }

        plan.remove = list(remove.values())
        plan.cut_back = list(cut_back.values())
        plan.keep_vouched = keep
        plan.unproven = list(unproven.values())
        plan.holders = [h for h in holders.values()
                        if h.node_id not in unproven]
        return plan

    def _unproven_nodes_for(self, unit_id: str) -> List[str]:
        conn = self.store._get_conn()
        with self.store._lock:
            return [r[0] for r in conn.execute(
                "SELECT node_id FROM node_provenance WHERE unit_id = ?",
                ("unproven:" + unit_id,))]

    def _legacy_carriers(self, primary: List[str]) -> List[Node]:
        """Nodes with no proven owner whose text contains the forgotten
        wording word for word. They are scrubbed and kept, never deleted."""
        out: Dict[str, Node] = {}
        conn = self.store._get_conn()
        with self.store._lock:
            for w in primary:
                for (nid,) in conn.execute(
                        "SELECT n.node_id FROM nodes n WHERE "
                        "instr(lower(n.label || char(10) || n.content), ?) > 0 "
                        "AND NOT EXISTS (SELECT 1 FROM node_provenance p WHERE "
                        "p.node_id = n.node_id AND substr(p.unit_id,1,9) != "
                        "'unproven:')", (w.lower(),)):
                    if is_tombstone(nid) or nid in out:
                        continue
                    n = self.store.get_node(nid)
                    if n is not None:
                        out[nid] = n
        return list(out.values())

    # ── the verb ──────────────────────────────────────────

    def forget(self, node_id: str, cascade: bool = False,
               actor: str = "forget", dry_run: bool = False) -> Dict:
        with self.store._capture_lock:
            if is_tombstone(node_id):
                return {"status": "skipped", "verb": VERB, "node_id": node_id,
                        "forgotten": [], "count": 0, "reason": "is_tombstone"}
            plan = self.plan(node_id, cascade)
            if plan.target is None:
                return {"status": "not_found", "verb": VERB,
                        "node_id": node_id, "forgotten": [], "count": 0}
            if plan.refusal is not None:
                return {"status": "refused", "verb": VERB, "node_id": node_id,
                        "forgotten": [], "count": 0, **plan.refusal}
            if dry_run:
                return self._summary(plan, preview=True)
            return self._execute(plan, actor)

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

    def _suppression_specs(self, plan: Plan) -> List[Dict]:
        """One spec per stable forgotten unit, built BEFORE anything is cut so
        the fingerprints are of the text as it was."""
        store = self.store
        specs = []
        wording = [n.content for n in plan.remove + plan.cut_back] \
            + [n.label for n in plan.remove + plan.cut_back] \
            + [plan.target.content, plan.target.label]
        for unit_id in plan.units:
            if is_shared_channel(unit_id):
                specs.append({"unit_id": unit_id, "stable": False})
                continue
            info = store.get_unit(unit_id) or {}
            path = info.get("path")
            keys = list(info.get("path_keys") or [])
            file_level = unit_id.startswith(("session:", "file:", "skill:"))
            file_backed = is_file_backed(unit_id)
            real = None
            if path and is_real_file(path):
                real = real_path(path)
                keys = sorted(set(keys) | set(identity_keys(path)))
            texts = []
            for nid in self._members(unit_id):
                n = store.get_node(nid)
                if n is not None and is_holder(n):
                    texts.append(n.content)
            specs.append({
                "unit_id": unit_id, "stable": True,
                "kind": "file" if file_backed else "permanent",
                "path": real or path, "path_keys": keys,
                "content_fps": fingerprints(texts),
                "wording_fps": fingerprints(wording),
                "file_fp": (file_fingerprint(real)
                            if (real and file_level) else None),
            })
        return specs

    def _execute(self, plan: Plan, actor: str) -> Dict:
        store = self.store
        target = plan.target
        now = datetime.now(timezone.utc)
        strings = plan.strings
        remove_ids = [n.node_id for n in plan.remove]
        cut_ids = [n.node_id for n in plan.cut_back]
        unproven_ids = [n.node_id for n in plan.unproven]
        holder_ids = [n.node_id for n in plan.holders]
        keep_ids = [k["node"].node_id for k in plan.keep_vouched]
        forgotten_units = set(plan.units)
        changed_ids = list(dict.fromkeys(cut_ids + unproven_ids + holder_ids))

        before = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        manifest_notes = self._manifest_notes(remove_ids + cut_ids)
        specs = self._suppression_specs(plan)
        stats = {
            "audit_rows_scrubbed": 0, "judgments_removed": 0,
            "judgments_queries_cleared": 0, "receipts_cleared": 0,
            "candidates_removed": 0, "candidates_scrubbed": 0,
            "manifest_rows_removed": 0, "pending_embeds_removed": 0,
            "wording_cuts": 0,
        }
        wording_cut: List[Dict] = []
        tombstones = 0
        removed_done: List[str] = []
        suppressed: List[Dict] = []
        not_suppressed: List[Dict] = []
        scrubbed_unproven: List[str] = []

        with store.transaction() as conn:
            # 1. Content-free suppression records (the unit stays out).
            for spec in specs:
                if not spec["stable"]:
                    not_suppressed.append({
                        "unit_id": spec["unit_id"], "note": RECAPTURE_NOTE})
                    continue
                sid = store.add_suppression(
                    spec["unit_id"], spec["kind"], spec["path"],
                    spec["path_keys"], spec["content_fps"],
                    spec["wording_fps"], spec["file_fp"], actor)
                store.record_audit(
                    f"suppression:{sid}", "suppress", actor=actor,
                    after={"unit_id": spec["unit_id"], "kind": spec["kind"],
                           "verb": VERB}, ts=now)
                suppressed.append({"unit_id": spec["unit_id"],
                                   "kind": spec["kind"]})
            # A forgotten unit with no spec (not stable) is simply deleted.

            # 2. What the recall surfaces remember about these nodes, read
            #    BEFORE the nodes go (deleting a node strips its id).
            affected = set(remove_ids + cut_ids + changed_ids)
            episodes = self._episodes_for(conn, affected)
            receipt_ids = self._receipts_for(conn, affected)

            # 3. Audit: snapshots of the removed nodes are dropped, and the
            #    forgotten wording is cut from every other row's snapshot.
            stats["audit_rows_scrubbed"] = self._scrub_audit(
                conn, set(remove_ids), strings)

            # 4. Lineage: surviving children of a removed node point at a
            #    content-free marker, as before.
            tombstones += self._repoint_children(remove_ids, now)

            # 5. Cut the wording out of the stored conversation (and out of
            #    unproven nodes that carry it).
            for nid in holder_ids + [u for u in unproven_ids if u not in holder_ids]:
                info = self._cut_node(conn, nid, strings, cut_back=False)
                if info["cuts"]:
                    stats["wording_cuts"] += info["cuts"]
                    if nid in holder_ids:
                        wording_cut.append({"context_node_id": nid,
                                            "occurrences": info["cuts"]})
                    else:
                        scrubbed_unproven.append(nid)
                    store.record_audit(nid, "forget_cut", actor=actor,
                                       after={"verb": VERB}, ts=now)
            # 5b. Unproven nodes are also cut back to their label.
            for n in plan.unproven:
                if is_holder(n):
                    continue  # a conversation is cut, never reduced to a label
                if self._cut_node(conn, n.node_id, strings, cut_back=True)["changed"]:
                    store.record_audit(n.node_id, "cut_back", actor=actor,
                                       after={"verb": VERB}, ts=now)
            # 6. Shared nodes whose wording only the forgotten unit supplied.
            for nid in cut_ids:
                self._cut_node(conn, nid, strings, cut_back=True)
                store.delete_provenance(nid, forgotten_units)
                store.record_audit(nid, "cut_back", actor=actor,
                                   after={"verb": VERB}, ts=now)
            for nid in keep_ids:
                store.delete_provenance(nid, forgotten_units)

            # 7. The forgotten nodes themselves.
            for nid in remove_ids:
                store.record_audit(
                    nid, VERB, actor=actor, before=None,
                    after={"_tombstone": True, "original_node_id": nid,
                           "deleted_at": now.isoformat(), "verb": VERB},
                    ts=now)
                if store.delete_node(nid):
                    removed_done.append(nid)

            # 8. Everything else that quotes them.
            q = self._scrub_queries(conn, episodes, receipt_ids, strings)
            stats.update(q)
            stats["candidates_scrubbed"] = self._scrub_candidates(conn, strings)
        # ── committed ──

        gone = removed_done
        problems = self._purge_vectors(gone + changed_ids + cut_ids)
        training = self._scrub_training(gone, strings)
        if self.clustering is not None:
            try:
                self.clustering.run()
            except Exception:  # noqa: BLE001 - never fail a forget on this
                pass
        storage = store.scrub_storage()
        after = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        with store._lock:
            residue = scrub.find_residue(store._get_conn(), strings)

        return self._finish(
            plan, before, after, tombstones, removed_done, stats, wording_cut,
            suppressed, not_suppressed, scrubbed_unproven, manifest_notes,
            problems, training, storage, residue)

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

    def _scrub_audit(self, conn, removed: Set[str], strings: List[str]) -> int:
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
            nb, cb = scrub.cut_json_text(bj, strings) if bj else (bj, 0)
            na, ca = scrub.cut_json_text(aj, strings) if aj else (aj, 0)
            nact, cact = scrub.cut_text(actor or "", strings)
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
                  cut_back: bool) -> Dict:
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
        conn.execute(
            "UPDATE nodes SET label = ?, content = ?, event_time_text = ?, "
            "source_context = ?, metadata = ? WHERE node_id = ?",
            (new_label[:200], new_content, new_ett, new_sctx,
             json.dumps(meta), node_id))
        self.store._enqueue_embed_stmt(conn, node_id)
        return {"cuts": cuts, "changed": True}

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

    def _leftovers(self, plan, specs_units, manifest_notes, problems, training,
                   storage, residue, kept_nodes) -> List[Dict]:
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
        exp = store.get_meta("exports_written")
        if exp:
            left.append({
                "kind": "exports",
                "note": "exports written to disk before this forget still "
                        "contain it"})
        snap = Path(str(store.db_path) + ".snapshots")
        if snap.is_dir() and any(snap.iterdir()):
            left.append({
                "kind": "snapshots", "path": str(snap),
                "note": "backups made by 'revien watch' still contain it"})
        for k in kept_nodes:
            left.append({
                "kind": "other_units", "node_id": k["node_id"],
                "note": "another live unit vouched for this node's wording, "
                        "so it was kept", "by": k["by"]})
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

    def _finish(self, plan, before, after, tombstones, removed, stats,
                wording_cut, suppressed, not_suppressed, scrubbed_unproven,
                manifest_notes, problems, training, storage, residue) -> Dict:
        target = plan.target
        kept_nodes = [{"node_id": k["node"].node_id, "by": k["by"]}
                      for k in plan.keep_vouched]
        leftovers = self._leftovers(
            plan, None, manifest_notes, problems, training, storage, residue,
            kept_nodes)
        vault = any(l["kind"] == "vault_note" for l in leftovers)
        scope = "forgotten from Revien" if vault else "gone everywhere"
        # What the final scan still finds in Revien's own tables.
        remaining = dict(residue)
        radius = self._blast_radius(plan, suppressed, not_suppressed, removed)
        result = {
            "status": "forgotten", "verb": VERB, "node_id": target.node_id,
            "cascade": plan.cascade, "forgotten": list(removed),
            "count": len(removed),
            "removed_nodes": len(removed),
            "tombstones_created": tombstones,
            "nodes_before": before["nodes"], "nodes_after": after["nodes"],
            "edges_before": before["edges"], "edges_after": after["edges"],
            "cut_back": [n.node_id for n in plan.cut_back]
            + [n for n in scrubbed_unproven],
            "kept": [{"node_id": k["node"].node_id, "by": k["by"],
                      "reason": "another live unit vouched for the name and "
                                "the wording"} for k in plan.keep_vouched],
            "unproven": [n.node_id for n in plan.unproven],
            "wording_cut": wording_cut,
            "suppressed": suppressed,
            "not_suppressed": not_suppressed,
            "blast_radius": radius,
            "details": stats,
            "training_db": training,
            "storage": storage,
            "remaining": remaining,
            "leftovers": leftovers,
            "scope": scope,
        }
        if plan.cascade_note:
            result["cascade_note"] = plan.cascade_note
        result["message"] = self._message(result)
        return result

    def _blast_radius(self, plan, suppressed, not_suppressed, removed) -> str:
        n = len(removed)
        noun = "node" if n == 1 else "nodes"
        parts = [f"This forgets {n} {noun}"]
        if suppressed:
            names = ", ".join(describe(s["unit_id"]) for s in suppressed)
            parts.append(f"and suppresses future imports from {names}")
        elif not_suppressed:
            parts.append("and cannot suppress future imports: it is "
                         + RECAPTURE_NOTE)
        else:
            parts.append("and suppresses nothing: no source was recorded "
                         "for it")
        return " ".join(parts) + "."

    def _message(self, r: Dict) -> str:
        lines = [f"{r['scope'].capitalize()}. {r['blast_radius']}"]
        if r["cut_back"]:
            lines.append(f"{len(r['cut_back'])} shared node(s) cut back to "
                         "their label.")
        if r["kept"]:
            lines.append(f"{len(r['kept'])} node(s) kept because another "
                         "unit vouched for them.")
        if r["unproven"]:
            lines.append(f"{len(r['unproven'])} node(s) with unproven "
                         "ownership were kept and listed.")
        for w in r["wording_cut"]:
            lines.append(f"Cut {w['occurrences']} occurrence(s) from context "
                         f"node {w['context_node_id']}.")
        for l in r["leftovers"]:
            if l["kind"] == "vault_note":
                lines.append(f"Forgotten from Revien only: {l['note_text']}. "
                             f"Run '{l['fix']}' to remove it from the note.")
            else:
                lines.append(f"Left in place: {l['note']}.")
        return " ".join(lines)

    def _summary(self, plan: Plan, preview: bool) -> Dict:
        units = plan.units
        sup = []
        nosup = []
        for u in units:
            if is_shared_channel(u):
                nosup.append({"unit_id": u, "note": RECAPTURE_NOTE})
            else:
                sup.append({"unit_id": u,
                            "kind": "file" if is_file_backed(u) else "permanent"})
        return {
            "status": "preview", "verb": VERB, "node_id": plan.target.node_id,
            "cascade": plan.cascade,
            "would_remove": [n.node_id for n in plan.remove],
            "would_cut_back": [n.node_id for n in plan.cut_back],
            "would_keep": [{"node_id": k["node"].node_id, "by": k["by"]}
                           for k in plan.keep_vouched],
            "unproven": [n.node_id for n in plan.unproven],
            "would_cut_wording_from": [h.node_id for h in plan.holders],
            "suppressed": sup, "not_suppressed": nosup,
            "count": len(plan.remove), "forgotten": [],
            "blast_radius": self._blast_radius(
                plan, sup, nosup, [n.node_id for n in plan.remove]).replace(
                    "forgets", "would forget"),
            **({"cascade_note": plan.cascade_note} if plan.cascade_note else {}),
        }
