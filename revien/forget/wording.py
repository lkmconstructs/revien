"""Forget a literal string: ``revien forget --wording`` (docs/FORGET_SPEC.md §2a).

For a secret that should never have been recorded (a key, a token, a password).
The user names the exact characters; only those characters go:

* every occurrence is cut out of every node (a conversation's stored text
  included), out of every audit row (the rows of nodes that were deleted,
  edited or refreshed away before the forget as well), out of recall queries
  and receipts, supersession candidates and edges, ``training.db`` and the
  ``revien watch`` snapshots; the vectors of the nodes it touched are removed
  and the nodes re-embedded from the cut text;
* the cut is the string, not the sentence around it, and a line break the
  text wrapped it at does not hide it;
* the string goes on a global denylist as keyed fingerprints only (its length,
  a keyed rolling hash and a keyed HMAC), so the same characters never land
  again from any source or session. The record, the audit row of the forget,
  the log and the response never contain the string.

Matching is by raw characters and case-sensitive, not by whole words: a secret
sits inside ``?token=...&x`` and ``ghp_...`` where there is no word boundary.
The price is that a very short string would cut ordinary text, so a string
shorter than ``MIN_WORDING`` characters is refused, and the preview says how
much a string would cut before it does.
"""

import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from revien.forget import keys, scrub
from revien.graph.store import StoreBusy
from revien.graph.units import is_file_backed

#: The shortest string ``--wording`` accepts, in characters (line wraps
#: ignored). Below it a string is common enough to cut ordinary text; the
#: smallest secrets people actually paste (tokens, keys, passwords) are longer.
MIN_WORDING = 6

#: table -> (key column, plain text columns, JSON columns)
_TARGETS = {
    "nodes": ("node_id",
              ("label", "content", "event_time_text", "source_context"),
              ("metadata",)),
    "edges": ("edge_id", ("source_context",), ("metadata",)),
    "audit_log": ("id", ("actor",), ("before_json", "after_json")),
    "supersession_candidates": ("id", ("reason",), ("trace",)),
    "use_judgments": ("id", ("reason", "query"), ("shared_terms",)),
    "recall_receipts": ("recall_id", ("query",), ()),
}

TOO_SHORT = (f"--wording needs at least {MIN_WORDING} characters: a shorter "
             "string would cut ordinary text. Nothing was changed.")


def secret_regex(text: str) -> "re.Pattern[bytes]":
    """A bytes pattern for the string as it may sit in the raw database
    file: a line wrap inside it, raw or JSON-escaped, does not hide it."""
    canon = scrub.canonical(text).encode("utf-8")
    wrap = rb"(?:(?:\r?\n|\\r?\\n)[ \t]*)?"
    return re.compile(wrap.join(re.escape(canon[i:i + 1]) for i in range(len(canon))))


class WordingForget:
    """Mixed into ForgetEngine."""

    # ── what a string would cut (read only) ──────────────

    def _scan_literal(self, conn, lit, apply: bool, live: bool = False,
                      nodes_only_ids: Optional[set] = None) -> Dict:
        """Find (and with ``apply`` cut) the string in every table that can
        hold it. Returns counts and the ids of the nodes it touched."""
        out = {"nodes": 0, "occurrences": 0, "node_ids": [], "tables": {}}
        strings = [lit]
        for table, (keycol, plain, jsoncols) in _TARGETS.items():
            cols = list(plain) + list(jsoncols)
            try:
                rows = conn.execute(
                    f'SELECT "{keycol}", ' + ", ".join(f'"{c}"' for c in cols)
                    + f' FROM "{table}"').fetchall()
            except sqlite3.OperationalError:
                continue                     # a snapshot from before that table
            hit_rows = 0
            for row in rows:
                key, vals = row[0], list(row[1:])
                new_vals, cuts = [], 0
                for col, val in zip(cols, vals):
                    if not isinstance(val, str) or not val:
                        new_vals.append(val)
                        continue
                    if col in jsoncols:
                        new, n = scrub.cut_json_text(val, strings)
                    else:
                        new, n = scrub.cut_text(val, strings)
                    cuts += n
                    new_vals.append(new)
                if not cuts:
                    continue
                hit_rows += 1
                out["occurrences"] += cuts
                if table == "nodes":
                    out["nodes"] += 1
                    out["node_ids"].append(key)
                if apply:
                    if table == "nodes":
                        li, ci = cols.index("label"), cols.index("content")
                        if not (new_vals[li] or "").strip():
                            new_vals[li] = "[removed]"
                        if not (new_vals[ci] or "").strip():
                            new_vals[ci] = "[removed]"
                        new_vals[li] = new_vals[li][:200]
                    sets = ", ".join(f'"{c}" = ?' for c in cols)
                    conn.execute(f'UPDATE "{table}" SET {sets} WHERE "{keycol}" = ?',
                                 (*new_vals, key))
                    if table == "nodes" and live:
                        self.store._enqueue_embed_stmt(conn, key)
            if hit_rows:
                out["tables"][table] = hit_rows
        return out

    def _wording_check(self, wording: str):
        canon = scrub.canonical(wording)
        if len(canon.strip()) < MIN_WORDING:
            return None, {"status": "refused", "verb": "forget", "wording": True,
                          "reason": "wording_too_short", "forgotten": [], "count": 0,
                          "message": TOO_SHORT}
        return scrub.Literal(wording), None

    # ── the verb ─────────────────────────────────────────

    def forget_wording(self, wording: str, actor: str = "forget",
                       dry_run: bool = False) -> Dict:
        """Forget a literal string (see the module docstring). The string is
        never put in the result, an audit row, a record or a log."""
        lit, refusal = self._wording_check(wording)
        if refusal is not None:
            return refusal
        with self.store._capture_lock:
            if dry_run:
                with self.store._lock:
                    scan = self._scan_literal(self.store._get_conn(), lit, apply=False)
                return {
                    "status": "preview", "verb": "forget", "wording": True,
                    "would_cut_occurrences": scan["occurrences"],
                    "would_cut_nodes": scan["nodes"],
                    "would_cut_in": dict(scan["tables"]),
                    "count": 0, "forgotten": [],
                    "blast_radius": self._wording_radius(scan, preview=True)}
            record = {
                "unit_id": f"forget:{uuid.uuid4()}", "scope": "global",
                "path_keys": [], "wording_fps": [], "sentence_fps": [],
                "short_fps": [], "window_fps": [],
                "literal_fps": [keys.literal_entry(scrub.canonical(wording))]}
            self._warnings = []
            try:
                self._record_denylist([record], actor)
            except StoreBusy:
                return self._busy(None, committed=False)
            try:
                return self._execute_wording(lit, wording, actor)
            except StoreBusy:
                return self._busy(None, committed=True)

    @staticmethod
    def _wording_radius(scan: Dict, preview: bool = False) -> str:
        n, m = scan["occurrences"], scan["nodes"]
        verb = "would cut" if preview else "cut"
        return (f"This {verb} {n} occurrence(s) of the string from {m} node(s) "
                f"and {sum(v for t, v in scan['tables'].items() if t != 'nodes')} "
                "row(s) of the audit trail, recall records and edges, and keeps "
                "the same characters out of every later capture from any source.")

    def _execute_wording(self, lit, wording: str, actor: str) -> Dict:
        store = self.store
        now = datetime.now(timezone.utc)
        before = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        with store.transaction(immediate=True, wait=self.BUSY_WAIT_SECONDS) as conn:
            scan = self._scan_literal(conn, lit, apply=True, live=True)
            for nid in scan["node_ids"]:
                store.record_audit(nid, "forget_cut", actor=actor,
                                   after={"verb": "forget", "wording": True}, ts=now)
            store.record_audit(
                "wording:" + uuid.uuid4().hex[:12], "forget", actor=actor,
                after={"verb": "forget", "wording": True,
                       "occurrences": scan["occurrences"], "nodes": scan["nodes"]},
                ts=now)
        # ── committed ──
        touched = list(scan["node_ids"])
        manifest_notes = self._manifest_notes(touched)
        problems = self._purge_vectors(touched)
        training = self._scrub_training(touched, [lit])
        model = self._model_after_forget(now)
        snapshots = self._scrub_snapshots(
            (), (), [lit], frozenset(), scrub_fn=lambda p: self._scrub_snapshot_wording(p, lit))
        if self.clustering is not None:
            try:
                self.clustering._communities.clear()
                self.clustering._node_to_community.clear()
                self.clustering.run()
            except Exception:  # noqa: BLE001
                pass
        storage = store.scrub_storage()
        after = {"nodes": store.count_nodes(), "edges": store.count_edges()}
        with store._lock:
            residue = scrub.find_residue(store._get_conn(), [lit],
                                         ignore=scrub.NOT_CONTENT_TABLES)
        raw_hits = self._raw_hits(lit)
        files = self._source_files_holding(touched, lit)
        leftovers = self._wording_leftovers(
            manifest_notes, problems, training, storage, snapshots, model,
            raw_hits, files)
        partial = self._partial_copies(touched, lit)
        if partial:
            leftovers.append({
                "kind": "partial_copy", "node_ids": partial,
                "note": f"a piece of the string (not the whole of it) is left "
                        f"at the edge of {len(partial)} node(s) that held it; "
                        "Revien cuts only the whole string you named, so "
                        "name that piece with another --wording if it is "
                        "also secret"})
        remaining = dict(residue)
        gone = not leftovers and not remaining
        result = {
            "status": "forgotten", "verb": "forget", "wording": True,
            "forgotten": [], "count": 0, "removed_nodes": 0,
            "nodes_cut": scan["nodes"], "occurrences_cut": scan["occurrences"],
            "cut_in": dict(scan["tables"]),
            "nodes_before": before["nodes"], "nodes_after": after["nodes"],
            "edges_before": before["edges"], "edges_after": after["edges"],
            "still_held_by": sorted({u for f in files for u in f["units"]}),
            "training_db": training, "snapshots": snapshots, "storage": storage,
            "remaining": remaining, "leftovers": leftovers,
            "scope": (("nothing found" if not scan["occurrences"] else "gone everywhere")
                      if gone else "forgotten from Revien"),
            "model": model, "model_stale": model["stale"],
            "warnings": list(getattr(self, "_warnings", [])),
            "blast_radius": self._wording_radius(scan),
            "reentry": {
                "scope": "everywhere",
                "stops": ["this exact string (a line break inside it does not "
                          "hide it), in any new sentence from any source or "
                          "session: it is cut out of a capture before anything "
                          "is extracted or stored"],
                "does_not_stop": [
                    "a different spelling of it, a part of it, or the same "
                    "secret encoded another way (base64, URL-encoded): each "
                    "is a different string, so name each form",
                    "a copy that already left this machine (a cloud "
                    "extractor, a provider, a backup): rotate a leaked key "
                    "first, forget second"]},
        }
        result["message"] = self._wording_message(result)
        return result

    # ── the rest of the store and the files ──────────────

    def _scrub_snapshot_wording(self, path, lit) -> List[str]:
        notes: List[str] = []
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("PRAGMA secure_delete = ON")
            scan = self._scan_literal(conn, lit, apply=True, live=False)
            has_vec = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'vec_nodes'").fetchone() is not None
            vec_ok = False
            if has_vec:
                try:
                    self.store._load_vec_extension(conn)
                    vec_ok = True
                except Exception as e:  # noqa: BLE001
                    if scan["node_ids"]:
                        notes.append("its vector index could not be opened to "
                                     f"remove the vectors of the cut nodes ({e})")
            if vec_ok and scan["node_ids"]:
                conn.executemany("DELETE FROM vec_nodes WHERE node_id = ?",
                                 [(i,) for i in scan["node_ids"]])
            conn.commit()
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError as e:
                if not (has_vec and not vec_ok):
                    notes.append(f"could not be rewritten ({e})")
        finally:
            conn.close()
        return notes

    def _raw_hits(self, lit) -> List[str]:
        pat = secret_regex(str(lit))
        hits = []
        base = str(self.store.db_path)
        for suffix, name in (("", "database file"), ("-wal", "write-ahead log"),
                             ("-shm", "shared-memory file")):
            p = base + suffix
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        if pat.search(fh.read()):
                            hits.append(name)
                except OSError:
                    pass
        return hits

    def _source_files_holding(self, node_ids, lit) -> List[Dict]:
        """The user's own files, behind the units of the nodes that were cut,
        that still contain the string (Revien does not edit them)."""
        pat = secret_regex(str(lit))
        units: Dict[str, str] = {}
        for nid in node_ids:
            for r in self.store.get_provenance(nid):
                uid = r["unit_id"]
                if uid.startswith("unproven:"):
                    uid = uid[len("unproven:"):]
                if uid and is_file_backed(uid):
                    info = self.store.get_unit(uid) or {}
                    if info.get("path"):
                        units[uid] = info["path"]
        out = []
        for uid, path in sorted(units.items()):
            try:
                if Path(path).stat().st_size <= 64 * 1024 * 1024 and \
                        pat.search(Path(path).read_bytes()):
                    out.append({"unit_id": uid, "path": path, "units": [uid]})
            except OSError:
                continue
        return out

    def _wording_leftovers(self, manifest_notes, problems, training, storage,
                           snapshots, model, raw_hits, files) -> List[Dict]:
        store = self.store
        left: List[Dict] = []
        for f in files:
            left.append({"kind": "source_file", "unit_id": f["unit_id"],
                         "path": f["path"],
                         "note": "your own file still contains it; Revien does "
                                 "not edit your files"})
        for stem in manifest_notes:
            left.append({"kind": "vault_note", "note": stem,
                         "note_text": f"the distilled note '{stem}' in your "
                                      "vault still shows it",
                         "fix": "revien distill-vault"})
        if store.get_meta("exports_written"):
            left.append({"kind": "exports",
                         "note": "an export written to disk before this "
                                 "forget may still contain it"})
        for u in snapshots.get("unreadable", []):
            left.append({"kind": "snapshots",
                         "note": f"a `revien watch` backup could not be fully "
                                 f"scrubbed and may still contain it ({u})"})
        models = Path(os.path.expanduser("~")) / ".revien" / "models"
        if model["stale"] and models.is_dir() and any(models.iterdir()):
            left.append({"kind": "neural_model", "path": str(models),
                         "note": "the trained ranking model cannot unlearn and "
                                 "is marked stale until it is retrained "
                                 "(POST /v1/training/run)"})
        if problems:
            left.append({"kind": "vectors", "note": problems})
        for p in training.get("problems", []):
            left.append({"kind": "training_db", "note": p})
        if raw_hits:
            left.append({"kind": "storage",
                         "note": "the string was still found in the "
                                 + ", ".join(raw_hits) + " after the rewrite"})
        elif not storage.get("vacuumed") or not storage.get("wal_truncated"):
            left.append({"kind": "storage",
                         "note": "the database file could not be fully "
                                 "rewritten or the write-ahead log truncated: "
                                 + (storage.get("note") or "another connection is open")})
        return left

    def _partial_copies(self, node_ids, lit) -> List[str]:
        """Nodes that held the whole string and now begin or end with a long
        piece of it (a label cut where a line wrapped). Read only: nothing but
        the whole string is ever cut."""
        part = scrub.canonical(str(lit))
        floor = max(scrub.MIN_EDGE, (len(part) + 3) // 4)
        found = []
        with self.store._lock:
            conn = self.store._get_conn()
            for nid in node_ids:
                row = conn.execute("SELECT label, content FROM nodes WHERE node_id = ?",
                                   (nid,)).fetchone()
                if row is None:
                    continue
                for text in row:
                    t = (text or "").strip()
                    if any(t.endswith(part[:k]) or t.startswith(part[-k:])
                           for k in range(floor, len(part))):
                        found.append(nid)
                        break
        return found

    def _wording_message(self, r: Dict) -> str:
        if r["scope"] == "nothing found":
            lines = ["Nothing was found and nothing was changed: the string is "
                     "not in this database. It is still kept out of every later "
                     "capture."]
        else:
            lines = [f"{r['scope'][0].upper()}{r['scope'][1:]}. {r['blast_radius']}"]
        for w in r["warnings"]:
            lines.append(w)
        re_ = r["reentry"]
        lines.append("Stops coming back: " + " | ".join(re_["stops"])
                     + ". Does not stop: " + " | ".join(re_["does_not_stop"]) + ".")
        for l in r["leftovers"]:
            if l["kind"] == "vault_note":
                lines.append(f"Forgotten from Revien only: {l['note_text']}. "
                             f"Run '{l['fix']}' to remove it from the note.")
            else:
                lines.append(f"Left in place: {l['note']}.")
        if r["remaining"]:
            lines.append("Still found in: " + ", ".join(sorted(r["remaining"])) + ".")
        return " ".join(lines)
