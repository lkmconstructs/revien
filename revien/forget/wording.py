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

import json
import os
import re
import sqlite3
import uuid
import zlib
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

INSIDE_WORDS = (
    "{n} of the {m} occurrence(s) of this string sit inside a longer word "
    "(for example 'secret' inside 'secretary', which would be left as "
    "'ary'). The string is cut exactly as named, wherever it is, so those "
    "words would be damaged. Nothing was changed. Name a longer, more "
    "specific string, or confirm that you want them cut "
    "(--include-longer-words, or \"allow_inside_words\": true).")

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
        out = {"nodes": 0, "occurrences": 0, "node_ids": [], "tables": {},
               "inside_words": 0, "fragments": 0}
        part = scrub.canonical(str(lit))
        pat = scrub._pattern([lit])
        fragments: Dict[str, List[str]] = {}
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
                new_vals, cuts, content_cut = [], 0, 0
                for col, val in zip(cols, vals):
                    if not isinstance(val, str) or not val:
                        new_vals.append(val)
                        continue
                    if col in jsoncols:
                        new, n = scrub.cut_json_text(val, strings)
                    else:
                        new, n = scrub.cut_text(val, strings)
                    if n:
                        out["inside_words"] += scrub.inside_longer_words(
                            val, strings, col in jsoncols)
                    cuts += n
                    if col == "content":
                        content_cut = n
                    new_vals.append(new)
                if table == "nodes" and content_cut:
                    # A label (or a preview, a context line) is the content
                    # cut off somewhere: when this node's content held the
                    # string and the field stops exactly inside the occurrence,
                    # the piece it kept goes too (never a piece anywhere else).
                    orig_content = vals[cols.index("content")]
                    spans = [(m.start(), m.end()) for m in pat.finditer(orig_content)]
                    for fld in ("label", "event_time_text", "source_context"):
                        i = cols.index(fld)
                        if new_vals[i] != vals[i]:
                            continue          # the whole string was cut from it
                        piece = scrub.aligned_edge_fragment(
                            vals[i], orig_content, spans, part)
                        if piece:
                            new_vals[i], _ = scrub.cut_edge_fragment(
                                new_vals[i], part, piece)
                            fragments.setdefault(key, []).append(piece)
                            out["fragments"] += 1
                            cuts += 1
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
        if fragments:
            self._cut_fragments_in_audit(conn, fragments, part, apply, out)
        return out

    @staticmethod
    def _cut_fragments_in_audit(conn, fragments, part, apply, out) -> None:
        """The same pieces, cut from the audit snapshots of the same nodes
        (the create row quotes the label as it was)."""
        rows = 0
        for node_id, pieces in fragments.items():
            try:
                found = conn.execute(
                    "SELECT id, before_json, after_json FROM audit_log "
                    "WHERE node_id = ?", (node_id,)).fetchall()
            except sqlite3.OperationalError:
                continue
            for rid, bj, aj in found:
                new = []
                changed = False
                for raw in (bj, aj):
                    if not raw:
                        new.append(raw)
                        continue
                    try:
                        value = json.loads(raw)
                    except (TypeError, ValueError):
                        new.append(raw)
                        continue

                    def walk(v):
                        nonlocal changed
                        if isinstance(v, str):
                            for piece in pieces:
                                v2, cut = scrub.cut_edge_fragment(v, part, piece)
                                if cut:
                                    changed = True
                                    return v2
                            return v
                        if isinstance(v, list):
                            return [walk(x) for x in v]
                        if isinstance(v, dict):
                            return {k: walk(x) for k, x in v.items()}
                        return v

                    value = walk(value)
                    new.append(json.dumps(value))
                if changed:
                    rows += 1
                    if apply:
                        conn.execute("UPDATE audit_log SET before_json = ?, "
                                     "after_json = ? WHERE id = ?",
                                     (new[0], new[1], rid))
        if rows:
            out["tables"]["audit_log"] = out["tables"].get("audit_log", 0) + rows

    def _wording_check(self, wording: str):
        canon = scrub.canonical(wording)
        if len(canon.strip()) < MIN_WORDING:
            return None, {"status": "refused", "verb": "forget", "wording": True,
                          "reason": "wording_too_short", "forgotten": [], "count": 0,
                          "message": TOO_SHORT}
        return scrub.Literal(wording), None

    # ── the verb ─────────────────────────────────────────

    def forget_wording(self, wording: str, actor: str = "forget",
                       dry_run: bool = False,
                       allow_inside_words: bool = False) -> Dict:
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
                    "inside_longer_words": scan["inside_words"],
                    "count": 0, "forgotten": [],
                    "blast_radius": self._wording_radius(scan, preview=True)}
            with self.store._lock:
                scan = self._scan_literal(self.store._get_conn(), lit, apply=False)
            if scan["inside_words"] and not allow_inside_words:
                return {
                    "status": "refused", "verb": "forget", "wording": True,
                    "reason": "inside_longer_words", "forgotten": [], "count": 0,
                    "would_cut_occurrences": scan["occurrences"],
                    "inside_longer_words": scan["inside_words"],
                    "message": INSIDE_WORDS.format(
                        n=scan["inside_words"], m=scan["occurrences"])}
            record = {
                "unit_id": f"forget:{uuid.uuid4()}", "scope": "global",
                "path_keys": [], "wording_fps": [], "sentence_fps": [],
                "short_fps": [], "window_fps": [],
                "literal_fps": [keys.literal_entry(scrub.canonical(wording))]}
            self._warnings = []
            self._key_problem = None
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
        inside = scan.get("inside_words", 0)
        warn = (f"{inside} of them sit inside a longer word and would be cut "
                "out of it. " if inside else "")
        return (f"This {verb} {n} occurrence(s) of the string. {warn}"
                f"It reaches {m} node(s) "
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
        orphaned = self._key_leftover()
        if orphaned:
            leftovers.append(orphaned)
        partial, unreadable = self._partial_copies(lit)
        for path in unreadable:
            leftovers.append({
                "kind": "snapshot_unreadable", "path": path,
                "note": f"snapshot could not be read: {path}"})
        if partial:
            shown = partial[:10]
            where = "; ".join(
                f"{p['surface']} {p.get('node_id') or 'row ' + str(p.get('row'))} "
                f"{p['field']} "
                f"({p['characters']} characters)" for p in shown)
            more = (f" and {len(partial) - len(shown)} more"
                    if len(partial) > len(shown) else "")
            leftovers.append({
                "kind": "partial_copy", "pieces": partial,
                "note": f"a piece of the string, not the whole of it, sits at "
                        f"the edge of a field in {len(partial)} place(s): "
                        f"{where}{more}. It was not cut because it is not "
                        "where an occurrence was cut from the same node; name "
                        "the piece with another --wording if it is also secret"})
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

    @staticmethod
    def _fragment_report(conn, part, surface: str) -> List[Dict]:
        """Pieces of the string (MIN_EDGE characters or more) at the edge of a
        text field, on one database connection: every node field, and every
        audit row by the content of its cells. Where and how long only."""
        found = []

        def best(text):
            return max((k for _kind, k in scrub.edge_fragments(
                text or "", part, scrub.MIN_EDGE, report=True)), default=0)

        def strings_of(raw):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return [raw]
            return list(scrub._iter_strings(value))

        try:
            for nid, label, content, ett, sctx, meta in conn.execute(
                    "SELECT node_id, label, content, event_time_text, "
                    "source_context, metadata FROM nodes"):
                for field, text in (("label", label), ("content", content),
                                    ("event_time_text", ett),
                                    ("source_context", sctx)):
                    k = best(text)
                    if k:
                        found.append({"surface": surface + "node", "node_id": nid,
                                      "field": field, "characters": k})
                k = max((best(x) for x in strings_of(meta or "")), default=0)
                if k:
                    found.append({"surface": surface + "node", "node_id": nid,
                                  "field": "metadata", "characters": k})
        except sqlite3.OperationalError:
            pass
        try:
            for rid, nid, op, actor, bj, aj in conn.execute(
                    "SELECT id, node_id, op, actor, before_json, after_json FROM audit_log"):
                for field, raw in (("actor", actor), ("before_json", bj),
                                   ("after_json", aj)):
                    if not raw:
                        continue
                    if field == "actor" and op not in scrub._MERGE_OPS:
                        continue        # Revien's own actor ("api:forget"), not content
                    k = max((best(x) for x in strings_of(raw)), default=0)
                    if k:
                        found.append({"surface": surface + "audit_log",
                                      "node_id": nid, "row": rid, "field": field,
                                      "characters": k})
        except sqlite3.OperationalError:
            pass
        # the other tables that keep text: edges, candidates, judgments, receipts
        for table, (keycol, plain, jsoncols) in _TARGETS.items():
            if table in ("nodes", "audit_log"):
                continue
            cols = list(plain) + list(jsoncols)
            try:
                rows = conn.execute(
                    f'SELECT "{keycol}", ' + ", ".join(f'"{c}"' for c in cols)
                    + f' FROM "{table}"').fetchall()
            except sqlite3.OperationalError:
                continue
            for row in rows:
                for field, raw in zip(cols, row[1:]):
                    if not isinstance(raw, str) or not raw:
                        continue
                    k = max((best(x) for x in strings_of(raw)), default=0)
                    if k:
                        found.append({"surface": surface + table, "row": row[0],
                                      "field": field, "characters": k})
        return found

    def _partial_copies(self, lit):
        """Where a piece of the string still sits: the live database (every
        node field, every audit row) and every `revien watch` snapshot, plain
        or gzip. Read only. A piece is cut only where an occurrence was cut
        from the same node (see scrub.aligned_edge_fragment); anywhere else it
        is only reported, and the answer is then not "gone everywhere"."""
        import gzip
        import shutil
        import tempfile

        part = scrub.canonical(str(lit))
        unreadable: List[str] = []
        with self.store._lock:
            found = self._fragment_report(self.store._get_conn(), part, "")
        snap = Path(str(self.store.db_path) + ".snapshots")
        if snap.is_dir():
            for f in sorted(snap.iterdir()):
                if not (f.name.endswith(".db") or f.name.endswith(".db.gz")):
                    continue
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = f
                        if f.name.endswith(".gz"):
                            path = Path(tmp) / "snap.db"
                            with gzip.open(f, "rb") as src, open(path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                        try:
                            found.extend(self._fragment_report(
                                conn, part, f"snapshot {f.name} "))
                        finally:
                            conn.close()
                except (OSError, EOFError, zlib.error, sqlite3.Error):
                    # A .gz cut short mid-write raises EOFError; bad deflate
                    # data raises zlib.error. Both are unreadable, not fatal.
                    unreadable.append(str(f))
        return found, unreadable

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
            if l["kind"] == "forget_key":
                continue                     # said above, as a warning
            if l["kind"] == "vault_note":
                lines.append(f"Forgotten from Revien only: {l['note_text']}. "
                             f"Run '{l['fix']}' to remove it from the note.")
            else:
                lines.append(f"Left in place: {l['note']}.")
        if r["remaining"]:
            lines.append("Still found in: " + ", ".join(sorted(r["remaining"])) + ".")
        return " ".join(lines)
