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

from revien.forget import inventory, keys, scrub
from revien.graph.store import StoreBusy
from revien.graph.units import is_file_backed

#: The shortest string ``--wording`` accepts, in characters (line wraps
#: ignored). Below it a string is common enough to cut ordinary text; the
#: smallest secrets people actually paste (tokens, keys, passwords) are longer.
MIN_WORDING = 6

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

    @staticmethod
    def _derive_cut(value, part, pat, ctx=None):
        """Cut, in a decoded JSON value, the half of an occurrence that a
        derived string (a label, a preview) kept next to a ``content`` string
        that held it. Returns (value, pieces). Derived = a verbatim stretch of
        that content that stops exactly inside the occurrence."""
        pieces: List[str] = []

        def walk(v, ctx):
            if isinstance(v, dict):
                content = v.get("content")
                if isinstance(content, str) and content:
                    spans = [(m.start(), m.end()) for m in pat.finditer(content)]
                    if spans:
                        ctx = (content, spans)
                out = {}
                for k, x in v.items():
                    if isinstance(x, str) and ctx and not (k == "content" and x is v.get("content")):
                        piece = scrub.aligned_edge_fragment(x, ctx[0], ctx[1], part)
                        if piece:
                            x, _ = scrub.cut_edge_fragment(x, part, piece)
                            pieces.append(piece)
                    else:
                        x = walk(x, ctx)
                    out[k] = x
                return out
            if isinstance(v, list):
                return [walk(x, ctx) for x in v]
            return v

        return walk(value, ctx), pieces

    def _scan_literal(self, conn, lit, apply: bool, live: bool = False,
                      nodes_only_ids: Optional[set] = None) -> Dict:
        """Find (and with ``apply`` cut) the string in every text column of
        the store (revien/forget/inventory.py, read from the schema of
        ``conn``: the live database and each snapshot alike). Returns counts,
        the ids of the nodes it touched and the identities it renamed."""
        out = {"nodes": 0, "occurrences": 0, "node_ids": [], "tables": {},
               "inside_words": 0, "fragments": 0, "renamed": {},
               "renamed_ids": [], "unrenamed": []}
        strings = [lit]
        part = scrub.canonical(str(lit))
        pat = scrub._pattern([lit])
        fragments: Dict[str, List[str]] = {}
        sources: Dict[str, tuple] = {}
        # the captured texts (context records) as they were before any cut:
        # the evidence a derived stretch is judged against
        ctx_orig: Dict[str, tuple] = {}
        try:
            for cid, ctext in conn.execute(
                    "SELECT node_id, content FROM nodes WHERE node_type = 'context'"):
                if ctext:
                    ctx_orig[cid] = (ctext, [(m.start(), m.end())
                                             for m in pat.finditer(ctext)])
        except sqlite3.OperationalError:
            pass
        # Units whose cut form is already another unit: renaming their
        # ownership rows would merge two sources into one, so they are kept
        # as they are and listed (identity_unrenamed).
        blocked = set()
        try:
            units = [r[0] for r in conn.execute("SELECT unit_id FROM capture_units")]
            have = set(units)
            for u in units:
                cut, n = scrub.cut_json_text(u, strings)
                if n and cut != u and cut in have:
                    blocked.add(u)
        except sqlite3.OperationalError:
            pass
        blocked_sources = set()
        try:
            srcs = {r[0] for r in conn.execute(
                "SELECT DISTINCT source_id FROM nodes WHERE source_id IS NOT NULL")}
            for u in srcs:
                cut, n = scrub.cut_json_text(u, strings)
                if n and cut != u and cut in srcs:
                    blocked_sources.add(u)
        except sqlite3.OperationalError:
            pass
        for table, cols in inventory.columns(conn).items():
            try:
                rows = conn.execute(
                    "SELECT rowid, " + ", ".join(f'"{c}"' for c in cols)
                    + f' FROM "{table}"').fetchall()
            except sqlite3.OperationalError:
                continue
            nkey = None
            if table == "nodes":
                try:
                    nkey = {r[0]: r[1] for r in conn.execute(
                        'SELECT rowid, node_id FROM nodes')}
                except sqlite3.OperationalError:
                    nkey = {}
            hit_rows = 0
            for row in rows:
                rowid, vals = row[0], list(row[1:])
                new_vals, cuts, content_cut = list(vals), 0, 0
                node_id = nkey.get(rowid) if nkey is not None else None
                orig_content = vals[cols.index("content")] if table == "nodes" else None
                spans = []
                if orig_content:
                    spans = [(m.start(), m.end()) for m in pat.finditer(orig_content)]
                for i, (col, val) in enumerate(zip(cols, vals)):
                    as_bytes = isinstance(val, (bytes, bytearray))
                    if as_bytes:
                        val = bytes(val).decode("utf-8", "ignore")
                    if not isinstance(val, str) or not val:
                        continue
                    if table == "node_provenance" and col == "unit_id" and val in blocked:
                        if ["node_provenance.unit_id"] not in out["unrenamed"]:
                            out["unrenamed"].append(["node_provenance.unit_id"])
                        continue
                    if table == "nodes" and col == "source_id" and val in blocked_sources:
                        if ["nodes.source_id"] not in out["unrenamed"]:
                            out["unrenamed"].append(["nodes.source_id"])
                        continue
                    work = val
                    got = []
                    # a derived string next to its content, cut where it stops
                    # inside the occurrence (before the whole-string cut
                    # removes the occurrence it is measured against)
                    if table == "nodes" and spans and col in (
                            "label", "event_time_text", "source_context"):
                        piece = scrub.aligned_edge_fragment(
                            work, orig_content, spans, part,
                            others=self._evidence(conn, node_id, (), ctx_orig))
                        if piece:
                            work, _ = scrub.cut_edge_fragment(work, part, piece)
                            got.append(piece)
                    elif work[:1] in "{[" and (table == "nodes" or "content" in work):
                        try:
                            decoded = json.loads(work)
                        except (TypeError, ValueError):
                            decoded = None
                        if isinstance(decoded, (dict, list)):
                            ctx = (orig_content, spans) if (table == "nodes" and spans) else None
                            decoded, pieces = self._derive_cut(decoded, part, pat, ctx)
                            if pieces:
                                work = json.dumps(decoded)
                                got.extend(pieces)
                    new, n = scrub.cut_json_text(work, strings)
                    if n:
                        out["inside_words"] += scrub.inside_longer_words(
                            val, strings, True)
                    if n or got:
                        new_vals[i] = new.encode("utf-8") if as_bytes else new
                        cuts += n + len(got)
                        out["fragments"] += len(got)
                        if col == "content":
                            content_cut = n
                        if got and node_id:
                            fragments.setdefault(node_id, []).extend(got)
                if not cuts:
                    continue
                hit_rows += 1
                out["occurrences"] += cuts
                if table == "nodes":
                    out["nodes"] += 1
                    out["node_ids"].append(node_id)
                    if content_cut:
                        sources[node_id] = (orig_content, spans)
                renames = [(table, c) for c, a, b in zip(cols, vals, new_vals)
                           if a != b and (table, c) in inventory.IDENTITY]
                if apply:
                    if table == "nodes":
                        li, ci = cols.index("label"), cols.index("content")
                        if not (new_vals[li] or "").strip():
                            new_vals[li] = "[removed]"
                        if not (new_vals[ci] or "").strip():
                            new_vals[ci] = "[removed]"
                        new_vals[li] = new_vals[li][:200]
                    sets = ", ".join(f'"{c}" = ?' for c in cols)
                    try:
                        conn.execute(f'UPDATE "{table}" SET {sets} WHERE rowid = ?',
                                     (*new_vals, rowid))
                    except sqlite3.IntegrityError:
                        # the cut form of an identity is already taken: keep
                        # the row as it was (the string is still in it, and
                        # that is reported, never hidden)
                        out["unrenamed"].append(
                            sorted({f"{t}.{c}" for t, c in renames})
                            or [f"{table}"])
                        continue
                    if table == "nodes" and live:
                        self.store._enqueue_embed_stmt(conn, node_id)
                for t, c in renames:
                    key = f"{t}.{c}"
                    out["renamed"][key] = out["renamed"].get(key, 0) + 1
                    if len(out["renamed_ids"]) < 20:
                        out["renamed_ids"].append(new_vals[cols.index(c)])
            if hit_rows:
                out["tables"][table] = hit_rows
        self._cut_siblings(conn, sources, part, apply, out, fragments, ctx_orig)
        if fragments:
            self._cut_fragments_in_audit(conn, fragments, part, apply, out)
        return out

    @staticmethod
    def _evidence(conn, node_id, extra_units, ctx_orig):
        """(content, spans) of the captured text (context records) of every
        unit that vouches for ``node_id`` (and of ``extra_units``), as it was
        before this forget cut anything, without ``node_id`` itself."""
        units = set(extra_units)
        try:
            units.update(r[0] for r in conn.execute(
                "SELECT unit_id FROM node_provenance WHERE node_id = ?", (node_id,)))
            ids = set()
            for u in units:
                ids.update(r[0] for r in conn.execute(
                    "SELECT p.node_id FROM node_provenance p JOIN nodes n "
                    "ON n.node_id = p.node_id WHERE p.unit_id = ? "
                    "AND n.node_type = 'context'", (u,)))
        except sqlite3.OperationalError:
            return []
        ids.discard(node_id)
        out = []
        for oid in ids:
            if oid in ctx_orig:
                out.append(ctx_orig[oid])
            else:
                r = conn.execute("SELECT content FROM nodes WHERE node_id = ?",
                                 (oid,)).fetchone()
                out.append((r[0] if r and r[0] else "", []))
        return out

    def _cut_siblings(self, conn, sources, part, apply, out, fragments, ctx_orig) -> None:
        """Nodes written by the same capture as a node the string was cut from
        (a fact or a topic taken from a conversation) keep a derived stretch of
        it: the first words of the sentence up to where the extractor stopped.
        Cut it only where it stops exactly inside the cut occurrence."""
        if not sources:
            return
        try:
            conn.execute("SELECT 1 FROM node_provenance LIMIT 1")
        except sqlite3.OperationalError:
            return
        fields = ("label", "content", "event_time_text", "source_context")
        for nid, (content, spans) in sources.items():
            units = [r[0] for r in conn.execute(
                "SELECT unit_id FROM node_provenance WHERE node_id = ?", (nid,))]
            sib = set()
            for u in units:
                sib.update(r[0] for r in conn.execute(
                    "SELECT node_id FROM node_provenance WHERE unit_id = ?", (u,)))
            sib.discard(nid)
            for sid in sib:
                row = conn.execute(
                    "SELECT " + ", ".join(fields) + " FROM nodes WHERE node_id = ?",
                    (sid,)).fetchone()
                if row is None:
                    continue
                new = list(row)
                got = []
                mine = self._evidence(conn, sid, units, ctx_orig)
                for i, text in enumerate(row):
                    piece = scrub.aligned_edge_fragment(
                        text or "", content, spans, part, others=mine)
                    if piece:
                        new[i], _ = scrub.cut_edge_fragment(text, part, piece)
                        got.append(piece)
                if not got:
                    continue
                out["fragments"] += len(got)
                out["occurrences"] += len(got)
                out["tables"]["nodes"] = out["tables"].get("nodes", 0) + 1
                fragments.setdefault(sid, []).extend(got)
                if apply:
                    if not (new[0] or "").strip():
                        new[0] = "[removed]"
                    if not (new[1] or "").strip():
                        new[1] = "[removed]"
                    conn.execute(
                        "UPDATE nodes SET " + ", ".join(f"{f} = ?" for f in fields)
                        + " WHERE node_id = ?", (*new, sid))
                    if sid not in out["node_ids"]:
                        out["node_ids"].append(sid)
                        out["nodes"] += 1
                        self.store._enqueue_embed_stmt(conn, sid)

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
        self._snapshot_cuts = 0
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
            residue = scrub.find_residue(store._get_conn(), [lit])
        raw_hits = self._raw_hits(lit)
        files = self._source_files_holding(touched, lit)
        leftovers = self._wording_leftovers(
            manifest_notes, problems, training, storage, snapshots, model,
            raw_hits, files)
        orphaned = self._key_leftover()
        if orphaned:
            leftovers.append(orphaned)
        left = self._leftover_scan(lit)
        partial, unreadable = left["pieces"], left["unreadable"]
        for path in unreadable:
            leftovers.append({
                "kind": "snapshot_unreadable", "path": path,
                "note": f"snapshot could not be read: {path}"})
        for hit in left["snapshot_hits"]:
            leftovers.append({
                "kind": "snapshot_residue", **hit,
                "note": f"snapshot {hit['snapshot']} still holds the string "
                        f"({hit['where']}); `revien watch` snapshots are not "
                        "edited by any other verb"})
        for cols in scan["unrenamed"]:
            leftovers.append({
                "kind": "identity_unrenamed", "columns": cols,
                "note": "an identity (" + ", ".join(cols) + ") holds the "
                        "string and could not be renamed, because the cut "
                        "form is already taken by another record; no verb "
                        "removes it"})
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
            "renamed": dict(scan["renamed"]), "renamed_ids": list(scan["renamed_ids"]),
            "scope": (("nothing found" if not (scan["occurrences"] or self._snapshot_cuts)
                       else "gone everywhere")
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
            self._snapshot_cuts = getattr(self, "_snapshot_cuts", 0) + scan["occurrences"]
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

    @staticmethod
    def _piece_len(part: str) -> int:
        """How long a stretch of the string has to be to be looked for in raw
        bytes as "a piece of it": 16 characters, or all but one of a shorter
        string; 0 when the string is too short to have a piece worth finding."""
        k = min(len(part) - 1, max(scrub.edge_floor(part), 16))
        return k if k >= scrub.MIN_EDGE else 0

    def _raw_scan(self, data: bytes, lit) -> str:
        """"" when ``data`` holds neither the string (a line wrap, raw or
        JSON-escaped, does not hide it) nor a long piece of it at either end;
        else what it holds."""
        if secret_regex(str(lit)).search(data):
            return "the whole string"
        part = scrub.canonical(str(lit))
        k = self._piece_len(part)
        if k:
            for piece in {part[:k], part[-k:]}:
                if piece.encode("utf-8") in data:
                    return f"{k} characters of it"
        return ""

    def _raw_hits(self, lit) -> List[str]:
        hits = []
        base = str(self.store.db_path)
        for suffix, name in (("", "database file"), ("-wal", "write-ahead log"),
                             ("-shm", "shared-memory file")):
            p = base + suffix
            if os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        what = self._raw_scan(fh.read(), lit)
                    if what:
                        hits.append(name if what == "the whole string"
                                    else f"{name} ({what})")
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
            whole = [h for h in raw_hits if "(" not in h]
            pieces = [h for h in raw_hits if "(" in h]
            parts = []
            if whole:
                parts.append("the string was still found in the " + ", ".join(whole))
            if pieces:
                parts.append("a piece of the string (not the whole of it) was "
                             "found in the " + ", ".join(pieces))
            left.append({"kind": "storage",
                         "note": "; ".join(parts) + " after the rewrite"})
        elif not storage.get("vacuumed") or not storage.get("wal_truncated"):
            left.append({"kind": "storage",
                         "note": "the database file could not be fully "
                                 "rewritten or the write-ahead log truncated: "
                                 + (storage.get("note") or "another connection is open")})
        return left

    @staticmethod
    def _fragment_report(conn, part, surface: str) -> List[Dict]:
        """Pieces of the string (MIN_EDGE characters or more) at the start or
        end of a text, in every column of the inventory and every audit row,
        on one database connection. Where and how long only; a cell that is
        exactly one of Revien's own constant actor names is skipped."""
        found = []

        def best(text):
            return max((k for _kind, k in scrub.edge_fragments(
                text or "", part, scrub.MIN_EDGE, report=True)), default=0)

        def strings_of(raw):
            if raw[:1] not in ("{", "["):
                return [raw]
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return [raw]
            return list(scrub._iter_strings(value))

        for table, cols in inventory.columns(conn).items():
            extra = ""
            try:
                has_nid = any(r[1] == "node_id" for r in conn.execute(
                    f'PRAGMA table_info("{table}")'))
                if has_nid and table in ("nodes", "audit_log"):
                    extra = ', "node_id"'
                rows = conn.execute(
                    "SELECT rowid, " + ", ".join(f'"{c}"' for c in cols) + extra
                    + f' FROM "{table}"').fetchall()
            except sqlite3.OperationalError:
                continue
            nid_at = len(cols) + 1 if extra else None
            surf = surface + ("node" if table == "nodes" else table)
            for row in rows:
                for field, raw in zip(cols, row[1:1 + len(cols)]):
                    if isinstance(raw, (bytes, bytearray)):
                        raw = bytes(raw).decode("utf-8", "ignore")
                    if not isinstance(raw, str) or not raw or inventory.is_own(raw):
                        continue
                    k = max((best(x) for x in strings_of(raw)
                             if not inventory.is_own(x)), default=0)
                    if k:
                        item = {"surface": surf, "field": field, "characters": k}
                        if nid_at:
                            item["node_id"] = row[nid_at]
                        if table != "nodes":
                            item["row"] = row[0]
                        found.append(item)
        return found

    def _snapshot_files(self):
        snap = Path(str(self.store.db_path) + ".snapshots")
        if snap.is_dir():
            for f in sorted(snap.iterdir()):
                if f.name.endswith(".db") or f.name.endswith(".db.gz"):
                    yield f

    def _leftover_scan(self, lit) -> Dict:
        """Where a piece of the string, or the string, still sits: the live
        database (every text column, every audit row) and every `revien
        watch` snapshot, plain or gzip, by column and by raw bytes. Read only.
        Anything found keeps the answer from being "gone everywhere"."""
        import gzip
        import shutil
        import tempfile

        part = scrub.canonical(str(lit))
        out = {"pieces": [], "unreadable": [], "snapshot_hits": []}
        with self.store._lock:
            out["pieces"] = self._fragment_report(self.store._get_conn(), part, "")
        for f in self._snapshot_files():
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = f
                    if f.name.endswith(".gz"):
                        path = Path(tmp) / "snap.db"
                        with gzip.open(f, "rb") as src, open(path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    what = self._raw_scan(Path(path).read_bytes(), lit)
                    if what:
                        out["snapshot_hits"].append(
                            {"snapshot": f.name, "where": f"raw bytes ({what})"})
                    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                    try:
                        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
                        for table, n in scrub.find_residue(conn, [lit]).items():
                            out["snapshot_hits"].append(
                                {"snapshot": f.name, "where": f"table {table}",
                                 "rows": n})
                        out["pieces"].extend(self._fragment_report(
                            conn, part, f"snapshot {f.name} "))
                    finally:
                        conn.close()
            except (OSError, EOFError, zlib.error, sqlite3.Error, ValueError):
                out["unreadable"].append(str(f))
        return out

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
