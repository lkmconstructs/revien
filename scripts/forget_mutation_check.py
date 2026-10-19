"""Mutation check for the forget work (docs/FORGET_SPEC.md §9, invariant 6).

Each mutation removes or breaks ONE piece of guarding code in a throwaway copy
of the repository, then runs the tests that are meant to guard it. A mutation
that the tests do not catch (the targeted tests still pass) is a hole in the
tests: this script exits non-zero if there is one.

    python scripts/forget_mutation_check.py [name ...]

Run it with the clean test environment (the one that imports revien from the
checkout, not from somewhere else). Nothing in the working tree is touched.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

INV = "tests/test_forget_invariants.py"
ENG = "tests/test_forget_engine.py"
SUP = "tests/test_forget_suppression.py"
VERBS = "tests/test_forget_verbs.py"
RESTORE = "tests/test_forget_restore.py"
UP = "tests/test_forget_upgrade.py"
COV = "tests/test_forget_coverage.py"
LOG = "tests/test_forget_changelog.py"
REV = "tests/test_forget_review.py"

# (name, invariant, file, old, new, [pytest targets])
M = [
    ("audit-snapshots-not-scrubbed", 1, "revien/forget/engine.py",
     "            stats[\"audit_rows_scrubbed\"] = self._scrub_audit(\n                conn, set(remove_ids), set(changed_ids), all_strings)",
     "            stats[\"audit_rows_scrubbed\"] = 0",
     [INV + " -k pipeline", ENG + " -k Unicode"]),
    ("usage-queries-not-cleared", 1, "revien/forget/engine.py",
     "            q = self._scrub_queries(conn, episodes, receipt_ids, all_strings)",
     "            q = {}",
     [INV + " -k pipeline"]),
    ("candidates-not-scrubbed", 1, "revien/forget/engine.py",
     "            stats[\"candidates_scrubbed\"] = self._scrub_candidates(conn, all_strings)",
     "            stats[\"candidates_scrubbed\"] = 0",
     [INV + " -k pipeline"]),
    ("vectors-not-purged", 1, "revien/forget/engine.py",
     "        problems = self._purge_vectors(gone + changed_ids)",
     "        problems = None",
     [INV + " -k pipeline"]),
    ("training-db-not-scrubbed", 1, "revien/forget/engine.py",
     "        training = self._scrub_training(gone, strings)",
     "        training = {\"checked\": 0, \"rows_removed\": 0, \"problems\": []}",
     [INV + " -k pipeline"]),
    ("storage-not-scrubbed", 1, "revien/forget/engine.py",
     "        storage = store.scrub_storage()",
     "        storage = {\"secure_delete\": True, \"wal_truncated\": True, \"vacuumed\": True, \"note\": \"\"}",
     [INV + " -k pipeline"]),
    ("secure-delete-off", 1, "revien/graph/store.py",
     "            self._conn.execute(\"PRAGMA secure_delete = ON\")",
     "            pass",
     [VERBS + " -k secure_delete"]),
    ("wording-not-cut-from-context", 2, "revien/forget/engine.py",
     "        plan.holders = list(holders.values())",
     "        plan.holders = []",
     [ENG + " -k SingleNode"]),
    ("vouched-node-removed", 2, "revien/forget/engine.py",
     "        if full:\n            return \"keep\", full",
     "        if full:\n            return \"remove\", full",
     [ENG + " -k vouched", ENG + " -k Shared"]),
    ("name-only-node-removed", 2, "revien/forget/engine.py",
     "        if name:\n            return \"cut_back\", name",
     "        if name:\n            return \"remove\", name",
     [ENG + " -k name_only"]),
    ("unproven-node-removed", 2, "revien/forget/engine.py",
     "        if not rows or any(is_unproven(r[\"unit_id\"]) for r in rows):\n            return \"unproven\", []",
     "        if False:\n            return \"unproven\", []",
     [ENG + " -k Unproven", UP]),
    ("unproven-backfilled-as-proven", 7, "revien/graph/forget_store.py",
     "        if node_type in (\"context\", \"skill\") and ikey and likely:\n            return likely, 1, 1, likely",
     "        if likely:\n            return likely, 1, 1, likely",
     [UP]),
    ("gate-disabled", 3, "revien/forget/gate.py",
     "    records = store.list_suppressions(\"active\", include_wording=True)\n    if not records:\n        return None\n    fp = fingerprint(content)",
     "    return None\n    records = []\n    fp = fingerprint(content)",
     [SUP + " -k 'recaptured or File or Permanent or Other'"]),
    ("name-level-match-lifts", 3, "revien/forget/gate.py",
     "            return Block(rec, \"no evidence the file changed\")",
     "            continue",
     [SUP + " -k 'made_up_path or other_real_file or case_changed'"]),
    ("inode-not-recorded", 3, "revien/graph/units.py",
     "        if st.st_ino:\n            keys.append(f\"inode:{st.st_dev}:{st.st_ino}\")",
     "        pass",
     [SUP + " -k 'hard_link or moved'"]),
    ("case-folding-dropped", 3, "revien/graph/units.py",
     "    return os.path.normcase(real_path(path)).casefold()",
     "    return real_path(path)",
     [SUP + " -k case_changed"]),
    ("file-change-ignored-timestamp-lifts", 3, "revien/forget/gate.py",
     "            if current is None or current == rec[\"file_fp\"]:",
     "            if current is None:",
     [SUP + " -k 'unchanged or timestamp'"]),
    ("forgotten-sentences-come-back-when-the-file-grows", 3, "revien/ingestion/pipeline.py",
     "            input_data.content, _left_out = _gate.strip_forgotten(\n                self.store, unit, input_data.content)",
     "            _left_out = 0",
     [SUP + " -k 'growing or new_evidence'"]),
    ("pipeline-skips-the-gate", 3, "revien/ingestion/pipeline.py",
     "        blocked = self._gate_check(input_data, unit)\n        if blocked is not None:\n            return self._suppressed_output(blocked, fenced_spans)\n        # A source that changed",
     "        # A source that changed",
     [SUP, COV + " -k pipeline"]),
    ("openai-adapter-skips-the-gate", 3, "revien/adapters/openai_adapter.py",
     "        block = forget_gate.check(self.store, unit, \"\\n\".join(texts))",
     "        block = None",
     [SUP + " -k openai", COV + " -k openai"]),
    ("graph-import-skips-the-gate", 3, "revien/graph/store.py",
     "            if (forget_gate.check(self, ref, node.content) is not None\n                    or forget_gate.check(\n                        self, forget_gate.UnitRef(None), node.label) is not None):",
     "            if False:",
     [SUP + " -k graph_import", COV + " -k import_graph"]),
    ("skills-skip-the-gate", 3, "revien/skills/ingest.py",
     "        if forget_gate.check(store, unit, fresh.content) is not None:",
     "        if False:",
     [SUP + " -k skill", COV + " -k skill"]),
    ("reconcile-skips-the-gate", 3, "revien/distill.py",
     "        if forget_gate.check(self.store, unit, text) is not None:\n            return None\n        if forget_gate.strip_forgotten(self.store, unit, text)[1]:\n            return None",
     "        pass",
     [SUP + " -k distilled"]),
    ("proposals-skip-the-gate", 3, "revien/skills/proposals.py",
     "            if _pattern_suppressed(store, pattern):",
     "            if False:",
     [SUP + " -k proposals"]),
    ("shared-channel-gets-suppressed", 3, "revien/forget/engine.py",
     "            if is_shared_channel(unit_id):\n                specs.append({\"unit_id\": unit_id, \"stable\": False})\n                continue",
     "            if False:\n                continue",
     [SUP + " -k shared", INV + " -k pipeline"]),
    ("sweep-becomes-forget", 4, "revien/graph/operations.py",
     "                if self.delete_node(nid, actor=actor, op=op):",
     "                if self.forget_node(nid, actor=actor).get(\"count\"):",
     [VERBS + " -k sweep", "tests/test_graph.py -k Retention"]),
    ("delete-is-recorded-as-forget", 4, "revien/graph/operations.py",
     "                node_id, op, actor=actor, before=None,\n                after={\"original_node_id\": node_id, \"verb\": op,",
     "                node_id, \"forget\", actor=actor, before=None,\n                after={\"original_node_id\": node_id, \"verb\": op,",
     [VERBS + " -k delete", "tests/test_graph.py -k Retention"]),
    ("cleanup-is-recorded-as-delete", 4, "revien/graph/operations.py",
     "        return self._retention_sweep(mode, days, op=\"cleanup\",",
     "        return self._retention_sweep(mode, days, op=\"delete\",",
     [VERBS + " -k cleanup", "tests/test_graph.py -k scheduled"]),
    ("rest-delete-suppresses", 4, "revien/daemon/server.py",
     "        deleted = ops.delete_node(node_id, actor=\"api:delete\")",
     "        deleted = bool(ops.forget_node(node_id, actor=\"api:delete\").get(\"count\"))",
     [VERBS + " -k delete"]),
    ("old-daemon-heartbeat-ignored-by-rest", 8, "revien/daemon/server.py",
     "        old = daemons_lacking(store, FORGET_REQUIRES, exclude_pid=os.getpid())",
     "        old = []",
     [VERBS + " -k rest_route_refuses"]),
    ("old-daemon-ignored-by-cli", 8, "revien/cli.py",
     "    old = _old_daemon_problem(db_path, daemon_url,\n                              need_features=FORGET_REQUIRES, what=\"forget\")",
     "    old = None",
     [VERBS + " -k Refuses"]),
    ("sync-reset-needs-no-yes", 8, "revien/cli.py",
     "    if forgets and not yes:",
     "    if False:",
     [VERBS + " -k needs_yes"]),
    ("late-forget-not-enforced-after-the-fresh-write", 8, "revien/ingestion/pipeline.py",
     "        _undone = _gate.enforce_after_write(\n            self.store, unit, input_data.content,\n            [nid for nid, _l, _c in newly_created],\n            [v for v in",
     "        _undone = (lambda *a, **k: None)(\n            self.store, unit, input_data.content,\n            [nid for nid, _l, _c in newly_created],\n            [v for v in",
     [REV + " -k 'lands_after or mid_capture'"]),
    ("late-forget-written-over-refresh", 8, "revien/ingestion/pipeline.py",
     "        with self.store._capture_lock, self.store.transaction():\n            blocked = self._gate_check(input_data, unit)\n            if blocked is not None:\n                return self._suppressed_output(blocked, fenced_spans)\n            _gate.register(self.store, unit)",
     "        with self.store._capture_lock, self.store.transaction():\n            _gate.register(self.store, unit)",
     [SUP + " -k 'Mid and keyed'"]),
    ("scheduler-counts-suppressed-as-ingested", 8, "revien/daemon/scheduler.py",
     "                if getattr(out, \"suppressed\", False):",
     "                if False:",
     [VERBS + " -k started_before"]),
    ("restore-bare-name-fans-out", 7, "revien/forget/restore.py",
     "    cands = [r for r in records if base and (",
     "    return {\"status\": \"ok\", \"mode\": \"single\", \"lifts\": [_summarize(r) for r in records if base and (os.path.basename(r[\"path\"] or \"\").casefold() == base)]}\n    cands = [r for r in records if base and (",
     [RESTORE + " -k bare_name"]),
    ("restore-pattern-not-listed-first", 7, "revien/cli.py",
     "        if plan[\"mode\"] == \"bulk\" and not yes:\n            if not click.confirm(\"Lift all of these?\"):",
     "        if False:\n            if not click.confirm(\"Lift all of these?\"):",
     [RESTORE + " -k lists_what"]),
    ("restore-not-audited", 7, "revien/forget/restore.py",
     "            _audit_lift(store, rec, actor, plan[\"mode\"])",
     "            pass",
     [RESTORE + " -k audited"]),
    ("refusal-removed", 2, "revien/forget/engine.py",
     "            plan.refusal = {\n                \"reason\": \"wording_not_found\",",
     "            plan.refusal = None\n            _unused = {\n                \"reason\": \"wording_not_found\",",
     [ENG + " -k Refusal", VERBS + " -k refusal"]),
    ("owner-not-recorded-at-write", 2, "revien/graph/store.py",
     "            if owner is not None:\n                unit_id, label_ok, wording_ok = owner\n                self._record_provenance_stmt(\n                    conn, node.node_id, unit_id, label_ok, wording_ok)",
     "            pass",
     [ENG, INV + " -k pipeline"]),
    ("merge-not-recorded", 2, "revien/ingestion/dedup.py",
     "        if unit_id is None:\n            return\n        same_words",
     "        return\n        same_words",
     [ENG + " -k 'vouched or merge'"]),
    ("fingerprint-is-a-plain-hash", 3, "revien/forget/keys.py",
     "    return hmac.new(key, normalize_text(text), hashlib.sha256).hexdigest()",
     "    return hashlib.sha256(normalize_text(text)).hexdigest()",
     [SUP + " -k ContentFree"]),
    ("counts-do-not-match-the-database", 5, "revien/forget/engine.py",
     "            \"removed_nodes\": len(removed),",
     "            \"removed_nodes\": len(removed) + 1,",
     [INV + " -k pipeline"]),
    ("message-claims-gone-everywhere-with-a-vault-note", 5, "revien/forget/engine.py",
     "        scope = \"gone everywhere\" if not leftovers and not remaining \\\n            else \"forgotten from Revien\"",
     "        scope = \"gone everywhere\"",
     [INV + " -k pipeline", SUP + " -k distilled"]),
    # ── the review round: word bounds, other units, every field, the rule ──
    ("matching-is-substring", 1, "revien/forget/scrub.py",
     "    return pre + re.escape(s) + post",
     "    return re.escape(s)",
     [REV + " -k 'WordBounded or word_bounded'"]),
    ("unproven-node-is-rewritten", 2, "revien/forget/engine.py",
     "                unproven[n.node_id] = n\n        everywhere_holders",
     "                legacy_cut[n.node_id] = n\n        everywhere_holders",
     [REV + " -k UnprovenIsLeftAlone"]),
    ("audit-actor-not-scrubbed", 1, "revien/forget/engine.py",
     "                nact, cact = scrub.cut_text(actor or \"\", strings)",
     "                nact, cact = actor, 0",
     [REV + " -k merge_entrys_actor"]),
    ("event-time-text-not-cut", 1, "revien/forget/engine.py",
     "        n_ett, a = scrub.cut_text(ett or \"\", strings)",
     "        n_ett, a = ett, 0",
     [REV + " -k event_time_text"]),
    ("source-context-not-cut", 1, "revien/forget/engine.py",
     "        n_sctx, b = scrub.cut_text(sctx or \"\", strings)",
     "        n_sctx, b = sctx, 0",
     [REV + " -k source_context"]),
    ("metadata-not-cut", 1, "revien/forget/engine.py",
     "        n_meta, c = scrub.cut_json_text(meta or \"{}\", strings)",
     "        n_meta, c = meta, 0",
     [REV + " -k 'cut_reaches and metadata'"]),
    ("cut-nodes-keep-their-vectors", 1, "revien/forget/engine.py",
     "        problems = self._purge_vectors(gone + changed_ids)",
     "        problems = self._purge_vectors(gone)",
     [REV + " -k vector_of_a_cut_node"]),
    ("watch-snapshots-not-scrubbed", 1, "revien/forget/engine.py",
     "        snapshots = self._scrub_snapshots(\n            remove_ids, set(changed_ids), strings, episodes)",
     "        snapshots = {\"scrubbed\": 0, \"unreadable\": []}",
     [REV + " -k Snapshots"]),
    ("provenance-outlives-its-node", 2, "revien/graph/store.py",
     "            conn.execute(\n                \"DELETE FROM node_provenance WHERE node_id = ?\", (node_id,))",
     "            pass",
     [REV + " -k provenance_goes"]),
    ("denylist-not-checked-in-the-write-transaction", 3, "revien/forget/gate.py",
     "    with store.transaction(immediate=True) as conn:\n        block = check(store, unit, content)",
     "    return None\n    with store.transaction(immediate=True) as conn:\n        block = check(store, unit, content)",
     [REV + " -k 'lands_after or mid_capture'"]),
    ("denylist-is-per-source-only", 3, "revien/forget/gate.py",
     "        global_fps |= set(rec[\"sentence_fps\"])\n        file_level",
     "        file_level",
     [REV + " -k 'copy_of_the_forgotten or every_other_writer'"]),
    ("short-fragments-not-bound-to-identity", 3, "revien/forget/gate.py",
     "            scoped |= set(rec[\"short_fps\"])",
     "            pass",
     [REV + " -k short_fragment"]),
    ("plain-forget-reaches-other-units", 2, "revien/forget/engine.py",
     "        everywhere_holders: Dict[str, Node] = {}\n        if everywhere:",
     "        everywhere_holders: Dict[str, Node] = {}\n        if True:",
     [REV + " -k 'per_source'"]),
    ("everywhere-reaches-nothing", 2, "revien/forget/engine.py",
     "        if everywhere:\n            # Every unit that holds the wording, not only this node's source.",
     "        if False:\n            # Every unit that holds the wording, not only this node's source.",
     [REV + " -k everywhere_reaches"]),
    ("model-not-marked-stale", 1, "revien/forget/engine.py",
     "            store.set_meta(\"neural_model_stale\", now.isoformat())",
     "            pass",
     [REV + " -k marked_stale"]),
    ("changelog-names-a-command-that-is-gone", 5, "revien/cli.py",
     "@main.command()\n@click.argument(\"target\", required=False)",
     "@main.command(name=\"restore-gone\")\n@click.argument(\"target\", required=False)",
     [LOG]),
]


def run(name, inv, rel, old, new, targets, root):
    work = Path(tempfile.mkdtemp(prefix="revien_mut_"))
    try:
        shutil.copytree(root / "revien", work / "revien")
        shutil.copytree(root / "tests", work / "tests")
        for extra in ("scripts", "docs", "CHANGELOG.md", "README.md", "GATES.md", "setup.py"):
            src = root / extra
            if src.is_dir():
                shutil.copytree(src, work / extra)
            elif src.exists():
                shutil.copy(src, work / extra)
        target = work / rel
        text = target.read_text(encoding="utf-8")
        if old not in text:
            return "STALE", f"snippet not found in {rel}"
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        failed = []
        for t in targets:
            cmd = [sys.executable, "-m", "pytest", *shlex.split(t), "-q", "-x",
                   "-p", "no:cacheprovider", "--basetemp", str(work / "bt")]
            res = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if res.returncode == 5:
                return "NO-TESTS", f"{t} collected nothing"
            if res.returncode != 0:
                failed.append(t)
        return ("CAUGHT" if failed else "SURVIVED"), ", ".join(failed)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    wanted = set(sys.argv[1:])
    survivors = 0
    for name, inv, rel, old, new, targets in M:
        if wanted and name not in wanted:
            continue
        status, detail = run(name, inv, rel, old, new, targets, REPO)
        print(f"{status:9s} inv{inv} {name:52s} {detail}")
        sys.stdout.flush()
        if status != "CAUGHT":
            survivors += 1
    print(f"\n{survivors} mutation(s) not caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
