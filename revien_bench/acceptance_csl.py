"""Claim Sovereignty Layer — Leg B END-TO-END acceptance.

Not 340 probes. Real claims through the WHOLE assembled path on the live ingest
pipeline, watched end-to-end, reporting what physically happened to the data.

For each scenario: a fresh graph, ingest the claims in order through
IngestionPipeline (csl wired), then inspect the data —
  SUPERSEDED  : the first claim's node is soft-invalidated (historical)
  QUEUED      : a candidate row references it; BOTH claims still live
  PRESERVED   : neither; both claims stand

SAFETY INVARIANTS (must hold, exit non-zero if violated):
  a sensitive or protected claim is NEVER auto-superseded (never silently erased).

Recognizer: resolved from env (REVIEN_SENSITIVITY_BACKEND + key) — qwen/cloud for
us. Without it the gate runs sensitive-floor + distrust-tripwire (interim). The
report names which layer caught each sensitive case, so defense-in-depth is
MEASURED, not assumed.

Usage:
  python -m revien_bench.acceptance_csl                      # floor/scope/tripwire (no recognizer)
  REVIEN_SENSITIVITY_BACKEND=openrouter REVIEN_SENSITIVITY_MODEL=qwen/... \
    OPENROUTER_API_KEY=... python -m revien_bench.acceptance_csl   # full, recognizer wired
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

# Keep the acceptance focused on CSL: no semantic embedding on ingest.
os.environ.setdefault("REVIEN_SEMANTIC", "0")

from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.pipeline import IngestionPipeline, IngestionInput
from revien.ingestion.supersession_ingest import build_governor

SUPERSEDED, QUEUED, PRESERVED = "SUPERSEDED", "QUEUED", "PRESERVED"


def _fresh():
    path = tempfile.mktemp(suffix=".db", prefix="csl_accept_")
    store = GraphStore(path)
    ops = GraphOperations(store)
    governor = build_governor(store, ops)  # env recognizer (or floor+tripwire)
    pipe = IngestionPipeline(store, csl=governor)
    return store, governor, pipe


def _ingest(pipe, text):
    out = pipe.ingest(IngestionInput(
        source_id="accept", content=text, timestamp=datetime.now(timezone.utc)))
    return out.context_node_id, out.governance


def run_scenario(name, claims, expect):
    store, governor, pipe = _fresh()
    print(f"\n{'='*78}\n{name}\n{'-'*78}")
    first_id = None
    all_outcomes = []
    for i, text in enumerate(claims):
        nid, outcomes = _ingest(pipe, text)
        if i == 0:
            first_id = nid
        tag = "  (claim 1)" if i == 0 else ""
        print(f"  ingest: {text!r}{tag}")
        for o in outcomes:
            print(f"     -> {o.action.upper()}: {o.reason}")
            print(f"        effect: {o.data_effect}")
            print(f"        trace : {' | '.join(o.trace)}")
            all_outcomes.append(o)

    node = store.get_node(first_id)
    superseded = node is not None and node.invalidated_at is not None
    cands = [c for c in store.list_candidates() if c["existing_node_id"] == first_id]
    if superseded:
        got = SUPERSEDED
    elif cands:
        got = QUEUED
    else:
        got = PRESERVED

    # Which layer acted (for the sensitive cases).
    layer = ""
    for o in all_outcomes:
        if "semantic_sensitive" in o.reason:
            layer = "  [caught by: SEMANTIC RECOGNIZER]"
        elif o.reason.startswith("tripwire_distrust"):
            layer = "  [caught by: DISTRUST TRIPWIRE]"
        elif "sensitive_floor" in " ".join(o.trace):
            layer = "  [caught by: SENSITIVE FLOOR]"
    if got == PRESERVED and not layer:
        # floor fires as NO_CONFLICT (no outcome emitted) — detect via re-classify note
        layer = "  [preserved: no scoped contradiction or floored]"

    ok = got == expect
    verdict = "PASS" if ok else "MISMATCH"
    print(f"  RESULT: claim 1 is {got}  (expected {expect})  -> {verdict}{layer}")
    return {"name": name, "expect": expect, "got": got, "ok": ok,
            "safety": expect in (QUEUED, PRESERVED)}


def main() -> int:
    rec = build_governor(GraphStore(tempfile.mktemp(suffix=".db")),  # probe only
                         GraphOperations(GraphStore(tempfile.mktemp(suffix=".db")))).gate.recognizer
    mode = (f"recognizer WIRED ({rec.backend}/{rec.model})" if rec is not None
            else "recognizer NOT wired -> sensitive floor + distrust tripwire only")
    print(f"CSL Leg B acceptance — {mode}")

    scenarios = [
        # (name, [claims in order], expected category for claim 1)
        ("1. Happy path — literal preference 'update' (acceptance #1)",
         ["I like tea.", "I prefer coffee now."], PRESERVED),
        ("1b. Happy path — single-valued favourite flip (auto-supersede SHOULD fire)",
         ["My favourite drink is tea.", "My favourite drink is coffee."], SUPERSEDED),
        ("2. Coded disclosure survives — residual recognizer-miss (acceptance #2)",
         ["I finally crossed the carpet last Sunday and they read my name out from the platform.",
          "Actually I'm heading back to the hall this Sunday."], PRESERVED),
        ("2b. NEGATION GAP — 'I don't love X anymore' is unclassified, so the "
         "contradiction is never detected -> preserved (SAFE, not erased)",
         ["I love being off the bottle.", "Honestly I don't love being off the bottle anymore."], PRESERVED),
        ("3. Protected claim not clobbered by frustration (acceptance #3)",
         ["I have Hashimoto's.", "I'm so done with this."], PRESERVED),
        ("3b. Protected claim, real status contradiction -> review, never auto",
         ["I'm married.", "I'm single now."], QUEUED),
        ("4. Plain neutral technical 'update' (acceptance #4)",
         ["the build passed.", "the build failed."], PRESERVED),
        ("4b. NEGATION GAP — 'I'm not swamped anymore' is unclassified, so the "
         "flip is never detected -> preserved (SAFE, but supersede stays dormant)",
         ["I'm swamped at work.", "I'm not swamped anymore."], PRESERVED),
        ("5. Misnamed-sensitive REACHES the recognizer (favourite-substitution) "
         "-> must route to review, never auto-erase",
         ["My favourite app is my sobriety tracker.", "My favourite app is TikTok now."], QUEUED),
    ]

    results = [run_scenario(*s) for s in scenarios]

    print(f"\n{'='*78}\nSUMMARY\n{'-'*78}")
    safety_breaches = []
    mismatches = []
    for r in results:
        flag = "ok " if r["ok"] else "MISMATCH"
        print(f"  [{flag}] {r['name'][:60]:60} got={r['got']:10} expect={r['expect']}")
        if not r["ok"]:
            mismatches.append(r)
        # SAFETY breach = a case that must be preserved/queued got SUPERSEDED.
        if r["safety"] and r["got"] == SUPERSEDED:
            safety_breaches.append(r)

    print(f"\n  scenarios: {len(results)}   mismatches: {len(mismatches)}   "
          f"SAFETY breaches (sensitive/protected auto-erased): {len(safety_breaches)}")
    if safety_breaches:
        print("\n  !!! SAFETY BREACH — a sensitive/protected claim was silently superseded:")
        for r in safety_breaches:
            print(f"      {r['name']}")
    if safety_breaches:
        return 2
    if mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
