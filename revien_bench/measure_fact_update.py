"""Does the memory hold the CURRENT TRUTH after an ordinary fact changes?

The real test of a memory system: state a fact, later change it, and check what
the memory now holds. For each (old, new) pair we ingest both through the live
pipeline (CSL wired) into a fresh graph and inspect the data:

  UPDATED : the old claim is retired (soft-invalidated) and the new stands.
            -> memory holds the current truth.  THIS is success.
  QUEUED  : surfaced for human review; both still live (acceptable for SENSITIVE,
            a MISS for ordinary facts — the memory is still stale until a human acts).
  STALE   : nothing happened; both claims live -> on recall the OLD fact still
            surfaces as current.  FAILURE.

Ordinary facts (non-sensitive) SHOULD end UPDATED. Sensitive controls MUST NOT be
auto-superseded (UPDATED on a sensitive control = a safety breach); QUEUED/STALE
are both safe there.

For each STALE ordinary failure we also print the gate's decision reason and the
two claims' classifications, so the fix is informed by the actual failure mode.

Usage:  python -m revien_bench.measure_fact_update
        REVIEN_SENS_FIXTURES not used here; corpus is fact_changes.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

WORKERS = max(1, int(os.environ.get("REVIEN_FU_WORKERS", "8")))

os.environ.setdefault("REVIEN_SEMANTIC", "0")

from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.pipeline import IngestionPipeline, IngestionInput
from revien.ingestion.supersession_ingest import build_governor
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, SupersessionGate

CORPUS = Path(__file__).parent / "fixtures" / "fact_changes.json"

UPDATED, QUEUED, STALE = "UPDATED", "QUEUED", "STALE"

# PRECISION guard: compatible restatements + cross-dimension statements that must
# NOT supersede (the old fact is still true). UPDATED here = a wrongful erase.
RESTATEMENTS = [
    ("I live in Boston.", "I really love living in Boston."),
    ("I work at Google as a software engineer.", "I've been at Google for three years."),
    ("I drive a Honda Civic.", "My Civic is due for an oil change."),
    ("My favourite show is Succession.", "I'm rewatching Succession, my favourite show."),
    ("My favourite coffee is a flat white.", "I had my usual flat white this morning."),
    ("I rent an apartment downtown.", "My downtown apartment has a great view."),
    ("I'm a teacher at Lincoln Elementary.", "I really love teaching at Lincoln."),
    ("I live in Denver.", "I work at Meta now."),                       # different dimensions
    ("My favourite show is Severance.", "My favourite food is ramen."),  # different fav category
    ("I drive a Tesla Model 3.", "I live in Austin now."),              # different dimensions
]


def _end_state(old: str, new: str) -> str:
    """Ingest old then new into a fresh graph; report what the data holds."""
    store = GraphStore(tempfile.mktemp(suffix=".db", prefix="fc_"))
    ops = GraphOperations(store)
    pipe = IngestionPipeline(store, csl=build_governor(store, ops))
    o1 = pipe.ingest(IngestionInput(source_id="fc", content=old,
                                    timestamp=datetime.now(timezone.utc)))
    old_id = o1.context_node_id
    pipe.ingest(IngestionInput(source_id="fc", content=new,
                               timestamp=datetime.now(timezone.utc)))
    node = store.get_node(old_id)
    if node is not None and node.invalidated_at is not None:
        return UPDATED
    if any(c["existing_node_id"] == old_id for c in store.list_candidates()):
        return QUEUED
    return STALE


def _diagnose(old: str, new: str, clf: ClaimClassifier, gate: SupersessionGate):
    co, cn = clf.classify(old), clf.classify(new)
    d = gate.evaluate(Claim(old, co), Claim(new, cn))
    fmt = lambda c: f"{c.classification_status.value}/{c.claim_type.value if c.claim_type else '-'}"
    return d.reason, fmt(co), fmt(cn)


def main() -> int:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    ordinary = data["ordinary"]
    sensitive = data.get("sensitive_controls", [])
    clf, gate = ClaimClassifier(), SupersessionGate()  # offline diagnosis (no recognizer)

    print(f"FACT-UPDATE MEASUREMENT — {len(ordinary)} ordinary changes, "
          f"{len(sensitive)} sensitive controls\n" + "=" * 78)

    by_outcome = Counter()
    by_dim = defaultdict(Counter)
    by_style = defaultdict(Counter)
    fail_reasons = Counter()
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(lambda p: _end_state(p["old"], p["new"]), ordinary))
    for p, out in zip(ordinary, outs):
        by_outcome[out] += 1
        by_dim[p["dimension"]][out] += 1
        by_style[p.get("change_style", "?")][out] += 1
        if out != UPDATED:
            reason, co, cn = _diagnose(p["old"], p["new"], clf, gate)
            fail_reasons[f"{out}:{reason}"] += 1
            failures.append((out, p["dimension"], p["old"], p["new"], reason, co, cn))

    total = len(ordinary)
    upd = by_outcome[UPDATED]
    rate = upd / total if total else 0.0

    print(f"\nORDINARY FACT-CHANGES — does memory end holding the current truth?")
    print(f"  UPDATED (current truth held): {upd}/{total}  = {rate:.0%}   <- the number that matters")
    print(f"  QUEUED  (surfaced, still stale until reviewed): {by_outcome[QUEUED]}/{total}")
    print(f"  STALE   (old fact still live, change lost):     {by_outcome[STALE]}/{total}")

    print(f"\n  by change_style:")
    for style, c in sorted(by_style.items()):
        t = sum(c.values())
        print(f"    {style:22} UPDATED {c[UPDATED]}/{t}")
    print(f"\n  by dimension:")
    for dim, c in sorted(by_dim.items()):
        t = sum(c.values())
        print(f"    {dim:22} UPDATED {c[UPDATED]}/{t}  (queued {c[QUEUED]}, stale {c[STALE]})")

    print(f"\n  failure modes (why ordinary changes did NOT update):")
    for reason, n in fail_reasons.most_common():
        print(f"    {n:3}  {reason}")

    print(f"\n  sample failures (outcome | dim | old -> new | reason | cls(old)->cls(new)):")
    for out, dim, old, new, reason, co, cn in failures[:18]:
        print(f"    [{out}] {dim}: {old!r} -> {new!r}")
        print(f"           reason={reason}  cls: {co} -> {cn}")

    # Sensitive controls: must NOT auto-erase.
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        s_outs = list(ex.map(lambda p: _end_state(p["old"], p["new"]), sensitive))
    breaches = [p for p, out in zip(sensitive, s_outs) if out == UPDATED]
    print(f"\nSENSITIVE CONTROLS — must never auto-erase: "
          f"{len(breaches)}/{len(sensitive)} breaches (want 0)")
    for p in breaches:
        print(f"    !!! BREACH: {p['old']!r} -> {p['new']!r}")

    # Precision: restatements / cross-dimension must NOT supersede.
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        r_outs = list(ex.map(lambda on: _end_state(on[0], on[1]), RESTATEMENTS))
    wrongful = [on for on, out in zip(RESTATEMENTS, r_outs) if out == UPDATED]
    print(f"\nPRECISION — restatements/independent that must NOT supersede: "
          f"{len(wrongful)}/{len(RESTATEMENTS)} wrongful erases (want 0)")
    for o, n in wrongful:
        print(f"    !!! WRONGFUL SUPERSEDE: {o!r} -> {n!r}")

    print("\n" + "=" * 78)
    print(f"HEADLINE: memory holds current truth on {rate:.0%} of ordinary fact-changes "
          f"({upd}/{total}); sensitive breaches: {len(breaches)}; wrongful erases: {len(wrongful)}")
    # Clean only if updates work, nothing sensitive erased, nothing wrongly erased.
    return 0 if (rate >= 0.90 and not breaches and not wrongful) else 1


if __name__ == "__main__":
    raise SystemExit(main())
