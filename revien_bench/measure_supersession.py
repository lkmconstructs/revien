"""Claim Sovereignty Layer — Leg 3: measure the supersession gate's AUTO-FIRE RATE.

The auto-fire rate is the gate that answers "rule-based enough, or hybrid backend
before launch": of the supersession decisions that SHOULD happen (a real
contradiction needing resolution), what fraction auto-fire vs route to human
review (candidate) vs version (iron-grip/historical)?

Reports, on the labeled claim-pair fixtures:
  * AUTO-FIRE RATE — auto / real-contradiction decisions (the headline).
  * the disposition split — auto / candidate / version / missed.
  * UNSAFE AUTO-FIRES — auto on a pair that should NOT have (the dangerous,
    silent failure). Must be ZERO; listed if not.
  * decision accuracy vs the labeled expectation.
  * the scenario mix (the auto-fire rate is conditional on it — stated, not hidden).

Run: python -m revien_bench.measure_supersession
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from revien.claims import ClassificationStatus
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, Lock, SupersessionAction, SupersessionGate

_FIX = Path(__file__).resolve().parent / "fixtures" / "supersession_fixtures.json"


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def measure() -> Dict:
    fixtures = json.loads(_FIX.read_text(encoding="utf-8"))
    clf = ClaimClassifier()
    gate = SupersessionGate()

    actions = Counter()
    unsafe_autos: List[dict] = []
    missed: List[dict] = []
    wrong: List[dict] = []
    rows = []
    # "Engaged" = both claims confidently classified, so the gate could actually
    # reason about scope/contradiction. Decisions where a claim abstained are a
    # CLASSIFIER-coverage miss, not a gate-logic outcome — separated so gate logic
    # is measured on its own.
    engaged_total = engaged_correct = engaged_contra = engaged_auto = 0

    for f in fixtures:
        existing = Claim(f["existing"], clf.classify(f["existing"]),
                         Lock(f.get("existing_lock", "normal")))
        new = Claim(f["new"], clf.classify(f["new"]))
        dec = gate.evaluate(existing, new)
        act = dec.action.value
        actions[act] += 1
        expected = f["expected"]
        engaged = (existing.result.classification_status is ClassificationStatus.CLASSIFIED
                   and new.result.classification_status is ClassificationStatus.CLASSIFIED)

        rows.append((f, dec))
        if engaged:
            engaged_total += 1
            if act == expected:
                engaged_correct += 1
            if f["contradiction"]:
                engaged_contra += 1
                if act == "auto_supersede":
                    engaged_auto += 1
        if act != expected:
            wrong.append({"existing": f["existing"], "new": f["new"],
                          "expected": expected, "got": act, "reason": dec.reason,
                          "tags": f["tags"], "engaged": engaged})
        # Unsafe auto: fired when the label says it must not.
        if act == "auto_supersede" and expected != "auto_supersede":
            unsafe_autos.append({"existing": f["existing"], "new": f["new"],
                                 "expected": expected, "tags": f["tags"]})
        # Missed: a real contradiction that the gate didn't act on at all.
        if f["contradiction"] and act == "no_conflict":
            missed.append({"existing": f["existing"], "new": f["new"], "tags": f["tags"]})

    contradiction_decisions = [f for f in fixtures if f["contradiction"]]
    n_contra = len(contradiction_decisions)
    n_auto = actions["auto_supersede"]
    n_candidate = actions["candidate"]
    n_version = actions["version_locked"]
    n_missed = len(missed)
    n_correct = len(fixtures) - len(wrong)

    return {
        "n_fixtures": len(fixtures),
        "n_contradiction": n_contra,
        "auto_fire_rate": (n_auto, n_contra),
        "disposition": {
            "auto_supersede": n_auto,
            "candidate": n_candidate,
            "version_locked": n_version,
            "missed_no_conflict": n_missed,
        },
        "unsafe_autos": unsafe_autos,
        "missed": missed,
        "wrong": wrong,
        "decision_accuracy": (n_correct, len(fixtures)),
        "gate_engaged": (engaged_total, len(fixtures)),
        "gate_logic_accuracy": (engaged_correct, engaged_total),
        "engaged_auto_fire": (engaged_auto, engaged_contra),
        "scenario_mix": dict(Counter(t for f in fixtures for t in f["tags"]
                                     if t in ("auto", "protected", "irongrip",
                                              "historical", "stable", "scope_guard",
                                              "compatible", "precond2", "known_miss"))),
        "rows": rows,
    }


def _print(r: Dict) -> None:
    print("\n===== Leg 3 — supersession gate: auto-fire measurement =====")
    print(f"fixtures: {r['n_fixtures']}  (real-contradiction decisions: {r['n_contradiction']})")

    af_n, af_d = r["auto_fire_rate"]
    print("\n-- headline --")
    print(f"  AUTO-FIRE RATE : {af_n}/{af_d}  ({_pct(af_n, af_d)})   <- of real supersession decisions")
    d = r["disposition"]
    print(f"  disposition    : auto={d['auto_supersede']}  candidate={d['candidate']}  "
          f"version={d['version_locked']}  missed={d['missed_no_conflict']}")
    ua = r["unsafe_autos"]
    print(f"  UNSAFE AUTO-FIRES (must be 0): {len(ua)}   <- the silent-corruption guard")
    da_n, da_d = r["decision_accuracy"]
    print(f"  decision accuracy: {da_n}/{da_d}  ({_pct(da_n, da_d)})")
    ge_n, ge_d = r["gate_engaged"]
    gl_n, gl_d = r["gate_logic_accuracy"]
    ea_n, ea_d = r["engaged_auto_fire"]
    print("\n-- gate logic vs classifier coverage (the diagnosis) --")
    print(f"  gate ENGAGED (both classified): {ge_n}/{ge_d}  ({_pct(ge_n, ge_d)})  "
          f"<- rest are classifier-coverage misses, not gate decisions")
    print(f"  gate-logic accuracy (engaged) : {gl_n}/{gl_d}  ({_pct(gl_n, gl_d)})  "
          f"<- is the gate ROUTING right on what it sees")
    print(f"  auto-fire among engaged contras: {ea_n}/{ea_d}  ({_pct(ea_n, ea_d)})")

    print("\n-- scenario mix (auto-fire rate is conditional on this) --")
    for k, v in sorted(r["scenario_mix"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v}")

    if ua:
        print("\n-- !!! UNSAFE AUTO-FIRES — auto on a pair that must not !!! --")
        for u in ua:
            print(f"  [{u['expected']} -> auto] ({','.join(u['tags'])})  "
                  f"{u['existing']!r} <= {u['new']!r}")
    if r["missed"]:
        print(f"\n-- missed contradictions ({len(r['missed'])}) — safe (old claim stands), detector ceiling --")
        for m in r["missed"]:
            print(f"  ({','.join(m['tags'])})  {m['existing']!r} <= {m['new']!r}")
    if r["wrong"]:
        print(f"\n-- decision mismatches vs label ({len(r['wrong'])}) --")
        for w in r["wrong"]:
            print(f"  [{w['expected']} -> {w['got']}] ({w['reason']}) ({','.join(w['tags'])})")
            print(f"      {w['existing']!r} <= {w['new']!r}")


def main() -> int:
    _print(measure())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
