"""Claim Sovereignty Layer — Leg 2.5d: measure the claim classifier in isolation.

Runs the rule-based classifier over the labeled fixtures and reports the confusion
matrix + the numbers that decide whether Leg 3 may consume claim_type:

  * CONFIDENT ERROR RATE — classified-but-wrong over all answerable items. The
    safety metric: a confident wrong type is what causes confident wrong
    supersession. This is the number to hold a threshold against.
  * coverage — fraction the classifier commits to (classified). Conservative
    design trades coverage for safety; abstention (low_confidence/unclassified)
    is NOT counted as error.
  * paraphrase vs canonical accuracy — the stress test. A rule classifier is
    expected to bleed on paraphrases; the gap is reported honestly, not hidden.
  * trap pass rate — the invariant cases (negative sentiment != contradiction).
  * compound detection — did multi-claim turns get flagged for candidate_only?

Run: python -m revien_bench.measure_classifier
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from revien.claims import ClassificationStatus
from revien.ingestion.claim_classifier import ClaimClassifier

_FIX = Path(__file__).resolve().parent / "fixtures" / "claim_fixtures.json"


def _load() -> List[dict]:
    return json.loads(_FIX.read_text(encoding="utf-8"))


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def measure(fixtures: Optional[List[dict]] = None) -> Dict:
    fixtures = fixtures if fixtures is not None else _load()
    clf = ClaimClassifier()

    # Partition the corpus by what each item is testing.
    compound_items = [f for f in fixtures if f.get("compound")]
    unclass_items = [f for f in fixtures if f.get("type") is None and not f.get("compound")]
    typed_items = [f for f in fixtures if f.get("type") is not None and not f.get("compound")]

    # ── Type scoring on the typed (non-compound) items ────────────────────────
    committed = correct = confident_wrong = 0
    dur_committed = dur_correct = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    confident_errors: List[tuple] = []
    subset = defaultdict(lambda: {"n": 0, "committed": 0, "correct": 0, "confident_wrong": 0})

    def tagset(f):
        out = ["paraphrase"] if "paraphrase" in f["tags"] else (["canonical"] if "canonical" in f["tags"] else [])
        for t in ("boundary", "protected", "trap", "hard", "durability_override", "locomo"):
            if t in f["tags"]:
                out.append(t)
        return out or ["other"]

    for f in typed_items:
        res = clf.classify(f["text"])
        true_t = f["type"]
        pred_t = res.claim_type.value if res.claim_type else None
        is_classified = res.classification_status is ClassificationStatus.CLASSIFIED
        col = pred_t if is_classified else res.classification_status.value
        confusion[true_t][col] += 1

        buckets = tagset(f)
        for b in buckets:
            subset[b]["n"] += 1

        if is_classified:
            committed += 1
            for b in buckets:
                subset[b]["committed"] += 1
            if pred_t == true_t:
                correct += 1
                for b in buckets:
                    subset[b]["correct"] += 1
                # durability scored only where type was committed correctly
                dur_committed += 1
                if res.durability.value == f["durability"]:
                    dur_correct += 1
            else:
                confident_wrong += 1
                confident_errors.append((f["text"], true_t, pred_t, buckets))
                for b in buckets:
                    subset[b]["confident_wrong"] += 1

    # ── Unclassifiable items: correct iff NOT confidently classified ──────────
    unclass_ok = sum(
        1 for f in unclass_items
        if clf.classify(f["text"]).classification_status is not ClassificationStatus.CLASSIFIED
    )
    unclass_false_commit = [
        f["text"] for f in unclass_items
        if clf.classify(f["text"]).classification_status is ClassificationStatus.CLASSIFIED
    ]

    # ── Compound items: correct iff the compound flag fired ──────────────────
    compound_ok = sum(1 for f in compound_items if clf.classify(f["text"]).compound)
    compound_missed = [f["text"] for f in compound_items if not clf.classify(f["text"]).compound]

    n_typed = len(typed_items)
    report = {
        "n_fixtures": len(fixtures),
        "n_typed": n_typed,
        "type_coverage": (committed, n_typed),
        "type_accuracy_committed": (correct, committed),
        "confident_error_rate": (confident_wrong, n_typed),
        "durability_accuracy_committed": (dur_correct, dur_committed),
        "unclassifiable_correct": (unclass_ok, len(unclass_items)),
        "compound_detected": (compound_ok, len(compound_items)),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "subset": {k: dict(v) for k, v in subset.items()},
        "confident_errors": confident_errors,
        "unclass_false_commit": unclass_false_commit,
        "compound_missed": compound_missed,
    }
    return report


def _print(r: Dict) -> None:
    print("\n===== Leg 2.5d — claim classifier confusion / error report =====")
    print(f"fixtures: {r['n_fixtures']}  (typed non-compound: {r['n_typed']})")
    cw, nt = r["confident_error_rate"]
    cov_n, cov_d = r["type_coverage"]
    acc_n, acc_d = r["type_accuracy_committed"]
    dn, dd = r["durability_accuracy_committed"]
    print("\n-- headline --")
    print(f"  CONFIDENT ERROR RATE : {cw}/{nt}  ({_pct(cw, nt)})   <- the safety number")
    print(f"  coverage (committed) : {cov_n}/{cov_d}  ({_pct(cov_n, cov_d)})")
    print(f"  accuracy WHEN committed: {acc_n}/{acc_d}  ({_pct(acc_n, acc_d)})")
    print(f"  durability acc (committed+right type): {dn}/{dd}  ({_pct(dn, dd)})")
    uo_n, uo_d = r["unclassifiable_correct"]
    cp_n, cp_d = r["compound_detected"]
    print(f"  unclassifiable held back: {uo_n}/{uo_d}  ({_pct(uo_n, uo_d)})")
    print(f"  compound flagged       : {cp_n}/{cp_d}  ({_pct(cp_n, cp_d)})")

    print("\n-- accuracy by subset (n / committed / correct / confident-wrong) --")
    order = ["canonical", "paraphrase", "hard", "boundary", "protected",
             "durability_override", "trap", "locomo"]
    for k in order:
        s = r["subset"].get(k)
        if not s:
            continue
        print(f"  {k:20s} n={s['n']:>3} committed={s['committed']:>3} "
              f"correct={s['correct']:>3} confident_wrong={s['confident_wrong']:>2}")

    print("\n-- confusion (true -> predicted / abstention) --")
    for true_t in sorted(r["confusion"]):
        row = r["confusion"][true_t]
        cells = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {true_t:20s} -> {cells}")

    if r["confident_errors"]:
        print(f"\n-- CONFIDENT ERRORS ({len(r['confident_errors'])}) — fix or accept --")
        for text, true_t, pred_t, buckets in r["confident_errors"]:
            print(f"  [{true_t} -> {pred_t}] ({','.join(buckets)})  {text!r}")
    if r["unclass_false_commit"]:
        print(f"\n-- vague turns wrongly committed ({len(r['unclass_false_commit'])}) --")
        for t in r["unclass_false_commit"]:
            print(f"  {t!r}")
    if r["compound_missed"]:
        print(f"\n-- compound turns missed ({len(r['compound_missed'])}) --")
        for t in r["compound_missed"]:
            print(f"  {t!r}")


def main() -> int:
    _print(measure())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
