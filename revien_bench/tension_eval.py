"""
revien_bench.tension_eval — B1's OWN eval track (identity memory).

Standing rule: new identity-memory features get their own eval — LoCoMo and
the vault fixtures do not measure tension/COEXIST at all. This harness runs
the FULL path (rule classifier -> supersession gate -> tension recognizer)
over a small labeled pair fixture and reports:

  * coexist recall     — labeled-tension pairs that actually COEXIST
  * false-fire count   — control pairs (compatible/retraction/sentiment,
                         the CLAIM_TAXONOMY §7.2 shapes) that wrongly COEXIST
  * classifier bound   — labeled-tension pairs that never even reached the
                         hook because the RULE CLASSIFIER failed to classify
                         one side (the known CSL coverage limit — reported
                         separately so recognizer quality isn't blamed for
                         classifier misses, and vice versa)

Requires a wired recognizer: REVIEN_TENSION_BACKEND (ollama local default;
cloud backends disclose once, per house rule). No backend -> exits with the
explanation instead of printing a meaningless number.

Results land in results/tension/ — their own namespace, no overall_f1 key,
never blended with the conversational or vault tables.

Usage:
    REVIEN_TENSION_BACKEND=ollama python -m revien_bench.tension_eval
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, SupersessionAction, SupersessionGate

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
FIXTURE = _PKG_DIR / "fixtures" / "tension_pairs.json"


def _build_recognizer():
    if not os.environ.get("REVIEN_TENSION_BACKEND"):
        return None
    from revien.tension import LLMTensionRecognizer
    rec = LLMTensionRecognizer()
    return rec if rec.is_available() else None


def run_eval(out_dir: Path, fixture_path: Path = FIXTURE) -> Optional[Dict]:
    recognizer = _build_recognizer()
    if recognizer is None:
        print(
            "tension_eval needs a wired recognizer: set REVIEN_TENSION_BACKEND\n"
            "(ollama = local default; openrouter/openai/anthropic need a key\n"
            "and disclose the cloud egress once). No number without the real\n"
            "path — a stubbed recognizer here would measure nothing."
        )
        return None

    # Transport probe: a recognizer with an unreachable backend silently
    # abstains on every pair, which would print as recall 0.0 and read as a
    # recognizer-quality verdict. One probe call separates "backend down"
    # from "backend bad" before any number is printed.
    recognizer.recognize_pair("I love quiet mornings.", "I love loud parties.")
    if not recognizer.is_available():
        print(
            f"tension_eval: backend {recognizer.backend!r} is wired but the "
            f"transport FAILED on a probe call — no number printed (all "
            f"verdicts would be abstain). Start the backend (ollama serve?) "
            f"or set REVIEN_TENSION_BACKEND to a reachable one."
        )
        return None

    pairs = json.loads(fixture_path.read_text(encoding="utf-8"))["pairs"]
    classifier = ClaimClassifier()
    gate = SupersessionGate(tension_recognizer=recognizer)

    rows: List[Dict] = []
    for p in pairs:
        existing = Claim(text=p["existing"], result=classifier.classify(p["existing"]))
        new = Claim(text=p["new"], result=classifier.classify(p["new"]))
        decision = gate.evaluate(existing, new)
        coexisted = decision.action is SupersessionAction.COEXIST
        # The hook can only fire if both sides classified into the same
        # tension type — a scope_overlap of 0/0.5 means the CLASSIFIER, not
        # the recognizer, is what missed.
        reached_hook = any(t.startswith("tension") for t in decision.trace)
        rows.append({
            **p,
            "action": decision.action.value,
            "reason": decision.reason,
            "coexisted": coexisted,
            "reached_hook": reached_hook,
            "trace": decision.trace,
        })

    tension_rows = [r for r in rows if r["label"] == "tension"]
    control_rows = [r for r in rows if r["label"] != "tension"]
    hits = [r for r in tension_rows if r["coexisted"]]
    classifier_blocked = [r for r in tension_rows
                          if not r["coexisted"] and not r["reached_hook"]]
    recognizer_missed = [r for r in tension_rows
                         if not r["coexisted"] and r["reached_hook"]]
    false_fires = [r for r in control_rows if r["coexisted"]]

    report = {
        "fixture": str(fixture_path),
        "backend": recognizer.backend,
        "model": recognizer.model,
        "n_pairs": len(rows),
        "n_tension": len(tension_rows),
        "coexist_recall": round(len(hits) / len(tension_rows), 4) if tension_rows else None,
        "classifier_blocked": len(classifier_blocked),
        "recognizer_missed": len(recognizer_missed),
        "false_fires": len(false_fires),
        "false_fire_pairs": [
            {"existing": r["existing"], "new": r["new"], "label": r["label"]}
            for r in false_fires
        ],
        "network_calls": recognizer.network_calls,
        "rows": rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_tension.json"
    )
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== Revien tension eval (B1) ===")
    print(f"backend        : {recognizer.backend} / {recognizer.model}")
    print(f"pairs          : {len(rows)} ({len(tension_rows)} tension, "
          f"{len(control_rows)} control)")
    print(f"coexist recall : {report['coexist_recall']} "
          f"({len(hits)}/{len(tension_rows)})")
    print(f"  blocked at classifier (known CSL bound): {len(classifier_blocked)}")
    print(f"  missed by recognizer                   : {len(recognizer_missed)}")
    print(f"false fires    : {len(false_fires)} / {len(control_rows)} controls "
          f"(MUST be 0 — §7.2)")
    for r in false_fires:
        print(f"  FALSE FIRE [{r['label']}]: {r['existing']!r} vs {r['new']!r}")
    print(f"network calls  : {recognizer.network_calls}")
    print(f"results JSON   : {out_path}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Revien B1 tension eval")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results" / "tension"))
    ap.add_argument("--fixture", default=str(FIXTURE))
    args = ap.parse_args(argv)
    report = run_eval(Path(args.out), Path(args.fixture))
    return 0 if report is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
