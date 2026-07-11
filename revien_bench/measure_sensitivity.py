"""Claim Sovereignty Layer — Trigger 2: measure semantic sensitive recognition.

Scores the recognizer against the acceptance bar (acceptance spec):
  * 100% candidate routing on sensitive fixtures (seeded-domain AND unseeded).
  * 0 confident-neutral on sensitive fixtures (the unsafe outcome).
  * measured FALSE-POSITIVE rate on neutral controls + everyday neutrals.
  * abstention is reported, not penalized (sensitive | abstain both = candidate).

Also runs the recognizer end-to-end through the supersession gate to confirm a
sensitive existing claim that would auto-supersede instead routes to candidate.

Run: REVIEN_SEMANTIC=1 python -m revien_bench.measure_sensitivity
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from revien.ingestion.claim_classifier import ClaimClassifier
from revien.sensitivity import SemanticSensitivityRecognizer, SensitivityRoute
from revien.supersession import Claim, SupersessionAction, SupersessionGate

_FIX = Path(__file__).resolve().parent / "fixtures" / "sensitivity_fixtures.json"


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def measure() -> Dict:
    fx = json.loads(_FIX.read_text(encoding="utf-8"))
    rec = SemanticSensitivityRecognizer()
    if not rec.is_available():
        return {"available": False}

    def route_all(texts: List[str]):
        return [(t, rec.recognize(t)) for t in texts]

    sensitive = route_all(fx["sensitive_no_lexeme"] + fx["sensitive_unseeded"])
    seeded = route_all(fx["sensitive_no_lexeme"])
    unseeded = route_all(fx["sensitive_unseeded"])
    controls = route_all(fx["neutral_controls"])
    everyday = route_all(fx["neutral_everyday"])
    neutral = controls + everyday

    sens_candidate = [(t, v) for t, v in sensitive if v.routes_candidate]
    sens_leaked = [(t, v) for t, v in sensitive if not v.routes_candidate]  # UNSAFE
    neut_fp = [(t, v) for t, v in neutral if v.routes_candidate]            # false positives

    # Adversarial leak set (wf_194fc6c9) — the genuine recognizer's success
    # criterion: drive these to 0 confident_neutral. Reported, never asserted —
    # the embedding recognizer leaks them by construction (it is INSUFFICIENT).
    adv = route_all(fx.get("trigger2_adversarial_leaks", []))
    adv_leaks = [(t, v) for t, v in adv if not v.routes_candidate]

    return {
        "available": True,
        "n_sensitive": len(sensitive),
        "sensitive_candidate": (len(sens_candidate), len(sensitive)),
        "sensitive_leaked": sens_leaked,   # must be empty
        "seeded_candidate": (sum(1 for _, v in seeded if v.routes_candidate), len(seeded)),
        "unseeded_candidate": (sum(1 for _, v in unseeded if v.routes_candidate), len(unseeded)),
        "neutral_fp": (len(neut_fp), len(neutral)),
        "controls_fp": (sum(1 for _, v in controls if v.routes_candidate), len(controls)),
        "everyday_fp": (sum(1 for _, v in everyday if v.routes_candidate), len(everyday)),
        "route_dist": Counter(v.route.value for _, v in sensitive + neutral),
        "adversarial_leaks": (len(adv_leaks), len(adv)),
        "neutral_fp_items": neut_fp,
        "rows": {"sensitive": sensitive, "neutral": neutral},
        "thresholds": (rec.sens_threshold, rec.neut_threshold),
    }


def _gate_endtoend() -> List[tuple]:
    """A sensitive existing claim that WOULD auto-supersede must instead route
    candidate once the recognizer is wired into the gate."""
    clf = ClaimClassifier()
    gate = SupersessionGate(recognizer=SemanticSensitivityRecognizer())
    pairs = [
        ("I love being off the bottle.", "I don't enjoy being off the bottle."),
        ("I'm a former smoker.", "I don't enjoy being a former smoker."),
        ("I finally paid off the bankruptcy.", "I don't enjoy being debt free."),
    ]
    out = []
    for e, n in pairs:
        d = gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n)))
        out.append((e, d.action.value, d.reason))
    return out


def _print(r: Dict) -> None:
    print("\n===== Trigger 2 — semantic sensitive recognition =====")
    if not r.get("available"):
        print("  recognizer UNAVAILABLE (no local embedding layer). Cannot measure.")
        return
    st, nt = r["thresholds"]
    print(f"thresholds: sens>={st}  neut>={nt}")
    sc_n, sc_d = r["sensitive_candidate"]
    print("\n-- acceptance bar --")
    print(f"  sensitive -> candidate : {sc_n}/{sc_d}  ({_pct(sc_n, sc_d)})   <- must be 100%")
    print(f"  UNSAFE leaks (sens->neutral): {len(r['sensitive_leaked'])}   <- must be 0")
    se_n, se_d = r["seeded_candidate"]
    un_n, un_d = r["unseeded_candidate"]
    print(f"    seeded-domain  : {se_n}/{se_d}  ({_pct(se_n, se_d)})")
    print(f"    UNSEEDED domain: {un_n}/{un_d}  ({_pct(un_n, un_d)})   <- recognizes property, not vocab")
    fp_n, fp_d = r["neutral_fp"]
    cf_n, cf_d = r["controls_fp"]
    ef_n, ef_d = r["everyday_fp"]
    print(f"  false positives (neutral->candidate): {fp_n}/{fp_d}  ({_pct(fp_n, fp_d)})")
    print(f"    adversarial controls: {cf_n}/{cf_d}   everyday: {ef_n}/{ef_d}")
    print(f"  route distribution: {dict(r['route_dist'])}")
    al_n, al_d = r["adversarial_leaks"]
    print(f"\n  ADVERSARIAL LEAKS (must reach 0 for Trigger 2 to close): {al_n}/{al_d}  ({_pct(al_n, al_d)})")
    print(f"    ^ embedding recognizer is INSUFFICIENT — these prove it keys on surface, not meaning")

    if r["sensitive_leaked"]:
        print("\n-- !!! UNSAFE: sensitive routed confident-neutral !!! --")
        for t, v in r["sensitive_leaked"]:
            print(f"  sens={v.sens_sim} neut={v.neut_sim}  {t!r}")
    if r["neutral_fp_items"]:
        print("\n-- false positives (neutral -> candidate; SAFE, a workload cost) --")
        for t, v in r["neutral_fp_items"]:
            print(f"  [{v.route.value}] sens={v.sens_sim} neut={v.neut_sim}  {t!r}")

    print("\n-- per-fixture detail --")
    for label in ("sensitive", "neutral"):
        print(f"  [{label}]")
        for t, v in r["rows"][label]:
            print(f"    {v.route.value:17} sens={v.sens_sim:.3f} neut={v.neut_sim:.3f}  {t!r}")


def main() -> int:
    r = measure()
    _print(r)
    if r.get("available"):
        print("\n-- gate end-to-end (recognizer wired; sensitive must NOT auto) --")
        for text, action, reason in _gate_endtoend():
            flag = "OK" if action != "auto_supersede" else "!! AUTO !!"
            print(f"  {flag}  {action:14} ({reason})  {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
