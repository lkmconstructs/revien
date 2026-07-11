"""Trigger 2 — verify the LLM sensitive recognizer against the closing criterion.

Runs the LLM recognizer (backend/model via env; gpt-4.1 for the reference verify)
over the known sensitive set + the 20 adversarial leaks the embedding recognizer
failed, and over the neutral controls. Reports:
  * confident_neutral leaks on sensitive — MUST be 0 (the closing criterion).
  * false-positive rate on neutral controls + everyday neutrals.
  * network call count (cost proxy).

Cost (gpt-4.1, ~one short call per fixture, <=4 output tokens): ~50-60 calls for
this fixture battery, well under $1. Run:
  OPENAI_API_KEY=<key> REVIEN_SENSITIVITY_BACKEND=openai \\
      python -m revien_bench.measure_sensitivity_llm
"""

from __future__ import annotations

import json
from pathlib import Path

from revien.sensitivity import SensitivityRoute
from revien.sensitivity_llm import LLMSensitivityRecognizer

_FIX = Path(__file__).resolve().parent / "fixtures" / "sensitivity_fixtures.json"


def main() -> int:
    fx = json.loads(_FIX.read_text(encoding="utf-8"))
    rec = LLMSensitivityRecognizer()
    print(f"backend={rec.backend} model={rec.model} available={rec.is_available()}")
    if not rec.is_available():
        print("  -> recognizer UNAVAILABLE (set the backend's API key, or run ollama). "
              "Degraded-safety means it would abstain-all; cannot verify recognition.")
        return 1

    sensitive = (fx["sensitive_no_lexeme"] + fx["sensitive_unseeded"]
                 + fx["trigger2_adversarial_leaks"])
    neutral = fx["neutral_controls"] + fx["neutral_everyday"]

    def run(texts):
        out = []
        for t in texts:
            v = rec.recognize(t)
            out.append((t, v.route))
        return out

    s = run(sensitive)
    n = run(neutral)

    leaks = [(t, r) for t, r in s if r is SensitivityRoute.CONFIDENT_NEUTRAL]
    fp = [(t, r) for t, r in n if r is not SensitivityRoute.CONFIDENT_NEUTRAL]

    print(f"\nsensitive items: {len(s)}   neutral items: {len(n)}   calls: {rec.network_calls}")
    print(f"\n  LEAKS (sensitive -> confident_neutral): {len(leaks)}/{len(s)}   <- MUST be 0")
    print(f"  false positives (neutral -> candidate): {len(fp)}/{len(n)}  "
          f"({100.0*len(fp)/len(n):.1f}%)")

    if leaks:
        print("\n  !!! LEAKS — Trigger 2 NOT closed by this model !!!")
        for t, r in leaks:
            print(f"    {t!r}")
    else:
        print("\n  no leaks on the known sensitive + adversarial-leak battery.")
    if fp:
        print("\n  false positives (neutral judged sensitive/unsure — safe, workload cost):")
        for t, r in fp:
            print(f"    [{r.value}] {t!r}")

    print("\n  NOTE: passing this battery is necessary, not sufficient — the full "
          "adversarial pass (fresh euphemisms/indirect/unseeded) is the real gate.")
    return 0 if not leaks else 2


if __name__ == "__main__":
    raise SystemExit(main())
