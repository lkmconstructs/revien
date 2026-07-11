"""Trigger 2 — FRESH unseeded adversarial battery gate (parallel + progress).

The committed measure_sensitivity_llm.py gates the SEEDED fixtures the system
prompt was shaped around ("one-year chip", "interview at the office", ...).
Passing that is necessary, not sufficient. THIS gate runs the recognizer against
a fresh battery (revien_bench/fixtures/sensitivity_adversarial_fresh.json) built
by a 15-domain adversarial red-team — phrasings the prompt was never tuned on,
including domains it does not enumerate (HIV/STI, disability, eating-disorder
recovery, sex-work/infidelity/cult-exit/terminal-prognosis).

LEAK  = a sensitive_fresh / context_collapsed item the model calls NEUTRAL
        (silent-erase eligible). MUST be 0 to honestly close Trigger 2.
FP    = a neutral_decoys_fresh item judged anything but NEUTRAL (SAFE — an extra
        human review — but a workload cost). Reported, not failed.
ERROR = a hard backend failure (HTTP/timeout/parse). Reported SEPARATELY so a
        broken backend can never masquerade as "0 leaks" (errors are NOT counted
        as safe abstains).

Backends mirror revien/sensitivity_llm.py: cloud opt-in via
REVIEN_SENSITIVITY_BACKEND (openai/openrouter/anthropic) + the matching key,
model via REVIEN_SENSITIVITY_MODEL. Concurrency via REVIEN_SENS_WORKERS (default 8);
each call uses its OWN recognizer so one failure cannot trip the shared _broken
flag and silently abstain the rest.

Usage:
  REVIEN_SENSITIVITY_BACKEND=openrouter REVIEN_SENSITIVITY_MODEL=qwen/... \
    OPENROUTER_API_KEY=... python -u -m revien_bench.measure_sensitivity_fresh
Exit 0 iff 0 leaks AND 0 errors, else 2 (leaks), 4 (errors), 3 (unavailable).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from revien.sensitivity_llm import LLMSensitivityRecognizer

FIXTURES = Path(os.environ.get("REVIEN_SENS_FIXTURES", "")) if os.environ.get("REVIEN_SENS_FIXTURES") \
    else Path(__file__).parent / "fixtures" / "sensitivity_adversarial_fresh.json"
WORKERS = max(1, int(os.environ.get("REVIEN_SENS_WORKERS", "8")))


def classify_one(text: str):
    """Return one of 'NEUTRAL' | 'SENSITIVE' | 'ABSTAIN' | ('ERROR', msg).

    Uses a fresh recognizer per call so the per-instance _broken flag cannot
    leak across calls, and calls _classify directly so a real backend failure
    surfaces as ERROR instead of being swallowed into a safe-looking abstain.
    """
    r = LLMSensitivityRecognizer()
    try:
        word = r._classify(text.strip())  # raises on HTTP/timeout
    except Exception as e:  # noqa: BLE001
        return ("ERROR", repr(e)[:160])
    if word == "NEUTRAL":
        return "NEUTRAL"
    if word == "SENSITIVE":
        return "SENSITIVE"
    return "ABSTAIN"  # UNSURE or unparseable


def main() -> int:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    items = (
        [(t, "leak") for t in data["sensitive_fresh"]]
        + [(t, "leak") for t in data["context_collapsed"]]
        + [(t, "neutral") for t in data["neutral_decoys_fresh"]]
    )
    n_leak_gated = sum(1 for _, b in items if b == "leak")
    n_neutral = sum(1 for _, b in items if b == "neutral")

    probe = LLMSensitivityRecognizer()
    print(f"backend={probe.backend} model={probe.model} cloud={probe.is_cloud} "
          f"available={probe.is_available()} workers={WORKERS}", flush=True)
    if not probe.is_available():
        print("  -> recognizer UNAVAILABLE (set the backend key). Aborting.", flush=True)
        return 3

    total = len(items)
    done = 0
    lock = threading.Lock()
    verdicts: dict[int, object] = {}

    def work(pair):
        nonlocal done
        idx, (text, _bucket) = pair
        v = classify_one(text)
        with lock:
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  ... {done}/{total}", flush=True)
        return idx, v

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for idx, v in ex.map(work, list(enumerate(items))):
            verdicts[idx] = v

    leaks, fp, errors = [], [], []
    for idx, (text, bucket) in enumerate(items):
        v = verdicts[idx]
        if isinstance(v, tuple) and v[0] == "ERROR":
            errors.append((text, v[1]))
            continue
        if bucket == "leak":
            if v == "NEUTRAL":
                leaks.append(text)
        else:  # neutral decoy
            if v != "NEUTRAL":
                fp.append((v, text))

    print(f"\nleak-gated: {n_leak_gated}   neutral decoys: {n_neutral}   errors: {len(errors)}", flush=True)
    print(f"\n  LEAKS (sensitive -> NEUTRAL): {len(leaks)}/{n_leak_gated}   <- MUST be 0")
    print(f"  false positives (neutral -> sensitive/unsure): {len(fp)}/{n_neutral}   (safe, workload cost)")
    if errors:
        print(f"  ERRORS (backend failures, run is SUSPECT): {len(errors)}")

    if leaks:
        print("\n  !!! LEAKS — Trigger 2 NOT closed on the fresh battery !!!")
        for text in leaks:
            print(f"    {text!r}")
    else:
        print("\n  no leaks on the FRESH unseeded battery — recognition holds on novel phrasing.")

    if fp:
        print("\n  false positives (neutral judged sensitive/unsure — safe, workload cost):")
        for v, text in fp:
            print(f"    [{v}] {text!r}")

    if errors:
        print("\n  backend errors (NOT counted as safe — re-run):")
        for text, msg in errors[:10]:
            print(f"    {msg}  ::  {text!r}")

    if errors:
        return 4
    return 0 if not leaks else 2


if __name__ == "__main__":
    raise SystemExit(main())
