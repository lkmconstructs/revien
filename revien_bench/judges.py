"""
revien_bench.judges — The LLM-judge track: end-to-end binary CORRECT/WRONG
scoring of a predicted answer against the LoCoMo gold answer.

This is a SEPARATE, second track from the official LoCoMo token-F1 metric
(revien_bench.metrics.f1_score). F1 measures lexical overlap with the gold
answer; an LLM judge measures whether a human grader would call the answer
right, the way an end-to-end memory-QA harness (e.g. the OpenViking pattern:
per-system ingest + QA + an LLM judge comparing prediction to gold) reports
accuracy. The two numbers are NEVER blended — see runner.py / report.py.

"f1" (the default judge spec) means NO LLM judge at all: `build_judge("f1")`
returns None, and the runner falls back to F1-only scoring exactly as before
this track existed.

Transport, disclosure, cost estimation, and the frozen/hashed-prompt pattern
are ALL mirrored from answerers.py so a judge run behaves identically to an
LLM answerer run: same env vars (OLLAMA_HOST, OPENAI_API_KEY, etc.), same
one-time cloud disclosure to stderr, same REVIEN_ANSWERER_TIMEOUT-governed
socket timeout, same coarse cost-per-1K-token estimate. stdlib urllib only.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import answerers as A

# Frozen, hashed prompt — same freeze discipline as answerers.ANSWER_PROMPT_SHA256:
# a silent edit would change every judge verdict, so load_judge_prompt() raises
# if the on-disk file no longer matches this digest.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge.txt"
JUDGE_PROMPT_SHA256 = "9f64753745baf8b076ff61f251b3f89fe7f56531068877f754d958436d7c594c"


def load_judge_prompt() -> str:
    """Read the frozen judge prompt, verifying its sha256 (mirrors
    answerers.load_answer_prompt — see there for the CRLF-normalization note).
    A missing file raises FileNotFoundError from read_bytes() itself — no
    separate pre-read exists() check needed."""
    raw = _PROMPT_PATH.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != JUDGE_PROMPT_SHA256:
        raise ValueError(
            f"judge prompt sha256 drifted: expected {JUDGE_PROMPT_SHA256}, "
            f"got {digest}. The prompt is frozen — update JUDGE_PROMPT_SHA256 "
            f"deliberately if you truly intend to change every judge verdict."
        )
    return raw.decode("utf-8")


def assemble_judge_prompt(
    question: str, gold: str, prediction: str, category_name: str = ""
) -> str:
    """Fill the frozen template with question/gold/prediction/category.

    Single source of truth for judge-prompt assembly, so a unit test can prove
    the wiring without a network call.
    """
    template = load_judge_prompt()
    return (
        template.replace("{category}", category_name or "")
        .replace("{question}", question or "")
        .replace("{gold}", gold or "")
        .replace("{prediction}", prediction or "")
    )


@dataclass
class Verdict:
    """One judge call's result. `network_calls`/`cost_usd` describe THIS call
    only (0 / $0.0 for a local Ollama judge; 1 / an estimate for a cloud one) —
    callers accumulate across calls themselves, same convention as the
    answerer's self.network_calls / self.cost_usd_estimate counters."""

    correct: bool
    raw: str
    network_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None


def _parse_verdict(raw: str) -> Tuple[bool, Optional[str]]:
    """First word of the reply, case-insensitive, in {CORRECT, WRONG}.

    Anything else (empty, garbage, a hedge) is an error: correct=False, and the
    caller counts it toward judge_errors — a judge that can't be parsed is
    never silently scored as a pass.

    F12: leading NON-LETTER characters (markdown bold "**", a list marker
    "1.", a leading quote) are stripped before taking the first word, so
    "**CORRECT**" and "1. CORRECT" parse cleanly. This only strips a prefix
    — "The answer is CORRECT" still starts with a letter ('T'), so nothing
    is stripped and it still errors (the model must lead with the verdict).
    """
    text = (raw or "").strip()
    text = re.sub(r"^[^A-Za-z]+", "", text)
    m = re.match(r"[A-Za-z]+", text)
    word = (m.group(0) if m else "").upper()
    if word == "CORRECT":
        return True, None
    if word == "WRONG":
        return False, None
    return False, f"unparseable judge output: {text[:120]!r}"


# ── Judge-specific cloud disclosure ────────────────────────────────────────
# A SEPARATE disclosed-set from answerers._DISCLOSED_PROVIDERS (F6): a run
# with --answerer openai:x --judge openai:x sends question+memory to OpenAI
# TWICE, for two different reasons (answering vs judging), and each purpose
# gets its own one-time disclosure — sharing the reader's set would silently
# swallow the judge's disclosure whenever the reader had already fired one.
_DISCLOSED_JUDGE_PROVIDERS: set = set()


def _disclose_judge_cloud(provider: str) -> None:
    """One-time stderr warning when the question/gold/prediction leave the
    machine FOR JUDGING. Local judges (f1, ollama) never call this."""
    if provider in _DISCLOSED_JUDGE_PROVIDERS:
        return
    _DISCLOSED_JUDGE_PROVIDERS.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending the question, gold answer and prediction "
        f"to {provider} for judging - this leaves your machine. Use "
        f"--judge f1 to keep it local.\n"
    )
    sys.stderr.flush()


class OllamaJudge:
    """LOCAL LLM judge via native Ollama /api/chat (http://localhost:11434).

    Zero egress (loopback) — mirrors OllamaAnswerer exactly, including the
    "never increments network_calls" convention (a loopback call never leaves
    the machine, so it is not counted as network egress).
    """

    def __init__(self, model: str, url: Optional[str] = None):
        self.model = model
        # F9: resolve OLLAMA_HOST fresh (not a stale module-import-time
        # value) via the shared answerers.resolve_ollama_host — sovereignty's
        # network_egress_zero inspects this SAME resolution to decide whether
        # this judge is actually local.
        self.url = A.resolve_ollama_host(url)
        self.name = f"ollama:{model}"
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def judge(
        self, question: str, gold: str, prediction: str, category_name: str = ""
    ) -> Verdict:
        prompt = assemble_judge_prompt(question, gold, prediction, category_name)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        t0 = time.perf_counter()
        try:
            data = A._http_post_json(f"{self.url}/api/chat", payload, headers={})
        except Exception as e:  # noqa: BLE001 - one bad judge call must not kill the run
            return Verdict(
                correct=False, raw="",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"{type(e).__name__}: {e}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        msg = (data.get("message") or {}).get("content", "")
        correct, err = _parse_verdict(msg)
        return Verdict(correct=correct, raw=msg, latency_ms=latency_ms, error=err)


class APIJudge:
    """Cloud LLM judge. OpenAI-compatible /v1/chat/completions for
    openai/openrouter/together, AND Anthropic /v1/messages for claude.

    Mirrors APIAnswerer: stdlib urllib only, discloses ONCE before any
    question/gold/prediction leaves the machine, reads the key from the
    provider-appropriate env var (same key envs as the answerer track — a
    reader and a judge on the same provider share one key), and self-counts
    network_calls / cost_usd_estimate.
    """

    def __init__(self, provider: str, model: str):
        provider = (provider or "").strip().lower()
        self.provider = provider
        self.model = model
        self.name = f"{provider}:{model}"
        self.is_anthropic = provider in A._ANTHROPIC
        if self.is_anthropic:
            cfg = A._ANTHROPIC[provider]
        elif provider in A._OPENAI_COMPAT:
            cfg = A._OPENAI_COMPAT[provider]
        else:
            raise ValueError(
                f"unknown cloud provider {provider!r}; expected one of "
                f"{sorted(set(A._OPENAI_COMPAT) | set(A._ANTHROPIC))}"
            )
        self.base_url = cfg["base_url"]
        self.key_env = cfg["key_env"]
        self.anthropic_version = cfg.get("version")
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def _api_key(self) -> str:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise RuntimeError(
                f"{self.key_env} not set; required for --judge {self.name}"
            )
        return key

    def judge(
        self, question: str, gold: str, prediction: str, category_name: str = ""
    ) -> Verdict:
        prompt = assemble_judge_prompt(question, gold, prediction, category_name)
        # Disclose BEFORE the network call (fires even if the request fails).
        # Judge-specific set/message (F6) — separate from the answerer's.
        _disclose_judge_cloud(self.provider)
        t0 = time.perf_counter()
        attempted = False
        try:
            key = self._api_key()
            cfg = A._ProviderCfg(
                model=self.model, base_url=self.base_url, is_anthropic=self.is_anthropic,
                anthropic_version=self.anthropic_version, api_key=key,
            )
            # F7: increment BEFORE the request — a failed call (HTTP 500,
            # timeout) still left the machine and must be counted as egress.
            attempted = True
            self.network_calls += 1
            text, in_tok, out_tok = A._chat_once(cfg, prompt, max_tokens=8)
        except Exception as e:  # noqa: BLE001 - one bad judge call must not kill the run
            return Verdict(
                correct=False,
                raw="",
                network_calls=1 if attempted else 0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"{type(e).__name__}: {e}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cost = A.estimate_cost_usd(self.provider, in_tok, out_tok)
        self.cost_usd_estimate += cost
        correct, err = _parse_verdict(text)
        return Verdict(
            correct=correct, raw=text, network_calls=1, cost_usd=cost,
            latency_ms=latency_ms, error=err,
        )


def build_judge(spec: str = "f1"):
    """Factory. Parses the judge spec and constructs the right backend.

    Accepted specs:
        f1                        -> None (no LLM judge; F1-only, default)
        ollama:<model>            -> OllamaJudge (LOCAL, loopback, zero egress)
        openai:<model>            -> APIJudge (OpenAI /v1, discloses)
        openrouter:<model>        -> APIJudge (OpenRouter /v1, discloses)
        together:<model>          -> APIJudge (Together /v1, discloses)
        claude:<model>            -> APIJudge (Anthropic /v1/messages, discloses)

    Misconfigured specs fail loud (ValueError) rather than silently degrading.
    No cloud call is made at construction time — only on .judge().
    """
    spec = (spec or "f1").strip()
    if spec.lower() == "f1":
        return None

    provider, model = A._parse_spec(spec)

    if provider == "ollama":
        if not model:
            raise ValueError("ollama judge requires a model: ollama:<model>")
        return OllamaJudge(model)

    if provider in A._OPENAI_COMPAT or provider in A._ANTHROPIC:
        if not model:
            raise ValueError(f"{provider} judge requires a model: {provider}:<model>")
        return APIJudge(provider, model)

    raise ValueError(
        f"unknown judge spec {spec!r}; expected 'f1', 'ollama:<model>', or one of "
        f"{sorted(set(A._OPENAI_COMPAT) | set(A._ANTHROPIC))}:<model>"
    )
