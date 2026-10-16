"""
revien_bench.decompose - BENCHMARK-ONLY query decomposition row.

Multi-hop LoCoMo questions need ~3 specific turns at once, and one query
embedding resembles only one of them. This row asks an LLM to split the
question into 2-3 sub-questions, recalls each, and lets the runner UNION the
results before the reader. It is NOT a product path: recall itself stays
local and LLM-free; only this harness row calls out, and it is egress-labeled
exactly like the judge.

"none" (default) means no decomposer at all: `build_decomposer("none")`
returns None and the runner behaves exactly as before.

Transport, disclosure, cost estimate and the frozen/hashed-prompt pattern are
mirrored from judges.py / answerers.py (shared `_chat_once` / `_ProviderCfg`).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import answerers as A

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "decompose.txt"
DECOMPOSE_PROMPT_SHA256 = "5913933c08ced47047341652389603a18c19263e01ba8ba15afd346ef729bd0e"
# Same reasoning-model lesson as the judge: hidden reasoning tokens spend the
# budget before any visible content. Usage bills only what is spent.
DECOMPOSE_MAX_TOKENS = 512
MAX_SUBQUERIES = 3


def load_decompose_prompt() -> str:
    """Read the frozen prompt, verifying its sha256 (CRLF-normalized, as in
    judges.load_judge_prompt)."""
    raw = _PROMPT_PATH.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DECOMPOSE_PROMPT_SHA256:
        raise ValueError(
            f"decompose prompt sha256 drifted: expected {DECOMPOSE_PROMPT_SHA256}, "
            f"got {digest}. The prompt is frozen - update DECOMPOSE_PROMPT_SHA256 "
            f"deliberately if you truly intend to change every sub-query."
        )
    return raw.decode("utf-8")


def assemble_decompose_prompt(question: str) -> str:
    return load_decompose_prompt().replace("{question}", question or "")


_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.)]|\(?[a-zA-Z][.)])\s+")


def parse_subqueries(raw: str, original: str) -> List[str]:
    """Reply text -> [original, sub1, ...sub3]. Lines are stripped of
    bullets/numbering, empties and duplicates (of each other or of the
    original) dropped, sub-questions capped at 3. The ORIGINAL is always
    first, so the union step can never do worse than the original query."""
    seen = {(original or "").strip().lower()}
    subs: List[str] = []
    for line in (raw or "").splitlines():
        text = _BULLET_RE.sub("", line).strip().strip('"').strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        subs.append(text)
        if len(subs) == MAX_SUBQUERIES:
            break
    return [original] + subs


@dataclass
class Decomposition:
    """One decompose call's result. `network_calls`/`cost_usd` describe THIS
    call only (same convention as judges.Verdict)."""

    queries: List[str]
    raw: str = ""
    network_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None


_DISCLOSED_DECOMPOSE_PROVIDERS: set = set()


def _disclose_decompose_cloud(provider: str) -> None:
    """One-time stderr warning, own set and wording (not shared with the
    reader's or judge's)."""
    if provider in _DISCLOSED_DECOMPOSE_PROVIDERS:
        return
    _DISCLOSED_DECOMPOSE_PROVIDERS.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending the question to {provider} to split it "
        f"into sub-questions - this leaves your machine. Use --decompose none "
        f"to keep it local.\n"
    )
    sys.stderr.flush()


def _fallback(question: str, t0: float, err: str, calls: int = 0) -> Decomposition:
    return Decomposition(
        queries=[question], network_calls=calls,
        latency_ms=(time.perf_counter() - t0) * 1000.0, error=err,
    )


def _finish(question: str, text: str, t0: float, calls: int, cost: float) -> Decomposition:
    queries = parse_subqueries(text, question)
    # An empty reply is an error; a reply that only repeats the question
    # (the atomic case) is a legitimate [original].
    err = None if (text or "").strip() else "empty decomposer output"
    return Decomposition(
        queries=queries, raw=text, network_calls=calls, cost_usd=cost,
        latency_ms=(time.perf_counter() - t0) * 1000.0, error=err,
    )


class OllamaDecomposer:
    """LOCAL decomposer via native Ollama /api/chat. Loopback-only is judged
    by sovereignty.network_egress_zero (same resolve_ollama_host the
    answerers use); a loopback call is never counted as network egress."""

    def __init__(self, model: str, url: Optional[str] = None):
        self.model = model
        self.url = A.resolve_ollama_host(url)
        self.name = f"ollama:{model}"
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def decompose(self, question: str) -> Decomposition:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": assemble_decompose_prompt(question)}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        t0 = time.perf_counter()
        try:
            data = A._http_post_json(f"{self.url}/api/chat", payload, headers={})
        except Exception as e:  # noqa: BLE001 - one bad call must not kill the run
            return _fallback(question, t0, f"{type(e).__name__}: {e}")
        text = (data.get("message") or {}).get("content", "")
        return _finish(question, text, t0, 0, 0.0)


class APIDecomposer:
    """Cloud decomposer (OpenAI-compatible or Anthropic), mirroring APIJudge:
    discloses once, counts every attempted call BEFORE the request, keeps a
    labelled cost estimate."""

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

    def decompose(self, question: str) -> Decomposition:
        _disclose_decompose_cloud(self.provider)
        t0 = time.perf_counter()
        attempted = False
        try:
            key = os.environ.get(self.key_env, "")
            if not key:
                raise RuntimeError(
                    f"{self.key_env} not set; required for --decompose {self.name}"
                )
            cfg = A._ProviderCfg(
                model=self.model, base_url=self.base_url, is_anthropic=self.is_anthropic,
                anthropic_version=self.anthropic_version, api_key=key,
            )
            attempted = True
            self.network_calls += 1
            text, in_tok, out_tok = A._chat_once(
                cfg, assemble_decompose_prompt(question), max_tokens=DECOMPOSE_MAX_TOKENS
            )
        except Exception as e:  # noqa: BLE001
            return _fallback(question, t0, f"{type(e).__name__}: {e}",
                             calls=1 if attempted else 0)
        cost = A.estimate_cost_usd(self.provider, in_tok, out_tok)
        self.cost_usd_estimate += cost
        return _finish(question, text, t0, 1, cost)


def build_decomposer(spec: str = "none"):
    """none -> None | ollama:<model> (local) | openai|openrouter|together|claude:<model>
    (cloud, discloses). Misconfigured specs fail loud. No call at construction."""
    spec = (spec or "none").strip()
    if spec.lower() == "none":
        return None
    provider, model = A._parse_spec(spec)
    if provider == "ollama":
        if not model:
            raise ValueError("ollama decomposer requires a model: ollama:<model>")
        return OllamaDecomposer(model)
    if provider in A._OPENAI_COMPAT or provider in A._ANTHROPIC:
        if not model:
            raise ValueError(f"{provider} decomposer requires a model: {provider}:<model>")
        return APIDecomposer(provider, model)
    raise ValueError(
        f"unknown decompose spec {spec!r}; expected 'none', 'ollama:<model>', or one of "
        f"{sorted(set(A._OPENAI_COMPAT) | set(A._ANTHROPIC))}:<model>"
    )
