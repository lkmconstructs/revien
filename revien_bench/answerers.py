"""
revien_bench.answerers — The "reader" layer. HEADLINE build: ExtractiveAnswerer only.

The answerer takes the question + the top-K retrieved nodes (the SAME context the
LLM answerers would get, when those exist) and produces an answer string. Keeping
the reader pluggable means the only variable across tracks is the reader, so
retrieval quality is isolated.

ExtractiveAnswerer (zero-LLM, deterministic, $0, zero-egress)
-------------------------------------------------------------
No model, no network, no randomness. It selects the best SPAN/sentence from the
retrieved node content by lexical overlap with the question, and REFUSES (emits a
canonical "no information available" / "not mentioned" string) when no candidate
clears a minimum-overlap floor — which is exactly what the adversarial (cat 5)
category rewards (hallucination resistance from absence-of-node).

LLM readers (ready for an end-to-end read the moment a model/key is available)
-----------------------------------------------------------------------------
    class OllamaAnswerer(Answerer): ...   # local LLM reader (ollama:<model>)
    class APIAnswerer(Answerer): ...      # OpenAI-compatible / Anthropic cloud
All readers consume the SAME `RetrievedContext` the extractive reader gets, so
retrieval quality stays the isolated variable. The LLM readers assemble a
frozen, hashed prompt (prompts/answerer.txt) that instructs the model to answer
CONCISELY from ONLY the provided context and to say "No information available"
when the context doesn't support an answer (handles adversarial questions).

Transport is stdlib urllib ONLY — no SDKs, no new dependencies. Cloud providers
disclose once to stderr, mirroring leg-4 extractor_llm._disclose_cloud.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

# Canonical refusal string. Contains BOTH official adversarial markers so the
# official matcher scores it as a correct refusal regardless of which marker the
# grader checks. (metrics.is_refusal recognizes either.)
REFUSAL = "No information available; this was not mentioned."

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "what", "when", "where", "why", "who", "whom", "which", "how", "to",
    "of", "in", "on", "at", "by", "for", "with", "and", "or", "but", "not",
    "that", "this", "it", "as", "from", "about", "into", "over", "after",
    "i", "we", "you", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "tell", "give", "name",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 1}


@dataclass
class RetrievedContext:
    """The context handed to a reader: ordered node contents from recall()."""
    query: str
    contents: List[str] = field(default_factory=list)  # ranked node contents
    labels: List[str] = field(default_factory=list)      # parallel node labels

    def sentences(self) -> List[str]:
        """Flatten retrieved contents into candidate sentences, rank-preserving."""
        out: List[str] = []
        for content in self.contents:
            for sent in _SENT_SPLIT.split(content or ""):
                sent = sent.strip()
                if sent:
                    out.append(sent)
        return out


class Answerer(Protocol):
    """A reader: turn a question + retrieved context into an answer string."""

    name: str

    def answer(self, ctx: RetrievedContext) -> str: ...


class ExtractiveAnswerer:
    """Deterministic, zero-LLM span/sentence selector.

    Scores each candidate sentence by lexical overlap with the question's
    content tokens (Jaccard-ish: shared / question-token-count, with a small
    bonus for sentences that strip the speaker prefix cleanly). Returns the
    best sentence's most-relevant span. Refuses when the best score is below
    `min_overlap` (no node supports an answer) — the adversarial-resistant path.
    """

    name = "extractive"

    def __init__(self, min_overlap: float = 0.15, max_answer_chars: int = 240):
        self.min_overlap = min_overlap
        self.max_answer_chars = max_answer_chars
        # Local reader: never leaves the machine. Kept for uniform accounting.
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def answer(self, ctx: RetrievedContext) -> str:
        q_tokens = _content_tokens(ctx.query)
        if not q_tokens:
            # Degenerate question (all stopwords): fall back to top sentence.
            sents = ctx.sentences()
            return self._trim(sents[0]) if sents else REFUSAL

        best_sent = None
        best_score = 0.0
        for sent in ctx.sentences():
            s_tokens = _content_tokens(sent)
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            if overlap == 0:
                continue
            # Overlap normalized by question size (recall of question terms),
            # with a mild length penalty so a huge sentence doesn't win on raw
            # term count alone. Deterministic — no randomness.
            score = overlap / len(q_tokens)
            score *= 1.0 / (1.0 + 0.002 * max(0, len(sent) - 80))
            if score > best_score:
                best_score = score
                best_sent = sent

        if best_sent is None or best_score < self.min_overlap:
            return REFUSAL
        return self._trim(self._strip_speaker(best_sent))

    @staticmethod
    def _strip_speaker(sent: str) -> str:
        """Drop a leading 'Speaker: ' prefix carried from the turn content."""
        return re.sub(r"^[A-Z][A-Za-z0-9 _-]{0,40}:\s*", "", sent).strip()

    def _trim(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.max_answer_chars:
            return text
        return text[: self.max_answer_chars].rsplit(" ", 1)[0] + "..."


# ── LLM reader scaffolding ────────────────────────────────────────────────────
# Frozen, hashed prompt. The benchmark pins the prompt by sha256 so a silent
# edit (which would change every LLM answer) is caught: load_answer_prompt()
# raises if the on-disk file no longer matches this digest.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "answerer.txt"
ANSWER_PROMPT_SHA256 = "eaab3885c6591c81491c08ab762fbeeab6d2893c1d49922cb887b29a8fe5e487"

# How much retrieved context to hand the model. The retrieved node contents are
# already the top-K from recall(); we cap total characters so a pathological
# context can't blow the model's window or the request size.
MAX_CONTEXT_CHARS = 4000

# HTTP socket timeout (seconds) for EVERY LLM answerer request. A stalled
# connection (dead Ollama, hung cloud endpoint, half-open socket) must raise a
# socket.timeout rather than block the whole benchmark forever — a single hung
# call was one of the two ways a 20-min run lost everything. Env-overridable via
# REVIEN_ANSWERER_TIMEOUT (seconds); falls back to 60s if unset/invalid.
_DEFAULT_REQUEST_TIMEOUT = 60.0


def _resolve_timeout() -> float:
    """Read REVIEN_ANSWERER_TIMEOUT (seconds); fall back to the 60s default.

    Invalid / non-positive values fall back to the default rather than wedging
    the transport with a 0/negative timeout.
    """
    raw = os.environ.get("REVIEN_ANSWERER_TIMEOUT")
    if raw is None:
        return _DEFAULT_REQUEST_TIMEOUT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_REQUEST_TIMEOUT
    return val if val > 0 else _DEFAULT_REQUEST_TIMEOUT


# Resolved once at import for the module-level constant other code reads, but the
# transport re-reads the env on each call so a test (or operator) can override it
# without re-importing. Mirrors the extractor/embedder transport budget.
REQUEST_TIMEOUT = _resolve_timeout()

# Cloud providers (disclose once before any text leaves the machine). Ollama is
# LOCAL and never discloses. `CLOUD_PROVIDERS` is the PUBLIC source of truth the
# sovereignty egress check imports — keep it in sync with the readers below.
_CLOUD_PROVIDERS = {"openai", "openrouter", "together", "claude"}
CLOUD_PROVIDERS = frozenset(_CLOUD_PROVIDERS)
# Local readers (never leave the machine).
LOCAL_PROVIDERS = frozenset({"extractive", "ollama"})

# ── Cost estimation (ESTIMATE — not a billed figure) ──────────────────────────
# Best-effort per-provider USD price table, dollars per 1K tokens (input, output).
# These are coarse public list-price anchors used only to surface a non-zero,
# clearly-labelled cost ESTIMATE on cloud runs so a paid run never reports $0.00.
# Local readers (extractive/ollama) cost $0 and never enter this table.
COST_PER_1K_TOKENS_USD = {
    "openai": (0.00015, 0.00060),      # ~gpt-4o-mini list price anchor
    "openrouter": (0.00050, 0.00150),  # provider-dependent; mid anchor
    "together": (0.00020, 0.00060),    # OSS-model hosting anchor
    "claude": (0.00080, 0.00400),      # ~claude-3-5-haiku list price anchor
}
# Fallback when a provider isn't in the table (still labelled an estimate).
_DEFAULT_COST_PER_1K = (0.0005, 0.0015)


def estimate_cost_usd(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Coarse USD cost ESTIMATE for one cloud call. Local providers -> 0.0.

    Token-based per-1K pricing; labelled an estimate everywhere it surfaces.
    """
    provider = (provider or "").strip().lower()
    if provider not in CLOUD_PROVIDERS:
        return 0.0
    in_rate, out_rate = COST_PER_1K_TOKENS_USD.get(provider, _DEFAULT_COST_PER_1K)
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate


def _approx_tokens(text: str) -> int:
    """Crude token estimate (~4 chars/token) when the API omits a usage block."""
    return max(1, len((text or "")) // 4)


def parse_provider(spec: str) -> Tuple[str, Optional[str]]:
    """Public: split a reader spec into (provider, model). See `_parse_spec`."""
    return _parse_spec(spec)

# OpenAI-compatible /v1/chat/completions endpoints + the env var holding the key.
_OPENAI_COMPAT = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "key_env": "TOGETHER_API_KEY",
    },
}
# Anthropic native /v1/messages.
_ANTHROPIC = {
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "key_env": "ANTHROPIC_API_KEY",
        "version": "2023-06-01",
    },
}

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def load_answer_prompt() -> str:
    """Read the frozen prompt template, verifying its sha256.

    Raises if the file is missing or its digest drifted from the pinned value,
    so an accidental prompt edit fails loud (every LLM answer depends on it).
    """
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"answer prompt missing at {_PROMPT_PATH}")
    raw = _PROMPT_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ANSWER_PROMPT_SHA256:
        raise ValueError(
            f"answer prompt sha256 drifted: expected {ANSWER_PROMPT_SHA256}, "
            f"got {digest}. The prompt is frozen — update ANSWER_PROMPT_SHA256 "
            f"deliberately if you truly intend to change every LLM answer."
        )
    return raw.decode("utf-8")


def _format_context(ctx: RetrievedContext) -> str:
    """Render the retrieved nodes into a compact, ranked context block.

    Rank-preserving, labelled, char-capped. Empty when nothing was retrieved
    (the prompt then instructs the model to say "No information available").
    """
    lines: List[str] = []
    total = 0
    for i, content in enumerate(ctx.contents):
        text = (content or "").strip()
        if not text:
            continue
        label = ctx.labels[i] if i < len(ctx.labels) else ""
        prefix = f"[{i + 1}] " + (f"({label}) " if label else "")
        entry = prefix + text
        if total + len(entry) > MAX_CONTEXT_CHARS:
            entry = entry[: max(0, MAX_CONTEXT_CHARS - total)]
            if entry:
                lines.append(entry)
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


def assemble_prompt(ctx: RetrievedContext) -> str:
    """Fill the frozen template with the retrieved context + question.

    Single source of truth for prompt assembly across all LLM readers, so the
    unit test can prove the concise-answer instruction + context wiring without
    a network call.
    """
    template = load_answer_prompt()
    context = _format_context(ctx) or "(no memory context retrieved)"
    return template.replace("{context}", context).replace("{question}", ctx.query)


# ── Cloud disclosure (mirrors leg-4 extractor_llm._disclose_cloud) ─────────────
_DISCLOSED_PROVIDERS: set = set()


def _disclose_cloud(provider: str) -> None:
    """One-time stderr warning when the question + memory leave the machine.

    Same style/voice as leg 4's extractor disclosure. Local readers (ollama)
    never call this.
    """
    if provider in _DISCLOSED_PROVIDERS:
        return
    _DISCLOSED_PROVIDERS.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending the question and retrieved memory to "
        f"{provider} for answering - this leaves your machine. Use "
        f"--answerer extractive (offline) or ollama:<model> (local) to keep it "
        f"on-device.\n"
    )
    sys.stderr.flush()


def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON, return parsed JSON. stdlib urllib only — no SDKs.

    The socket timeout is re-resolved from REVIEN_ANSWERER_TIMEOUT on every call
    so a stalled connection RAISES (socket.timeout) instead of blocking forever.
    A timeout is normalized to a RuntimeError so the runner can catch one bad
    call, record the QA as unanswered, and continue — a hung call can't kill the
    whole run.
    """
    timeout = _resolve_timeout()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover - network path
        body = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except socket.timeout as e:  # pragma: no cover - network path
        raise RuntimeError(
            f"timeout after {timeout}s contacting {url} "
            f"(set REVIEN_ANSWERER_TIMEOUT to adjust)"
        ) from e
    except urllib.error.URLError as e:  # pragma: no cover - network path
        # urllib wraps socket.timeout as URLError(reason=socket.timeout) on some
        # platforms; surface either as a clean, catchable RuntimeError.
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.timeout):
            raise RuntimeError(
                f"timeout after {timeout}s contacting {url} "
                f"(set REVIEN_ANSWERER_TIMEOUT to adjust)"
            ) from e
        raise RuntimeError(f"network error contacting {url}: {reason}") from e


class OllamaAnswerer:
    """LOCAL LLM reader via native Ollama /api/chat (http://localhost:11434).

    Zero egress (loopback). stdlib urllib only. Given the retrieved context +
    question, returns a short answer or the model's "No information available"
    when the context doesn't support one.
    """

    def __init__(self, model: str, url: Optional[str] = None):
        self.model = model
        self.url = (url or OLLAMA_URL).rstrip("/")
        self.name = f"ollama:{model}"
        # LOCAL (loopback): no off-device egress, $0. Kept for uniform accounting.
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def answer(self, ctx: RetrievedContext) -> str:
        prompt = assemble_prompt(ctx)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        data = _http_post_json(f"{self.url}/api/chat", payload, headers={})
        # Ollama /api/chat -> {"message": {"role": ..., "content": "..."}}
        msg = (data.get("message") or {}).get("content", "")
        return _clean_answer(msg)


class APIAnswerer:
    """Cloud LLM reader. OpenAI-compatible /v1/chat/completions for
    openai/openrouter/together, AND Anthropic /v1/messages for claude.

    stdlib urllib only — no SDKs. Discloses ONCE before any text leaves the
    machine, mirroring the leg-4 extractor. API key is read from the
    provider-appropriate env var; absence raises a clear error (no silent
    fabrication of a cloud answer).
    """

    def __init__(self, provider: str, model: str):
        provider = (provider or "").strip().lower()
        self.provider = provider
        self.model = model
        self.name = f"{provider}:{model}"
        self.is_anthropic = provider in _ANTHROPIC
        if self.is_anthropic:
            cfg = _ANTHROPIC[provider]
        elif provider in _OPENAI_COMPAT:
            cfg = _OPENAI_COMPAT[provider]
        else:
            raise ValueError(
                f"unknown cloud provider {provider!r}; expected one of "
                f"{sorted(set(_OPENAI_COMPAT) | set(_ANTHROPIC))}"
            )
        self.base_url = cfg["base_url"]
        self.key_env = cfg["key_env"]
        self.anthropic_version = cfg.get("version")
        # Network/cost accounting (read by the runner for the results JSON).
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def _api_key(self) -> str:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise RuntimeError(
                f"{self.key_env} not set; required for --answerer {self.name}"
            )
        return key

    def answer(self, ctx: RetrievedContext) -> str:
        prompt = assemble_prompt(ctx)
        # Disclose BEFORE the network call (fires even if the request fails).
        _disclose_cloud(self.provider)
        key = self._api_key()

        if self.is_anthropic:
            payload = {
                "model": self.model,
                "max_tokens": 256,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "x-api-key": key,
                "anthropic-version": self.anthropic_version,
            }
            data = _http_post_json(
                f"{self.base_url}/messages", payload, headers=headers
            )
            self.network_calls += 1
            # Anthropic -> {"content": [{"type": "text", "text": "..."}], ...}
            parts = data.get("content") or []
            text = "".join(
                p.get("text", "") for p in parts if p.get("type") == "text"
            )
            # Anthropic usage -> {"input_tokens": .., "output_tokens": ..}.
            usage = data.get("usage") or {}
            in_tok = usage.get("input_tokens") or _approx_tokens(prompt)
            out_tok = usage.get("output_tokens") or _approx_tokens(text)
            self.cost_usd_estimate += estimate_cost_usd(self.provider, in_tok, out_tok)
            return _clean_answer(text)

        # OpenAI-compatible /v1/chat/completions.
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {key}"}
        data = _http_post_json(
            f"{self.base_url}/chat/completions", payload, headers=headers
        )
        self.network_calls += 1
        # OpenAI -> {"choices": [{"message": {"content": "..."}}]}
        choices = data.get("choices") or []
        text = (choices[0].get("message") or {}).get("content", "") if choices else ""
        # OpenAI usage -> {"prompt_tokens": .., "completion_tokens": ..}.
        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens") or _approx_tokens(prompt)
        out_tok = usage.get("completion_tokens") or _approx_tokens(text)
        self.cost_usd_estimate += estimate_cost_usd(self.provider, in_tok, out_tok)
        return _clean_answer(text)


def _clean_answer(text: str) -> str:
    """Normalize an LLM answer: strip whitespace and a leading 'Answer:' echo."""
    text = (text or "").strip()
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text or REFUSAL


def _parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split 'provider:model' -> (provider, model). 'extractive' -> ('extractive', None)."""
    spec = (spec or "extractive").strip()
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.strip().lower(), model.strip()
    return spec.lower(), None


def build_answerer(spec: str = "extractive", **kwargs) -> Answerer:
    """Factory. Parses the reader spec and constructs the right backend.

    Accepted specs:
        extractive               -> ExtractiveAnswerer (zero-LLM, default)
        ollama:<model>           -> OllamaAnswerer (LOCAL, loopback, zero egress)
        openai:<model>           -> APIAnswerer (OpenAI /v1, discloses)
        openrouter:<model>       -> APIAnswerer (OpenRouter /v1, discloses)
        together:<model>         -> APIAnswerer (Together /v1, discloses)
        claude:<model>           -> APIAnswerer (Anthropic /v1/messages, discloses)

    Misconfigured specs fail loud (ValueError) rather than silently degrading.
    No cloud call is made at construction time — only on .answer().
    """
    provider, model = _parse_spec(spec)

    if provider == "extractive":
        return ExtractiveAnswerer(**kwargs)

    if provider == "ollama":
        if not model:
            raise ValueError("ollama answerer requires a model: ollama:<model>")
        return OllamaAnswerer(model, **kwargs)

    if provider in _OPENAI_COMPAT or provider in _ANTHROPIC:
        if not model:
            raise ValueError(f"{provider} answerer requires a model: {provider}:<model>")
        return APIAnswerer(provider, model)

    raise ValueError(
        f"unknown answerer spec {spec!r}; expected 'extractive', "
        f"'ollama:<model>', or one of "
        f"{sorted(set(_OPENAI_COMPAT) | set(_ANTHROPIC))}:<model>"
    )
