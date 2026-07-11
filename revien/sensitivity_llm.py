"""Claim Sovereignty Layer — Trigger 2: LLM semantic sensitive recognition.

The embedding recognizer (revien/sensitivity.py) was adversarially proven to key on
surface construction, not meaning (96.7% stream abstention to reach zero leaks).
This recognizer reasons about MEANING: it asks an LLM the one question that defines
cost-of-erasure — "would silently auto-erasing this statement betray the user?" —
and routes SENSITIVE / NEUTRAL / UNSURE, where UNSURE (and any failure) abstains.

Backend-pluggable, mirroring revien/ingestion/extractor_llm.py: LOCAL ollama
(zero-cloud, the production default for a SAFETY path) or CLOUD openai/anthropic
(opt-in, verification/reference). A sensitivity recognizer that sends claims to the
cloud is itself an egress of possibly-sensitive content — cloud backends disclose
loudly; ship local ollama in production.

Same interface as SemanticSensitivityRecognizer (recognize -> SensitivityVerdict),
so it drops into SupersessionGate(recognizer=...). Degraded-safety: any backend
failure -> the recognizer is unavailable and abstains, so the gate routes every
would-be-auto to candidate (no auto-supersede without a working recognizer).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Optional

from revien.sensitivity import SensitivityRoute, SensitivityVerdict

REQUEST_TIMEOUT = 30.0
LOCAL_BACKENDS = ("ollama",)
CLOUD_BACKENDS = ("openai", "openrouter", "anthropic")

_BACKEND_DEFAULTS = {
    "ollama": {"url": os.environ.get("REVIEN_OLLAMA_URL", "http://localhost:11434/api/chat"),
               "model": "llama3.1", "key_env": None},
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "model": "gpt-4.1", "key_env": "OPENAI_API_KEY"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "model": "openai/gpt-4.1", "key_env": "OPENROUTER_API_KEY"},
    "anthropic": {"url": "https://api.anthropic.com/v1/messages",
                  "model": "claude-3-5-haiku-latest", "key_env": "ANTHROPIC_API_KEY"},
}

SYSTEM_PROMPT = """You are a privacy safety filter for a personal, user-owned memory system. The system may AUTOMATICALLY overwrite or delete a stored statement when a newer statement appears to contradict it, with NO human review.

Decide whether silently auto-erasing the given statement would risk BETRAYING the user. A statement is SENSITIVE if it is a vulnerable personal self-disclosure whose quiet erasure would feel like a betrayal -- for example (NOT exhaustive): addiction/recovery/sobriety, gender identity or transition, sexual orientation or coming out, mental health, medical diagnoses or treatment, pregnancy/fertility/miscarriage/abortion, immigration or legal status, criminal record/incarceration/parole, financial hardship/bankruptcy/debt, religious belief or loss of faith, family estrangement, abuse or trauma, or ANY comparably vulnerable disclosure even in a domain not listed here.

Recognize MEANING, not keywords. Catch euphemisms ("I picked up my one-year chip", "I'm finally clean"), indirect or implied disclosures ("the test came back positive", "I haven't spoken to my dad in years", "the transfer didn't take"), and disclosures buried in procedural, logistical, or upbeat phrasing ("my lawyer is filing the bankruptcy paperwork", "I love being off the bottle").

Two patterns are especially easy to misread as mundane: (1) IN-GROUP CODED LANGUAGE -- ordinary words carrying a second, community-specific meaning that quietly flags a protected identity (insider shorthand for a sexual orientation, a faith status or exit, a recovery program, a gender transition, or a diagnosis). (2) AVOIDANCE AND SAFETY LOGISTICS -- routes deliberately not driven, places or checkpoints avoided, documents kept at someone else's address, scripts for handling calls from officials, or precautions that only make sense if the speaker is in a precarious or hidden status (undocumented, fleeing abuse, under supervision). When a statement plausibly fits either pattern, answer UNSURE rather than NEUTRAL. Do NOT flag the same words when they are plainly literal in context (an actual relative, an ordinary traffic detour, a person who happens to be named).

Be conservative. Many statements are sensitive ONLY in light of who the speaker is or context you cannot see. For example: "I have my interview at the office next month" could be an ordinary job interview OR an immigration/asylum interview; "I brought my girlfriend home to meet my parents" could be an ordinary relationship milestone OR a coming-out; "I'm starting a new round on Thursday" could be a board game OR chemotherapy. When a statement could plausibly be a vulnerable disclosure under SOME reasonable speaker context, answer UNSURE -- do NOT default to NEUTRAL.

Reserve NEUTRAL for statements that are mundane under ANY reasonable context (chores, work tasks, food, weather, hobbies, scheduling, software builds, shopping). If you have to imagine a specific innocent backstory to call it neutral, it is not NEUTRAL -- it is UNSURE. The cost of a wrong NEUTRAL (silently erasing a real disclosure) is far higher than a wrong UNSURE (an extra human review).

Answer with EXACTLY ONE WORD: SENSITIVE, NEUTRAL, or UNSURE."""

_DISCLOSED: set = set()
_VERDICT_RE = re.compile(r"\b(SENSITIVE|NEUTRAL|UNSURE)\b", re.I)


def _disclose_cloud(provider: str) -> None:
    if provider in _DISCLOSED:
        return
    _DISCLOSED.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending CLAIM TEXT to {provider} to judge its "
        f"sensitivity - this leaves your machine and may itself be a sensitive "
        f"disclosure. Use REVIEN_SENSITIVITY_BACKEND=ollama (local) in production.\n")
    sys.stderr.flush()


class LLMSensitivityRecognizer:
    """LLM cost-of-erasure recognizer. recognize(text) -> SensitivityVerdict."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        backend = (backend or os.environ.get("REVIEN_SENSITIVITY_BACKEND", "ollama")).lower().strip()
        if backend not in _BACKEND_DEFAULTS:
            backend = "ollama"
        cfg = _BACKEND_DEFAULTS[backend]
        self.backend = backend
        self.url = cfg["url"]
        self.model = model or os.environ.get("REVIEN_SENSITIVITY_MODEL", cfg["model"])
        self.key_env = cfg["key_env"]
        self.api_key = os.environ.get(self.key_env, "") if self.key_env else ""
        self.is_cloud = backend in CLOUD_BACKENDS
        # Subclass surface: a sibling recognizer (e.g. the tension
        # recognizer) overrides these to reuse the transport unchanged.
        self.system_prompt = SYSTEM_PROMPT
        self.verdict_re = _VERDICT_RE
        self.max_tokens = 4
        self.disclosure_purpose = "judge its sensitivity"
        self.backend_env = "REVIEN_SENSITIVITY_BACKEND"
        self._broken = False
        self.network_calls = 0

    def is_available(self) -> bool:
        if self._broken:
            return False
        if self.key_env and not self.api_key:
            return False
        return True

    # ── routing ───────────────────────────────────────────────────────────────
    def recognize(self, text: str) -> SensitivityVerdict:
        if not (text and text.strip()):
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=self.is_available())
        if not self.is_available():
            # No backend -> cannot reason -> abstain (degraded-safety, never auto).
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=False)
        try:
            word = self._classify(text.strip())
        except Exception:  # noqa: BLE001 - any failure -> abstain, mark broken
            self._broken = True
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=False)
        # Conservative mapping: ONLY a clean NEUTRAL auto-clears. SENSITIVE ->
        # candidate; UNSURE or any unparseable answer -> abstain -> candidate.
        if word == "NEUTRAL":
            return SensitivityVerdict(SensitivityRoute.CONFIDENT_NEUTRAL, 0.0, 0.0, available=True)
        if word == "SENSITIVE":
            return SensitivityVerdict(SensitivityRoute.SENSITIVE, 0.0, 0.0, available=True)
        return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=True)

    def _classify(self, text: str) -> Optional[str]:
        if self.is_cloud:
            _disclose_cloud(self.backend, self.disclosure_purpose, self.backend_env)
        self.network_calls += 1
        if self.backend == "anthropic":
            raw = self._call_anthropic(text)
        elif self.backend == "ollama":
            raw = self._call_ollama(text)
        else:
            raw = self._call_openai_compatible(text)
        m = self.verdict_re.search(raw or "")
        return m.group(1).upper() if m else None

    def _call_openai_compatible(self, text: str) -> str:
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": text}]}
        data = self._post(self.url, payload,
                          {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, text: str) -> str:
        payload = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0,
                   "system": self.system_prompt, "messages": [{"role": "user", "content": text}]}
        data = self._post(self.url, payload,
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                           "Content-Type": "application/json"})
        return "".join(b.get("text", "") for b in (data.get("content") or []) if isinstance(b, dict))

    def _call_ollama(self, text: str) -> str:
        payload = {"model": self.model, "stream": False, "options": {"temperature": 0.0},
                   "messages": [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": text}]}
        data = self._post(self.url, payload, {"Content-Type": "application/json"})
        if isinstance(data.get("message"), dict):
            return data["message"].get("content", "")
        return data.get("response", "")

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{self.backend} HTTP {e.code}: {body}") from e
