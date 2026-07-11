"""Trigger 2 LLM recognizer — offline tests (mapping, conservatism, degraded-safety).

The network path is verified against gpt-4.1 in measure_sensitivity_llm.py + the
adversarial pass; these pin the parts that must hold WITHOUT a backend: the
conservative verdict mapping and the degraded-safety fallback.
"""

import pytest

from revien.sensitivity import SensitivityRoute
from revien.sensitivity_llm import LLMSensitivityRecognizer


def _stub(monkeypatch, word):
    r = LLMSensitivityRecognizer(backend="openai")
    monkeypatch.setattr(r, "api_key", "x")          # pretend available
    monkeypatch.setattr(r, "_classify", lambda text: word)
    return r


def test_neutral_is_the_only_auto_clear(monkeypatch):
    """Only a clean NEUTRAL routes confident_neutral (auto-eligible)."""
    assert _stub(monkeypatch, "NEUTRAL").recognize("x").route is SensitivityRoute.CONFIDENT_NEUTRAL


def test_sensitive_routes_candidate(monkeypatch):
    assert _stub(monkeypatch, "SENSITIVE").recognize("x").route is SensitivityRoute.SENSITIVE


@pytest.mark.parametrize("word", ["UNSURE", None, "garbage", ""])
def test_unsure_or_unparseable_abstains(monkeypatch, word):
    """Conservative: anything that is not a clean NEUTRAL/SENSITIVE abstains (candidate)."""
    v = _stub(monkeypatch, word).recognize("x")
    assert v.route is SensitivityRoute.ABSTAIN
    assert v.routes_candidate


def test_no_key_is_unavailable_and_abstains(monkeypatch):
    """Degraded-safety: a cloud backend with no key is unavailable -> abstain, never auto."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = LLMSensitivityRecognizer(backend="openai")
    assert r.is_available() is False
    v = r.recognize("I love being off the bottle.")
    assert v.route is SensitivityRoute.ABSTAIN
    assert v.available is False


def test_backend_failure_marks_broken_and_abstains(monkeypatch):
    """A runtime call failure -> abstain + broken (so the gate stays fail-safe)."""
    r = LLMSensitivityRecognizer(backend="openai")
    monkeypatch.setattr(r, "api_key", "x")
    def boom(text):
        raise RuntimeError("network down")
    monkeypatch.setattr(r, "_classify", boom)
    v = r.recognize("anything")
    assert v.route is SensitivityRoute.ABSTAIN
    assert r.is_available() is False  # marked broken


def test_default_backend_is_local_for_production_safety():
    """Production default is local ollama (zero-cloud) unless explicitly overridden."""
    import os
    if os.environ.get("REVIEN_SENSITIVITY_BACKEND"):
        pytest.skip("backend overridden by env")
    assert LLMSensitivityRecognizer().backend == "ollama"
