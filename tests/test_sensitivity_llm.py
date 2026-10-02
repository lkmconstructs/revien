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


# ── Real _classify path (transport mocked, NOT _classify) ───────────────────
# Every test above stubs _classify, so the code inside it never ran under
# test. That hid a TypeError present since 0.3.0: _classify called
# _disclose_cloud(backend, purpose, env) and the helper took one argument, so
# every CLOUD call raised, recognize() swallowed it, and the recognizer
# abstained on everything while looking merely cautious.

def _cloud(monkeypatch, reply, backend="openrouter", cls=LLMSensitivityRecognizer):
    import revien.sensitivity_llm as mod
    monkeypatch.setattr(mod, "_DISCLOSED", set())
    r = cls(backend=backend)
    monkeypatch.setattr(r, "api_key", "x")
    seen = {}

    def fake_post(url, payload, headers):
        seen["payload"] = payload
        return {"choices": [{"message": {"content": reply}}]}

    monkeypatch.setattr(r, "_post", fake_post)
    return r, seen


def test_cloud_classify_path_runs_and_clears_neutral(monkeypatch, capsys):
    r, seen = _cloud(monkeypatch, "NEUTRAL")
    v = r.recognize("Mara drives a 2015 hatchback.")
    assert v.route is SensitivityRoute.CONFIDENT_NEUTRAL
    assert v.available is True
    assert r._broken is False
    assert r.network_calls == 1
    assert "leaves your machine" in capsys.readouterr().err


def test_cloud_classify_path_routes_sensitive(monkeypatch):
    r, _ = _cloud(monkeypatch, "SENSITIVE")
    assert r.recognize("Theo was diagnosed last year.").route is SensitivityRoute.SENSITIVE
    assert r._broken is False


def test_tension_recognizer_cloud_classify_does_not_raise(monkeypatch):
    from revien.tension import LLMTensionRecognizer
    r, _ = _cloud(monkeypatch, "NEUTRAL", cls=LLMTensionRecognizer)
    r._classify("Sam prefers tea. / Sam prefers coffee.")   # must not raise
    assert r.network_calls == 1


def test_disclosure_is_once_per_provider_and_purpose(monkeypatch, capsys):
    from revien.tension import LLMTensionRecognizer
    import revien.sensitivity_llm as mod
    monkeypatch.setattr(mod, "_DISCLOSED", set())
    lines = []
    for cls in (LLMSensitivityRecognizer, LLMTensionRecognizer, LLMSensitivityRecognizer):
        r = cls(backend="openrouter")
        monkeypatch.setattr(r, "api_key", "x")
        monkeypatch.setattr(r, "_post", lambda u, p, h: {"choices": [{"message": {"content": "NEUTRAL"}}]})
        r._classify("x")
    err = capsys.readouterr().err
    assert err.count("WARNING: Revien is sending CLAIM TEXT") == 2   # two purposes, one provider
    assert "REVIEN_SENSITIVITY_BACKEND=ollama" in err and "REVIEN_TENSION_BACKEND=ollama" in err


def test_reply_budget_default_and_override(monkeypatch):
    r, seen = _cloud(monkeypatch, "NEUTRAL")
    r.recognize("x")
    assert seen["payload"]["max_tokens"] >= 64          # reasoning models need room
    monkeypatch.setenv("REVIEN_RECOGNIZER_MAX_TOKENS", "512")
    r2, seen2 = _cloud(monkeypatch, "NEUTRAL")
    r2.recognize("x")
    assert seen2["payload"]["max_tokens"] == 512


def test_empty_cloud_reply_abstains_without_breaking(monkeypatch):
    r, _ = _cloud(monkeypatch, None)        # reasoning model, budget exhausted
    v = r.recognize("x")
    assert v.route is SensitivityRoute.ABSTAIN
