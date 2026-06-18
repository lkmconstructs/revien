"""
Egress-honesty + network/cost accounting tests. OFFLINE: the HTTP layer is
mocked, NO real API call ever happens, NO key is required for the egress-check
assertions. Proves the credibility bug is fixed:

  * all-local config (rule/fastembed/extractive)  -> egress PASS, cloud_backends
    empty, network_calls 0, cost_usd 0.0.
  * a cloud ANSWERER (openai) -> egress FAIL naming 'openai', even when the
    measured cloud_calls counter is 0 (config-derived, not HTTP-derived).
  * a cloud EXTRACTOR / EMBEDDER (openai) -> egress FAIL naming that backend.
  * the APIAnswerer self-counts network_calls (>0) and accumulates a labelled
    cost ESTIMATE (>0) over a mocked transport — no real request.
  * local readers (extractive / ollama) report network_calls 0 and $0.0.
"""

import os

import pytest

from revien_bench import answerers as A
from revien_bench import sovereignty as S


# ── helper: temporarily set the backend env, always restored ──────────────────
def _with_env(**kv):
    class _Ctx:
        def __enter__(self):
            self._prev = {k: os.environ.get(k) for k in kv}
            for k, v in kv.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, *exc):
            for k, v in self._prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Ctx()


# ── 1. all-local config: PASS / no cloud backends / 0 / $0 ────────────────────
def test_egress_all_local_passes():
    with _with_env(REVIEN_EXTRACTOR="rule", REVIEN_EMBEDDER="fastembed"):
        check = S.network_egress_zero(cloud_calls=0, answerer="extractive")
    assert check.passed, check.detail
    assert check.detail["cloud_backends"] == []
    assert check.detail["all_backends_local"] is True
    assert check.detail["cloud_calls"] == 0


def test_egress_local_ollama_answerer_passes():
    # ollama reader is loopback-local; still a PASS.
    with _with_env(REVIEN_EXTRACTOR="rule", REVIEN_EMBEDDER="fastembed"):
        check = S.network_egress_zero(cloud_calls=0, answerer="ollama:llama3")
    assert check.passed, check.detail
    assert check.detail["cloud_backends"] == []


# ── 2. cloud ANSWERER: FAIL naming 'openai' even with 0 measured calls ────────
def test_egress_cloud_answerer_fails_naming_openai():
    # THE credibility bug: cloud answerer, but cloud_calls happens to read 0.
    # Must STILL fail and name 'openai' — the decision is config-derived.
    with _with_env(REVIEN_EXTRACTOR="rule", REVIEN_EMBEDDER="fastembed"):
        check = S.network_egress_zero(cloud_calls=0, answerer="openai:gpt-4o-mini")
    assert not check.passed, check.detail
    assert any("answerer=openai" in b for b in check.detail["cloud_backends"]), check.detail
    assert check.detail["answerer"] == "openai"
    assert check.detail["answerer_local"] is False


def test_egress_cloud_answerer_fails_with_positive_calls():
    with _with_env(REVIEN_EXTRACTOR="rule", REVIEN_EMBEDDER="fastembed"):
        check = S.network_egress_zero(cloud_calls=12, answerer="claude:haiku")
    assert not check.passed, check.detail
    assert any("answerer=claude" in b for b in check.detail["cloud_backends"])


# ── 3. cloud EXTRACTOR / EMBEDDER also fail and are named ─────────────────────
def test_egress_cloud_extractor_fails_naming_it():
    with _with_env(REVIEN_EXTRACTOR="openai", REVIEN_EMBEDDER="fastembed"):
        check = S.network_egress_zero(cloud_calls=0, answerer="extractive")
    assert not check.passed, check.detail
    assert any("extractor=openai" in b for b in check.detail["cloud_backends"])


def test_egress_cloud_embedder_fails_naming_it():
    with _with_env(REVIEN_EXTRACTOR="rule", REVIEN_EMBEDDER="openai"):
        check = S.network_egress_zero(cloud_calls=0, answerer="extractive")
    assert not check.passed, check.detail
    assert any("embedder=openai" in b for b in check.detail["cloud_backends"])


# ── 4. network/cost accounting on the answerers (mocked transport) ────────────
def test_local_readers_report_zero_calls_and_cost():
    ext = A.build_answerer("extractive")
    assert ext.network_calls == 0 and ext.cost_usd_estimate == 0.0
    oll = A.build_answerer("ollama:llama3")
    assert oll.network_calls == 0 and oll.cost_usd_estimate == 0.0


def test_cloud_answerer_counts_calls_and_estimates_cost(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Mocked transport returns a usage block; NO real HTTP.
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {
            "choices": [{"message": {"content": "Redis"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
    )
    ans = A.build_answerer("openai:gpt-4o-mini")
    ctx = A.RetrievedContext(query="q", contents=["c"], labels=["l"])
    ans.answer(ctx)
    ans.answer(ctx)
    assert ans.network_calls == 2          # counted both mocked calls
    assert ans.cost_usd_estimate > 0.0     # non-zero labelled estimate
    # 2 calls * (1000/1k * 0.00015 + 500/1k * 0.00060) = 2 * 0.00045 = 0.0009
    assert ans.cost_usd_estimate == pytest.approx(0.0009, rel=1e-6)


def test_cost_estimate_falls_back_without_usage_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {"content": [{"type": "text", "text": "ok"}]},
    )
    ans = A.build_answerer("claude:haiku")
    ans.answer(A.RetrievedContext(query="q", contents=["c"], labels=["l"]))
    assert ans.network_calls == 1
    assert ans.cost_usd_estimate > 0.0  # estimated from char-based token approx


def test_estimate_cost_local_provider_is_zero():
    assert A.estimate_cost_usd("extractive", 9999, 9999) == 0.0
    assert A.estimate_cost_usd("ollama", 9999, 9999) == 0.0
    assert A.estimate_cost_usd("openai", 1000, 0) == pytest.approx(0.00015, rel=1e-6)
