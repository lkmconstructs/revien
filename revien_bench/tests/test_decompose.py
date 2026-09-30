"""
Tests for the benchmark-only --decompose row (revien_bench.decompose).
OFFLINE: the HTTP layer is mocked (same way as test_judges.py).
"""

import json
from types import SimpleNamespace

import pytest

from revien_bench import answerers as A
from revien_bench import decompose as D
from revien_bench import runner as R
from revien_bench import sovereignty as S
from revien_bench.tests.test_checkpoint_resume import _two_convs, _FAKE_SHA

Q = "What did Alice and Bob each cook last summer?"


# ── factory ──────────────────────────────────────────────────────────────────
def test_build_none_is_none():
    for spec in ("none", "NONE", "", None):
        assert D.build_decomposer(spec) is None


def test_build_specs_and_bad_specs():
    assert D.build_decomposer("ollama:llama3").name == "ollama:llama3"
    assert D.build_decomposer("openai:gpt-4o-mini").name == "openai:gpt-4o-mini"
    assert D.build_decomposer("claude:haiku").name == "claude:haiku"
    for bad in ("frobnicate:m", "ollama", "openai"):
        with pytest.raises(ValueError):
            D.build_decomposer(bad)


# ── parse ────────────────────────────────────────────────────────────────────
def test_parse_strips_bullets_numbers_and_blanks():
    raw = "1. What did Alice cook?\n\n- What did Bob cook?\n* When was summer?\n"
    assert D.parse_subqueries(raw, Q) == [
        Q, "What did Alice cook?", "What did Bob cook?", "When was summer?"]


def test_parse_caps_at_three_subqueries():
    raw = "\n".join(f"sub question {i}" for i in range(6))
    out = D.parse_subqueries(raw, Q)
    assert out[0] == Q and len(out) == 4


def test_parse_atomic_passthrough_and_original_always_first():
    assert D.parse_subqueries(Q, Q) == [Q]
    assert D.parse_subqueries(Q.upper(), Q) == [Q]  # dupe of original dropped
    assert D.parse_subqueries("", Q) == [Q]
    assert D.parse_subqueries("a b c\na b c", Q) == [Q, "a b c"]  # dedupe


# ── frozen prompt ────────────────────────────────────────────────────────────
def test_prompt_sha_gate(monkeypatch):
    assert "{question}" in D.load_decompose_prompt()
    assert Q in D.assemble_decompose_prompt(Q)
    monkeypatch.setattr(D, "DECOMPOSE_PROMPT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="drifted"):
        D.load_decompose_prompt()


# ── cloud transport: counts, cost, discloses once; errors fall back ──────────
def _openai_reply(text):
    return lambda url, payload, headers: {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 20},
    }


def test_cloud_decomposer_counts_cost_and_discloses_once(monkeypatch, capsys):
    D._DISCLOSED_DECOMPOSE_PROVIDERS.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(A, "_http_post_json", _openai_reply("Alice dish?\nBob dish?"))
    dec = D.build_decomposer("openai:gpt-4o-mini")
    r1 = dec.decompose(Q)
    dec.decompose(Q)
    assert r1.queries == [Q, "Alice dish?", "Bob dish?"]
    assert r1.network_calls == 1 and r1.cost_usd > 0 and r1.error is None
    assert dec.network_calls == 2
    assert dec.cost_usd_estimate == pytest.approx(2 * r1.cost_usd)
    err = capsys.readouterr().err
    assert err.count("leaves your machine") == 1
    assert "to split it into sub-questions" in err


def test_cloud_decomposer_error_falls_back_and_still_counts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def boom(url, payload, headers):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(A, "_http_post_json", boom)
    dec = D.build_decomposer("openai:gpt-4o-mini")
    r = dec.decompose(Q)
    assert r.queries == [Q] and r.error and "HTTP 500" in r.error
    assert r.network_calls == 1 and dec.network_calls == 1


def test_missing_key_falls_back_without_counting(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dec = D.build_decomposer("openai:gpt-4o-mini")
    r = dec.decompose(Q)
    assert r.queries == [Q] and r.error and dec.network_calls == 0


def test_empty_reply_is_an_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(A, "_http_post_json", _openai_reply(""))
    r = D.build_decomposer("openai:x").decompose(Q)
    assert r.queries == [Q] and r.error


# ── ollama: local, never counted, never disclosed ────────────────────────────
def test_ollama_decomposer_local(monkeypatch, capsys):
    D._DISCLOSED_DECOMPOSE_PROVIDERS.clear()
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {"message": {"content": "a?\nb?"}})
    dec = D.build_decomposer("ollama:llama3")
    r = dec.decompose(Q)
    assert r.queries == [Q, "a?", "b?"]
    assert dec.network_calls == 0 and r.cost_usd == 0.0
    assert "leaves your machine" not in capsys.readouterr().err


def test_ollama_loopback_rule(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    assert S.network_egress_zero(decompose="ollama:m").passed
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    chk = S.network_egress_zero(decompose="ollama:m")
    assert not chk.passed
    assert "decompose=ollama host is not loopback" in chk.detail["cloud_backends"]


# ── egress ───────────────────────────────────────────────────────────────────
def test_egress_cloud_decomposer_fails_naming_itself():
    chk = S.network_egress_zero(cloud_calls=0, decompose="openrouter:some/model")
    assert not chk.passed
    assert "decompose=openrouter" in chk.detail["cloud_backends"]
    assert chk.detail["decompose_local"] is False


def test_egress_none_and_default_pass():
    assert S.network_egress_zero(decompose="none").passed
    assert S.network_egress_zero().passed


# ── fusion ───────────────────────────────────────────────────────────────────
def _res(nid, score):
    return SimpleNamespace(node_id=nid, score=score)


def _resp(results, scores=None, filtered=None, anchors=None):
    diag = {"scores": scores or {}, "filtered": filtered or {},
            "anchors": {"all": anchors or []}}
    return SimpleNamespace(results=results, diagnostics=diag)


def test_fuse_keeps_best_score_sorts_and_truncates():
    a = _resp([_res("n1", 0.5), _res("n2", 0.4)],
              scores={"n1": 0.5, "n2": 0.4}, filtered={"n9": "stale"}, anchors=["x"])
    b = _resp([_res("n2", 0.9), _res("n3", 0.45)],
              scores={"n2": 0.9, "n3": 0.45}, filtered={"n9": "other", "n8": "dup"},
              anchors=["x", "y"])
    results, diag = R._fuse_responses([a, b], top_n=3)
    assert [(r.node_id, r.score) for r in results] == [("n2", 0.9), ("n1", 0.5), ("n3", 0.45)]
    results2, _ = R._fuse_responses([a, b], top_n=2)
    assert [r.node_id for r in results2] == ["n2", "n1"]
    assert diag["scores"]["n2"] == 0.9
    assert diag["filtered"] == {"n9": "stale", "n8": "dup"}  # first recall wins
    assert diag["anchors"]["all"] == ["x", "y"]


def test_fuse_with_no_diagnostics():
    a = SimpleNamespace(results=[_res("n1", 0.5)], diagnostics=None)
    b = SimpleNamespace(results=[_res("n2", 0.6)], diagnostics=None)
    results, diag = R._fuse_responses([a, b], top_n=10)
    assert [r.node_id for r in results] == ["n2", "n1"] and diag is None


# ── fingerprint ──────────────────────────────────────────────────────────────
def test_fingerprint_differs_and_none_is_unchanged():
    base = R._run_fingerprint()
    assert R._run_fingerprint("none") == base
    assert R._run_fingerprint("openai:x") != base
    assert R._run_fingerprint("openai:x") != R._run_fingerprint("openai:y")
    from pathlib import Path
    assert (R._checkpoint_path(Path("o"), "c", "a", "f1", "openai:x")
            != R._checkpoint_path(Path("o"), "c", "a", "f1"))


# ── runner ───────────────────────────────────────────────────────────────────
class _Stub:
    def __init__(self, extra=()):
        self.name = "stub:dec"
        self.network_calls = 0
        self.cost_usd_estimate = 0.0
        self.extra = list(extra)

    def decompose(self, question):
        self.network_calls += 1
        self.cost_usd_estimate += 0.002
        return D.Decomposition(
            queries=[question] + self.extra, network_calls=1, cost_usd=0.002)


def _run(tmp_path, monkeypatch, name, stub=None):
    monkeypatch.setattr(R, "load_locomo", lambda _p: _two_convs())
    monkeypatch.setattr(R, "read_locked_hash", lambda: _FAKE_SHA)
    if stub is not None:
        monkeypatch.setattr(R.D, "build_decomposer", lambda _s: stub)
    return R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=tmp_path / "ds.json", out_dir=tmp_path / name,
        fresh=True, decompose_name="stub:dec" if stub is not None else "none",
    )


_VOLATILE = {"recall_latency_ms", "answer_latency_ms", "decompose_latency_ms",
             "n_subqueries", "decompose_error"}


def _stable(rows):
    return [{k: v for k, v in r.items() if k not in _VOLATILE} for r in rows]


def test_runner_identity_stub_matches_no_decompose(tmp_path, monkeypatch):
    base = _run(tmp_path, monkeypatch, "base")
    dec = _run(tmp_path, monkeypatch, "dec", _Stub())
    assert _stable(dec["per_question"]) == _stable(base["per_question"])
    assert "decompose" not in base
    assert base["config"]["decompose"] == "none"
    assert all(r["n_subqueries"] == 1 for r in dec["per_question"])


def test_runner_block_counts_and_egress(tmp_path, monkeypatch):
    stub = _Stub(extra=["Where did Alice go?"])
    rep = _run(tmp_path, monkeypatch, "blk", stub)
    n = rep["n_questions"]
    block = rep["decompose"]
    assert block["spec"] == "stub:dec" and block["model"] == "stub:dec"
    assert block["prompt_sha256"] == D.DECOMPOSE_PROMPT_SHA256
    assert block["network_calls"] == n == stub.network_calls
    assert block["cost_usd"] == pytest.approx(0.002 * n)
    assert block["cost_usd_is_estimate"] is True
    assert block["decompose_errors"] == 0
    assert block["mean_subqueries"] == 2.0
    assert block["taxonomy_basis"] == "merged_subrecall_diagnostics"
    # Top-level totals include the decomposer.
    assert rep["network_calls"] == n
    assert rep["cost_usd"] == pytest.approx(0.002 * n)
    # Persisted JSON carries the block.
    written = sorted((tmp_path / "blk").glob("*.json"))
    assert json.loads(written[-1].read_text(encoding="utf-8"))["decompose"]["network_calls"] == n
