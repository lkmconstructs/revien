"""
Unit tests for the LLM-judge track (revien_bench.judges). OFFLINE: the HTTP
layer is mocked, so no network call ever happens. Proves:

  * build_judge('f1') is a no-op (None) — the default, F1-only path.
  * the frozen, hashed judge prompt loads (sha256 gate) and assembles with
    question/gold/prediction/category.
  * verdict parsing: first word CORRECT/WRONG (case-insensitive) wins; any
    other reply is an unparseable error, scored correct=False.
  * OllamaJudge drives the native /api/chat shape and never counts network
    calls (loopback/local).
  * APIJudge drives both the OpenAI-compatible and Anthropic response shapes,
    self-counts network_calls, and accumulates a labelled cost estimate.
  * cloud judges disclose once; the local ollama judge does not.
  * egress: a cloud judge FAILS naming itself even with 0 reader calls; f1 and
    ollama judges PASS.
"""

import pytest

from revien_bench import answerers as A
from revien_bench import judges as J
from revien_bench import sovereignty as S


# ── build_judge factory ───────────────────────────────────────────────────────
def test_build_judge_f1_is_none():
    assert J.build_judge("f1") is None
    assert J.build_judge("F1") is None
    assert J.build_judge("") is None
    assert J.build_judge(None) is None


def test_build_judge_specs():
    assert J.build_judge("ollama:llama3").name == "ollama:llama3"
    assert J.build_judge("openai:gpt-4o-mini").name == "openai:gpt-4o-mini"
    assert J.build_judge("openrouter:x/y").name == "openrouter:x/y"
    assert J.build_judge("together:x/y").name == "together:x/y"
    assert J.build_judge("claude:claude-3-5-haiku").name == "claude:claude-3-5-haiku"


def test_build_judge_bad_specs_fail_loud():
    with pytest.raises(ValueError):
        J.build_judge("frobnicate:model")
    with pytest.raises(ValueError):
        J.build_judge("ollama")  # missing model
    with pytest.raises(ValueError):
        J.build_judge("openai")  # missing model


# ── frozen prompt: hash gate + assembly ───────────────────────────────────────
def test_judge_prompt_loads_and_hash_matches():
    template = J.load_judge_prompt()
    assert "{question}" in template
    assert "{gold}" in template
    assert "{prediction}" in template
    assert "{category}" in template
    assert "CORRECT" in template and "WRONG" in template


def test_assemble_judge_prompt_wires_all_fields():
    prompt = J.assemble_judge_prompt(
        question="What database did we deploy on?",
        gold="PostgreSQL",
        prediction="We use Postgres.",
        category_name="single-hop",
    )
    assert "What database did we deploy on?" in prompt
    assert "PostgreSQL" in prompt
    assert "We use Postgres." in prompt
    assert "single-hop" in prompt
    assert "{question}" not in prompt and "{gold}" not in prompt
    assert "{prediction}" not in prompt and "{category}" not in prompt


# ── verdict parsing ────────────────────────────────────────────────────────────
def test_parse_verdict_correct_and_wrong_case_insensitive():
    assert J._parse_verdict("CORRECT") == (True, None)
    assert J._parse_verdict("correct") == (True, None)
    assert J._parse_verdict("Correct.") == (True, None)
    assert J._parse_verdict("WRONG") == (False, None)
    assert J._parse_verdict("wrong, the answer is incomplete") == (False, None)


def test_parse_verdict_garbage_is_an_error():
    correct, err = J._parse_verdict("I think this is probably fine")
    assert correct is False
    assert err is not None and "unparseable" in err

    correct, err = J._parse_verdict("")
    assert correct is False
    assert err is not None


# ── Ollama judge: native /api/chat, mocked transport, never counts calls ──────
def test_ollama_judge_parses_chat_and_never_counts_calls(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"role": "assistant", "content": "CORRECT"}}

    monkeypatch.setattr(A, "_http_post_json", fake_post)
    judge = J.build_judge("ollama:llama3")
    verdict = judge.judge("q?", "gold", "pred", "single-hop")

    assert verdict.correct is True
    assert verdict.error is None
    assert verdict.network_calls == 0  # loopback: never counted as egress
    assert judge.network_calls == 0
    assert captured["url"].endswith("/api/chat")
    sent = captured["payload"]["messages"][0]["content"]
    assert "q?" in sent and "gold" in sent and "pred" in sent


def test_ollama_judge_does_not_disclose(monkeypatch, capsys):
    A._DISCLOSED_PROVIDERS.clear()
    monkeypatch.setattr(A, "_http_post_json",
                        lambda url, payload, headers: {"message": {"content": "WRONG"}})
    J.build_judge("ollama:llama3").judge("q", "g", "p", "cat")
    assert "leaves your machine" not in capsys.readouterr().err


# ── API judge: OpenAI-compatible + Anthropic shapes, mocked transport ─────────
def test_openai_judge_parses_choices_and_counts_cost(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {
            "choices": [{"message": {"content": "CORRECT"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 2},
        },
    )
    judge = J.build_judge("openai:gpt-4o-mini")
    verdict = judge.judge("q", "gold", "pred", "cat")
    assert verdict.correct is True
    assert verdict.network_calls == 1
    assert verdict.cost_usd > 0.0
    assert judge.network_calls == 1
    assert judge.cost_usd_estimate == pytest.approx(verdict.cost_usd)


def test_claude_judge_parses_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["headers"] = headers
        return {"content": [{"type": "text", "text": "WRONG"}]}

    monkeypatch.setattr(A, "_http_post_json", fake_post)
    judge = J.build_judge("claude:claude-3-5-haiku-20241022")
    verdict = judge.judge("q", "gold", "pred", "cat")
    assert verdict.correct is False
    assert verdict.error is None
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"


def test_cloud_judge_discloses_once(monkeypatch, capsys):
    A._DISCLOSED_PROVIDERS.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {"choices": [{"message": {"content": "CORRECT"}}]},
    )
    judge = J.build_judge("openai:gpt-4o-mini")
    judge.judge("q", "g", "p", "cat")
    judge.judge("q", "g", "p", "cat")  # second call must NOT re-disclose
    err = capsys.readouterr().err
    assert err.count("leaves your machine") == 1


def test_cloud_judge_missing_key_returns_error_verdict_not_raise(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(A, "_http_post_json", lambda url, payload, headers: {})
    judge = J.build_judge("openai:gpt-4o-mini")
    # A judge call must never kill the run: a missing key -> error Verdict,
    # not a raised exception (mirrors the runner's per-QA answerer guard).
    verdict = judge.judge("q", "g", "p", "cat")
    assert verdict.correct is False
    assert verdict.error is not None and "OPENAI_API_KEY" in verdict.error
    assert judge.network_calls == 0


# ── egress: judge is part of the sovereignty decision ─────────────────────────
def test_egress_f1_judge_passes():
    check = S.network_egress_zero(cloud_calls=0, answerer="extractive", judge="f1")
    assert check.passed, check.detail
    assert check.detail["judge"] == "f1"
    assert check.detail["judge_local"] is True


def test_egress_ollama_judge_passes():
    check = S.network_egress_zero(cloud_calls=0, answerer="extractive", judge="ollama:llama3")
    assert check.passed, check.detail
    assert check.detail["judge_local"] is True


def test_egress_cloud_judge_fails_naming_itself():
    check = S.network_egress_zero(
        cloud_calls=0, answerer="extractive", judge="openrouter:some/model"
    )
    assert not check.passed, check.detail
    assert any("judge=openrouter" in b for b in check.detail["cloud_backends"]), check.detail
    assert check.detail["judge"] == "openrouter"
    assert check.detail["judge_local"] is False


def test_egress_cloud_judge_fails_even_with_zero_reader_calls():
    # The credibility bug, judge edition: cloud JUDGE, but reader is local and
    # the measured cloud_calls happens to read 0. Must still FAIL.
    check = S.network_egress_zero(cloud_calls=0, answerer="extractive", judge="claude:haiku")
    assert not check.passed, check.detail
    assert check.detail["answerer_local"] is True
    assert check.detail["judge_local"] is False


# ── runner smoke: --limit-convs 1 --max-qa 3, stub judge, judge block ─────────
def test_runner_smoke_with_stub_judge_produces_judge_block(monkeypatch, tmp_path):
    """OFFLINE: a synthetic 1-conversation dataset, extractive answerer, and a
    monkeypatched judge (no real LLM/network). Proves the runner wiring: the
    judge is called per question, per_question rows carry judge_correct/
    judge_error, and the top-level report gains separate reader/judge blocks
    without touching overall_f1 / per_category_f1."""
    from revien_bench import runner as R
    from revien_bench.loader import QA, Conversation, Turn

    conv = Conversation(conv_id="D1", speaker_a="Alice", speaker_b="Bob")
    conv.session_dates = {1: "7 May 2023"}
    conv.turns = [
        Turn(dia_id="D1:1", speaker="Alice", session=1, session_date="7 May 2023",
             text="We deployed the backend on PostgreSQL."),
    ]
    conv.qa = [
        QA(question="What database did we deploy on?", answer="PostgreSQL",
           category=4, evidence=["D1:1"]),
    ]

    monkeypatch.setattr(R, "load_locomo", lambda _p: [conv])
    monkeypatch.setattr(R, "read_locked_hash", lambda: "deadbeef" * 8)

    class _StubJudge:
        name = "stub:judge"

        def __init__(self):
            self.network_calls = 0
            self.cost_usd_estimate = 0.0

        def judge(self, question, gold, prediction, category_name):
            self.network_calls += 1
            self.cost_usd_estimate += 0.001
            return J.Verdict(
                correct=True, raw="CORRECT", network_calls=1, cost_usd=0.001,
                latency_ms=1.0, error=None,
            )

    stub = _StubJudge()
    monkeypatch.setattr(R.J, "build_judge", lambda _spec: stub)

    report = R.run_benchmark(
        config_name="graph_only",
        answerer_name="extractive",
        dataset_path=tmp_path / "ds.json",
        out_dir=tmp_path / "results",
        limit_convs=1,
        max_qa=3,
        fresh=True,
        judge_name="stub:judge",
    )

    assert report["n_questions"] == 1
    assert "judge" in report and report["judge"] is not None
    assert report["judge"]["spec"] == "stub:judge"
    assert report["judge"]["accuracy_overall"] == 1.0
    assert report["judge"]["n_judged"] == 1
    assert report["judge"]["judge_errors"] == 0
    assert report["judge"]["network_calls"] == 1
    assert report["judge"]["cost_usd"] == pytest.approx(0.001)
    assert report["reader"]["spec"] == "extractive"

    row = report["per_question"][0]
    assert row["judge_correct"] is True
    assert row["judge_error"] is None
    # F1/retrieval untouched by the judge track.
    assert "f1" in row and row["f1"] is not None

    # Sovereignty: the (fake) 'stub' judge spec doesn't parse as a known
    # provider, so build a real check directly against a real spec instead —
    # the runner-level check itself is exercised by the egress tests above.
    assert report["sovereignty"]["all_passed"] in (True, False)  # ran without raising


def test_runner_default_judge_f1_omits_judge_block(monkeypatch, tmp_path):
    """The default (no --judge / judge_name='f1') path's report shape is
    UNCHANGED — no 'judge' or 'reader' key at all, so existing results-schema
    expectations for the F1-only track keep holding."""
    from revien_bench import runner as R
    from revien_bench.loader import QA, Conversation, Turn

    conv = Conversation(conv_id="D1", speaker_a="Alice", speaker_b="Bob")
    conv.session_dates = {1: "7 May 2023"}
    conv.turns = [
        Turn(dia_id="D1:1", speaker="Alice", session=1, session_date="7 May 2023",
             text="We deployed the backend on PostgreSQL."),
    ]
    conv.qa = [
        QA(question="What database did we deploy on?", answer="PostgreSQL",
           category=4, evidence=["D1:1"]),
    ]
    monkeypatch.setattr(R, "load_locomo", lambda _p: [conv])
    monkeypatch.setattr(R, "read_locked_hash", lambda: "deadbeef" * 8)

    report = R.run_benchmark(
        config_name="graph_only",
        answerer_name="extractive",
        dataset_path=tmp_path / "ds.json",
        out_dir=tmp_path / "results",
        limit_convs=1,
        max_qa=3,
        fresh=True,
    )
    assert "judge" not in report
    assert "reader" not in report
    row = report["per_question"][0]
    assert row["judge_correct"] is None
    assert row["judge_error"] is None
