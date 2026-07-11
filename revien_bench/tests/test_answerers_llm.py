"""
Unit tests for the LLM answerer backends. OFFLINE: the HTTP layer is mocked, so
no network call ever happens. Proves:

  * build_answerer wiring for every spec (extractive | ollama:* | openai:* |
    openrouter:* | together:* | claude:*).
  * the frozen, hashed prompt loads (sha256 gate) and assembles with the
    retrieved context + question + the concise-answer / refusal instructions.
  * OllamaAnswerer drives the native /api/chat shape and returns a clean answer.
  * APIAnswerer drives both the OpenAI-compatible /v1/chat/completions shape and
    the Anthropic /v1/messages shape, parsing each response correctly.
  * cloud providers disclose; the local ollama reader does not.
  * absent API key fails loud (no silent fabrication).
"""

import json

import pytest

from revien_bench import answerers as A


# ── prompt: frozen hash gate + assembly ──────────────────────────────────────
def test_prompt_loads_and_hash_matches():
    template = A.load_answer_prompt()
    assert "{context}" in template
    assert "{question}" in template
    assert "CONCISELY" in template
    assert "No information available" in template


def test_assemble_prompt_wires_context_and_question():
    ctx = A.RetrievedContext(
        query="What database did we deploy on?",
        contents=["We deployed the backend on PostgreSQL.", "Bob prefers JWT."],
        labels=["db-choice", "auth"],
    )
    prompt = A.assemble_prompt(ctx)
    assert "PostgreSQL" in prompt              # context embedded
    assert "What database did we deploy on?" in prompt  # question embedded
    assert "CONCISELY" in prompt               # concise instruction present
    assert "No information available" in prompt  # refusal instruction present
    assert "{context}" not in prompt and "{question}" not in prompt  # fully filled


def test_assemble_prompt_empty_context_falls_back():
    ctx = A.RetrievedContext(query="anything?", contents=[], labels=[])
    prompt = A.assemble_prompt(ctx)
    assert "no memory context retrieved" in prompt


# ── factory wiring ────────────────────────────────────────────────────────────
def test_build_answerer_specs():
    assert type(A.build_answerer("extractive")).__name__ == "ExtractiveAnswerer"
    assert A.build_answerer("ollama:llama3").name == "ollama:llama3"
    assert A.build_answerer("openai:gpt-4o-mini").name == "openai:gpt-4o-mini"
    assert A.build_answerer("openrouter:x/y").name == "openrouter:x/y"
    assert A.build_answerer("together:x/y").name == "together:x/y"
    assert A.build_answerer("claude:claude-3-5-haiku").name == "claude:claude-3-5-haiku"


def test_build_answerer_bad_specs_fail_loud():
    with pytest.raises(ValueError):
        A.build_answerer("frobnicate:model")
    with pytest.raises(ValueError):
        A.build_answerer("ollama")  # missing model
    with pytest.raises(ValueError):
        A.build_answerer("openai")  # missing model


# ── Ollama: native /api/chat, mocked transport ───────────────────────────────
def test_ollama_answerer_parses_chat(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"message": {"role": "assistant", "content": "Answer: PostgreSQL"}}

    monkeypatch.setattr(A, "_http_post_json", fake_post)
    ans = A.build_answerer("ollama:llama3")
    ctx = A.RetrievedContext(query="What database?",
                             contents=["We use PostgreSQL."], labels=["db"])
    out = ans.answer(ctx)

    assert out == "PostgreSQL"  # 'Answer:' echo stripped by _clean_answer
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "llama3"
    assert captured["payload"]["stream"] is False
    # The frozen prompt (with context + question) is what we sent.
    sent = captured["payload"]["messages"][0]["content"]
    assert "PostgreSQL" in sent and "What database?" in sent and "CONCISELY" in sent


def test_ollama_does_not_disclose(monkeypatch, capsys):
    A._DISCLOSED_PROVIDERS.clear()
    monkeypatch.setattr(A, "_http_post_json",
                        lambda url, payload, headers: {"message": {"content": "ok"}})
    A.build_answerer("ollama:llama3").answer(
        A.RetrievedContext(query="q", contents=["c"], labels=["l"])
    )
    assert "leaves your machine" not in capsys.readouterr().err


# ── OpenAI-compatible: /v1/chat/completions, mocked transport ─────────────────
def test_openai_answerer_parses_choices(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "Redis"}}]}

    monkeypatch.setattr(A, "_http_post_json", fake_post)
    out = A.build_answerer("openai:gpt-4o-mini").answer(
        A.RetrievedContext(query="Where are tokens stored?",
                           contents=["Tokens live in Redis."], labels=["store"])
    )
    assert out == "Redis"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-4o-mini"


def test_openrouter_uses_its_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    captured = {}
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: captured.update(url=url, h=headers) or {
            "choices": [{"message": {"content": "x"}}]
        },
    )
    A.build_answerer("openrouter:meta/llama").answer(
        A.RetrievedContext(query="q", contents=["c"], labels=["l"])
    )
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["h"]["Authorization"] == "Bearer or-test"


# ── Anthropic: /v1/messages, mocked transport ─────────────────────────────────
def test_claude_answerer_parses_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "PostgreSQL"}]}

    monkeypatch.setattr(A, "_http_post_json", fake_post)
    out = A.build_answerer("claude:claude-3-5-haiku-20241022").answer(
        A.RetrievedContext(query="What database?",
                           contents=["We use PostgreSQL."], labels=["db"])
    )
    assert out == "PostgreSQL"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-3-5-haiku-20241022"


def test_cloud_discloses_once(monkeypatch, capsys):
    A._DISCLOSED_PROVIDERS.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        A, "_http_post_json",
        lambda url, payload, headers: {"choices": [{"message": {"content": "x"}}]},
    )
    ans = A.build_answerer("openai:gpt-4o-mini")
    ctx = A.RetrievedContext(query="q", contents=["c"], labels=["l"])
    ans.answer(ctx)
    ans.answer(ctx)  # second call must NOT re-disclose
    err = capsys.readouterr().err
    assert err.count("leaves your machine") == 1


def test_missing_api_key_fails_loud(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Disclosure happens before the key check; suppress the actual POST so we
    # only exercise the key-absence path.
    monkeypatch.setattr(A, "_http_post_json", lambda url, payload, headers: {})
    ans = A.build_answerer("openai:gpt-4o-mini")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        ans.answer(A.RetrievedContext(query="q", contents=["c"], labels=["l"]))


def test_empty_llm_answer_becomes_refusal():
    assert A._clean_answer("") == A.REFUSAL
    assert A._clean_answer("   ") == A.REFUSAL
    assert A._clean_answer("Answer:  Madrid") == "Madrid"
