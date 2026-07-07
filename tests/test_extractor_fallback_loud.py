"""
OPEN 3: the LLM extractor's fallback must be LOUD in aggregate, not just
per-call. A sustained backend outage (the quota-429 leak class) escalates
once per outage with an unmissable "ingestion is effectively rule-based"
warning, and extractor.status() exposes the degraded state for daemons.
"""

import pytest

from revien.ingestion.extractor_llm import LLMExtractor


@pytest.fixture
def broken_extractor(monkeypatch):
    ex = LLMExtractor(backend="ollama")

    def boom(_content):
        raise ConnectionError("backend down")

    monkeypatch.setattr(ex, "_call_model", boom)
    return ex


class TestLoudFallback:
    def test_escalates_once_after_consecutive_failures(self, broken_extractor, capsys):
        for i in range(broken_extractor.ESCALATE_AFTER + 2):
            result = broken_extractor.extract(f"User: turn {i}", source_id=f"t{i}")
            assert result.context_node is not None  # fallback still extracts
        err = capsys.readouterr().err
        assert err.count("EFFECTIVELY RULE-BASED") == 1, \
            "escalation fires exactly once per outage, not per call"
        assert broken_extractor.status()["degraded"] is True
        assert broken_extractor.status()["total_fallbacks"] == \
            broken_extractor.ESCALATE_AFTER + 2

    def test_success_resets_and_rearms(self, broken_extractor, capsys, monkeypatch):
        for i in range(broken_extractor.ESCALATE_AFTER):
            broken_extractor.extract(f"User: turn {i}")
        assert broken_extractor.status()["degraded"] is True

        # Backend recovers for one call...
        monkeypatch.setattr(
            broken_extractor, "_call_model",
            lambda _c: {"entities": [], "facts": [], "decisions": [],
                        "preferences": [], "topics": []},
        )
        broken_extractor.extract("User: healthy turn")
        assert broken_extractor.status()["degraded"] is False
        assert broken_extractor.consecutive_fallbacks == 0

        # ...then dies again: escalation must re-arm and fire a second time.
        capsys.readouterr()

        def boom(_c):
            raise ConnectionError("down again")

        monkeypatch.setattr(broken_extractor, "_call_model", boom)
        for i in range(broken_extractor.ESCALATE_AFTER):
            broken_extractor.extract(f"User: turn {i}")
        assert "EFFECTIVELY RULE-BASED" in capsys.readouterr().err

    def test_below_threshold_no_escalation(self, broken_extractor, capsys):
        for i in range(broken_extractor.ESCALATE_AFTER - 1):
            broken_extractor.extract(f"User: turn {i}")
        err = capsys.readouterr().err
        assert "EFFECTIVELY RULE-BASED" not in err
        assert "falling back to rule-based" in err  # per-call lines still there
