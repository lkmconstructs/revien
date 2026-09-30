"""LLM readers see each retrieved memory's date; the extractive path does not.

OFFLINE: no network, no real dataset. The prompt TEMPLATE is sha256-gated and
untouched; only the text substituted into {context} carries the date.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from revien_bench import answerers as A
from revien_bench import runner as R
from revien_bench.tests.test_checkpoint_resume import _two_convs, _FAKE_SHA


def _ctx(dates):
    return A.RetrievedContext(
        query="When did Alice go?",
        contents=["Alice went to the lake yesterday.", "Bob likes tea."],
        labels=["trip", ""],
        dates=dates,
    )


def test_dated_memory_rendering():
    prompt = A.assemble_prompt(_ctx(["2023-05-07T00:00:00+00:00", None]))
    assert "[1] (trip) [7 May 2023] Alice went to the lake yesterday." in prompt
    # No date -> unchanged rendering.
    assert "[2] Bob likes tea." in prompt


def test_no_dates_is_unchanged():
    assert A._format_context(_ctx([])) == A._format_context(_ctx([None, None]))
    assert "[2] Bob likes tea." in A._format_context(_ctx([]))
    assert "[1] (trip) Alice went" in A._format_context(_ctx([]))


def test_unparseable_date_is_ignored():
    out = A._format_context(_ctx(["not-a-date", None]))
    assert "[1] (trip) Alice went" in out


def test_day_has_no_leading_zero():
    out = A._format_context(_ctx(["2023-06-02T00:00:00+00:00", None]))
    assert "[2 June 2023] " in out


def test_template_file_unchanged():
    # load_answer_prompt raises if the on-disk template drifted from its digest.
    assert "{context}" in A.load_answer_prompt()


def test_extractive_ignores_dates():
    dated = _ctx(["2023-05-07T00:00:00+00:00", "2023-05-08T00:00:00+00:00"])
    plain = _ctx([])
    assert dated.sentences() == plain.sentences()
    ext = A.ExtractiveAnswerer()
    assert ext.answer(dated) == ext.answer(plain)
    assert "2023" not in ext.answer(dated)


def test_fingerprint_changes_with_reader_context(monkeypatch):
    dated_fp = R._run_fingerprint()
    monkeypatch.setattr(R, "READER_CONTEXT", "undated")
    assert R._run_fingerprint() != dated_fp
    other = R._checkpoint_path(Path("x"), "c", "a")
    monkeypatch.setattr(R, "READER_CONTEXT", "dated-resolved")
    assert R._checkpoint_path(Path("x"), "c", "a") != other


@pytest.fixture
def work_dir():
    d = Path(tempfile.mkdtemp(suffix="_dates_test"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_results_json_records_reader_context_and_runner_fills_dates(
    work_dir, monkeypatch
):
    monkeypatch.setattr(R, "load_locomo", lambda _p: _two_convs())
    monkeypatch.setattr(R, "read_locked_hash", lambda: _FAKE_SHA)
    seen = []

    class Spy:
        name = "extractive"
        network_calls = 0
        cost_usd_estimate = 0.0

        def __init__(self):
            self._inner = A.ExtractiveAnswerer()

        def answer(self, ctx):
            seen.append(ctx)
            return self._inner.answer(ctx)

    monkeypatch.setattr(R.A, "build_answerer", lambda _n: Spy())
    report = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=work_dir / "ds.json", out_dir=work_dir / "results",
        fresh=True,
    )
    assert report["reader_context"] == "dated-resolved"
    written = sorted((work_dir / "results").glob("*.json"))
    assert json.loads(written[-1].read_text(encoding="utf-8"))["reader_context"] == "dated-resolved"
    assert seen and all(len(c.dates) == len(c.contents) for c in seen)


def test_score_qa_fills_dates_from_recall(monkeypatch):
    from types import SimpleNamespace

    def res(nid, content, when):
        return SimpleNamespace(
            node_id=nid, content=content, label="L", score=0.5,
            score_breakdown={}, path=[], recorded_at=when,
        )

    resp = SimpleNamespace(
        results=[res("n1", "Alice went to the lake yesterday.",
                     "2023-05-07T00:00:00+00:00"),
                 res("n2", "Bob likes tea.", None)],
        diagnostics=None, nodes_examined=2, retrieval_time_ms=1.0,
    )
    engine = SimpleNamespace(recall=lambda *a, **k: resp)
    conv = _two_convs()[0]
    got = {}

    class Spy:
        name = "spy"

        def answer(self, ctx):
            got["dates"] = ctx.dates
            got["contents"] = ctx.contents
            return "x"

    try:
        R._score_qa(None, engine, conv.qa[0], conv, Spy())
    except Exception:
        pass  # scoring internals need a real store; ctx was captured first
    assert got["dates"] == ["2023-05-07T00:00:00+00:00", None]
