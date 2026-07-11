"""
Checkpoint / resume + answerer-timeout tests. OFFLINE: no real dataset, no
network, no API key. Proves the two interruption-robustness fixes:

JOB 1 — per-conversation checkpoint + resume (runner):
  * after a conversation completes, ONE JSON line is appended to the checkpoint
    and flushed, so an interruption loses at most the in-progress conversation.
  * a resumed run SKIPS conversations already in the checkpoint (keyed by
    conv_id) and does NOT re-answer them, yet the final aggregate covers BOTH
    the resumed and the newly-run conversations.
  * the dataset-SHA guard prevents a changed dataset from falsely resuming.
  * --fresh ignores + deletes the checkpoint and re-runs everything.

JOB 2 — HTTP timeout + per-QA failure handling:
  * a stalled HTTP layer (socket.timeout) makes the LLM answerer RAISE a clean,
    catchable RuntimeError instead of blocking forever.
  * the runner catches a per-QA answerer exception, records the QA as
    unanswered (empty prediction -> F1 0, answer_error set) and CONTINUES the
    run rather than dying.
"""

import json
import os
import shutil
import socket
import tempfile
import urllib.error
from pathlib import Path

import pytest

from revien_bench import answerers as A
from revien_bench import runner as R
from revien_bench.loader import QA, Conversation, Turn


_FAKE_SHA = "deadbeef" * 8  # 64-hex placeholder dataset SHA


# ── synthetic two-conversation dataset (no network, no real file) ─────────────
def _conv(cid: str, db: str, tok: str, qtok: str) -> Conversation:
    c = Conversation(conv_id=cid, speaker_a="Alice", speaker_b="Bob")
    c.session_dates = {1: "7 May 2023"}
    c.turns = [
        Turn(dia_id=f"{cid}:1", speaker="Alice", session=1, session_date="7 May 2023",
             text=f"We deployed the backend on {db}."),
        Turn(dia_id=f"{cid}:2", speaker="Bob", session=1, session_date="7 May 2023",
             text=f"The {tok} tokens live in Redis."),
    ]
    c.qa = [
        QA(question=f"What database for {qtok}?", answer=db, category=4,
           evidence=[f"{cid}:1"]),
    ]
    return c


def _two_convs():
    return [
        _conv("D1", "PostgreSQL", "JWT", "alpha"),
        _conv("D2", "MySQL", "OAuth", "beta"),
    ]


class _SpyAnswerer:
    """Extractive-backed answerer that records every question it is asked.

    Lets a test prove a resumed conversation's QAs are NOT re-answered.
    """

    name = "extractive"

    def __init__(self):
        self._inner = A.ExtractiveAnswerer()
        self.asked = []
        self.network_calls = 0
        self.cost_usd_estimate = 0.0

    def answer(self, ctx: A.RetrievedContext) -> str:
        self.asked.append(ctx.query)
        return self._inner.answer(ctx)


@pytest.fixture
def work_dir():
    """A throwaway working directory (stdlib tempfile; pytest tmp_path's base is
    permission-locked in this environment)."""
    d = Path(tempfile.mkdtemp(suffix="_ckpt_test"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def patched_loader(monkeypatch, work_dir):
    """Point the runner at the synthetic dataset + a fixed dataset SHA."""
    monkeypatch.setattr(R, "load_locomo", lambda _p: _two_convs())
    monkeypatch.setattr(R, "read_locked_hash", lambda: _FAKE_SHA)
    return work_dir


# ── JOB 1: checkpoint is written per conversation ─────────────────────────────
def test_checkpoint_written_per_conversation(patched_loader, monkeypatch):
    out_dir = patched_loader / "results"
    spy = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy)

    report = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir, fresh=True,
    )

    # The checkpoint path carries the run's env+code fingerprint, and the
    # config's env is applied only INSIDE run_benchmark — so the path can't
    # be recomputed out here. The report carries it (by design: it's the
    # run's identity).
    ckpt = Path(report["resume"]["checkpoint_path"])
    assert ckpt.exists()
    lines = [l for l in ckpt.read_text(encoding="utf-8").splitlines() if l.strip()]
    # One line per completed conversation.
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert {r["conv_id"] for r in recs} == {"D1", "D2"}
    # Each record carries the dataset SHA + the per-question rows.
    for r in recs:
        assert r["dataset_sha"] == _FAKE_SHA
        assert len(r["rows"]) == 1
    # Both conversations ran fresh; aggregate covers both.
    assert report["n_questions"] == 2
    assert report["resume"]["conversations_ran"] == 2
    assert report["resume"]["conversations_resumed"] == 0


# ── JOB 1: interrupt after conv 1, resume, conv 1 SKIPPED, aggregate covers both
def test_resume_skips_completed_and_aggregates_all(patched_loader, monkeypatch):
    out_dir = patched_loader / "results"

    # ── Run 1: only conversation D1 (simulate an interruption before D2). ──
    spy1 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy1)
    rep1 = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir,
        limit_convs=1, fresh=True,
    )
    assert rep1["n_questions"] == 1
    assert {q for q in spy1.asked} == {"What database for alpha?"}  # only D1 asked

    ckpt = Path(rep1["resume"]["checkpoint_path"])
    assert ckpt.exists()
    assert len([l for l in ckpt.read_text().splitlines() if l.strip()]) == 1  # D1 only

    # ── Run 2: full dataset, RESUME (not fresh). D1 must be skipped. ──
    spy2 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy2)
    rep2 = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir,
        fresh=False,  # resume
    )

    # D1 was resumed from the checkpoint -> its QA was NOT re-answered.
    assert "What database for alpha?" not in spy2.asked, "D1 should NOT be re-answered"
    assert spy2.asked == ["What database for beta?"], "only D2 should be answered fresh"

    # The final aggregate covers BOTH conversations (resumed D1 + new D2).
    assert rep2["n_questions"] == 2
    assert rep2["resume"]["conversations_resumed"] == 1
    assert rep2["resume"]["conversations_ran"] == 1
    convs_in_report = {row["conv"] for row in rep2["per_question"]}
    assert convs_in_report == {"D1", "D2"}

    # Both conversations now in the checkpoint (D1 once, D2 appended).
    final_lines = [l for l in ckpt.read_text().splitlines() if l.strip()]
    assert len(final_lines) == 2


# ── JOB 1: dataset-SHA guard prevents a false resume ──────────────────────────
def test_dataset_sha_guard_blocks_false_resume(patched_loader, monkeypatch):
    out_dir = patched_loader / "results"
    spy1 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy1)
    R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir,
        limit_convs=1, fresh=True,
    )

    # The dataset "changes": its locked SHA is now different. The old checkpoint
    # line (stamped with the OLD sha) must be IGNORED -> D1 is re-run, not resumed.
    monkeypatch.setattr(R, "read_locked_hash", lambda: "feedface" * 8)
    spy2 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy2)
    rep = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir,
        limit_convs=1, fresh=False,  # resume attempt, but SHA mismatch
    )
    assert rep["resume"]["conversations_resumed"] == 0  # guard blocked the resume
    assert "What database for alpha?" in spy2.asked     # D1 was re-answered


# ── JOB 1: --fresh ignores + deletes the checkpoint ───────────────────────────
def test_fresh_flag_reruns_everything(patched_loader, monkeypatch):
    out_dir = patched_loader / "results"
    spy1 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy1)
    R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir, fresh=True,
    )
    spy2 = _SpyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: spy2)
    rep = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir, fresh=True,
    )
    # --fresh: nothing resumed, both conversations re-answered.
    assert rep["resume"]["conversations_resumed"] == 0
    assert rep["resume"]["conversations_ran"] == 2
    assert "What database for alpha?" in spy2.asked
    assert "What database for beta?" in spy2.asked


# ── JOB 2: a stalled socket makes the answerer RAISE (not hang) ───────────────
def test_answerer_raises_on_socket_timeout(monkeypatch):
    def _hang(req, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(A.urllib.request, "urlopen", _hang)
    ans = A.OllamaAnswerer("llama3")
    with pytest.raises(RuntimeError, match="timeout"):
        ans.answer(A.RetrievedContext(query="q", contents=["c"], labels=["l"]))


def test_answerer_raises_on_urlerror_wrapped_timeout(monkeypatch):
    def _hang(req, timeout=None):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr(A.urllib.request, "urlopen", _hang)
    ans = A.OllamaAnswerer("llama3")
    with pytest.raises(RuntimeError, match="timeout"):
        ans.answer(A.RetrievedContext(query="q", contents=["c"], labels=["l"]))


def test_timeout_env_override_is_read(monkeypatch):
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"message": {"content": "ok"}}'

    def _fake_urlopen(req, timeout=None):
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setenv("REVIEN_ANSWERER_TIMEOUT", "7")
    monkeypatch.setattr(A.urllib.request, "urlopen", _fake_urlopen)
    A.OllamaAnswerer("llama3").answer(
        A.RetrievedContext(query="q", contents=["c"], labels=["l"])
    )
    assert captured["timeout"] == 7.0  # env override flowed into urlopen


# ── JOB 2: the runner records a failed QA as unanswered and CONTINUES ─────────
def test_runner_records_failed_qa_and_continues(patched_loader, monkeypatch):
    out_dir = patched_loader / "results"

    class _FlakyAnswerer:
        """Times out on the FIRST question, answers the rest — proves one bad
        call is recorded as unanswered without killing the run."""

        name = "extractive"

        def __init__(self):
            self.calls = 0
            self.network_calls = 0
            self.cost_usd_estimate = 0.0

        def answer(self, ctx):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("timeout after 60s contacting http://x")
            return "MySQL"

    flaky = _FlakyAnswerer()
    monkeypatch.setattr(R.A, "build_answerer", lambda _name: flaky)

    report = R.run_benchmark(
        config_name="graph_only", answerer_name="extractive",
        dataset_path=patched_loader / "ds.json", out_dir=out_dir, fresh=True,
    )

    # The run did NOT die: both conversations' QAs are present.
    assert report["n_questions"] == 2
    rows = {row["conv"]: row for row in report["per_question"]}

    # D1's only QA failed -> empty prediction, F1 0, answer_error captured.
    d1 = rows["D1"]
    assert d1["prediction"] == ""
    assert d1["f1"] == 0.0
    assert d1["answer_error"] and "timeout" in d1["answer_error"]

    # D2 still answered normally after the failure -> run continued.
    d2 = rows["D2"]
    assert d2["answer_error"] is None
    assert d2["prediction"] != ""
