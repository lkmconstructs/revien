"""
F8: revien_bench.report's stale "LLM-judge accuracy deferred to track B"
footer must not appear once a judge block is actually present in the
report — otherwise the report contradicts itself (an "End-to-end QA (LLM
judge)" section above, and a footer below claiming that's deferred).
"""

from revien_bench import report as R


def _base_report(**overrides):
    rep = {
        "config": {"name": "graph_only", "answerer": "extractive", "cluster": False},
        "dataset": {"path": "x", "conversations": 1, "sha256": "deadbeef"},
        "environment": {"revien_version": "0.0.0", "python": "3.x", "platform": "test"},
        "timestamp": "2026-01-01T00:00:00+00:00",
        "n_questions": 1,
        "cost_usd": 0.0,
        "network_calls": 0,
        "overall_f1": 1.0,
        "per_category_f1": {},
        "retrieval": {"recall@1": None, "recall@3": None, "recall@5": None,
                      "recall@10": None, "mrr": None, "ndcg@10": None,
                      "n_with_evidence": 0},
        "latency_ms": {"recall": {"p50": 0, "p90": 0, "p99": 0, "mean": 0}},
        "ingest_turns_per_sec": 0.0,
        "sovereignty": {"all_passed": True, "checks": []},
    }
    rep.update(overrides)
    return rep


def test_footer_defers_to_track_b_without_judge():
    md = R.render(_base_report())
    assert "deferred to track B" in md


def test_footer_does_not_claim_deferred_when_judge_block_present():
    rep = _base_report(
        reader={"spec": "extractive", "model": "extractive", "prompt_sha256": None},
        judge={
            "spec": "ollama:llama3", "model": "llama3", "prompt_sha256": "abc",
            "accuracy_overall": 1.0, "accuracy_denominator": 1,
            "per_category_accuracy": {}, "judge_errors": 0, "n_judged": 1,
            "network_calls": 0, "cost_usd": 0.0, "cost_usd_is_estimate": True,
        },
    )
    md = R.render(rep)
    assert "## End-to-end QA (LLM judge)" in md
    assert "deferred to track B" not in md


# ── honest judge denominator: correct/denominator, judged, errors ─────────

def test_judge_accuracy_line_shows_correct_over_denominator_judged_errors():
    rep = _base_report(
        reader={"spec": "extractive", "model": "extractive", "prompt_sha256": None},
        judge={
            "spec": "ollama:llama3", "model": "llama3", "prompt_sha256": "abc",
            "accuracy_overall": 0.75, "accuracy_denominator": 4, "n_correct": 3,
            "per_category_accuracy": {}, "judge_errors": 2, "n_judged": 6,
            "network_calls": 0, "cost_usd": 0.0, "cost_usd_is_estimate": True,
        },
    )
    md = R.render(rep)
    assert "Overall accuracy: 0.75" in md
    assert "correct/denominator = 3/4" in md
    assert "judged 6" in md
    assert "errors 2" in md


def test_judge_accuracy_line_backfills_n_correct_when_absent():
    """A report JSON written before n_correct existed must still render a
    correct count, backed out from accuracy * denominator, rather than
    crashing or printing 'None'."""
    rep = _base_report(
        reader={"spec": "extractive", "model": "extractive", "prompt_sha256": None},
        judge={
            "spec": "ollama:llama3", "model": "llama3", "prompt_sha256": "abc",
            "accuracy_overall": 0.5, "accuracy_denominator": 2,
            "per_category_accuracy": {}, "judge_errors": 0, "n_judged": 2,
            "network_calls": 0, "cost_usd": 0.0, "cost_usd_is_estimate": True,
        },
    )
    md = R.render(rep)
    assert "correct/denominator = 1/2" in md


def test_judge_accuracy_none_renders_as_no_judged_answers():
    """When every judged question errored, accuracy_overall is None (F5) —
    the report must say so plainly, never print 'None' or a fabricated
    0.0."""
    rep = _base_report(
        reader={"spec": "extractive", "model": "extractive", "prompt_sha256": None},
        judge={
            "spec": "ollama:llama3", "model": "llama3", "prompt_sha256": "abc",
            "accuracy_overall": None, "accuracy_denominator": 0, "n_correct": 0,
            "per_category_accuracy": {}, "judge_errors": 3, "n_judged": 3,
            "network_calls": 0, "cost_usd": 0.0, "cost_usd_is_estimate": True,
        },
    )
    md = R.render(rep)
    assert "Overall accuracy: n/a (no judged answers)" in md
    assert "Overall accuracy: None" not in md
