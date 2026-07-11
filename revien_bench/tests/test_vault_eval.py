"""
Smoke tests for the vault eval. Runs the REAL harness on the fixture corpus in
graph-only mode (fast, deterministic, no embedder needed) and asserts the
report's shape and its isolation guarantees — not specific scores, which are
the eval's own job to report honestly.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from revien_bench.vault_eval import (
    FIXTURE_CONVS,
    FIXTURE_QA,
    FIXTURE_VAULT,
    fixture_sha,
    run_vault_eval,
)


@pytest.fixture(scope="module")
def report():
    out = Path(tempfile.mkdtemp(prefix="revien_vault_eval_"))
    try:
        yield run_vault_eval(out, semantic_enabled=False)
    finally:
        shutil.rmtree(out, ignore_errors=True)


class TestFixtureIntegrity:
    def test_fixture_files_exist(self):
        assert FIXTURE_VAULT.is_dir()
        assert len(list(FIXTURE_VAULT.rglob("*.md"))) >= 15
        assert FIXTURE_CONVS.exists() and FIXTURE_QA.exists()

    def test_gold_ids_resolve_to_fixture_content(self):
        """Every gold id must point at a real note or a real turn — a typo in
        the QA file must fail here, not silently zero a metric."""
        qa = json.loads(FIXTURE_QA.read_text(encoding="utf-8"))["qa"]
        notes = {p.relative_to(FIXTURE_VAULT).as_posix() for p in FIXTURE_VAULT.rglob("*.md")}
        turns = {t["id"] for t in json.loads(FIXTURE_CONVS.read_text(encoding="utf-8"))["turns"]}
        for item in qa:
            for gid in item["gold"]:
                kind, _, ref = gid.partition(":")
                if kind == "note":
                    assert ref in notes, f"gold note missing: {ref}"
                elif kind == "conv":
                    assert ref in turns, f"gold turn missing: {ref}"
                else:
                    pytest.fail(f"unknown gold id space: {gid}")

    def test_fixture_sha_is_stable(self):
        assert fixture_sha() == fixture_sha()


class TestReportShape:
    def test_metrics_and_taxonomy_present(self, report):
        assert report["eval"] == "vault_fixture_v1"
        r = report["retrieval"]
        for key in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"):
            assert 0.0 <= r[key] <= 1.0
        assert set(report["per_category"]) == {"single_note", "cross_note", "attachment"}
        fa = report["retrieval_failure_analysis"]
        assert fa["rows_classified"] == report["n_questions"]

    def test_attachment_ground_truth_measured(self, report):
        a = report["attachment"]
        assert a["n_expected"] > 0 and a["n_fragile"] > 0
        assert 0.0 <= a["attach_rate_expected"] <= 1.0
        # Every fixture turn got a verdict.
        assert len(a["rows"]) == report["ingest"]["conversation_turns"]

    def test_isolation_from_conversational_bench(self, report):
        """AND-not-OR, mechanically: the vault report must name its corpus,
        carry the dataset SHA, and never emit the conversational headline key
        (overall_f1) that could get pasted into the wrong table."""
        assert "overall_f1" not in report
        assert report["dataset_sha256"] == fixture_sha()
        assert "never blend" in report["corpus_note"]

    def test_results_written_to_vault_namespace(self, report):
        assert "_vault_eval.json" in report["_out_path"]
