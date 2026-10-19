"""
The contract for judge_use() (revien/usage/judge.py): both signals must agree.

Every case sits at an extreme (all of a memory's words shared vs none,
similarity 0.95 vs 0.05), so any sensible thresholds pass. Exactly where the
bars sit is the rule-writer's call, calibrated in shadow mode.
"""

from revien.usage import UseEvidence, judge_use

ALL_TERMS = ["concurrent", "postgresql", "sqlite", "worker", "writes"]


def _evidence(shared, similarity):
    return UseEvidence(shared_terms=shared, memory_terms=len(ALL_TERMS),
                       similarity=similarity)


def test_both_signals_strong_is_used():
    verdict = judge_use(_evidence(ALL_TERMS, 0.95))
    assert verdict.used and not verdict.abstain


def test_words_without_meaning_is_not_used():
    verdict = judge_use(_evidence(ALL_TERMS, 0.05))
    assert not verdict.used and not verdict.abstain


def test_meaning_without_words_is_not_used():
    verdict = judge_use(_evidence([], 0.95))
    assert not verdict.used and not verdict.abstain


def test_neither_signal_is_not_used():
    verdict = judge_use(_evidence([], 0.05))
    assert not verdict.used and not verdict.abstain


def test_missing_similarity_abstains():
    verdict = judge_use(_evidence(ALL_TERMS, None))
    assert verdict.abstain and not verdict.used


def test_every_verdict_explains_itself():
    for evidence in (_evidence(ALL_TERMS, 0.95), _evidence([], 0.05),
                     _evidence(ALL_TERMS, None)):
        assert judge_use(evidence).reason.strip()
