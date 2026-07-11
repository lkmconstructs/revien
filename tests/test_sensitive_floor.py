"""Claim Sovereignty Layer — interim sensitive floor + queue instrumentation.

The floor is a SAFETY mechanism: a not-confidently-classified claim might be a
sensitive claim the rule classifier could not name, so it is never
auto-superseded, and NO config can lower that. These tests try to break it
through every config lever and prove it holds — and exercise the workload-trigger
metrics.
"""

import pytest

from revien.claims import SENSITIVE_FLOOR, PROTECTED_DEFAULT, ClassificationStatus
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import (
    Claim,
    Lock,
    SupersessionAction,
    SupersessionGate,
    SupersessionMetrics,
)


@pytest.fixture
def clf():
    return ClaimClassifier()


def _adversarial_gates():
    """Every config a future protected-set feature might produce, incl. fully
    de-protected and a zero confidence bar — the config back door, opened wide."""
    return [
        SupersessionGate(),
        SupersessionGate(protected_set=frozenset()),
        SupersessionGate(protected_set=frozenset(), new_conf_bar=0.0),
        SupersessionGate(protected_set=PROTECTED_DEFAULT, new_conf_bar=0.0),
    ]


def test_unclassified_sensitive_claim_classifies_as_nothing(clf):
    """Premise check: the named example really is unclassified by the rules."""
    assert clf.classify("I'm sober.").classification_status is ClassificationStatus.UNCLASSIFIED


def test_floor_holds_against_every_config(clf):
    """An unclassified existing claim is never auto-superseded — under ANY gate
    config, including a fully de-protected set and a zero confidence bar."""
    existing = Claim("I'm sober.", clf.classify("I'm sober."))
    attackers = ["I'm drinking again.", "I had a drink last night.",
                 "I love drinking now.", "I don't stay sober anymore."]
    for gate in _adversarial_gates():
        for atk in attackers:
            d = gate.evaluate(existing, Claim(atk, clf.classify(atk)))
            assert d.action is not SupersessionAction.AUTO_SUPERSEDE
            assert "sensitive_floor:existing_not_classified" in d.trace


def test_floor_checked_first_before_scope(clf):
    """The floor runs ahead of all configurable logic — trace shows it fired and
    no scope/contradiction reasoning happened."""
    d = SupersessionGate(protected_set=frozenset()).evaluate(
        Claim("I'm sober.", clf.classify("I'm sober.")),
        Claim("I'm drinking again.", clf.classify("I'm drinking again.")))
    assert d.trace[0] == "sensitive_floor:existing_not_classified"
    assert d.scope_overlap == 0.0


def test_low_confidence_existing_also_floored(clf):
    """low_confidence (not just unclassified) is below the bar -> floored too."""
    # find a low_confidence existing claim if the classifier produces one
    res = clf.classify("My grandma is from Sweden.")
    if res.classification_status is ClassificationStatus.LOW_CONFIDENCE:
        d = SupersessionGate().evaluate(
            Claim("My grandma is from Sweden.", res),
            Claim("My grandma is from Norway.", clf.classify("My grandma is from Norway.")))
        assert d.action is not SupersessionAction.AUTO_SUPERSEDE
        assert "sensitive_floor:existing_not_classified" in d.trace


def test_floor_does_not_overblock_classified_claims(clf):
    """The floor protects only the unclassified default; a classified non-protected
    claim still supersedes normally (the floor isn't a blanket no-op)."""
    d = SupersessionGate().evaluate(
        Claim("I love painting.", clf.classify("I love painting.")),
        Claim("Honestly, I don't like painting anymore.", clf.classify("Honestly, I don't like painting anymore.")))
    assert d.action is SupersessionAction.AUTO_SUPERSEDE


def test_floor_level_is_the_named_sensitive_set():
    assert SupersessionGate.FLOOR_LEVEL is SENSITIVE_FLOOR
    assert SENSITIVE_FLOOR == PROTECTED_DEFAULT


# ── workload-trigger metrics ──────────────────────────────────────────────────

def test_queue_metrics_track_depth_and_rate(clf):
    gate = SupersessionGate()
    m = SupersessionMetrics()
    pairs = [
        ("I love painting.", "I don't like painting anymore."),   # auto
        ("My favorite food is sushi.", "My favorite food is ramen now."),  # auto
        ("I'm single.", "I'm married now."),                      # candidate (protected)
        ("I'm married.", "I'm so annoyed at Sam today."),       # no_conflict (scope)
    ]
    for e, n in pairs:
        m.record(gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n))))
    assert m.total == 4
    assert m.auto == 2
    assert m.candidate_queue_depth == m.candidate + m.version_locked
    snap = m.snapshot()
    assert snap["total"] == 4 and "candidate_queue_depth" in snap and "auto_fire_rate" in snap


def test_floor_catches_are_counted_separately_not_pooled_into_no_conflict(clf):
    """Safety observability: a sensitive-heavy stream of floored claims must be
    visible (sensitive_floor_caught), not hidden inside no_conflict / a quiet queue."""
    gate = SupersessionGate()
    m = SupersessionMetrics()
    for _ in range(5):
        d = gate.evaluate(Claim("I'm sober.", clf.classify("I'm sober.")),
                          Claim("I'm drinking again.", clf.classify("I'm drinking again.")))
        m.record(d)
    assert m.sensitive_floor_caught == 5
    assert m.candidate_queue_depth == 0           # the queue looks quiet...
    assert m.snapshot()["sensitive_floor_caught"] == 5   # ...but the safety overlay shows the volume
