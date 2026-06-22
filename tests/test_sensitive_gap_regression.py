"""Claim Sovereignty Layer — sensitive-recognition gap REGRESSION PIN.

The adversarial floor verification (wf_fe8f5b82) found that the interim sensitive
floor covers only manifestation 1 of the recognition gap (sensitive content the
rules classify as NOTHING). Manifestation 2 — sensitive content the rules
confidently MISNAME into a non-protected type ("I love being sober" ->
preference_habit) — threads between the floor (keyed on status) and the protected
guard (keyed on type) and is AUTO_SUPERSEDED under the default config today.

This file PINS both behaviors so the gap is measurable and cannot be silently
forgotten:
  * manifestation 1 is floored (a passing safety assertion);
  * manifestation 2 currently auto-supersedes — asserted as the KNOWN GAP. When
    the hybrid backend's sensitive-recognition (Trigger 2, HYBRID_BACKEND_TRIGGERS.md)
    lands and routes these to candidate/protected, THIS test will fail — that
    failure is the signal the gap closed; update the assertion then.

These are the backend's success criterion: every SENSITIVE_GAP_CASES pair must
stop auto-superseding.
"""

import pytest

from revien.claims import ClassificationStatus
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, SupersessionAction, SupersessionGate

# Manifestation-2 cases: sensitive disclosure phrased through a preference frame,
# with a same-type contradictor that also classifies. The backend must route ALL
# of these away from auto_supersede.
SENSITIVE_GAP_CASES = [
    ("I love being sober.", "I don't enjoy being sober."),
    ("I enjoy being single.", "I don't enjoy being single anymore."),
    ("I love being in recovery.", "I don't enjoy being in recovery."),
]

# Manifestation-1 cases: bare sensitive statements the rules classify as nothing.
SENSITIVE_FLOORED_CASES = [
    ("I am sober.", "I'm drinking again."),
    ("I am transitioning.", "I stopped transitioning."),
]


@pytest.fixture
def clf():
    return ClaimClassifier()


@pytest.fixture
def gate():
    return SupersessionGate()


@pytest.mark.parametrize("existing,new", SENSITIVE_FLOORED_CASES)
def test_manifestation1_unclassified_is_floored(clf, gate, existing, new):
    """SAFETY (must hold): bare sensitive statements are unclassified -> floored."""
    res = clf.classify(existing)
    assert res.classification_status is ClassificationStatus.UNCLASSIFIED
    d = gate.evaluate(Claim(existing, res), Claim(new, clf.classify(new)))
    assert d.action is not SupersessionAction.AUTO_SUPERSEDE
    assert "sensitive_floor:existing_not_classified" in d.trace


@pytest.mark.parametrize("existing,new", SENSITIVE_GAP_CASES)
def test_manifestation2_misnamed_is_the_known_gap(clf, gate, existing, new):
    """KNOWN GAP (pinned): preference-framed sensitive content is confidently typed
    non-protected and currently AUTO_SUPERSEDES under the default config. When the
    hybrid backend closes Trigger 2 this flips to candidate/protected and this
    assertion must be updated — its failure is the gap-closed signal."""
    res = clf.classify(existing)
    assert res.classification_status is ClassificationStatus.CLASSIFIED
    assert res.is_protected() is False  # the reason it threads both guards
    d = gate.evaluate(Claim(existing, res), Claim(new, clf.classify(new)))
    # Pinned current (unsafe) behavior — DO NOT "fix" by editing the classifier
    # rules; this closes with the backend. See HYBRID_BACKEND_TRIGGERS.md Trigger 2.
    assert d.action is SupersessionAction.AUTO_SUPERSEDE


def test_gap_cases_are_the_backend_success_criterion(clf, gate):
    """Documents the measurable target: count of gap cases still auto-superseding.
    Backend is done (for this gap) when this count reaches 0."""
    still_leaking = 0
    for existing, new in SENSITIVE_GAP_CASES:
        d = gate.evaluate(Claim(existing, clf.classify(existing)),
                          Claim(new, clf.classify(new)))
        if d.action is SupersessionAction.AUTO_SUPERSEDE:
            still_leaking += 1
    # Today every gap case leaks; the backend must drive this to 0.
    assert still_leaking == len(SENSITIVE_GAP_CASES)
