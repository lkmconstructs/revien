"""Claim Sovereignty Layer — sensitive-recognition gap REGRESSION PIN.

Three populations, pinned so the safety story is measurable and cannot drift:

  1. Manifestation 1 (unclassified) — "I am sober" classifies as nothing.
     Caught by the interim SENSITIVE FLOOR. (safety assertion)
  2. Manifestation 2, LEXED — sensitive content confidently misnamed as a
     non-protected type BUT naming a core sensitive lexeme ("I love being sober"
     -> preference_habit, lexeme "sober"). Caught by the interim DISTRUST
     TRIPWIRE -> candidate. (was the live gap; tripwire closed THIS slice)
  3. Manifestation 2, UNLEXED — sensitive MEANING with no core lexeme ("I love
     being off the bottle"). The tripwire is lexemes-not-meaning, so it MISSES
     these and they still AUTO_SUPERSEDE. This is the residual gap and the proof
     that the tripwire does NOT close Trigger 2. Pinned as the SEMANTIC backend's
     success criterion: drive these to 0 auto-supersede. When the backend lands
     and does, THIS assertion flips — that failure is the gap-closed signal.

See HYBRID_BACKEND_TRIGGERS.md. The tripwire is the interim promise; the semantic
backend is the fix.
"""

import pytest

from revien.claims import ClassificationStatus
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, SupersessionAction, SupersessionGate

SENSITIVE_FLOORED_CASES = [
    ("I am sober.", "I'm drinking again."),
    ("I am transitioning.", "I stopped transitioning."),
]

# Lexed manifestation-2: the tripwire catches these (core lexeme present).
TRIPWIRE_CAUGHT_CASES = [
    ("I love being sober.", "I don't enjoy being sober."),
    ("I enjoy being single.", "I don't enjoy being single anymore."),
    ("I love being in recovery.", "I don't enjoy being in recovery."),
]

# Unlexed manifestation-2: sensitive meaning, NO core lexeme -> tripwire misses ->
# still auto-superseded. The semantic backend (Trigger 2) must close these.
TRIPWIRE_MISS_CASES = [
    ("I love being off the bottle.", "I don't enjoy being off the bottle."),
    ("I love being a former smoker.", "I don't enjoy being a former smoker."),
    ("I love being childfree.", "I don't enjoy being childfree."),
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


@pytest.mark.parametrize("existing,new", TRIPWIRE_CAUGHT_CASES)
def test_manifestation2_lexed_is_caught_by_tripwire(clf, gate, existing, new):
    """The interim tripwire closes the LEXED slice: misnamed-but-lexed sensitive
    content routes to candidate with a labeled reason, never auto-superseded."""
    res = clf.classify(existing)
    assert res.classification_status is ClassificationStatus.CLASSIFIED
    assert res.is_protected() is False  # threads both type-keyed guards
    d = gate.evaluate(Claim(existing, res), Claim(new, clf.classify(new)))
    assert d.action is SupersessionAction.CANDIDATE
    assert d.reason.startswith("tripwire_distrust:")


@pytest.mark.parametrize("existing,new", TRIPWIRE_MISS_CASES)
def test_manifestation2_unlexed_still_leaks_trigger2_open(clf, gate, existing, new):
    """RESIDUAL GAP (pinned): sensitive meaning with no core lexeme is NOT caught —
    proof the tripwire is lexemes-not-meaning and does not close Trigger 2. This
    still AUTO_SUPERSEDES today; the SEMANTIC backend must flip it. DO NOT close by
    adding lexemes to chase these — that is the rule-stuffing the backend replaces."""
    res = clf.classify(existing)
    d = gate.evaluate(Claim(existing, res), Claim(new, clf.classify(new)))
    assert d.action is SupersessionAction.AUTO_SUPERSEDE


def test_backend_success_criteria_are_measurable(clf, gate):
    """Two measurable targets for the semantic backend, recorded here."""
    def autos(cases):
        return sum(
            1 for e, n in cases
            if gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n))).action
            is SupersessionAction.AUTO_SUPERSEDE)
    # Tripwire has already driven the LEXED slice to zero auto-supersede.
    assert autos(TRIPWIRE_CAUGHT_CASES) == 0
    # The UNLEXED slice still fully leaks — the backend's job is to drive this to 0.
    assert autos(TRIPWIRE_MISS_CASES) == len(TRIPWIRE_MISS_CASES)
