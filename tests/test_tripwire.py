"""Claim Sovereignty Layer — distrust tripwire tests (the six invariants).

Tests the catches AND the deliberate false positives. The false positives are
asserted as PROOF OF HUMILITY (invariant 6), not flaws: the tripwire is
deliberately over-broad because betrayal with a stack trace costs more than an
extra human review. Narrowing it to kill these would be the defect.
"""

import pytest

from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import (
    Claim,
    SupersessionAction,
    SupersessionGate,
    SupersessionMetrics,
)
from revien.tripwire import DistrustTripwire


@pytest.fixture
def tw():
    return DistrustTripwire()


@pytest.fixture
def clf():
    return ClaimClassifier()


# ── catches (every core domain) ───────────────────────────────────────────────

@pytest.mark.parametrize("text,domain", [
    ("I love being sober.", "recovery"),
    ("I'm in recovery and proud.", "recovery"),
    ("I started my transition.", "identity_transition"),
    ("I use they/them pronouns.", "identity_transition"),
    ("I'm single and happy.", "relationship_status"),
    ("My diagnosis changed everything.", "health_diagnosis"),
    ("I'm nursing my newborn.", "health_diagnosis"),
    ("My faith is important to me.", "faith_religion"),
    ("I'm politically progressive.", "political_belief"),
])
def test_core_domains_catch(tw, text, domain):
    assert tw.check(text) == domain


def test_inspects_normalized_form(tw):
    """Invariant 3: raw AND normalized — caps and trailing punctuation still catch."""
    assert tw.check("I'm SOBER!!!") == "recovery"
    assert tw.check("Married, finally.") == "relationship_status"


# ── invariant 4: config-floor (add-only, never removable) ─────────────────────

def test_operator_can_add_a_domain(tw):
    custom = DistrustTripwire(extra_domains={"custom_pii": [r"badge.?number"]})
    assert custom.check("my badge number is 7") == "custom_pii"
    assert custom.covers_core()  # core untouched by the addition


def test_operator_can_add_lexemes_to_a_core_domain():
    custom = DistrustTripwire(extra_domains={"recovery": [r"off the bottle"]})
    assert custom.check("I'm off the bottle now") == "recovery"  # new lexeme works
    assert custom.check("I am sober") == "recovery"              # core still works


def test_core_cannot_be_removed_via_config():
    """Passing an empty or weak override for a core domain CANNOT shrink it."""
    for attack in ({}, {"recovery": []}, {"recovery": ["xyz"]},
                   {"relationship_status": []}):
        t = DistrustTripwire(extra_domains=attack)
        assert t.covers_core()
        assert t.check("I am sober") == "recovery"
        assert t.check("I'm single") == "relationship_status"


# ── invariants 1 & 2: strictly additive, candidate-only ───────────────────────

def test_tripwire_downgrades_auto_to_candidate(clf):
    gate = SupersessionGate()
    d = gate.evaluate(Claim("I love being sober.", clf.classify("I love being sober.")),
                      Claim("I don't enjoy being sober.", clf.classify("I don't enjoy being sober.")))
    assert d.action is SupersessionAction.CANDIDATE
    assert d.reason == "tripwire_distrust:recovery"


def test_tripwire_fires_on_new_claim_too(clf):
    """A sensitive NEW claim also routes to candidate (the disclosure is the new state)."""
    gate = SupersessionGate()
    d = gate.evaluate(Claim("I love wine.", clf.classify("I love wine.")),
                      Claim("I don't like wine, I'm sober now.", clf.classify("I don't like wine, I'm sober now.")))
    # whatever the gate would have done, a sensitive claim present -> not auto
    assert d.action is not SupersessionAction.AUTO_SUPERSEDE


def test_tripwire_never_creates_auto_or_removes_protection(clf):
    """Invariant 1: the tripwire only makes decisions MORE conservative. Across a
    mixed batch, no decision is AUTO when the tripwire fired, and a tripwire firing
    never overrides a protected/floor/version outcome into auto."""
    gate = SupersessionGate()
    tw = DistrustTripwire()
    batch = [
        ("I love being sober.", "I don't enjoy being sober."),     # sensitive -> candidate
        ("I'm single.", "I'm married now."),                       # protected -> candidate
        ("I am sober.", "I'm drinking again."),                    # unclassified -> floor
        ("I love painting.", "I don't like painting anymore."),    # neutral -> auto
    ]
    for e, n in batch:
        d = gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n)))
        if tw.check(e) or tw.check(n):
            assert d.action is not SupersessionAction.AUTO_SUPERSEDE


def test_neutral_claims_still_auto_supersede(clf):
    """The tripwire is targeted, not a blanket block — neutral content still autos."""
    gate = SupersessionGate()
    for e, n in [("I love painting.", "I don't like painting anymore."),
                 ("My favorite food is sushi.", "My favorite food is ramen now.")]:
        d = gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n)))
        assert d.action is SupersessionAction.AUTO_SUPERSEDE


# ── invariant 6: deliberate false positives are PROOF OF HUMILITY, not flaws ───

@pytest.mark.parametrize("text,domain", [
    ("I recovered from the flu last week.", "recovery"),      # not addiction recovery
    ("I'm single-minded about my goals.", "relationship_status"),  # not relationship status
    ("We had a god-awful time.", "faith_religion"),           # not faith
    ("I voted for the best pizza place.", "political_belief"),  # not politics
])
def test_false_positives_are_intended_humility(tw, text, domain):
    """These trip the wire WITHOUT being real sensitive disclosures. That over-catch
    is INTENDED (invariant 6): an extra human review is cheap; auto-erasing a real
    disclosure is betrayal with a stack trace. Do NOT 'fix' these by narrowing the
    lexemes — that reintroduces the harm the tripwire exists to prevent."""
    assert tw.check(text) == domain


# ── full metrics set ──────────────────────────────────────────────────────────

def test_metrics_count_tripwire_catches_by_domain(clf):
    gate = SupersessionGate()
    m = SupersessionMetrics()
    pairs = [
        ("I love being sober.", "I don't enjoy being sober."),    # recovery
        ("I love being in recovery.", "I don't enjoy being in recovery."),  # recovery
        ("I enjoy being single.", "I don't enjoy being single anymore."),   # relationship_status
        ("I love painting.", "I don't like painting anymore."),   # neutral -> auto
    ]
    for e, n in pairs:
        m.record(gate.evaluate(Claim(e, clf.classify(e)), Claim(n, clf.classify(n))))
    assert m.tripwire_caught == 3
    assert m.tripwire_by_domain.get("recovery") == 2
    assert m.tripwire_by_domain.get("relationship_status") == 1
    assert m.snapshot()["tripwire_caught"] == 3
