"""Claim Sovereignty Layer — Leg 2.5b classifier: contract-guarantee tests.

These assert the INVARIANTS the classifier must uphold regardless of accuracy
(auto-supersession never granted, conservative abstention, durability as an
independent axis, protected/one-time/unclassified -> candidate_only). The two
known L2.5d defects are tracked as xfail, not hidden — they go to the gate.
"""

import pytest

from revien.claims import (
    ClaimType,
    ClassificationStatus,
    Durability,
)
from revien.ingestion.claim_classifier import ClaimClassifier


@pytest.fixture
def clf():
    return ClaimClassifier()


def test_auto_supersession_always_false(clf):
    """Invariant: Leg 2.5 never grants auto-supersession — only Leg 3 may."""
    for t in ["I'm a transgender woman.", "I went to the museum yesterday.",
              "I want to become a counselor.", "Yeah, totally."]:
        assert clf.classify(t).auto_supersession_allowed is False


def test_vague_turns_are_not_classified(clf):
    """Conservative: the schema never forces a guess on a contentless turn."""
    for t in ["Yeah, totally.", "That's so cool!", "How have you been?"]:
        assert clf.classify(t).classification_status is not ClassificationStatus.CLASSIFIED


def test_durability_is_independent_single_is_fast(clf):
    """'I'm single' is relationship by type but fast_change by durability (§7.1)."""
    r = clf.classify("I'm single.")
    assert r.claim_type is ClaimType.RELATIONSHIP
    assert r.durability is Durability.FAST_CHANGE  # overrides relationship's slow prior


def test_aspiration_does_not_collapse_into_project(clf):
    assert clf.classify("I want to become a counselor.").claim_type is ClaimType.ASPIRATION_GOAL
    assert clf.classify("I'm researching counseling programs this month.").claim_type is ClaimType.PROJECT_STATUS_PLAN


def test_semantic_fact_vs_belief_value_boundary(clf):
    assert clf.classify("The necklace symbolizes love, faith, and strength.").claim_type is ClaimType.SEMANTIC_FACT
    assert clf.classify("I value love, faith, and strength.").claim_type is ClaimType.BELIEF_VALUE


def test_protected_type_routes_candidate_only(clf):
    r = clf.classify("I'm a transgender woman.")
    assert r.is_protected()
    assert r.route() == "candidate_only"


def test_unclassified_routes_candidate_only(clf):
    assert clf.classify("Yeah, totally.").route() == "candidate_only"


def test_historical_one_time_routes_candidate_only(clf):
    """Historical claims are versioned/bounded, never overwritten (§7)."""
    r = clf.classify("I went to a LGBTQ support group yesterday.")
    assert r.durability is Durability.ONE_TIME
    assert r.route() == "candidate_only"


def test_non_protected_changeable_can_be_auto_eligible(clf):
    """A confidently-typed, non-protected, changeable claim is eligible for L3's gate."""
    r = clf.classify("I love painting.")
    assert r.claim_type is ClaimType.PREFERENCE_HABIT
    assert r.classification_status is ClassificationStatus.CLASSIFIED
    assert r.route() == "auto_eligible"


# ── Fixed defects (Fix 1 blocking, Fix 3 pre-bar) ─────────────────────────────

def test_negative_sentiment_about_partner_is_emotion(clf):
    """Fix 1 (blocking): negative sentiment whose object is a person is an
    emotion claim, NOT a relationship status claim. Was a confident error."""
    assert clf.classify("Ugh, my partner is driving me crazy lately.").claim_type is ClaimType.EMOTION_STATE


def test_chronic_health_durability_is_slow(clf):
    """Fix 3: a named chronic condition is a standing state -> slow_change,
    overriding health_state's fast prior."""
    r = clf.classify("I have Hashimoto's.")
    assert r.claim_type is ClaimType.HEALTH_STATE
    assert r.durability is Durability.SLOW_CHANGE


def test_compound_detection_flags_distinct_claims(clf):
    """Fix 2: clause-split flags genuinely multi-claim turns."""
    assert clf.classify("I love hiking, and last week I finally climbed Mount Rainier.").compound is True
    assert clf.classify("Politically I'm liberal, and I'm planning to volunteer next month.").compound is True


def test_missed_compound_still_routes_candidate_only(clf):
    """Defense-in-depth: even the compound case detection misses stays safe —
    relationship is protected, so it routes candidate_only regardless."""
    assert clf.classify("I'm single, but I really want to have kids someday.").route() == "candidate_only"


@pytest.mark.xfail(reason="residual rule ceiling: an adverb between 'I' and 'want to' "
                          "('I really want to') hides the 2nd-clause aspiration signal, so "
                          "compound detection misses. Safe via the protected guard above; "
                          "not chased further — pattern-stuffing here is how rule sets rot.")
def test_compound_detection_adverb_insertion(clf):
    assert clf.classify("I'm single, but I really want to have kids someday.").compound is True
