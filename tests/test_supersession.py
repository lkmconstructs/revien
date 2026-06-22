"""Claim Sovereignty Layer — Leg 3 supersession gate: guarantee tests.

The gate's CONTRACT: aggressive detection, conservative resolution. Auto-supersede
only when every axis clears; protected/irongrip/historical/stable/unclassified all
route away from auto. The headline safety invariant — zero unsafe auto-fires — is
asserted directly against the fixture corpus.
"""

import pytest

from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import Claim, Lock, SupersessionAction, SupersessionGate


@pytest.fixture
def clf():
    return ClaimClassifier()


@pytest.fixture
def gate():
    return SupersessionGate()


def _c(clf, text, lock=Lock.NORMAL):
    return Claim(text, clf.classify(text), lock)


def test_scope_guard_emotion_cannot_touch_relationship(clf, gate):
    """Invariant: negative sentiment (emotion) can't supersede a relationship claim."""
    d = gate.evaluate(_c(clf, "I'm married."), _c(clf, "I'm so annoyed at Jesse today."))
    assert d.action is SupersessionAction.NO_CONFLICT
    assert d.scope_overlap < 1.0


def test_scope_guard_shared_entity_different_dimension(clf, gate):
    """The founder-claim guard: shared entity, different predicate dimension -> no conflict."""
    d = gate.evaluate(_c(clf, "Revien is my company."),
                      _c(clf, "I'm so frustrated with Revien today."))
    assert d.action is SupersessionAction.NO_CONFLICT


def test_preference_polarity_auto_supersedes(clf, gate):
    """Non-protected, changeable, classified, scoped contradiction -> auto."""
    d = gate.evaluate(_c(clf, "I love painting."),
                      _c(clf, "Honestly, I don't like painting anymore."))
    assert d.action is SupersessionAction.AUTO_SUPERSEDE


def test_compatible_same_type_is_no_conflict(clf, gate):
    """Same type + subject but different object is NOT a contradiction (love both)."""
    d = gate.evaluate(_c(clf, "I love painting."), _c(clf, "I love hiking."))
    assert d.action is SupersessionAction.NO_CONFLICT


def test_protected_relationship_routes_candidate(clf, gate):
    """Protected category never auto-supersedes, even on a clean status change."""
    d = gate.evaluate(_c(clf, "I'm single."), _c(clf, "I'm married now."))
    assert d.action is SupersessionAction.CANDIDATE


def test_irongrip_versions_not_auto(clf, gate):
    """Iron-grip versions on explicit instruction only — precedence over protected."""
    d = gate.evaluate(_c(clf, "I'm married.", Lock.IRONGRIP), _c(clf, "I'm divorced now."))
    assert d.action is SupersessionAction.VERSION_LOCKED


def test_historical_one_time_routes_candidate(clf, gate):
    """Historical claims are versioned/bounded, never overwritten."""
    d = gate.evaluate(_c(clf, "I went to the museum yesterday."),
                      _c(clf, "Actually I didn't go to the museum, I went to the aquarium."))
    assert d.action is SupersessionAction.CANDIDATE


def test_unclassified_new_never_auto(clf, gate):
    """Precondition 2: an unclassified new claim can't drive supersession."""
    d = gate.evaluate(_c(clf, "I love painting."),
                      _c(clf, "yeah, not really my thing these days."))
    assert d.action is SupersessionAction.NO_CONFLICT


def test_no_unsafe_autofire_across_corpus():
    """THE safety invariant: nothing auto-fires that the label forbids, and the
    gate logic is correct on every pair it engages."""
    from revien_bench.measure_supersession import measure
    r = measure()
    assert r["unsafe_autos"] == []
    engaged_correct, engaged_total = r["gate_logic_accuracy"]
    assert engaged_correct == engaged_total  # 100% routing on what it sees
