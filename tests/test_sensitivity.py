"""Claim Sovereignty Layer — Trigger 2 semantic recognition tests.

Skipped when the local embedding layer (revien[semantic]) is unavailable. The
acceptance-bar coverage (100% sensitive, 0 unsafe, measured FP, abstention) lives
in revien_bench/measure_sensitivity.py; these pin the invariants and the
degraded-safety fallback.
"""

import pytest

from revien.ingestion.claim_classifier import ClaimClassifier
from revien.sensitivity import SemanticSensitivityRecognizer, SensitivityRoute
from revien.supersession import Claim, SupersessionAction, SupersessionGate


@pytest.fixture(scope="module")
def rec():
    r = SemanticSensitivityRecognizer()
    if not r.is_available():
        pytest.skip("local embedding layer (revien[semantic]) not available")
    return r


# disclosures with NO seeded lexeme — the manifestation-2 / unseeded gap
SENSITIVE = [
    "I love being off the bottle.", "I'm finally clean.", "I'm a former smoker.",
    "I was assigned female at birth.", "I'm childfree by choice.",
    "I finally paid off the bankruptcy.", "I'm undocumented.",
    "I left the church.", "I terminated the pregnancy.",
]
# surface-word decoys — must read as NEUTRAL (the hard separation)
NEUTRAL_DECOYS = [
    "The recovery script failed.", "The transition animation is broken.",
    "The bottle is off the shelf.",
]


@pytest.mark.parametrize("text", SENSITIVE)
def test_sensitive_meaning_routes_candidate(rec, text):
    """100% of sensitive disclosures route candidate (sensitive or abstain),
    never confident_neutral — including unseeded domains."""
    v = rec.recognize(text)
    assert v.routes_candidate, f"LEAKED as {v.route.value}: {text!r} sens={v.sens_sim}"
    assert v.route is not SensitivityRoute.CONFIDENT_NEUTRAL


@pytest.mark.parametrize("text", NEUTRAL_DECOYS)
def test_surface_word_decoys_read_neutral(rec, text):
    """The adversarial controls share vocabulary with sensitive disclosures but
    carry no disclosure meaning — they must read confident_neutral."""
    assert rec.recognize(text).route is SensitivityRoute.CONFIDENT_NEUTRAL


def test_abstention_is_a_real_route(rec):
    """Abstention exists and routes candidate — a claim near neither bank is not
    forced into a confident call (the conservative-bands safety net)."""
    v = rec.recognize("The childfree event has no attendees under 18.")
    assert v.route is SensitivityRoute.ABSTAIN
    assert v.routes_candidate


def test_gate_routes_classified_sensitive_to_candidate(rec):
    """End-to-end: a confidently-misnamed sensitive claim that WOULD auto-supersede
    instead routes candidate once the recognizer is wired (primary recognition)."""
    clf = ClaimClassifier()
    gate = SupersessionGate(recognizer=rec)
    d = gate.evaluate(Claim("I love being off the bottle.", clf.classify("I love being off the bottle.")),
                      Claim("I don't enjoy being off the bottle.", clf.classify("I don't enjoy being off the bottle.")))
    assert d.action is SupersessionAction.CANDIDATE
    assert d.reason.startswith("semantic_sensitive:")


def test_neutral_preference_can_still_auto(rec):
    """The recognizer is conservative, not a blanket block — a confidently-neutral
    preference change still proceeds to auto."""
    clf = ClaimClassifier()
    gate = SupersessionGate(recognizer=rec)
    d = gate.evaluate(Claim("I love painting.", clf.classify("I love painting.")),
                      Claim("I don't like painting anymore.", clf.classify("I don't like painting anymore.")))
    assert d.action is SupersessionAction.AUTO_SUPERSEDE


def test_unavailable_recognizer_abstains_degraded_safe():
    """No embeddings -> the recognizer cannot assess -> abstains (candidate), never
    silently confident-neutral. Degraded-safety, not unsafe."""
    class _BrokenEmbedder:
        def embed(self, texts):
            raise RuntimeError("no model")
    r = SemanticSensitivityRecognizer(embedder=_BrokenEmbedder())
    assert r.is_available() is False
    v = r.recognize("I love being off the bottle.")
    assert v.route is SensitivityRoute.ABSTAIN
    assert v.available is False
