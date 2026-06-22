"""Claim Sovereignty Layer — Leg 3: gated supersession.

Detect contradictions aggressively, resolve CONSERVATIVELY. A new claim may
supersede an existing one ONLY when every axis clears; otherwise the conflict is
surfaced for human review (candidate) or versioned (iron-grip / historical),
never silently overwritten.

The gate (auto-supersede requires ALL):
  * both claims `classified` (never low_confidence / unclassified)   [precond 2]
  * the new claim is single-claim (not compound)                      [precond 1]
  * a SCOPED contradiction exists — same subject AND same predicate
    dimension (claim_type), with incompatible value. scope_overlap is a
    FIRST-CLASS score, not folded into confidence: it is what stops
    "I'm frustrated with Revien today" (emotion) from touching
    "I founded Revien" (identity) despite the shared entity.
  * the existing claim's durability is CHANGEABLE (fast/slow) — never
    stable (identity) and never one_time (historical: versioned/bounded,
    never overwritten)
  * the existing claim is NOT protected (health/identity/belief/relationship
    by default) — protected categories always route to candidate
  * the existing claim is NOT iron-grip — iron-grip versions on explicit
    instruction only, never auto
  * the new claim is explicit + high-confidence

MVP honesty: contradiction detection is rule-based and CONSERVATIVE. It catches
single-valued status changes (relationship/location), favorite-value changes, and
polarity flips on a shared object. It does NOT yet reason about cross-subject
implication ("I'm nursing" vs "Silas weaned") — those are missed, which is SAFE
(a missed contradiction leaves the old claim standing, caught later) rather than
dangerous (a wrong supersession silently corrupts memory). Defense-in-depth is a
backstop, not a license: the gate's primary checks must be right on their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from revien.claims import (
    ClaimType,
    ClassificationResult,
    ClassificationStatus,
    Durability,
    PROTECTED_DEFAULT,
    SENSITIVE_FLOOR,
)
from revien.tripwire import DistrustTripwire, verify_tripwire


class Lock(str, Enum):
    """Per-claim lock authority. 'protected' is derived from claim_type ∈ the
    protected set; this enum carries the explicit user lock."""
    NORMAL = "normal"
    IRONGRIP = "irongrip"


class SupersessionAction(str, Enum):
    NO_CONFLICT = "no_conflict"        # nothing to do
    AUTO_SUPERSEDE = "auto_supersede"  # gate fully cleared — replace
    CANDIDATE = "candidate"            # conflict surfaced for human review
    VERSION_LOCKED = "version_locked"  # iron-grip: version on explicit instruction only


@dataclass
class Claim:
    """A claim under supersession consideration: text + its classification + lock."""
    text: str
    result: ClassificationResult
    lock: Lock = Lock.NORMAL


@dataclass
class SupersessionDecision:
    action: SupersessionAction
    reason: str
    scope_overlap: float
    contradiction: bool
    trace: List[str] = field(default_factory=list)


# Types where a subject holds ONE value at a time — a different value is a
# contradiction even without a shared object. (Preferences/events are
# multi-valued: you can love many things, attend many events.)
_SINGLE_VALUED = frozenset({
    ClaimType.RELATIONSHIP, ClaimType.IDENTITY, ClaimType.CURRENT_STATE,
    ClaimType.HEALTH_STATE,
})

_RELATIONSHIP_STATUS = ("single", "married", "divorced", "engaged", "dating", "widowed")
_NEG = re.compile(
    r"\b(?:not|don'?t|doesn'?t|didn'?t|isn'?t|wasn'?t|weren'?t|aren'?t|won'?t|"
    r"hasn'?t|haven'?t|hadn'?t|can'?t|couldn'?t|wouldn'?t|no longer|stopped|quit|"
    r"anymore|never|cancell?ed|gave up|broke up|ended)\b", re.I)
_STOP = {
    "i", "im", "i'm", "am", "is", "are", "was", "were", "a", "an", "the", "my",
    "to", "of", "in", "on", "and", "but", "so", "really", "now", "anymore",
    "actually", "honestly", "that", "this", "it", "with", "for", "about",
    "no", "not", "longer", "still", "just", "ve", "have", "had", "been",
}


def _subject(text: str) -> str:
    """Crude subject: first-person -> 'self'; else the first capitalized entity."""
    t = text.strip()
    if re.match(r"(?i)^(?:ugh|argh|honestly|actually|yeah|well|so)?[,\s]*i\b", t) or \
       re.search(r"\bmy\b", t, re.I):
        return "self"
    m = re.search(r"\b([A-Z][a-z]+)\b", t)
    return m.group(1).lower() if m else "unknown"


def _content_tokens(text: str) -> Set[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return {w for w in toks if w not in _STOP and len(w) > 2}


def _polarity(text: str) -> int:
    """+1 affirmative, -1 negated."""
    return -1 if _NEG.search(text) else 1


def _status_value(text: str) -> Optional[str]:
    low = text.lower()
    for s in _RELATIONSHIP_STATUS:
        if re.search(rf"\b{s}\b", low):
            return s
    return None


def _favorite_value(text: str) -> Optional[str]:
    m = re.search(r"\bfavou?rite\s+\w+\s+is\s+([a-z]+)", text.lower())
    return m.group(1) if m else None


def _place(text: str) -> Optional[str]:
    m = re.search(r"\b(?:from|live in|moved to|based in|grew up in)\s+([a-z]+)", text.lower())
    return m.group(1) if m else None


class SupersessionGate:
    """The conservative supersession gate (Leg 3)."""

    SCOPE_FULL = 0.999   # same subject AND same predicate dimension
    NEW_CONF_BAR = 0.60  # new claim must be confidently classified

    def __init__(self, protected_set: Set[ClaimType] = PROTECTED_DEFAULT,
                 new_conf_bar: float = NEW_CONF_BAR,
                 tripwire: Optional[DistrustTripwire] = None):
        self.protected_set = protected_set
        self.new_conf_bar = new_conf_bar
        # Interim distrust tripwire (content-level, strictly additive). Default-on.
        # BEHAVIORAL config-floor enforcement (invariant 4): an injected tripwire
        # that fails to trip the core reproduced-harm sentinels — a blinded subclass,
        # a duck-typed no-op, a shrunk lexicon — is REFUSED. We fall back to a
        # known-good DistrustTripwire so the core can never be removed through the
        # injection/subclass back door (which a lexical-only check could not see).
        candidate_tw = tripwire if tripwire is not None else DistrustTripwire()
        self.tripwire = candidate_tw if verify_tripwire(candidate_tw) else DistrustTripwire()

    # ── scope_overlap: a first-class score ────────────────────────────────────
    def _scope_overlap(self, e: Claim, n: Claim) -> float:
        """Same subject AND same predicate dimension (claim_type). 0 unless BOTH
        claims are confidently classified (precondition 2 enforced here)."""
        if (e.result.classification_status is not ClassificationStatus.CLASSIFIED
                or n.result.classification_status is not ClassificationStatus.CLASSIFIED):
            return 0.0
        subject_match = _subject(e.text) == _subject(n.text)
        type_match = e.result.claim_type == n.result.claim_type
        if subject_match and type_match:
            return 1.0
        return 0.5 if (subject_match or type_match) else 0.0

    # ── contradiction (only meaningful at full scope) ─────────────────────────
    def _contradicts(self, e: Claim, n: Claim) -> bool:
        ct = e.result.claim_type  # == n's at full scope
        et, nt = e.text, n.text

        # Single-valued status frames: a different value is a contradiction.
        if ct is ClaimType.RELATIONSHIP:
            ev, nv = _status_value(et), _status_value(nt)
            if ev and nv and ev != nv:
                return True
        if ct in (ClaimType.IDENTITY,):
            pe, pn = _place(et), _place(nt)
            if pe and pn and pe != pn:
                return True
        fe, fn = _favorite_value(et), _favorite_value(nt)
        if fe and fn and fe != fn:
            return True

        # Polarity flip on a shared object (works for any type).
        shared = _content_tokens(et) & _content_tokens(nt)
        if shared and _polarity(et) != _polarity(nt):
            return True

        # Single-valued state change with an explicit negation in the new claim
        # (e.g. "I'm swamped" -> "I'm not busy anymore") even w/o lexical overlap.
        if ct in _SINGLE_VALUED and _polarity(nt) == -1 and _polarity(et) == 1:
            return True

        return False

    # ── the gate ──────────────────────────────────────────────────────────────
    # The non-configurable protection level the interim floor pins the generic
    # default to. Named for verification; the floor below does not read config.
    FLOOR_LEVEL = SENSITIVE_FLOOR

    def evaluate(self, existing: Claim, new: Claim) -> SupersessionDecision:
        trace: List[str] = []

        # ── INTERIM SENSITIVE FLOOR (non-configurable, checked FIRST) ─────────
        # A claim that is not confidently classified might BE a sensitive claim
        # the rule classifier could not name ("I'm sober" -> unclassified). It is
        # therefore NEVER auto-superseded. This guard runs ahead of all
        # configurable logic (protected_set, thresholds) and reads NO config, so
        # the protection cannot be lowered through the config back door — the floor
        # holds even with protected_set=frozenset(). NOTE: that promise is scoped
        # to the UNCLASSIFIED default only; emptying protected_set still de-protects
        # CLASSIFIED named-sensitive claims by design (governance). And the floor
        # covers only manifestation 1 of the recognition gap (sensitive content
        # classified as NOTHING) — content confidently MISNAMED into a non-protected
        # type ("I love being sober" -> preference_habit) is NOT floored; that is
        # the hybrid backend's job (HYBRID_BACKEND_TRIGGERS.md Trigger 2).
        if existing.result.classification_status is not ClassificationStatus.CLASSIFIED:
            trace.append("sensitive_floor:existing_not_classified")
            return SupersessionDecision(
                SupersessionAction.NO_CONFLICT,
                "sensitive_floor_unclassified_existing_never_auto", 0.0, False, trace)

        so = self._scope_overlap(existing, new)
        trace.append(f"scope_overlap={so}")

        if so < self.SCOPE_FULL:
            # Different subject or predicate dimension — not about the same thing.
            return SupersessionDecision(
                SupersessionAction.NO_CONFLICT, "out_of_scope", so, False, trace)

        contradiction = self._contradicts(existing, new)
        trace.append(f"contradiction={contradiction}")
        if not contradiction:
            return SupersessionDecision(
                SupersessionAction.NO_CONFLICT, "scoped_but_compatible", so, False, trace)

        # ── a scoped contradiction exists — resolve conservatively ──
        if existing.lock is Lock.IRONGRIP:
            trace.append("irongrip")
            return SupersessionDecision(
                SupersessionAction.VERSION_LOCKED,
                "irongrip_versions_on_explicit_only", so, True, trace)

        if existing.result.durability is Durability.ONE_TIME:
            trace.append("one_time_historical")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE,
                "historical_versioned_not_overwritten", so, True, trace)

        if existing.result.durability is Durability.STABLE:
            trace.append("stable")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, "stable_not_auto", so, True, trace)

        if existing.result.is_protected(self.protected_set):
            trace.append("protected")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, "protected_requires_review", so, True, trace)

        if new.result.compound:
            trace.append("new_compound")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, "compound_candidate_only", so, True, trace)

        if new.result.claim_type_confidence < self.new_conf_bar:
            trace.append("low_new_confidence")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, "new_not_confident_enough", so, True, trace)

        if existing.result.durability not in (Durability.FAST_CHANGE, Durability.SLOW_CHANGE):
            trace.append("durability_unknown")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, "durability_not_changeable", so, True, trace)

        # ── DISTRUST TRIPWIRE (strictly additive, content-level, type-independent) ──
        # The gate would auto-supersede. Before it does, distrust the classifier's
        # type: if the EXISTING or NEW claim's raw/normalized CONTENT names a
        # sensitive domain, route to candidate instead — covering the
        # confidently-misnamed manifestation ("I love being sober" -> preference_habit)
        # the type-keyed protected guard misses. This can ONLY downgrade auto ->
        # candidate (invariant 1); it never reaches this line on a non-auto path.
        # It does NOT close Trigger 2 (lexemes, not meaning — it misses unlexed
        # sensitive content); it is the interim promise, not the fix.
        tripped = self.tripwire.check(existing.text) or self.tripwire.check(new.text)
        if tripped:
            trace.append(f"tripwire:{tripped}")
            return SupersessionDecision(
                SupersessionAction.CANDIDATE, f"tripwire_distrust:{tripped}", so, True, trace)

        # Every axis cleared: classified, single-claim, scoped contradiction,
        # changeable, non-protected, non-irongrip, confident new claim.
        trace.append("all_clear")
        return SupersessionDecision(
            SupersessionAction.AUTO_SUPERSEDE, "gate_cleared", so, True, trace)


@dataclass
class SupersessionMetrics:
    """Production instrumentation for the candidate-queue WORKLOAD trigger.

    Per the gate ruling, the coverage decision ships rule-based and the
    candidate-queue depth is instrumented in REAL USE — volume, not a benchmark,
    decides whether the hybrid backend's workload trigger fires. Feed every gate
    decision via ``record``; read ``candidate_queue_depth`` (items awaiting human
    adjudication) and the dispositions over time.

    This is the WORKLOAD trigger only. It is independent of — and must never be
    read as covering — the SAFETY trigger (sensitive-recognition), which fires
    regardless of queue depth. See HYBRID_BACKEND_TRIGGERS.md.
    """
    auto: int = 0
    candidate: int = 0
    version_locked: int = 0
    no_conflict: int = 0
    # SAFETY-relevant overlays (not dispositions). A floor catch has
    # action=NO_CONFLICT and a tripwire catch action=CANDIDATE, so without these
    # counters sensitive activity would blur into the ordinary disposition counts
    # and a sensitive-heavy stream could read as a quiet/normal queue. Counted
    # separately so a workload-defer decision can't be blind to sensitive volume —
    # reinforces the Trigger-1/Trigger-2 firewall.
    sensitive_floor_caught: int = 0
    tripwire_caught: int = 0
    tripwire_by_domain: Dict[str, int] = field(default_factory=dict)

    def record(self, decision: SupersessionDecision) -> None:
        a = decision.action
        if a is SupersessionAction.AUTO_SUPERSEDE:
            self.auto += 1
        elif a is SupersessionAction.CANDIDATE:
            self.candidate += 1
        elif a is SupersessionAction.VERSION_LOCKED:
            self.version_locked += 1
        else:
            self.no_conflict += 1
        if "sensitive_floor:existing_not_classified" in decision.trace:
            self.sensitive_floor_caught += 1
        if decision.reason.startswith("tripwire_distrust:"):
            self.tripwire_caught += 1
            domain = decision.reason.split(":", 1)[1]
            self.tripwire_by_domain[domain] = self.tripwire_by_domain.get(domain, 0) + 1

    @property
    def total(self) -> int:
        return self.auto + self.candidate + self.version_locked + self.no_conflict

    @property
    def candidate_queue_depth(self) -> int:
        """Decisions awaiting human adjudication (candidate + iron-grip version)."""
        return self.candidate + self.version_locked

    @property
    def auto_fire_rate(self) -> float:
        """Auto-fires over decisions that ACTED (excludes no_conflict no-ops)."""
        acted = self.auto + self.candidate + self.version_locked
        return self.auto / acted if acted else 0.0

    def snapshot(self) -> dict:
        return {
            "total": self.total,
            "auto": self.auto,
            "candidate": self.candidate,
            "version_locked": self.version_locked,
            "no_conflict": self.no_conflict,
            "candidate_queue_depth": self.candidate_queue_depth,
            "sensitive_floor_caught": self.sensitive_floor_caught,
            "tripwire_caught": self.tripwire_caught,
            "tripwire_by_domain": dict(self.tripwire_by_domain),
            "auto_fire_rate": round(self.auto_fire_rate, 4),
        }
