"""Claim Sovereignty Layer — claim vocabulary (the CLAIM_TAXONOMY.md contract, in code).

Enums + the classification-result shape + the default protected set. This module
is the single source of truth for the taxonomy vocabulary; the classifier
(revien/ingestion/claim_classifier.py) and, later, Leg 3 supersession both import
from here. No store/node coupling — Leg 2.5 is measured in isolation before Leg 3
consumes any of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ClaimType(str, Enum):
    """The 12 blessed claim types (CLAIM_TAXONOMY.md §3). No merges."""
    IDENTITY = "identity"
    RELATIONSHIP = "relationship"
    PREFERENCE_HABIT = "preference_habit"
    CURRENT_STATE = "current_state"
    HEALTH_STATE = "health_state"
    EMOTION_STATE = "emotion_state"
    HISTORICAL_EVENT = "historical_event"
    PROJECT_STATUS_PLAN = "project_status_plan"
    SCHEDULE = "schedule"
    BELIEF_VALUE = "belief_value"
    ASPIRATION_GOAL = "aspiration_goal"
    SEMANTIC_FACT = "semantic_fact"


class Durability(str, Enum):
    """How changeable a claim is by nature (§4). Classified PER CLAIM, not by type."""
    STABLE = "stable"
    SLOW_CHANGE = "slow_change"
    FAST_CHANGE = "fast_change"
    ONE_TIME = "one_time"
    UNKNOWN = "unknown"


class ClassificationStatus(str, Enum):
    """Outcome of the type-classification attempt (§5). Not a 13th type."""
    CLASSIFIED = "classified"
    LOW_CONFIDENCE = "low_confidence"
    UNCLASSIFIED = "unclassified"


# Default hard-floor-protected set (§7.3). Operator-configurable, but the learning
# loop may NEVER modify it. Membership here means: barred from automatic mutation
# regardless of classification confidence.
PROTECTED_DEFAULT = frozenset({
    ClaimType.HEALTH_STATE,
    ClaimType.IDENTITY,
    ClaimType.BELIEF_VALUE,
    ClaimType.RELATIONSHIP,
})

# Durability PRIOR per type (§7.1). A PRIOR ONLY — per-claim signals may override
# it. Implementations must never collapse to `durability = DEFAULTS[claim_type]`;
# that shortcut is the rot the contract names.
DURABILITY_PRIOR = {
    ClaimType.IDENTITY: Durability.STABLE,
    ClaimType.SEMANTIC_FACT: Durability.STABLE,
    ClaimType.RELATIONSHIP: Durability.SLOW_CHANGE,
    ClaimType.PREFERENCE_HABIT: Durability.SLOW_CHANGE,
    ClaimType.BELIEF_VALUE: Durability.SLOW_CHANGE,
    ClaimType.ASPIRATION_GOAL: Durability.SLOW_CHANGE,
    ClaimType.CURRENT_STATE: Durability.FAST_CHANGE,
    ClaimType.EMOTION_STATE: Durability.FAST_CHANGE,
    ClaimType.HEALTH_STATE: Durability.FAST_CHANGE,
    ClaimType.PROJECT_STATUS_PLAN: Durability.FAST_CHANGE,
    ClaimType.SCHEDULE: Durability.FAST_CHANGE,
    ClaimType.HISTORICAL_EVENT: Durability.ONE_TIME,
}


@dataclass
class ClassificationResult:
    """The canonical classifier output record (CLAIM_TAXONOMY.md §10).

    `auto_supersession_allowed` is ALWAYS False out of Leg 2.5 — only Leg 3,
    applying its full gate, may grant auto-supersession. `compound` marks a turn
    carrying multiple distinct claims; per the gate ruling, compound turns are
    candidate_only at Leg 3 (recorded here so Leg 3 can enforce it).
    """
    claim_type: Optional[ClaimType]
    claim_type_confidence: float
    classification_status: ClassificationStatus
    durability: Durability
    durability_confidence: float
    boundary_notes: List[str] = field(default_factory=list)
    auto_supersession_allowed: bool = False  # invariant: always False out of L2.5
    compound: bool = False

    def is_protected(self, protected_set=PROTECTED_DEFAULT) -> bool:
        """True if this claim's type is in the protected set (§7.3)."""
        return self.claim_type is not None and self.claim_type in protected_set

    def route(self, protected_set=PROTECTED_DEFAULT) -> str:
        """Disposition toward Leg 3 (§6, §10): 'candidate_only' or 'auto_eligible'.

        Conservative by construction: anything unclassified / low-confidence /
        unknown-durability / compound / protected is candidate_only. Only a
        confidently-classified, non-protected, changeable, non-compound claim is
        even ELIGIBLE for Leg 3's gate (which may still route it to candidate).
        """
        if self.classification_status is not ClassificationStatus.CLASSIFIED:
            return "candidate_only"
        if self.compound:
            return "candidate_only"
        if self.durability in (Durability.UNKNOWN, Durability.ONE_TIME):
            # one_time = historical: versioned/bounded, never overwritten (§7).
            return "candidate_only"
        if self.is_protected(protected_set):
            return "candidate_only"
        return "auto_eligible"

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type.value if self.claim_type else None,
            "claim_type_confidence": round(self.claim_type_confidence, 3),
            "classification_status": self.classification_status.value,
            "durability": self.durability.value,
            "durability_confidence": round(self.durability_confidence, 3),
            "boundary_notes": list(self.boundary_notes),
            "auto_supersession_allowed": self.auto_supersession_allowed,
            "compound": self.compound,
        }
