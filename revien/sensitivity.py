"""Claim Sovereignty Layer — Trigger 2: semantic sensitive recognition.

The tripwire catches sensitive LEXEMES; this recognizes sensitive MEANING,
independent of whether the wording hits a seeded term. It closes the gap the
tripwire only patches ("I love being off the bottle" — sensitive, no lexeme).

Mechanism (local-first, no cloud on the default path): embed the claim with the
local sentence embedder (bge-small) and compare to two prototype banks —
SENSITIVE disclosures spanning many domains (recovery, identity, health,
finance-ruin, immigration, criminal-record, religious-deconversion, reproductive,
family-estrangement, abuse, ...) and NEUTRAL everyday/technical statements. The
routing is TWO-SIDED and conservative:

    sens_sim >= SENS_THRESHOLD                       -> SENSITIVE  -> candidate
    sens_sim < SENS_THRESHOLD and neut_sim >= NEUT   -> CONFIDENT_NEUTRAL -> auto-ok
    otherwise (near neither bank)                    -> ABSTAIN    -> candidate

ABSTENTION IS REQUIRED, NOT A FAILURE. A claim that is neither clearly sensitive
nor confidently neutral routes candidate — because a backend that COMMITS to
"non-sensitive" on an ambiguous disclosure is the exact confident-misclassification
that created this gap. This is the same conservative-bands principle as the claim
classifier: allowed to be unsure and route safe.

This recognizes COST-OF-ERASURE as a property, not vocabulary: an unseeded-domain
disclosure embeds near the sensitive bank (vulnerable self-disclosure clusters in
embedding space) while a surface-word decoy ("the recovery script failed") embeds
near the neutral bank. The prototype banks are anchors, not an exhaustive list —
the abstain band is the safety net under a TRULY novel disclosure that is near
neither bank.

Requires the local embedding layer (`revien[semantic]`). Without it the recognizer
is UNAVAILABLE and the system runs in degraded-safety (tripwire-only) mode — which
is NOT launch-safe. See HYBRID_BACKEND_TRIGGERS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

# ── Prototype banks ───────────────────────────────────────────────────────────
# SENSITIVE: first-person vulnerable self-disclosures spanning MANY domains —
# deliberately beyond the tripwire's seeded four, so recognition is of the
# cost-of-erasure property, not a seeded vocabulary. Broad coverage widens the
# region of embedding space treated as sensitive; the abstain band covers the rest.
SENSITIVE_PROTOTYPES: List[str] = [
    # recovery / substance
    "I am in recovery from addiction.", "I have been sober for a year.",
    "I stopped drinking alcohol.", "I am a recovering alcoholic.",
    "I quit smoking and stay away from it now.",
    # gender / sexuality identity
    "I am transgender.", "I am transitioning my gender.",
    "I am questioning my gender identity.", "I was assigned female at birth.",
    "I am gay.", "I came out as queer.",
    # health / mental health
    "I was diagnosed with a serious medical condition.",
    "I struggle with depression and anxiety.", "I am in therapy for my mental health.",
    "I am in treatment for an illness.", "I am nursing my newborn.",
    # legal / immigration / record
    "I am undocumented and do not have legal status.",
    "I have a criminal record.", "I was incarcerated.", "I got my record expunged.",
    # finance
    "I went through bankruptcy and financial ruin.",
    "I am in serious debt and financial trouble.",
    # faith
    "I left my religion and lost my faith.", "I deconverted from the church.",
    # reproductive / family
    "I had an abortion.", "I terminated a pregnancy.", "I had a miscarriage.",
    "I am estranged from my family.", "I am childfree by choice.",
    "I cannot have children.",
    # abuse / trauma
    "I am a survivor of abuse.", "I experienced trauma I do not talk about.",
]

# NEUTRAL: everyday + technical statements, incl. surface-word decoys that share
# vocabulary with sensitive disclosures but carry no disclosure meaning.
NEUTRAL_PROTOTYPES: List[str] = [
    "I like painting and hiking on weekends.", "My favorite food is sushi.",
    "I prefer tea over coffee.", "The meeting is scheduled for three o'clock.",
    "I painted the fence in the backyard.", "The software build passed all tests.",
    "The script failed with an error.", "I went to the grocery store.",
    "I enjoy reading science fiction novels.",
    "The animation is broken in the latest release.",
    "I am working on a new project at the office.", "The bottle is on the top shelf.",
    "We watched a movie last night.", "I bought a new laptop.",
    "The recovery of the lost file took an hour.",
    "The data migration finished successfully.", "I reorganized my bookshelf.",
]


class SensitivityRoute(str, Enum):
    SENSITIVE = "sensitive"                 # -> candidate
    ABSTAIN = "abstain"                     # -> candidate (required, not a failure)
    CONFIDENT_NEUTRAL = "confident_neutral" # -> auto-eligible


@dataclass
class SensitivityVerdict:
    route: SensitivityRoute
    sens_sim: float
    neut_sim: float
    available: bool = True

    @property
    def routes_candidate(self) -> bool:
        """True for SENSITIVE and ABSTAIN — both route to human review."""
        return self.route in (SensitivityRoute.SENSITIVE, SensitivityRoute.ABSTAIN)


def _normalize(vec: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SemanticSensitivityRecognizer:
    """Embedding-prototype semantic sensitivity recognizer (Trigger 2).

    Conservative two-sided routing with required abstention. The embedder and
    prototype embeddings load lazily on first ``recognize`` so construction is
    cheap and import-safe; if the embedding layer is unavailable the recognizer
    reports ``available=False`` and abstains (routes candidate) — degraded-safety,
    never a silent confident-neutral.
    """

    # Thresholds are conservative and tunable against measure_sensitivity.py.
    SENS_THRESHOLD = 0.66   # >= -> sensitive (data gap: neutral<=0.64, sensitive>=0.69)
    NEUT_THRESHOLD = 0.62   # >= (and sens<SENS) -> confident neutral; else abstain

    def __init__(self, embedder=None,
                 sens_prototypes: Optional[List[str]] = None,
                 neut_prototypes: Optional[List[str]] = None,
                 sens_threshold: Optional[float] = None,
                 neut_threshold: Optional[float] = None):
        self._embedder = embedder
        self._sens_texts = sens_prototypes if sens_prototypes is not None else SENSITIVE_PROTOTYPES
        self._neut_texts = neut_prototypes if neut_prototypes is not None else NEUTRAL_PROTOTYPES
        self.sens_threshold = sens_threshold if sens_threshold is not None else self.SENS_THRESHOLD
        self.neut_threshold = neut_threshold if neut_threshold is not None else self.NEUT_THRESHOLD
        self._sens_vecs: Optional[List[List[float]]] = None
        self._neut_vecs: Optional[List[List[float]]] = None
        self._available: Optional[bool] = None

    def _ensure(self) -> bool:
        """Lazily build the embedder + prototype vectors. Returns availability."""
        if self._available is not None:
            return self._available
        try:
            if self._embedder is None:
                from revien.semantic.index import build_embedder
                self._embedder = build_embedder()
            self._sens_vecs = [_normalize(v) for v in self._embedder.embed(self._sens_texts)]
            self._neut_vecs = [_normalize(v) for v in self._embedder.embed(self._neut_texts)]
            self._available = bool(self._sens_vecs and self._neut_vecs)
        except Exception:  # noqa: BLE001 - any failure => unavailable, abstain
            self._available = False
        return self._available

    def is_available(self) -> bool:
        return self._ensure()

    def recognize(self, text: str) -> SensitivityVerdict:
        if not (text and text.strip()):
            # Empty content cannot be confirmed neutral -> abstain (safe).
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=self._available is True)
        if not self._ensure():
            # No embeddings -> cannot assess -> abstain (degraded safety, never auto).
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=False)
        try:
            v = _normalize(self._embedder.embed([text])[0])
        except Exception:  # noqa: BLE001
            return SensitivityVerdict(SensitivityRoute.ABSTAIN, 0.0, 0.0, available=False)
        sens = max(_cos(v, p) for p in self._sens_vecs)
        neut = max(_cos(v, p) for p in self._neut_vecs)
        if sens >= self.sens_threshold:
            route = SensitivityRoute.SENSITIVE
        elif neut >= self.neut_threshold:
            route = SensitivityRoute.CONFIDENT_NEUTRAL
        else:
            route = SensitivityRoute.ABSTAIN
        return SensitivityVerdict(route, round(sens, 4), round(neut, 4), available=True)
