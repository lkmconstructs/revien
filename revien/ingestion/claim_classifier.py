"""Claim Sovereignty Layer — Leg 2.5b: rule-based claim classifier.

Zero-dependency, deterministic classifier producing the CLAIM_TAXONOMY.md §10
record: claim_type (+ confidence + status), durability (+ confidence, an
INDEPENDENT axis), boundary_notes, compound, auto_supersession_allowed (always
False). Rule-based on purpose — measured in isolation (L2.5d) before Leg 3
consumes it. An LLM backend may come later; it is never first (too expensive,
too easy to hide uncertainty behind plausible prose).

Design (per the gate ruling):
  * CONSERVATIVE bands — prefer "unknown" over a wrong guess. A single strong,
    uncontested signal classifies; weak/contested signals fall to low_confidence
    or unclassified. The schema never forces a guess.
  * MARGIN-based confidence — a near-tie between two types is its own ambiguity
    detector and drives the boundary_notes the contract §8 wants.
  * Durability is classified PER CLAIM from tense/transience signals, with the
    type only a prior it may override (§7.1) — never durability = DEFAULTS[type].
  * COMPOUND turns (≥2 distinct strong claims) are flagged; Leg 3 treats them as
    candidate_only.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from revien.claims import (
    ClaimType,
    ClassificationResult,
    ClassificationStatus,
    Durability,
    DURABILITY_PRIOR,
)

# Signal weights.
_STRONG = 0.65
_MEDIUM = 0.42
_WEAK = 0.25

# Conservative bands (tunable in L2.5d against the confusion matrix).
_CLASSIFY_HI = 0.60   # >= -> classified
_LOW_FLOOR = 0.35     # [floor, hi) -> low_confidence ; < floor -> unclassified

# Known boundary pairs (§8) — when the top two collide here, annotate it.
_BOUNDARY_PAIRS = {
    frozenset({ClaimType.SEMANTIC_FACT, ClaimType.BELIEF_VALUE}),
    frozenset({ClaimType.ASPIRATION_GOAL, ClaimType.PROJECT_STATUS_PLAN}),
    frozenset({ClaimType.SCHEDULE, ClaimType.PROJECT_STATUS_PLAN}),
}

# ── Type signals: (regex, weight) per claim type ──────────────────────────────
_TYPE_PATTERNS: Dict[ClaimType, List[Tuple[str, float]]] = {
    ClaimType.IDENTITY: [
        (r"\bi(?:'m| am) (?:a |an )?(?:transgender|trans|nonbinary|non-binary|gay|lesbian|bisexual|queer|cisgender)\b", _STRONG),
        (r"\bi was born in\b", _STRONG),
        (r"\bi identify as\b", _STRONG),
        (r"\bmy name is\b", _STRONG),
        (r"\bi(?:'m| am) (?:a |an )?(?:swedish|german|american|british|canadian|french|italian|spanish|irish|dutch|polish)\b", _MEDIUM),
        (r"\bi(?:'m| am) (?:originally )?from [a-z]+\b", _MEDIUM),
    ],
    ClaimType.RELATIONSHIP: [
        # A bare person-mention is only MEDIUM — it is weak evidence of a
        # relationship STATUS claim (you mention your partner while talking about
        # something else). The explicit status assertions below are the STRONG
        # signal. This is what stops "my partner is driving me crazy" (an emotion
        # whose object is a person) from being read as a relationship claim.
        (r"\bmy (?:husband|wife|partner|boyfriend|girlfriend|spouse|fianc[eé]e?)\b", _MEDIUM),
        (r"\bi(?:'m| am) (?:married|single|divorced|engaged|dating|widowed|seeing someone)\b", _STRONG),
        (r"\bwe(?:'ve| have) been (?:married|together|friends|dating)\b", _STRONG),
        (r"\bmy (?:mom|mum|dad|mother|father|sister|brother|son|daughter|kids?|children|grandma|grandpa|grandmother|grandfather|cousin|aunt|uncle|best friend)\b", _MEDIUM),
        (r"\bi have (?:a |an |one |two |three |\d+ )?(?:kids?|children|sons?|daughters?)\b", _MEDIUM),
    ],
    ClaimType.PREFERENCE_HABIT: [
        (r"\bi (?:like|love|enjoy|prefer|adore)\b", _STRONG),
        (r"\bi (?:hate|dislike|can'?t stand|don'?t (?:like|enjoy))\b", _STRONG),
        (r"\bmy favou?rite\b", _STRONG),
        (r"\bi (?:usually|often|always|regularly|tend to)\b", _MEDIUM),
        (r"\bi(?:'m| am) (?:a fan of|really into)\b", _MEDIUM),
    ],
    ClaimType.CURRENT_STATE: [
        (r"\bi(?:'m| am) (?:swamped|slammed|overwhelmed|stuck|behind|in over my head)\b", _STRONG),
        (r"\bright now i(?:'m| am)\b", _MEDIUM),
        (r"\b(?:currently|at the moment) i(?:'m| am)\b", _MEDIUM),
        (r"\bthese days i(?:'m| am)\b", _WEAK),
    ],
    ClaimType.HEALTH_STATE: [
        (r"\bi(?:'m| am) (?:sick|ill|tired|exhausted|achy|nauseous|pregnant|nursing|burnt out|burned out)\b", _STRONG),
        (r"\bi have (?:a |an )?(?:cold|cough|fever|flu|headache|migraine|injury|hashimoto'?s|diabetes|cancer|asthma|the flu)\b", _STRONG),
        (r"\bi(?:'ve| have) been (?:diagnosed|recovering|feeling unwell)\b", _STRONG),
        (r"\bmy (?:back|head|knee|stomach|shoulder) (?:hurts|aches|is killing me)\b", _MEDIUM),
    ],
    ClaimType.EMOTION_STATE: [
        (r"\bi feel (?:so |really |very )?[a-z]+\b", _STRONG),
        (r"\bi(?:'m| am) (?:so |really |very |feeling )?(?:happy|sad|excited|thrilled|frustrated|anxious|nervous|proud|angry|upset|grateful|thankful|scared|worried|glad|stressed|annoyed)\b", _STRONG),
        # Colloquial frustration framing — the claim is an emotion even when its
        # OBJECT is a person ("negative sentiment alone is not contradiction").
        (r"\bdriving me (?:crazy|nuts|insane|up the wall|mad)\b", _STRONG),
        (r"\b(?:annoyed|frustrated|fed up|furious|irritated|upset|mad|sick) (?:with|at|by|about|of)\b", _STRONG),
        (r"\bi was (?:so |really |very )?(?:happy|sad|excited|thrilled|frustrated|anxious|proud|scared|grateful)\b", _MEDIUM),
        (r"\bmakes me feel\b", _MEDIUM),
        (r"\b(?:ugh|argh|ngl)\b", _WEAK),
    ],
    ClaimType.HISTORICAL_EVENT: [
        (r"\bi (?:went|attended|visited|launched|graduated|completed|finished|moved|travel?led|hosted|threw)\b", _STRONG),
        (r"\b(?:yesterday|last (?:week|month|year|night)|\d+ (?:days?|weeks?|months?|years?) ago)\b", _MEDIUM),
        (r"\bi (?:did|saw|made|ran|took|met|had|got) [a-z]+.{0,30}\b(?:yesterday|ago|last (?:week|month|year))\b", _MEDIUM),
    ],
    ClaimType.PROJECT_STATUS_PLAN: [
        (r"\bi(?:'m| am) (?:researching|working on|building|developing|organi[sz]ing|preparing|putting together)\b", _STRONG),
        (r"\bi(?:'m| am) in the (?:middle|process) of\b", _STRONG),
        (r"\bmy plan is to\b", _STRONG),
        (r"\bi started (?:a |an |my )?[a-z]+", _MEDIUM),
        (r"\b(?:i've|i have)? ?started [a-z]+ing\b", _MEDIUM),
        (r"\bi(?:'m| am) (?:trying|looking) to\b", _WEAK),
    ],
    ClaimType.SCHEDULE: [
        (r"\bi(?:'m| am) (?:going|planning) to .{0,40}(?:next (?:week|month)|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tomorrow|this weekend|in (?:january|february|march|april|may|june|july|august|september|october|november|december))\b", _STRONG),
        (r"\bi have .{0,30}(?:next week|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|tomorrow|this weekend)\b", _MEDIUM),
        (r"\b(?:next week|next month|tomorrow|this weekend) i(?:'m| am|'ll| will)\b", _MEDIUM),
    ],
    ClaimType.BELIEF_VALUE: [
        (r"\bi believe\b", _STRONG),
        (r"\bi value\b", _STRONG),
        (r"\bi think .{0,40}(?:is important|matters|is wrong|is right|is key)\b", _STRONG),
        (r"\b(?:politically|religiously) i\b", _STRONG),
        (r"\bi(?:'m| am) (?:liberal|conservative|progressive|religious|spiritual|atheist|christian|catholic|muslim|jewish|buddhist|hindu)\b", _STRONG),
        (r"\b(?:is|are) (?:so |really )?important to me\b", _MEDIUM),
        (r"\bi prioriti[sz]e\b", _MEDIUM),
        (r"\bmatters? (?:a lot )?to me\b", _MEDIUM),
    ],
    ClaimType.ASPIRATION_GOAL: [
        (r"\bi want to (?:be|become)\b", _STRONG),
        (r"\bi want to [a-z]+\b", _MEDIUM),  # "want to <anything>" is desire/goal-shaped
        # Desire for a life-STATE, not an action: "I want a quiet life",
        # "I long for financial freedom". Scoped to state nouns so "I want
        # a sandwich" stays out — an appetite is not an aspiration. (B1
        # eval's classifier-blocked pair: verb-directed wants matched,
        # noun-phrase state wants fell to unclassified, never reached
        # the tension gate.)
        # STRONG is earned by the noun scoping (precision lives in the state-
        # noun list), unlike the want-to-verb catch-all above where "I want
        # to eat lunch" forces MEDIUM conservatism.
        (r"\bi (?:want|long for|wish for|dream of) (?:a|an|the)?\s*"
         r"(?:\w+\s+){0,3}?(?:life|lifestyle|future|career|family|home|"
         r"retirement|marriage|freedom|stability|independence)\b", _STRONG),
        (r"\bi (?:hope|dream|aspire|aim) to\b", _STRONG),
        (r"\bmy (?:goal|dream|ambition) is\b", _STRONG),
        (r"\bsomeday i(?:'d| would| want| hope)\b", _STRONG),
        (r"\bi(?:'d| would) love to (?:be|become|have|own|open)\b", _MEDIUM),
        (r"\bi(?:'m| am) keen on\b", _WEAK),
    ],
    ClaimType.SEMANTIC_FACT: [
        (r"\b(?:symboli[sz]es?|represents?|signifies|stands for)\b", _STRONG),
        (r"\b(?:it|this|that|the \w+) (?:symboli[sz]es|represents|signifies|means)\b", _STRONG),
        (r"\bthe \w+ (?:is|was|supports?|contains?|includes?|offers?|provides?)\b", _MEDIUM),
    ],
}

# ── Durability signals (independent axis) ─────────────────────────────────────
_DUR_ONETIME = re.compile(
    r"\b(?:yesterday|last (?:week|month|year|night)|\d+ (?:days?|weeks?|months?|years?) ago"
    r"|i (?:went|attended|visited|launched|graduated|completed|finished|hosted))\b", re.I)
_DUR_FAST = re.compile(
    r"\b(?:right now|currently|at the moment|today|this week|lately|these days"
    r"|swamped|slammed|busy|overwhelmed|single|dating|nursing|pregnant|sick|tired|exhausted|achy"
    r"|happy|sad|excited|thrilled|frustrated|anxious|nervous|proud|angry|upset|stressed|annoyed"
    r"|researching|working on|planning to|next week|tomorrow|this weekend)\b", re.I)
_DUR_STABLE = re.compile(
    r"\b(?:born|always been|i(?:'m| am) (?:a |an )?(?:transgender|trans|nonbinary|gay|lesbian|swedish|german|american|british|canadian|french)"
    r"|originally from|symboli[sz]es|represents|stands for)\b", re.I)
_DUR_SLOW = re.compile(
    r"\b(?:usually|often|always|regularly|i (?:like|love|enjoy|prefer|believe|value)"
    r"|want to (?:be|become)|hope to|aspire|my (?:goal|dream)|prioriti[sz]e|favou?rite)\b", re.I)
# Chronic standing conditions are slow_change, not the health fast-prior (Fix 3).
_DUR_CHRONIC = re.compile(
    r"\b(?:hashimoto'?s|diabetes|cancer|asthma|arthritis|hypertension|lupus|crohn'?s"
    r"|epilepsy|fibromyalgia|chronic|thyroid)\b", re.I)

# Clause connectors — compound turns join distinct claims with these (Fix 2).
_CLAUSE_SPLIT = re.compile(r",\s*(?:and|but)\s+|;\s*|\.\s+", re.I)


class ClaimClassifier:
    """Rule-based claim classifier. Stateless; compile patterns once."""

    def __init__(self) -> None:
        self._compiled: Dict[ClaimType, List[Tuple[re.Pattern, float]]] = {
            ct: [(re.compile(p, re.I), w) for p, w in pats]
            for ct, pats in _TYPE_PATTERNS.items()
        }

    def _type_scores(self, text: str) -> Dict[ClaimType, float]:
        scores: Dict[ClaimType, float] = {}
        for ct, pats in self._compiled.items():
            s = 0.0
            for rx, w in pats:
                if rx.search(text):
                    s += w
            if s > 0:
                scores[ct] = min(1.0, s)
        return scores

    def _detect_compound(self, text: str) -> bool:
        """Compound = distinct claims joined in one turn (clause-split, Fix 2).

        Split on clause connectors; if two clauses carry DIFFERENT top claim types
        (each clearing MEDIUM), the turn is compound. More precise than "two strong
        signals anywhere", which conflated single boundary-ambiguous claims with
        genuine multi-claim turns.
        """
        parts = [p for p in _CLAUSE_SPLIT.split(text) if p and p.strip()]
        if len(parts) < 2:
            return False
        tops = []
        for p in parts:
            sc = self._type_scores(p)
            if sc:
                t, s = max(sc.items(), key=lambda kv: kv[1])
                if s >= _MEDIUM:
                    tops.append(t)
        return len(set(tops)) >= 2

    def _durability(self, text: str, claim_type) -> Tuple[Durability, float]:
        """Classify durability independently; type prior only as a fallback (§7.1)."""
        # A named chronic condition on a health claim is a STANDING state —
        # slow_change, overriding health_state's fast prior (Fix 3).
        if claim_type is ClaimType.HEALTH_STATE and _DUR_CHRONIC.search(text):
            return Durability.SLOW_CHANGE, 0.70
        onetime = _DUR_ONETIME.search(text)
        fast = _DUR_FAST.search(text)
        stable = _DUR_STABLE.search(text)
        slow = _DUR_SLOW.search(text)
        # Explicit signal priority. A completed past event is one_time even if a
        # transient word also appears; otherwise transient > stable > slow.
        if onetime and not fast:
            return Durability.ONE_TIME, 0.80
        if fast:
            return Durability.FAST_CHANGE, 0.80
        if onetime:
            return Durability.ONE_TIME, 0.75
        if stable:
            return Durability.STABLE, 0.80
        if slow:
            return Durability.SLOW_CHANGE, 0.70
        # No explicit signal: fall back to the type prior (a PRIOR, lower conf).
        if claim_type is not None:
            return DURABILITY_PRIOR[claim_type], 0.50
        return Durability.UNKNOWN, 0.30

    def classify(self, text: str) -> ClassificationResult:
        text = (text or "").strip()
        scores = self._type_scores(text)
        notes: List[str] = []

        if not scores:
            # Nothing matched — honestly unclassified, durability unknown.
            return ClassificationResult(
                claim_type=None, claim_type_confidence=0.0,
                classification_status=ClassificationStatus.UNCLASSIFIED,
                durability=Durability.UNKNOWN, durability_confidence=0.30,
                boundary_notes=["no_signal"],
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_type, top_score = ranked[0]
        second_type, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        # Margin-based confidence: a clean win keeps the top score; a near-tie is
        # penalized toward the floor — the ambiguity detector the contract wants.
        margin = (top_score - second_score) / top_score if top_score > 0 else 1.0
        confidence = top_score * (0.5 + 0.5 * margin)

        # Compound: distinct claims joined in one turn (clause-split, Fix 2).
        compound = self._detect_compound(text)
        if compound:
            notes.append("compound")

        # Boundary annotation: the close pair is a known §8 seam.
        if second_type is not None and second_score >= _MEDIUM:
            pair = frozenset({top_type, second_type})
            if pair in _BOUNDARY_PAIRS:
                notes.append(f"boundary:{top_type.value}|{second_type.value}")

        # Conservative banding.
        if confidence >= _CLASSIFY_HI and not compound:
            status = ClassificationStatus.CLASSIFIED
            claim_type = top_type
        elif confidence >= _LOW_FLOOR:
            status = ClassificationStatus.LOW_CONFIDENCE
            claim_type = top_type  # retained as a hint, but not confidently classified
        else:
            status = ClassificationStatus.UNCLASSIFIED
            claim_type = None

        durability, dur_conf = self._durability(
            text, claim_type if claim_type is not None else top_type)
        # If we could not confidently classify the TYPE and there is no explicit
        # durability signal, durability is unknown rather than a prior-from-a-guess.
        if status is ClassificationStatus.UNCLASSIFIED and dur_conf <= 0.50:
            durability, dur_conf = Durability.UNKNOWN, 0.30

        return ClassificationResult(
            claim_type=claim_type,
            claim_type_confidence=confidence,
            classification_status=status,
            durability=durability,
            durability_confidence=dur_conf,
            boundary_notes=notes,
            auto_supersession_allowed=False,  # invariant — only Leg 3 may grant
            compound=compound,
        )
