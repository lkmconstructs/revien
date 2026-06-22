"""Claim Sovereignty Layer — interim distrust tripwire (Leg 3 safety net).

A "distrust trigger" (Ash's framing): when a claim's raw/normalized CONTENT names
a sensitive domain, the system DISTRUSTS its own classifier's type and routes the
claim to human review — even though the type came out non-protected. This covers
the confidently-misnamed manifestation of the sensitive-recognition gap ("I love
being sober" -> preference_habit -> would otherwise auto-supersede and silently
erase the disclosure). The stakes: a trust product auto-erasing a recovery or
transition disclosure is betrayal with a stack trace — worse than any number of
extra reviews. The tripwire is the interim promise it won't, while the real fix
(semantic recognition, HYBRID_BACKEND_TRIGGERS.md Trigger 2) is built.

================================ SIX INVARIANTS ================================
(committed verbatim — the tripwire's contract)

  1. Strictly additive. The tripwire ONLY routes to candidate; it never grants
     auto-supersession, never removes protection, never alters a claim's type,
     durability, or content. It can only make a decision more conservative.
  2. Candidate-only. A tripwire match routes to human review (candidate). It never
     auto-acts and never resolves a conflict on its own.
  3. Type-independent. The tripwire inspects the raw AND normalized claim CONTENT,
     independent of claim_type. It does not trust the classifier's type — that
     distrust is the point; it covers the confidently-misnamed manifestation the
     type-keyed protected guard misses.
  4. Core domain is a config-floor. Operators may ADD domains and lexemes; they may
     NEVER remove the reproduced-harm core set. The learning loop may never modify
     it. No config, attribute, or subclass can shrink the core.
  5. The tripwire does not close Trigger 2. It is lexemes, not meaning; it WILL
     miss sensitive content phrased without a core lexeme. It structurally cannot
     satisfy the sensitive-recognition test and never retires it. It may be demoted
     only after the semantic backend's MEASURED performance proves it safe.
  6. False positives are humility, not failure. The tripwire is deliberately
     over-broad. Over-catching routes extra claims to human review — safe — and
     that bias is intended. Narrowing it to reduce false positives is a defect, not
     a fix: betrayal with a stack trace costs more than an extra review.
================================================================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ── Reproduced-harm CORE domains (the config-floor — add-only, never removable) ──
# Deliberately broad (invariant 6). Each entry is a regex fragment matched with a
# leading word boundary against lowercased + normalized content. Stems are used on
# purpose so inflections catch (recover -> recovery/recovering). Known, intended
# false positives (e.g. "recovery from the flu", "single-minded") are PROOF OF
# HUMILITY, not bugs — see tests/test_tripwire.py.
CORE_DOMAINS: Dict[str, List[str]] = {
    "recovery": [
        r"sober", r"sobriet", r"recover", r"relaps", r"addict", r"rehab",
        r"in recovery", r"twelve.?step", r"12.?step", r"clean and sober",
    ],
    "identity_transition": [
        r"transition", r"transgender", r"nonbinary", r"non-binary", r"deadnam",
        r"pronoun", r"genderqueer", r"gender identity", r"coming out",
    ],
    "relationship_status": [
        r"single", r"married", r"divorc", r"widow", r"engaged", r"separated",
        r"my husband", r"my wife", r"my partner", r"my spouse",
    ],
    "health_diagnosis": [
        r"diagnos", r"cancer", r"hashimoto", r"diabet", r"depress", r"anxiety",
        r"bipolar", r"\bhiv\b", r"disorder", r"chronic", r"illness", r"disease",
        r"nursing", r"pregnan", r"miscarriag",
    ],
    "faith_religion": [
        r"religio", r"\bfaith\b", r"christian", r"muslim", r"islam", r"jewish",
        r"buddhis", r"atheis", r"\bgod\b", r"church", r"mosque", r"synagogue",
        r"\bpray", r"spiritual",
    ],
    "political_belief": [
        r"liberal", r"conservativ", r"progressive", r"politic", r"democrat",
        r"republican", r"\bvote", r"left.?wing", r"right.?wing",
    ],
}


def _normalize(text: str) -> str:
    """Punctuation-light, lowercased form so 's.o.b.e.r' / 'sober!' still match."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


class DistrustTripwire:
    """Content-level sensitive-domain screen. Returns the tripped domain or None.

    The CORE domains are a config-floor: ``extra_domains`` are MERGED IN (union);
    nothing can remove a core lexeme. There is deliberately no removal API.
    """

    def __init__(self, extra_domains: Optional[Dict[str, List[str]]] = None):
        # Start from a copy of CORE, then UNION any operator additions. Union, never
        # replace — so a passed-in domain that overlaps a core domain can only ADD
        # lexemes, never shrink it (invariant 4).
        domains: Dict[str, List[str]] = {d: list(lex) for d, lex in CORE_DOMAINS.items()}
        if extra_domains:
            for d, lex in extra_domains.items():
                domains[d] = sorted(set(domains.get(d, [])) | set(lex))
        self._domains = domains
        self._compiled = {
            d: re.compile(r"\b(?:" + "|".join(lex) + r")", re.I)
            for d, lex in domains.items()
        }
        # Hard guarantee: every core lexeme is still present (config-floor proof).
        for d, lex in CORE_DOMAINS.items():
            assert set(lex) <= set(self._domains[d]), f"core domain {d} was shrunk"

    def check(self, text: str) -> Optional[str]:
        """Return the first sensitive domain the content trips, or None.

        Inspects BOTH the raw lowercased text and the normalized form (invariant 3).
        """
        if not text:
            return None
        forms = (text.lower(), _normalize(text))
        for domain, rx in self._compiled.items():
            if any(rx.search(f) for f in forms):
                return domain
        return None

    def covers_core(self) -> bool:
        """True iff the effective lexicon still contains the entire core set."""
        return all(set(lex) <= set(self._domains[d]) for d, lex in CORE_DOMAINS.items())
