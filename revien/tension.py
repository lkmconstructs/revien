"""
Revien Tension Recognizer — B1: is this pair a genuine COEXISTING tension?

WHY: the rule gate detects contradictions it can SEE — value flips, polarity
retractions. The identity-memory case it cannot see is two AFFIRMATIVE claims
pulling in opposite directions with zero shared vocabulary: "I want closeness"
+ "I want space". Both are true of the same person at once; neither should
supersede or queue the other. That is semantic opposition — recognizer
territory, same verdict as CSL Leg B's cross-phrasing residue.

This mirrors the sensitivity recognizer exactly (and SUBCLASSES it for the
transport): opt-in via REVIEN_TENSION_BACKEND, ollama-local default, one-time
cloud disclosure, never raises, unavailable -> abstain. Conservative mapping:
ONLY a clean TENSION verdict draws the coexist edge — UNSURE/COMPATIBLE/
unavailable all fall through to the gate's unchanged NO_CONFLICT path. A
missed tension costs a missing edge (recoverable); a false tension edge is
graph noise that the false-merge-style audit would have to mop.

CLAIM_TAXONOMY §7.2 guardrails are IN the prompt: sentiment is not
contradiction, similarity is not contradiction, and a retraction is an
update, not a tension.

Cost profile: the gate consults this on scoped-but-lexically-compatible pairs
of tension-class claim types — broader than the sensitivity recognizer's
would-be-auto trigger. Wire a LOCAL backend (default) or accept per-pair
cloud cost knowingly.
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from revien.sensitivity_llm import LLMSensitivityRecognizer, _BACKEND_DEFAULTS


class TensionRoute(str, Enum):
    TENSION = "tension"          # opposing pulls, both live -> coexist
    COMPATIBLE = "compatible"    # no opposition -> unchanged no-conflict
    ABSTAIN = "abstain"          # unsure/unavailable -> unchanged no-conflict


@dataclass
class TensionVerdict:
    route: TensionRoute
    available: bool = True

    @property
    def is_tension(self) -> bool:
        return self.route is TensionRoute.TENSION


TENSION_PROMPT = """You judge whether two statements by the SAME person express a genuine standing TENSION: two pulls that can BOTH be true of them at once and should BOTH be remembered (for example "I want closeness" and "I want space"; "I love my independence" and "I hate being alone").

Answer TENSION only when BOTH statements are affirmative standing positions, desires, values, or feelings, AND they pull in opposing directions while both remaining live.

NOT a tension — answer COMPATIBLE:
- A retraction or update: "I don't like X anymore" after "I like X" is a correction that replaces the old claim, not a tension.
- Mere negative sentiment: "I'm frustrated with X today" against a standing view of X. Emotion about a thing is not opposition to it.
- Similar, overlapping, or restated positions: semantic similarity is not opposition.
- Statements about different subjects or clearly different topics.

If you cannot confidently tell, answer UNSURE.

The two statements follow, separated by a line containing only "---".

Answer with EXACTLY ONE WORD: TENSION, COMPATIBLE, or UNSURE."""

_TENSION_VERDICT_RE = re.compile(r"\b(TENSION|COMPATIBLE|UNSURE)\b", re.I)


class LLMTensionRecognizer(LLMSensitivityRecognizer):
    """Pair-level tension recognizer riding the sensitivity transport.

    recognize_pair(a, b) -> TensionVerdict. The inherited single-text
    recognize() is NOT meaningful here — use recognize_pair.
    """

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None):
        backend = (backend or os.environ.get(
            "REVIEN_TENSION_BACKEND", "ollama")).lower().strip()
        if backend not in _BACKEND_DEFAULTS:
            backend = "ollama"
        super().__init__(backend=backend, model=None)
        # Model override chain: explicit arg > REVIEN_TENSION_MODEL > backend default.
        self.model = model or os.environ.get("REVIEN_TENSION_MODEL", self.model)
        self.system_prompt = TENSION_PROMPT
        self.verdict_re = _TENSION_VERDICT_RE
        self.disclosure_purpose = "judge whether two claims are in tension"
        self.backend_env = "REVIEN_TENSION_BACKEND"

    def recognize_pair(self, existing_text: str, new_text: str) -> TensionVerdict:
        a = (existing_text or "").strip()
        b = (new_text or "").strip()
        if not a or not b:
            return TensionVerdict(TensionRoute.ABSTAIN, available=self.is_available())
        if not self.is_available():
            return TensionVerdict(TensionRoute.ABSTAIN, available=False)
        try:
            word = self._classify(f"{a}\n---\n{b}")
        except Exception:  # noqa: BLE001 - any failure -> abstain, mark broken
            self._broken = True
            return TensionVerdict(TensionRoute.ABSTAIN, available=False)
        if word == "TENSION":
            return TensionVerdict(TensionRoute.TENSION)
        if word == "COMPATIBLE":
            return TensionVerdict(TensionRoute.COMPATIBLE)
        return TensionVerdict(TensionRoute.ABSTAIN)
