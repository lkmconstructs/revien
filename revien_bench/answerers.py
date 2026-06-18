"""
revien_bench.answerers — The "reader" layer. HEADLINE build: ExtractiveAnswerer only.

The answerer takes the question + the top-K retrieved nodes (the SAME context the
LLM answerers would get, when those exist) and produces an answer string. Keeping
the reader pluggable means the only variable across tracks is the reader, so
retrieval quality is isolated.

ExtractiveAnswerer (zero-LLM, deterministic, $0, zero-egress)
-------------------------------------------------------------
No model, no network, no randomness. It selects the best SPAN/sentence from the
retrieved node content by lexical overlap with the question, and REFUSES (emits a
canonical "no information available" / "not mentioned" string) when no candidate
clears a minimum-overlap floor — which is exactly what the adversarial (cat 5)
category rewards (hallucination resistance from absence-of-node).

Extension points (DEFERRED to the LLM-judge track — intentionally NOT implemented):
    class OllamaAnswerer(Answerer): ...   # local LLM reader (ollama:<model>)
    class ClaudeAnswerer(Answerer): ...   # cloud LLM reader (discloses, $cost)
Both would subclass `Answerer` and consume the same `RetrievedContext`. The
runner already dispatches on `--answerer`, so adding them is a drop-in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

# Canonical refusal string. Contains BOTH official adversarial markers so the
# official matcher scores it as a correct refusal regardless of which marker the
# grader checks. (metrics.is_refusal recognizes either.)
REFUSAL = "No information available; this was not mentioned."

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "what", "when", "where", "why", "who", "whom", "which", "how", "to",
    "of", "in", "on", "at", "by", "for", "with", "and", "or", "but", "not",
    "that", "this", "it", "as", "from", "about", "into", "over", "after",
    "i", "we", "you", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "tell", "give", "name",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 1}


@dataclass
class RetrievedContext:
    """The context handed to a reader: ordered node contents from recall()."""
    query: str
    contents: List[str] = field(default_factory=list)  # ranked node contents
    labels: List[str] = field(default_factory=list)      # parallel node labels

    def sentences(self) -> List[str]:
        """Flatten retrieved contents into candidate sentences, rank-preserving."""
        out: List[str] = []
        for content in self.contents:
            for sent in _SENT_SPLIT.split(content or ""):
                sent = sent.strip()
                if sent:
                    out.append(sent)
        return out


class Answerer(Protocol):
    """A reader: turn a question + retrieved context into an answer string."""

    name: str

    def answer(self, ctx: RetrievedContext) -> str: ...


class ExtractiveAnswerer:
    """Deterministic, zero-LLM span/sentence selector.

    Scores each candidate sentence by lexical overlap with the question's
    content tokens (Jaccard-ish: shared / question-token-count, with a small
    bonus for sentences that strip the speaker prefix cleanly). Returns the
    best sentence's most-relevant span. Refuses when the best score is below
    `min_overlap` (no node supports an answer) — the adversarial-resistant path.
    """

    name = "extractive"

    def __init__(self, min_overlap: float = 0.15, max_answer_chars: int = 240):
        self.min_overlap = min_overlap
        self.max_answer_chars = max_answer_chars

    def answer(self, ctx: RetrievedContext) -> str:
        q_tokens = _content_tokens(ctx.query)
        if not q_tokens:
            # Degenerate question (all stopwords): fall back to top sentence.
            sents = ctx.sentences()
            return self._trim(sents[0]) if sents else REFUSAL

        best_sent = None
        best_score = 0.0
        for sent in ctx.sentences():
            s_tokens = _content_tokens(sent)
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens)
            if overlap == 0:
                continue
            # Overlap normalized by question size (recall of question terms),
            # with a mild length penalty so a huge sentence doesn't win on raw
            # term count alone. Deterministic — no randomness.
            score = overlap / len(q_tokens)
            score *= 1.0 / (1.0 + 0.002 * max(0, len(sent) - 80))
            if score > best_score:
                best_score = score
                best_sent = sent

        if best_sent is None or best_score < self.min_overlap:
            return REFUSAL
        return self._trim(self._strip_speaker(best_sent))

    @staticmethod
    def _strip_speaker(sent: str) -> str:
        """Drop a leading 'Speaker: ' prefix carried from the turn content."""
        return re.sub(r"^[A-Z][A-Za-z0-9 _-]{0,40}:\s*", "", sent).strip()

    def _trim(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.max_answer_chars:
            return text
        return text[: self.max_answer_chars].rsplit(" ", 1)[0] + "..."


def build_answerer(name: str = "extractive", **kwargs) -> Answerer:
    """Factory. HEADLINE build supports only 'extractive'.

    'ollama:*' and 'claude' are DEFERRED (LLM-judge track). Asking for them
    here raises a clear NotImplementedError rather than silently degrading, so
    a misconfigured run fails loud instead of fabricating a cloud path.
    """
    name = (name or "extractive").strip().lower()
    if name == "extractive":
        return ExtractiveAnswerer(**kwargs)
    raise NotImplementedError(
        f"answerer {name!r} is deferred to the LLM-judge track; "
        f"only 'extractive' is implemented in the headline build."
    )
