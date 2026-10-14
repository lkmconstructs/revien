"""
Evidence that an answer drew on a recalled memory (N1).

Two independent signals, kept apart so the judgment rule can demand that both
agree:

* shared terms — distinctive words of the memory that reappear in the
  answer. Words the human prompt or the recall query already contained are
  removed first: if the user asked about "Postgres", an answer that says
  "Postgres" proves nothing about the memory.
* similarity — the best cosine between the memory and any paragraph of the
  answer, from the configured embedder (local by default). None when the
  semantic layer is off, so the rule can tell "no evidence" from "weak
  evidence".
"""

from __future__ import annotations

import math
import re
from typing import Callable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from revien.ingestion.extractor import STOP_WORDS
from revien.retrieval.bm25 import tokenize
from revien.semantic.index import SemanticIndex

from .episodes import RecallEpisode, RecalledMemory

EmbedFn = Callable[[Sequence[str]], Optional[List[List[float]]]]

# Paragraph cap per answer: bounds embedding cost on very long answers.
MAX_ANSWER_PARAGRAPHS = 32
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class UseEvidence(BaseModel):
    """What one answer shares with one memory."""

    # Distinctive memory words that reappear in the answer (prompt and query
    # words already removed), sorted.
    shared_terms: List[str] = Field(default_factory=list)
    # How many distinctive words the memory had left after that removal — the
    # ceiling shared_terms could reach.
    memory_terms: int = 0
    # Best cosine between the memory and any answer paragraph; None when the
    # semantic layer is off.
    similarity: Optional[float] = None


def distinctive_terms(text: str) -> Set[str]:
    """Words that can carry evidence: not on BM25's or the extractor's
    stopword lists, and at least 4 characters or containing a digit
    (versions, ports, ids)."""
    return {
        token for token in tokenize(text)
        if token not in STOP_WORDS
        and (len(token) >= 4 or any(ch.isdigit() for ch in token))
    }


def answer_paragraphs(text: str) -> List[str]:
    """The answer split on blank lines — the unit similarity is scored on, so
    one paragraph that leans on a memory is not diluted by the rest."""
    parts = [p.strip() for p in _PARAGRAPH_BREAK.split(text or "") if p.strip()]
    return parts[:MAX_ANSWER_PARAGRAPHS]


def memory_text(memory: RecalledMemory) -> str:
    """The memory as the semantic index embeds a node, so the two compare."""
    return SemanticIndex._node_text(memory.label, memory.content)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def gather_evidence(
    episode: RecallEpisode, embed: Optional[EmbedFn] = None
) -> List[Tuple[RecalledMemory, UseEvidence]]:
    """Evidence for every memory in one episode, in order. The answer's
    paragraphs and all the memories go to the embedder in one call."""
    excluded = distinctive_terms(episode.query) | distinctive_terms(episode.prompt)
    answer_terms = distinctive_terms(episode.response)
    paragraphs = answer_paragraphs(episode.response)
    texts = [memory_text(m) for m in episode.memories]

    vectors = None
    if embed is not None and paragraphs and texts:
        vectors = embed(paragraphs + texts)
        if vectors is None or len(vectors) != len(paragraphs) + len(texts):
            vectors = None
    paragraph_vectors = vectors[: len(paragraphs)] if vectors else []

    gathered = []
    for i, memory in enumerate(episode.memories):
        terms = distinctive_terms(texts[i]) - excluded
        similarity = None
        if vectors:
            memory_vector = vectors[len(paragraphs) + i]
            similarity = max(_cosine(memory_vector, p) for p in paragraph_vectors)
        gathered.append((memory, UseEvidence(
            shared_terms=sorted(terms & answer_terms),
            memory_terms=len(terms),
            similarity=similarity,
        )))
    return gathered
