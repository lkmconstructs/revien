"""Revien BM25 lexical lane — pure-stdlib Okapi BM25 candidate ranking.

WHY: the shipped keyword lane (``_keyword_search``) is presence-only — any
node whose label+content contains a query keyword ranks the same as any
other hit, so a document that happens to repeat a common word (a "launch"
hub with a hundred mentions) is indistinguishable from the one document
that actually holds the rare, query-specific term ("heliotrope launch
code"). BM25 scores TERM RARITY (inverse document frequency) and SATURATING
term frequency (repeats past a point stop helping, length-normalized), so
the document that is actually distinctive for the query wins instead of
the document that is merely long or generic. Production sweep on the live
graph measured recall@10 0.5814 -> 0.6395 (REVIEN_LEXICAL=bm25, REVIEN_HYBRID
=rrf) before this leg existed here — this module is the validated ranking
math from that overlay, ported near-verbatim.

This module ranks candidate node identifiers only. It owns no persistence,
mutates no graph state, and does not replace the semantic or graph-walk
layers — engine.py's ``_bm25_candidates``/``_lexical_candidates`` decide
when it runs (REVIEN_LEXICAL=bm25, default off) and how its scores compose
with everything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "hers", "him", "his", "how", "i", "in",
    "is", "it", "its", "last", "may", "me", "might", "my", "next", "no",
    "not", "of", "on", "or", "our", "she", "should", "some", "that",
    "the", "their", "them", "they", "this", "to", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
})


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; punctuation/hyphens/underscores are boundaries,
    stopwords dropped. Same shape as ``_keyword_search``'s word split, just
    regex-driven so hyphenated/punctuated identifiers still tokenize."""
    return [
        token for token in _TOKEN_RE.findall(str(text).lower())
        if token and token not in _STOPWORDS
    ]


def bm25_rank(
    query: str,
    documents: Sequence[Tuple[str, str]],
    *,
    top_n: Optional[int] = None,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """Rank ``(identifier, text)`` documents by Okapi BM25 score against
    ``query``.

    Zero-overlap documents are omitted (a BM25 score of 0 carries no
    query-relevance signal, so it isn't a candidate). Equal scores preserve
    INPUT ORDER — Revien's deterministic recall contract requires that a tie
    resolve the same way on every run, not by whatever order a dict or set
    happened to produce.

    Args:
        query: Natural language query.
        documents: ``(node_id, "label content")`` pairs — the same shape
            engine.py's ``_bm25_candidates`` builds from the graph.
        top_n: Cap on returned candidates. ``None`` returns everything that
            scored above zero. ``0`` returns an empty list (a legitimate,
            distinct request from "no cap").
        k1: Term-frequency saturation. Higher lets repeated terms keep
            adding score longer before flattening.
        b: Length normalization strength (0 = ignore document length,
            1 = fully normalize by it).

    Returns:
        ``[(node_id, score), ...]`` sorted by descending score, ties broken
        by input position.
    """
    if not isinstance(k1, (int, float)) or isinstance(k1, bool):
        raise TypeError("k1 must be a finite positive number")
    k1 = float(k1)
    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 must be a finite positive number")
    if not isinstance(b, (int, float)) or isinstance(b, bool):
        raise TypeError("b must be a finite number between 0 and 1")
    b = float(b)
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("b must be a finite number between 0 and 1")
    if top_n is not None:
        if not isinstance(top_n, int) or isinstance(top_n, bool):
            raise TypeError("top_n must be a nonnegative integer or None")
        if top_n < 0:
            raise ValueError("top_n must be a nonnegative integer or None")
        if top_n == 0:
            return []

    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not documents:
        return []

    tokenized = [tokenize(text) for _, text in documents]
    document_count = len(tokenized)
    average_length = sum(len(tokens) for tokens in tokenized) / document_count
    if average_length == 0:
        return []

    document_frequency: Counter = Counter()
    query_term_set = set(query_terms)
    for tokens in tokenized:
        document_frequency.update(query_term_set.intersection(tokens))

    scored = []
    for position, ((identifier, _), tokens) in enumerate(zip(documents, tokenized)):
        frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * document_length / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0) / denominator
            )
        if score > 0:
            scored.append((identifier, score, position))

    scored.sort(key=lambda row: (-row[1], row[2]))
    ranked = [(identifier, score) for identifier, score, _ in scored]
    return ranked if top_n is None else ranked[:top_n]
