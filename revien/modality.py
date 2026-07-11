"""Claim Sovereignty Layer — Leg 1: modality-aware miss classification.

A retrieval miss has two very different causes, and conflating them hides bugs:

  * retrieval_failure  — the answer IS available to the system in text the
                         retriever could have surfaced, but ranking/recall didn't.
                         This is a retrieval-quality problem worth fixing.
  * unavailable_modality — the answer lives ONLY in a medium the system never
                         read (e.g. a shared photo with no vision pass). No amount
                         of better ranking recovers it; it is honestly out of
                         scope for a text-only system, not a retrieval defect.

These helpers read the Leg 1 node flags (`answerable_by_text`, `vision_processed`)
to draw that line. They are pure: no store access, no I/O.
"""

from __future__ import annotations

from typing import Iterable

from revien.graph.schema import Node

# Classification labels (stable strings — callers branch/report on them).
RETRIEVAL_FAILURE = "retrieval_failure"
UNAVAILABLE_MODALITY = "unavailable_modality"


def answer_available_in_text(node: Node) -> bool:
    """True iff this node's answer content is present in text the system indexed.

    Either the evidence was answerable by text to begin with, or a vision pass has
    since read any attached non-text medium into text. Both make the content
    reachable by a text retriever.
    """
    return bool(node.answerable_by_text or node.vision_processed)


def classify_miss(gold_nodes: Iterable[Node]) -> str:
    """Classify a retrieval MISS given the gold-evidence nodes that were sought.

    Returns ``unavailable_modality`` when NONE of the gold nodes carry answer
    content the system can see in text (every one is an unread non-text medium) —
    the miss was structurally unavoidable for a text-only system. Otherwise the
    answer was reachable and the miss is a genuine ``retrieval_failure``.

    With no gold nodes (nothing to locate) we cannot blame modality, so the
    conservative label is ``retrieval_failure``.
    """
    gold_nodes = list(gold_nodes)
    if gold_nodes and not any(answer_available_in_text(n) for n in gold_nodes):
        return UNAVAILABLE_MODALITY
    return RETRIEVAL_FAILURE
