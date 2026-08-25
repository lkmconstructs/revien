"""
Revien Deduplication — Prevents duplicate nodes from polluting the graph.
Layers: exact label match -> fuzzy match (Levenshtein distance + ratio) ->
OPT-IN semantic match (cosine over the embeddings already in vec_nodes).
"""

import os
import re
from typing import List, Optional, Tuple, TYPE_CHECKING

from revien.graph.schema import Edge, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations

if TYPE_CHECKING:  # import for typing only — dedup must not require the layer
    from revien.semantic.index import SemanticIndex


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Cheap negation-marker detector for the contradiction guard. Catches the
# explicit markers ("doesn't like X" vs "likes X"); affect antonyms
# ("loves"/"hates") are out of scope here — those score far enough apart in
# embedding space or land in supersession's lap, where contradictions belong.
_NEGATION_RE = re.compile(
    r"\b(not|no|never|none|without|nothing|neither|nor)\b|n[o']t\b",
    re.IGNORECASE,
)


def _has_negation(*texts: Optional[str]) -> bool:
    return any(_NEGATION_RE.search(t) for t in texts if t)


class Deduplicator:
    """
    Deduplication engine for ingestion.
    Before creating a node, checks if a semantically equivalent node exists.
    If found, increments the existing node's access_count and returns it
    instead of creating a duplicate.
    """

    # Cosine floor for a semantic merge. BENCHED (per-user conversational
    # harness, 10 user graphs, LLM extractor, 2026-08):
    # 0.90 -> 14 merges / 40% generic recall, 0.85 -> 62 / 40%,
    # 0.80 -> 166 / 70%, 0.75 -> 279 / still 70% — a clean knee at 0.80
    # (below it, +68% merge volume for zero recall gain; sampled merges at
    # 0.80 were all legitimate consolidations). This default only matters to
    # callers who opted into REVIEN_SEMANTIC_DEDUP in the first place.
    # Override: REVIEN_SEMANTIC_DEDUP_THRESHOLD.
    SEMANTIC_DEDUP_THRESHOLD = 0.80
    # How many nearest neighbours to consider before giving up on a same-type
    # above-threshold match.
    SEMANTIC_DEDUP_TOP_K = 8

    def __init__(
        self,
        store: GraphStore,
        ops: GraphOperations,
        semantic: Optional["SemanticIndex"] = None,
    ):
        self.store = store
        self.ops = ops
        # Semantic dedup (paraphrase-swarm fix): OPT-IN via
        # REVIEN_SEMANTIC_DEDUP=1 — default off keeps ingestion byte-identical
        # until the A/B numbers exist. Inert without a live semantic index
        # (the embeddings this queries live in its vec_nodes table).
        self.semantic = semantic
        self.semantic_dedup_enabled = _env_flag("REVIEN_SEMANTIC_DEDUP")
        try:
            self.semantic_dedup_threshold = float(os.environ.get(
                "REVIEN_SEMANTIC_DEDUP_THRESHOLD",
                self.SEMANTIC_DEDUP_THRESHOLD,
            ))
        except ValueError:  # bad experiment knob must never crash ingest
            self.semantic_dedup_threshold = self.SEMANTIC_DEDUP_THRESHOLD
        # Contradiction guard: never merge a pair whose negation-marker
        # presence differs ("likes present tense" / "doesn't like present
        # tense" can clear 0.90 cosine — that pair belongs to supersession,
        # not merge). REVIEN_SEMANTIC_DEDUP_NEGATION_GUARD=0 disables for
        # sweeps that want to measure the guard's cost.
        self.negation_guard = _env_flag(
            "REVIEN_SEMANTIC_DEDUP_NEGATION_GUARD", default=True
        )

    def deduplicate_node(
        self, candidate: Node, allow_semantic: bool = True
    ) -> Tuple[Node, bool]:
        """
        Check if a semantically equivalent node exists.

        Args:
            candidate: The node the extractor proposes to create.
            allow_semantic: Gate for the semantic layer of THIS call — the
                capture path passes False (semantic dedup means embedding the
                candidate inline, and defer_embed exists precisely so capture
                never waits on a model load). The lexical layers always run.

        Returns:
            (node, is_new): The node to use and whether it was newly created.
            If a duplicate exists, returns the existing node (with incremented
            access_count) and is_new=False.
        """
        # Context nodes are never deduplicated — each session is unique
        if candidate.node_type == NodeType.CONTEXT:
            stored = self.store.add_node(candidate)
            return stored, True

        # 1. Exact label match (NORMALIZED, same type). Precision guard: a
        # merge that happened ONLY because of normalization (raw lowercase
        # labels differ) is written to the audit log with BOTH labels — the
        # reviewable surface for false merges ("Lincoln" the city absorbing
        # "Lincoln" the person is invisible to any surface rule; the audit
        # list is where a human catches it). Benches surface these per run.
        existing = self._find_exact_match(candidate)
        if existing:
            if existing.label.lower() != candidate.label.lower():
                self.store._record_node_audit(
                    existing.node_id,
                    "normalized_merge",
                    actor=f"{candidate.label!r} -> {existing.label!r}",
                    after_node=existing,
                )
            self.ops.touch_node(existing.node_id)
            return existing, False

        # 2. Fuzzy match (Levenshtein < 3), same type
        existing = self._find_fuzzy_match(candidate)
        if existing:
            self.ops.touch_node(existing.node_id)
            return existing, False

        # 3. Semantic match (OPT-IN): paraphrases of the same claim
        # ("dislikes when pacing slows" / "felt the pacing died mid-book")
        # share no surface form the lexical layers can see, and each one that
        # slips through becomes another same-type node crowding top-N recall
        # — the paraphrase swarm. Cosine against the embeddings already in
        # vec_nodes; merge is non-destructive (candidate is simply never
        # created, survivor is reinforced) and leaves a `semantic_merge`
        # audit entry — the reviewable surface for false merges, same
        # contract as normalized_merge above.
        if allow_semantic:
            semantic_hit = self._find_semantic_match(candidate)
            if semantic_hit is not None:
                existing, cosine = semantic_hit
                self.store._record_node_audit(
                    existing.node_id,
                    "semantic_merge",
                    actor=(
                        f"{candidate.label!r} -> {existing.label!r} "
                        f"(cosine={cosine:.3f})"
                    ),
                    after_node=existing,
                )
                self.ops.touch_node(existing.node_id)
                return existing, False

        # 4. No match — create new node
        stored = self.store.add_node(candidate)
        return stored, True

    def deduplicate_nodes(
        self, candidates: List[Node]
    ) -> List[Tuple[Node, bool]]:
        """Deduplicate a batch of candidate nodes."""
        results = []
        for candidate in candidates:
            results.append(self.deduplicate_node(candidate))
        return results

    def _find_exact_match(self, candidate: Node) -> Optional[Node]:
        """Find a node with the exact same label and type."""
        return self.ops.find_node_by_label(
            candidate.label, node_type=candidate.node_type
        )

    def _find_semantic_match(
        self, candidate: Node
    ) -> Optional[Tuple[Node, float]]:
        """Nearest same-type node at or above the cosine threshold, or None.

        Constraints, in order of why they exist:
          * same node_type only — a preference never absorbs a fact, however
            close the embeddings sit (type carries claim semantics);
          * invalidated (superseded) nodes never win — a retired claim must
            not swallow its own replacement back out of existence;
          * negation-marker parity (guard, default on) — contradictions go
            to supersession, not merge.
        """
        if not self.semantic_dedup_enabled:
            return None
        sem = self.semantic
        if sem is None or not sem.is_enabled:
            return None
        # Embed the exact text form index_node stores, so the comparison is
        # like-for-like with what is already in vec_nodes.
        text = sem._node_text(candidate.label, candidate.content)
        if not text:
            return None
        candidate_negated = _has_negation(candidate.label, candidate.content)
        for node_id, cosine in sem.find_similar(
            text, top_k=self.SEMANTIC_DEDUP_TOP_K
        ):
            if cosine < self.semantic_dedup_threshold:
                break  # nearest-first: everything after is farther
            node = self.store.get_node(node_id)
            if node is None or node.node_type != candidate.node_type:
                continue
            if getattr(node, "invalidated_at", None) is not None:
                continue
            if self.negation_guard and _has_negation(
                node.label, node.content
            ) != candidate_negated:
                continue
            return node, cosine
        return None

    def _find_fuzzy_match(self, candidate: Node) -> Optional[Node]:
        """Find a similar node using Levenshtein distance and ratio matching.

        Negation-parity guard: "likes present tense" vs
        "does not like present tense" clears the 0.75 ratio floor — the edit
        distance IS the negation — and merging them silently swallows a
        contradiction that belongs to supersession. Same guard as the
        semantic layer, but UNCONDITIONAL: this fixes a known corruption
        class, not an experiment of its own (the P1 entity-anchor correction
        set the precedent — regression fixes don't get flags).
        """
        matches = self.ops.find_nodes_by_label_fuzzy(
            candidate.label, max_distance=5, min_ratio=0.75
        )

        candidate_negated = _has_negation(candidate.label, candidate.content)
        for match in matches:
            if match.node_type != candidate.node_type:
                continue
            if _has_negation(match.label, match.content) != candidate_negated:
                continue
            return match
        return None
