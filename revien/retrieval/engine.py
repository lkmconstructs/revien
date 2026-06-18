"""
Revien Retrieval Engine — Query processing, graph walking, scoring, and ranking.
The full retrieval pipeline: parse query → find anchors → walk graph → score → rank.

Scoring layers (in apply order inside recall):
  1. Three-factor base score (recency + frequency + proximity)  [base, stdlib]
  2. Neural adjustment (TF-IDF + LogisticRegression)            [opt-in extra]
  3. Community boost (same-community-as-anchor bonus)           [leg 2]
  4. Confidence multiplier (post-factor, with lazy decay)       [leg 1, base]

Final: final_score = (neural_adjusted_base + community_boost) * effective_confidence

The neural layer is OPT-IN (pip install revien[neural]). Its import is guarded:
when numpy/scikit-learn are absent, neural is silently disabled and every other
layer runs unchanged.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.graph.clustering import CommunityDetector
from revien.ingestion.extractor import RuleBasedExtractor
# Neural reranker is opt-in. NeuralScorer/TrainingLoop import cleanly without
# the `neural` extra installed (NeuralScorer self-disables when numpy/sklearn
# are missing; TrainingLoop is pure sqlite3 stdlib).
from revien.neural.scorer_model import NeuralScorer
from revien.neural.training import TrainingLoop
# Semantic/vector layer is opt-in (pip install revien[semantic]). SemanticIndex
# imports cleanly without the extra and self-disables (is_enabled False), so
# recall() runs the unchanged graph path when it is absent or REVIEN_SEMANTIC=0.
from revien.semantic.index import SemanticIndex
from .scorer import ScoreBreakdown, ScoringConfig, ThreeFactorScorer
from .walker import GraphWalker


@dataclass
class RetrievalResult:
    """A single node result with its score and path."""
    node_id: str
    node_type: str
    label: str
    content: str
    score: float
    score_breakdown: Dict[str, float]
    path: List[str]


@dataclass
class RetrievalResponse:
    """Complete response to a retrieval query."""
    query: str
    results: List[RetrievalResult]
    nodes_examined: int
    retrieval_time_ms: float
    neural_active: bool = False


class RetrievalEngine:
    """
    Full retrieval pipeline:
    1. Parse query to extract entities/topics
    2. Find anchor nodes in the graph
    2b. Community-first routing — boost nodes in query-relevant communities
    3. Walk the graph from anchors
    4. Score all reachable nodes (three-factor + neural + community + confidence)
    5. Rank and return top N
    6. Log retrieval for neural training (if neural extra installed)
    """

    # Community membership boost — nodes in the same community as anchors score higher
    COMMUNITY_BOOST = 0.15

    # Semantic blend — when the opt-in semantic layer surfaces a node, add a
    # bounded bonus proportional to how far the node's similarity exceeds a
    # floor. Mirrors COMMUNITY_BOOST in magnitude so it nudges ranking without
    # dominating the three-factor base.
    SEMANTIC_BOOST = 0.15
    # How many nearest neighbours to consider at recall time.
    SEMANTIC_TOP_K = 10
    # Similarity floor. Embedding models (e.g. bge-small) score almost every
    # node as mildly related to almost any query, which would flatten ranking
    # if every neighbour became an equal-weight anchor. We only treat a node as
    # a semantic anchor / give it a boost when its similarity clears this floor,
    # and the boost is scaled by (sim - floor) so weak matches contribute ~0.
    # This keeps keyword queries unchanged-or-better while still letting a
    # keyword-less query reach a genuinely-close node.
    SEMANTIC_SIM_FLOOR = 0.55

    def __init__(
        self,
        store: GraphStore,
        scoring_config: Optional[ScoringConfig] = None,
        max_depth: int = 3,
        clustering: Optional[CommunityDetector] = None,
        model_dir: Optional[str] = None,
        training_db: Optional[str] = None,
        semantic: Optional[SemanticIndex] = None,
    ):
        self.store = store
        self.ops = GraphOperations(store)
        self.extractor = RuleBasedExtractor()
        self.scorer = ThreeFactorScorer(scoring_config)
        self.walker = GraphWalker(store, max_depth=max_depth)
        self.max_depth = max_depth
        self.clustering = clustering

        # Neural components (opt-in). These construct cleanly even without the
        # `neural` extra: NeuralScorer.is_neural stays False and adjust_score()
        # is a pass-through, so recall() degrades to base scoring.
        self.neural_scorer = NeuralScorer(model_dir=model_dir)
        self.training_loop = TrainingLoop(db_path=training_db, model_dir=model_dir)

        # Semantic/vector layer (opt-in). Constructs cleanly even without the
        # `semantic` extra: SemanticIndex.is_enabled stays False and all of its
        # methods no-op, so recall() degrades to the exact graph-only path.
        self.semantic = semantic if semantic is not None else SemanticIndex(store)

    def recall(
        self,
        query: str,
        top_n: int = 5,
        min_score: float = 0.01,
        now: Optional[datetime] = None,
        include_invalidated: bool = False,
    ) -> RetrievalResponse:
        """
        Query the memory graph and return ranked results.

        Args:
            query: Natural language query
            top_n: Maximum results to return (default 5, max 20)
            min_score: Minimum composite score threshold
            now: Current time for recency scoring (defaults to UTC now)
            include_invalidated: When False (default), soft-invalidated nodes
                (invalidated_at set) are excluded from results. Set True to
                surface them. Provenance is non-destructive — invalidated nodes
                are retained and recoverable, just hidden from default recall.
                When no node is invalidated this flag changes nothing, so recall
                is byte-identical to the pre-6a behavior.

        Returns:
            RetrievalResponse with ranked nodes and timing data
        """
        start_time = time.perf_counter()
        top_n = min(top_n, 20)

        if now is None:
            now = datetime.now(timezone.utc)

        # 1. Parse query — extract entities and topics
        anchor_ids = self._find_anchors(query)

        # 2. If no anchors found, try keyword search across all nodes
        if not anchor_ids:
            anchor_ids = self._keyword_search(query)

        # 2a. Hybrid semantic anchors (opt-in). When the semantic layer is
        # enabled, embed the query, pull the nearest stored nodes, and UNION
        # them into the anchor set. This is what lets a keyword-less query
        # (no entity/keyword overlap with any node) still find relevant nodes.
        # When the layer is disabled, semantic_sims is empty and the anchor set
        # is byte-for-byte what it was before this leg.
        semantic_sims: Dict[str, float] = {}
        if self.semantic.is_enabled:
            for node_id, sim in self.semantic.search(query, top_k=self.SEMANTIC_TOP_K):
                # Only nodes that clear the floor act as semantic anchors. This
                # is what lets a keyword-less query reach a genuinely-close node
                # without near-uniform mild similarity reshuffling keyword hits.
                if sim < self.SEMANTIC_SIM_FLOOR:
                    continue
                semantic_sims[node_id] = sim
                if node_id not in anchor_ids:
                    anchor_ids.append(node_id)

        # 2b. Community-first routing — identify which communities are relevant
        relevant_communities: set = set()
        if anchor_ids and self.clustering and self.clustering.is_clustered:
            relevant_communities = set(
                self.clustering.get_communities_for_anchors(anchor_ids)
            )

        # 3. Walk graph from anchors
        if anchor_ids:
            node_distances = self.walker.walk(anchor_ids)
            node_paths = self.walker.walk_with_paths(anchor_ids)
        else:
            node_distances = {}
            node_paths = {}

        # 4. Score all reachable nodes
        scored_results: List[RetrievalResult] = []
        nodes_examined = len(node_distances)

        for node_id, distance in node_distances.items():
            node = self.store.get_node(node_id)
            if node is None:
                continue

            # Skip context nodes from results (they're structural, not content)
            if node.node_type == NodeType.CONTEXT:
                continue

            # Provenance (leg 6a): exclude soft-invalidated nodes by default.
            # No-op when nothing is invalidated, so recall stays byte-identical.
            if node.invalidated_at is not None and not include_invalidated:
                continue

            # Three-factor base score (recency + frequency + proximity)
            breakdown = self.scorer.score(
                last_accessed=node.last_accessed,
                access_count=node.access_count,
                graph_distance=distance,
                now=now,
            )

            # Neural adjustment (opt-in). Pass-through when no trained model /
            # neural extra absent — base_score == breakdown.composite in that case.
            base_score = self.neural_scorer.adjust_score(
                base_score=breakdown.composite,
                node_label=node.label,
                query=query,
            )

            # Community boost — nodes in same community as anchors get a boost
            community_boost = 0.0
            if relevant_communities and self.clustering:
                node_community = self.clustering.get_community(node_id)
                if node_community is not None and node_community in relevant_communities:
                    community_boost = self.COMMUNITY_BOOST

            # Semantic boost (opt-in). When the semantic layer surfaced this
            # node for the query, add a bounded similarity-weighted bonus.
            # semantic_sims is empty whenever the layer is disabled, so this
            # branch never runs and the score expression below is unchanged.
            semantic_boost = 0.0
            if semantic_sims:
                sim = semantic_sims.get(node_id)
                if sim is not None:
                    # Scale by how far similarity clears the floor, normalized so
                    # a perfect match (sim==1) yields the full SEMANTIC_BOOST and
                    # a just-at-floor match yields ~0. Floor-gated above, so sim
                    # here is always >= SEMANTIC_SIM_FLOOR.
                    span = max(1e-6, 1.0 - self.SEMANTIC_SIM_FLOOR)
                    norm = (sim - self.SEMANTIC_SIM_FLOOR) / span
                    semantic_boost = self.SEMANTIC_BOOST * norm

            # Layer 1 (leg 1): confidence multiplier as a post-factor.
            # Lazy decay for INFERRED nodes is applied here, reusing leg-1's
            # GraphOperations._apply_decay helper (no duplicated decay logic).
            effective_confidence = self._effective_confidence(node, now)
            final_score = (base_score + community_boost + semantic_boost) * effective_confidence

            if final_score < min_score:
                continue

            # Build path labels
            path_ids = node_paths.get(node_id, [node_id])
            path_labels = []
            for pid in path_ids:
                pnode = self.store.get_node(pid)
                if pnode:
                    path_labels.append(pnode.label)

            score_breakdown = {
                "recency": breakdown.recency,
                "frequency": breakdown.frequency,
                "proximity": breakdown.proximity,
                "base_composite": breakdown.composite,
                "community_boost": community_boost,
                "effective_confidence": effective_confidence,
                "neural_adjusted": self.neural_scorer.is_neural,
            }
            # Only surface the semantic component when the opt-in layer is
            # enabled, so the disabled-path breakdown is byte-for-byte unchanged.
            if self.semantic.is_enabled:
                score_breakdown["semantic_boost"] = semantic_boost

            scored_results.append(RetrievalResult(
                node_id=node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                content=node.content,
                score=final_score,
                score_breakdown=score_breakdown,
                path=path_labels,
            ))

        # 5. Rank by final score, return top N
        scored_results.sort(key=lambda r: r.score, reverse=True)
        top_results = scored_results[:top_n]

        # 6. Touch retrieved nodes (update access tracking)
        for result in top_results:
            self.ops.touch_node(result.node_id)

        # 7. Log retrieval for neural training. The TrainingLoop is pure
        # stdlib, so signals accumulate even when the neural extra is absent —
        # training itself just no-ops until numpy/sklearn are installed.
        self.training_loop.log_retrieval(
            query=query,
            results=[
                {
                    "node_id": r.node_id,
                    "label": r.label,
                    "node_type": r.node_type,
                    "score": r.score,
                }
                for r in top_results
            ],
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResponse(
            query=query,
            results=top_results,
            nodes_examined=nodes_examined,
            retrieval_time_ms=round(elapsed_ms, 2),
            neural_active=self.neural_scorer.is_neural,
        )

    def _effective_confidence(self, node: Node, now: datetime) -> float:
        """
        Layer 1: effective confidence used as the recall post-factor — PURE.

        INFERRED nodes decay -0.01/week since last_referenced (floored at
        DECAY_FLOOR); EXTRACTED, DERIVED, CORRECTED and pinned nodes are immune.
        Delegates the decay MATH to GraphOperations._compute_decayed_confidence,
        which performs NO writes — so recall NEVER persists or audits. Decay is
        only persisted by the explicit maintenance pass (_apply_decay). This
        keeps reads pure: no write latency, no decay-spam in the audit log.

        The `now` arg is accepted for API parity but unused — decay anchors to
        wall-clock now inside the pure helper.
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node.confidence
        return self.ops._compute_decayed_confidence(node)

    def _find_anchors(self, query: str) -> List[str]:
        """
        Extract entities/topics from query and find matching nodes in the graph.
        These become the starting points for graph traversal.
        """
        # Use the same extraction logic as ingestion
        extraction = self.extractor.extract(query, source_id="__query__")
        anchor_ids = []

        # Look for extracted entities/topics in the graph
        for candidate in extraction.nodes:
            if candidate.node_type == NodeType.CONTEXT:
                continue  # Skip the query's own context node

            # Try exact match first
            existing = self.ops.find_node_by_label(
                candidate.label, node_type=candidate.node_type
            )
            if existing:
                anchor_ids.append(existing.node_id)
                continue

            # Try fuzzy match
            fuzzy = self.ops.find_nodes_by_label_fuzzy(candidate.label, max_distance=3)
            for match in fuzzy:
                if match.node_id not in anchor_ids:
                    anchor_ids.append(match.node_id)

        return anchor_ids

    def _keyword_search(self, query: str) -> List[str]:
        """
        Fallback: search node labels and content for query keywords.
        Used when entity extraction finds no anchors.
        """
        words = set(query.lower().split())
        # Remove very common words
        stop = {"what", "did", "we", "about", "the", "is", "are", "was",
                "were", "how", "when", "where", "why", "who", "do", "does",
                "a", "an", "in", "on", "at", "to", "for", "of", "with",
                "that", "this", "it", "and", "or", "but", "not", "no",
                "have", "has", "had", "be", "been", "will", "would",
                "can", "could", "should", "may", "might", "my", "our",
                "i", "you", "he", "she", "they", "me", "us", "them",
                "last", "next", "any", "some"}
        keywords = words - stop

        if not keywords:
            return []

        anchor_ids = []
        all_nodes = self.store.list_nodes(limit=999999)

        for node in all_nodes:
            if node.node_type == NodeType.CONTEXT:
                continue
            text = f"{node.label} {node.content}".lower()
            # Score by keyword overlap
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                anchor_ids.append(node.node_id)

        return anchor_ids[:10]  # Cap at 10 anchors to prevent explosion

    def mark_used(self, node_id: str, query: Optional[str] = None) -> None:
        """
        Mark a node as actually used after retrieval.
        Call this when the user references or acts on retrieved information.
        Provides positive training signal AND reinforces edge weights along the path.
        """
        # 1. Log training signal (pure stdlib — works without neural extra)
        self.training_loop.mark_used(node_id, query)

        # 2. Touch the node (bump access count + last_accessed)
        self.ops.touch_node(node_id)

        # 2b. Provenance hook (leg 6a): record the access in the audit trail.
        # touch_node suppresses its own generic "update"; this is the single
        # "access" entry. Defensive — never breaks the underlying op.
        accessed = self.store.get_node(node_id)
        if accessed is not None:
            self.store._record_node_audit(
                node_id, "access", after_node=accessed,
            )

        # 3. Reinforce edges connected to this node (retrieval-driven learning)
        REINFORCEMENT_DELTA = 0.05  # Small boost per usage
        edges = self.store.get_edges_for_node(node_id)
        for edge in edges:
            new_weight = min(1.0, edge.weight + REINFORCEMENT_DELTA)
            if new_weight != edge.weight:
                self.store.update_edge_weight(edge.edge_id, new_weight)

    def get_training_stats(self) -> Dict:
        """Get neural training statistics."""
        return {
            "training": self.training_loop.get_stats(),
            "scorer": self.neural_scorer.get_stats(),
        }

    def force_train(self) -> bool:
        """Force training run regardless of threshold. For testing/manual triggers.

        Returns False when the neural extra is not installed (training no-ops).
        """
        return self.training_loop.train()
