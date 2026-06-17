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

    def __init__(
        self,
        store: GraphStore,
        scoring_config: Optional[ScoringConfig] = None,
        max_depth: int = 3,
        clustering: Optional[CommunityDetector] = None,
        model_dir: Optional[str] = None,
        training_db: Optional[str] = None,
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

    def recall(
        self,
        query: str,
        top_n: int = 5,
        min_score: float = 0.01,
        now: Optional[datetime] = None,
    ) -> RetrievalResponse:
        """
        Query the memory graph and return ranked results.

        Args:
            query: Natural language query
            top_n: Maximum results to return (default 5, max 20)
            min_score: Minimum composite score threshold
            now: Current time for recency scoring (defaults to UTC now)

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

            # Layer 1 (leg 1): confidence multiplier as a post-factor.
            # Lazy decay for INFERRED nodes is applied here, reusing leg-1's
            # GraphOperations._apply_decay helper (no duplicated decay logic).
            effective_confidence = self._effective_confidence(node, now)
            final_score = (base_score + community_boost) * effective_confidence

            if final_score < min_score:
                continue

            # Build path labels
            path_ids = node_paths.get(node_id, [node_id])
            path_labels = []
            for pid in path_ids:
                pnode = self.store.get_node(pid)
                if pnode:
                    path_labels.append(pnode.label)

            scored_results.append(RetrievalResult(
                node_id=node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                content=node.content,
                score=final_score,
                score_breakdown={
                    "recency": breakdown.recency,
                    "frequency": breakdown.frequency,
                    "proximity": breakdown.proximity,
                    "base_composite": breakdown.composite,
                    "community_boost": community_boost,
                    "effective_confidence": effective_confidence,
                    "neural_adjusted": self.neural_scorer.is_neural,
                },
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
        Layer 1 (leg 1): effective confidence used as the recall post-factor.

        INFERRED nodes decay -0.01/week since last_referenced; EXTRACTED,
        DERIVED, CORRECTED and pinned nodes are immune. Decayed confidence is
        lazily persisted when the change is meaningful (> 0.005).

        This does NOT reimplement decay — it delegates to leg-1's
        GraphOperations._apply_decay (the single source of truth for the decay
        rate, the INFERRED/pinned immunity rules, and the lazy-persist
        threshold), then returns the resulting confidence. Non-decaying nodes
        short-circuit so we avoid a redundant store round-trip.

        Note (seam): leg-1's _apply_decay anchors decay to the wall-clock
        datetime.now(), so the `now` argument here is accepted for API parity
        with the server engine but the decay clock is owned by leg 1. For
        live recall now == wall clock, so behavior matches.
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node.confidence
        decayed = self.ops._apply_decay(node)
        return decayed.confidence

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
