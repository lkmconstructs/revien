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

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# One-shot flag for the graph-only degrade warning (per process, not per engine
# — bench runs construct hundreds of engines and one warning is the message).
_SEMANTIC_OFF_WARNED = False

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
from .scorer import ScoreBreakdown, ScoringConfig, ThreeFactorScorer, _env_float
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
    # Semantic-layer visibility: recall quality differs ~10x between the hybrid
    # and graph-only paths (LoCoMo recall@10 0.47 vs 0.05), so every response
    # says WHICH path served it. semantic_note is None when active, else a
    # one-line reason the layer is off — the caller must be able to see a
    # degrade, not infer it from mysteriously worse results.
    semantic_active: bool = False
    semantic_note: Optional[str] = None
    # Populated only when recall(debug=True): anchor sets, walked distances,
    # per-node final scores, and filter reasons. This is what per-query
    # retrieval failure analysis (extraction/seed/walk/ranking miss) reads.
    diagnostics: Optional[Dict[str, Any]] = None


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

    # Semantic-FIRST ranking — when the opt-in semantic layer is enabled, a node
    # the query embeds close to is ranked PRIMARILY by that similarity, and the
    # three-factor graph composite only REFINES it (weight GRAPH_REFINE). This is
    # what makes recall query-discriminative: a genuinely relevant node outranks
    # high-frequency/high-proximity hub nodes, instead of being given a weak
    # additive nudge that the hubs swamp (the v1 bug: same hubs returned for
    # every query). When the layer is disabled this path is inert and the graph
    # composite is used unchanged.
    GRAPH_REFINE = 0.25
    # How many nearest neighbours to pull as semantic anchors/candidates.
    SEMANTIC_TOP_K = 30
    # Similarity floor for treating a node as a semantic match. bge-small scores
    # genuinely-relevant turns in roughly the 0.3-0.6 band, so the old 0.55 floor
    # rejected most real matches and recall fell back to hub-walking. 0.30 admits
    # real matches while excluding pure noise; the candidates are already the
    # top-K nearest, so relative rank — not an absolute threshold — carries the
    # signal.
    SEMANTIC_SIM_FLOOR = 0.30

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
        # No explicit config -> defaults + env overrides (ScoringConfig.from_env;
        # unset env == exact defaults). This is what lets the bench sweep ranking
        # knobs without code edits.
        self.scorer = ThreeFactorScorer(scoring_config or ScoringConfig.from_env())
        # Weighted walk (A1): edge strength = weight, optionally * edge
        # confidence. The strength only moves scores when the scorer's
        # edge_weight_blend knob (REVIEN_EDGE_WEIGHT_BLEND) is > 0.
        self.walker = GraphWalker(
            store,
            max_depth=max_depth,
            use_edge_confidence=os.environ.get(
                "REVIEN_EDGE_CONFIDENCE_IN_WALK", "0"
            ).strip().lower() in ("1", "true", "yes", "on"),
        )
        self.max_depth = max_depth
        self.clustering = clustering

        # Ranking knobs (see class constants for semantics). Env-overridable for
        # sweeps; the class constants remain the shipped defaults. The miss
        # taxonomy says 72% of semantic-path misses are `outranked` — these are
        # the levers that decide that ranking.
        self.semantic_top_k = int(_env_float("REVIEN_SEMANTIC_TOP_K", self.SEMANTIC_TOP_K))
        self.semantic_sim_floor = _env_float(
            "REVIEN_SEMANTIC_SIM_FLOOR", self.SEMANTIC_SIM_FLOOR
        )
        self.graph_refine = _env_float("REVIEN_GRAPH_REFINE", self.GRAPH_REFINE)
        self.community_boost = _env_float("REVIEN_COMMUNITY_BOOST", self.COMMUNITY_BOOST)
        # Frequency feedback-loop gate — DEFAULT OFF (sweep-shipped July 2026):
        # recall() touching its own results made access_count a popularity
        # prior contaminated by the engine's own behavior (being returned →
        # higher frequency → returned again), measured at -21% recall@10 vs
        # honest frequency at full scale. Only mark_used() — a caller
        # confirming the memory was actually useful — feeds access_count.
        # Set REVIEN_TOUCH_ON_RECALL=1 to restore the old behavior.
        self.touch_on_recall = os.environ.get(
            "REVIEN_TOUCH_ON_RECALL", "0"
        ).strip().lower() in ("1", "true", "yes", "on")

        # Neural components (opt-in). These construct cleanly even without the
        # `neural` extra: NeuralScorer.is_neural stays False and adjust_score()
        # is a pass-through, so recall() degrades to base scoring.
        self.neural_scorer = NeuralScorer(model_dir=model_dir)
        self.training_loop = TrainingLoop(db_path=training_db, model_dir=model_dir)

        # Semantic/vector layer (SPINE — core deps as of the promote-to-spine
        # change). Constructs cleanly even when the deps are absent (source
        # install): SemanticIndex.is_enabled stays False and all of its methods
        # no-op, so recall() degrades to the graph-only path — but LOUDLY:
        self.semantic = semantic if semantic is not None else SemanticIndex(store)
        self._warn_if_semantic_inactive()

    def _warn_if_semantic_inactive(self) -> None:
        """One warning per process when recall will run graph-only. Graph-only
        recall has no query-relevance signal beyond keyword overlap (LoCoMo
        recall@10: 0.05 vs 0.47 hybrid) — an engine silently running in that
        mode is the bug this warning exists to catch."""
        global _SEMANTIC_OFF_WARNED
        if self.semantic.is_enabled or _SEMANTIC_OFF_WARNED:
            return
        _SEMANTIC_OFF_WARNED = True
        sys.stderr.write(
            f"[revien] recall is running GRAPH-ONLY (keyword) retrieval - "
            f"semantic layer inactive: {self.semantic.inactive_reason()}. "
            f"Recall quality is significantly degraded. Set "
            f"REVIEN_SEMANTIC=require to make this fatal.\n"
        )
        sys.stderr.flush()

    def recall(
        self,
        query: str,
        top_n: int = 5,
        min_score: float = 0.01,
        now: Optional[datetime] = None,
        include_invalidated: bool = False,
        include_context: bool = False,
        debug: bool = False,
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
            debug: When True, the response carries a ``diagnostics`` dict
                (anchor sets by origin, walked node distances, per-node final
                scores including sub-threshold ones, and filter reasons) so a
                caller can classify WHY a given node was or wasn't returned.
                Default False: zero overhead, response unchanged.

        Returns:
            RetrievalResponse with ranked nodes and timing data
        """
        start_time = time.perf_counter()
        top_n = min(top_n, 20)

        if now is None:
            now = datetime.now(timezone.utc)

        # 1. Parse query — extract entities and topics
        entity_anchor_ids = self._find_anchors(query)
        anchor_ids = list(entity_anchor_ids)

        # 2. If no anchors found, try keyword search across all nodes
        keyword_anchor_ids: List[str] = []
        if not anchor_ids:
            keyword_anchor_ids = self._keyword_search(query)
            anchor_ids = list(keyword_anchor_ids)

        # 2a. Hybrid semantic anchors (opt-in). When the semantic layer is
        # enabled, embed the query, pull the nearest stored nodes, and UNION
        # them into the anchor set. This is what lets a keyword-less query
        # (no entity/keyword overlap with any node) still find relevant nodes.
        # When the layer is disabled, semantic_sims is empty and the anchor set
        # is byte-for-byte what it was before this leg.
        semantic_sims: Dict[str, float] = {}
        if self.semantic.is_enabled:
            for node_id, sim in self.semantic.search(query, top_k=self.semantic_top_k):
                # Only nodes that clear the floor act as semantic anchors. This
                # is what lets a keyword-less query reach a genuinely-close node
                # without near-uniform mild similarity reshuffling keyword hits.
                if sim < self.semantic_sim_floor:
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

        # 3. Walk graph from anchors — ONE traversal yields distances, paths,
        # and path strengths (this used to be two full BFS passes per recall).
        if anchor_ids:
            node_distances, node_paths, node_strengths = self.walker.walk_full(
                anchor_ids
            )
        else:
            node_distances = {}
            node_paths = {}
            node_strengths = {}

        # 4. Score all reachable nodes. One bulk fetch for the whole walked
        # frontier — a SELECT per node here was a measured latency driver.
        scored_results: List[RetrievalResult] = []
        nodes_examined = len(node_distances)
        nodes_map = self.store.get_nodes_bulk(node_distances.keys())
        # Debug bookkeeping (leg: per-query failure analysis). Cheap dicts,
        # built only when asked for.
        diag_scores: Dict[str, float] = {}
        diag_filtered: Dict[str, str] = {}

        for node_id, distance in node_distances.items():
            node = nodes_map.get(node_id)
            if node is None:
                if debug:
                    diag_filtered[node_id] = "missing"
                continue

            # CONTEXT nodes are the verbatim turns — answer-bearing content for
            # conversational memory. Surface them by default; callers wanting only
            # distilled extract nodes can pass include_context=False.
            if node.node_type == NodeType.CONTEXT and not include_context:
                if debug:
                    diag_filtered[node_id] = "context_excluded"
                continue

            # Provenance (leg 6a): exclude soft-invalidated nodes by default.
            # No-op when nothing is invalidated, so recall stays byte-identical.
            if node.invalidated_at is not None and not include_invalidated:
                if debug:
                    diag_filtered[node_id] = "invalidated"
                continue

            # Three-factor base score (recency + frequency + proximity).
            # Recency scores CONTENT time — when the memory was said
            # (recorded_at), falling back to when it entered the graph
            # (created_at). Scoring last_accessed here made "recency" mean
            # recently-touched: it correlated with retrieval popularity, was
            # constant in any evaluation querying at a historical `now`, and
            # buried old-but-true facts behind whatever was returned last.
            breakdown = self.scorer.score(
                timestamp=node.recorded_at or node.created_at,
                access_count=node.access_count,
                graph_distance=distance,
                now=now,
                path_strength=node_strengths.get(node_id, 1.0),
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
                    community_boost = self.community_boost

            # Layer 1 (leg 1): confidence multiplier as a post-factor.
            effective_confidence = self._effective_confidence(node, now)

            # Semantic-FIRST ranking (opt-in). When the semantic layer surfaced
            # this node for the query, similarity is the PRIMARY term and the
            # graph composite only refines it (GRAPH_REFINE) — so query-relevant
            # nodes outrank frequency/proximity hubs. semantic_sims is empty when
            # the layer is disabled, so sim is None and the graph-only expression
            # below runs unchanged (byte-identical to the pre-semantic path).
            sim = semantic_sims.get(node_id)
            if sim is not None:
                final_score = (sim + self.graph_refine * base_score + community_boost) * effective_confidence
            else:
                final_score = (base_score + community_boost) * effective_confidence

            if debug:
                diag_scores[node_id] = final_score

            if final_score < min_score:
                continue

            # Build path labels (path nodes are all within the walked set,
            # so the bulk map already has them).
            path_ids = node_paths.get(node_id, [node_id])
            path_labels = []
            for pid in path_ids:
                pnode = nodes_map.get(pid)
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
                score_breakdown["semantic_sim"] = sim if sim is not None else 0.0
            # Same rule for path strength: only in the breakdown when the
            # weighted-walk blend is actually shaping the score.
            if self.scorer.config.edge_weight_blend > 0.0:
                score_breakdown["path_strength"] = round(
                    node_strengths.get(node_id, 1.0), 4
                )

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

        # 6. Touch retrieved nodes (update access tracking). Env-gated
        # (REVIEN_TOUCH_ON_RECALL=0 disables) because this is the frequency
        # feedback loop: being RETURNED bumps access_count, which raises the
        # frequency score, which gets the node returned again — retrieval
        # popularity masquerading as relevance. With the gate off, only
        # mark_used() (a caller confirming the memory was actually useful)
        # feeds the frequency signal. Default ON (shipped behavior unchanged)
        # pending the sweep verdict.
        if self.touch_on_recall:
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

        diagnostics: Optional[Dict[str, Any]] = None
        if debug:
            diagnostics = {
                "anchors": {
                    "entity": entity_anchor_ids,
                    "keyword": keyword_anchor_ids,
                    "semantic": list(semantic_sims.keys()),
                    "all": list(anchor_ids),
                },
                "node_distances": dict(node_distances),
                "scores": diag_scores,
                "filtered": diag_filtered,
                "max_depth": self.max_depth,
            }

        return RetrievalResponse(
            query=query,
            results=top_results,
            nodes_examined=nodes_examined,
            retrieval_time_ms=round(elapsed_ms, 2),
            neural_active=self.neural_scorer.is_neural,
            semantic_active=self.semantic.is_enabled,
            semantic_note=self.semantic.inactive_reason(),
            diagnostics=diagnostics,
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

        # SQL-side substring search (same semantics as the old Python scan:
        # any-keyword hit on label+content, newest first, CONTEXT excluded,
        # capped at 10 anchors). The old list_nodes(limit=999999) full scan
        # was the single biggest recall latency driver (OPEN 2).
        matches = self.store.search_nodes_keyword(
            keywords, limit=10, exclude_context=True
        )
        return [n.node_id for n in matches]

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
