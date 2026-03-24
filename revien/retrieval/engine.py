"""
Revien Retrieval Engine — Query processing, graph walking, scoring, and ranking.
The full retrieval pipeline: parse query → find anchors → walk graph → score → rank.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.extractor import RuleBasedExtractor
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


class RetrievalEngine:
    """
    Full retrieval pipeline:
    1. Parse query to extract entities/topics
    2. Find anchor nodes in the graph
    3. Walk the graph from anchors
    4. Score all reachable nodes
    5. Rank and return top N
    """

    def __init__(
        self,
        store: GraphStore,
        scoring_config: Optional[ScoringConfig] = None,
        max_depth: int = 3,
    ):
        self.store = store
        self.ops = GraphOperations(store)
        self.extractor = RuleBasedExtractor()
        self.scorer = ThreeFactorScorer(scoring_config)
        self.walker = GraphWalker(store, max_depth=max_depth)
        self.max_depth = max_depth

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

            # Score
            breakdown = self.scorer.score(
                last_accessed=node.last_accessed,
                access_count=node.access_count,
                graph_distance=distance,
                now=now,
            )

            if breakdown.composite < min_score:
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
                score=breakdown.composite,
                score_breakdown={
                    "recency": breakdown.recency,
                    "frequency": breakdown.frequency,
                    "proximity": breakdown.proximity,
                },
                path=path_labels,
            ))

        # 5. Rank by composite score, return top N
        scored_results.sort(key=lambda r: r.score, reverse=True)
        top_results = scored_results[:top_n]

        # 6. Touch retrieved nodes (update access tracking)
        for result in top_results:
            self.ops.touch_node(result.node_id)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalResponse(
            query=query,
            results=top_results,
            nodes_examined=nodes_examined,
            retrieval_time_ms=round(elapsed_ms, 2),
        )

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
