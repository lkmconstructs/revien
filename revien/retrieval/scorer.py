"""
Revien Three-Factor Scorer — Scores candidate nodes on recency, frequency, and proximity.
This is the core ranking algorithm that makes Revien's retrieval surgical.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a node's composite score."""
    recency: float
    frequency: float
    proximity: float
    composite: float


@dataclass
class ScoringConfig:
    """Configurable weights and parameters for the scoring engine."""
    # Factor weights (must sum to 1.0)
    recency_weight: float = 0.35
    frequency_weight: float = 0.30
    proximity_weight: float = 0.35

    # Recency: exponential decay
    recency_half_life_days: float = 7.0  # Half-life in days

    # Frequency: logarithmic scaling
    frequency_diminishing_threshold: int = 50  # Diminishing returns after this

    # Proximity: graph distance decay
    proximity_decay_per_hop: float = 0.3  # Score reduction per hop
    proximity_max_depth: int = 3  # Maximum hops to consider


class ThreeFactorScorer:
    """
    Scores candidate nodes using three independent factors:
    - Recency: How recently was this node accessed?
    - Frequency: How often has this node been retrieved?
    - Proximity: How close is this node to the query anchor nodes in the graph?
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score(
        self,
        last_accessed: datetime,
        access_count: int,
        graph_distance: int,
        now: Optional[datetime] = None,
    ) -> ScoreBreakdown:
        """
        Compute composite score for a candidate node.

        Args:
            last_accessed: When the node was last accessed/retrieved
            access_count: How many times the node has been retrieved
            graph_distance: Shortest path distance from query anchor nodes (0 = anchor itself)
            now: Current time (defaults to UTC now)

        Returns:
            ScoreBreakdown with individual factor scores and composite
        """
        if now is None:
            now = datetime.now(timezone.utc)

        recency = self._score_recency(last_accessed, now)
        frequency = self._score_frequency(access_count)
        proximity = self._score_proximity(graph_distance)

        composite = (
            self.config.recency_weight * recency
            + self.config.frequency_weight * frequency
            + self.config.proximity_weight * proximity
        )

        return ScoreBreakdown(
            recency=round(recency, 4),
            frequency=round(frequency, 4),
            proximity=round(proximity, 4),
            composite=round(composite, 4),
        )

    def _score_recency(self, last_accessed: datetime, now: datetime) -> float:
        """
        Exponential decay from last_accessed.
        Score = 0.5 ^ (days_since / half_life)
        Recent nodes score close to 1.0; old nodes decay toward 0.
        """
        # Ensure both datetimes are timezone-aware for comparison
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta = now - last_accessed
        days_since = max(delta.total_seconds() / 86400.0, 0.0)
        half_life = self.config.recency_half_life_days

        if half_life <= 0:
            return 1.0 if days_since == 0 else 0.0

        return math.pow(0.5, days_since / half_life)

    def _score_frequency(self, access_count: int) -> float:
        """
        Logarithmic scaling of access_count with diminishing returns.
        Score = log(1 + count) / log(1 + threshold)
        Capped at 1.0 after threshold.
        """
        threshold = self.config.frequency_diminishing_threshold
        if threshold <= 0:
            return 1.0

        raw = math.log(1 + access_count) / math.log(1 + threshold)
        return min(raw, 1.0)

    def _score_proximity(self, graph_distance: int) -> float:
        """
        Graph distance decay.
        Distance 0 (anchor node itself) = 1.0
        Each hop reduces score by decay_per_hop.
        Beyond max_depth = 0.0
        """
        if graph_distance < 0:
            return 0.0
        if graph_distance > self.config.proximity_max_depth:
            return 0.0

        decay = self.config.proximity_decay_per_hop
        return max(1.0 - (graph_distance * decay), 0.0)
