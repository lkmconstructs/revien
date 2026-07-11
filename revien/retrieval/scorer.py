"""
Revien Three-Factor Scorer — Scores candidate nodes on recency, frequency, and proximity.
This is the core ranking algorithm that makes Revien's retrieval surgical.
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


def _env_float(name: str, default: float) -> float:
    """Read a float env override; malformed values fall back to the default
    (a bad experiment knob must never crash recall)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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

    # Recency: exponential decay over CONTENT time (recorded_at). 365d makes
    # recency a gentle tiebreak, not a burial: the 7-day default predated the
    # content-time semantics and zeroed anything said more than a month ago —
    # sweep-measured at full scale (LoCoMo 1,986 QA), 7d cost 2.6x on recall@1
    # vs this default. Override: REVIEN_RECENCY_HALF_LIFE_DAYS.
    recency_half_life_days: float = 365.0  # Half-life in days

    # Frequency: logarithmic scaling
    frequency_diminishing_threshold: int = 50  # Diminishing returns after this

    # Proximity: graph distance decay
    proximity_decay_per_hop: float = 0.3  # Score reduction per hop
    proximity_max_depth: int = 3  # Maximum hops to consider

    # Weighted walk (A1): how much of the proximity term comes from PATH
    # STRENGTH (product of edge weights along the shortest-hop path — the
    # walker's strengths dict) vs pure hop-count decay.
    #   proximity = (1 - blend) * hop_decay + blend * path_strength
    # 0.0 (default) is byte-identical to shipped hop-only behavior; 1.0 ranks
    # entirely by how strong the connecting edges are. This is what finally
    # READS the edge weights that mark_used() reinforcement and author-drawn
    # vault edges (0.8) vs extractor co-occurrence guesses (0.3) have been
    # writing. Override: REVIEN_EDGE_WEIGHT_BLEND.
    edge_weight_blend: float = 0.0

    @classmethod
    def from_env(cls) -> "ScoringConfig":
        """Build a config with env overrides for the ranking knobs the miss
        taxonomy points at (72% of semantic-path misses are `outranked`).
        Unset env == exact defaults, so the default path is byte-identical.
        Weights are NOT auto-renormalized — pass a full set that sums to 1.0
        when sweeping, so what you measured is what you configured."""
        d = cls()
        return cls(
            recency_weight=_env_float("REVIEN_RECENCY_WEIGHT", d.recency_weight),
            frequency_weight=_env_float("REVIEN_FREQUENCY_WEIGHT", d.frequency_weight),
            proximity_weight=_env_float("REVIEN_PROXIMITY_WEIGHT", d.proximity_weight),
            recency_half_life_days=_env_float(
                "REVIEN_RECENCY_HALF_LIFE_DAYS", d.recency_half_life_days
            ),
            proximity_decay_per_hop=_env_float(
                "REVIEN_PROXIMITY_DECAY_PER_HOP", d.proximity_decay_per_hop
            ),
            edge_weight_blend=_env_float(
                "REVIEN_EDGE_WEIGHT_BLEND", d.edge_weight_blend
            ),
        )


class ThreeFactorScorer:
    """
    Scores candidate nodes using three independent factors:
    - Recency: How recent is this memory's CONTENT (when it was said/recorded)?
    - Frequency: How often has this node been retrieved?
    - Proximity: How close is this node to the query anchor nodes in the graph?
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score(
        self,
        timestamp: datetime,
        access_count: int,
        graph_distance: int,
        now: Optional[datetime] = None,
        path_strength: float = 1.0,
    ) -> ScoreBreakdown:
        """
        Compute composite score for a candidate node.

        Args:
            timestamp: The node's CONTENT time — when the memory was said or
                recorded (recorded_at, falling back to created_at). Scoring
                access time here instead (last_accessed) makes "recency" mean
                recently-TOUCHED, which correlates with retrieval popularity,
                not with how fresh the remembered fact is.
            access_count: How many times the node has been retrieved
            graph_distance: Shortest path distance from query anchor nodes (0 = anchor itself)
            now: Current time (defaults to UTC now)
            path_strength: Product of edge strengths along the node's
                shortest-hop path from the anchors (walker strengths dict;
                anchors = 1.0). Inert unless edge_weight_blend > 0.

        Returns:
            ScoreBreakdown with individual factor scores and composite
        """
        if now is None:
            now = datetime.now(timezone.utc)

        recency = self._score_recency(timestamp, now)
        frequency = self._score_frequency(access_count)
        proximity = self._score_proximity(graph_distance, path_strength)

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

    def _score_recency(self, timestamp: datetime, now: datetime) -> float:
        """
        Exponential decay from the node's content time.
        Score = 0.5 ^ (days_since / half_life)
        Recent nodes score close to 1.0; old nodes decay toward 0.
        """
        # Ensure both datetimes are timezone-aware for comparison
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta = now - timestamp
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

    def _score_proximity(
        self, graph_distance: int, path_strength: float = 1.0
    ) -> float:
        """
        Graph distance decay, optionally blended with path strength.
        Distance 0 (anchor node itself) = 1.0
        Each hop reduces the hop term by decay_per_hop.
        Beyond max_depth = 0.0 regardless of strength (reachability gates
        stay hop-based; strength shapes ranking within reach).

        With edge_weight_blend > 0:
            proximity = (1 - blend) * hop_decay + blend * path_strength
        At the default blend 0.0 this returns the hop term unchanged —
        byte-identical to the pre-weighted-walk scorer.
        """
        if graph_distance < 0:
            return 0.0
        if graph_distance > self.config.proximity_max_depth:
            return 0.0

        decay = self.config.proximity_decay_per_hop
        hop = max(1.0 - (graph_distance * decay), 0.0)

        blend = self.config.edge_weight_blend
        if blend <= 0.0:
            return hop
        blend = min(blend, 1.0)
        strength = min(max(path_strength, 0.0), 1.0)
        return (1.0 - blend) * hop + blend * strength
