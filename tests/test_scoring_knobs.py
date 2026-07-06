"""
Tests for the env-tunable ranking knobs (sweep support). Unset env must be
byte-identical to the shipped defaults — the knobs exist for experiments, not
to move the default path.
"""

import os
import tempfile

import pytest

from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.retrieval.scorer import ScoringConfig
from revien.semantic.index import SemanticIndex


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


class TestScoringConfigFromEnv:
    def test_unset_env_is_exact_defaults(self, monkeypatch):
        for var in ("REVIEN_RECENCY_WEIGHT", "REVIEN_FREQUENCY_WEIGHT",
                    "REVIEN_PROXIMITY_WEIGHT", "REVIEN_RECENCY_HALF_LIFE_DAYS",
                    "REVIEN_PROXIMITY_DECAY_PER_HOP"):
            monkeypatch.delenv(var, raising=False)
        assert ScoringConfig.from_env() == ScoringConfig()

    def test_env_overrides_apply(self, monkeypatch):
        monkeypatch.setenv("REVIEN_RECENCY_HALF_LIFE_DAYS", "365")
        monkeypatch.setenv("REVIEN_FREQUENCY_WEIGHT", "0.0")
        cfg = ScoringConfig.from_env()
        assert cfg.recency_half_life_days == 365.0
        assert cfg.frequency_weight == 0.0
        # Untouched knobs keep their defaults.
        assert cfg.proximity_weight == ScoringConfig().proximity_weight

    def test_malformed_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("REVIEN_RECENCY_HALF_LIFE_DAYS", "not-a-number")
        assert ScoringConfig.from_env().recency_half_life_days == 365.0


class TestEngineKnobs:
    def test_defaults_match_class_constants(self, store, monkeypatch):
        for var in ("REVIEN_SEMANTIC_TOP_K", "REVIEN_SEMANTIC_SIM_FLOOR",
                    "REVIEN_GRAPH_REFINE", "REVIEN_COMMUNITY_BOOST"):
            monkeypatch.delenv(var, raising=False)
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.semantic_top_k == RetrievalEngine.SEMANTIC_TOP_K
        assert eng.semantic_sim_floor == RetrievalEngine.SEMANTIC_SIM_FLOOR
        assert eng.graph_refine == RetrievalEngine.GRAPH_REFINE
        assert eng.community_boost == RetrievalEngine.COMMUNITY_BOOST

    def test_env_overrides_engine_knobs(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC_TOP_K", "100")
        monkeypatch.setenv("REVIEN_SEMANTIC_SIM_FLOOR", "0.15")
        monkeypatch.setenv("REVIEN_GRAPH_REFINE", "0.10")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        assert eng.semantic_top_k == 100
        assert eng.semantic_sim_floor == 0.15
        assert eng.graph_refine == 0.10

    def test_explicit_scoring_config_wins_over_env(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_RECENCY_HALF_LIFE_DAYS", "365")
        cfg = ScoringConfig(recency_half_life_days=14.0)
        eng = RetrievalEngine(
            store, scoring_config=cfg,
            semantic=SemanticIndex(store, enabled=False),
        )
        assert eng.scorer.config.recency_half_life_days == 14.0


def _fact(store, label, content, recorded_at=None):
    from datetime import datetime, timezone
    from revien.graph.schema import Node, NodeType, SourceType
    now = datetime.now(timezone.utc)
    node = Node(
        node_type=NodeType.FACT, label=label, content=content,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        created_at=now, last_accessed=now, recorded_at=recorded_at,
    )
    return store.add_node(node)


class TestContentTimeRecency:
    """Recency must score WHEN THE MEMORY WAS SAID (recorded_at), not when it
    was last touched. The old access-time recency was constant in any
    historical-`now` evaluation and correlated with retrieval popularity."""

    def test_recency_scores_recorded_at(self, store):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        old = _fact(store, "postgres decision",
                    "we chose postgres", recorded_at=now - timedelta(days=70))
        fresh = _fact(store, "postgres migration",
                      "postgres migration is done", recorded_at=now)

        # Explicit sharp half-life so the content-time contrast is visible
        # (the shipped 365d default decays 70 days to only ~0.88).
        eng = RetrievalEngine(
            store, scoring_config=ScoringConfig(recency_half_life_days=7.0),
            semantic=SemanticIndex(store, enabled=False),
        )
        resp = eng.recall("postgres", now=now)
        rec = {r.node_id: r.score_breakdown["recency"] for r in resp.results}
        # 70 days at a 7-day half-life: ~0.001 vs ~1.0. last_accessed is `now`
        # for BOTH nodes — under access-time recency these would tie.
        assert rec[fresh.node_id] > 0.9
        assert rec[old.node_id] < 0.01

    def test_recency_falls_back_to_created_at(self, store):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        node = _fact(store, "redis cache", "redis for sessions", recorded_at=None)
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("redis", now=now)
        rec = {r.node_id: r.score_breakdown["recency"] for r in resp.results}
        # created_at is ~now, so fallback recency is ~1.0 (not a crash, not 0).
        assert rec[node.node_id] > 0.9


class TestTouchOnRecallGate:
    def test_default_recall_does_not_self_touch(self, store, monkeypatch):
        """Shipped default (sweep, July 2026): recall() must not feed its own
        frequency signal. Only mark_used() increments access_count."""
        monkeypatch.delenv("REVIEN_TOUCH_ON_RECALL", raising=False)
        node = _fact(store, "docker setup", "docker on staging")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        resp = eng.recall("docker")
        assert any(r.node_id == node.node_id for r in resp.results)
        assert store.get_node(node.node_id).access_count == 0
        # mark_used still feeds the signal — that's the honest path.
        eng.mark_used(node.node_id)
        assert store.get_node(node.node_id).access_count == 1

    def test_gate_on_restores_self_touching(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_TOUCH_ON_RECALL", "1")
        node = _fact(store, "docker setup", "docker on staging")
        eng = RetrievalEngine(store, semantic=SemanticIndex(store, enabled=False))
        eng.recall("docker")
        assert store.get_node(node.node_id).access_count == 1
