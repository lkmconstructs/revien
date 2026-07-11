"""
Cross-encoder head reranker (A1): reordering logic, gating, and degrade
behavior — all with an injected scorer, no model download. The real-model
path is exercised by the bench sweep, not unit tests.
"""

import os
import tempfile

import pytest

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.rerank import CrossEncoderReranker


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


class FakeResult:
    def __init__(self, content, score):
        self.content = content
        self.score = score
        self.score_breakdown = {}


def _results(*contents):
    # Base-scored order: first argument = highest base score.
    return [FakeResult(c, 1.0 - i * 0.1) for i, c in enumerate(contents)]


class TestRerankerUnit:
    def test_enabled_by_default_and_opt_out(self, monkeypatch):
        # Shipped contract (July 11 2026): DEFAULT ON when fastembed is
        # present; REVIEN_RERANK=0 opts out. (conftest pins =0 for the
        # suite, so the true default needs the var removed.)
        monkeypatch.delenv("REVIEN_RERANK", raising=False)
        assert CrossEncoderReranker().is_enabled

        monkeypatch.setenv("REVIEN_RERANK", "0")
        rr = CrossEncoderReranker()
        assert not rr.is_enabled
        assert "opted out" in rr.inactive_reason()
        results = _results("a", "b")
        assert rr.rerank("q", results) is results  # pure pass-through

    def test_default_model_and_depth_are_the_measured_shape(self, monkeypatch):
        monkeypatch.delenv("REVIEN_RERANK_MODEL", raising=False)
        monkeypatch.delenv("REVIEN_RERANK_TOP_K", raising=False)
        rr = CrossEncoderReranker()
        assert rr.model_name == "revien/ms-marco-MiniLM-L-6-v2-int8"
        assert rr.top_k == 20

    def test_injected_scorer_reorders_head(self):
        # Scorer prefers content "gold" — it must rise above base order.
        rr = CrossEncoderReranker(
            scorer=lambda q, texts: [1.0 if t == "gold" else 0.0 for t in texts]
        )
        assert rr.is_enabled
        out = rr.rerank("q", _results("hub", "noise", "gold"))
        assert [r.content for r in out] == ["gold", "hub", "noise"]
        assert out[0].score_breakdown["rerank_score"] == 1.0

    def test_base_score_not_overwritten(self):
        rr = CrossEncoderReranker(scorer=lambda q, t: [0.5] * len(t))
        results = _results("a", "b")
        base_scores = [r.score for r in results]
        out = rr.rerank("q", results)
        assert [r.score for r in out] == base_scores

    def test_tail_beyond_top_k_keeps_order(self):
        # top_k=2: only the first two are rescored; "gold" at position 3
        # stays in the tail exactly where it was.
        rr = CrossEncoderReranker(
            top_k=2,
            scorer=lambda q, texts: [1.0 if t == "b" else 0.0 for t in texts],
        )
        out = rr.rerank("q", _results("a", "b", "gold", "d"))
        assert [r.content for r in out] == ["b", "a", "gold", "d"]
        assert "rerank_score" not in out[2].score_breakdown

    def test_runtime_failure_degrades_to_passthrough(self, capsys):
        def boom(q, texts):
            raise RuntimeError("onnx exploded")

        rr = CrossEncoderReranker(scorer=boom)
        results = _results("a", "b")
        out = rr.rerank("q", results)
        assert out is results
        assert not rr.is_enabled
        assert "runtime error" in rr.inactive_reason()
        assert "DISABLED" in capsys.readouterr().err
        # Subsequent calls stay pass-through without re-raising.
        assert rr.rerank("q", results) is results


class TestRerankerEngineIntegration:
    def _seed(self, store):
        anchor = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Quicksilver", content="hub",
        ))
        gold = store.add_node(Node(
            node_type=NodeType.FACT, label="gold fact",
            content="the answer lives here",
        ))
        decoy = store.add_node(Node(
            node_type=NodeType.FACT, label="decoy fact",
            content="plausible but wrong",
        ))
        for target in (decoy, gold):  # decoy linked first = same base shape
            store.add_edge(Edge(
                edge_type=EdgeType.RELATED_TO,
                source_node_id=anchor.node_id,
                target_node_id=target.node_id,
                weight=0.5,
            ))
        return gold, decoy

    def test_engine_applies_reranker_before_top_n(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        gold, decoy = self._seed(store)
        rr = CrossEncoderReranker(
            scorer=lambda q, texts: [
                1.0 if "answer" in t else 0.0 for t in texts
            ]
        )
        engine = RetrievalEngine(store, reranker=rr)
        response = engine.recall("quicksilver", top_n=10)
        contents = [r.content for r in response.results]
        assert contents.index("the answer lives here") < contents.index(
            "plausible but wrong"
        )

    def test_engine_with_rerank_opted_out_is_unchanged(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        monkeypatch.setenv("REVIEN_RERANK", "0")
        self._seed(store)
        engine = RetrievalEngine(store)
        assert not engine.reranker.is_enabled
        response = engine.recall("quicksilver", top_n=10)
        for r in response.results:
            assert "rerank_score" not in r.score_breakdown
