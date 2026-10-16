"""ONNX Runtime telemetry is switched off before any model/session exists.
Stub onnxruntime + stub fastembed; no real model loads, no network."""

import sys
import types

import pytest

from revien.semantic import _ort
from revien.semantic import index as idx
from revien.semantic import rerank as rr


@pytest.fixture
def seq(monkeypatch):
    calls = []
    ort = types.ModuleType("onnxruntime")
    ort.disable_telemetry_events = lambda: calls.append("disable")

    class TE:
        def __init__(self, *a, **k):
            calls.append("TextEmbedding")

    class TCE:
        def __init__(self, *a, **k):
            calls.append("TextCrossEncoder")

        @classmethod
        def add_custom_model(cls, *a, **k):
            pass

    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = TE
    rer = types.ModuleType("fastembed.rerank")
    ce = types.ModuleType("fastembed.rerank.cross_encoder")
    ce.TextCrossEncoder = TCE
    for name, mod in [("onnxruntime", ort), ("fastembed", fe),
                      ("fastembed.rerank", rer),
                      ("fastembed.rerank.cross_encoder", ce)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(_ort, "_DONE", False)
    monkeypatch.setattr(idx, "_FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(rr, "_FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(rr, "_register_builtin_variants", lambda: None)
    return calls


def test_disabled_once_and_before_models(seq):
    idx.FastEmbedProvider()._ensure_model()
    idx.FastEmbedProvider()._ensure_model()
    rr.CrossEncoderReranker()._ensure_model()
    rr.CrossEncoderReranker()._ensure_model()
    assert seq.count("disable") == 1
    assert seq[0] == "disable"
    assert seq == ["disable", "TextEmbedding", "TextEmbedding",
                   "TextCrossEncoder", "TextCrossEncoder"]


def test_reranker_alone_disables_first(seq):
    rr.CrossEncoderReranker()._ensure_model()
    assert seq == ["disable", "TextCrossEncoder"]


def test_missing_onnxruntime_false_no_raise(monkeypatch):
    monkeypatch.setattr(_ort, "_DONE", False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises
    assert _ort.disable_ort_telemetry() is False
    assert _ort.ort_telemetry_state() is None


def test_raising_switch_never_raises(monkeypatch):
    ort = types.ModuleType("onnxruntime")

    def boom():
        raise RuntimeError("x")

    ort.disable_telemetry_events = boom
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(_ort, "_DONE", False)
    assert _ort.disable_ort_telemetry() is False


def test_sovereignty_detail_carries_field(seq):
    from revien_bench import sovereignty as S
    assert S.network_egress_zero().detail["ort_telemetry_disabled"] is True


def test_sovereignty_detail_none_when_absent(monkeypatch):
    from revien_bench import sovereignty as S
    monkeypatch.setattr(_ort, "_DONE", False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert S.network_egress_zero().detail["ort_telemetry_disabled"] is None


def test_semantic_status_field(seq):
    from revien.graph.store import GraphStore
    s = idx.SemanticIndex(GraphStore(db_path=":memory:"))
    assert s.status()["ort_telemetry_disabled"] is True
