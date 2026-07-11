"""
Shared pytest configuration.

The opt-in semantic/vector layer (leg 5) defaults to ON whenever its extra
(sqlite-vec) is importable in the environment. The bulk of the test suite
asserts the BASE graph-only retrieval contract (anchor selection, three-factor
+ community + confidence scoring, ranking, latency budgets). When the semantic
extra happens to be installed in the dev/CI environment, leaving it default-on
would change those base-contract outcomes.

So we pin REVIEN_SEMANTIC=0 for the suite by default: existing tests keep
validating the unchanged graph-only engine regardless of whether the extra is
installed. The dedicated semantic tests (tests/test_semantic.py) opt back in
EXPLICITLY by constructing SemanticIndex(..., enabled=True) — they do not rely
on the env default — so the hybrid path is still fully exercised there.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _disable_semantic_by_default(monkeypatch):
    """Force the semantic layer off for base-contract tests unless a test
    explicitly enables it via SemanticIndex(enabled=True)."""
    monkeypatch.setenv("REVIEN_SEMANTIC", "0")
    yield


@pytest.fixture(autouse=True)
def _disable_rerank_by_default(monkeypatch):
    """Same rationale for the cross-encoder reranker (DEFAULT ON in
    production since July 11 2026): base-contract tests assert graph/scoring
    behavior and must stay deterministic, fast, and offline — a default-on
    reranker would add model inference (and a cold-cache download) to every
    recall in the suite. Rerank tests exercise the layer explicitly via
    injected scorers or monkeypatched env; the real-model path is measured
    by the bench, not unit tests."""
    monkeypatch.setenv("REVIEN_RERANK", "0")
    yield
