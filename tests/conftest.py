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
