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


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """A forget writes a key under the Revien home and scrubs training.db
    under the user's home. No test may touch the real ones: every test gets a
    throwaway home."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("REVIEN_HOME", str(home / ".revien"))
    from revien.forget import keys
    keys._cache.clear()
    yield
    keys._cache.clear()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_probe: use the real loopback health probe (revien/loopback_probe.py) "
        "instead of the default stub that reports 'no daemon answered'")


@pytest.fixture(autouse=True)
def _no_real_daemon_probe(request, monkeypatch):
    """`revien sync reset` asks a loopback daemon for /v1/health. A developer
    machine may really have one on 7437, which would make unrelated tests
    refuse or flake. Stub the probe to 'nothing answered' unless a test marks
    itself real_probe."""
    if request.node.get_closest_marker("real_probe"):
        yield
        return
    import revien.loopback_probe as lp

    monkeypatch.setattr(lp, "probe_daemon", lambda url: None)
    yield
