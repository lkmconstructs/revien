"""
B1 — Tension/COEXIST: gate hook, governor edge-drawing, queue resolution.

All recognizer behavior is stubbed (deterministic, $0). The real-LLM
recognizer quality is measured by revien_bench/tension_eval.py, opt-in.

CLAIM_TAXONOMY §7.2 controls are load-bearing here: a retraction must keep
its supersession handling (never coexist), negative sentiment must not fire
the hook (affirmative-affirmative guard), and no-recognizer must be
byte-identical to pre-B1 behavior.
"""

import os
import tempfile

import pytest

from revien.claims import ClaimType, ClassificationResult, ClassificationStatus, Durability
from revien.graph.operations import GraphOperations
from revien.graph.schema import EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.supersession_ingest import ClaimGovernor
from revien.supersession import Claim, SupersessionAction, SupersessionGate
from revien.tension import TensionRoute, TensionVerdict


# ── Stubs ─────────────────────────────────────────────────

class StubTension:
    """Deterministic recognizer: returns a fixed route, counts consultations."""

    def __init__(self, route=TensionRoute.TENSION):
        self.route = route
        self.calls = []

    def recognize_pair(self, a, b):
        self.calls.append((a, b))
        return TensionVerdict(self.route)


def _claim(text, claim_type=ClaimType.PREFERENCE_HABIT,
           status=ClassificationStatus.CLASSIFIED,
           durability=Durability.SLOW_CHANGE, confidence=0.65):
    return Claim(text=text, result=ClassificationResult(
        claim_type=claim_type,
        claim_type_confidence=confidence,
        classification_status=status,
        durability=durability,
        durability_confidence=0.6,
    ))


# Both sides classify as preference_habit with the REAL classifier (verified),
# so governor end-to-end tests exercise the full classify->gate->edge path.
TENSION_A = "I love working from home."
TENSION_B = "I love the energy of working in the office."


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


# ── Gate-level ────────────────────────────────────────────

class TestTensionGate:
    def test_tension_verdict_yields_coexist(self):
        stub = StubTension(TensionRoute.TENSION)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(_claim("I want closeness with people."),
                          _claim("I want plenty of alone time."))
        assert d.action is SupersessionAction.COEXIST
        assert d.reason == "tension_coexist"
        assert "tension_recognized" in d.trace
        assert len(stub.calls) == 1

    def test_compatible_verdict_falls_through_unchanged(self):
        stub = StubTension(TensionRoute.COMPATIBLE)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(_claim("I want more sleep."),
                          _claim("I want more rest."))
        assert d.action is SupersessionAction.NO_CONFLICT
        assert d.reason == "scoped_but_compatible"
        assert "tension_checked:compatible" in d.trace

    def test_abstain_falls_through_unchanged(self):
        stub = StubTension(TensionRoute.ABSTAIN)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(_claim("I want more sleep."),
                          _claim("I want more rest."))
        assert d.action is SupersessionAction.NO_CONFLICT

    def test_no_recognizer_is_byte_identical(self):
        gate = SupersessionGate()
        d = gate.evaluate(_claim("I want closeness with people."),
                          _claim("I want plenty of alone time."))
        assert d.action is SupersessionAction.NO_CONFLICT
        assert d.reason == "scoped_but_compatible"
        assert not any(t.startswith("tension") for t in d.trace)

    def test_retraction_never_reaches_the_hook(self):
        # Polarity flip on a shared object IS a rule-detectable contradiction
        # — it takes the existing supersession path and the recognizer is
        # never consulted (§7.2: a retraction is an update, not a tension).
        stub = StubTension(TensionRoute.TENSION)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(_claim("I love working from home."),
                          _claim("I don't love working from home anymore."))
        assert d.action is not SupersessionAction.COEXIST
        assert stub.calls == []

    def test_negated_pair_blocked_by_affirmative_guard(self):
        # A negated side that ISN'T a detected contradiction (no shared
        # object tokens) must not consult the recognizer: tension is
        # affirmative-affirmative by definition.
        stub = StubTension(TensionRoute.TENSION)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(_claim("I love quiet evenings."),
                          _claim("I don't enjoy crowded parties."))
        assert d.action is SupersessionAction.NO_CONFLICT
        assert stub.calls == []

    def test_single_valued_type_never_consults(self):
        stub = StubTension(TensionRoute.TENSION)
        gate = SupersessionGate(tension_recognizer=stub)
        d = gate.evaluate(
            _claim("I am from Boston.", claim_type=ClaimType.IDENTITY),
            _claim("I am a morning person.", claim_type=ClaimType.IDENTITY),
        )
        assert d.action is not SupersessionAction.COEXIST
        assert stub.calls == []

    def test_metrics_count_coexist(self):
        from revien.supersession import SupersessionMetrics
        stub = StubTension(TensionRoute.TENSION)
        gate = SupersessionGate(tension_recognizer=stub)
        m = SupersessionMetrics()
        d = gate.evaluate(_claim("I want closeness with people."),
                          _claim("I want plenty of alone time."))
        m.record(d)
        assert m.coexist == 1
        assert m.total == 1


# ── Governor end-to-end ───────────────────────────────────

class TestGovernorCoexist:
    def _governor(self, store, route=TensionRoute.TENSION):
        ops = GraphOperations(store)
        gate = SupersessionGate(tension_recognizer=StubTension(route))
        return ClaimGovernor(store, ops, gate=gate)

    def _ingest_pair(self, store):
        old = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="claim 1", content=TENSION_A,
        ))
        new = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="claim 2", content=TENSION_B,
        ))
        return old, new

    def test_coexist_draws_edge_both_live_nothing_queued(self, store):
        gov = self._governor(store)
        old, new = self._ingest_pair(store)
        outcomes = gov.govern(new)

        assert [o.action for o in outcomes] == ["coexist"]
        assert "BOTH claims live" in outcomes[0].data_effect
        # Both claims live:
        assert store.get_node(old.node_id).invalidated_at is None
        assert store.get_node(new.node_id).invalidated_at is None
        # Tension edge drawn:
        edges = [e for e in store.get_edges_for_node(old.node_id)
                 if e.edge_type is EdgeType.CONFLICTS_WITH]
        assert len(edges) == 1
        assert edges[0].confidence_set_by == "csl_tension_gate"
        # Nothing queued:
        assert store.count_candidates() == 0

    def test_re_govern_is_idempotent_no_duplicate_edge(self, store):
        gov = self._governor(store)
        old, new = self._ingest_pair(store)
        gov.govern(new)
        gov.govern(new)  # re-ingest scenario: governor re-compares every time
        edges = [e for e in store.get_edges_for_node(old.node_id)
                 if e.edge_type is EdgeType.CONFLICTS_WITH]
        assert len(edges) == 1

    def test_compatible_pair_mutates_nothing(self, store):
        gov = self._governor(store, route=TensionRoute.COMPATIBLE)
        old, new = self._ingest_pair(store)
        outcomes = gov.govern(new)
        assert outcomes == []
        assert store.get_edges_for_node(old.node_id) == []
        assert store.count_candidates() == 0


class TestCoexistResolution:
    def test_coexist_candidate_draws_edge_and_resolves(self, store):
        ops = GraphOperations(store)
        gov = ClaimGovernor(store, ops)
        old = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="c1", content="I need stability.",
        ))
        new = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="c2", content="I crave change.",
        ))
        cid = store.add_candidate(old.node_id, new.node_id, "candidate", "test")

        assert gov.coexist_candidate(cid) is True
        edges = [e for e in store.get_edges_for_node(old.node_id)
                 if e.edge_type is EdgeType.CONFLICTS_WITH]
        assert len(edges) == 1
        assert store.count_candidates(unresolved_only=True) == 0
        resolved = store.list_candidates(unresolved_only=False)[0]
        assert resolved["resolution"] == "coexist"
        # Both claims still live — resolution mutates neither.
        assert store.get_node(old.node_id).invalidated_at is None
        assert store.get_node(new.node_id).invalidated_at is None

    def test_unknown_or_resolved_candidate_returns_false(self, store):
        ops = GraphOperations(store)
        gov = ClaimGovernor(store, ops)
        assert gov.coexist_candidate(9999) is False


# ── Tension surfacing (recall flag + tensions view) ───────

def _tension_graph(store):
    """Two live claims joined by a CONFLICTS_WITH edge, reachable from an
    entity anchor (keyword search anchors on non-context nodes, so a
    context-only graph is unreachable in graph-only mode — mirror the real
    ingest shape: entity -> RELATED_TO -> context turns)."""
    from revien.graph.schema import Edge
    anchor = store.add_node(Node(
        node_type=NodeType.ENTITY, label="Belonging",
        content="closeness and space needs",
    ))
    a = store.add_node(Node(
        node_type=NodeType.CONTEXT, label="closeness",
        content="I want closeness with the people I love.",
    ))
    b = store.add_node(Node(
        node_type=NodeType.CONTEXT, label="space",
        content="I want space to be alone with my thoughts.",
    ))
    for target in (a, b):
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=anchor.node_id, target_node_id=target.node_id,
            weight=0.5,
        ))
    store.add_edge(Edge(
        edge_type=EdgeType.CONFLICTS_WITH,
        source_node_id=a.node_id, target_node_id=b.node_id,
        weight=0.8, source_context="tension_coexist",
    ))
    return a, b, anchor


class TestTensionsView:
    def test_live_pairs_listed_with_both_claims(self, store):
        a, b, _ = _tension_graph(store)
        pairs = store.list_tension_pairs()
        assert len(pairs) == 1
        contents = {pairs[0]["a"]["content"], pairs[0]["b"]["content"]}
        assert a.content in contents and b.content in contents

    def test_invalidated_side_hides_pair_unless_asked(self, store):
        a, b, _ = _tension_graph(store)
        ops = GraphOperations(store)
        ops.invalidate_node(b.node_id, reason="test", construct_id="test")
        assert store.list_tension_pairs() == []
        history = store.list_tension_pairs(live_only=False)
        assert len(history) == 1


class TestRecallTensionSurfacing:
    def test_flag_attaches_live_counterpart(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        from revien.retrieval.engine import RetrievalEngine
        a, b, _ = _tension_graph(store)
        engine = RetrievalEngine(store)
        response = engine.recall(
            "closeness people love", top_n=10,
            include_context=True, include_tensions=True,
        )
        by_id = {r.node_id: r for r in response.results}
        assert a.node_id in by_id
        tensions = by_id[a.node_id].tensions
        assert len(tensions) == 1
        assert tensions[0]["node_id"] == b.node_id
        assert "space" in tensions[0]["content"]

    def test_flag_off_is_byte_identical(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        from revien.retrieval.engine import RetrievalEngine
        a, _, _ = _tension_graph(store)
        engine = RetrievalEngine(store)
        response = engine.recall(
            "closeness people love", top_n=10, include_context=True,
        )
        by_id = {r.node_id: r for r in response.results}
        assert by_id[a.node_id].tensions == []

    def test_invalidated_counterpart_not_surfaced(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_SEMANTIC", "0")
        from revien.retrieval.engine import RetrievalEngine
        a, b, _ = _tension_graph(store)
        GraphOperations(store).invalidate_node(
            b.node_id, reason="test", construct_id="test")
        engine = RetrievalEngine(store)
        response = engine.recall(
            "closeness people love", top_n=10,
            include_context=True, include_tensions=True,
        )
        by_id = {r.node_id: r for r in response.results}
        assert by_id[a.node_id].tensions == []
