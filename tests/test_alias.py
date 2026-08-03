"""
Alias leg — evidence-backed ALIAS_OF edges + recall anchor expansion.

Covers: candidate blocking (never all-pairs), precision-first evidence
scoring for both routes (name_form / conceptual), the hard guards
(CONFLICTS_WITH, node_type, invalidated, idempotency), the audited
edge-mutation path (create + reverse), recall anchor expansion (and its
env gate + invalidation-respecting behavior), the daemon API, and the
consolidate.py toggle. Mirrors the house test style in test_consolidate.py
and test_graph.py.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.alias import AliasPassResult, run_alias_pass
from revien.consolidate import Consolidator
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
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


@pytest.fixture
def ops(store):
    return GraphOperations(store)


def _entity(store, label, content=None):
    return store.add_node(Node(
        node_type=NodeType.ENTITY, label=label, content=content or label,
    ))


def _context(store, label, content=None):
    return store.add_node(Node(
        node_type=NodeType.CONTEXT, label=label, content=content or label,
        source_type=SourceType.EXTRACTED, confidence=1.0,
    ))


def _link(store, a_id, b_id, edge_type=EdgeType.RELATED_TO, weight=0.5):
    return store.add_edge(Edge(
        edge_type=edge_type, source_node_id=a_id, target_node_id=b_id, weight=weight,
    ))


class _FakeEmbedder:
    """Deterministic, dependency-free embedder (EmbeddingProvider protocol):
    a fixed label -> vector map, no fastembed/model load. Mirrors
    tests/test_semantic.py's _MockEmbedder pattern."""

    def __init__(self, vectors: dict):
        self._vectors = vectors

    @property
    def dim(self):
        return len(next(iter(self._vectors.values())))

    @property
    def is_cloud(self):
        return False

    def embed(self, texts):
        return [self._vectors[t] for t in texts]


def _semantic_with(store, vectors: dict) -> SemanticIndex:
    """A SemanticIndex wired to the fake embedder above. alias.py only ever
    calls is_enabled + _get_embedder().embed(...) — never vec0 storage — so
    forcing _enabled True is safe/portable regardless of whether sqlite-vec
    happens to be installed in this environment (same override
    test_semantic.py's _InMemoryVectorIndex uses, just without a subclass
    since no vec0 method is ever exercised here)."""
    sem = SemanticIndex(store, embedder=_FakeEmbedder(vectors), enabled=True)
    sem._enabled = True
    return sem


# ── Unit: candidate scoring ────────────────────────────────────────────

class TestNameFormRoute:
    """Non-subset name_form shape: 'Sam Rivera' / 'Sam R.' overlaps via
    substring containment, not a strict token-set subset ({'sam','r'} isn't
    a subset of {'sam','rivera'} or vice versa — 'r' != 'rivera'), so it
    does NOT hit the mandatory-embedding subset rule (see TestSubsetShape).
    Corroboration bar here: >=2 distinct shared neighbors (REVIEN_ALIAS_
    COOC_MIN, same as conceptual) OR embedding_sim >= REVIEN_ALIAS_SIM_NAME.
    """

    def test_name_form_pair_with_two_shared_neighbors_creates_edge(self, store):
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        c1 = _context(store, "turn1", "Sam R. said hi")
        c2 = _context(store, "turn2", "Sam R. said bye")
        _link(store, c1.node_id, rivera.node_id)
        _link(store, c1.node_id, sam_r.node_id)
        _link(store, c2.node_id, rivera.node_id)
        _link(store, c2.node_id, sam_r.node_id)

        result = run_alias_pass(store)
        assert result.ran is True
        assert result.edges_created == 1
        assert result.sample[0]["method"] == "name_form"

        edges = [e for e in store.get_edges_for_node(rivera.node_id)
                 if e.edge_type is EdgeType.ALIAS_OF]
        assert len(edges) == 1
        assert edges[0].metadata["method"] == "name_form"
        assert edges[0].metadata["cooccurrence"] == 2
        assert edges[0].invalidated_at is None

    def test_name_form_single_shared_neighbor_no_longer_sufficient(self, store):
        # Tightened bar (adversarial-review fix): ONE shared neighbor used
        # to corroborate name_form; it no longer does — the bar matches
        # conceptual's (>=2), and there's no embedder here to clear the OR.
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        ctx = _context(store, "turn", "Sam R. said hi")
        _link(store, ctx.node_id, rivera.node_id)
        _link(store, ctx.node_id, sam_r.node_id)

        result = run_alias_pass(store)
        assert result.edges_created == 0

    def test_name_form_without_corroboration_no_edge(self, store):
        # Shares a token ("sam") but NO shared context/topic neighbor and no
        # embedder — precision-first: no corroboration, no edge.
        sam = _entity(store, "Sam")
        _entity(store, "Sam R.")
        result = run_alias_pass(store)
        assert result.edges_created == 0
        assert not [e for e in store.get_edges_for_node(sam.node_id)
                    if e.edge_type is EdgeType.ALIAS_OF]

    def test_name_form_corroborated_by_embedding_sim_alone(self, store):
        # No shared context/topic neighbor, but embedding sim clears the
        # name_form floor (REVIEN_ALIAS_SIM_NAME) — corroboration via the OR.
        # "Sam Rivera" / "Sam R." is a non-subset overlap (containment, not
        # a strict token-set subset), so this exercises the general bar,
        # not the mandatory-embedding subset rule (see TestSubsetShape).
        vectors = {"Sam Rivera": [1.0, 0.0], "Sam R.": [0.95, 0.05]}
        sem = _semantic_with(store, vectors)
        _entity(store, "Sam Rivera")
        _entity(store, "Sam R.")
        result = run_alias_pass(store, semantic=sem, sim_name=0.85)
        assert result.edges_created == 1
        assert result.sample[0]["method"] == "name_form"


class TestSubsetShape:
    """The subset shape (adversarial-review fix, generalized past the
    original single-token carve-out): one label's normalized token set is a
    STRICT subset of the other's, at ANY token count — 'sam'/'sam r'
    (single-token), but just as much 'new york'/'new york times' or
    'ford'/'ford foundation' (multi-token). Token-identical whether it's a
    name variant of the SAME entity (Sam / Sam R.) or a qualified superset
    that's a DIFFERENT thing entirely (John / John's laptop; New York /
    New York Times — a newspaper, not the city). Co-occurrence alone can
    never draw this shape, at any count; only embedding_sim can."""

    def test_possessive_pair_not_aliased_without_embedder(self, store):
        # The reviewer's original false-positive repro: John and his laptop
        # get mentioned together in every conversation about the laptop —
        # 2 shared contexts is easy to rack up and must NOT be enough.
        john = _entity(store, "John")
        laptop = _entity(store, "John's Laptop")
        c1 = _context(store, "turn1", "John updated his laptop drivers")
        c2 = _context(store, "turn2", "John's laptop battery died again")
        _link(store, c1.node_id, john.node_id)
        _link(store, c1.node_id, laptop.node_id)
        _link(store, c2.node_id, john.node_id)
        _link(store, c2.node_id, laptop.node_id)

        result = run_alias_pass(store)
        assert result.edges_created == 0
        assert not [e for e in store.get_edges_for_node(john.node_id)
                    if e.edge_type is EdgeType.ALIAS_OF]
        assert result.note is not None
        assert "subset" in result.note.lower()

    def test_same_entity_subset_pair_needs_embedding_not_cooccurrence(self, store):
        # "Sam" / "Sam R." IS the same entity, but it's still the subset
        # shape — even here, co-occurrence (however much) must not draw the
        # edge; only embedding evidence can.
        sam = _entity(store, "Sam")
        sam_r = _entity(store, "Sam R.")
        c1 = _context(store, "turn1", "Sam R. said hi")
        c2 = _context(store, "turn2", "Sam R. said bye")
        _link(store, c1.node_id, sam.node_id)
        _link(store, c1.node_id, sam_r.node_id)
        _link(store, c2.node_id, sam.node_id)
        _link(store, c2.node_id, sam_r.node_id)

        no_embedder = run_alias_pass(store)
        assert no_embedder.edges_created == 0

        vectors = {"Sam": [1.0, 0.0], "Sam R.": [0.95, 0.05]}
        sem = _semantic_with(store, vectors)
        with_embedder = run_alias_pass(store, semantic=sem, sim_name=0.85)
        assert with_embedder.edges_created == 1
        assert with_embedder.sample[0]["method"] == "name_form"

    def _seed_new_york(self, store):
        """'New York' is a subset of 'New York Times' — the reviewer's residual repro:
        a MULTI-token subset pair the original single-token-only carve-out
        missed. Two shared contexts, easy to rack up, must not be enough on
        their own."""
        ny = _entity(store, "New York")
        nyt = _entity(store, "New York Times")
        c1 = _context(store, "turn1", "New York Times covered the New York mayor's race")
        c2 = _context(store, "turn2", "New York Times reporters based in New York")
        _link(store, c1.node_id, ny.node_id)
        _link(store, c1.node_id, nyt.node_id)
        _link(store, c2.node_id, ny.node_id)
        _link(store, c2.node_id, nyt.node_id)
        return ny, nyt

    def test_new_york_times_not_aliased_without_embedder(self, store):
        ny, nyt = self._seed_new_york(store)
        result = run_alias_pass(store)
        assert result.edges_created == 0
        assert not [e for e in store.get_edges_for_node(ny.node_id)
                    if e.edge_type is EdgeType.ALIAS_OF]
        assert result.note is not None
        assert "subset" in result.note.lower()

    def test_new_york_times_not_aliased_when_embedding_sim_low(self, store):
        # Embedder IS available, but scores them as genuinely different
        # things — co-occurrence must not override that low similarity.
        ny, nyt = self._seed_new_york(store)
        vectors = {"New York": [1.0, 0.0, 0.0], "New York Times": [0.0, 1.0, 0.0]}
        sem = _semantic_with(store, vectors)
        result = run_alias_pass(store, semantic=sem, sim_name=0.85)
        assert result.edges_created == 0

    def test_new_york_times_aliased_when_embedding_sim_high(self, store):
        # High embedding similarity IS sufficient corroboration for the
        # subset shape (the mandatory-embedding rule is a floor, not a ban).
        ny, nyt = self._seed_new_york(store)
        vectors = {"New York": [1.0, 0.0], "New York Times": [0.95, 0.05]}
        sem = _semantic_with(store, vectors)
        result = run_alias_pass(store, semantic=sem, sim_name=0.85)
        assert result.edges_created == 1
        assert result.sample[0]["method"] == "name_form"


class TestConceptualRoute:
    LOW_DIM_VECTORS = {
        "Offline Mode": [1.0, 1.0, 0.0],
        "Roadmap 2026": [1.0, 0.95, 0.0],
        "Printer Setup": [0.0, 0.0, 1.0],
    }

    def _seed(self, store):
        a = _entity(store, "Offline Mode")
        b = _entity(store, "Roadmap 2026")
        # Two DISTINCT shared CONTEXT neighbors -> cooccurrence >= 2.
        c1 = _context(store, "turn1", "discussing offline mode and the roadmap")
        c2 = _context(store, "turn2", "more about offline mode and the roadmap")
        _link(store, c1.node_id, a.node_id)
        _link(store, c1.node_id, b.node_id)
        _link(store, c2.node_id, a.node_id)
        _link(store, c2.node_id, b.node_id)
        return a, b

    def test_conceptual_needs_both_gates(self, store):
        a, b = self._seed(store)
        sem = _semantic_with(store, self.LOW_DIM_VECTORS)
        result = run_alias_pass(
            store, semantic=sem, sim_concept=0.90, cooc_min=2,
        )
        assert result.edges_created == 1
        assert result.sample[0]["method"] == "conceptual"
        edges = [e for e in store.get_edges_for_node(a.node_id)
                 if e.edge_type is EdgeType.ALIAS_OF]
        assert edges[0].metadata["cooccurrence"] == 2

    def test_conceptual_sim_alone_without_cooccurrence_min_no_edge(self, store):
        # High similarity but only ONE shared neighbor (< cooc_min=2).
        a = _entity(store, "Offline Mode")
        b = _entity(store, "Roadmap 2026")
        c1 = _context(store, "turn1", "discussing offline mode and the roadmap")
        _link(store, c1.node_id, a.node_id)
        _link(store, c1.node_id, b.node_id)
        sem = _semantic_with(store, self.LOW_DIM_VECTORS)
        result = run_alias_pass(store, semantic=sem, sim_concept=0.90, cooc_min=2)
        assert result.edges_created == 0

    def test_conceptual_below_sim_floor_no_edge(self, store):
        a, b = self._seed(store)
        vectors = dict(self.LOW_DIM_VECTORS)
        vectors["Roadmap 2026"] = [0.0, 1.0, 1.0]  # now far from Offline Mode
        sem = _semantic_with(store, vectors)
        result = run_alias_pass(store, semantic=sem, sim_concept=0.90, cooc_min=2)
        assert result.edges_created == 0

    def test_conceptual_disabled_without_semantic_reports_note(self, store):
        a, b = self._seed(store)
        result = run_alias_pass(store, semantic=None)
        assert result.edges_created == 0  # no name overlap, no embedder -> nothing
        assert result.note is not None
        assert "conceptual" in result.note.lower()

    def test_conceptual_disabled_when_embedder_raises(self, store):
        a, b = self._seed(store)

        class _BoomEmbedder:
            dim = 2
            is_cloud = False

            def embed(self, texts):
                raise RuntimeError("model unavailable")

        sem = SemanticIndex(store, embedder=_BoomEmbedder(), enabled=True)
        sem._enabled = True
        result = run_alias_pass(store, semantic=sem)
        assert result.edges_created == 0
        assert result.note is not None
        assert "embedder failed" in result.note


class TestGuards:
    # Non-subset pair (containment, not a strict token-set subset) with 2
    # shared neighbors — evidence that WOULD otherwise clear the (tightened)
    # name_form bar, so these tests prove their OWN guard is what blocked
    # the edge, not an evidence shortfall.
    def _evidenced_pair(self, store):
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        c1 = _context(store, "turn1", "Sam R. said hi")
        c2 = _context(store, "turn2", "Sam R. said bye")
        _link(store, c1.node_id, rivera.node_id)
        _link(store, c1.node_id, sam_r.node_id)
        _link(store, c2.node_id, rivera.node_id)
        _link(store, c2.node_id, sam_r.node_id)
        return rivera, sam_r

    def test_conflicts_with_blocks_alias(self, store):
        rivera, sam_r = self._evidenced_pair(store)
        _link(store, rivera.node_id, sam_r.node_id, edge_type=EdgeType.CONFLICTS_WITH)

        result = run_alias_pass(store)
        assert result.edges_created == 0

    def test_invalidated_node_skipped(self, store, ops):
        rivera, sam_r = self._evidenced_pair(store)
        ops.invalidate_node(sam_r.node_id, reason="test", construct_id="test")

        result = run_alias_pass(store)
        assert result.edges_created == 0
        assert result.entities_considered == 1  # sam_r excluded up front

    def test_normalized_equal_labels_never_alias(self, store):
        # Two live nodes wearing one normalized label ("The Wolves" /
        # "the wolves") are dedup's problem — an ALIAS_OF between them is a
        # self-loop in disguise. Evidence is stacked so ONLY the equal-label
        # guard can be what blocks the edge.
        a = _entity(store, "The Wolves")
        b = _entity(store, "the wolves")
        c1 = _context(store, "turn1", "the wolves played")
        c2 = _context(store, "turn2", "the wolves won")
        for c in (c1, c2):
            _link(store, c.node_id, a.node_id)
            _link(store, c.node_id, b.node_id)

        result = run_alias_pass(store)
        assert result.edges_created == 0

    def test_idempotent_no_duplicate_on_rerun(self, store):
        rivera, sam_r = self._evidenced_pair(store)

        first = run_alias_pass(store)
        second = run_alias_pass(store)
        assert first.edges_created == 1
        assert second.edges_created == 0
        edges = [e for e in store.get_edges_for_node(rivera.node_id)
                 if e.edge_type is EdgeType.ALIAS_OF]
        assert len(edges) == 1


class TestBlocking:
    def test_no_all_pairs_blowup(self, store):
        # Four entities, NO shared normalized tokens at all. Embedding
        # vectors are crafted so exactly two MUTUAL top-1 pairs exist
        # (Apple<->Banana, Cherry<->Date); the other four cross-pairs never
        # appear as anyone's nearest neighbor. With top_k=1, candidate
        # generation must find exactly those 2 pairs, not all C(4,2)=6.
        vectors = {
            "Apple": [1.0, 0.0, 0.0, 0.0],
            "Banana": [0.9, 0.1, 0.0, 0.0],
            "Cherry": [0.0, 0.0, 1.0, 0.0],
            "Date": [0.0, 0.0, 0.9, 0.1],
        }
        for label in vectors:
            _entity(store, label)
        sem = _semantic_with(store, vectors)

        result = run_alias_pass(store, semantic=sem, top_k=1)
        assert result.candidates_considered == 2
        assert result.candidates_considered < 6  # never all-pairs


class TestReport:
    def test_max_entities_cap_skips_with_note(self, store):
        for i in range(3):
            _entity(store, f"entity {i}")
        result = run_alias_pass(store, max_entities=2)
        assert result.ran is False
        assert "REVIEN_ALIAS_MAX_ENTITIES" in result.note
        assert result.edges_created == 0

    def test_env_overrides_are_read_when_no_explicit_arg(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_ALIAS_MAX_ENTITIES", "1")
        for i in range(3):
            _entity(store, f"entity {i}")
        result = run_alias_pass(store)
        assert result.ran is False


# ── Audit: alias edge creation/removal ─────────────────────────────────

class TestAudit:
    def _make_pair(self, store):
        # Non-subset pair, 2 shared neighbors — clears the (tightened)
        # name_form bar without needing an embedder.
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        c1 = _context(store, "turn1", "Sam R. said hi")
        c2 = _context(store, "turn2", "Sam R. said bye")
        _link(store, c1.node_id, rivera.node_id)
        _link(store, c1.node_id, sam_r.node_id)
        _link(store, c2.node_id, rivera.node_id)
        _link(store, c2.node_id, sam_r.node_id)
        return rivera, sam_r

    def test_alias_creation_is_audited(self, store):
        rivera, sam_r = self._make_pair(store)
        run_alias_pass(store)
        edge = [e for e in store.get_edges_for_node(rivera.node_id)
                if e.edge_type is EdgeType.ALIAS_OF][0]
        hist = store.get_edge_audit(edge.edge_id)
        assert [h["op"] for h in hist] == ["create"]
        assert hist[0]["after"]["edge_type"] == "alias_of"

    def test_removal_soft_invalidates_and_audits(self, store, ops):
        rivera, sam_r = self._make_pair(store)
        run_alias_pass(store)
        edge = [e for e in store.get_edges_for_node(rivera.node_id)
                if e.edge_type is EdgeType.ALIAS_OF][0]

        removed = ops.invalidate_edge(edge.edge_id, reason="wrong", construct_id="bash")
        assert removed.invalidated_at is not None
        # Row retained — never deleted.
        assert store.get_edge(edge.edge_id) is not None

        ops_seen = [h["op"] for h in store.get_edge_audit(edge.edge_id)]
        assert ops_seen == ["create", "invalidate"]

    def test_invalidate_edge_idempotent(self, store, ops):
        rivera, sam_r = self._make_pair(store)
        run_alias_pass(store)
        edge = [e for e in store.get_edges_for_node(rivera.node_id)
                if e.edge_type is EdgeType.ALIAS_OF][0]
        first = ops.invalidate_edge(edge.edge_id, reason="a", construct_id="x")
        second = ops.invalidate_edge(edge.edge_id, reason="b", construct_id="y")
        assert first.invalidated_at == second.invalidated_at

    def test_add_edge_audited_rolls_back_on_audit_failure(self, store, monkeypatch):
        a = _entity(store, "a")
        b = _entity(store, "b")

        def boom(*args, **kwargs):
            raise RuntimeError("audit boom")
        monkeypatch.setattr(store, "record_audit", boom)
        edge = Edge(edge_type=EdgeType.ALIAS_OF,
                    source_node_id=a.node_id, target_node_id=b.node_id)
        with pytest.raises(RuntimeError, match="audit boom"):
            store.add_edge_audited(edge)
        assert store.get_edge(edge.edge_id) is None, "edge insert must roll back"

    def test_add_edge_unaudited_path_unchanged(self, store):
        # add_edge (existing callers) stays unaudited — no audit row appears.
        a = _entity(store, "a")
        b = _entity(store, "b")
        edge = store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO, source_node_id=a.node_id,
            target_node_id=b.node_id,
        ))
        assert store.get_edge_audit(edge.edge_id) == []


# ── Recall: anchor expansion ────────────────────────────────────────────

class TestRecallExpansion:
    def _seed_alias_and_fact(self, store, ops):
        """'Sam Rivera' (multi-word -> rule-extractor anchors it exactly)
        ALIAS_OF 'Sam R.' (a distinct node), which carries a fact reachable
        only through it."""
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        edge = Edge(edge_type=EdgeType.ALIAS_OF,
                    source_node_id=rivera.node_id, target_node_id=sam_r.node_id,
                    metadata={"method": "manual", "embedding_sim": None, "cooccurrence": 0})
        store.add_edge_audited(edge, actor="test")
        fact = store.add_node(Node(
            node_type=NodeType.FACT, label="Sam R. likes hiking",
            content="Sam R. likes hiking on weekends.",
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))
        _link(store, fact.node_id, sam_r.node_id)
        return rivera, sam_r, fact, edge

    # max_depth=1 makes the effect observable and unambiguous: the ALIAS_OF
    # edge itself already connects rivera->sam_r at ordinary graph distance 1
    # regardless of anchor expansion (an edge is an edge — the walker isn't
    # touched by this leg, same as CONFLICTS_WITH is walked unfiltered
    # today). What anchor expansion changes is whether sam_r is ALSO seeded
    # at distance 0 directly. With max_depth=1, that's the difference
    # between the fact (one hop further, off sam_r) falling inside the walk
    # budget or outside it — a real, assertable behavioral difference rather
    # than a same-reachability score nuance.

    def test_alias_expansion_reaches_aliased_facts(self, store, ops):
        rivera, sam_r, fact, edge = self._seed_alias_and_fact(store, ops)
        engine = RetrievalEngine(store, max_depth=1)
        resp = engine.recall("Tell me about Sam Rivera", top_n=10, min_score=0.0)
        labels = {r.label for r in resp.results}
        assert fact.label in labels

    def test_alias_expansion_disabled_by_env_gate(self, store, ops, monkeypatch):
        # Positive control PAIRED with the negative assertion: recall must
        # still return SOMETHING (the exact-matched anchor itself) so the
        # absence of `fact` below is evidence the gate did its job, not
        # evidence recall silently returned nothing at all (a no-op
        # expansion would otherwise pass this test vacuously).
        rivera, sam_r, fact, edge = self._seed_alias_and_fact(store, ops)
        monkeypatch.setenv("REVIEN_ALIAS", "0")
        engine = RetrievalEngine(store, max_depth=1)
        resp = engine.recall("Tell me about Sam Rivera", top_n=10, min_score=0.0)
        labels = {r.label for r in resp.results}
        assert rivera.label in labels  # positive control
        assert fact.label not in labels  # gate suppressed the expansion

    def test_invalidated_alias_edge_not_expanded(self, store, ops):
        # FIX 1 proof: max_depth=3 is generous enough that the LIVE alias
        # edge routes the ORDINARY walk to the fact with expansion playing
        # no special role (rivera -alias-> sam_r -related_to-> fact is only
        # 2 hops) — that's the positive control. After --remove
        # (invalidate_edge), the SAME edge must no longer route the walk AT
        # ALL (not just stop being an anchor-expansion seed) — get_neighbors_
        # bulk/get_neighbors_weighted_bulk now filter invalidated_at, so a
        # reversed alias is disconnected everywhere, matching what
        # `revien aliases --remove` promises.
        rivera, sam_r, fact, edge = self._seed_alias_and_fact(store, ops)
        engine = RetrievalEngine(store, max_depth=3)

        before = engine.recall("Tell me about Sam Rivera", top_n=10, min_score=0.0)
        before_labels = {r.label for r in before.results}
        assert fact.label in before_labels  # positive control: live edge routes the walk

        ops.invalidate_edge(edge.edge_id, reason="wrong", construct_id="test")

        after = engine.recall("Tell me about Sam Rivera", top_n=10, min_score=0.0)
        after_labels = {r.label for r in after.results}
        assert rivera.label in after_labels  # recall still works at all
        assert fact.label not in after_labels  # the walk no longer crosses it

    def test_expansion_is_one_hop_only(self, store, ops):
        # A -alias- B -alias- C: anchoring on A must reach B but NOT C (no
        # transitive chaining in v1). Multi-word labels so the rule
        # extractor's proper-noun regex anchors "Node Alpha" exactly.
        a = _entity(store, "Node Alpha")
        b = _entity(store, "Node Beta")
        c = _entity(store, "Node Gamma")
        store.add_edge_audited(Edge(edge_type=EdgeType.ALIAS_OF,
                                     source_node_id=a.node_id, target_node_id=b.node_id))
        store.add_edge_audited(Edge(edge_type=EdgeType.ALIAS_OF,
                                     source_node_id=b.node_id, target_node_id=c.node_id))

        engine = RetrievalEngine(store)
        anchors = engine._find_anchors("Node Alpha")
        assert a.node_id in anchors
        assert b.node_id in anchors
        assert c.node_id not in anchors


# ── Daemon API ──────────────────────────────────────────────────────────

class TestDaemonAliasEdge:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from revien.daemon.server import create_app

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        app = create_app(db_path=path)
        with TestClient(app) as c:
            yield c
        os.unlink(path)

    def test_post_edges_accepts_alias_of(self, client):
        graph = client.post("/v1/graph/import", json={
            "nodes": [
                {"node_type": "entity", "label": "Sam Rivera", "content": "Sam Rivera"},
                {"node_type": "entity", "label": "Sam R.", "content": "Sam R."},
                {"node_type": "fact", "label": "Sam R. likes hiking",
                 "content": "Sam R. likes hiking on weekends."},
            ],
            "edges": [],
        })
        assert graph.status_code == 200
        nodes = {n["label"]: n["node_id"] for n in client.get("/v1/graph").json()["nodes"]}

        link = client.post("/v1/edges", json={
            "edge_type": "related_to",
            "source_node_id": nodes["Sam R. likes hiking"],
            "target_node_id": nodes["Sam R."],
        })
        assert link.status_code == 200

        alias = client.post("/v1/edges", json={
            "edge_type": "alias_of",
            "source_node_id": nodes["Sam Rivera"],
            "target_node_id": nodes["Sam R."],
            "source_context": "manual alias",
        })
        assert alias.status_code == 200
        assert alias.json()["edge_type"] == "alias_of"

        edge_types = {e["edge_type"] for e in client.get("/v1/graph").json()["edges"]}
        assert "alias_of" in edge_types

        recall = client.post("/v1/recall", json={
            "query": "Tell me about Sam Rivera",
            "top_n": 10,
            "min_score": 0.0,
        })
        assert recall.status_code == 200
        labels = {r["label"] for r in recall.json()["results"]}
        assert "Sam R. likes hiking" in labels


# ── Consolidate toggle ──────────────────────────────────────────────────

class TestConsolidateToggle:
    def test_default_run_does_not_fire_alias_pass(self, store):
        sam = _entity(store, "Sam")
        _entity(store, "Sam R.")
        report = Consolidator(store).run(recluster=False)
        assert report.alias_ran is False
        assert report.alias_edges_created == 0

    def test_alias_true_fires_and_reports_counts(self, store):
        rivera = _entity(store, "Sam Rivera")
        sam_r = _entity(store, "Sam R.")
        c1 = _context(store, "turn1", "Sam R. said hi")
        c2 = _context(store, "turn2", "Sam R. said bye")
        _link(store, c1.node_id, rivera.node_id)
        _link(store, c1.node_id, sam_r.node_id)
        _link(store, c2.node_id, rivera.node_id)
        _link(store, c2.node_id, sam_r.node_id)

        report = Consolidator(store).run(recluster=False, alias=True)
        assert report.alias_ran is True
        assert report.alias_edges_created == 1
        assert len(report.alias_sample) == 1


# ── Clustering exclusion (adversarial-review fix) ───────────────────────

class TestClusteringExcludesAliasEdges:
    """ALIAS_OF is a recall-routing edge, not a semantic-community one — it
    must never reshape Louvain communities. Exercises _load_graph directly
    (no clustering algorithm run needed) since there's no existing
    clustering test file/pattern to extend."""

    def test_alias_edge_excluded_from_community_graph(self, store):
        from revien.graph.clustering import CommunityDetector

        a = _entity(store, "Node Alpha")
        b = _entity(store, "Node Beta")
        store.add_edge_audited(Edge(
            edge_type=EdgeType.ALIAS_OF, source_node_id=a.node_id, target_node_id=b.node_id,
        ))

        detector = CommunityDetector(db_path=store.db_path)
        conn = store._get_conn()
        G = detector._load_graph(conn)

        assert G.has_node(a.node_id) and G.has_node(b.node_id)
        assert not G.has_edge(a.node_id, b.node_id)

    def test_invalidated_non_alias_edge_excluded_too(self, store, ops):
        from revien.graph.clustering import CommunityDetector

        a = _entity(store, "Node Gamma")
        b = _entity(store, "Node Delta")
        edge = store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO, source_node_id=a.node_id, target_node_id=b.node_id,
        ))
        ops.invalidate_edge(edge.edge_id, reason="test", construct_id="test")

        detector = CommunityDetector(db_path=store.db_path)
        conn = store._get_conn()
        G = detector._load_graph(conn)

        assert not G.has_edge(a.node_id, b.node_id)

    def test_live_non_alias_edge_still_included(self, store):
        from revien.graph.clustering import CommunityDetector

        a = _entity(store, "Node Epsilon")
        b = _entity(store, "Node Zeta")
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO, source_node_id=a.node_id, target_node_id=b.node_id,
        ))

        detector = CommunityDetector(db_path=store.db_path)
        conn = store._get_conn()
        G = detector._load_graph(conn)

        assert G.has_edge(a.node_id, b.node_id)  # unaffected: not alias, not invalidated
