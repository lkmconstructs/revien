"""
Tests for entity normalization (prong A: one definition of label equivalence)
and known-entity mention linking (prong B: the gazetteer pass that draws the
edges the regex extractor is blind to).
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from revien.graph.normalize import normalize_label, normalize_text
from revien.graph.operations import GraphOperations
from revien.graph.schema import EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
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


def _entity(store, label, curated=True):
    """Test entity; curated by default because mention linking is gated to
    curated (vault-origin) entities — the measured scope where it pays."""
    now = datetime.now(timezone.utc)
    return store.add_node(Node(
        node_type=NodeType.ENTITY, label=label, content=label,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        created_at=now, last_accessed=now,
        metadata={"curated": True} if curated else {},
    ))


def _pipeline(store):
    return IngestionPipeline(store, semantic=SemanticIndex(store, enabled=False))


class TestNormalizeLabel:
    def test_surface_forms_collapse(self):
        forms = ["Providence-Core", "providence core", "Providence Core",
                 "Providence_Core", "PROVIDENCE.CORE"]
        assert len({normalize_label(f) for f in forms}) == 1

    def test_possessive_and_punctuation(self):
        assert normalize_label("Mira's Design") == normalize_label("mira design")
        assert normalize_label("  Atlas   Server!  ") == "atlas server"

    def test_no_plural_or_alias_folding(self):
        # Deliberate non-goals: these are DIFFERENT normalized forms.
        assert normalize_label("Redis") != normalize_label("Redi")
        assert normalize_label("PostgreSQL") != normalize_label("postgres")

    def test_text_matches_label_space(self):
        assert " " + normalize_label("Atlas Server") + " " in \
            " " + normalize_text("the atlas-server needs a fan") + " "


class TestNormalizedLookup:
    def test_find_node_by_label_across_surface_forms(self, store):
        node = _entity(store, "Providence Core")
        ops = GraphOperations(store)
        for form in ("Providence-Core", "providence_core", "PROVIDENCE CORE"):
            found = ops.find_node_by_label(form, node_type=NodeType.ENTITY)
            assert found is not None and found.node_id == node.node_id

    def test_dedup_merges_surface_variants(self, store):
        """'The Atlas Server' (extractor, article included) and 'Atlas Server'
        (clean form) are one entity — normalized dedup must merge them."""
        pipeline = _pipeline(store)
        pipeline.ingest(IngestionInput(
            source_id="s1", content="User: The Atlas Server is our main box.",
        ))
        pipeline.ingest(IngestionInput(
            source_id="s2", content="User: Atlas Server maintenance runs Friday.",
        ))
        entities = [
            n for n in store.list_nodes(node_type=NodeType.ENTITY, limit=1000)
            if normalize_label(n.label) == "atlas server"
        ]
        assert len(entities) == 1, \
            f"surface variants must merge, got {[e.label for e in entities]}"

    def test_leading_article_stripped_on_labels_not_text(self):
        assert normalize_label("The Atlas Server") == "atlas server"
        # In running text 'the' is just a word — padding handles boundaries.
        assert normalize_text("the atlas server hums") == "the atlas server hums"


class TestMentionLinking:
    def test_lowercase_hyphen_mention_attaches(self, store):
        """The vault-eval fragile-variant class: a known entity mentioned in a
        surface form the regex extractor cannot see still gets the edge."""
        entity = _entity(store, "Atlas Server")
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="turn1",
            content="User: the atlas-server needs a new case fan, ordered one.",
        ))
        neighbors = store.get_neighbors(entity.node_id)
        assert out.context_node_id in neighbors, \
            "lowercase/hyphenated mention must link to the known entity"

    def test_mention_edge_weight_and_type(self, store):
        entity = _entity(store, "Postgres Cluster")
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="turn2",
            content="User: we should bump the postgres cluster to version 17.",
        ))
        edges = [
            e for e in store.get_edges_for_node(entity.node_id)
            if out.context_node_id in (e.source_node_id, e.target_node_id)
            and e.edge_type == EdgeType.RELATED_TO
        ]
        assert edges and edges[0].weight == IngestionPipeline.MENTION_EDGE_WEIGHT

    def test_short_labels_never_link(self, store):
        """A short entity name must not attach to every sentence containing it
        as an ordinary word. ('Ivy', not 'Go' — 'go' is in the extractor's own
        tech-entity list and links via a different, pre-existing path.)"""
        entity = _entity(store, "Ivy")
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="turn3", content="User: the ivy on the wall is spreading.",
        ))
        assert out.context_node_id not in store.get_neighbors(entity.node_id)

    def test_word_boundary_no_substring_hits(self, store):
        entity = _entity(store, "Halcyon")
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="turn4",
            content="User: the halcyonic weather made for a good walk.",
        ))
        assert out.context_node_id not in store.get_neighbors(entity.node_id)

    def test_no_duplicate_edge_when_extractor_already_linked(self, store):
        """A properly-capitalized mention gets ONE connection (via extractor
        dedup or gazetteer), not stacked duplicates."""
        _entity(store, "Kubernetes Cluster")
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="turn5",
            content="User: The Kubernetes Cluster restart fixed the timeout.",
        ))
        ops = GraphOperations(store)
        entity = ops.find_node_by_label("Kubernetes Cluster", node_type=NodeType.ENTITY)
        edges = [
            e for e in store.get_edges_for_node(entity.node_id)
            if out.context_node_id in (e.source_node_id, e.target_node_id)
        ]
        # At most one edge per (direction, type) pair — no stacking.
        keys = {(e.source_node_id, e.target_node_id, e.edge_type) for e in edges}
        assert len(keys) == len(edges)

    def test_distinct_entities_never_merge_on_substring(self, store):
        """Asher's Lincoln case: normalization is exact-equality on canonical
        forms, never substring — 'Lincoln' and 'Lincoln Elementary' stay
        distinct nodes forever."""
        a = _entity(store, "Lincoln")
        b = _entity(store, "Lincoln Elementary")
        ops = GraphOperations(store)
        found = ops.find_node_by_label("Lincoln", node_type=NodeType.ENTITY)
        assert found.node_id == a.node_id
        found_school = ops.find_node_by_label("lincoln-elementary", node_type=NodeType.ENTITY)
        assert found_school.node_id == b.node_id

    def test_normalization_only_merges_are_audited(self, store):
        """The precision surface: a merge that happened ONLY because of
        normalization writes an audit row carrying BOTH labels, and the bench
        report function surfaces the pair."""
        from revien_bench.failure_analysis import normalization_merge_report
        pipeline = _pipeline(store)
        pipeline.ingest(IngestionInput(
            source_id="s1", content="User: The Atlas Server is our main box.",
        ))
        pipeline.ingest(IngestionInput(
            source_id="s2", content="User: Atlas Server maintenance runs Friday.",
        ))
        report = normalization_merge_report(store)
        assert report["count"] >= 1
        assert any("Atlas Server" in p for p in report["pairs"])

    def test_plain_case_merges_are_not_flagged(self, store):
        """Raw case-insensitive merges existed before normalization — they
        must NOT pollute the precision surface."""
        from revien_bench.failure_analysis import normalization_merge_report
        pipeline = _pipeline(store)
        pipeline.ingest(IngestionInput(
            source_id="s1", content="User: We deploy on Kubernetes today.",
        ))
        pipeline.ingest(IngestionInput(
            source_id="s2", content="User: We deploy on Kubernetes tomorrow too.",
        ))
        assert normalization_merge_report(store)["count"] == 0

    def test_machine_extracted_entities_are_not_scanned(self, store):
        """Curated gating: mention linking pays on curated (vault) entities
        and bought zero conversational recall at real cost — so machine-
        extracted entities are NOT gazetteer targets."""
        entity = _entity(store, "Nightjar Relay", curated=False)
        pipeline = _pipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="t1", content="User: the nightjar-relay deploy is friday.",
        ))
        assert out.context_node_id not in store.get_neighbors(entity.node_id)

    def test_curated_entity_created_mid_stream_is_linkable(self, store):
        """A curated entity born in turn N (vault-style, via links) is
        linkable by turn N+1 within the same pipeline instance (cache
        append, not reload)."""
        pipeline = _pipeline(store)
        pipeline.ingest(IngestionInput(
            source_id="vault:Nightjar Relay.md#nightjar-relay",
            content="Nightjar Relay\nThe new relay service.",
            links=["Nightjar Relay"], curated=True,
        ))
        out2 = pipeline.ingest(IngestionInput(
            source_id="t2", content="User: the nightjar-relay deploy is friday.",
        ))
        ops = GraphOperations(store)
        entity = ops.find_node_by_label("Nightjar Relay", node_type=NodeType.ENTITY)
        assert entity is not None
        assert out2.context_node_id in store.get_neighbors(entity.node_id)