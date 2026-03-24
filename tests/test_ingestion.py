"""
Test 2: Ingestion Pipeline
Feed a sample conversation into the ingestion engine.
Verify extraction of entities, topics, decisions, facts.
Verify edges connect extracted nodes to context.
Verify deduplication on second ingestion.
"""

import os
import tempfile

import pytest

from revien.graph.schema import NodeType, EdgeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.extractor import RuleBasedExtractor
from revien.ingestion.dedup import Deduplicator
from revien.ingestion.pipeline import IngestionPipeline, IngestionInput


# The sample conversation from the build spec
SAMPLE_CONVERSATION = """User: We need to decide on the pricing for the enterprise tier.
Assistant: Based on our analysis, I recommend $499/month with a 20% annual discount.
User: That works. Let's go with that. Also, make sure the deployment uses PostgreSQL, not MySQL. We decided that last week.
Assistant: Confirmed. Enterprise tier at $499/month, 20% annual discount, PostgreSQL for the database layer. I'll update the architecture doc."""


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def pipeline(store):
    return IngestionPipeline(store)


@pytest.fixture
def extractor():
    return RuleBasedExtractor()


# ── Raw Extractor Tests ───────────────────────────────────

class TestRuleBasedExtractor:
    def test_extracts_context_node(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION, source_id="test-session")
        assert result.context_node is not None
        assert result.context_node.node_type == NodeType.CONTEXT
        assert result.context_node.source_id == "test-session"

    def test_extracts_entity_nodes(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION)
        entity_labels = {
            n.label.lower()
            for n in result.nodes
            if n.node_type == NodeType.ENTITY
        }
        # Must extract at least: enterprise tier, PostgreSQL, MySQL
        assert any("enterprise" in l for l in entity_labels), \
            f"Expected 'enterprise tier' entity, got: {entity_labels}"
        assert any("postgresql" in l or "postgres" in l for l in entity_labels), \
            f"Expected 'PostgreSQL' entity, got: {entity_labels}"

    def test_extracts_at_least_3_entities(self, extractor):
        """Spec requires at least 3 entity nodes."""
        result = extractor.extract(SAMPLE_CONVERSATION)
        entities = [n for n in result.nodes if n.node_type == NodeType.ENTITY]
        assert len(entities) >= 3, \
            f"Expected >= 3 entities, got {len(entities)}: {[n.label for n in entities]}"

    def test_extracts_decision_nodes(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION)
        decisions = [n for n in result.nodes if n.node_type == NodeType.DECISION]
        assert len(decisions) >= 1, "Expected at least 1 decision node"
        # Decision should reference pricing
        decision_text = " ".join(d.content.lower() for d in decisions)
        assert "499" in decision_text or "pricing" in decision_text, \
            f"Decision should reference $499 pricing, got: {[d.content for d in decisions]}"

    def test_extracts_fact_nodes(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION)
        facts = [n for n in result.nodes if n.node_type == NodeType.FACT]
        assert len(facts) >= 1, "Expected at least 1 fact node"
        # Should capture the PostgreSQL fact
        fact_text = " ".join(f.content.lower() for f in facts)
        has_postgres = "postgresql" in fact_text or "postgres" in fact_text or "database" in fact_text
        has_price = "499" in fact_text
        assert has_postgres or has_price, \
            f"Expected PostgreSQL or pricing fact, got: {[f.content for f in facts]}"

    def test_extracts_topic_nodes(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION)
        topics = [n for n in result.nodes if n.node_type == NodeType.TOPIC]
        assert len(topics) >= 1, \
            f"Expected >= 1 topic, got {len(topics)}: {[n.label for n in topics]}"

    def test_edges_connect_to_context(self, extractor):
        result = extractor.extract(SAMPLE_CONVERSATION)
        context_id = result.context_node.node_id
        non_context_nodes = [n for n in result.nodes if n.node_type != NodeType.CONTEXT]
        # Every non-context node should have at least one edge to the context node
        for node in non_context_nodes:
            edges_to_context = [
                e for e in result.edges
                if (e.source_node_id == node.node_id and e.target_node_id == context_id)
                or (e.source_node_id == context_id and e.target_node_id == node.node_id)
            ]
            assert len(edges_to_context) >= 1, \
                f"Node '{node.label}' ({node.node_type}) has no edge to context"


# ── Pipeline Integration Tests ────────────────────────────

class TestIngestionPipeline:
    def test_ingest_sample_conversation(self, pipeline, store):
        result = pipeline.ingest(IngestionInput(
            source_id="test-session-1",
            content=SAMPLE_CONVERSATION,
            content_type="conversation",
        ))
        assert result.nodes_created > 0
        assert result.edges_created > 0
        assert result.total_nodes_in_graph > 0

    def test_ingest_creates_correct_node_types(self, pipeline, store):
        pipeline.ingest(IngestionInput(
            source_id="test-session-1",
            content=SAMPLE_CONVERSATION,
        ))
        # Check we have a mix of node types
        all_nodes = store.list_nodes(limit=999)
        types_found = {n.node_type for n in all_nodes}
        assert NodeType.CONTEXT in types_found, "Missing context node"
        assert NodeType.ENTITY in types_found, "Missing entity nodes"
        assert NodeType.DECISION in types_found, "Missing decision nodes"

    def test_dedup_prevents_duplicates(self, pipeline, store):
        """Second ingestion of same content should not create duplicate nodes."""
        result1 = pipeline.ingest(IngestionInput(
            source_id="test-session-1",
            content=SAMPLE_CONVERSATION,
        ))
        nodes_after_first = result1.total_nodes_in_graph

        result2 = pipeline.ingest(IngestionInput(
            source_id="test-session-2",
            content=SAMPLE_CONVERSATION,
        ))

        # Second ingestion should create 1 new context node (contexts are always unique)
        # but entities, decisions, facts should be deduplicated
        assert result2.nodes_deduplicated > 0, \
            "Expected some nodes to be deduplicated on second ingestion"

        # Total node increase should be much less than first ingestion
        new_nodes = result2.total_nodes_in_graph - nodes_after_first
        assert new_nodes < result1.nodes_created, \
            f"Dedup failed: second ingestion created {new_nodes} nodes vs {result1.nodes_created} first time"

    def test_dedup_increments_access_count(self, pipeline, store):
        """Deduplicated nodes should have access_count incremented."""
        pipeline.ingest(IngestionInput(
            source_id="test-1",
            content=SAMPLE_CONVERSATION,
        ))

        # Find an entity that should exist
        ops = GraphOperations(store)
        # Look for PostgreSQL entity
        all_nodes = store.list_nodes(node_type=NodeType.ENTITY, limit=999)
        pg_node = None
        for n in all_nodes:
            if "postgres" in n.label.lower():
                pg_node = n
                break

        if pg_node:
            original_count = pg_node.access_count

            # Ingest again
            pipeline.ingest(IngestionInput(
                source_id="test-2",
                content=SAMPLE_CONVERSATION,
            ))

            # Check access count increased
            updated = store.get_node(pg_node.node_id)
            assert updated.access_count > original_count, \
                f"Access count not incremented: {updated.access_count} <= {original_count}"

    def test_decision_contains_full_pricing(self, pipeline, store):
        """Decision node should contain the full pricing decision text."""
        pipeline.ingest(IngestionInput(
            source_id="test-1",
            content=SAMPLE_CONVERSATION,
        ))
        decisions = store.list_nodes(node_type=NodeType.DECISION, limit=999)
        assert len(decisions) >= 1, "No decision nodes found"

        # At least one decision should mention $499
        decision_contents = [d.content for d in decisions]
        has_pricing = any("499" in c for c in decision_contents)
        assert has_pricing, \
            f"No decision mentions $499: {decision_contents}"

    def test_postgresql_fact_captured(self, pipeline, store):
        """Fact or entity node should capture PostgreSQL requirement."""
        pipeline.ingest(IngestionInput(
            source_id="test-1",
            content=SAMPLE_CONVERSATION,
        ))
        all_nodes = store.list_nodes(limit=999)
        pg_nodes = [
            n for n in all_nodes
            if "postgresql" in n.label.lower() or "postgres" in n.label.lower()
            or "postgresql" in n.content.lower()
        ]
        assert len(pg_nodes) >= 1, \
            "No node captures PostgreSQL requirement"

    def test_enterprise_tier_entity_exists(self, pipeline, store):
        """Entity node for 'enterprise tier' should exist."""
        pipeline.ingest(IngestionInput(
            source_id="test-1",
            content=SAMPLE_CONVERSATION,
        ))
        entities = store.list_nodes(node_type=NodeType.ENTITY, limit=999)
        enterprise_nodes = [
            n for n in entities
            if "enterprise" in n.label.lower()
        ]
        assert len(enterprise_nodes) >= 1, \
            f"No 'enterprise tier' entity: {[n.label for n in entities]}"

    def test_edges_stored_in_graph(self, pipeline, store):
        """Ingested edges should be persisted in the store."""
        result = pipeline.ingest(IngestionInput(
            source_id="test-1",
            content=SAMPLE_CONVERSATION,
        ))
        assert result.edges_created > 0
        assert store.count_edges() > 0

        # Context node should have edges
        context = store.get_node(result.context_node_id)
        assert context is not None
        edges = store.get_edges_for_node(context.node_id)
        assert len(edges) > 0, "Context node has no edges"


# ── Edge Cases ────────────────────────────────────────────

class TestIngestionEdgeCases:
    def test_empty_content(self, pipeline):
        result = pipeline.ingest(IngestionInput(
            source_id="empty",
            content="",
        ))
        # Should still create a context node at minimum
        assert result.nodes_created >= 1

    def test_short_content(self, pipeline):
        result = pipeline.ingest(IngestionInput(
            source_id="short",
            content="Hello, how are you?",
        ))
        assert result.nodes_created >= 1

    def test_code_content(self, pipeline):
        code = """User: Can you write a function to parse JSON?
Assistant: Sure, here's a Python function:
def parse_json(data):
    import json
    return json.loads(data)
User: Let's go with that approach. Always use json.loads, never eval."""
        result = pipeline.ingest(IngestionInput(
            source_id="code-session",
            content=code,
            content_type="code",
        ))
        assert result.nodes_created >= 1
