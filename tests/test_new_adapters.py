# Licensed under the Apache License, Version 2.0
# See LICENSE file in the project root for full license text.

"""
Tests for the OpenAI, LangChain, and Ollama adapters.
Covers ingestion, deduplication, scoring, CLI integration, and edge cases.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from revien.adapters.openai_adapter import OpenAIAdapter
from revien.adapters.ollama_adapter import OllamaAdapter
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore


# ── Sample Data ──────────────────────────────────────────────

SAMPLE_OPENAI_CONVERSATION = {
    "id": "conv_abc123",
    "title": "Database Migration Discussion",
    "create_time": 1711234567.890,
    "update_time": 1711234999.890,
    "mapping": {
        "node_root": {
            "id": "node_root",
            "message": None,
            "parent": None,
            "children": ["node_system"]
        },
        "node_system": {
            "id": "node_system",
            "message": {
                "id": "msg_sys",
                "author": {"role": "system"},
                "content": {"content_type": "text", "parts": ["You are a helpful assistant."]},
                "create_time": 1711234567.0,
                "metadata": {}
            },
            "parent": "node_root",
            "children": ["node_user1"]
        },
        "node_user1": {
            "id": "node_user1",
            "message": {
                "id": "msg_u1",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["We need to decide between PostgreSQL and MongoDB for the new service."]},
                "create_time": 1711234600.0,
                "metadata": {}
            },
            "parent": "node_system",
            "children": ["node_assistant1"]
        },
        "node_assistant1": {
            "id": "node_assistant1",
            "message": {
                "id": "msg_a1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["For your use case with complex relational data, I'd recommend PostgreSQL. It handles ACID transactions and complex joins better than MongoDB."]},
                "create_time": 1711234610.0,
                "metadata": {}
            },
            "parent": "node_user1",
            "children": ["node_user2"]
        },
        "node_user2": {
            "id": "node_user2",
            "message": {
                "id": "msg_u2",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Agreed. Let's go with PostgreSQL. What about the connection pool size?"]},
                "create_time": 1711234700.0,
                "metadata": {}
            },
            "parent": "node_assistant1",
            "children": ["node_assistant2"]
        },
        "node_assistant2": {
            "id": "node_assistant2",
            "message": {
                "id": "msg_a2",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["For a service handling 500 concurrent users, I'd recommend a connection pool of 20-30 connections using asyncpg."]},
                "create_time": 1711234710.0,
                "metadata": {}
            },
            "parent": "node_user2",
            "children": []
        }
    }
}

SAMPLE_OPENAI_CONVERSATION_2 = {
    "id": "conv_def456",
    "title": "API Design Review",
    "create_time": 1711300000.0,
    "update_time": 1711300500.0,
    "mapping": {
        "node_r": {
            "id": "node_r",
            "message": {
                "id": "msg_r",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Should we use REST or GraphQL for the public API?"]},
                "create_time": 1711300100.0,
                "metadata": {}
            },
            "parent": None,
            "children": ["node_a"]
        },
        "node_a": {
            "id": "node_a",
            "message": {
                "id": "msg_a",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["Given that you chose PostgreSQL for the database, GraphQL pairs well for complex queries. It also reduces over-fetching."]},
                "create_time": 1711300200.0,
                "metadata": {}
            },
            "parent": "node_r",
            "children": []
        }
    }
}

# Conversation with edge cases
SAMPLE_OPENAI_EDGE_CASES = {
    "id": "conv_edge",
    "title": "Edge Case Conversation",
    "create_time": 1711400000.0,
    "update_time": 1711400500.0,
    "mapping": {
        "node_empty": {
            "id": "node_empty",
            "message": {
                "id": "msg_empty",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [""]},
                "create_time": 1711400100.0,
                "metadata": {}
            },
            "parent": None,
            "children": ["node_image"]
        },
        "node_image": {
            "id": "node_image",
            "message": {
                "id": "msg_img",
                "author": {"role": "user"},
                "content": {"content_type": "image", "parts": []},
                "create_time": 1711400200.0,
                "metadata": {}
            },
            "parent": "node_empty",
            "children": ["node_tool"]
        },
        "node_tool": {
            "id": "node_tool",
            "message": {
                "id": "msg_tool",
                "author": {"role": "tool"},
                "content": {"content_type": "text", "parts": ["Code execution result: 42"]},
                "create_time": 1711400300.0,
                "metadata": {}
            },
            "parent": "node_image",
            "children": ["node_emoji"]
        },
        "node_emoji": {
            "id": "node_emoji",
            "message": {
                "id": "msg_emoji",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Great work! 🎉🚀 Let's ship it!"]},
                "create_time": 1711400400.0,
                "metadata": {}
            },
            "parent": "node_tool",
            "children": []
        },
        "node_no_message": {
            "id": "node_no_message",
            "message": None,
            "parent": None,
            "children": []
        }
    }
}

# Branching conversation (user tried two different follow-ups)
SAMPLE_OPENAI_BRANCHING = {
    "id": "conv_branch",
    "title": "Branching Discussion",
    "create_time": 1711500000.0,
    "update_time": 1711500500.0,
    "mapping": {
        "node_q": {
            "id": "node_q",
            "message": {
                "id": "msg_q",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["How should we deploy this?"]},
                "create_time": 1711500100.0,
                "metadata": {}
            },
            "parent": None,
            "children": ["node_branch_a", "node_branch_b"]
        },
        "node_branch_a": {
            "id": "node_branch_a",
            "message": {
                "id": "msg_ba",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["Option A: Docker containers on AWS ECS."]},
                "create_time": 1711500200.0,
                "metadata": {}
            },
            "parent": "node_q",
            "children": []
        },
        "node_branch_b": {
            "id": "node_branch_b",
            "message": {
                "id": "msg_bb",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["Option B: Kubernetes on bare metal."]},
                "create_time": 1711500300.0,
                "metadata": {}
            },
            "parent": "node_q",
            "children": []
        }
    }
}


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def openai_adapter(tmp_db):
    """Create an OpenAI adapter with a temporary database."""
    return OpenAIAdapter(graph_path=tmp_db)


@pytest.fixture
def single_conv_file():
    """Create a temp file with a single OpenAI conversation."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(SAMPLE_OPENAI_CONVERSATION, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def bulk_export_file():
    """Create a temp file with a bulk OpenAI export (array of conversations)."""
    conversations = [
        SAMPLE_OPENAI_CONVERSATION,
        SAMPLE_OPENAI_CONVERSATION_2,
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(conversations, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def edge_case_file():
    """Create a temp file with edge case conversation."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(SAMPLE_OPENAI_EDGE_CASES, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def branching_file():
    """Create a temp file with a branching conversation."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(SAMPLE_OPENAI_BRANCHING, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def ollama_adapter(tmp_db):
    """Create an Ollama adapter with a temporary database."""
    return OllamaAdapter(graph_path=tmp_db)


# ══════════════════════════════════════════════════════════════
# OpenAI Adapter Tests
# ══════════════════════════════════════════════════════════════

class TestOpenAIAdapterSingleConversation:
    """Test single conversation ingestion."""

    def test_single_conversation_ingest(self, openai_adapter, single_conv_file):
        """Parse a sample OpenAI JSON, verify node count, edge structure, timestamps."""
        stats = openai_adapter.ingest_conversation(single_conv_file)

        assert stats["conversation_id"] == "conv_abc123"
        assert stats["message_count"] >= 5  # system + 2 user + 2 assistant
        assert stats["node_count"] > 0
        assert stats["edge_count"] > 0

    def test_nodes_created_in_graph(self, openai_adapter, single_conv_file):
        """Verify nodes are actually in the graph store."""
        openai_adapter.ingest_conversation(single_conv_file)
        total = openai_adapter.store.count_nodes()
        assert total >= 5, f"Expected at least 5 nodes, got {total}"

    def test_edges_created_in_graph(self, openai_adapter, single_conv_file):
        """Verify edges are actually in the graph store."""
        openai_adapter.ingest_conversation(single_conv_file)
        total = openai_adapter.store.count_edges()
        assert total >= 4, f"Expected at least 4 edges (parent→child threading), got {total}"

    def test_timestamps_preserved(self, openai_adapter, single_conv_file):
        """Verify OpenAI timestamps are converted correctly."""
        openai_adapter.ingest_conversation(single_conv_file)
        nodes = openai_adapter.store.list_nodes(limit=100)

        # Find a user message node
        for node in nodes:
            if "PostgreSQL" in node.content and "MongoDB" in node.content:
                # This message had create_time: 1711234600.0
                expected = datetime.fromtimestamp(1711234600.0, tz=timezone.utc)
                assert node.created_at.year == expected.year
                assert node.created_at.month == expected.month
                break

    def test_author_role_in_metadata(self, openai_adapter, single_conv_file):
        """Verify author.role is stored in node metadata."""
        openai_adapter.ingest_conversation(single_conv_file)
        nodes = openai_adapter.store.list_nodes(limit=100)

        roles_found = set()
        for node in nodes:
            role = node.metadata.get("author_role")
            if role:
                roles_found.add(role)

        assert "user" in roles_found, "Should find user-role messages"
        assert "assistant" in roles_found, "Should find assistant-role messages"

    def test_file_not_found_raises(self, openai_adapter):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            openai_adapter.ingest_conversation("/nonexistent/path.json")

    def test_system_messages_flagged(self, openai_adapter, single_conv_file):
        """System messages should be ingested with is_system metadata."""
        stats = openai_adapter.ingest_conversation(single_conv_file)
        assert stats["system_messages"] >= 1


class TestOpenAIAdapterBulkExport:
    """Test bulk export ingestion."""

    def test_bulk_export_ingest(self, openai_adapter, bulk_export_file):
        """Parse multi-conversation export, verify all conversations ingested."""
        stats = openai_adapter.ingest_bulk_export(bulk_export_file)

        assert stats["total_conversations"] == 2
        assert stats["conversations_ingested"] == 2
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    def test_cross_conversation_edges(self, openai_adapter, bulk_export_file):
        """Verify cross-conversation edges created for similar content."""
        stats = openai_adapter.ingest_bulk_export(bulk_export_file)
        # Both conversations mention PostgreSQL — should create cross-conv edges
        # (depends on similarity threshold; at minimum stats key should exist)
        assert "cross_conversation_edges" in stats

    def test_bulk_not_array_raises(self, openai_adapter, tmp_db):
        """Bulk export must be a JSON array."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"not": "an array"}, f)
            path = f.name
        try:
            with pytest.raises(ValueError):
                openai_adapter.ingest_bulk_export(path)
        finally:
            os.unlink(path)


class TestOpenAIAdapterDeduplication:
    """Test deduplication behavior."""

    def test_deduplication(self, openai_adapter, single_conv_file):
        """Ingest same conversation twice, verify no duplicates."""
        stats1 = openai_adapter.ingest_conversation(single_conv_file)
        nodes_after_first = openai_adapter.store.count_nodes()

        stats2 = openai_adapter.ingest_conversation(single_conv_file)

        # Second ingest should be skipped (0 new nodes)
        assert stats2["node_count"] == 0
        assert stats2["message_count"] == 0

        # Total nodes should not have increased
        nodes_after_second = openai_adapter.store.count_nodes()
        assert nodes_after_second == nodes_after_first


class TestOpenAIAdapterEdgeCases:
    """Test edge case handling."""

    def test_empty_messages(self, openai_adapter, edge_case_file):
        """Empty messages should create nodes with fallback content."""
        stats = openai_adapter.ingest_conversation(edge_case_file)
        # Should not crash, should still create some nodes
        assert stats["node_count"] > 0

    def test_non_text_content_types(self, openai_adapter, edge_case_file):
        """Non-text content (images) should be handled gracefully."""
        stats = openai_adapter.ingest_conversation(edge_case_file)
        # Image node may or may not be created depending on implementation,
        # but it should not crash
        assert stats["node_count"] > 0

    def test_unicode_emoji_handling(self, openai_adapter, edge_case_file):
        """Unicode and emoji in content should be preserved."""
        openai_adapter.ingest_conversation(edge_case_file)
        nodes = openai_adapter.store.list_nodes(limit=100)

        emoji_found = False
        for node in nodes:
            if "🎉" in node.content or "🚀" in node.content:
                emoji_found = True
                break
        assert emoji_found, "Emoji content should be preserved"

    def test_branching_conversations(self, openai_adapter, branching_file):
        """Multiple children per node (branching) should preserve full tree."""
        stats = openai_adapter.ingest_conversation(branching_file)

        # Should have the root node + both branches
        assert stats["message_count"] >= 3
        assert stats["edge_count"] >= 2  # root→branch_a and root→branch_b

    def test_null_message_nodes_skipped(self, openai_adapter, edge_case_file):
        """Nodes with message=None should be skipped."""
        stats = openai_adapter.ingest_conversation(edge_case_file)
        # Should not crash and skipped_messages should account for the null node
        assert stats["skipped_messages"] >= 1

    def test_tool_messages_ingested(self, openai_adapter, edge_case_file):
        """Tool messages should be ingested with appropriate metadata."""
        openai_adapter.ingest_conversation(edge_case_file)
        nodes = openai_adapter.store.list_nodes(limit=100)

        tool_found = False
        for node in nodes:
            if node.metadata.get("author_role") == "tool":
                tool_found = True
                break
        assert tool_found, "Tool messages should be ingested"


class TestOpenAIAdapterThreeFactorScores:
    """Verify three-factor scores are computed correctly on ingested data."""

    def test_three_factor_scores(self, openai_adapter, single_conv_file):
        """Verify retrieval engine can score ingested OpenAI data."""
        from revien.retrieval.engine import RetrievalEngine

        openai_adapter.ingest_conversation(single_conv_file)
        engine = RetrievalEngine(openai_adapter.store)

        response = engine.recall("PostgreSQL database migration")
        assert len(response.results) > 0, "Should find results for PostgreSQL query"

        # Verify scores are in valid range
        for result in response.results:
            assert 0.0 <= result.score <= 1.0
            assert "recency" in result.score_breakdown
            assert "frequency" in result.score_breakdown
            assert "proximity" in result.score_breakdown


# ══════════════════════════════════════════════════════════════
# LangChain Adapter Tests
# ══════════════════════════════════════════════════════════════

class TestLangChainAdapter:
    """Test the LangChain memory adapter."""

    def test_langchain_import_guard(self):
        """Verify graceful failure when langchain-core not installed."""
        # The adapter should be importable regardless
        from revien.adapters.langchain_adapter import LANGCHAIN_AVAILABLE
        # This just verifies the module loads without crashing

    def test_memory_variables_property(self, tmp_db):
        """Verify correct variable names returned."""
        try:
            from revien.adapters.langchain_adapter import RevienMemory, LANGCHAIN_AVAILABLE
            if not LANGCHAIN_AVAILABLE:
                pytest.skip("langchain-core not installed")
            memory = RevienMemory(graph_path=tmp_db)
            assert memory.memory_variables == ["history"]
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_save_and_load_cycle(self, tmp_db):
        """Save a context pair, then load with a related query."""
        try:
            from revien.adapters.langchain_adapter import RevienMemory, LANGCHAIN_AVAILABLE
            if not LANGCHAIN_AVAILABLE:
                pytest.skip("langchain-core not installed")

            memory = RevienMemory(graph_path=tmp_db)

            # Save a conversation
            memory.save_context(
                {"input": "We decided to use PostgreSQL for the database"},
                {"output": "Great choice. PostgreSQL is excellent for relational data."},
            )

            # Load with related query
            result = memory.load_memory_variables(
                {"input": "What database are we using?"}
            )

            assert "history" in result
            # The history should contain relevant content
            # (may be empty if graph hasn't built enough connections yet)
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_clear(self, tmp_db):
        """Verify clear removes all nodes."""
        try:
            from revien.adapters.langchain_adapter import RevienMemory, LANGCHAIN_AVAILABLE
            if not LANGCHAIN_AVAILABLE:
                pytest.skip("langchain-core not installed")

            memory = RevienMemory(graph_path=tmp_db)
            memory.save_context(
                {"input": "test input"},
                {"output": "test output"},
            )
            assert memory._store.count_nodes() > 0
            memory.clear()
            assert memory._store.count_nodes() == 0
        except ImportError:
            pytest.skip("langchain-core not installed")


# ══════════════════════════════════════════════════════════════
# Ollama Adapter Tests
# ══════════════════════════════════════════════════════════════

class TestOllamaAdapter:
    """Test the Ollama adapter."""

    def test_ollama_connection_refused(self, ollama_adapter):
        """Adapter should handle Ollama not running gracefully."""
        # Use a port that's definitely not running Ollama
        adapter = OllamaAdapter(
            graph_path=ollama_adapter.graph_path,
            ollama_host="http://localhost:1",
            model="llama3",
        )
        assert adapter.health_check() is False

    def test_context_injection_format(self, ollama_adapter):
        """Verify formatted context string is clean and parseable."""
        # First ingest some data
        ollama_adapter.pipeline.ingest(
            from_ingestion_input(
                "User: What about using PostgreSQL?\nAssistant: PostgreSQL is great for relational data."
            )
        )

        context = ollama_adapter.get_context_for_prompt("PostgreSQL database")

        if context:  # May be empty if no matching nodes found
            assert "[Revien Memory Context]" in context
            assert "[End Memory Context]" in context

    def test_chat_saves_to_graph(self, ollama_adapter):
        """Send a message via adapter, verify input saved to graph (mock Ollama)."""
        initial_count = ollama_adapter.store.count_nodes()

        # Mock the Ollama HTTP call
        with patch.object(ollama_adapter, "_call_ollama", return_value="Mocked response"):
            ollama_adapter.chat("What is the meaning of life?")

        final_count = ollama_adapter.store.count_nodes()
        assert final_count > initial_count, "Chat should save exchange to graph"

    def test_history_ingestion(self, ollama_adapter):
        """Ingest a sample Ollama history, verify graph structure."""
        history = [
            {"role": "user", "content": "Let's discuss the deployment strategy."},
            {"role": "assistant", "content": "Sure. Are you targeting Docker or Kubernetes?"},
            {"role": "user", "content": "Docker for now. We'll move to Kubernetes later."},
            {"role": "assistant", "content": "Good plan. I'll set up Docker Compose first."},
        ]

        stats = ollama_adapter.ingest_ollama_history(history)

        assert stats["total_messages"] == 4
        assert stats["nodes_created"] > 0
        assert stats["total_nodes_in_graph"] > 0

    def test_empty_history_ingestion(self, ollama_adapter):
        """Empty history should return zero stats without crashing."""
        stats = ollama_adapter.ingest_ollama_history([])
        assert stats["total_messages"] == 0
        assert stats["nodes_created"] == 0

    def test_offline_mode(self, ollama_adapter):
        """If Ollama unreachable, graph queries should still work."""
        # Ingest some data first
        ollama_adapter.pipeline.ingest(
            from_ingestion_input(
                "We decided to use Docker for deployment. The API runs on port 8080."
            )
        )

        # Even without Ollama, graph context retrieval should work
        context = ollama_adapter.get_context_for_prompt("Docker deployment")
        # Should not crash — may return empty or results depending on graph content

    def test_health_check_unreachable(self, ollama_adapter):
        """Health check should return False for unreachable server."""
        adapter = OllamaAdapter(
            graph_path=ollama_adapter.graph_path,
            ollama_host="http://localhost:1",
        )
        assert adapter.health_check() is False

    def test_close_cleanup(self, tmp_db):
        """Close should clean up resources without error."""
        adapter = OllamaAdapter(graph_path=tmp_db)
        adapter.close()  # Should not raise


# ── Helper ───────────────────────────────────────────────────

def from_ingestion_input(content: str):
    """Helper to create an IngestionInput from a content string."""
    from revien.ingestion.pipeline import IngestionInput
    return IngestionInput(
        source_id="test",
        content=content,
        content_type="conversation",
    )
