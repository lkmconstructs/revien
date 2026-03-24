"""
Test 5: Claude Code Adapter (and other adapters)
Create a mock Claude Code JSONL session log.
Point the Claude Code adapter at it.
Verify the adapter reads the log and produces valid content for ingestion.
Verify the ingested content produces correct graph nodes.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.file_watcher import FileWatcherAdapter
from revien.adapters.generic_api import GenericAPIAdapter
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


# ── Mock Data ─────────────────────────────────────────────

MOCK_CLAUDE_CODE_SESSION = [
    {
        "type": "human",
        "content": "We need to set up the authentication system. Let's use JWT tokens with a 24-hour expiry.",
        "timestamp": "2026-03-24T10:00:00Z"
    },
    {
        "type": "assistant",
        "content": "I'll set up JWT authentication with 24-hour token expiry. I'll use the PyJWT library and store refresh tokens in Redis.",
        "timestamp": "2026-03-24T10:00:15Z"
    },
    {
        "type": "tool_use",
        "content": [{"type": "tool_use", "name": "write_file", "input": {"path": "auth.py"}}],
        "timestamp": "2026-03-24T10:00:20Z"
    },
    {
        "type": "tool_result",
        "content": "File written successfully.",
        "timestamp": "2026-03-24T10:00:21Z"
    },
    {
        "type": "human",
        "content": "Perfect. We decided to use PostgreSQL for the user database, not SQLite. Make sure the connection pool is set to 20.",
        "timestamp": "2026-03-24T10:01:00Z"
    },
    {
        "type": "assistant",
        "content": "Confirmed. I'll configure PostgreSQL with a connection pool of 20. I prefer using asyncpg over psycopg2 for async support.",
        "timestamp": "2026-03-24T10:01:15Z"
    },
    {
        "type": "human",
        "content": [
            {"type": "text", "text": "Let's go with asyncpg then. Also deploy this to the staging server at 192.168.1.50."}
        ],
        "timestamp": "2026-03-24T10:02:00Z"
    },
    {
        "type": "assistant",
        "content": [
            {"type": "text", "text": "Got it. Going with asyncpg for PostgreSQL, deploying to staging at 192.168.1.50. I'll update the deployment config."}
        ],
        "timestamp": "2026-03-24T10:02:15Z"
    }
]

MOCK_FILE_CONTENT = """# Meeting Notes - Sprint 14

## Decisions
- We decided to migrate from REST to GraphQL for the public API
- Frontend team will use Apollo Client
- Backend uses Strawberry (Python GraphQL library)

## Action Items
- Jesse: Set up GraphQL schema by Friday
- Alice: Update API documentation
- Deploy to staging by end of week
"""


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_claude_dir():
    """Create a mock Claude Code session directory with JSONL log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure: projects/my-project/conversations/
        project_dir = Path(tmpdir) / "projects" / "my-project" / "conversations"
        project_dir.mkdir(parents=True)

        # Write mock JSONL session log
        session_file = project_dir / "session-001.jsonl"
        with open(session_file, "w") as f:
            for msg in MOCK_CLAUDE_CODE_SESSION:
                f.write(json.dumps(msg) + "\n")

        yield tmpdir


@pytest.fixture
def mock_watch_dir():
    """Create a mock watch directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a markdown file
        md_file = Path(tmpdir) / "meeting-notes.md"
        md_file.write_text(MOCK_FILE_CONTENT)

        # Write a JSON conversation file
        json_file = Path(tmpdir) / "conversation.json"
        json_file.write_text(json.dumps({
            "conversations": [
                {"role": "user", "content": "We should use Docker for deployment."},
                {"role": "assistant", "content": "Agreed. I'll create a Dockerfile."},
            ]
        }))

        # Write a JSONL file
        jsonl_file = Path(tmpdir) / "chat.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"role": "user", "content": "Set the port to 8080."}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "Port configured to 8080."}) + "\n")

        yield tmpdir


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


# ── Claude Code Adapter Tests ─────────────────────────────

class TestClaudeCodeAdapter:
    def test_reads_jsonl_session_log(self, mock_claude_dir):
        """Adapter should read the JSONL file and produce content."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        assert len(results) >= 1, "Should find at least one session"

    def test_extracts_human_assistant_messages(self, mock_claude_dir):
        """Should extract human and assistant messages, skip tool_use/tool_result."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))

        content = results[0]["content"]
        assert "User:" in content, "Should have User: prefixed messages"
        assert "Assistant:" in content, "Should have Assistant: prefixed messages"
        # Should NOT include tool_use noise
        assert "write_file" not in content, "Should skip tool_use messages"
        assert "File written successfully" not in content, "Should skip tool_result"

    def test_handles_content_blocks(self, mock_claude_dir):
        """Should handle both string content and content block arrays."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))

        content = results[0]["content"]
        # Message 7 uses content blocks format
        assert "asyncpg" in content, "Should extract text from content blocks"
        assert "192.168.1.50" in content, "Should extract staging server IP"

    def test_produces_valid_ingestion_content(self, mock_claude_dir):
        """Adapter output should have all required fields for ingestion."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))

        for result in results:
            assert "content" in result
            assert "content_type" in result
            assert "timestamp" in result
            assert "metadata" in result
            assert result["content_type"] == "conversation"
            assert result["metadata"]["adapter"] == "claude_code"

    def test_ingested_content_creates_correct_nodes(self, mock_claude_dir, store):
        """Ingesting adapter output should produce correct graph nodes."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))

        pipeline = IngestionPipeline(store)
        for result in results:
            pipeline.ingest(IngestionInput(
                source_id=result.get("source_id", "claude-code-test"),
                content=result["content"],
                content_type=result["content_type"],
                metadata=result.get("metadata", {}),
            ))

        # Verify nodes were created
        all_nodes = store.list_nodes(limit=999)
        assert len(all_nodes) > 0, "Should create nodes from adapter content"

        # Check for expected entities
        entity_labels = {n.label.lower() for n in all_nodes if n.node_type == NodeType.ENTITY}
        has_postgres = any("postgres" in l for l in entity_labels)
        has_jwt = any("jwt" in l for l in entity_labels)
        assert has_postgres or has_jwt, \
            f"Should extract PostgreSQL or JWT entity, got: {entity_labels}"

        # Check for decisions
        decisions = [n for n in all_nodes if n.node_type == NodeType.DECISION]
        assert len(decisions) >= 1, "Should extract at least one decision"

    def test_since_filter_works(self, mock_claude_dir):
        """Content from before 'since' timestamp should be excluded."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        # Use a future date — nothing should match
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(future))
        assert len(results) == 0, "Future since date should return no results"

    def test_health_check_valid_dir(self, mock_claude_dir):
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        assert run_async(adapter.health_check()) is True

    def test_health_check_invalid_dir(self):
        adapter = ClaudeCodeAdapter(session_dir="/nonexistent/path")
        assert run_async(adapter.health_check()) is False

    def test_extracts_project_name(self, mock_claude_dir):
        """Should extract project name from directory structure."""
        adapter = ClaudeCodeAdapter(session_dir=mock_claude_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        if results:
            assert "my-project" in results[0].get("source_id", ""), \
                "Source ID should include project name"


# ── File Watcher Adapter Tests ────────────────────────────

class TestFileWatcherAdapter:
    def test_reads_markdown_files(self, mock_watch_dir):
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        md_results = [r for r in results if r["metadata"]["filename"] == "meeting-notes.md"]
        assert len(md_results) == 1
        assert "GraphQL" in md_results[0]["content"]

    def test_reads_jsonl_files(self, mock_watch_dir):
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        jsonl_results = [r for r in results if r["metadata"]["filename"] == "chat.jsonl"]
        assert len(jsonl_results) == 1
        assert "8080" in jsonl_results[0]["content"]

    def test_reads_json_files(self, mock_watch_dir):
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        json_results = [r for r in results if r["metadata"]["filename"] == "conversation.json"]
        assert len(json_results) == 1
        assert "Docker" in json_results[0]["content"]

    def test_since_filter(self, mock_watch_dir):
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(future))
        assert len(results) == 0

    def test_health_check(self, mock_watch_dir):
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        assert run_async(adapter.health_check()) is True

    def test_health_check_nonexistent(self):
        adapter = FileWatcherAdapter(watch_dir="/nonexistent")
        assert run_async(adapter.health_check()) is False

    def test_ignores_unsupported_extensions(self, mock_watch_dir):
        """Should not read .py, .jpg, etc."""
        Path(mock_watch_dir, "script.py").write_text("print('hello')")
        adapter = FileWatcherAdapter(watch_dir=mock_watch_dir)
        since = datetime(2020, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(since))
        filenames = [r["metadata"]["filename"] for r in results]
        assert "script.py" not in filenames


# ── Generic API Adapter Tests ─────────────────────────────

class TestGenericAPIAdapter:
    def test_default_parser_list_format(self):
        """Default parser should handle list-of-dicts format."""
        adapter = GenericAPIAdapter(url="http://localhost:9999")
        data = {
            "conversations": [
                {"content": "Hello from API", "timestamp": "2026-01-01T00:00:00Z"},
                {"content": "Second message", "timestamp": "2026-01-01T01:00:00Z"},
            ]
        }
        results = adapter._default_parser(data)
        assert len(results) == 2
        assert results[0]["content"] == "Hello from API"

    def test_default_parser_direct_list(self):
        """Should handle direct list format."""
        adapter = GenericAPIAdapter(url="http://localhost:9999")
        data = [
            {"content": "Message 1"},
            {"text": "Message 2"},
        ]
        results = adapter._default_parser(data)
        assert len(results) == 2

    def test_default_parser_string_items(self):
        """Should handle list of plain strings."""
        adapter = GenericAPIAdapter(url="http://localhost:9999")
        data = ["First conversation", "Second conversation"]
        results = adapter._default_parser(data)
        assert len(results) == 2
        assert results[0]["content"] == "First conversation"

    def test_health_check_unreachable(self):
        """Health check should return False for unreachable URLs."""
        adapter = GenericAPIAdapter(url="http://localhost:99999")
        assert run_async(adapter.health_check()) is False
