"""
Test: Context Fence (leg 6c)
Recalled memory that re-enters a prompt must not re-enter the graph as new
memory. Unit-test each marker family the fence strips, then exercise the
REVIEN_FENCE=0 escape hatch and the empty-after-fence skip through the real
pipeline.
"""

import os
import tempfile

import pytest

from revien.graph.store import GraphStore
from revien.ingestion.fence import fence_content
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


# ── Fixtures ──────────────────────────────────────────────

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


# ── fence_content: one marker family at a time ────────────

class TestSystemReminder:
    def test_strips_well_formed_pair(self):
        text = (
            "User: what's on my calendar?\n"
            "<system-reminder>\nInternal harness note, not conversation.\n"
            "</system-reminder>\n"
            "Assistant: You have a 2pm call."
        )
        result = fence_content(text)
        assert "<system-reminder>" not in result.content
        assert "Internal harness note" not in result.content
        assert "You have a 2pm call" in result.content
        assert "what's on my calendar" in result.content
        assert result.stripped_spans == 1
        assert result.stripped_chars > 0
        assert result.markers == ["system_reminder"]

    def test_strips_multiple_spans(self):
        text = (
            "<system-reminder>first</system-reminder>\n"
            "User: hello\n"
            "<system-reminder>second</system-reminder>\n"
            "Assistant: hi"
        )
        result = fence_content(text)
        assert "first" not in result.content
        assert "second" not in result.content
        assert "User: hello" in result.content
        assert "Assistant: hi" in result.content
        assert result.stripped_spans == 2

    def test_unclosed_opening_tag_strips_to_end_of_text(self):
        """The harness always closes its reminders; an opening tag with no
        matching close means the transcript was truncated mid-injection —
        everything from there to EOF is injected content, not conversation.
        Reported under its OWN marker family (`_truncated`), never disguised
        as a tidy pair removal — a whole-tail strip is a much bigger claim."""
        text = (
            "User: hello\n"
            "<system-reminder>\nthis got cut off mid-block and never closed"
        )
        result = fence_content(text)
        assert "User: hello" in result.content
        assert "<system-reminder>" not in result.content
        assert "cut off mid-block" not in result.content
        assert result.stripped_spans == 1
        assert result.markers == ["system_reminder_truncated"]

    def test_mid_line_quoted_literal_survives_intact(self):
        """A bare marker string sitting mid-line — e.g. quoted inside a code
        sample — is not a harness injection (the harness always emits at
        line-start) and must be left completely untouched, not treated as an
        unclosed opener and stripped to EOF."""
        text = (
            "User: here's the guard clause I added:\n"
            "if '<system-reminder>' in text:\n"
            "    strip_it()\n"
            "Assistant: Looks right to me."
        )
        result = fence_content(text)
        assert result.content == text
        assert result.stripped_spans == 0
        assert result.stripped_chars == 0
        assert result.markers == []
        assert "strip_it" in result.content
        assert "Looks right to me" in result.content


class TestMemoryContextFence:
    def test_strips_well_formed_pair(self):
        text = (
            "User: what did we decide about the database?\n"
            "[Revien Memory Context]\n"
            "The following context is retrieved from persistent memory:\n"
            "- [Score: 91%] (2 days ago) Database Choice: PostgreSQL, not MySQL.\n"
            "\n[End Memory Context]\n"
            "Assistant: You decided on PostgreSQL."
        )
        result = fence_content(text)
        assert "[Revien Memory Context]" not in result.content
        assert "PostgreSQL, not MySQL" not in result.content
        assert "what did we decide" in result.content
        assert "You decided on PostgreSQL" in result.content
        assert result.markers == ["memory_context"]

    def test_unclosed_opening_strips_to_end_of_text(self):
        text = (
            "User: hi\n"
            "[Revien Memory Context]\nsome recalled content that got truncated"
        )
        result = fence_content(text)
        assert "User: hi" in result.content
        assert "[Revien Memory Context]" not in result.content
        assert "truncated" not in result.content
        assert result.markers == ["memory_context_truncated"]

    def test_mid_line_quoted_literal_survives_intact(self):
        """Same line-anchoring guard as system-reminder — this marker family
        shares the _strip_paired helper."""
        text = (
            "User: the fence looks for the literal string "
            "'[Revien Memory Context]' in the input.\n"
            "Assistant: right, that's the ollama_adapter fence marker."
        )
        result = fence_content(text)
        assert result.content == text
        assert result.stripped_spans == 0
        assert result.markers == []


class TestHermesHeader:
    def test_strips_header_and_list_lines(self):
        text = (
            "User: remind me what I said\n"
            "## Relevant memory (Revien)\n"
            "- Prefers PostgreSQL over MySQL\n"
            "- Deploys to staging at 192.168.1.50\n"
            "Assistant: You prefer PostgreSQL."
        )
        result = fence_content(text)
        assert "## Relevant memory (Revien)" not in result.content
        assert "Prefers PostgreSQL over MySQL" not in result.content
        assert "192.168.1.50" not in result.content
        assert "remind me what I said" in result.content
        assert "You prefer PostgreSQL" in result.content
        assert result.markers == ["hermes_header"]

    def test_stops_at_first_non_list_line(self):
        text = (
            "## Relevant memory (Revien)\n"
            "- item one\n"
            "Not a list line, real conversation resumes here.\n"
        )
        result = fence_content(text)
        assert "item one" not in result.content
        assert "Not a list line, real conversation resumes here." in result.content


class TestLangchainContextBlock:
    def test_strips_block_through_results_to_next_h2(self):
        text = (
            "## Relevant Context (from 2 nodes)\n"
            "\n"
            "### Result 1: Database Choice\n"
            "Type: decision | Score: 0.912\n"
            "\nPostgreSQL, not MySQL.\n"
            "\n"
            "### Result 2: Deploy Target\n"
            "Type: fact | Score: 0.845\n"
            "\nStaging at 192.168.1.50.\n"
            "\n[Retrieved in 12.34ms]\n"
            "## Actual next section\n"
            "This is real content that must survive."
        )
        result = fence_content(text)
        assert "Relevant Context (from" not in result.content
        assert "Database Choice" not in result.content
        assert "192.168.1.50" not in result.content
        assert "Retrieved in 12.34ms" not in result.content
        assert "## Actual next section" in result.content
        assert "This is real content that must survive." in result.content
        assert result.markers == ["langchain_context"]

    def test_runs_to_end_of_text_when_no_next_h2(self):
        text = (
            "Some real conversation.\n"
            "## Relevant Context (from 1 nodes)\n"
            "### Result 1: Thing\n"
            "Type: entity | Score: 0.5\n"
        )
        result = fence_content(text)
        assert "Some real conversation." in result.content
        assert "Relevant Context" not in result.content
        assert "Result 1: Thing" not in result.content


# ── Cross-cutting behavior ────────────────────────────────

class TestFenceGeneral:
    def test_noop_on_clean_text(self):
        text = "User: hello there\nAssistant: hi, how can I help?"
        result = fence_content(text)
        assert result.content == text
        assert result.stripped_spans == 0
        assert result.stripped_chars == 0
        assert result.markers == []

    def test_noop_preserves_newline_runs_on_marker_free_content(self):
        """The newline collapse is cleanup for holes STRIPPING leaves behind
        — it must never fire on marker-free content just because that
        content happens to contain its own 3+ newline run. Fence-on must be
        byte-identical to fence-off for content with nothing to strip."""
        text = "before\n\n\n\n\nafter — no markers anywhere in this text"
        result = fence_content(text)
        assert result.content == text
        assert result.stripped_spans == 0
        assert result.stripped_chars == 0
        assert result.markers == []

    def test_idempotent(self):
        text = (
            "User: question\n"
            "<system-reminder>noise</system-reminder>\n"
            "[Revien Memory Context]\nrecalled stuff\n[End Memory Context]\n"
            "Assistant: answer"
        )
        once = fence_content(text)
        twice = fence_content(once.content)
        assert twice.content == once.content
        assert twice.stripped_spans == 0
        assert twice.markers == []

    def test_empty_after_fence(self):
        text = "<system-reminder>only injected content, nothing else</system-reminder>"
        result = fence_content(text)
        assert result.content.strip() == ""

    def test_collapses_newline_runs_left_by_stripping(self):
        """The collapse DOES fire when stripping actually happened and left
        a 3+ newline run behind — this is the positive case for FIX 1,
        paired with the no-op test above for the negative case."""
        text = "before\n\n<system-reminder>noise</system-reminder>\n\nafter"
        result = fence_content(text)
        assert result.stripped_spans == 1
        assert result.markers == ["system_reminder"]
        assert "\n\n\n" not in result.content
        assert "before" in result.content and "after" in result.content
        assert "noise" not in result.content

    def test_mixed_marker_document(self):
        # NOTE: the langchain block (d) strips forward to the next "## "
        # h2 boundary OR end of text — per spec, with nothing left to anchor
        # a boundary after it, that means to EOF. So real content that must
        # survive is placed BEFORE the langchain block, which is last.
        text = (
            "User: catch me up.\n"
            "<system-reminder>\nharness note\n</system-reminder>\n"
            "Here is real content one.\n"
            "[Revien Memory Context]\n"
            "- [Score: 80%] (1 day ago) Old Fact: something recalled\n"
            "\n[End Memory Context]\n"
            "Here is real content two.\n"
            "## Relevant memory (Revien)\n"
            "- another recalled line\n"
            "Here is real content three.\n"
            "## Relevant Context (from 1 nodes)\n"
            "### Result 1: Thing\n"
            "Type: entity | Score: 0.5\n"
            "\nrecalled body\n"
            "\n[Retrieved in 1.00ms]\n"
        )
        result = fence_content(text)
        for gone in (
            "harness note", "something recalled", "another recalled line",
            "recalled body", "Retrieved in 1.00ms",
        ):
            assert gone not in result.content, f"{gone!r} should have been fenced"
        assert "catch me up" in result.content
        assert "real content one" in result.content
        assert "real content two" in result.content
        assert "real content three" in result.content
        assert set(result.markers) == {
            "system_reminder", "memory_context", "hermes_header",
            "langchain_context",
        }
        assert result.stripped_spans == 4


# ── Pipeline wiring ────────────────────────────────────────

class TestPipelineFenceWiring:
    def test_default_on_strips_system_reminder_before_ingest(self, pipeline, store):
        content = (
            "User: what's the plan?\n"
            "<system-reminder>\nInternal harness note, never conversation.\n"
            "</system-reminder>\n"
            "Assistant: Ship Tuesday."
        )
        output = pipeline.ingest(IngestionInput(
            source_id="fence-test-1",
            content=content,
        ))
        assert output.fenced_spans == 1
        node = store.get_node(output.context_node_id)
        assert node is not None
        assert "Internal harness note" not in node.content
        assert "what's the plan" in node.content

    def test_revien_fence_0_disables_stripping(self, monkeypatch, pipeline, store):
        monkeypatch.setenv("REVIEN_FENCE", "0")
        content = (
            "User: what's the plan?\n"
            "<system-reminder>\nInternal harness note, never conversation.\n"
            "</system-reminder>\n"
            "Assistant: Ship Tuesday."
        )
        output = pipeline.ingest(IngestionInput(
            source_id="fence-test-2",
            content=content,
        ))
        assert output.fenced_spans == 0
        node = store.get_node(output.context_node_id)
        assert node is not None
        assert "Internal harness note" in node.content

    def test_empty_after_fence_skips_ingest_entirely(self, pipeline, store):
        content = "<system-reminder>only injected content</system-reminder>"
        before_nodes = store.count_nodes()
        output = pipeline.ingest(IngestionInput(
            source_id="fence-test-3",
            content=content,
        ))
        assert output.context_node_id == ""
        assert output.nodes_created == 0
        assert store.count_nodes() == before_nodes
