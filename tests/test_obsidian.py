"""
Tests for the Obsidian vault adapter + curated-ingest path.

Covers: markdown parsing (frontmatter/headings/wikilinks/tags), heading
chunking, the adapter's mtime-gated scan, pipeline link->edge transcription,
curated confidence, and the CSL curated shield (machine claims can never
silently auto-supersede human-curated memory).
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from revien.adapters.obsidian import (
    ObsidianVaultAdapter,
    _parse_frontmatter,
    chunk_note,
)
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.ingestion.supersession_ingest import ClaimGovernor
from revien.semantic.index import SemanticIndex
from revien.supersession import SupersessionAction


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
def vault():
    # tempfile.mkdtemp, not pytest's tmp_path — the pytest temp root has
    # broken ACLs on this machine (repo-wide convention: tempfile only).
    import shutil
    d = Path(tempfile.mkdtemp(prefix="revien_vault_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _fetch(adapter, since=None):
    since = since or datetime.fromtimestamp(0, tz=timezone.utc)
    return asyncio.run(adapter.fetch_new_content(since))


# ── Parsing units ──────────────────────────────────────────────────────

class TestFrontmatter:
    def test_inline_tags_and_date(self):
        fm, body = _parse_frontmatter(
            "---\ntags: [infra, decisions]\ndate: 2026-05-09\n---\nBody here."
        )
        assert fm["tags"] == ["infra", "decisions"]
        assert fm["date"].year == 2026 and fm["date"].month == 5
        assert body == "Body here."

    def test_block_list_tags(self):
        fm, _ = _parse_frontmatter(
            "---\ntags:\n  - infra\n  - '#quoted'\ncreated: 2025-11-17\n---\nx"
        )
        assert fm["tags"] == ["infra", "quoted"]
        assert fm["date"].year == 2025

    def test_no_frontmatter_passthrough(self):
        fm, body = _parse_frontmatter("Just a note.\n---\nnot frontmatter")
        assert fm == {}
        assert body.startswith("Just a note.")


class TestChunking:
    def test_preamble_and_headings(self):
        body = "Intro line.\n\n## Setup\nSetup text.\n\n## Decision\nWe chose X."
        chunks = chunk_note(body, "My Note")
        assert [h for h, _ in chunks] == ["My Note", "Setup", "Decision"]
        assert chunks[2][1] == "We chose X."

    def test_no_headings_single_chunk(self):
        chunks = chunk_note("Only body text.", "Flat Note")
        assert chunks == [("Flat Note", "Only body text.")]

    def test_empty_sections_dropped(self):
        chunks = chunk_note("## A\n\n## B\ncontent", "N")
        assert [h for h, _ in chunks] == ["B"]


# ── Adapter scan ───────────────────────────────────────────────────────

class TestVaultScan:
    def test_chunks_links_tags_and_source_ids(self, vault):
        (vault / "Postgres Decision.md").write_text(
            "---\ntags: [infra]\ndate: 2026-04-01\n---\n"
            "We picked [[PostgreSQL]] over [[MySQL|the other one]].\n"
            "## Rationale\nBecause of [[Fernweh Server#specs]] and #performance.\n",
            encoding="utf-8",
        )
        items = _fetch(ObsidianVaultAdapter(str(vault)))
        assert len(items) == 2  # preamble chunk + Rationale chunk

        pre, rat = items
        assert pre["source_id"] == "vault:Postgres Decision.md#postgres-decision"
        assert pre["curated"] is True
        assert "PostgreSQL" in pre["links"] and "MySQL" in pre["links"]
        assert "Postgres Decision" in pre["links"]  # own-title anchor
        assert pre["timestamp"].startswith("2026-04-01")  # frontmatter beats mtime

        assert rat["metadata"]["heading"] == "Rationale"
        assert "Fernweh Server" in rat["links"]  # #heading anchor stripped
        assert "performance" in rat["metadata"]["tags"]  # inline tag
        assert "infra" in rat["metadata"]["tags"]  # frontmatter tag rides all chunks

    def test_excluded_dirs_and_mtime_gate(self, vault):
        (vault / ".obsidian").mkdir()
        (vault / ".obsidian" / "config.md").write_text("internal", encoding="utf-8")
        note = vault / "old.md"
        note.write_text("Old note body.", encoding="utf-8")

        adapter = ObsidianVaultAdapter(str(vault))
        assert len(_fetch(adapter)) == 1  # .obsidian excluded

        # Nothing modified since the future -> nothing fetched.
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert _fetch(adapter, since=future) == []

    def test_headings_are_not_inline_tags(self, vault):
        (vault / "n.md").write_text("## Heading\nBody #realtag here.", encoding="utf-8")
        items = _fetch(ObsidianVaultAdapter(str(vault)))
        assert items[0]["metadata"]["tags"] == ["realtag"]


# ── Pipeline integration: links become edges, curated becomes confidence ──

class TestCuratedIngest:
    def _ingest_chunk(self, store, **overrides):
        pipeline = IngestionPipeline(store, semantic=SemanticIndex(store, enabled=False))
        kwargs = dict(
            source_id="vault:Postgres Decision.md#postgres-decision",
            content="Postgres Decision\nWe picked PostgreSQL over MySQL.",
            content_type="note",
            timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
            links=["Postgres Decision", "PostgreSQL", "MySQL"],
            curated=True,
            metadata={"adapter": "obsidian"},
        )
        kwargs.update(overrides)
        return pipeline.ingest(IngestionInput(**kwargs)), pipeline

    def test_links_become_entities_and_edges(self, store):
        out, _ = self._ingest_chunk(store)
        ops = GraphOperations(store)
        for label in ("Postgres Decision", "PostgreSQL", "MySQL"):
            entity = ops.find_node_by_label(label, node_type=NodeType.ENTITY)
            assert entity is not None, f"link {label!r} should be an ENTITY node"
            # Author-drawn edge from the chunk's CONTEXT to the entity.
            neighbors = store.get_neighbors(entity.node_id)
            assert out.context_node_id in neighbors, \
                f"context should link to {label!r}"

    def test_link_edges_are_strong(self, store):
        from revien.graph.schema import EdgeType
        out, _ = self._ingest_chunk(store)
        ops = GraphOperations(store)
        entity = ops.find_node_by_label("PostgreSQL", node_type=NodeType.ENTITY)
        edges = store.get_edges_for_node(entity.node_id)
        # The extractor adds its own weaker MENTIONED_BY edge; the author-drawn
        # link edge is the RELATED_TO one and must carry curated weight.
        link_edges = [
            e for e in edges
            if e.edge_type == EdgeType.RELATED_TO
            and out.context_node_id in (e.source_node_id, e.target_node_id)
        ]
        assert link_edges and all(e.weight >= 0.8 for e in link_edges)

    def test_curated_confidence_and_flag(self, store):
        out, _ = self._ingest_chunk(store)
        ctx = store.get_node(out.context_node_id)
        assert ctx.confidence == 1.0
        assert ctx.metadata.get("curated") is True
        assert ctx.recorded_at is not None

    def test_second_chunk_reuses_entities(self, store):
        self._ingest_chunk(store)
        out2, _ = self._ingest_chunk(
            store,
            source_id="vault:Postgres Decision.md#rationale",
            content="Postgres Decision — Rationale\nJSONB support decided it.",
            links=["Postgres Decision", "PostgreSQL"],
        )
        ops = GraphOperations(store)
        # Still exactly ONE entity per label — both chunks hang off the same
        # anchors, which is what makes the note's graph walkable as a unit.
        all_entities = [
            n for n in store.list_nodes(node_type=NodeType.ENTITY, limit=1000)
            if n.label == "PostgreSQL"
        ]
        assert len(all_entities) == 1
        assert out2.context_node_id in store.get_neighbors(all_entities[0].node_id)


# ── CSL curated shield ─────────────────────────────────────────────────

class _AlwaysSupersedeGate:
    """Stub gate: every pair is a would-be auto-supersession."""

    class _Decision:
        def __init__(self):
            self.action = SupersessionAction.AUTO_SUPERSEDE
            self.reason = "stub: contradiction"
            self.trace = ["stub"]

    def evaluate(self, existing, new):
        return self._Decision()


class TestCuratedShield:
    def _node(self, store, content, curated):
        from revien.graph.schema import Node, SourceType
        now = datetime.now(timezone.utc)
        return store.add_node(Node(
            node_type=NodeType.CONTEXT,
            label=content[:40],
            content=content,
            source_type=SourceType.EXTRACTED,
            confidence=1.0 if curated else 0.7,
            created_at=now,
            last_accessed=now,
            metadata={"curated": True} if curated else {},
        ))

    def test_machine_claim_cannot_auto_supersede_curated(self, store):
        ops = GraphOperations(store)
        curated = self._node(store, "We use PostgreSQL for the backend.", curated=True)
        machine = self._node(store, "We use MySQL for the backend.", curated=False)

        gov = ClaimGovernor(store, ops, gate=_AlwaysSupersedeGate())
        outcomes = gov.govern(machine)

        target = [o for o in outcomes if o.existing_node_id == curated.node_id]
        assert target, "governor should have evaluated the curated claim"
        assert target[0].action == "candidate", \
            "curated memory must go to the review queue, never silent supersession"
        assert "curated_shield" in target[0].reason
        # The curated claim is untouched — still live.
        assert store.get_node(curated.node_id).invalidated_at is None

    def test_curated_claim_may_supersede_machine(self, store):
        ops = GraphOperations(store)
        machine = self._node(store, "We use MySQL for the backend.", curated=False)
        curated = self._node(store, "We use PostgreSQL for the backend.", curated=True)

        gov = ClaimGovernor(store, ops, gate=_AlwaysSupersedeGate())
        outcomes = gov.govern(curated)

        target = [o for o in outcomes if o.existing_node_id == machine.node_id]
        assert target and target[0].action == "auto_supersede", \
            "the human's word outranks the machine's"
        assert store.get_node(machine.node_id).invalidated_at is not None
