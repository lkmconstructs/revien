"""
Tests for distill-out (leg 2): Revien's memory written into the vault as
markdown. Every hard rule from the design gets its own assertion:
writes only in the distill folder / marker-gated overwrite and prune /
no re-ingest of distilled notes / deterministic + idempotent / vault-echo
guard / invalidated claims excluded / pure read (graph never mutated).
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from revien.adapters.obsidian import ObsidianVaultAdapter
from revien.distill import VaultDistiller, _safe_filename
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore


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
    d = Path(tempfile.mkdtemp(prefix="revien_distill_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _node(store, node_type, label, content=None, curated=False, recorded=None,
          invalidated=False, confidence=0.9):
    now = datetime.now(timezone.utc)
    n = Node(
        node_type=node_type,
        label=label,
        content=content or label,
        source_type=SourceType.EXTRACTED,
        confidence=confidence,
        created_at=now,
        last_accessed=now,
        recorded_at=recorded,
        metadata={"curated": True} if curated else {"adapter": "claude_code"},
    )
    if invalidated:
        n.invalidated_at = now
    return store.add_node(n)


def _edge(store, a, b):
    store.add_edge(Edge(
        edge_type=EdgeType.RELATED_TO,
        source_node_id=a.node_id, target_node_id=b.node_id, weight=0.8,
    ))


def _seed_entity(store, label="PostgreSQL"):
    """Entity with two machine claims and one related entity."""
    entity = _node(store, NodeType.ENTITY, label)
    fact = _node(store, NodeType.FACT, "chosen db",
                 "We chose PostgreSQL over MySQL.",
                 recorded=datetime(2026, 4, 1, tzinfo=timezone.utc))
    decision = _node(store, NodeType.DECISION, "jsonb",
                     "Decided to use JSONB columns for flexible metadata.")
    related = _node(store, NodeType.ENTITY, "Hetzner Server")
    _edge(store, fact, entity)
    _edge(store, decision, entity)
    _edge(store, entity, related)
    return entity


class TestRendering:
    def test_note_content_provenance_and_wikilinks(self, store, vault):
        _seed_entity(store)
        summary = VaultDistiller(store, str(vault)).distill()
        assert summary["status"] == "ok" and summary["notes"] >= 1

        note = (vault / "Revien" / "PostgreSQL.md").read_text(encoding="utf-8")
        assert "revien: derived" in note
        assert "# PostgreSQL — Revien memory" in note
        assert "We chose PostgreSQL over MySQL." in note
        assert "2026-04-01" in note                      # content date rendered
        assert "confidence 0.90" in note                 # trust rendered
        assert "[[Hetzner Server]]" in note              # threads into the graph pane
        # Index note generated and links the entity.
        index = (vault / "Revien" / "_Revien Index.md").read_text(encoding="utf-8")
        assert "[[PostgreSQL]]" in index

    def test_invalidated_claims_excluded(self, store, vault):
        entity = _seed_entity(store)
        dead = _node(store, NodeType.FACT, "old", "We use MySQL.", invalidated=True)
        _edge(store, dead, entity)
        VaultDistiller(store, str(vault)).distill()
        note = (vault / "Revien" / "PostgreSQL.md").read_text(encoding="utf-8")
        assert "We use MySQL." not in note

    def test_vault_echo_guard(self, store, vault):
        """An entity whose only claims came FROM the vault is not echoed back."""
        entity = _node(store, NodeType.ENTITY, "My Own Note")
        vault_claim = _node(store, NodeType.FACT, "own", "Curated content.", curated=True)
        _edge(store, vault_claim, entity)
        summary = VaultDistiller(store, str(vault)).distill()
        assert summary["skipped_vault_echo"] == 1
        assert not (vault / "Revien" / "My Own Note.md").exists()

    def test_safe_filenames(self):
        assert _safe_filename('bad:/\\*"name?') == "badname"
        assert _safe_filename("   ") == "entity"

    def test_junk_entities_never_break_wikilinks(self, store, vault):
        """Extractor noise (labels spanning newlines / carrying brackets) must
        degrade to 'not linked', never to broken markdown in the vault."""
        entity = _seed_entity(store)
        junk = _node(store, NodeType.ENTITY, "Deployment\nRuns")
        junk2 = _node(store, NodeType.ENTITY, "bad [[label]]")
        _edge(store, entity, junk)
        _edge(store, entity, junk2)
        VaultDistiller(store, str(vault)).distill()
        note = (vault / "Revien" / "PostgreSQL.md").read_text(encoding="utf-8")
        assert "[[Deployment" not in note
        assert "bad [[label]]" not in note
        assert "[[Hetzner Server]]" in note  # clean links still render


class TestSafetyRails:
    def test_writes_only_inside_distill_folder(self, store, vault):
        (vault / "User Note.md").write_text("mine", encoding="utf-8")
        _seed_entity(store)
        VaultDistiller(store, str(vault)).distill()
        written = {p.relative_to(vault).as_posix() for p in vault.rglob("*.md")}
        assert (vault / "User Note.md").read_text(encoding="utf-8") == "mine"
        assert all(p == "User Note.md" or p.startswith("Revien/") for p in written)

    def test_never_overwrites_unmarked_file(self, store, vault):
        """A user file sitting at OUR target path is inviolate."""
        out = vault / "Revien"
        out.mkdir()
        (out / "PostgreSQL.md").write_text("user's own postgres note", encoding="utf-8")
        _seed_entity(store)
        VaultDistiller(store, str(vault)).distill()
        assert (out / "PostgreSQL.md").read_text(encoding="utf-8") == \
            "user's own postgres note"

    def test_prune_is_marker_gated(self, store, vault):
        out = vault / "Revien"
        out.mkdir()
        # A stale generated note (marked) and a user note (unmarked).
        (out / "Stale Entity.md").write_text(
            "---\nrevien: derived\nentity: Stale Entity\n---\nold view\n",
            encoding="utf-8",
        )
        (out / "Keep Me.md").write_text("user note in the folder", encoding="utf-8")
        _seed_entity(store)
        summary = VaultDistiller(store, str(vault)).distill()
        assert summary["pruned"] == 1
        assert not (out / "Stale Entity.md").exists()
        assert (out / "Keep Me.md").exists()

    def test_distill_is_pure_read(self, store, vault):
        _seed_entity(store)
        nodes_before = store.count_nodes()
        edges_before = store.count_edges()
        audit_before = len(store.get_all_audit())
        VaultDistiller(store, str(vault)).distill()
        assert store.count_nodes() == nodes_before
        assert store.count_edges() == edges_before
        assert len(store.get_all_audit()) == audit_before


class TestIdempotency:
    def test_second_run_writes_nothing(self, store, vault):
        _seed_entity(store)
        d = VaultDistiller(store, str(vault))
        first = d.distill()
        assert first["written"] >= 1
        second = d.distill()
        assert second["written"] == 0
        assert second["unchanged"] == first["written"] + first["unchanged"]

    def test_content_is_deterministic(self, store, vault):
        _seed_entity(store)
        d = VaultDistiller(store, str(vault))
        d.distill()
        note_path = vault / "Revien" / "PostgreSQL.md"
        content1 = note_path.read_text(encoding="utf-8")
        note_path.unlink()
        d.distill()
        assert note_path.read_text(encoding="utf-8") == content1


class TestNoEchoLoop:
    def test_distilled_notes_are_not_reingested(self, store, vault):
        _seed_entity(store)
        VaultDistiller(store, str(vault)).distill()
        # A user note ingests; every distilled note is skipped.
        (vault / "Real Note.md").write_text("Actual user content.", encoding="utf-8")
        adapter = ObsidianVaultAdapter(str(vault))
        items = asyncio.run(
            adapter.fetch_new_content(datetime.fromtimestamp(0, tz=timezone.utc))
        )
        sources = {i["source_id"] for i in items}
        assert any("Real Note" in s for s in sources)
        assert not any("Revien/" in s for s in sources), \
            "distilled notes must never feed back into the graph"
