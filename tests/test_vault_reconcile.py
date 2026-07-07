"""
Tests for the editable-vault leg: reconciling edits to distilled notes back
into the graph. The three gestures (correct / delete / add) and the safety
properties that justify the manifest — idempotency (no echo), no-false-reject
(the Mentat catch: a claim ingested after the user last saw the note must not
be forgotten), and round-trip stability.
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from revien.distill import VaultDistiller, VaultReconciler, _ANCHOR_RE, _parse_claim_line
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.semantic.index import SemanticIndex


def _recon(store, vault):
    # Disabled semantic index keeps the correctness tests fast and
    # deterministic (a live index would model-load per created node).
    return VaultReconciler(store, str(vault), semantic=SemanticIndex(store, enabled=False))


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
    d = Path(tempfile.mkdtemp(prefix="revien_reconcile_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _node(store, node_type, label, content, curated=False):
    now = datetime.now(timezone.utc)
    return store.add_node(Node(
        node_type=node_type, label=label, content=content,
        source_type=SourceType.EXTRACTED,
        confidence=1.0 if curated else 0.7,
        created_at=now, last_accessed=now, recorded_at=now,
        metadata={"curated": True} if curated else {"adapter": "claude_code"},
    ))


def _entity_with_claims(store, label="Atlas Server"):
    ent = _node(store, NodeType.ENTITY, label, label)
    fact = _node(store, NodeType.FACT, "ram", "Has 64GB of RAM.")
    dec = _node(store, NodeType.DECISION, "upgrade", "Decided to upgrade to 128GB.")
    for c in (fact, dec):
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                            source_node_id=ent.node_id, target_node_id=c.node_id,
                            weight=0.8))
    return ent, fact, dec


def _note_path(vault, label="Atlas Server"):
    return vault / "Revien" / f"{label}.md"


def _live_contents(store, entity_id):
    out = []
    for nid in store.get_neighbors(entity_id):
        n = store.get_node(nid)
        if n and n.invalidated_at is None and n.node_type != NodeType.ENTITY:
            out.append(n.content)
    return out


# ── Parsing units ──────────────────────────────────────────────────────

class TestParsing:
    def test_anchor_and_text_extracted(self):
        line = "Has 64GB of RAM. *(claude_code, 2026-07-07, confidence 0.70)* <!--rv:abc-123-->"
        anchor, text = _parse_claim_line(line)
        assert anchor == "abc-123"
        assert text == "Has 64GB of RAM."

    def test_user_added_line_has_no_anchor(self):
        anchor, text = _parse_claim_line("Runs Ubuntu 24.04.")
        assert anchor is None
        assert text == "Runs Ubuntu 24.04."

    def test_edited_text_with_leftover_provenance(self):
        # Anchors are node UUIDs (hex + dashes) — the regex matches those tightly
        # so a malformed <!--rv:...--> can't be mistaken for a real anchor.
        line = "Now 128GB. *(claude_code, confidence 0.70)* <!--rv:abcdef12-3456-->"
        anchor, text = _parse_claim_line(line)
        assert anchor == "abcdef12-3456" and text == "Now 128GB."


# ── Distill writes anchors + manifest ──────────────────────────────────

class TestDistillManifest:
    def test_anchors_and_manifest_written(self, store, vault):
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        text = _note_path(vault).read_text(encoding="utf-8")
        anchors = set(_ANCHOR_RE.findall(text))
        assert fact.node_id in anchors and dec.node_id in anchors
        manifest = store.get_distill_manifest("Atlas Server")
        assert {r["anchor_node_id"] for r in manifest} == {fact.node_id, dec.node_id}
        assert all(r["entity_id"] == ent.node_id for r in manifest)


# ── The three gestures ─────────────────────────────────────────────────

class TestReconcileGestures:
    def _distill_then_edit(self, store, vault, transform):
        _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        p.write_text(transform(p.read_text(encoding="utf-8")), encoding="utf-8")
        return _recon(store, vault).reconcile()

    def test_edit_supersedes(self, store, vault):
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        p.write_text(
            p.read_text(encoding="utf-8").replace(
                "Has 64GB of RAM.", "Has 128GB of RAM after the upgrade."),
            encoding="utf-8")
        rec = _recon(store, vault).reconcile()
        assert rec["corrected"] == 1
        assert store.get_node(fact.node_id).invalidated_at is not None
        live = _live_contents(store, ent.node_id)
        assert "Has 128GB of RAM after the upgrade." in live
        assert "Has 64GB of RAM." not in live
        # The correction carries a CORRECTS edge to the superseded claim.
        corrects = [
            e for e in store.get_edges_for_node(fact.node_id)
            if e.edge_type == EdgeType.CORRECTS and e.target_node_id == fact.node_id
        ]
        assert corrects, "correction must CORRECTS-supersede the old claim"

    def test_delete_forgets(self, store, vault):
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        kept = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                if "Decided to upgrade" not in ln]
        p.write_text("\n".join(kept) + "\n", encoding="utf-8")
        rec = _recon(store, vault).reconcile()
        assert rec["forgotten"] == 1
        assert store.get_node(dec.node_id).invalidated_at is not None
        # Soft, not hard — the row and content are retained.
        assert store.get_node(dec.node_id).content == "Decided to upgrade to 128GB."

    def test_add_teaches(self, store, vault):
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        lines = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            lines.append(ln)
            if ln.strip() == "## Facts":
                lines.append("- Runs Ubuntu 24.04.")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rec = _recon(store, vault).reconcile()
        assert rec["added"] == 1
        live = _live_contents(store, ent.node_id)
        assert "Runs Ubuntu 24.04." in live
        # Added claim is curated (human-authored) and typed by its section.
        added = [store.get_node(n) for n in store.get_neighbors(ent.node_id)]
        ubuntu = next(n for n in added if n and n.content == "Runs Ubuntu 24.04.")
        assert ubuntu.metadata.get("curated") is True
        assert ubuntu.node_type == NodeType.FACT


# ── Safety properties (the reason the manifest exists) ─────────────────

class TestReconcileSafety:
    def test_idempotent_no_echo(self, store, vault):
        _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        rec = _recon(store, vault).reconcile()
        assert (rec["corrected"], rec["added"], rec["forgotten"]) == (0, 0, 0)

    def test_no_false_reject_of_unseen_claims(self, store, vault):
        """The Mentat catch: a claim ingested AFTER the user last saw the note
        must not be forgotten when they reconcile the unedited note. The
        manifest (what was shown), not the live graph, is the delete referent."""
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        # New machine claim arrives, never distilled into the note.
        unseen = _node(store, NodeType.FACT, "region", "Hosted in eu-central.")
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                            source_node_id=ent.node_id, target_node_id=unseen.node_id,
                            weight=0.8))
        rec = _recon(store, vault).reconcile()
        assert rec["forgotten"] == 0
        assert store.get_node(unseen.node_id).invalidated_at is None

    def test_round_trip_stable(self, store, vault):
        _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        p.write_text(p.read_text(encoding="utf-8").replace(
            "Has 64GB of RAM.", "Has 128GB now."), encoding="utf-8")
        r1 = _recon(store, vault).reconcile()
        r2 = _recon(store, vault).reconcile()
        assert r1["corrected"] == 1
        assert (r2["corrected"], r2["added"], r2["forgotten"]) == (0, 0, 0)

    def test_only_manifest_notes_are_touched(self, store, vault):
        """A user note that Revien never rendered (no manifest) is not
        reconciled, even if it sits in the distill folder."""
        (vault / "Revien").mkdir(parents=True)
        stray = vault / "Revien" / "My Note.md"
        stray.write_text("---\ntitle: mine\n---\n## Facts\n- something\n", encoding="utf-8")
        rec = _recon(store, vault).reconcile()
        assert (rec["corrected"], rec["added"], rec["forgotten"]) == (0, 0, 0)
        assert stray.read_text(encoding="utf-8").startswith("---\ntitle: mine")

    def test_missing_file_does_not_mass_invalidate(self, store, vault):
        """If the user deletes the whole note file, we do NOT invalidate all its
        claims (could be an accident or a move) — next distill regenerates it."""
        ent, fact, dec = _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        _note_path(vault).unlink()
        rec = _recon(store, vault).reconcile()
        assert rec["forgotten"] == 0
        assert store.get_node(fact.node_id).invalidated_at is None


class TestReconcileHardening:
    """Regressions for the bugs the adversarial pass found — each was a
    CONFIRMED data-loss or correctness defect before the hardening fix."""

    def _distill(self, store, vault):
        VaultDistiller(store, str(vault)).distill()

    def _add_under_facts(self, vault, text, label="Atlas Server"):
        p = _note_path(vault, label)
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            out.append(ln)
            if ln.strip() == "## Facts":
                out.append(f"- {text}")
        p.write_text("\n".join(out) + "\n", encoding="utf-8")

    def test_add_is_stable_across_repeated_reconciles(self, store, vault):
        """#1: an added line was re-added forever and its prior node forgotten
        each pass. Now redistill anchors it, so repeats are inert."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        self._add_under_facts(vault, "Runs Ubuntu 24.04.")
        r1 = _recon(store, vault).reconcile()
        r2 = _recon(store, vault).reconcile()
        r3 = _recon(store, vault).reconcile()
        assert r1["added"] == 1
        assert (r2["added"], r2["forgotten"], r3["added"], r3["forgotten"]) == (0, 0, 0, 0)
        assert _live_contents(store, ent.node_id).count("Runs Ubuntu 24.04.") == 1

    def test_edit_with_intext_anchor_token_preserved(self, store, vault):
        """#2: an edit whose text contains an anchor-shaped token lost the claim
        entirely. Now only the TRAILING anchor counts."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        p = _note_path(vault)
        new = [
            ln.replace("Has 64GB of RAM.", "See <!--rv:00000000-0000--> now 128GB.")
            if "Has 64GB of RAM" in ln else ln
            for ln in p.read_text(encoding="utf-8").splitlines()
        ]
        p.write_text("\n".join(new) + "\n", encoding="utf-8")
        r = _recon(store, vault).reconcile()
        assert r["forgotten"] == 0
        assert any("now 128GB" in c for c in _live_contents(store, ent.node_id))

    def test_add_with_intext_anchor_token(self, store, vault):
        """#3: a user-typed claim containing an anchor-shaped token was dropped."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        self._add_under_facts(vault, "Anchors look like <!--rv:12345678-90ab--> here.")
        r = _recon(store, vault).reconcile()
        assert r["added"] == 1
        assert any("Anchors look like" in c for c in _live_contents(store, ent.node_id))

    def test_section_rename_does_not_forget(self, store, vault):
        """#5: renaming a heading dropped its claims out of the section scan and
        mass-forgot them. Delete is now whole-file anchor presence."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        p = _note_path(vault)
        p.write_text(p.read_text(encoding="utf-8").replace("## Facts", "## Hardware"),
                     encoding="utf-8")
        r = _recon(store, vault).reconcile()
        assert r["forgotten"] == 0
        assert store.get_node(fact.node_id).invalidated_at is None

    def test_added_emphasis_line_kept_whole(self, store, vault):
        """#6: an added line ending in *(...)* had it stripped as provenance.
        Provenance is now stripped only from anchored lines."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        self._add_under_facts(vault, "Prefer bash *(not zsh)*")
        _recon(store, vault).reconcile()
        assert "Prefer bash *(not zsh)*" in _live_contents(store, ent.node_id)

    def test_truncated_file_does_not_mass_invalidate(self, store, vault):
        """Corrupt/truncated note must never be read as 'user deleted everything'."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        p = _note_path(vault)
        p.write_text(p.read_text(encoding="utf-8")[:20], encoding="utf-8")  # truncated
        r = _recon(store, vault).reconcile()
        assert r["forgotten"] == 0
        assert store.get_node(fact.node_id).invalidated_at is None
        assert store.get_node(dec.node_id).invalidated_at is None

    def test_shared_claim_delete_not_global(self, store, vault):
        """A claim rendered into two notes must not be globally nuked by a delete
        in one of them."""
        e1 = _node(store, NodeType.ENTITY, "Server A", "Server A")
        e2 = _node(store, NodeType.ENTITY, "Server B", "Server B")
        shared = _node(store, NodeType.FACT, "shared", "Both share a rack.")
        a2 = _node(store, NodeType.FACT, "a2", "A runs nginx.")
        b2 = _node(store, NodeType.FACT, "b2", "B runs postgres.")
        for e, extra in ((e1, a2), (e2, b2)):
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=e.node_id,
                                target_node_id=shared.node_id, weight=0.8))
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=e.node_id,
                                target_node_id=extra.node_id, weight=0.8))
        self._distill(store, vault)
        pa = _note_path(vault, "Server A")
        kept = [ln for ln in pa.read_text(encoding="utf-8").splitlines()
                if "Both share a rack" not in ln]
        pa.write_text("\n".join(kept) + "\n", encoding="utf-8")
        _recon(store, vault).reconcile()
        assert store.get_node(shared.node_id).invalidated_at is None

    def test_free_text_preserved_in_other_notes(self, store, vault):
        """The bug the SECOND adversarial pass caught: reconciling one note must
        not regenerate (and wipe free-text in) other notes."""
        _entity_with_claims(store, label="Atlas Server")
        e2 = _node(store, NodeType.ENTITY, "Halcyon", "Halcyon")
        for lbl, content in (("x", "Written in Python."), ("y", "Uses SQLite.")):
            n = _node(store, NodeType.FACT, lbl, content)
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=e2.node_id,
                                target_node_id=n.node_id, weight=0.8))
        self._distill(store, vault)
        pb = _note_path(vault, "Halcyon")
        pb.write_text(pb.read_text(encoding="utf-8")
                      + "\n## My notes\nAsk Mira about onboarding.\n", encoding="utf-8")
        # Edit a claim in the OTHER note, triggering reconcile work.
        pa = _note_path(vault, "Atlas Server")
        pa.write_text(pa.read_text(encoding="utf-8").replace(
            "Has 64GB of RAM.", "Has 128GB of RAM."), encoding="utf-8")
        _recon(store, vault).reconcile()
        assert "Ask Mira about onboarding." in pb.read_text(encoding="utf-8")

    def test_free_text_preserved_in_edited_note(self, store, vault):
        """Free-text adjacent to an edited claim in the SAME note survives too."""
        ent, fact, dec = _entity_with_claims(store)
        self._distill(store, vault)
        p = _note_path(vault)
        txt = p.read_text(encoding="utf-8").replace("Has 64GB of RAM.", "Has 128GB of RAM.")
        txt += "\n## My reminders\nCheck the RAID battery.\n"
        p.write_text(txt, encoding="utf-8")
        _recon(store, vault).reconcile()
        assert "Check the RAID battery." in p.read_text(encoding="utf-8")

    def test_vault_origin_curated_not_rendered(self, store, vault):
        """A vault-origin curated claim is managed in the user's source note, not
        the distilled view — so it isn't rendered or reconcilable here."""
        ent = _node(store, NodeType.ENTITY, "Halcyon", "Halcyon")
        machine = _node(store, NodeType.FACT, "m", "Written in Python.")
        curated = _node(store, NodeType.FACT, "c", "Local-first reading app.", curated=True)
        for c in (machine, curated):
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=ent.node_id,
                                target_node_id=c.node_id, weight=0.8))
        self._distill(store, vault)
        text = _note_path(vault, "Halcyon").read_text(encoding="utf-8")
        assert "Written in Python." in text
        assert "Local-first reading app." not in text


class _SpyIndex:
    """Records index_node calls so we can assert reconciled nodes are made
    semantically searchable (add_node does not index; the reconciler must)."""
    is_enabled = True

    def __init__(self):
        self.indexed = []

    def index_node(self, node_id, label, content):
        self.indexed.append((node_id, content))
        return True


class TestReconcileSemanticIndexing:
    def test_corrections_and_adds_are_indexed(self, store, vault):
        """The gap the adversarial pass targeted: a corrected/added claim must
        be embedded, or the fix is invisible to semantic recall."""
        _entity_with_claims(store)
        VaultDistiller(store, str(vault)).distill()
        p = _note_path(vault)
        lines = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            lines.append(ln.replace("Has 64GB of RAM.", "Has 128GB of RAM."))
            if ln.strip() == "## Facts":
                lines.append("- Runs Ubuntu 24.04.")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        spy = _SpyIndex()
        VaultReconciler(store, str(vault), semantic=spy).reconcile()
        indexed = [c for _, c in spy.indexed]
        assert "Has 128GB of RAM." in indexed, "correction must be indexed"
        assert "Runs Ubuntu 24.04." in indexed, "added claim must be indexed"
