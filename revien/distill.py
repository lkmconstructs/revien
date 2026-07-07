"""
Revien Distill-Out (+ Reconcile-In) — memory as files you can open AND edit.

Two halves of one loop:

  * VaultDistiller writes a markdown VIEW of the graph into a dedicated folder
    inside the user's Obsidian vault: one note per entity, every claim carrying
    its provenance and an invisible `<!--rv:node_id-->` anchor, related entities
    as [[wikilinks]]. Pure read of the graph.
  * VaultReconciler reads the user's EDITS to those notes back into the graph:
    correct a claim (curated node CORRECTS-supersedes the old), delete a line
    (soft-invalidate — forgotten, reversibly), add a line (new curated claim).
    This is the write-back path; it mutates the graph, deliberately.

HARD RULES (the reason this is safe to point at a real vault):
  * The distiller WRITES ONLY inside `<vault>/<folder>/` (default "Revien"),
    only overwrites/prunes files carrying `revien: derived` frontmatter, and is
    deterministic (same graph -> byte-identical files, no phantom churn).
  * The raw Obsidian ingest adapter still SKIPS `revien: derived` notes — they
    never enter normal ingestion. Edits come back ONLY through the reconciler,
    which acts on DIFFS against a persisted per-note manifest (the last-rendered
    snapshot). An unedited claim hashes equal and is a no-op, so re-reading a
    note is inert — no echo loop.
  * Deletion is judged against what was last SHOWN (the manifest), never the
    live graph, so a claim ingested after the user last saw the note can't be
    false-rejected.

Vault-echo guard: an entity whose only attached claims came FROM the vault is
skipped at distill — the user already wrote that note. An entity earns a
distilled note by holding at least `min_claims` machine-side claims.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore

MARKER_KEY = "revien"
MARKER_VALUE = "derived"
DEFAULT_FOLDER = "Revien"

# Per-claim anchor: invisible in Obsidian's reading view, the immutable join
# key between a distilled line and its graph node (and the manifest row).
_ANCHOR_RE = re.compile(r"<!--\s*rv:([0-9a-fA-F-]+)\s*-->")


def _claim_text(node: Node) -> str:
    """The exact text rendered for a claim — whitespace-normalized content.
    Render and manifest MUST agree on this so hashes match on reconcile."""
    return " ".join((node.content or node.label).split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# Claim-bearing node types rendered into a distilled note, in section order.
_SECTIONS: List[Tuple[NodeType, str]] = [
    (NodeType.DECISION, "Decisions"),
    (NodeType.FACT, "Facts"),
    (NodeType.PREFERENCE, "Preferences"),
    (NodeType.EVENT, "Events"),
]


def _safe_filename(label: str) -> str:
    """Entity label -> filesystem/Obsidian-safe filename stem."""
    cleaned = re.sub(r"[^\w\- ]+", "", label).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "entity"


def _linkable(label: str) -> bool:
    """A label that can live inside [[...]] without breaking markdown. Junk
    graph entities (extractor noise spanning newlines, bracket-bearing labels)
    must degrade to 'not linked', never to broken output."""
    return bool(label.strip()) and not re.search(r"[\n\r\[\]|#]", label)


def _provenance(node: Node) -> str:
    """One human-readable parenthetical per claim: where it came from, when
    it was said, how much we trust it."""
    md = node.metadata or {}
    if md.get("curated"):
        note = md.get("note")
        source = f"vault: {note}" if note else "vault"
    else:
        adapter = md.get("adapter")
        source = adapter if adapter else "conversation"
    parts = [source]
    if node.recorded_at is not None:
        parts.append(node.recorded_at.strftime("%Y-%m-%d"))
    parts.append(f"confidence {node.confidence:.2f}")
    return ", ".join(parts)


class VaultDistiller:
    """Render the graph's entity views into `<vault>/<folder>/` markdown."""

    def __init__(
        self,
        store: GraphStore,
        vault_dir: str,
        folder: str = DEFAULT_FOLDER,
        min_claims: int = 1,
    ):
        self.store = store
        self.vault_dir = Path(vault_dir)
        self.out_dir = self.vault_dir / folder
        self.min_claims = max(1, int(min_claims))

    # ── Gather ─────────────────────────────────────────────

    def _claims_for(self, entity: Node) -> Tuple[List[Node], List[Node]]:
        """(claims, related_entities) attached to an entity — live nodes only."""
        claims: List[Node] = []
        related: List[Node] = []
        for nid in self.store.get_neighbors(entity.node_id):
            node = self.store.get_node(nid)
            if node is None or node.invalidated_at is not None:
                continue
            if node.node_type == NodeType.ENTITY:
                related.append(node)
            elif any(node.node_type == t for t, _ in _SECTIONS):
                claims.append(node)
        return claims, related

    # ── Render ─────────────────────────────────────────────

    def _render(
        self, entity: Node, claims: List[Node], related: List[Node]
    ) -> Tuple[str, List[Dict]]:
        """Render a note AND the manifest rows it implies. Returns
        (markdown, manifest_rows). Each claim line carries an invisible
        <!--rv:node_id--> anchor; the manifest records the same node_id +
        a hash of the rendered text, so a later reconcile can tell edits from
        deletes from adds against what was actually shown."""
        manifest_rows: List[Dict] = []
        lines = [
            "---",
            f"{MARKER_KEY}: {MARKER_VALUE}",
            f"entity: {entity.label}",
            "---",
            "",
            f"# {entity.label} — Revien memory",
            "",
            "> Generated by Revien from its memory graph. This note is editable:",
            "> correct a claim to supersede it, delete a line to forget it, add a",
            "> line under a heading to teach it — changes reconcile into the graph",
            "> on the next `revien sync-vault`. Provenance *(in italics)* and the",
            "> `<!--rv:…-->` anchors are bookkeeping — leave them be.",
        ]
        for node_type, title in _SECTIONS:
            section = sorted(
                (c for c in claims if c.node_type == node_type),
                key=lambda c: (
                    c.recorded_at.isoformat() if c.recorded_at else "",
                    c.label.lower(),
                    c.node_id,
                ),
            )
            if not section:
                continue
            lines += ["", f"## {title}", ""]
            for claim in section:
                text = _claim_text(claim)
                lines.append(
                    f"- {text} *({_provenance(claim)})* <!--rv:{claim.node_id}-->"
                )
                manifest_rows.append({
                    "entity_id": entity.node_id,
                    "anchor_node_id": claim.node_id,
                    "current_node_id": claim.node_id,
                    "content_hash": _hash(text),
                    "section": title,
                })
        if related:
            names = sorted(
                {
                    r.label for r in related
                    if r.label != entity.label and _linkable(r.label)
                },
                key=str.lower,
            )
            if names:
                lines += ["", "## Related", ""]
                lines.append(" · ".join(f"[[{n}]]" for n in names))
        return "\n".join(lines) + "\n", manifest_rows

    def _render_index(self, entries: List[Tuple[str, int]]) -> str:
        lines = [
            "---",
            f"{MARKER_KEY}: {MARKER_VALUE}",
            "entity: __index__",
            "---",
            "",
            "# Revien memory index",
            "",
            "> One note per entity Revien holds machine-side memory about.",
            "",
        ]
        for stem, n_claims in sorted(entries, key=lambda e: e[0].lower()):
            lines.append(f"- [[{stem}]] ({n_claims} claim{'s' if n_claims != 1 else ''})")
        return "\n".join(lines) + "\n"

    # ── Write / prune (marker-gated) ───────────────────────

    @staticmethod
    def _has_marker(path: Path) -> bool:
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:400]
        except OSError:
            return False
        m = re.match(r"\A---\s*\n(.*?)\n---", head, re.DOTALL)
        return bool(m) and bool(
            re.search(rf"^{MARKER_KEY}:\s*{MARKER_VALUE}\s*$", m.group(1), re.MULTILINE)
        )

    def _write_if_changed(self, path: Path, content: str) -> bool:
        if path.exists():
            if not self._has_marker(path):
                # A user file where we wanted to write: theirs. Skip loudly
                # in the summary rather than overwrite silently.
                return False
            try:
                if path.read_text(encoding="utf-8") == content:
                    return False  # unchanged — no phantom churn
            except OSError:
                pass
        path.write_text(content, encoding="utf-8", newline="\n")
        return True

    # ── Entry point ────────────────────────────────────────

    def distill(self) -> Dict:
        """Regenerate the distilled view. Returns a summary dict."""
        if not self.vault_dir.is_dir():
            return {"status": "error", "error": f"vault not found: {self.vault_dir}"}
        self.out_dir.mkdir(parents=True, exist_ok=True)

        entities = self.store.list_nodes(node_type=NodeType.ENTITY, limit=999999)
        generated: Dict[str, str] = {}  # filename stem -> content
        manifests: Dict[str, List[Dict]] = {}  # filename stem -> manifest rows
        skipped_vault_echo = 0
        index_entries: List[Tuple[str, int]] = []

        for entity in sorted(entities, key=lambda e: (e.label.lower(), e.node_id)):
            if entity.invalidated_at is not None or not entity.label.strip():
                continue
            claims, related = self._claims_for(entity)
            # Only machine-extracted + reconcile-born claims are rendered (and
            # thus reconcilable). Vault-origin curated claims live in the user's
            # own note; echoing them here would let a delete resurrect on sync.
            reconcilable = [c for c in claims if _reconcilable(c)]
            if len(reconcilable) < self.min_claims:
                if claims:
                    skipped_vault_echo += 1
                continue

            stem = _safe_filename(entity.label)
            if stem in generated:  # label collision after sanitizing
                stem = f"{stem} ({entity.node_id[:8]})"
            content, rows = self._render(entity, reconcilable, related)
            generated[stem] = content
            manifests[stem] = rows
            index_entries.append((stem, len(reconcilable)))

        if index_entries:
            generated["_Revien Index"] = self._render_index(index_entries)

        written = sum(
            1 for stem, content in generated.items()
            if self._write_if_changed(self.out_dir / f"{stem}.md", content)
        )

        # Persist the last-rendered snapshot per note. This is the referent the
        # reconciler diffs edited notes against; without it, deletion-as-
        # rejection has nothing safe to compare to (a fresh render reflects the
        # current graph, not what the user last saw).
        for stem in generated:
            self.store.replace_distill_manifest(stem, manifests.get(stem, []))
        # Clear manifest rows for notes we no longer generate (entity invalidated,
        # renamed, or dropped below min_claims) — otherwise their orphaned rows
        # would drive phantom deletes on the next reconcile.
        generated_stems = set(generated)
        for stem in self.store.list_manifest_note_stems():
            if stem not in generated_stems:
                self.store.replace_distill_manifest(stem, [])

        # Prune: OUR stale files only (marker-gated). User files are inviolate.
        pruned = 0
        keep = {f"{stem}.md" for stem in generated}
        for path in self.out_dir.glob("*.md"):
            if path.name not in keep and self._has_marker(path):
                path.unlink()
                pruned += 1

        return {
            "status": "ok",
            "notes": len(index_entries),
            "written": written,
            "unchanged": len(generated) - written,
            "pruned": pruned,
            "skipped_vault_echo": skipped_vault_echo,
            "out_dir": str(self.out_dir),
        }


# ── Reconcile: absorb edits to distilled notes back into the graph ─────────

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_CLAIM_LINE_RE = re.compile(r"^\s*-\s+(.*\S)\s*$")
_PROV_SUFFIX_RE = re.compile(r"\s*\*\([^*]*\)\*\s*$")
# The REAL anchor is the TRAILING one only. An anchor-shaped token sitting
# inside user-typed text is content, not an anchor — matching it would let a
# user mentioning "<!--rv:...-->" hijack or vanish a claim.
_TRAILING_ANCHOR_RE = re.compile(r"\s*<!--\s*rv:([0-9a-fA-F-]+)\s*-->\s*$")

# Section heading -> the node type a claim added under it becomes.
_SECTION_TYPE: Dict[str, NodeType] = {title: nt for nt, title in _SECTIONS}


def _reconcilable(node: Node) -> bool:
    """Which claims a distilled note owns — and may be edited/deleted through.
    Machine-extracted claims (no other editable home) and reconcile-born
    corrections, YES. Vault-origin curated claims, NO: those live in the user's
    own source note, and 'deleting' the echo here would only resurrect on the
    next sync. Excluding them removes the curated-claim data hazard entirely."""
    md = node.metadata or {}
    if not md.get("curated"):
        return True
    return md.get("source") == "vault_reconcile"


def _looks_intact(text: str) -> bool:
    """A distilled note trustworthy enough to judge DELETIONS from: it still
    carries our marker and the entity header. A truncated/corrupted file fails
    this, so lost anchors are never mistaken for user deletions."""
    if "— Revien memory" not in text:
        return False
    return bool(re.search(
        rf"^{MARKER_KEY}:\s*{MARKER_VALUE}\s*$", text[:400], re.MULTILINE))


def _parse_claim_line(body: str) -> Tuple[Optional[str], str]:
    """(anchor_node_id | None, claim_text) from a list line's body (already
    stripped of the leading '- '). The anchor is the TRAILING <!--rv:--> only;
    provenance is stripped ONLY from anchored (Revien-rendered) lines, so a
    user-added line ending in its own *(...)* keeps its full text."""
    tm = _TRAILING_ANCHOR_RE.search(body)
    if tm:
        anchor = tm.group(1)
        body = _PROV_SUFFIX_RE.sub("", body[:tm.start()])
        return anchor, " ".join(body.split())
    return None, " ".join(body.split())


class VaultReconciler:
    """Read edited distilled notes and reconcile them INTO the graph.

    The three gestures, each routed through machinery that already exists:
      * EDIT   a claim's text  -> a curated node CORRECTS-supersedes the old
                                  one (the curated shield makes your word win).
      * DELETE a claim's line   -> the node is soft-invalidated (forgotten,
                                  reversibly — nothing is hard-deleted).
      * ADD    a line           -> a new curated claim, attached to the entity.

    Safe by construction:
      * Only notes with a manifest (ones Revien rendered) are touched.
      * Only DIFFS against the manifest snapshot act — an unedited machine
        claim hashes equal and is a no-op, so re-reading a note is inert (no
        echo loop).
      * Deletion is judged against what was last SHOWN (the manifest), never
        against the live graph, so claims ingested after the user last saw the
        note can't be false-rejected.
    """

    def __init__(self, store: GraphStore, vault_dir: str, folder: str = DEFAULT_FOLDER,
                 semantic=None):
        from revien.graph.operations import GraphOperations
        from revien.semantic.index import SemanticIndex
        self.store = store
        self.ops = GraphOperations(store)
        self.out_dir = Path(vault_dir) / folder
        # Corrections and additions must be SEMANTICALLY searchable, or the
        # user corrects their memory and the fix is second-class in recall.
        # add_node (used below) does not index — only the ingest pipeline does —
        # so the reconciler indexes its own new nodes. Self-disables without
        # the semantic extra.
        self.semantic = semantic if semantic is not None else SemanticIndex(store)

    def reconcile(self) -> Dict:
        corrected = added = forgotten = notes = 0
        for stem in self.store.list_manifest_note_stems():
            path = self.out_dir / f"{stem}.md"
            if not path.exists() or not VaultDistiller._has_marker(path):
                # File gone or its marker stripped: don't mass-invalidate on an
                # absent/foreign file. Next distill regenerates it.
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            c, a, f, new_text = self._reconcile_note(stem, text)
            # Surgical writeback ONLY: appends anchors to the lines the user
            # added, nothing else. It never regenerates the note, so free-text
            # the user wrote (annotations, their own headings, reminders) is
            # preserved. (A global redistill here destroyed all user free-text —
            # the fix that traded one data-loss bug for another.)
            if new_text is not None and new_text != text:
                try:
                    path.write_text(new_text, encoding="utf-8", newline="\n")
                except OSError:
                    pass
            corrected += c
            added += a
            forgotten += f
            if c or a or f:
                notes += 1

        return {
            "status": "ok",
            "notes_reconciled": notes,
            "corrected": corrected,
            "added": added,
            "forgotten": forgotten,
        }

    def _reconcile_note(self, stem: str, text: str) -> Tuple[int, int, int, Optional[str]]:
        manifest = {r["anchor_node_id"]: r for r in self.store.get_distill_manifest(stem)}
        entity_id = self._entity_for(stem, manifest, text)
        if entity_id is None:
            return (0, 0, 0, None)

        corrected = added = 0
        processed: set = set()  # dedup repeated anchors within one note
        # Deletion is judged on WHOLE-FILE anchor presence, not per-section
        # parsing — so a renamed/removed heading (which drops lines out of the
        # section scan) can never be mistaken for the user deleting the claims.
        present = set(_ANCHOR_RE.findall(text))
        current_section: Optional[str] = None
        out_lines: List[str] = []  # the file, with anchors appended to adds
        file_changed = False

        for line in text.splitlines():
            hm = _HEADING_RE.match(line)
            if hm:
                current_section = hm.group(1).strip()
                out_lines.append(line)
                continue
            cm = _CLAIM_LINE_RE.match(line)
            if not cm:
                out_lines.append(line)
                continue
            anchor, claim_text = _parse_claim_line(cm.group(1))
            if not claim_text:
                out_lines.append(line)
                continue

            if anchor:
                out_lines.append(line)
                if anchor not in manifest or anchor in processed:
                    continue  # unknown/duplicate anchor -> ignore (dedup)
                processed.add(anchor)
                row = manifest[anchor]
                if _hash(claim_text) != row["content_hash"]:
                    self._apply_edit(stem, entity_id, row, claim_text, current_section)
                    corrected += 1
            elif current_section in _SECTION_TYPE:
                # An anchorless line under a claim heading — a new claim to teach.
                # Anchor it IN PLACE so the next reconcile treats it as known
                # (no re-add), without regenerating the note.
                new_node = self._apply_add(stem, entity_id, claim_text, current_section)
                out_lines.append(f"{line} <!--rv:{new_node.node_id}-->")
                file_changed = True
                added += 1
            else:
                out_lines.append(line)  # anchorless line outside a claim section

        # DELETE: a manifest anchor absent from the ENTIRE file — but only when
        # the file is structurally intact, so a truncated/corrupt note never
        # mass-forgets.
        forgotten = 0
        if _looks_intact(text):
            for anchor, row in list(manifest.items()):
                if anchor in present:
                    continue
                # A claim rendered into another note too must not be nuked
                # globally by a delete in this one — just drop this note's row.
                if self.store.manifest_refs_elsewhere(row["current_node_id"], stem):
                    self.store.delete_manifest_row(stem, anchor)
                    continue
                self.ops.invalidate_node(
                    row["current_node_id"],
                    reason="user deleted from distilled note",
                    construct_id="vault_reconcile",
                )
                self.store.delete_manifest_row(stem, anchor)
                forgotten += 1

        new_text = ("\n".join(out_lines) + "\n") if file_changed else None
        return (corrected, added, forgotten, new_text)

    def _entity_for(self, stem: str, manifest: Dict, text: str) -> Optional[str]:
        """The note's subject entity id — from the manifest, else resolved from
        the `entity:` frontmatter label."""
        for row in manifest.values():
            return row["entity_id"]
        m = re.search(r"^entity:\s*(.+?)\s*$", text, re.MULTILINE)
        if m:
            ent = self.ops.find_node_by_label(m.group(1).strip(), node_type=NodeType.ENTITY)
            if ent is not None:
                return ent.node_id
        return None

    def _new_claim_node(self, entity_id: str, text: str, section: Optional[str]) -> Node:
        from datetime import datetime, timezone
        from revien.graph.schema import Edge, EdgeType, SourceType
        node_type = _SECTION_TYPE.get(section or "", NodeType.FACT)
        now = datetime.now(timezone.utc)
        node = self.store.add_node(Node(
            node_type=node_type,
            label=text[:80],
            content=text,
            source_type=SourceType.EXTRACTED,
            confidence=1.0,
            created_at=now,
            last_accessed=now,
            recorded_at=now,
            metadata={"curated": True, "source": "vault_reconcile"},
        ))
        # Attach to the note's entity so it renders on the entity's note and is
        # reachable in the graph walk.
        self.store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=entity_id,
            target_node_id=node.node_id,
            weight=0.9,  # human-authored, high trust
        ))
        # Index for semantic recall (add_node does not; the pipeline would, but
        # reconcile bypasses it). No-op without the semantic extra.
        if self.semantic.is_enabled:
            self.semantic.index_node(node.node_id, node.label, node.content)
        return node

    def _apply_edit(self, stem, entity_id, row, claim_text, section) -> None:
        from revien.graph.schema import Edge, EdgeType
        new_node = self._new_claim_node(entity_id, claim_text, section)
        old_id = row["current_node_id"]
        # The correction supersedes the claim it replaces.
        self.store.add_edge(Edge(
            edge_type=EdgeType.CORRECTS,
            source_node_id=new_node.node_id,
            target_node_id=old_id,
            weight=1.0,
        ))
        self.ops.invalidate_node(
            old_id, reason=f"corrected in distilled note (superseded_by {new_node.node_id})",
            construct_id="vault_reconcile",
        )
        # Anchor (file join key) is unchanged; point it at the correction so the
        # next reconcile sees no phantom diff.
        self.store.update_manifest_row(
            stem, row["anchor_node_id"], new_node.node_id, _hash(claim_text)
        )

    def _apply_add(self, stem, entity_id, claim_text, section) -> Node:
        new_node = self._new_claim_node(entity_id, claim_text, section)
        self.store.add_manifest_row(
            stem, entity_id, new_node.node_id, new_node.node_id,
            _hash(claim_text), section or "",
        )
        return new_node
