"""
Revien Obsidian Vault Adapter — a vault is a pre-built knowledge graph.

An Obsidian vault isn't just a purer corpus — it's graph structure a human
already drew, which is exactly what Revien's own extractor is worst at
manufacturing (the miss taxonomy's `disconnected` bucket):

    [[wikilinks]]      -> real CONTEXT->ENTITY edges (author-drawn, weight 0.8)
    headings           -> chunk boundaries (no giant note-blobs)
    frontmatter dates  -> recorded_at (content time for recency)
    #tags / tags:      -> metadata (and searchable content)
    note title         -> the note's anchor ENTITY

Every unit ingests as CURATED: full confidence, and the CSL curated shield
guarantees a machine-side claim can never silently auto-supersede it — a
contradiction with human-curated memory goes to the candidate queue for
human review instead.

SCOPE NOTE (AND, not OR): the vault is a SECOND corpus beside conversational
memory. Its recall characteristics get their own eval and their own number —
never blended with the LoCoMo conversational-recall figures.

Known v1 trade: like the file watcher, a re-edited note re-ingests as new
CONTEXT units (mtime-gated, so only changed notes). Reconciling edited chunks
against their prior version is future work.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import RevienAdapter

# Vault folders that are never content.
EXCLUDED_DIRS = {".obsidian", ".trash", ".git"}

# [[target]], [[target|alias]], [[target#heading]], [[target#heading|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]\[#|]+)(?:#[^\]\[|]*)?(?:\|[^\]\[]*)?\]\]")
# Inline #tag — not headings (require non-space-# boundary and a letter start).
_INLINE_TAG_RE = re.compile(r"(?<![\w#])#([A-Za-z][\w/-]*)")
# ATX headings, any level.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# Frontmatter block at the very top of the note.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_DATE_KEYS = ("date", "created", "updated", "modified")


def _parse_frontmatter(text: str) -> Tuple[Dict, str]:
    """Extract a minimal frontmatter dict (tags + first parseable date) and
    return (frontmatter, body). Deliberately not a YAML parser — no new
    dependency for the two fields we need; unrecognized lines are ignored."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm: Dict = {}
    tags: List[str] = []
    collecting_tags = False
    for line in m.group(1).splitlines():
        stripped = line.strip()
        if collecting_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip("\"'#"))
                continue
            collecting_tags = False
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "revien":
            fm["revien"] = value.strip().strip("\"'").lower()
        elif key == "tags" or key == "tag":
            if not value:
                collecting_tags = True  # block list follows
            else:
                # Inline list: [a, b] or comma/space separated.
                cleaned = value.strip("[]")
                tags.extend(
                    t.strip().strip("\"'#") for t in re.split(r"[,\s]+", cleaned) if t.strip()
                )
        elif key in _DATE_KEYS and "date" not in fm:
            dt = _parse_date(value)
            if dt is not None:
                fm["date"] = dt
    if tags:
        fm["tags"] = [t for t in tags if t]
    return fm, text[m.end():]


def _parse_date(value: str) -> Optional[datetime]:
    """Best-effort ISO-ish date parse -> aware UTC datetime, else None."""
    value = value.strip().strip("\"'")
    if not value:
        return None
    for candidate in (value, value.replace(" ", "T", 1)):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def chunk_note(body: str, note_title: str) -> List[Tuple[str, str]]:
    """Split a note body into (heading, section_text) chunks by ATX headings.

    Content before the first heading is the note's own chunk (heading = the
    note title). A note with no headings is a single chunk. Empty sections
    are dropped."""
    matches = list(_HEADING_RE.finditer(body))
    chunks: List[Tuple[str, str]] = []
    preamble = body[: matches[0].start()] if matches else body
    if preamble.strip():
        chunks.append((note_title, preamble.strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section = body[m.end():end].strip()
        if section:
            chunks.append((m.group(2).strip(), section))
    return chunks


class ObsidianVaultAdapter(RevienAdapter):
    """Ingest an Obsidian vault: one curated unit per heading-section, with
    the author's own link structure transcribed into the graph."""

    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir)

    async def health_check(self) -> bool:
        return self.vault_dir.exists() and self.vault_dir.is_dir()

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Scan the vault for notes modified since `since`; return one item
        per heading-chunk, ready for the ingestion pipeline."""
        if not self.vault_dir.exists():
            return []
        since_ts = since.timestamp()
        items: List[Dict] = []
        for path in sorted(self.vault_dir.rglob("*.md")):
            rel = path.relative_to(self.vault_dir)
            if any(part in EXCLUDED_DIRS for part in rel.parts):
                continue
            if not path.is_file() or path.stat().st_mtime <= since_ts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            items.extend(self._note_to_items(rel, text, path.stat().st_mtime))
        return items

    def _note_to_items(self, rel_path: Path, text: str, mtime: float) -> List[Dict]:
        note_title = rel_path.stem
        fm, body = _parse_frontmatter(text)
        # Distill-out exclusion: notes Revien itself wrote (revien: derived)
        # are views OF the graph and must never feed back INTO it. Marker
        # check, not folder-name check — survives the user renaming or moving
        # the distill folder.
        if fm.get("revien") == "derived":
            return []
        # Content time: frontmatter date beats file mtime (an old note copied
        # into the vault yesterday is still an old memory).
        ts = fm.get("date") or datetime.fromtimestamp(mtime, tz=timezone.utc)
        note_tags = list(fm.get("tags", []))

        items: List[Dict] = []
        for heading, section in chunk_note(body, note_title):
            wikilinks = [w.strip() for w in _WIKILINK_RE.findall(section) if w.strip()]
            tags = sorted(set(note_tags) | set(_INLINE_TAG_RE.findall(section)))
            # The note's own title anchors every chunk; wikilink targets are
            # the author-drawn edges out. Tags ride as entity links too — a
            # tag IS a curated topic assignment.
            links = [note_title] + wikilinks + tags
            label_line = note_title if heading == note_title else f"{note_title} — {heading}"
            items.append({
                "content": f"{label_line}\n{section}",
                "content_type": "note",
                "timestamp": ts.isoformat(),
                "source_id": f"vault:{rel_path.as_posix()}#{_slug(heading)}",
                "links": links,
                "curated": True,
                "metadata": {
                    "adapter": "obsidian",
                    "vault": str(self.vault_dir),
                    "note": rel_path.as_posix(),
                    "heading": heading,
                    "tags": tags,
                    "wikilinks": wikilinks,
                },
            })
        return items
