"""
Revien Readwise Importer — turns a Readwise highlights CSV export into one
ImportUnit per highlight, ready for revien.importers.base.run_import.

Export shape (Readwise's own "Export" -> CSV, undocumented but stable): a
header row, then one row per highlight. Column order is NOT guaranteed
(Readwise has reshuffled it across export-tool versions), so this reads by
header name via csv.DictReader and tolerates any of the optional columns
being absent — only Highlight is required; a row without one is dropped.
"""

import csv
import hashlib
import io
import re
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from revien.importers.base import ImportUnit, slugify

_TAG_SPLIT_RE = re.compile(r"[,\s]+")


def _parse_highlighted_at(value: Optional[str]) -> Optional[datetime]:
    """ISO 8601 (with or without a trailing 'Z'), or Readwise's older
    '%Y-%m-%d %H:%M:%S' export format. Naive results are UTC (Readwise
    timestamps are always UTC on the wire); missing/unparseable -> None,
    never guessed."""
    if not value or not value.strip():
        return None
    text = value.strip()
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_tags(*raw_values: Optional[str]) -> List[str]:
    """Readwise carries tags in two columns (highlight-level "Tags",
    book-level "Document tags") — merge both. Split on comma OR
    whitespace (Readwise has exported both ", "-joined and space-joined
    lists across versions), strip a leading '#', drop empties, dedup
    while keeping first-seen order."""
    seen = set()
    tags: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        for piece in _TAG_SPLIT_RE.split(raw.strip()):
            tag = piece.strip().lstrip("#").strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def iter_units(path: str) -> Iterator[ImportUnit]:
    """One ImportUnit per highlight row in a Readwise CSV export.

    Readwise CSVs are never zipped, but base.open_export handles a direct
    .csv path the same as a direct .json path — read straight through it
    here rather than duplicating that branch."""
    from revien.importers.base import open_export

    raw = open_export(path)
    text = raw.decode("utf-8-sig")  # Excel-exported CSVs often carry a BOM
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        row = row or {}
        highlight = (row.get("Highlight") or "").strip()
        if not highlight:
            continue

        book_title = (row.get("Book Title") or "").strip()
        author = (row.get("Book Author") or "").strip()
        note = (row.get("Note") or "").strip()
        location = (row.get("Location") or "").strip()

        content = highlight
        if note:
            content = f"{highlight}\n\nNote: {note}"

        digest = hashlib.sha1(f"{highlight}{location}".encode("utf-8")).hexdigest()[:12]
        book_slug = slugify(book_title) if book_title else "untitled"
        source_id = f"readwise:{book_slug}:{digest}"

        metadata: Dict = {
            "title": book_title or None,
            "author": author or None,
            "location": location or None,
            "tags": _parse_tags(row.get("Tags"), row.get("Document tags")),
        }

        yield ImportUnit(
            source_id=source_id,
            ingest_key=source_id,
            content=content,
            content_type="document",
            timestamp=_parse_highlighted_at(row.get("Highlighted at")),
            origin_runtime="readwise",
            origin_source="import",
            session_key=book_slug if book_title else None,
            links=[book_title] if book_title else [],
            metadata=metadata,
        )
