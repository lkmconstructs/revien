"""
Shared date rendering for memory blocks shown to a consuming model.

Product promise: a model is never told something false about a memory. A
bracketed date plus the DATE_NOTE ("when each memory was said") is only true
when the stored recorded_at really is content/capture/import time. A date that
came from a FILE MTIME is when a file was last written, not when anything was
said, so it is never rendered and never triggers the note.

recorded_at_source vocabulary (IngestionInput.timestamp_source):
    content  the unit's own timestamp (a message, a transcript, a frontmatter date)
    capture  stamped at the moment Revien captured it live
    import   carried in from an import (the exported record's own time)
    mtime    file modification time - NOT a said-at time, never rendered

A result with no recorded source is unknown: no date, no note. Rows ingested
before the source existed are labeled by the user_version 4 backfill
(graph/origin.derive_recorded_at_source); those it cannot derive stay
unlabeled and undated.
"""

from typing import Optional

DATE_NOTE = (
    "(dates in brackets are when each memory was said; "
    "resolve 'yesterday' etc. against them)"
)

TIMESTAMP_SOURCES = ("content", "capture", "mtime", "import")
DATED_SOURCES = ("content", "capture", "import")


def said_date(recorded_at: Optional[str], source: Optional[str] = None) -> Optional[str]:
    """YYYY-MM-DD of when a memory was said, or None when unknown or when the
    only date we hold is a file mtime. recorded_at keeps the speaker's own UTC
    offset (see engine._iso_utc), so its first ten characters are the
    speaker's calendar day."""
    if not recorded_at or source not in DATED_SOURCES:
        return None
    return recorded_at[:10]
