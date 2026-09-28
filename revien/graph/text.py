"""
Revien text utilities — leaf module, zero adapter imports.

Anything importers/base.py needs (or any other zero-network code path)
belongs here rather than in revien.adapters.*: importing that package's
__init__ pulls in generic_api/ollama_adapter, which import httpx. An
importer that only wants a slug must never drag httpx along for the ride.
"""

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str, empty: str = "section") -> str:
    """Lowercase, non-alphanumeric runs collapsed to a single '-', leading
    and trailing '-' trimmed. Empty/all-punctuation input falls back to
    `empty` (obsidian.py wants 'section', importers/base.py wants
    'untitled' — each caller picks its own)."""
    return _SLUG_RE.sub("-", (text or "").lower()).strip("-") or empty
