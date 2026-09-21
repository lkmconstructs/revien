"""
Revien Skills — minimal frontmatter parser.

Deliberately NOT a YAML parser — same call as adapters/obsidian.py's own
`_parse_frontmatter` (obsidian.py:50-89): no new dependency (no pyyaml) for
the handful of keys a SKILL.md actually needs. This module reads exactly
four: name, description, triggers, version. Anything else in the
frontmatter block is ignored, not stored.

Supported shapes (authors mix these freely):
    name: ponytail
    description: One line.
    description: >
      Folded multi-line description, continued
      on indented lines until the next key or EOF.
    triggers: foo, bar, baz
    triggers: [foo, bar, baz]
    triggers:
      - foo
      - bar
    version: "1.2"

A file with no ``---`` frontmatter block returns ``({}, text)`` unchanged —
the whole file is then treated as the body.
"""

import re
from typing import Dict, List, Tuple

# Frontmatter block at the very top of the file — same shape as Obsidian's.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_KNOWN_KEYS = ("name", "description", "triggers", "version")
_BLOCK_SCALAR_MARKERS = (">", "|", ">-", "|-", ">+", "|+")


def parse_frontmatter(text: str) -> Tuple[Dict, str]:
    """Extract {name, description, triggers, version} (only the keys
    present) from the leading frontmatter block; return (frontmatter, body).

    ``triggers`` is always a list (possibly empty if present-but-blank; the
    key is simply absent from the dict if never set). ``body`` is
    everything after the closing ``---`` line, unmodified."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    fm: Dict = {}
    lines = m.group(1).splitlines()
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if not stripped or ":" not in stripped:
            i += 1
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key not in _KNOWN_KEYS:
            i += 1
            continue

        if key == "triggers":
            fm["triggers"], i = _parse_triggers(value, lines, i)
            continue

        if not value or value in _BLOCK_SCALAR_MARKERS:
            fm[key], i = _parse_block_scalar(lines, i)
            continue

        fm[key] = value.strip("\"'")
        i += 1

    return fm, text[m.end():]


def _parse_block_scalar(lines: List[str], i: int) -> Tuple[str, int]:
    """Fold indented continuation lines following a bare/`>`/`|` key into one
    space-joined string (blank lines are just skipped, not folded verbatim —
    this is a folded-scalar approximation, not literal YAML `|`)."""
    n = len(lines)
    collected: List[str] = []
    j = i + 1
    while j < n and (lines[j].strip() == "" or lines[j].startswith((" ", "\t"))):
        if lines[j].strip():
            collected.append(lines[j].strip())
        j += 1
    return " ".join(collected).strip(), j


def _parse_triggers(value: str, lines: List[str], i: int) -> Tuple[List[str], int]:
    """`triggers:` is either an inline comma/bracket list on the same line,
    or a following block list (`  - item`, obsidian-tags style)."""
    n = len(lines)
    if value:
        cleaned = value.strip("[]")
        items = [t.strip().strip("\"'") for t in cleaned.split(",") if t.strip()]
        return items, i + 1

    items: List[str] = []
    j = i + 1
    while j < n and lines[j].strip().startswith("- "):
        items.append(lines[j].strip()[2:].strip().strip("\"'"))
        j += 1
    return items, j
