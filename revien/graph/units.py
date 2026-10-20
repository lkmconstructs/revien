"""Unit identities: what a captured thing belongs to, and how to recognise it.

A *unit* is one source item with a stable identity (docs/FORGET_SPEC.md §3):
a Claude Code or Codex session, an OpenAI/ChatGPT/Claude conversation, a
watched file (by real path), a vault section, or a skill. Everything here is
pure naming and file identity. It touches no database, so the store, the
migration backfill, the gate and the forget engine all agree on the same
answer for the same input.

Unit id shapes (the prefix says which kind):

    session:<runtime>:<session key>   Claude Code / Codex session   (file)
    vault:<relpath>#<slug>            Obsidian vault section        (file)
    file:<real path, case-folded>     watched file                  (file)
    skill:<real path>                 SKILL.md                      (file)
    item:<stable key>                 an import / conversation      (no file)
    capture:<uuid>                    one capture through a shared
                                      channel (mcp, hermes, ollama,
                                      langchain, api feeds, the CLI):
                                      NOT a stable identity
    unproven:<likely unit>            a marker, never an owner: the
                                      ownership of a node that predates
                                      provenance could not be proven
"""

import os
import unicodedata
from typing import List, Optional

UNPROVEN_PREFIX = "unproven:"
CAPTURE_PREFIX = "capture:"

# Unit kinds whose identity is a file on disk. A file-backed unit is
# suppressed until the file changes; every other stable unit has no file
# behind it and is suppressed for good.
FILE_PREFIXES = ("session:", "vault:", "file:", "skill:")


def session_unit(runtime: str, session_key: str) -> str:
    return f"session:{runtime}:{session_key}"


def item_unit(key: str) -> str:
    return f"item:{key}"


def is_unproven(unit_id: str) -> bool:
    return unit_id.startswith(UNPROVEN_PREFIX)


def is_shared_channel(unit_id: str) -> bool:
    return unit_id.startswith(CAPTURE_PREFIX)


def is_file_backed(unit_id: str) -> bool:
    return unit_id.startswith(FILE_PREFIXES)


def describe(unit_id: str) -> str:
    """Human wording for a unit in messages ('session X', 'conversation Y')."""
    if unit_id.startswith("session:"):
        _, runtime, key = (unit_id.split(":", 2) + ["", ""])[:3]
        return f"{runtime} session {key}"
    if unit_id.startswith("item:"):
        return f"import {unit_id[5:]}"
    if unit_id.startswith("vault:"):
        return f"vault section {unit_id[6:]}"
    if unit_id.startswith("file:"):
        return f"file {unit_id[5:]}"
    if unit_id.startswith("skill:"):
        return f"skill {unit_id[6:]}"
    if unit_id.startswith(CAPTURE_PREFIX):
        return "a shared-channel capture"
    return unit_id


def node_likely_unit(
    origin_runtime: Optional[str], session_key: Optional[str],
    source_id: Optional[str], ingest_key: Optional[str],
) -> Optional[str]:
    """The unit a node PROBABLY belongs to, from the fields a node already
    carries. This is a hint for listing and for the upgrade marker. It is not
    proof and nothing is ever deleted because of it."""
    if ingest_key:
        if ingest_key.startswith("skill:"):
            return ingest_key
        if origin_runtime in ("claude-code", "codex") and session_key:
            return session_unit(origin_runtime, session_key)
        return item_unit(ingest_key)
    if origin_runtime in ("claude-code", "codex") and session_key:
        return session_unit(origin_runtime, session_key)
    src = source_id or ""
    if src.startswith(("openai:conversation:", "chatgpt:conversation:",
                       "claude:conversation:", "readwise:")):
        # an importer's source id is its ingest key: the conversation or
        # highlight id from the export
        return item_unit(src)
    if src.startswith("vault:"):
        return src
    if src.startswith("skill:"):
        return src
    return None


def real_path(path: str) -> str:
    """Absolute path with symlinks resolved (NFC-normalised). A path that
    does not exist still resolves lexically."""
    return unicodedata.normalize("NFC", os.path.realpath(os.fspath(path)))


def path_unit_key(path: str) -> str:
    """The case-folded real path used in file: unit ids and path keys."""
    return os.path.normcase(real_path(path)).casefold()


def identity_keys(path: str) -> List[str]:
    """Every key that names this file: its case-folded real path and, where
    the platform reports one, its device and inode (so a hard link, or the
    same file moved, matches)."""
    keys = ["path:" + path_unit_key(path)]
    try:
        st = os.stat(real_path(path))
        if st.st_ino:
            keys.append(f"inode:{st.st_dev}:{st.st_ino}")
    except OSError:
        pass
    return keys


def is_real_file(path: str) -> bool:
    """True only when ``path`` exists on disk as a regular file. A path that
    does not is never trusted as evidence about any file."""
    try:
        return os.path.isfile(real_path(path))
    except (OSError, ValueError):
        return False
