"""Ingestion suppression tombstones: which stable identity does a deleted node
bind to?

An explicit user forget/delete leaves a content-free row (identity, kind,
timestamp, reason, actor) so ordinary ingestion can never silently capture
that unit again. This module is the ONE place that decides what the identity
of a node's source is, and when there is none.

Identity kinds:
    source_id   the node's source_id, but ONLY when that string names ONE unit
                (a session file, a conversation, a watched file, a vault
                section). Shared source ids such as "mcp", "hermes",
                "langchain", "ollama_chat" or "api:<feed url>" name a whole
                channel; tombstoning them would silently block everything the
                channel ever sends, so they are never tombstoned and the
                caller is told so.
    ingest_key  the stable key a keyed adapter stamps on its context node
                (equal to source_id for claude-code/codex).
    session_id  claude-code / codex only: the session UUID the runtime itself
                assigned, independent of the project folder. Survives a moved
                project or a moved/renamed session file. Only bound when a
                real UUID is present.

A file-watcher file's identity is its BASENAME (``file:<name>``), which is
also the source_id its nodes carry, and, since the context node records where
it was read from, ALSO its real path (``file_path`` row, an absolute path with
symlinks resolved). The basename row alone would block every unrelated file
with the same name (every ``README.md``) for as long as the tombstone stands,
so a path-qualified tombstone blocks like this (``Store.find_ingest_suppression``):

* the same real path: always skipped;
* the same basename at another path: skipped only while a recorded path no
  longer exists on disk (the forgotten file was moved or deleted, and this may
  be it); allowed while every recorded path still exists (this is a different
  file);
* an item with no path (a graph import of an extracted node, an adapter that
  sends none): skipped, because nothing can tell it from the forgotten file.

Bounds, all disclosed to the user: a tombstone written before the context
node recorded its path (or for an extracted node of several same-named files)
has only the name and stays name-wide; the paths of forgotten files are stored
in the suppression (a path is as revealing as the name already stored);
after the forgotten file is deleted or moved, a new unrelated file with the
same name is skipped until the user restores that path; the same file moved
and later edited under a new name is a gap, as before. A moved file keeps
matching by name. Both identities stay content-free.

Gaps (no stable content-free identity exists): a file-watcher file renamed
(identity is its basename), a vault note renamed or moved (its relative
path), a skill moved (its resolved path), a Claude Code / Codex session that
the runtime copies into a NEW session id (resume/fork), and any shared channel.

A tombstone holds no content. Lifting it is an explicit act (restore or a
graph import with the override flag) and is audited.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from revien.graph.origin import derive_origin
from revien.graph.schema import NodeType

KIND_SOURCE_ID = "source_id"
KIND_INGEST_KEY = "ingest_key"
KIND_SESSION_ID = "session_id"
KIND_FILE_PATH = "file_path"

_UUID = (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
         r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CLAUDE_STEM = re.compile(rf"^({_UUID})$")
# Codex rollout stems: rollout-<local timestamp>-<thread uuid>
_CODEX_STEM = re.compile(rf"^rollout-.*?({_UUID})$")


def session_identity_for(source_id: str) -> Optional[str]:
    """The project-independent identity of a Claude Code / Codex session, or
    None. Both runtimes name a session file with an id THEY assign at session
    creation (a UUID; for Codex the UUID is the tail of ``rollout-<ts>-<uuid>``).
    The id does not depend on the project directory or on which dated folder
    the file sits in, so it survives a moved project, a moved file, or a
    renamed project folder. It is only used when a real UUID is present: a
    free-form stem ("main", "notes") is not globally unique, and binding it
    across projects would block unrelated sessions. Content-free."""
    sid = (source_id or "").strip()
    if not sid.startswith(("claude-code:", "codex:")):
        return None
    origin = derive_origin(sid)
    if not origin.session:
        return None
    if origin.runtime == "claude-code":
        m = _CLAUDE_STEM.match(origin.session)
    elif origin.runtime == "codex":
        m = _CODEX_STEM.match(origin.session)
    else:
        return None
    if not m:
        return None
    return f"{origin.runtime}-session:{m.group(1).lower()}"


def parent_session_identity(metadata: Optional[Dict]) -> Optional[str]:
    """The session identity named by a ``parent_session_id`` metadata field,
    or None. A Claude Code sub-agent transcript
    (``<session uuid>/subagents/agent-<id>.jsonl``) has its own file stem but
    every entry carries the PARENT's ``sessionId``; the adapter copies that
    one id (never any text) into metadata so a forgotten session's sub-agent
    files are covered by the session tombstone. Only a real UUID counts."""
    raw = (metadata or {}).get("parent_session_id")
    if not isinstance(raw, str):
        return None
    m = _CLAUDE_STEM.match(raw.strip())
    if not m:
        return None
    return f"claude-code-session:{m.group(1).lower()}"


def file_path_of(source_id: str, metadata: Optional[Dict]) -> Optional[str]:
    """The real path of a file-watcher file, or None. The adapter sends it as
    ``path``; the pipeline keeps it on the context node as ``source_path``.
    Symlinks are resolved so two spellings of one directory are one path."""
    if not (source_id or "").startswith("file:"):
        return None
    raw = (metadata or {}).get("source_path") or (metadata or {}).get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return os.path.realpath(raw.strip())


def unit_key(identity: str) -> str:
    """Rows that name the same unit share a key (a source_id row and the
    session_id row derived from it), so restore lifts them together."""
    return session_identity_for(identity) or identity

# source_id prefixes whose conventions are per-unit (verified against the
# adapters that produce them, see revien/graph/origin.py).
UNIT_SOURCE_PREFIXES = (
    "claude-code:",          # claude-code:{project}:{session_stem}
    "codex:",                # codex:{project}:{session_stem}
    "openai:conversation:",  # openai:conversation:{conv_id}
    "file:",                 # file:{name}        (file watcher; the BASENAME only)
    "skill:",                # skill:{resolved SKILL.md path}
    "vault:",                # vault:{relpath}#{heading slug}  (obsidian)
    "skill-proposal:",       # skill-proposal:{pattern hash}  (engine proposals)
)

REASON_RETENTION = "retention_expire"


def is_unit_source_id(source_id: str) -> bool:
    for prefix in UNIT_SOURCE_PREFIXES:
        if source_id.startswith(prefix) and len(source_id) > len(prefix):
            return True
    return False


def identities_for(
    source_id: str, metadata: Optional[Dict], node_type: NodeType,
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """(identities, gap). identities is a list of (identity, kind); gap is a
    short reason string when NO identity could be bound (else None)."""
    found: List[Tuple[str, str]] = []
    sid = (source_id or "").strip()
    if sid and is_unit_source_id(sid):
        found.append((sid, KIND_SOURCE_ID))
    path = file_path_of(sid, metadata)
    if path:
        found.append((path, KIND_FILE_PATH))
    key = ((metadata or {}).get("ingest_key") or "")
    if isinstance(key, str):
        key = key.strip()
    else:
        key = ""
    # Context nodes (pipeline) and skill nodes (skill ingest) are gated by
    # ingest_key at ingest time, so only they bind it.
    if key and node_type in (NodeType.CONTEXT, NodeType.SKILL):
        found.append((key, KIND_INGEST_KEY))
    session = session_identity_for(sid)
    if session:
        found.append((session, KIND_SESSION_ID))
    if found:
        return found, None
    if not sid:
        return [], "node has no source_id"
    return [], (f"source_id {sid!r} is a shared channel, not one unit; "
                "tombstoning it would block everything it sends")


# Identity prefixes whose identity is a NAME or PATH the user can change.
_RENAMEABLE = {
    "file:": "file name",
    "vault:": "note path",
    "skill:": "skill path",
}


def notices_for(
    identities: List[Tuple[str, str]], gap: Optional[str], count: int = 1,
    whole_unit: bool = False,
) -> List[str]:
    """Plain-language blast radius of a forget/delete, for CLI, REST and MCP
    responses. States only what the tombstone provably does."""
    if not identities:
        return [
            "Deleted now, but eligible for recapture because the source "
            f"supplied no stable item identity ({gap or 'no source identity'})."
        ]
    units = sorted({i for i, k in identities if k == KIND_SESSION_ID}
                   or {i for i, k in identities if k == KIND_FILE_PATH}
                   or {i for i, k in identities if k == KIND_SOURCE_ID}
                   or {i for i, _ in identities})
    what = "this node" if count == 1 else f"{count} nodes"
    if whole_unit:
        # A whole-session cascade: the exclusive nodes went with it, so the
        # "siblings remain" sentence would be false. What was kept is stated
        # separately by the forget response.
        out = [
            f"This forgets {what} and suppresses all future imports from "
            f"{', '.join(units)}. 'revien restore' lifts the suppression."
        ]
    else:
        out = [
            f"This forgets {what} and suppresses all future imports from "
            f"{', '.join(units)}. Sibling facts already captured from that "
            "source remain in the graph; forget them too if you want them gone. "
            "'revien restore' lifts the suppression."
        ]
    kinds = {k for _, k in identities}
    if any(k == KIND_SOURCE_ID and i.startswith("skill-proposal:")
           for i, k in identities):
        out.append(
            "This is an engine skill proposal: it is not proposed again "
            "while the suppression stands. Its identity is a hash of the "
            "pattern's step labels, so a pattern with different steps is a "
            "different proposal. The actions it was drawn from stay in the "
            "graph; forget them to stop the pattern being detected.")
    if KIND_SESSION_ID not in kinds:
        for ident, kind in identities:
            for prefix, noun in _RENAMEABLE.items():
                if kind == KIND_SOURCE_ID and ident.startswith(prefix):
                    if ident.startswith("file:"):
                        out.append(
                            "The suppression is bound to the file name and "
                            "path; if the file is renamed it is a different "
                            "identity and may be captured again.")
                    else:
                        out.append(
                            f"The suppression is bound to the {noun}; if it "
                            "is renamed or moved it is a different identity "
                            "and may be captured again.")
                    if ident.startswith("file:"):
                        paths = [i for i, k in identities if k == KIND_FILE_PATH]
                        if paths:
                            out.append(
                                f"This file is blocked by its path "
                                f"({', '.join(paths)}), which is stored in the "
                                "suppression. A file with the same name in "
                                "another folder is still captured while this "
                                "path exists; if this file is moved or deleted, "
                                "a file with the same name is skipped too (it "
                                "may be this one). 'revien restore <path>' "
                                "lifts only this file. 'revien suppressions' "
                                "lists what was skipped.")
                        else:
                            out.append(
                                "No path was recorded for this item, so the "
                                "suppression is the file's base name only: a "
                                "file with the same name anywhere in a watched "
                                "folder is skipped too, and the same file moved "
                                "to another folder stays skipped. "
                                "'revien suppressions' lists what was skipped.")
            if kind == KIND_SOURCE_ID and ident.startswith(
                    ("claude-code:", "codex:")):
                out.append(
                    "This session id is not a UUID, so the suppression is "
                    "bound to its project folder as well; moving the "
                    "session to another project may capture it again.")
    return out
