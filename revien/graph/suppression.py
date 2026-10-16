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

A tombstone holds no content. Lifting it is an explicit act (restore or a
graph import with the override flag) and is audited.
"""

from typing import Dict, List, Optional, Tuple

from revien.graph.schema import NodeType

KIND_SOURCE_ID = "source_id"
KIND_INGEST_KEY = "ingest_key"

# source_id prefixes whose conventions are per-unit (verified against the
# adapters that produce them, see revien/graph/origin.py).
UNIT_SOURCE_PREFIXES = (
    "claude-code:",          # claude-code:{project}:{session_stem}
    "codex:",                # codex:{project}:{session_stem}
    "openai:conversation:",  # openai:conversation:{conv_id}
    "file:",                 # file:{name}        (file watcher)
    "vault:",                # vault:{relpath}#{heading slug}  (obsidian)
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
    key = ((metadata or {}).get("ingest_key") or "")
    if isinstance(key, str):
        key = key.strip()
    else:
        key = ""
    # Only context nodes are gated by ingest_key at ingest time (skills use
    # their own store-level path, which is not gated), so only they bind it.
    if key and node_type == NodeType.CONTEXT:
        found.append((key, KIND_INGEST_KEY))
    if found:
        return found, None
    if not sid:
        return [], "node has no source_id"
    if node_type != NodeType.CONTEXT and (metadata or {}).get("ingest_key"):
        return [], "skill/keyed non-context nodes are not gated at ingest"
    return [], (f"source_id {sid!r} is a shared channel, not one unit; "
                "tombstoning it would block everything it sends")
