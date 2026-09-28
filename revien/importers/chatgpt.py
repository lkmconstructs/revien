"""
Revien ChatGPT Importer — turns a ChatGPT conversations.json export into
one ImportUnit per conversation, ready for revien.importers.base.run_import.

Export shape (OpenAI's own, undocumented but stable): an array of
conversation objects, each {title, create_time, update_time, mapping,
current_node}. `mapping` is NOT the displayed transcript — it's every
branch the user ever generated (edits, regenerations) as a tree keyed by
node id, each entry {id, parent, children[], message}. The single thread
ChatGPT actually shows you is the parent chain from `current_node` back to
the root. Anything off that path is an abandoned branch and must not be
ingested — see _displayed_path below.

revien/adapters/openai_adapter.py already walks this same mapping, but its
walk isn't a pure function: _ingest_conversation_data does the tree walk
and the store writes (self.store.add_node/add_edge) interleaved in the
same two passes (openai_adapter.py:352-428), and it ingests the WHOLE tree
as separate message nodes rather than the one displayed thread. Pulling a
pure helper out of it would mean either reaching into a private method
that's written to also mutate the store, or leaving it half-extracted —
not worth it for what's a ~20-line walk. Written fresh here; the adapter
is untouched and still owns its own callers.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from revien.importers.base import ImportUnit, render_turns

_KEPT_ROLES = {"user", "assistant"}


def _epoch_to_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _conversation_id(conv: Dict) -> str:
    """conversation.id, else conversation_id, else a stable hash of
    title+create_time — the same three-deep fallback openai_adapter.py
    uses (there: _extract_conversation_id), so a conversation missing
    both id fields still gets a REPEATABLE source_id across re-imports."""
    if conv.get("id"):
        return str(conv["id"])
    if conv.get("conversation_id"):
        return str(conv["conversation_id"])
    basis = f"{conv.get('title', '')}:{conv.get('create_time', '')}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _message_text(message: Dict) -> Optional[str]:
    """String parts only, joined; None if there's nothing keepable (empty,
    or every part is a non-string medium like an image asset pointer)."""
    parts = (message.get("content") or {}).get("parts") or []
    strings = [p for p in parts if isinstance(p, str) and p.strip()]
    if not strings:
        return None
    return "\n".join(strings)


def _displayed_path(mapping: Dict, current_node: Optional[str]) -> List[str]:
    """Node ids from root to the displayed leaf, in that order.

    Normal case: current_node is set and present in mapping — follow
    `parent` links back to the root, then reverse. Fallback (current_node
    missing or stale): no way to know which branch was "current", so take
    the longest root-to-leaf path — the fullest conversation on record,
    which is the closest honest guess without a current_node to trust.
    """
    if current_node and current_node in mapping:
        path = []
        node_id = current_node
        seen = set()
        while node_id is not None and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            path.append(node_id)
            node_id = mapping[node_id].get("parent")
        path.reverse()
        return path

    # Fallback: longest root-to-leaf path. A leaf has no children (or an
    # empty children list); walk each leaf back to its root via `parent`
    # and keep the longest. Iterated in mapping's own (insertion) order,
    # so ties resolve deterministically to the first-seen longest path.
    leaves = [
        node_id for node_id, node in mapping.items()
        if not node.get("children")
    ]
    best: List[str] = []
    for leaf in leaves:
        path = []
        node_id = leaf
        seen = set()
        while node_id is not None and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            path.append(node_id)
            node_id = mapping[node_id].get("parent")
        path.reverse()
        if len(path) > len(best):
            best = path
    return best


def iter_units(path: str) -> Iterator[ImportUnit]:
    """One ImportUnit per conversation in a ChatGPT export (.zip or a
    bare conversations.json)."""
    from revien.importers.base import open_export

    raw = open_export(path)
    data = json.loads(raw)
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        conv_id = _conversation_id(conv)
        mapping = conv.get("mapping") or {}
        title = conv.get("title") or "Untitled"

        kept: List[tuple] = []  # (role, text, create_time)
        for node_id in _displayed_path(mapping, conv.get("current_node")):
            message = mapping.get(node_id, {}).get("message")
            if not message:
                continue
            role = (message.get("author") or {}).get("role")
            if role not in _KEPT_ROLES:
                continue
            text = _message_text(message)
            if text is None:
                continue
            kept.append((role, text, message.get("create_time")))

        if not kept:
            # Nothing keepable (system-only thread, or every message was a
            # non-text medium) — an empty unit; run_import counts and
            # skips it rather than the importer silently dropping it.
            content = ""
        else:
            content = render_turns([(role, text) for role, text, _ in kept])

        message_times = [
            t for _, _, t in kept if t is not None
        ]
        timestamp = (
            _epoch_to_utc(min(message_times)) if message_times
            else _epoch_to_utc(conv.get("create_time"))
        )

        source_id = f"chatgpt:conversation:{conv_id}"
        yield ImportUnit(
            source_id=source_id,
            ingest_key=source_id,
            content=content,
            content_type="conversation",
            timestamp=timestamp,
            origin_runtime="chatgpt",
            origin_source="import",
            session_key=conv_id,
            links=[],
            metadata={"title": title, "messages": len(kept)},
        )
