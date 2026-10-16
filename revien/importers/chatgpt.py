"""
Revien ChatGPT Importer — turns a ChatGPT conversations.json export into
one ImportUnit per conversation, ready for revien.importers.base.run_import.

Export shape (OpenAI's own, undocumented but stable): an array of
conversation objects, each {title, create_time, update_time, mapping,
current_node}. `mapping` is NOT the displayed transcript — it's every
branch the user ever generated (edits, regenerations) as a tree keyed by
node id. The single thread ChatGPT shows you is the parent chain from
`current_node` back to the root; anything off that path is an abandoned
branch and must not be ingested — see _displayed_path below.

revien/adapters/openai_adapter.py walks the same mapping but interleaves
the walk with store writes and ingests the WHOLE tree, not one displayed
thread — not reusable here. Written fresh; the adapter is untouched.
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


def _to_float(value) -> Optional[float]:
    """Coerce a create_time value to float; None if it can't be (F3: an
    export with mixed-type create_time — a stray string, a null — must not
    crash min() over the whole batch; the un-coercible value is simply
    skipped rather than guessed)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _walk_to_root(mapping: Dict, leaf: str) -> List[str]:
    """Root-to-leaf node id path by following `parent` links from `leaf`
    back to the root (cycle-guarded), then reversing. Shared by both
    _displayed_path branches below (S3) — the only difference between them
    is which leaf(s) they start from."""
    path = []
    node_id = leaf
    seen = set()
    while node_id is not None and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(node_id)
        node_id = mapping[node_id].get("parent")
    path.reverse()
    return path


def _displayed_path(mapping: Dict, current_node: Optional[str]) -> List[str]:
    """Node ids from root to the displayed leaf, in that order.

    Normal case: current_node is set and present in mapping — walk it back
    to the root. Fallback (current_node missing or stale): no way to know
    which branch was "current", so take the longest root-to-leaf path over
    every leaf — the fullest conversation on record, the closest honest
    guess without a current_node to trust. Iterated in mapping's own
    (insertion) order, so ties resolve deterministically to the first-seen
    longest path.
    """
    if current_node and current_node in mapping:
        return _walk_to_root(mapping, current_node)

    leaves = [
        node_id for node_id, node in mapping.items()
        if not node.get("children")
    ]
    best: List[str] = []
    for leaf in leaves:
        path = _walk_to_root(mapping, leaf)
        if len(path) > len(best):
            best = path
    return best


def iter_units(path: str) -> Iterator[ImportUnit]:
    """One ImportUnit per conversation in a ChatGPT export (.zip or a
    bare conversations.json).

    F3: the export's top-level JSON must be an array of conversations (the
    documented shape) — anything else is one clear ValueError raised before
    any unit is produced, rather than silently treating a single stray
    object as a one-conversation export. A per-conversation parse failure
    (mid-batch) does NOT raise — this generator is dead after a raise, so
    one malformed conversation would silently kill every conversation after
    it. Instead it's caught and yielded as an error-marker ImportUnit that
    run_import logs and skips, so a 3-conversation export with one bad
    conversation in the middle still ingests the other two.
    """
    from revien.importers.base import open_export

    raw = open_export(path)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a JSON array of conversations at the top level "
            f"(ChatGPT's own conversations.json shape), got {type(data).__name__}."
        )

    for index, conv in enumerate(data):
        try:
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
                # Nothing keepable (system-only thread, or every message was
                # a non-text medium) — an empty unit; run_import counts and
                # skips it rather than the importer silently dropping it.
                content = ""
            else:
                content = render_turns([(role, text) for role, text, _ in kept])

            # Mixed-type create_time (a stray string, a null among floats)
            # must not crash min() — coerce, skip what can't coerce.
            message_times = [
                ft for _, _, t in kept if (ft := _to_float(t)) is not None
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
                session_key=conv_id,
                metadata={"title": title, "messages": len(kept)},
            )
        except Exception as exc:  # noqa: BLE001 - one bad conversation must not kill the batch
            marker = f"chatgpt:conversation:error:{index}"
            yield ImportUnit(
                source_id=marker,
                ingest_key=marker,
                content="",
                content_type="conversation",
                timestamp=None,
                origin_runtime="chatgpt",
                error=f"{type(exc).__name__}: {exc}",
            )
