"""
Revien Claude.ai Importer — turns a Claude.ai conversations.json export
into one ImportUnit per conversation, ready for
revien.importers.base.run_import.

Export shape (Claude.ai's own, undocumented but stable): an array of
conversation objects, each {uuid, name, created_at, updated_at,
chat_messages: [{uuid, sender, text, content, created_at, attachments}]}.
Unlike ChatGPT's export there's no branching tree here — chat_messages is
already the one displayed thread, in order — so this importer is a
straight walk, no current_node/parent-chain resolution needed.
"""

import json
from typing import Dict, Iterator, List, Optional

from revien.importers.base import ImportUnit, parse_iso_timestamp, render_turns

# Claude.ai's sender values -> the role render_turns() expects.
_SENDER_ROLE = {"human": "user", "assistant": "assistant"}


def _message_text(message: Dict) -> Optional[str]:
    """`text` if present and non-empty, else the text content blocks
    joined — a message can carry text only inside `content` (e.g. when the
    export's top-level `text` field was left blank but the block list
    wasn't)."""
    text = message.get("text")
    if text and text.strip():
        return text.strip()
    blocks = message.get("content") or []
    strings = [
        b.get("text", "").strip()
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
    ]
    if not strings:
        return None
    return "\n".join(strings)


def iter_units(path: str) -> Iterator[ImportUnit]:
    """One ImportUnit per conversation in a Claude.ai export (.zip or a
    bare conversations.json). F3: same non-array-top-level ValueError and
    per-conversation error-marker pattern as chatgpt.iter_units — see
    there for why."""
    from revien.importers.base import open_export

    raw = open_export(path)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected a JSON array of conversations at the top level "
            f"(Claude.ai's own conversations.json shape), got {type(data).__name__}."
        )

    for index, conv in enumerate(data):
        try:
            conv_id = conv.get("uuid") or ""
            title = conv.get("name") or "Untitled"

            kept: List[tuple] = []  # (role, text, created_at)
            for message in conv.get("chat_messages") or []:
                role = _SENDER_ROLE.get(message.get("sender"))
                if role is None:
                    continue
                text = _message_text(message)
                if text is None:
                    continue
                kept.append((role, text, message.get("created_at")))

            content = (
                render_turns([(role, text) for role, text, _ in kept]) if kept else ""
            )

            message_times = [
                dt for _, _, t in kept
                if t is not None and (dt := parse_iso_timestamp(t)) is not None
            ]
            timestamp = (
                min(message_times) if message_times
                else parse_iso_timestamp(conv.get("created_at"))
            )

            source_id = f"claude:conversation:{conv_id}"
            yield ImportUnit(
                source_id=source_id,
                ingest_key=source_id,
                content=content,
                content_type="conversation",
                timestamp=timestamp,
                origin_runtime="claude",
                session_key=conv_id,
                metadata={"title": title, "messages": len(kept)},
            )
        except Exception as exc:  # noqa: BLE001 - one bad conversation must not kill the batch
            marker = f"claude:conversation:error:{index}"
            yield ImportUnit(
                source_id=marker,
                ingest_key=marker,
                content="",
                content_type="conversation",
                timestamp=None,
                origin_runtime="claude",
                error=f"{type(exc).__name__}: {exc}",
            )
