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
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from revien.importers.base import ImportUnit, render_turns

# Claude.ai's sender values -> the role render_turns() expects.
_SENDER_ROLE = {"human": "user", "assistant": "assistant"}


def _parse_iso(value) -> Optional[datetime]:
    """Claude.ai timestamps are ISO 8601 UTC, typically with a trailing
    'Z' — Python's fromisoformat only accepts that suffix from 3.11, and
    this project targets >=3.10, so swap it for the offset form first."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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
    bare conversations.json)."""
    from revien.importers.base import open_export

    raw = open_export(path)
    data = json.loads(raw)
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
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
            _parse_iso(t) for _, _, t in kept if t is not None
        ]
        message_times = [t for t in message_times if t is not None]
        timestamp = (
            min(message_times) if message_times
            else _parse_iso(conv.get("created_at"))
        )

        source_id = f"claude:conversation:{conv_id}"
        yield ImportUnit(
            source_id=source_id,
            ingest_key=source_id,
            content=content,
            content_type="conversation",
            timestamp=timestamp,
            origin_runtime="claude",
            origin_source="import",
            session_key=conv_id,
            links=[],
            metadata={"title": title, "messages": len(kept)},
        )
