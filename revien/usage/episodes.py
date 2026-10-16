"""
Recall episodes from Claude Code transcripts (N1, metacognition tier 1).

A recall episode is one ``revien_recall`` call inside a session: the human
prompt it served, the query, the memories it returned, and the assistant's
answer that followed — the raw material for "did the answer actually draw on
this memory?".

Reads the real transcript shape (see ``ClaudeCodeAdapter``): the recall is a
``tool_use`` block on an assistant entry, its result a ``tool_result`` block
on the next user entry, and the answer is the text blocks after it. Thinking
blocks and tool inputs are never part of the answer.

An episode is emitted only once its turn has CLOSED: the turn's final message
(``stop_reason: "end_turn"``) has been written, or a new human prompt arrived.
A turn still being written is left alone — the file grows when it finishes,
so the next scan picks it up.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

RECALL_TOOL = "revien_recall"


class RecalledMemory(BaseModel):
    """One result of a recall, exactly as the model saw it."""

    node_id: str
    label: str = ""
    content: str = ""


class RecallEpisode(BaseModel):
    """One recall and the answer that followed it."""

    session_source_id: str
    tool_use_id: str
    query: str = ""
    prompt: str = ""  # the human prompt the recall served
    memories: List[RecalledMemory] = Field(default_factory=list)
    response: str = ""  # assistant text after the result, up to the turn's end
    recalled_at: Optional[datetime] = None

    @property
    def episode_key(self) -> str:
        """Ledger identity: ``{session source_id}#{tool_use_id}``."""
        return f"{self.session_source_id}#{self.tool_use_id}"


def is_recall_tool(name: str) -> bool:
    """``revien_recall`` itself, or the MCP-namespaced form Claude Code
    records (``mcp__<server>__revien_recall``)."""
    return name == RECALL_TOOL or name.endswith("__" + RECALL_TOOL)


def _blocks(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """An entry's content blocks: ``message.content`` (real shape) or
    ``content`` (flat shape). A bare string becomes one text block."""
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        content = obj.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def parse_recall_result(content: Any) -> List[RecalledMemory]:
    """The memories in a ``revien_recall`` tool result. The MCP tool returns
    the ``POST /v1/recall`` JSON shape; anything unparseable yields []."""
    try:
        payload = json.loads(_result_text(content))
    except (TypeError, ValueError):
        return []
    if (isinstance(payload, dict) and "results" not in payload
            and isinstance(payload.get("result"), dict)):
        payload = payload["result"]
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    memories = []
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("node_id"), str) and r["node_id"]:
            memories.append(RecalledMemory(
                node_id=r["node_id"],
                label=str(r.get("label") or ""),
                content=str(r.get("content") or ""),
            ))
    return memories


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_recall_episodes(
    lines: Iterable[str], session_source_id: str
) -> List[RecallEpisode]:
    """Every closed recall episode in one session transcript, in order.

    ``lines`` is the session JSONL (an open file works). Harness-injected
    (``isMeta``) and sidechain (subagent) entries are skipped, matching what
    the adapter ingests."""
    episodes: List[RecallEpisode] = []
    pending: Dict[str, Dict[str, str]] = {}  # tool_use_id -> query, prompt
    open_episodes: List[RecallEpisode] = []  # result in, answer accumulating
    answers: Dict[str, List[str]] = {}  # tool_use_id -> answer text blocks
    closing_message: Optional[object] = None  # id of the end_turn message
    last_prompt = ""

    def close_turn() -> None:
        nonlocal closing_message
        for episode in open_episodes:
            episode.response = "\n\n".join(answers.pop(episode.tool_use_id, [])).strip()
            if episode.response:
                episodes.append(episode)
        open_episodes.clear()
        closing_message = None

    for raw in lines:
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind not in ("user", "human", "assistant"):
            continue
        if obj.get("isMeta") or obj.get("isSidechain"):
            continue
        message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        blocks = _blocks(obj)

        # The end_turn message can span several entries (one per block).
        # Anything that is not another piece of it means the turn is over.
        if closing_message is not None and not (
            kind == "assistant" and message.get("id") == closing_message
        ):
            close_turn()

        if kind in ("user", "human"):
            results = [b for b in blocks if b.get("type") == "tool_result"]
            if results:
                for block in results:
                    tool_use_id = str(block.get("tool_use_id", ""))
                    call = pending.pop(tool_use_id, None)
                    if call is None or block.get("is_error"):
                        continue
                    memories = parse_recall_result(block.get("content"))
                    if not memories:
                        continue
                    open_episodes.append(RecallEpisode(
                        session_source_id=session_source_id,
                        tool_use_id=tool_use_id,
                        query=call["query"],
                        prompt=call["prompt"],
                        memories=memories,
                        recalled_at=_parse_ts(obj.get("timestamp")),
                    ))
                    answers[tool_use_id] = []
                continue
            text = "\n".join(
                str(b.get("text", "")) for b in blocks if b.get("type") == "text"
            ).strip()
            if text:
                close_turn()  # a new human prompt ends any turn still open
                last_prompt = text
            continue

        # Assistant entry: new recalls start here; answer text accumulates
        # on every episode whose result has already arrived.
        has_text = False
        for block in blocks:
            block_type = block.get("type")
            if block_type == "tool_use" and is_recall_tool(str(block.get("name", ""))):
                call_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                pending[str(block.get("id", ""))] = {
                    "query": str(call_input.get("query") or ""),
                    "prompt": last_prompt,
                }
            elif block_type == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    has_text = True
                    for episode in open_episodes:
                        answers[episode.tool_use_id].append(text)
        if open_episodes and has_text and message.get("stop_reason") == "end_turn":
            # A missing id gets a sentinel no later entry can match, so the
            # very next entry closes the turn.
            closing_message = message.get("id") or object()

    if closing_message is not None:
        close_turn()
    return episodes
