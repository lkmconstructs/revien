"""
Revien Claude Code Adapter — Reads Claude Code session history from JSONL logs.
The primary adoption hook. Claude Code has zero persistent memory. Revien gives it memory.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .base import RevienAdapter


# Common locations for Claude Code session logs
CLAUDE_CODE_PATHS = [
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude",
]


class ClaudeCodeAdapter(RevienAdapter):
    """
    Reads Claude Code JSONL conversation logs and produces content for ingestion.

    Real Claude Code transcripts nest the API message under ``message``:
    {
        "type": "user" | "assistant" | "attachment" | "system" | ...,
        "message": {"role": "user" | "assistant",
                    "content": "message text" | [{"type": "text" | "tool_use"
                                                  | "tool_result" | "thinking", ...}]},
        "isMeta": true,          # harness-injected user entry, not the human
        "isSidechain": true,     # subagent thread; its "user" is an agent
        "timestamp": "ISO-8601",
        ...
    }
    The flat shape ({"type": "human", "content": ...}) is still read, for
    older or simplified logs.
    """

    IDEMPOTENT_REREAD = True  # every item carries ingest_key

    def __init__(self, session_dir: Optional[str] = None):
        """
        Args:
            session_dir: Path to Claude Code session logs.
                         Auto-detected if not provided.
        """
        self.session_dir = Path(session_dir) if session_dir else self._auto_detect()

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Fetch conversations from Claude Code sessions modified since `since`."""
        results = []
        for jsonl_file in self.session_files(since):
            mtime = jsonl_file.stat().st_mtime
            conversation = self._parse_session_log(jsonl_file)
            if conversation and conversation.strip():
                # Derive project name from directory structure
                project_name = self._extract_project_name(jsonl_file)
                source_id = self.source_id_for(jsonl_file)

                results.append({
                    "content": conversation,
                    "content_type": "conversation",
                    "timestamp": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    "metadata": {
                        "adapter": "claude_code",
                        "project": project_name,
                        "session_file": jsonl_file.name,
                        "path": str(jsonl_file),
                    },
                    "source_id": source_id,
                    # Stable re-ingest identity (R3): the whole session file is
                    # re-fetched on every mtime bump (correct change detector);
                    # the key makes that re-ingest refresh the ONE existing
                    # context node instead of stacking a duplicate per sync.
                    "ingest_key": source_id,
                    # Origin Layer (WS0): known at read time, no need to
                    # round-trip through derive_origin(source_id) later.
                    "origin_runtime": "claude-code",
                    "origin_source": "live",
                    "project_key": project_name,
                    "session_key": jsonl_file.stem,
                })

        return results

    def session_files(self, since: datetime) -> List[Path]:
        """Session JSONL files under the session directory modified after
        ``since``. Shared by ingest and the usage observer, so both read the
        same files."""
        if self.session_dir is None or not self.session_dir.exists():
            return []
        since_ts = since.timestamp()
        files = []
        for jsonl_file in self.session_dir.rglob("*.jsonl"):
            if jsonl_file.is_file() and jsonl_file.stat().st_mtime > since_ts:
                files.append(jsonl_file)
        return files

    def source_id_for(self, jsonl_file: Path) -> str:
        """The source_id a session file ingests under:
        ``claude-code:{project}:{session}``."""
        return f"claude-code:{self._extract_project_name(jsonl_file)}:{jsonl_file.stem}"

    async def health_check(self) -> bool:
        """Check if Claude Code session directory exists."""
        return self.session_dir is not None and self.session_dir.exists()

    def _parse_session_log(self, filepath: Path) -> Optional[str]:
        """
        Parse a Claude Code JSONL session log into conversation text.
        Extracts human and assistant messages, skips tool use noise.
        """
        messages = []

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(obj, dict):
                        continue
                    # Not the human's words: harness-injected user entries
                    # (skill bodies, command caveats, agent reports) and
                    # sidechain entries, whose "user" is the parent agent
                    # (subagent transcripts under <session>/subagents/ are
                    # all sidechain, so they ingest as nothing).
                    if obj.get("isMeta") or obj.get("isSidechain"):
                        continue

                    msg_type = obj.get("type", "")
                    content = self._extract_content(obj)

                    if not content:
                        continue

                    # Only include conversational messages
                    if msg_type in ("human", "user"):
                        messages.append(f"User: {content}")
                    elif msg_type in ("assistant", "ai"):
                        messages.append(f"Assistant: {content}")
                    elif msg_type == "text" and content:
                        # Some formats use "text" type with role field
                        role = obj.get("role", "")
                        if role in ("human", "user"):
                            messages.append(f"User: {content}")
                        elif role in ("assistant", "ai"):
                            messages.append(f"Assistant: {content}")

        except Exception:
            return None

        return "\n".join(messages) if messages else None

    def _extract_content(self, obj: Dict) -> Optional[str]:
        """Extract text content from a message object: top-level ``content``
        (flat shape) or ``message.content`` (real transcripts). Only text
        blocks count; tool_use, tool_result and thinking blocks are skipped."""
        content = obj.get("content", "")
        message = obj.get("message")
        if not content and isinstance(message, dict):
            content = message.get("content", "")

        if isinstance(content, str):
            return content.strip() if content.strip() else None

        if isinstance(content, list):
            # Content blocks: [{"type": "text", "text": "..."}, ...]
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            parts.append(text.strip())
                elif isinstance(block, str):
                    if block.strip():
                        parts.append(block.strip())
            return "\n".join(parts) if parts else None

        return None

    def _extract_project_name(self, filepath: Path) -> str:
        """Extract project name from the file path."""
        parts = filepath.parts
        # Look for "projects" directory and use the next segment
        for i, part in enumerate(parts):
            if part == "projects" and i + 1 < len(parts):
                return parts[i + 1]
        # Fallback: use parent directory name
        return filepath.parent.name

    def _auto_detect(self) -> Optional[Path]:
        """Auto-detect Claude Code session log directory."""
        for path in CLAUDE_CODE_PATHS:
            if path.exists() and path.is_dir():
                return path
        return None
