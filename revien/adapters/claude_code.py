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

    Claude Code stores sessions as JSONL files where each line is a message object:
    {
        "type": "human" | "assistant" | "tool_use" | "tool_result" | ...,
        "content": "message text" | [...],
        "timestamp": "ISO-8601",
        ...
    }
    """

    def __init__(self, session_dir: Optional[str] = None):
        """
        Args:
            session_dir: Path to Claude Code session logs.
                         Auto-detected if not provided.
        """
        self.session_dir = Path(session_dir) if session_dir else self._auto_detect()

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Fetch conversations from Claude Code sessions modified since `since`."""
        if self.session_dir is None or not self.session_dir.exists():
            return []

        results = []
        since_ts = since.timestamp()

        # Find all JSONL files in the session directory tree
        for jsonl_file in self.session_dir.rglob("*.jsonl"):
            if not jsonl_file.is_file():
                continue

            mtime = jsonl_file.stat().st_mtime
            if mtime <= since_ts:
                continue

            conversation = self._parse_session_log(jsonl_file)
            if conversation and conversation.strip():
                # Derive project name from directory structure
                project_name = self._extract_project_name(jsonl_file)

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
                    "source_id": f"claude-code:{project_name}:{jsonl_file.stem}",
                })

        return results

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
        """Extract text content from a message object."""
        content = obj.get("content", "")

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
