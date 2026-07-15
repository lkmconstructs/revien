"""
Revien Codex Adapter — Reads Codex CLI session history from rollout JSONL logs.
Near-clone of the Claude Code adapter against ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl.
CLI sessions only; the unified desktop app's session storage is undocumented.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .base import RevienAdapter


_PATH_SEP_RE = re.compile(r"[\\/]+")


def _basename_cross_platform(raw: str) -> str:
    """Last path component of a cwd that may be Windows- OR POSIX-style, on
    ANY host OS. Codex records ``cwd`` in the session's native format; the
    adapter (and CI) may run on a different OS, so ``Path(cwd).name`` is wrong
    — on Linux it can't split a ``C:\\...`` path and returns the whole string.
    Split on both separators instead."""
    cleaned = raw.strip().rstrip("\\/")
    parts = [p for p in _PATH_SEP_RE.split(cleaned) if p]
    return parts[-1] if parts else ""


def default_codex_home() -> Path:
    """Codex home directory. CODEX_HOME env overrides ~/.codex (Codex's own rule)."""
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env)
    return Path.home() / ".codex"


# Codex injects session plumbing as user-role messages. Verified against real
# rollout files (2026-07): these wrappers are context, not conversation — skip.
_USER_NOISE_PREFIXES = (
    "<environment_context>",
    "<user_instructions>",
    "<turn_aborted>",
    "<recommended_plugins>",
    "<no retained transcript",
)


class CodexAdapter(RevienAdapter):
    """
    Reads Codex CLI rollout JSONL logs and produces content for ingestion.

    Verified rollout line shape (real session files, Codex CLI 2026-05/07;
    layout also documented by codex-trace):
    {
        "timestamp": "ISO-8601",
        "type": "session_meta" | "turn_context" | "response_item" | "event_msg" | ...,
        "payload": {...}
    }
    Conversation lives in response_item lines whose payload is
    {"type": "message", "role": "user"|"assistant"|"developer",
     "content": [{"type": "input_text"|"output_text", "text": "..."}]}.
    Older Codex versions wrote the response item bare (no envelope):
    {"type": "message", "role": ..., "content": [...]} — handled too.
    Reasoning, function calls, and event_msg lines are noise and skipped.
    """

    def __init__(self, session_dir: Optional[str] = None):
        """
        Args:
            session_dir: Path to Codex rollout session logs.
                         Auto-detected if not provided (CODEX_HOME env,
                         then ~/.codex/sessions).
        """
        self.session_dir = Path(session_dir) if session_dir else self._auto_detect()

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Fetch conversations from Codex sessions modified since `since`."""
        if self.session_dir is None or not self.session_dir.exists():
            return []

        results = []
        since_ts = since.timestamp()

        for jsonl_file in self.session_dir.rglob("rollout-*.jsonl"):
            if not jsonl_file.is_file():
                continue

            mtime = jsonl_file.stat().st_mtime
            if mtime <= since_ts:
                continue

            conversation, project_name = self._parse_rollout(jsonl_file)
            if conversation and conversation.strip():
                # Per-session source_id, matching the claude_code adapter's
                # granularity (adapter:project:session-stem) — sessions in one
                # project must not share provenance.
                project = project_name or "unknown"
                source_id = f"codex:{project}:{jsonl_file.stem}"

                results.append({
                    "content": conversation,
                    "content_type": "conversation",
                    "timestamp": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    "metadata": {
                        "adapter": "codex",
                        "project": project_name or "",
                        "session_file": jsonl_file.name,
                        "path": str(jsonl_file),
                    },
                    "source_id": source_id,
                    # Stable re-ingest identity (R3): the whole rollout file is
                    # re-fetched on every mtime bump (correct change detector);
                    # the key makes that re-ingest refresh the ONE existing
                    # context node instead of stacking a duplicate per sync.
                    "ingest_key": source_id,
                })

        return results

    async def health_check(self) -> bool:
        """Check if the Codex session directory exists."""
        return self.session_dir is not None and self.session_dir.exists()

    def _parse_rollout(self, filepath: Path) -> tuple:
        """
        Parse a Codex rollout JSONL log into (conversation text, project name).
        Extracts user and assistant messages, skips tool/reasoning/event noise.
        """
        messages = []
        project_name = None

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

                    line_type = obj.get("type", "")

                    # session_meta carries the working directory — the project.
                    if line_type == "session_meta":
                        payload = obj.get("payload")
                        if isinstance(payload, dict) and payload.get("cwd"):
                            project_name = _basename_cross_platform(
                                str(payload["cwd"])
                            ) or None
                        continue

                    # Envelope shape: {"type": "response_item", "payload": {...}}
                    # Bare shape (older Codex): the response item IS the line.
                    if line_type == "response_item":
                        item = obj.get("payload")
                    elif line_type == "message":
                        item = obj
                    else:
                        continue

                    if not isinstance(item, dict) or item.get("type") != "message":
                        continue

                    role = item.get("role", "")
                    content = self._extract_content(item)
                    if not content:
                        continue

                    if role == "user":
                        if content.lstrip().startswith(_USER_NOISE_PREFIXES):
                            continue
                        messages.append(f"User: {content}")
                    elif role == "assistant":
                        messages.append(f"Assistant: {content}")
                    # developer/system roles are instructions, not conversation.

        except Exception:
            return None, project_name

        return ("\n".join(messages) if messages else None), project_name

    def _extract_content(self, item: Dict) -> Optional[str]:
        """Extract text from a message item's content blocks."""
        content = item.get("content", "")

        if isinstance(content, str):
            return content.strip() if content.strip() else None

        if isinstance(content, list):
            # Blocks: [{"type": "input_text"|"output_text"|"text", "text": "..."}]
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("input_text", "output_text", "text"):
                        text = block.get("text", "")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                elif isinstance(block, str):
                    if block.strip():
                        parts.append(block.strip())
            return "\n".join(parts) if parts else None

        return None

    def _auto_detect(self) -> Optional[Path]:
        """Auto-detect the Codex session log directory."""
        sessions = default_codex_home() / "sessions"
        if sessions.exists() and sessions.is_dir():
            return sessions
        return None
