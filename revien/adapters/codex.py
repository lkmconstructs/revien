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

from .base import RevienAdapter, parse_message_timestamp


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

            conversation, project_name, first_ts = self._parse_rollout_ts(jsonl_file)
            if conversation and conversation.strip():
                # recorded_at = when the first message was SAID, not the
                # rollout file's mtime (moves on every append). No message
                # timestamp -> mtime, labelled as such.
                if first_ts is not None:
                    ts_iso, ts_source = first_ts.isoformat(), "content"
                else:
                    ts_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                    ts_source = "mtime"
                # Per-session source_id, matching the claude_code adapter's
                # granularity (adapter:project:session-stem) — sessions in one
                # project must not share provenance.
                project = project_name or "unknown"
                source_id = f"codex:{project}:{jsonl_file.stem}"

                results.append({
                    "content": conversation,
                    "content_type": "conversation",
                    "timestamp": ts_iso,
                    "timestamp_source": ts_source,
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
                    # Origin Layer (WS0): known at read time. project_key
                    # matches the same value baked into source_id above.
                    "origin_runtime": "codex",
                    "origin_source": "live",
                    "project_key": project,
                    "session_key": jsonl_file.stem,
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
        text, project, _ts = self._parse_rollout_ts(filepath)
        return text, project

    def _parse_rollout_ts(self, filepath: Path) -> tuple:
        """(conversation text | None, project name, earliest message
        timestamp | None)."""
        messages = []
        stamps = []
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

                    before = len(messages)
                    if role == "user":
                        if content.lstrip().startswith(_USER_NOISE_PREFIXES):
                            continue
                        messages.append(f"User: {content}")
                    elif role == "assistant":
                        messages.append(f"Assistant: {content}")
                    # developer/system roles are instructions, not conversation.
                    if len(messages) > before:
                        # Envelope lines carry the time on the line; bare
                        # (older) items may carry it on the item itself.
                        ts = parse_message_timestamp(
                            obj.get("timestamp", item.get("timestamp")))
                        if ts is not None:
                            stamps.append(ts)

        except Exception:
            return None, project_name, None

        return (("\n".join(messages) if messages else None), project_name,
                (min(stamps) if stamps else None))

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
