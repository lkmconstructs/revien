"""
Revien File Watcher Adapter — Watches a local directory for new/changed files.
The simplest adapter. Covers manual export workflows.
Supports: .md, .txt, .json, .jsonl files.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .base import RevienAdapter


class FileWatcherAdapter(RevienAdapter):
    """
    Watches a local directory for new or modified files.
    Ingests file content when changes are detected.
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ".jsonl", ".log"}

    def __init__(
        self,
        watch_dir: str,
        recursive: bool = True,
        extensions: Optional[set] = None,
    ):
        self.watch_dir = Path(watch_dir)
        self.recursive = recursive
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Scan directory for files modified since `since`."""
        if not self.watch_dir.exists():
            return []

        results = []
        since_ts = since.timestamp()

        pattern = "**/*" if self.recursive else "*"
        for filepath in self.watch_dir.glob(pattern):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in self.extensions:
                continue

            # Check modification time
            mtime = filepath.stat().st_mtime
            if mtime <= since_ts:
                continue

            content = self._read_file(filepath)
            if content and content.strip():
                results.append({
                    "content": content,
                    "content_type": self._detect_content_type(filepath),
                    "timestamp": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                    "metadata": {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "adapter": "file_watcher",
                    },
                    "source_id": f"file:{filepath.name}",
                })

        return results

    async def health_check(self) -> bool:
        """Check if the watch directory exists and is readable."""
        return self.watch_dir.exists() and self.watch_dir.is_dir()

    def _read_file(self, filepath: Path) -> Optional[str]:
        """Read file content, handling different formats."""
        try:
            if filepath.suffix == ".jsonl":
                return self._read_jsonl(filepath)
            elif filepath.suffix == ".json":
                return self._read_json(filepath)
            else:
                return filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _read_jsonl(self, filepath: Path) -> str:
        """Read JSONL file, extracting content from each line."""
        lines = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # Try common content fields
                        content = (
                            obj.get("content")
                            or obj.get("text")
                            or obj.get("message")
                            or ""
                        )
                        role = obj.get("role", "")
                        if content:
                            prefix = f"{role}: " if role else ""
                            lines.append(f"{prefix}{content}")
                except json.JSONDecodeError:
                    continue
        return "\n".join(lines)

    def _read_json(self, filepath: Path) -> str:
        """Read JSON file, extracting meaningful text."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, str):
                return data
            elif isinstance(data, list):
                # List of messages or strings
                parts = []
                for item in data:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        content = item.get("content") or item.get("text") or ""
                        role = item.get("role", "")
                        if content:
                            prefix = f"{role}: " if role else ""
                            parts.append(f"{prefix}{content}")
                return "\n".join(parts)
            elif isinstance(data, dict):
                return data.get("content") or data.get("text") or json.dumps(data)
            return str(data)
        except (json.JSONDecodeError, Exception):
            return filepath.read_text(encoding="utf-8", errors="replace")

    def _detect_content_type(self, filepath: Path) -> str:
        """Detect content type from file extension."""
        ext = filepath.suffix.lower()
        if ext in (".md", ".txt"):
            return "document"
        elif ext in (".json", ".jsonl", ".log"):
            return "conversation"
        return "note"
