from .base import RevienAdapter
from .file_watcher import FileWatcherAdapter
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .generic_api import GenericAPIAdapter
from .obsidian import ObsidianVaultAdapter
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter

# LangChain adapter: conditional import (peer dependency)
try:
    from .langchain_adapter import RevienMemory
    _LANGCHAIN_EXPORTS = ["RevienMemory"]
except ImportError:
    _LANGCHAIN_EXPORTS = []

def build_adapter_from_config(entry: dict):
    """Construct an adapter from a ``~/.revien/config.json`` adapters entry.

    This is the missing half of ``revien connect``: connect persists the
    entry, the daemon calls this at startup to make it LIVE on the sync
    scheduler. Returns None for unknown/malformed entries (the daemon logs
    and skips — one bad entry must not stop the rest).
    """
    entry = entry or {}
    adapter_type = entry.get("type", "")
    try:
        if adapter_type == "claude_code":
            return ClaudeCodeAdapter(session_dir=entry.get("session_dir"))
        if adapter_type == "codex":
            return CodexAdapter(session_dir=entry.get("session_dir"))
        if adapter_type == "file_watcher":
            return FileWatcherAdapter(watch_dir=entry["watch_dir"])
        if adapter_type == "generic_api":
            return GenericAPIAdapter(url=entry["url"])
        if adapter_type == "obsidian":
            return ObsidianVaultAdapter(vault_dir=entry["vault_dir"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


__all__ = [
    "RevienAdapter",
    "FileWatcherAdapter",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "GenericAPIAdapter",
    "ObsidianVaultAdapter",
    "OpenAIAdapter",
    "OllamaAdapter",
    "build_adapter_from_config",
    *_LANGCHAIN_EXPORTS,
]
