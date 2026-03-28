from .base import RevienAdapter
from .file_watcher import FileWatcherAdapter
from .claude_code import ClaudeCodeAdapter
from .generic_api import GenericAPIAdapter
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter

# LangChain adapter: conditional import (peer dependency)
try:
    from .langchain_adapter import RevienMemory
    _LANGCHAIN_EXPORTS = ["RevienMemory"]
except ImportError:
    _LANGCHAIN_EXPORTS = []

__all__ = [
    "RevienAdapter",
    "FileWatcherAdapter",
    "ClaudeCodeAdapter",
    "GenericAPIAdapter",
    "OpenAIAdapter",
    "OllamaAdapter",
    *_LANGCHAIN_EXPORTS,
]
