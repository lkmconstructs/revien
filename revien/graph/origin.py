"""
Revien Origin Layer — derives origin_runtime/origin_source/project_key/
session_key from a node's source_id.

source_id conventions are adapter-owned free text, not a designed schema —
each adapter picked its own shape independently (see the per-adapter receipts
below). This module is the ONE place that knows how to read them back, so the
003 migration backfill, store._ensure_db's in-place upgrade, and the pipeline's
fallback-when-the-caller-omits-origin path all agree on the same answer for
the same source_id.

Conventions (verified against the adapters that produce them):
    claude_code  claude-code:{project}:{session_stem}   (adapters/claude_code.py)
    codex        codex:{project}:{session_stem}         (adapters/codex.py)
    hermes       hermes                                  (hermes_provider.py)
    openai       openai:conversation:{conv_id}           (adapters/openai_adapter.py)
    obsidian     vault:{relpath}#{slug}                   (adapters/obsidian.py)
    file_watcher file:{name}                             (adapters/file_watcher.py)
    generic_api  api:{url}  (prefix configurable)         (adapters/generic_api.py)
    ollama       ollama_history | ollama_chat            (adapters/ollama_adapter.py)
    langchain    langchain  (session-scoped source_ids are caller-chosen free
                 text with no recognizable prefix and are deliberately left
                 unrecognized — see the module docstring above)

Anything else (empty source_id, or a prefix not in the table) derives to all
None — an honest "unknown", never a guess.
"""

from typing import NamedTuple, Optional


class Origin(NamedTuple):
    runtime: Optional[str]
    source: Optional[str]
    project: Optional[str]
    session: Optional[str]


_UNKNOWN = Origin(None, None, None, None)


# Fixed vocabulary (WS0 Leg — origin validation). A caller declaring
# origin_runtime/origin_source is claiming provenance; only these values are
# recognized. None (unset) is always allowed — that's "unknown", not
# "invalid". Kept here, next to derive_origin, so the declared-value gate and
# the source_id-inference table can never quietly drift apart.
RUNTIMES = frozenset({
    "claude-code", "codex", "hermes", "ollama", "openai", "langchain",
    "obsidian", "file", "api", "chatgpt", "claude", "readwise",
})
SOURCES = frozenset({"live", "import", "vault", "watch", "api"})


def validate_origin(runtime: Optional[str], source: Optional[str]) -> None:
    """Raise ValueError naming the offending value if either is set and not
    in the fixed vocabulary above. None is always fine for either field —
    only a value that claims to BE something gets checked."""
    if runtime is not None and runtime not in RUNTIMES:
        raise ValueError(
            f"Unknown origin_runtime: {runtime!r}. Valid: {sorted(RUNTIMES)}"
        )
    if source is not None and source not in SOURCES:
        raise ValueError(
            f"Unknown origin_source: {source!r}. Valid: {sorted(SOURCES)}"
        )


def _split_project_session(source_id: str, runtime: str) -> Origin:
    """Shared claude_code/codex shape: {runtime}:{project}:{session_stem}."""
    rest = source_id[len(runtime) + 1:]
    parts = rest.split(":", 1)
    project = parts[0] if len(parts) > 0 and parts[0] else None
    session = parts[1] if len(parts) > 1 and parts[1] else None
    return Origin(runtime, "live", project, session)


def derive_origin(source_id: str) -> Origin:
    """Pure function: source_id -> Origin(runtime, source, project, session).

    Never raises; unrecognized or empty input derives to Origin(None, None,
    None, None). Deliberately conservative — a prefix that merely LOOKS
    similar to a known convention is not enough; only exact, verified shapes
    match.
    """
    if not source_id:
        return _UNKNOWN

    if source_id.startswith("claude-code:"):
        return _split_project_session(source_id, "claude-code")
    if source_id.startswith("codex:"):
        return _split_project_session(source_id, "codex")
    if source_id == "hermes":
        return Origin("hermes", "live", None, None)
    if source_id.startswith("openai:conversation:"):
        conv_id = source_id[len("openai:conversation:"):] or None
        return Origin("openai", "import", None, conv_id)
    if source_id.startswith("vault:"):
        return Origin("obsidian", "vault", None, None)
    if source_id.startswith("file:"):
        return Origin("file", "watch", None, None)
    if source_id.startswith("api:"):
        return Origin("api", "api", None, None)
    if source_id in ("ollama_history", "ollama_chat"):
        return Origin("ollama", "live", None, None)
    if source_id == "langchain":
        return Origin("langchain", "live", None, None)

    return _UNKNOWN
