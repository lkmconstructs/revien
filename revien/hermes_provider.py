# Copyright 2025 LKM Constructs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Revien memory provider for Hermes Agent (NousResearch).

Hermes ships a pluggable external-memory system: a ``MemoryProvider`` ABC that
the agent drives automatically — recall relevant context before each turn,
persist each turn after the response, extract on session end, and inject a
provider block into the system prompt. Only ONE external provider is active at a
time, so winning the slot makes Revien *the* memory backend for that user.

This provider backs those hooks with an IN-PROCESS Revien stack over a single
SQLite database — the same wiring ``revien.mcp_server.build_mcp_server(db_path=)``
uses for its own stack (GraphStore + SemanticIndex + IngestionPipeline +
RetrievalEngine). No daemon, no REST hop: a zero-egress, single-file memory
backend, which is the differentiated pitch against the hosted incumbents
(Honcho / Mem0 / Supermemory).

Hook mapping (Hermes verb -> Revien op):
  - prefetch      -> engine.recall            (load prior context before a turn)
  - sync_turn     -> enqueue to a SINGLE serial worker that runs
                     pipeline.ingest(defer_embed=True) (persist now, NEVER
                     block the turn; one worker => writes never interleave on
                     the shared connection)
  - on_session_end-> flush the queue, then drain deferred embeds (else no-op)
  - get_tool_schemas / handle_tool_call
                  -> explicit revien_recall / revien_store tools
  - system_prompt_block -> one line: what Revien is, that recall is automatic

--------------------------------------------------------------------------------
PINNED INTERFACE — verified against hermes-agent v0.18.2 (release 2026.7.7.2,
2026-07-08, MIT), developer guide
``website/docs/developer-guide/memory-provider-plugin.md``:

  from agent.memory_provider import MemoryProvider

  @property
  def name(self) -> str
  def is_available(self) -> bool                          # NO network calls
  def initialize(self, session_id: str, **kwargs) -> None # kwargs: hermes_home
  def get_config_schema(self)
  def save_config(self, values: dict, hermes_home: str) -> None
  def get_tool_schemas(self)
  def handle_tool_call(self, tool_name, args, **kwargs) -> str
  # optional hooks:
  def system_prompt_block(self)
  def prefetch(self, query, *, session_id="")
  def queue_prefetch(self, query)
  def sync_turn(self, user_content, assistant_content, *, session_id="",
                messages=None)                            # MUST be non-blocking
  def on_session_end(self, messages)
  def on_pre_compress(self, messages)
  def on_memory_write(self, action, target, content)
  def shutdown(self)

  # entry point Hermes' plugin loader calls:
  def register(ctx) -> None:  ctx.register_memory_provider(<instance>)

RISK (top of the scoping note): this is a PRE-1.0 ABC on a hot release cadence.
The adapter is deliberately THIN — every Hermes-facing method delegates straight
to the existing engine surface and does no interpretation of its own — so an ABC
bump is a small, localized edit here, not a rework. The return SHAPES the docs
leave provider-defined (get_tool_schemas' key name, prefetch's container) are
called out inline where they're chosen; those are the churn-prone lines.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Peer-dependency guard (mirrors the langchain adapter and mcp_server): the
# core install never depends on Hermes. Everything that touches the SDK checks
# HERMES_AVAILABLE first, and constructing the provider without the SDK raises a
# clear ImportError instead of an AttributeError deep in a hook.
try:
    from agent.memory_provider import MemoryProvider

    HERMES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the SDK
    MemoryProvider = object  # type: ignore[assignment,misc]
    HERMES_AVAILABLE = False


# Revien-side imports are core (no guard) — this module only imports cleanly-
# importable engine pieces, so the mapping logic below is testable WITHOUT the
# Hermes SDK by constructing the provider's in-process stack directly.
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex
from revien.validation import (
    TOP_N_MAX,
    TOP_N_MIN,
    ValidationError,
    validate_ingest,
    validate_recall,
)


# Turn-conversation provenance, per the scoping note's sync_turn-volume risk:
# every Hermes turn lands as a single conversational unit from one source, so
# curated capture stays distinguishable from the per-turn firehose.
_HERMES_SOURCE_ID = "hermes"
_CONVERSATION = "conversation"

# Small recall window for the automatic pre-turn injection — prefetch runs on
# EVERY turn, so it stays cheap; the explicit revien_recall tool is where a
# model asks for more.
_PREFETCH_TOP_N = 5


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """Resolve the Revien database path, mirroring revien.cli._default_db_path.

    Precedence: explicit arg > ``REVIEN_DB`` env > ``~/.revien/revien.db``. This
    keeps a Hermes-driven Revien pointed at the SAME single file the daemon and
    the ``revien`` CLI use, so memory is shared across every entry point.
    """
    if db_path:
        return db_path
    env = os.environ.get("REVIEN_DB")
    if env:
        return env
    revien_dir = Path.home() / ".revien"
    revien_dir.mkdir(parents=True, exist_ok=True)
    return str(revien_dir / "revien.db")


class _MissingHermesStub:
    """Stub base for when hermes-agent is not installed.

    Constructing RevienMemoryProvider without the SDK raises immediately with an
    actionable message, rather than half-initializing against ``object``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "hermes-agent is required to use RevienMemoryProvider. "
            "Install it with: pip install revien[hermes]  (and hermes-agent)."
        )


class RevienMemoryProvider(MemoryProvider if HERMES_AVAILABLE else _MissingHermesStub):
    """Hermes ``MemoryProvider`` backed by an in-process Revien stack.

    Construct with an optional ``db_path`` (defaults to the shared
    ``~/.revien/revien.db``). Hermes calls :meth:`initialize` once at startup;
    the stack is built lazily there (and on first use) so import and
    registration stay cheap and side-effect-free.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if not HERMES_AVAILABLE:
            # Belt-and-suspenders: the stub base already raises, but a direct
            # subclass call shouldn't slip through.
            raise ImportError(
                "hermes-agent is required to use RevienMemoryProvider. "
                "Install it with: pip install revien[hermes]  (and hermes-agent)."
            )
        self._db_path = _resolve_db_path(db_path)
        self._session_id: str = ""
        self._store: Optional[GraphStore] = None
        self._pipeline: Optional[IngestionPipeline] = None
        self._engine: Optional[RetrievalEngine] = None
        # Per-turn writes run on ONE serial worker draining this queue — not a
        # thread per turn. GraphStore holds a single sqlite connection with no
        # lock, so concurrent ingests would interleave transactions and lose
        # writes; serializing through one worker removes the race while keeping
        # sync_turn non-blocking (an unbounded put returns immediately). The
        # queue carries turn text; None is the stop sentinel.
        self._sync_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._sync_worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()

    # ── stack wiring (mirrors build_mcp_server's own-stack branch) ──────────

    def _ensure_stack(self) -> None:
        """Build GraphStore + SemanticIndex + pipeline + engine over the db.

        Idempotent and lazy: safe to call from initialize() and from any hook
        that might fire first. Same construction as
        ``build_mcp_server(db_path=...)`` minus clustering (a daemon-side
        refinement the engine runs fine without).
        """
        if self._store is not None:
            return
        store = GraphStore(db_path=self._db_path)
        semantic = SemanticIndex(store)
        self._store = store
        self._pipeline = IngestionPipeline(store, semantic=semantic)
        self._engine = RetrievalEngine(store, semantic=semantic)

    # ── core properties / lifecycle ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "revien"

    def is_available(self) -> bool:
        """Config-presence check, NO network (per Hermes' rule).

        Revien is local-first: once the package is importable there is no
        external dependency to probe. A fresh db is created on demand, so the
        only thing to verify is that a db path is resolvable — which it always
        is (explicit > REVIEN_DB > ~/.revien). Returns True; kept as a method so
        a future gate (e.g. require an existing graph) has one place to live.
        """
        try:
            _resolve_db_path(self._db_path)
            return True
        except Exception:  # noqa: BLE001 - availability probe never raises
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Called once at agent startup. ``kwargs`` may carry ``hermes_home``.

        We build the stack here. hermes_home is accepted for contract parity but
        not used to locate the db — Revien's own ``~/.revien`` (or REVIEN_DB) is
        the source of truth so the graph is shared with the daemon/CLI, not
        siloed under the Hermes profile.
        """
        self._session_id = session_id or ""
        self._ensure_stack()

    # ── automatic memory hooks ──────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context BEFORE a turn — injected, not model-invoked.

        Maps to ``engine.recall``. Returns a formatted context block (empty
        string when there's nothing) — Hermes injects the returned context ahead
        of the turn. Kept string-shaped because that's what the system-prompt
        injection consumes; the structured payload is available via the explicit
        revien_recall tool.
        """
        if not query or not query.strip():
            return ""
        self._ensure_stack()
        assert self._engine is not None
        try:
            response = self._engine.recall(query=query, top_n=_PREFETCH_TOP_N)
        except Exception:  # noqa: BLE001 - a recall miss must never break a turn
            return ""
        return self._format_context(response)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Any]] = None,
    ) -> None:
        """Persist a completed turn. MUST be non-blocking (Hermes contract).

        The whole reason P3's ``defer_embed`` was a prerequisite: the write is
        queued to the semantic index's pending queue rather than embedded inline,
        AND the ingest runs off the turn thread — so the turn returns immediately
        and never waits on a cold embedding model.

        Concurrency: the turn text is handed to a SINGLE serial worker
        (``_sync_loop``), not a fresh thread per turn. GraphStore shares one
        sqlite connection with no lock; two concurrent ingests would interleave
        their multi-statement read-modify-write and one turn's commit would
        flush the other's half-written transaction — lost/interleaved writes.
        One worker draining a FIFO queue serializes every write and preserves
        turn order. The ``put`` is on an unbounded queue, so it returns at once
        (Hermes' non-blocking contract).

        The turn is ingested as ONE conversational unit (source_id="hermes",
        content_type="conversation") — the scoping note's sync_turn-volume
        mitigation: conversational provenance keeps the per-turn firehose
        distinguishable from curated capture, and defer_embed keeps it cheap.
        """
        text = self._format_turn(user_content, assistant_content)
        if not text:
            return
        self._ensure_stack()
        self._ensure_sync_worker()
        self._sync_queue.put(text)

    def on_session_end(self, messages: Any) -> None:
        """Session ended — flush deferred embeddings, else no-op.

        We do NOT do a second summary-grade ingest here: every turn was already
        persisted by sync_turn, so re-ingesting the transcript would double-write
        the firehose. Instead we drain any embeddings still pending from this
        session's deferred writes so recall is fully semantic by the next
        session. Best-effort and safe when the semantic layer is disabled
        (drain_pending is a no-op there).
        """
        # Wait for EVERY queued turn to be ingested before draining, so all of
        # this session's deferred nodes are included (not just the newest turn).
        self._flush_sync()
        if self._store is None:
            return
        try:
            self._pipeline.semantic.drain_pending()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - flush is best-effort
            pass

    # ── explicit tools ──────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose recall + store as explicit tools the model can invoke.

        Shape note (churn-prone): Hermes injects these into the tool list, so
        the descriptors follow the standard name/description/parameters(JSON-
        Schema) tool shape. If a Hermes bump renames the schema key (e.g.
        ``input_schema``), that is a one-line change here.
        """
        return [
            {
                "name": "revien_recall",
                "description": (
                    "Search the user's persistent Revien memory graph for prior "
                    "context — decisions, facts, preferences from past sessions. "
                    "Call before answering anything that may depend on history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language description of what to recall.",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Max results (default 5, max 20).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "revien_store",
                "description": (
                    "Store a durable fact, decision, or preference in the user's "
                    "Revien memory graph so future sessions can recall it. One "
                    "self-contained statement per call; skip chit-chat."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory text — a standalone declarative statement.",
                        },
                    },
                    "required": ["content"],
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs: Any
    ) -> str:
        """Dispatch an explicit tool call to the engine/pipeline.

        Returns a JSON STRING. The Hermes ABC contracts
        ``handle_tool_call -> str`` (verified against
        ``agent/memory_provider.py`` — Hermes splices the result into the
        model's tool-result text stream). The payload is the SAME dict shape
        ``revien_recall`` / ``revien_store`` produce in the MCP face
        (mcp_server.py), json.dumps'd here so an MCP client and a Hermes client
        carry identical data over the same graph — the wire type differs
        because the two protocols do (MCP serializes the dict itself; Hermes
        wants text).
        """
        self._ensure_stack()
        # Shared validation core (revien/validation.py) — same rules as
        # POST /v1/recall|/v1/ingest and the MCP tools. ANY failure —
        # validation, malformed args, or an engine/pipeline exception —
        # comes back in the payload shape this face already uses for errors
        # ({"error": ...}, json.dumps'd), never as an exception into Hermes.
        try:
            if args is None:
                args = {}
            elif not isinstance(args, dict):
                raise ValidationError(
                    f"Invalid args (object expected): {type(args).__name__}"
                )
            if tool_name == "revien_recall":
                top_n = self._coerce_top_n(args.get("top_n", _PREFETCH_TOP_N))
                query = args.get("query", "")
                validate_recall(query=query, top_n=top_n)
                payload = self._tool_recall(query=query, top_n=top_n)
            elif tool_name == "revien_store":
                content = args.get("content", "")
                validate_ingest(
                    content=content,
                    source_id=_HERMES_SOURCE_ID,
                    content_type="note",
                )
                payload = self._tool_store(content=content)
            else:
                payload = {"error": f"unknown tool: {tool_name}"}
        except ValidationError as e:
            payload = {"error": str(e)}
        except Exception as e:  # noqa: BLE001 - never raise into Hermes
            payload = {
                "error": f"revien internal error ({type(e).__name__}): {e}"
            }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _coerce_top_n(raw: Any) -> int:
        """Coerce a Hermes-supplied top_n under the same honesty rules as the
        other faces (pydantic/FastMCP schema coercion): an int passes, an
        int-valued float or int-string coerces ("5" / 5.0 -> 5), a bool is
        refused (True is not "1 result"), and a fractional float is refused
        rather than silently truncated (5.9 must not become 5)."""
        message = (
            f"Invalid top_n (integer {TOP_N_MIN}..{TOP_N_MAX} expected): {raw!r}"
        )
        if isinstance(raw, bool):
            raise ValidationError(message)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if raw.is_integer():
                return int(raw)
            raise ValidationError(message)
        if isinstance(raw, str):
            try:
                return int(raw.strip())
            except ValueError:
                raise ValidationError(message)
        raise ValidationError(message)

    def _tool_recall(self, query: str, top_n: int = _PREFETCH_TOP_N) -> Dict[str, Any]:
        assert self._engine is not None
        response = self._engine.recall(query=query, top_n=top_n)
        return {
            "query": response.query,
            "results": [
                {
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "label": r.label,
                    "content": r.content,
                    "score": r.score,
                    "path": r.path,
                }
                for r in response.results
            ],
            "nodes_examined": response.nodes_examined,
            "retrieval_time_ms": response.retrieval_time_ms,
            "semantic_active": response.semantic_active,
            "semantic_note": response.semantic_note,
        }

    def _tool_store(self, content: str) -> Dict[str, Any]:
        assert self._pipeline is not None
        # Explicit store is a deliberate write — embed inline (defer_embed=False)
        # so it is semantically recallable immediately, unlike the per-turn
        # firehose that sync_turn defers.
        result = self._pipeline.ingest(
            IngestionInput(
                source_id=_HERMES_SOURCE_ID,
                content=content,
                content_type="note",
            )
        )
        return {
            "context_node_id": result.context_node_id,
            "nodes_created": result.nodes_created,
            "nodes_deduplicated": result.nodes_deduplicated,
            "edges_created": result.edges_created,
            "total_nodes_in_graph": result.total_nodes_in_graph,
            "total_edges_in_graph": result.total_edges_in_graph,
        }

    # ── config schema (drives `hermes memory setup`) ────────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Field descriptors for ``hermes memory setup``.

        Keys per Hermes' descriptor contract: ``key``, ``description``, optional
        ``secret`` / ``required`` / ``env_var`` / ``default``. Revien is
        local-first, so the only knob is the db path; there is no token to
        collect because an in-process provider makes no network calls.
        """
        return [
            {
                "key": "db_path",
                "description": (
                    "Path to the Revien SQLite database (default "
                    "~/.revien/revien.db — shared with the revien daemon/CLI)."
                ),
                "required": False,
                "env_var": "REVIEN_DB",
                "default": "",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret config. Applies db_path if the setup flow set one."""
        db_path = (values or {}).get("db_path")
        if db_path:
            # Re-point the resolved path; rebuild lazily on next use.
            self._db_path = _resolve_db_path(db_path)
            self._store = None
            self._pipeline = None
            self._engine = None

    # ── system prompt injection ─────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        """One line telling the model what Revien is and that recall is automatic."""
        return (
            "Revien is the user's persistent memory graph. Relevant memories are "
            "recalled automatically before each turn; you may also call "
            "revien_recall to search it and revien_store to save a durable fact."
        )

    # ── cleanup ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Process exit — finish every queued write, stop the worker, close db."""
        self._flush_sync()
        worker = self._sync_worker
        if worker is not None and worker.is_alive():
            self._sync_queue.put(None)  # stop sentinel (FIFO: after all turns)
            worker.join(timeout=5.0)
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001 - shutdown never raises
                pass

    # ── serial sync worker ──────────────────────────────────────────────────

    def _ensure_sync_worker(self) -> None:
        """Start the single serial ingest worker if it isn't already running.
        One worker is what serializes writes on the shared sqlite connection."""
        if self._sync_worker is not None and self._sync_worker.is_alive():
            return
        with self._worker_lock:
            if self._sync_worker is not None and self._sync_worker.is_alive():
                return
            worker = threading.Thread(
                target=self._sync_loop, name="revien-sync-turn", daemon=True
            )
            self._sync_worker = worker
            worker.start()

    def _sync_loop(self) -> None:
        """Drain the sync queue serially. A per-item guard keeps one bad turn
        from wedging the queue (task_done always fires), so _flush_sync can
        never hang on an ingest error; None is the stop sentinel."""
        while True:
            item = self._sync_queue.get()
            try:
                if item is None:
                    return
                pipeline = self._pipeline
                if pipeline is not None:
                    pipeline.ingest(
                        IngestionInput(
                            source_id=_HERMES_SOURCE_ID,
                            content=item,
                            content_type=_CONVERSATION,
                            defer_embed=True,  # persist now, embed on drain/sweep
                        )
                    )
            except Exception:  # noqa: BLE001 - a sync failure never breaks Hermes
                pass
            finally:
                self._sync_queue.task_done()

    def _flush_sync(self) -> None:
        """Block until every enqueued turn has been ingested — ALL in-flight
        turns, not just the newest. Used by on_session_end/shutdown (and the
        tests). Safe when nothing was ever enqueued (returns at once)."""
        self._sync_queue.join()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_turn(user_content: str, assistant_content: str) -> str:
        """Render a turn as one conversational unit ("User: ... Assistant: ...").

        Matches the adapter transcript convention (claude_code / codex) so the
        rule-based extractor sees the shape it already handles.
        """
        parts: List[str] = []
        if user_content and user_content.strip():
            parts.append(f"User: {user_content.strip()}")
        if assistant_content and assistant_content.strip():
            parts.append(f"Assistant: {assistant_content.strip()}")
        return "\n".join(parts)

    @staticmethod
    def _format_context(response: Any) -> str:
        """Format a RetrievalResponse into an injectable context block."""
        if not getattr(response, "results", None):
            return ""
        lines = ["## Relevant memory (Revien)"]
        for r in response.results:
            lines.append(f"- {r.content}")
        return "\n".join(lines)


def register(ctx: Any) -> None:
    """Hermes plugin entry point — called by the memory-plugin discovery system.

    Registers a single provider instance. Named ``register`` and taking ``ctx``
    per the Hermes contract; ``ctx.register_memory_provider`` is the only ctx
    method used.
    """
    ctx.register_memory_provider(RevienMemoryProvider())
