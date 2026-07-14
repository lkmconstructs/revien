"""
Revien MCP server (LEG P5) — thin protocol face over the same engine surface
as ``POST /v1/recall`` and ``POST /v1/ingest``. No new engine behavior lives
here: the tools serialize exactly what the REST endpoints serialize, so an
MCP client and a REST client see the same memory.

Two transports, one server:
  - stdio: ``revien mcp`` runs this as its own process over the configured
    database. SQLite WAL lets it coexist with a running daemon.
  - streamable HTTP: the daemon mounts the same server at /mcp when
    REVIEN_MCP_HTTP is set (see revien/daemon/server.py). Default off.

The MCP SDK is an optional extra (``pip install revien[mcp]``) — this module
imports cleanly without it and callers check MCP_AVAILABLE, mirroring the
langchain peer-dependency pattern in revien/adapters/__init__.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

# Peer-dependency guard: the core install stays lean; everything below that
# touches the SDK checks MCP_AVAILABLE first.
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FastMCP = None  # type: ignore[assignment]
    TransportSecuritySettings = None  # type: ignore[assignment]
    MCP_AVAILABLE = False


_INSTRUCTIONS = (
    "Revien is the user's persistent memory graph. Recall from it before "
    "answering anything that depends on the user's history, preferences, or "
    "prior decisions; store durable facts and decisions back into it as they "
    "are made. It is local-first and survives across conversations."
)


def build_mcp_server(
    db_path: Optional[str] = None,
    *,
    engine: Any = None,
    pipeline: Any = None,
) -> "FastMCP":
    """Build the Revien FastMCP server.

    Two wiring modes:
      - ``build_mcp_server(db_path=...)`` owns its own stack (GraphStore +
        SemanticIndex + pipeline + engine) over that database — the stdio
        path, and what tests use with a temp db.
      - ``build_mcp_server(engine=..., pipeline=...)`` rides an existing
        stack — the daemon's /mcp mount, so MCP traffic and REST traffic hit
        the same shared engine/pipeline (and the same vec0 table).

    Raises ImportError when the SDK is absent — callers gate on MCP_AVAILABLE.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "The MCP SDK is not installed. Install with: pip install revien[mcp]"
        )

    # Own-stack mode opens a GraphStore this module is responsible for handing
    # back — it rides on the returned server object and close_mcp_server() is
    # the deterministic release (the CLI's try/finally, tests' teardown).
    # Ride-along mode attaches None: the daemon owns that store's lifecycle.
    owned_store = None
    try:
        if engine is None or pipeline is None:
            # Own stack — same wiring as create_app minus clustering
            # (community boosts are a daemon-side refinement; the engine
            # runs fine without).
            from revien.graph.store import GraphStore
            from revien.ingestion.pipeline import IngestionPipeline
            from revien.retrieval.engine import RetrievalEngine
            from revien.semantic.index import SemanticIndex

            store = GraphStore(db_path=db_path or "revien.db")
            owned_store = store
            semantic = SemanticIndex(store)
            if pipeline is None:
                pipeline = IngestionPipeline(store, semantic=semantic)
            if engine is None:
                engine = RetrievalEngine(store, semantic=semantic)

        server = _build_server(engine, pipeline)
    except BaseException:
        # Anything after the own-stack store opened — semantic/pipeline/
        # engine wiring, FastMCP construction, tool registration — failed:
        # close the store before propagating, or the handle leaks (and on
        # Windows keeps the db file locked). Ride-along mode owns nothing.
        if owned_store is not None:
            owned_store.close()
        raise

    # The store the own-stack branch opened (None in ride-along mode). The
    # sqlite connection's own finalizer at GC is the only fallback — callers
    # that opened a stack close it via close_mcp_server().
    server._revien_store = owned_store
    return server


def _build_server(engine: Any, pipeline: Any) -> "FastMCP":
    """FastMCP construction + tool registration over an existing stack.

    Split out of build_mcp_server so the factory can close an own-stack
    store when anything in here raises."""
    # streamable_http_path="/": the daemon routes /mcp itself (exact Route,
    # see daemon/server.py) — this only matters if someone runs the SDK's own
    # wrapper app, where "/" keeps the endpoint at the served root.
    # stateless_http + json_response: recall/store are independent calls with
    # no server-side session state worth keeping, and plain JSON responses
    # work with every client (ChatGPT connectors included) without an SSE
    # stream per request. Neither setting affects the stdio transport.
    server = FastMCP(
        "revien",
        instructions=_INSTRUCTIONS,
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        # The SDK default rejects any Host header outside localhost (421),
        # which kills the remote story outright — a Tailscale client sends
        # Host: <machine-name> even WITH a valid bearer. So the SDK check is
        # off, and the equivalent protections live at the mount boundary
        # (_CaptureAuthASGI in daemon/server.py): capture-token gate for
        # remote callers, and an Origin-header refusal for browser callers —
        # a rebinding/CSRF page runs AS loopback, so the loopback exemption
        # alone must never be the last line against browser JS.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )

    @server.tool()
    def revien_recall(
        query: str,
        top_n: int = 5,
        include_context: bool = False,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the user's persistent memory graph.

        Call this BEFORE answering anything that might depend on prior
        conversations: at the start of a session, when the user references
        past work ("that bug we fixed", "my usual setup"), or when you are
        about to say "I don't have context on that" — check memory first.

        Args:
            query: Natural-language description of what you need. Plain
                phrasing works best ("database migration decision", not a
                keyword soup).
            top_n: Maximum results (default 5, max 20). Raise it when
                surveying a topic rather than answering a point question.
            include_context: Also return the verbatim conversation/context
                nodes, not just extracted facts. Use when you need original
                wording.
            as_of: ISO-8601 timestamp for bi-temporal queries — "what was
                true AT this time". Superseded facts whose validity window
                covers the moment come back. Leave unset for current truth.

        Returns ranked results with content, score, and provenance path,
        plus semantic_note explaining if retrieval ran degraded.
        """
        as_of_dt = None
        if as_of:
            try:
                as_of_dt = datetime.fromisoformat(as_of)
            except ValueError:
                raise ValueError(f"Invalid as_of (ISO-8601 expected): {as_of}")

        response = engine.recall(
            query=query,
            top_n=top_n,
            include_context=include_context,
            as_of=as_of_dt,
        )
        # Mirror of POST /v1/recall's serialization (daemon/server.py) with
        # include_tensions off — the MCP face must not invent a new shape.
        return {
            "query": response.query,
            "results": [
                {
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "label": r.label,
                    "content": r.content,
                    "score": r.score,
                    "score_breakdown": r.score_breakdown,
                    "path": r.path,
                }
                for r in response.results
            ],
            "nodes_examined": response.nodes_examined,
            "retrieval_time_ms": response.retrieval_time_ms,
            "semantic_active": response.semantic_active,
            "semantic_note": response.semantic_note,
        }

    @server.tool()
    def revien_store(
        content: str,
        source_id: str = "mcp",
        content_type: str = "note",
        defer_embed: bool = False,
    ) -> Dict[str, Any]:
        """Store a durable memory in the user's persistent memory graph.

        Store decisions, stable facts, preferences, and outcomes — the things
        a future conversation will need: "chose Postgres over SQLite for the
        API because of concurrent writes", "user's staging server is
        blue.example.com". Do NOT store chit-chat, transient state, or
        content the user asked to keep out of memory. One coherent fact or
        decision per call; write it so it stands alone without this
        conversation as context.

        Args:
            content: The memory text. Self-contained declarative statements
                extract best.
            source_id: Where this came from (defaults to "mcp"). Set it to
                the client/agent name if you have one — it becomes provenance.
            content_type: "note" (default), "conversation", "document", or
                "code".
            defer_embed: Persist now, embed later. Set True only for
                latency-critical capture; the memory is keyword-searchable
                immediately and semantically searchable after the queue
                drains.

        Returns the created context node id and node/edge counts.
        """
        from revien.ingestion.pipeline import IngestionInput

        result = pipeline.ingest(
            IngestionInput(
                source_id=source_id,
                content=content,
                content_type=content_type,
                defer_embed=defer_embed,
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

    return server


def close_mcp_server(server: Any) -> None:
    """Release the GraphStore an own-stack ``build_mcp_server`` opened.

    No-op for ride-along servers (engine/pipeline were injected — the daemon
    owns that store) and safe to call twice: GraphStore.close is idempotent.
    ``revien mcp`` calls this in a try/finally around ``server.run()`` so the
    db file handle is released deterministically, not at interpreter exit.
    """
    store = getattr(server, "_revien_store", None)
    if store is not None:
        store.close()
