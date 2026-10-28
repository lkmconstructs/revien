"""
Revien REST API Server — FastAPI application exposing all Revien endpoints.
Runs as local daemon on port 7437 or as hosted service.
"""

import logging
import asyncio
import time
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import secrets

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic import ValidationError as SchemaValidationError

from revien import __version__, extensions, pairing
from revien.daemon.heartbeat import (
    DAEMON_FEATURES, HEARTBEAT_INTERVAL, clear_heartbeat, write_heartbeat)
from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.graph.clustering import CommunityDetector
from revien.ingestion.pipeline import IngestionInput, IngestionOutput, IngestionPipeline, TIMESTAMP_SOURCES
from revien.retrieval.engine import RetrievalEngine, RetrievalResponse
from revien.semantic.index import SemanticIndex
from revien.skills.ingest import sort_user_before_engine
from revien.skills.proposals import accept_proposal, decline_proposal
from revien.validation import ValidationError, validate_ingest, validate_recall


# ── Request/Response Models ───────────────────────────────

class IngestRequest(BaseModel):
    source_id: str
    content: str
    content_type: str = "conversation"
    timestamp: Optional[str] = None
    # Where `timestamp` came from (content/capture/mtime/import). Omitted:
    # "content" when the caller supplied a timestamp, else "capture".
    timestamp_source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Capture path: persist now, embed later — the response returns before any
    # embedding-model load. Queued nodes become recallable at the next semantic
    # recall (search drains the queue first) or the ~30s idle sweep. NOTE: a
    # verbatim-only capture is not keyword-anchorable in the gap (keyword
    # anchors exclude CONTEXT nodes) — drain-at-search is the guarantee.
    defer_embed: bool = False
    # Origin Layer (WS0) passthrough: an external caller that knows its own
    # provenance can set these; omitted origin_runtime falls back to
    # deriving it from source_id in the pipeline's stamp loop.
    origin_runtime: Optional[str] = None
    origin_source: Optional[str] = None
    project_key: Optional[str] = None
    session_key: Optional[str] = None


class IngestResponse(BaseModel):
    context_node_id: str
    nodes_created: int
    nodes_deduplicated: int
    edges_created: int
    total_nodes_in_graph: int
    total_edges_in_graph: int
    # True when the suppression gate refused the capture (a forget said it
    # must not come back); nothing was written. A capture that arrives
    # through a shared channel has no stable identity the gate can protect.
    suppressed: bool = False
    suppression_note: str = ""
    # forget.key is not the key the earlier forgets were made with: they
    # match nothing, so a secret they kept out can be captured again.
    warnings: List[str] = []


class ConsolidateRequest(BaseModel):
    """Dream-mode pass (B3.1). Reindex is a backfill, off by default;
    orphan invalidation is soft/reversible and strictly opt-in. alias (alias
    leg) is opt-in too — it's the one pass here that WRITES new edges rather
    than only reporting or demoting."""
    decay: bool = True
    recluster: bool = True
    reindex: bool = False
    invalidate_orphans: bool = False
    alias: bool = False


class RecallRequest(BaseModel):
    query: str
    top_n: int = 5
    min_score: float = 0.01
    include_invalidated: bool = False
    include_context: bool = False
    include_tensions: bool = False
    # Bi-temporal query time (B2), ISO-8601: "what was true AT this time?"
    as_of: Optional[str] = None
    # Wire format (LEG P2): "json" (default, unchanged response) or "toon"
    # — Token-Oriented Object Notation, the same payload serialized for
    # fewer tokens in a consuming LLM's context window (see revien/toon.py).
    format: str = "json"
    # Origin Layer (WS0 Leg B): filter candidates/results to one or more
    # origin_runtime values (e.g. "claude-code", or ["claude-code", "codex"]).
    # None (default): no filtering, response byte-identical.
    source: Optional[Union[str, List[str]]] = None
    # Usage signal (N1): issue a receipt for this recall. The response gains
    # recall_id + receipt_expires_at; hand recall_id back with the answer to
    # POST /v1/usage/observe within 24h, or it is flagged. False (default):
    # response byte-identical.
    receipt: bool = False
    # Who is recalling, recorded on the receipt (e.g. "langchain", "api").
    runtime: Optional[str] = None


class ObserveRequest(BaseModel):
    recall_id: str
    answer: str
    # The prompt the recall served, if the client has it: its words are
    # discounted as evidence, same as the query's.
    prompt: str = ""


class NodeUpdateRequest(BaseModel):
    label: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EdgeCreateRequest(BaseModel):
    edge_type: str
    source_node_id: str
    target_node_id: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_set_by: str = ""
    source_context: str = ""


class EdgeResponse(BaseModel):
    edge_id: str
    edge_type: str
    source_node_id: str
    target_node_id: str
    weight: float
    created_at: str
    metadata: Dict[str, Any]
    confidence: float
    confidence_set_by: str
    source_context: str


class NodeResponse(BaseModel):
    node_id: str
    node_type: str
    label: str
    content: str
    source_id: str
    created_at: str
    last_accessed: str
    access_count: int
    metadata: Dict[str, Any]
    # Bi-temporal validity (B2) — when the claim WAS TRUE. Null = unbounded.
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    # Origin Layer (WS0 Leg B): surfaced so a GET /v1/nodes?origin_runtime=…
    # caller can see the field it filtered on.
    origin_runtime: Optional[str] = None
    origin_source: Optional[str] = None
    project_key: Optional[str] = None
    session_key: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    node_count: int
    edge_count: int
    uptime_seconds: float
    version: str
    # What this daemon can do (revien/daemon/heartbeat.py:DAEMON_FEATURES).
    # A daemon from before fencing has no such field, which is how
    # `revien sync reset` recognises it. Names only: no path, no id.
    features: List[str] = []
    # "ok", or "mismatch" when forget.key is not the key the earlier forgets
    # were made with (they then match nothing). Says nothing else.
    forget_key: str = "ok"


class SyncResponse(BaseModel):
    status: str
    message: str


def _normalize_origin_runtime_param(
    raw: Optional[List[str]],
) -> Optional[List[str]]:
    """GET /v1/nodes?origin_runtime=... (G5) — accept BOTH shapes a caller
    might reasonably send: a repeated query param (FastAPI/Starlette gives
    that to us as ``raw`` already split, e.g. ``["claude-code", "codex"]``)
    AND a single comma-separated value (arrives as ``["claude-code,codex"]``
    — one raw item that still has commas in it, since Starlette doesn't
    split on commas itself). Every raw item is comma-split, then every
    resulting piece is stripped and blanks are dropped — so a repeated
    param, a comma-joined param, a single value, and an empty value
    (``?origin_runtime=``) all normalize the same way.

    None (param absent from the query string entirely) passes straight
    through as None — unfiltered, unchanged from before. Any other input
    normalizes to a list that MAY be empty (every segment was blank) —
    that empty-but-present list is what makes store.list_nodes fail
    closed (G4) instead of silently returning everything.
    """
    if raw is None:
        return None
    parts: List[str] = []
    for item in raw:
        parts.extend(item.split(","))
    return [p.strip() for p in parts if p.strip()]


# ── Capture auth (P3: remote capture is opt-in, token-gated) ─────────

# starlette's TestClient reports host "testclient"; it exercises the same
# in-process path as a local adapter, so it counts as loopback.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient", ""}


def check_capture_auth(client_host: Optional[str], auth_header: str) -> None:
    """Gate for /v1/ingest. Raises HTTPException on refusal.

    Loopback callers are never gated — the local adapter path is unchanged
    whether or not a token is configured. A remote caller is refused outright
    unless a pairing token is configured (remote capture is opt-in) — either
    ``REVIEN_CAPTURE_TOKEN`` or a minted ``revien token`` file, see
    ``revien.pairing.configured_token`` — and with one configured must present
    ``Authorization: Bearer <token>``. Comparison is constant-time.

    G10: the scheme ("Bearer") is compared case-insensitively — a client
    that sends ``bearer <token>`` (lowercase, RFC 6750 doesn't mandate a
    case) is not refused just for that. The header is split on the FIRST
    whitespace only (``partition``, not ``strip`` + exact-string compare)
    so the scheme and the token are judged independently: a wrong-case
    scheme with the right token now passes, but a right scheme with a
    malformed/padded token (extra whitespace inside what should be just
    the token) still fails, because the token half is compared with
    ``compare_digest`` byte-for-byte, no stripping of its own.
    """
    host = (client_host or "").strip().lower()
    if host in _LOOPBACK_HOSTS:
        return
    expected = pairing.configured_token()
    if not expected:
        raise HTTPException(
            403,
            "Remote access is disabled — run `revien token` on the Revien "
            "host and pair.",
        )
    scheme, _, provided = (auth_header or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        provided.encode(), expected.encode()
    ):
        raise HTTPException(401, "Invalid or missing capture token.")


def require_mutation_auth(client_host: Optional[str], auth_header: str) -> None:
    """Gate for remote-mutation endpoints (Leg D: skill accept/decline).

    Identical rule to ``check_capture_auth`` — loopback exempt, remote needs
    the configured pairing token as ``Bearer <token>`` — kept as a separate
    name so mutation routes read as a deliberate choice rather than reuse of
    the capture-specific gate.
    """
    check_capture_auth(client_host, auth_header)


def _require_mutation_auth_dep(request: Request) -> None:
    """FastAPI dependency wrapping require_mutation_auth — one implementation,
    wired via ``dependencies=[_mutation_auth]`` on every mutating route
    instead of a copy-pasted call + ``http_request: Request`` param each."""
    require_mutation_auth(
        request.client.host if request.client else "",
        request.headers.get("authorization", ""),
    )


_mutation_auth = Depends(_require_mutation_auth_dep)


class _CaptureAuthASGI:
    """ASGI wrapper applying check_capture_auth to a whole mount (P5: /mcp).

    MCP traffic bypasses the FastAPI route layer, so the ingest endpoint's
    dependency can't cover it — this re-applies the exact same gate at the
    ASGI boundary for EVERYTHING under the mount (recall-ish and ingest-ish
    alike: a recall response leaks memory just as surely as ingest writes it).
    Loopback exempt, remote needs Bearer REVIEN_CAPTURE_TOKEN — unchanged
    semantics, one source of truth.

    Browser gate: any request carrying an Origin header is refused unless
    that origin is allowlisted via REVIEN_MCP_ALLOWED_ORIGINS. The loopback
    exemption exists for LOCAL PROCESSES; a DNS-rebinding/CSRF page running
    in the user's browser is also loopback, and without this check its JS
    could call revien_recall and exfiltrate the whole graph. Native MCP
    clients (Claude, Codex, anything non-browser) send no Origin, so they
    are unaffected; browsers always send one on cross-origin POST.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _allowed_origins() -> set:
        raw = os.environ.get("REVIEN_MCP_ALLOWED_ORIGINS", "")
        return {o.strip().lower() for o in raw.split(",") if o.strip()}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            host = client[0] if client else ""
            auth = ""
            origin = ""
            for key, value in scope.get("headers", []):
                if key == b"authorization":
                    auth = value.decode("latin-1")
                elif key == b"origin":
                    origin = value.decode("latin-1")
            try:
                if origin and origin.strip().lower() not in self._allowed_origins():
                    # Name the refused origin so a legitimate non-browser
                    # client that unexpectedly sends one can be allowlisted
                    # from the error alone.
                    raise HTTPException(
                        403,
                        f"Browser-origin requests to the MCP endpoint are "
                        f"refused (origin: {origin!r}). Set "
                        f"REVIEN_MCP_ALLOWED_ORIGINS to allow a specific "
                        f"origin.",
                    )
                check_capture_auth(host, auth)
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _mcp_http_enabled() -> bool:
    """REVIEN_MCP_HTTP gate — default OFF; the shipped REST surface is
    byte-identical when unset."""
    return os.environ.get("REVIEN_MCP_HTTP", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── App Factory ───────────────────────────────────────────

async def _heartbeat_loop(store) -> None:
    """Write the daemon heartbeat now and every HEARTBEAT_INTERVAL seconds."""
    while True:
        try:
            write_heartbeat(store)
        except Exception:  # noqa: BLE001 - never take the daemon down
            pass
        await asyncio.sleep(HEARTBEAT_INTERVAL)


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Database path resolution is explicit argument > ``REVIEN_DB_PATH`` env var >
    ``revien.db``. The CLI passes an explicit path, but honoring the env var
    keeps direct ASGI/Docker usage from silently serving a fresh local database.
    """
    db_path = db_path or os.environ.get("REVIEN_DB_PATH", "revien.db")

    # Shared state — built before the FastAPI app so the optional MCP mount
    # (below) can ride the same engine/pipeline instead of opening a second
    # stack over the same database.
    store = GraphStore(db_path=db_path)
    ops = GraphOperations(store)
    # One shared semantic index (opt-in). Self-disables without the `semantic`
    # extra, so ingest/recall run the unchanged graph path. Sharing it means
    # ingest-time embeddings and recall-time search hit the same vec0 table.
    semantic = SemanticIndex(store)
    pipeline = IngestionPipeline(store, semantic=semantic)
    clustering = CommunityDetector(db_path=db_path)
    engine = RetrievalEngine(store, clustering=clustering, semantic=semantic)
    start_time = time.time()

    # Load existing community assignments (or run initial clustering)
    if not clustering.load_from_db():
        clustering.run()

    # ── Optional MCP surface (P5) ─────────────────────
    # Mounted only when the SDK is importable AND REVIEN_MCP_HTTP is set.
    # Default off: with the gate unset, the app carries no /mcp route and a
    # no-op lifespan — the shipped REST surface is unchanged.
    mcp_server = None
    if _mcp_http_enabled():
        from revien.mcp_server import MCP_AVAILABLE, build_mcp_server

        if MCP_AVAILABLE:
            mcp_server = build_mcp_server(engine=engine, pipeline=pipeline)

    def _forget_key_problem():
        from revien.forget import gate as _g
        try:
            return _g.key_problem(store)
        except Exception:  # noqa: BLE001
            return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Say it at start, loudly, if the earlier forgets no longer apply
        # (forget.key lost or replaced): the daemon would capture again what
        # they kept out. It starts anyway; refusing would only hide it.
        problem = _forget_key_problem()
        if problem:
            from revien.forget import gate as _g
            logging.getLogger("revien").error(
                "forget.key mismatch: %s", _g.key_problem_text(problem))
        # Startup: start the scheduler INSIDE uvicorn's running loop when the
        # daemon attached one (RevienDaemon sets app.state.scheduler before
        # uvicorn.run). AsyncIOScheduler.start() needs the running event
        # loop — started before uvicorn.run, as the daemon used to, it
        # either raises RuntimeError outright (apscheduler 3.11 + py3.13) or
        # binds a loop that never runs, so sync/drain/dream jobs never fired.
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is not None and not scheduler.is_running:
            scheduler.start()
        # A daemon (the app has a scheduler) says so in the database it
        # serves, so an operation that depends on fencing can tell. Refreshed
        # while it runs, removed on a clean exit. Best effort: a failure here
        # never stops the daemon.
        beat_task = None
        if scheduler is not None:
            beat_task = asyncio.ensure_future(_heartbeat_loop(store))
        # The MCP session manager needs a running task group for the life of
        # the app. Mounted sub-app lifespans don't run under Starlette, so it
        # is wired here. No-op when MCP is off.
        try:
            if mcp_server is not None:
                async with mcp_server.session_manager.run():
                    yield
            else:
                yield
        finally:
            # Shutdown, innermost-out. The MCP session manager (above) has
            # already exited by the time this runs. Stop the scheduler first
            # if the daemon attached one (its jobs write through the store),
            # then close the store — the ONE sqlite connection every layer
            # rides, semantic index included. Everything here is idempotent,
            # so a second teardown is a no-op. This is also what releases
            # the db file handle when a TestClient exits: without it, the
            # connection create_app opened lived until GC and Windows
            # refused the temp-db unlink (WinError 32).
            if beat_task is not None:
                beat_task.cancel()
                clear_heartbeat(store)
            scheduler = getattr(app.state, "scheduler", None)
            if scheduler is not None:
                scheduler.stop()
            store.close()

    app = FastAPI(
        title="Revien",
        description="Graph-based memory engine for AI systems. Memory that returns.",
        version=__version__,
        lifespan=lifespan,
    )

    # A Node or Edge that does not fit the schema (a label over 200
    # characters, say) is a refusal of what the request carried, not a crash:
    # 422, naming the field and the kind of problem only. pydantic's own text
    # quotes the value, and the value is the caller's text. Any other model
    # failing is a server bug and stays a 500.
    @app.exception_handler(SchemaValidationError)
    async def _schema_limit_without_the_text(request, exc):
        if exc.title not in ("Node", "Edge"):
            raise exc
        problems = [{"field": ".".join(str(p) for p in e.get("loc", ())) or exc.title,
                     "problem": e.get("type", "invalid")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": {
            "status": "refused", "reason": "schema_limit",
            "problems": problems,
            "message": (f"This request produced a {exc.title.lower()} the memory "
                        "schema does not accept. Nothing you sent is repeated "
                        "here.")}})

    if mcp_server is not None:
        from starlette.routing import Route as _StarletteRoute

        # streamable_http_app() is called for its side effect: it lazily
        # creates the session manager the lifespan above runs. Its wrapper
        # Starlette app is NOT mounted — a Mount("/mcp") + inner Route("/")
        # 307-redirects the bare POST /mcp to /mcp/, which not every MCP
        # client follows. Streamable HTTP is a single-endpoint transport
        # (POST/GET/DELETE on one path), so an exact Route serves /mcp with
        # no redirect. Capture auth wraps the whole surface — recall AND
        # store get the /v1/ingest gate.
        mcp_server.streamable_http_app()

        class _MCPEndpoint:
            """ASGI face over the session manager (class instance, so
            Starlette treats it as an ASGI app, not a request handler)."""

            async def __call__(self, scope, receive, send):
                await mcp_server.session_manager.handle_request(
                    scope, receive, send
                )

        app.router.routes.append(
            _StarletteRoute("/mcp", endpoint=_CaptureAuthASGI(_MCPEndpoint()))
        )

    # Store references on app for access by daemon/scheduler
    app.state.store = store
    app.state.ops = ops
    app.state.pipeline = pipeline
    app.state.engine = engine
    app.state.clustering = clustering
    app.state.semantic = semantic
    # What a forget needs beyond the store: the vectors to drop, the
    # training database to scrub, the community labels to rebuild.
    ops.semantic = semantic
    ops.clustering = clustering
    ops.training_dbs = (engine.training_loop.db_path,)
    # With neural ranking on, a forget retrains the model from the scrubbed
    # signals; with it off nothing trains and the model stays marked stale.
    ops.retrain = engine.force_train if engine.neural_enabled else None
    ops.retrain_reason = engine.last_train_failure if engine.neural_enabled else None

    # ── POST /v1/ingest ───────────────────────────────

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest, http_request: Request):
        """Ingest raw content. Builds graph nodes and edges."""
        check_capture_auth(
            http_request.client.host if http_request.client else "",
            http_request.headers.get("authorization", ""),
        )
        # Shared validation core (revien/validation.py) — same rules as the
        # MCP and Hermes faces. An unparseable timestamp is a 400 now, not a
        # silent drop; blank content/source_id and unknown content_type are
        # refused before they create empty memories.
        try:
            ts = validate_ingest(
                content=request.content,
                source_id=request.source_id,
                content_type=request.content_type,
                timestamp=request.timestamp,
            )
        except ValidationError as e:
            raise HTTPException(400, str(e))

        if request.timestamp_source is not None and request.timestamp_source not in TIMESTAMP_SOURCES:
            raise HTTPException(
                400, f"Unknown timestamp_source {request.timestamp_source!r}; "
                     f"expected one of {sorted(TIMESTAMP_SOURCES)}")
        input_data = IngestionInput(
            source_id=request.source_id,
            content=request.content,
            content_type=request.content_type,
            timestamp=ts,
            timestamp_source=request.timestamp_source
            or ("content" if ts is not None else "capture"),
            metadata=request.metadata,
            defer_embed=request.defer_embed,
            origin_runtime=request.origin_runtime,
            origin_source=request.origin_source,
            project_key=request.project_key,
            session_key=request.session_key,
        )
        try:
            result = pipeline.ingest(input_data)
        except SchemaValidationError:
            raise  # a ValueError too; answered above without the text
        except ValueError as e:
            # Origin Layer (WS0): an unknown declared origin_runtime/
            # origin_source is a client error, not a 500.
            raise HTTPException(400, str(e))

        # Notify clustering — recluster if threshold reached
        if clustering.notify_ingest():
            clustering.run()

        return IngestResponse(
            context_node_id=result.context_node_id,
            nodes_created=result.nodes_created,
            nodes_deduplicated=result.nodes_deduplicated,
            edges_created=result.edges_created,
            total_nodes_in_graph=result.total_nodes_in_graph,
            total_edges_in_graph=result.total_edges_in_graph,
            suppressed=result.suppressed,
            suppression_note=result.suppression_note,
            warnings=list(result.warnings),
        )

    # ── POST /v1/recall ───────────────────────────────

    @app.post("/v1/recall")
    async def recall(request: RecallRequest, http_request: Request):
        """Query memory. Returns ranked nodes by three-factor score."""
        if request.receipt:
            # A receipt writes a row (and can become a flag), so it takes the
            # same gate as other writes; plain recall stays ungated.
            require_mutation_auth(
                http_request.client.host if http_request.client else "",
                http_request.headers.get("authorization", ""),
            )
        if request.format not in ("json", "toon"):
            raise HTTPException(400, f"Invalid format (json|toon expected): {request.format}")
        # Shared validation core — blank query, out-of-range top_n (reject,
        # don't clamp: the engine's internal clamp stays as belt), and
        # negative min_score are 400s.
        try:
            validate_recall(
                query=request.query,
                top_n=request.top_n,
                min_score=request.min_score,
            )
        except ValidationError as e:
            raise HTTPException(400, str(e))
        as_of = None
        if request.as_of:
            try:
                as_of = datetime.fromisoformat(request.as_of)
            except ValueError:
                raise HTTPException(400, f"Invalid as_of (ISO-8601 expected): {request.as_of}")

        response = engine.recall(
            query=request.query,
            top_n=request.top_n,
            min_score=request.min_score,
            include_invalidated=request.include_invalidated,
            include_context=request.include_context,
            include_tensions=request.include_tensions,
            as_of=as_of,
            source=request.source,
        )
        payload = {
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
                    # Origin Layer (WS0 Leg B): present on EVERY result, None
                    # allowed — key always present so TOON's tabular reshape
                    # stays uniform.
                    "origin_runtime": r.origin_runtime,
                    "origin_source": r.origin_source,
                    "project_key": r.project_key,
                    # When it was said (ISO-8601, speaker offset) or None; always present.
                    "recorded_at": r.recorded_at,
                    "recorded_at_source": r.recorded_at_source,
                    # Only present when asked for — the flag-off response
                    # shape is byte-identical to pre-B1.
                    **({"tensions": r.tensions} if request.include_tensions else {}),
                }
                for r in response.results
            ],
            "nodes_examined": response.nodes_examined,
            "retrieval_time_ms": response.retrieval_time_ms,
            # Degrade visibility: which retrieval path served this query.
            # semantic_note is null when the hybrid path is active, else a
            # one-line reason recall is running graph-only (degraded).
            "semantic_active": response.semantic_active,
            "semantic_note": response.semantic_note,
            # Skills leg D2: draft engine-origin skill proposals relevant to
            # this query. Always present, possibly [].
            "skill_proposals": response.skill_proposals,
        }
        from revien.usage.observer import usage_mode

        if request.receipt and usage_mode() == "off":
            # REVIEN_USAGE=off: nothing will judge it, so a receipt would only
            # expire into a false flag.
            payload["recall_id"] = None
            payload["receipt_expires_at"] = None
            payload["receipt_note"] = ("no receipt issued: the usage signal is "
                                       "off (REVIEN_USAGE=off)")
        elif request.receipt and not response.results:
            # Nothing was recalled, so nothing could be used: a receipt would
            # only expire into a meaningless flag.
            payload["recall_id"] = None
            payload["receipt_expires_at"] = None
            payload["receipt_note"] = ("no receipt issued: the recall "
                                       "returned no results")
        elif request.receipt:
            from revien.usage import UsageObserver

            issued = UsageObserver(store, mode="shadow").issue_receipt(
                query=response.query,
                node_ids=[r.node_id for r in response.results],
                runtime=(request.runtime or "api").strip() or "api",
            )
            payload["recall_id"] = issued.recall_id
            payload["receipt_expires_at"] = issued.expires_at.isoformat()
        if request.format == "toon":
            # Same payload, TOON wire format (LEG P2). text/toon is the
            # spec's provisional media type. The json path above is
            # untouched — byte-identical to pre-P2.
            from revien.toon import serialize_recall

            return PlainTextResponse(
                serialize_recall(payload), media_type="text/toon; charset=utf-8"
            )
        return payload

    # ── GET /v1/nodes ─────────────────────────────────

    @app.get("/v1/nodes", response_model=List[NodeResponse])
    async def list_nodes(
        node_type: Optional[str] = Query(None),
        source_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        origin_runtime: Optional[List[str]] = Query(None),
        project_key: Optional[str] = Query(None),
    ):
        """List all nodes. Supports filtering by type, date, source.

        Origin Layer (WS0 Leg B): origin_runtime accepts EITHER a repeated
        query param (``?origin_runtime=claude-code&origin_runtime=codex``)
        OR a single comma-separated value (``?origin_runtime=claude-code,
        codex``) — see ``_normalize_origin_runtime_param`` (G5). project_key
        is a single exact match. Both are SQL prefilters via
        store.list_nodes — see its docstring.

        G4/G5 (fail closed): the param being ABSENT (None: no
        origin_runtime in the query string at all) means unfiltered — same
        as before. The param being PRESENT but resolving to an empty list
        (``?origin_runtime=``, or every comma-split segment blank) means a
        filter IS active and matches nothing: this returns ``[]``, it
        never silently falls back to unfiltered.
        """
        nt = None
        if node_type:
            try:
                nt = NodeType(node_type)
            except ValueError:
                raise HTTPException(400, f"Invalid node_type: {node_type}")

        nodes = store.list_nodes(
            node_type=nt,
            source_id=source_id or None,
            limit=limit,
            offset=offset,
            origin_runtime=_normalize_origin_runtime_param(origin_runtime),
            project_key=project_key or None,
        )
        return [_node_to_response(n) for n in nodes]

    # ── GET /v1/nodes/{id} ────────────────────────────

    @app.get("/v1/nodes/{node_id}")
    async def get_node(node_id: str):
        """Get a specific node with its edges and content."""
        result = ops.get_node_with_edges(node_id)
        if result is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return result

    # ── PUT /v1/nodes/{id} ────────────────────────────

    # Skill metadata keys whose transition belongs exclusively to
    # accept/decline/ingest — a plain PUT must never be able to smuggle a
    # skill from proposed to active, or forge origin/curated provenance.
    _SKILL_LOCKED_METADATA_KEYS = ("status", "origin", "curated")

    @app.put("/v1/nodes/{node_id}", response_model=NodeResponse, dependencies=[_mutation_auth])
    async def update_node(node_id: str, request: NodeUpdateRequest):
        """Update a node's label, content, or metadata."""
        kwargs = {}
        if request.label is not None:
            kwargs["label"] = request.label
        if request.content is not None:
            kwargs["content"] = request.content
        if request.metadata is not None:
            kwargs["metadata"] = request.metadata

        if not kwargs:
            raise HTTPException(400, "No fields to update")

        existing = store.get_node(node_id)
        if existing is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        if existing.node_type == NodeType.SKILL and request.metadata is not None:
            before_md = existing.metadata or {}
            for key in _SKILL_LOCKED_METADATA_KEYS:
                if request.metadata.get(key) != before_md.get(key):
                    raise HTTPException(
                        409,
                        f"Cannot change skill metadata field {key!r} via PUT "
                        f"/v1/nodes/{{id}} -- that transition belongs to "
                        f"skills accept/decline/ingest only.",
                    )

        updated = store.update_node(node_id, **kwargs)
        if updated is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return _node_to_response(updated)

    # ── DELETE /v1/nodes/{id} ─────────────────────────

    @app.delete("/v1/nodes/{node_id}", dependencies=[_mutation_auth])
    async def delete_node(node_id: str):
        """The DELETE verb: remove the node and its edges now. Permanent.
        Leaves a content-free delete audit entry and NO suppression record;
        the content can be captured again if its source still holds it. To
        keep it from coming back, use POST /v1/nodes/{id}/forget."""
        deleted = ops.delete_node(node_id, actor="api:delete")
        if not deleted:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {"status": "deleted", "node_id": node_id}

    # ── POST /v1/edges ─────────────────────────────────

    @app.post("/v1/edges", response_model=EdgeResponse, dependencies=[_mutation_auth])
    async def create_edge(request: EdgeCreateRequest):
        """Create an explicit typed edge between two existing nodes.

        Useful for human/agent adjudication edges like ``conflicts_with`` where
        both claims should remain live while the tension becomes first-class.
        """
        try:
            edge_type = EdgeType(request.edge_type)
        except ValueError:
            valid = ", ".join(extensions.all_edge_type_values())
            raise HTTPException(400, f"Invalid edge_type: {request.edge_type}. Valid: {valid}")

        if store.get_node(request.source_node_id) is None:
            raise HTTPException(404, f"Source node not found: {request.source_node_id}")
        if store.get_node(request.target_node_id) is None:
            raise HTTPException(404, f"Target node not found: {request.target_node_id}")

        new_edge = Edge(
            edge_type=edge_type,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            weight=request.weight,
            metadata=request.metadata,
            confidence=request.confidence,
            confidence_set_by=request.confidence_set_by,
            source_context=request.source_context,
        )
        if edge_type is EdgeType.ALIAS_OF:
            # Alias leg: a user-declared alias needs the SAME provenance an
            # inferred one gets (create audit row) — otherwise its later
            # --remove would write an 'invalidate' entry with no 'create'
            # before it, an incoherent trail. Every other edge_type keeps
            # the existing unaudited add_edge path, unchanged.
            edge = store.add_edge_audited(
                new_edge, actor=request.confidence_set_by
            )
        else:
            edge = store.add_edge(new_edge)
        return _edge_to_response(edge)

    # ── POST /v1/consolidate ──────────────────────────

    @app.post("/v1/consolidate", dependencies=[_mutation_auth])
    async def consolidate(request: ConsolidateRequest):
        """Run the dream-mode maintenance pass (B3.1): persist confidence
        decay, refresh clustering, optionally backfill the semantic index,
        and report orphaned nodes. Returns the full consolidation report —
        a dream you can't inspect is a black box, so there is no silent
        variant of this endpoint."""
        from revien.consolidate import Consolidator
        consolidator = Consolidator(
            store, ops, semantic=semantic, clustering=clustering,
        )
        report = consolidator.run(
            decay=request.decay,
            recluster=request.recluster,
            reindex=request.reindex,
            invalidate_orphans=request.invalidate_orphans,
            alias=request.alias,
        )
        return report.to_dict()

    # ── GET /v1/tensions ──────────────────────────────

    @app.get("/v1/tensions")
    async def list_tensions(live_only: bool = Query(True)):
        """The tensions view (B1): every recognized coexisting tension —
        pairs of claims joined by a CONFLICTS_WITH edge, both sides live.
        'What am I in tension with myself about?' as an endpoint.
        live_only=false includes pairs where a side was later invalidated
        (lineage/audit reading)."""
        pairs = store.list_tension_pairs(live_only=live_only)
        return {"count": len(pairs), "tensions": pairs}

    # ── GET /v1/graph ─────────────────────────────────

    @app.get("/v1/graph")
    async def export_graph():
        """Export the full graph as JSON. For inspection/backup."""
        graph = store.export_graph()
        return graph.model_dump(mode="json")

    # ── POST /v1/graph/import ─────────────────────────

    @app.post("/v1/graph/import", dependencies=[_mutation_auth])
    async def import_graph(
        graph_data: Dict[str, Any], mode: str = Query("refuse")
    ):
        """Import a graph from JSON. For restore/migration.

        mode=refuse (default) 409s against a non-empty database — an import
        must never silently eat an existing graph (the old shape hardcoded a
        wipe). mode=merge skips colliding IDs; mode=replace swaps the whole
        graph in one transaction (any failure leaves the original intact).
        """
        from revien.graph.store import ImportRefusedError, ImportValidationError

        if mode not in ("refuse", "merge", "replace"):
            raise HTTPException(400, f"Invalid mode {mode!r}: use refuse, merge or replace")
        try:
            graph = Graph.model_validate(graph_data)
        except Exception as e:
            # Say so when the cause is a type no extension registered: the
            # validator's own list of valid types only names the built-ins.
            hint = extensions.describe_unregistered(graph_data)
            raise HTTPException(
                400, f"Invalid graph data: {str(e)}" + (f"\n{hint}" if hint else ""))
        try:
            result = store.import_graph(graph, mode=mode, semantic=semantic)
        except ImportRefusedError as e:
            raise HTTPException(409, str(e))
        except ImportValidationError as e:
            raise HTTPException(400, f"Invalid graph data: {str(e)}")
        return {
            "status": "imported",
            "mode": mode,
            "nodes": result["nodes_added"],
            "edges": result["edges_added"],
            **result,
        }

    # ── POST /v1/cluster ─────────────────────────────

    @app.post("/v1/cluster", dependencies=[_mutation_auth])
    async def run_clustering():
        """Trigger community detection on the graph. Returns community summary."""
        communities = clustering.run()
        return {
            "status": "clustered",
            "community_count": len(communities),
            "communities": clustering.get_all_communities(),
        }

    # ── GET /v1/communities ──────────────────────────

    @app.get("/v1/communities")
    async def list_communities():
        """List detected communities with centroids and sizes."""
        if not clustering.is_clustered:
            return {
                "status": "unclustered",
                "community_count": 0,
                "communities": [],
            }
        return {
            "status": "clustered",
            "community_count": clustering.community_count,
            "communities": clustering.get_all_communities(),
        }

    # ── GET /v1/health ────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Health check. Returns node count, edge count, uptime."""
        return HealthResponse(
            status="healthy",
            node_count=store.count_nodes(),
            edge_count=store.count_edges(),
            uptime_seconds=round(time.time() - start_time, 2),
            version=__version__,
            features=list(DAEMON_FEATURES),
            forget_key="mismatch" if _forget_key_problem() else "ok",
        )

    # ── POST /v1/sync ─────────────────────────────────

    @app.post("/v1/sync", response_model=SyncResponse, dependencies=[_mutation_auth])
    async def sync():
        """Trigger manual sync with connected AI systems.

        Runs the scheduler's sync_all inline when a scheduler with live
        adapters is attached (the daemon attaches it at startup). Without
        one (bare create_app, e.g. tests/ASGI), reports that honestly
        instead of pretending a sync happened.
        """
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None or not scheduler.list_adapters():
            return SyncResponse(
                status="ok",
                message="No adapters registered — nothing to sync. "
                "Run 'revien connect <system>' then restart the daemon.",
            )
        results = await scheduler.sync_all()
        ingested = sum(
            r.get("items_ingested", 0) for r in results.values() if isinstance(r, dict)
        )
        return SyncResponse(
            status="ok",
            message=f"Synced {len(results)} adapter(s), {ingested} item(s) ingested.",
        )

    # ── POST /v1/mark_used ─────────────────────────────

    class MarkUsedRequest(BaseModel):
        node_id: str
        query: Optional[str] = None

    @app.post("/v1/mark_used", dependencies=[_mutation_auth])
    async def mark_used(request: MarkUsedRequest):
        """
        Mark a retrieved node as actually used.
        Call this when the user references or acts on retrieved information.
        Provides positive training signal AND reinforces connected edge weights.
        """
        node = store.get_node(request.node_id)
        if node is None:
            raise HTTPException(404, f"Node not found: {request.node_id}")
        engine.mark_used(request.node_id, request.query)
        return {"status": "marked", "node_id": request.node_id}

    # ── POST /v1/usage/observe ─────────────────────────

    @app.post("/v1/usage/observe", dependencies=[_mutation_auth])
    async def usage_observe(request: ObserveRequest):
        """Hand back the answer a receipted recall served. Judges each memory
        on the receipt (both signals must agree) and returns how each was
        handled. Shadow records only; REVIEN_USAGE=apply counts uses."""
        from revien.usage import ReceiptError, UsageObserver

        try:
            result = UsageObserver(
                store, engine=engine, semantic=semantic,
            ).observe_answer(request.recall_id, request.answer, request.prompt)
        except ReceiptError as e:
            status = {"unknown": 404, "expired": 410, "already_redeemed": 409,
                      "already_closed": 409, "empty_answer": 400}.get(e.code, 400)
            raise HTTPException(status, {"code": e.code, "detail": str(e)})
        return result.model_dump(mode="json")

    # ── GET /v1/usage/flags ────────────────────────────

    @app.get("/v1/usage/flags", dependencies=[_mutation_auth])
    async def usage_flags(include_dismissed: bool = Query(False)):
        """Recalls whose receipt expired unjudged. Kept until dismissed."""
        store.expire_recall_receipts()
        return {"flags": store.list_receipt_flags(include_dismissed)}

    # ── GET /v1/training/stats ─────────────────────────

    @app.get("/v1/training/stats")
    async def training_stats():
        """Get neural training statistics. Reports neural status even when the
        opt-in `neural` extra (numpy/sklearn) is not installed."""
        return engine.get_training_stats()

    # ── POST /v1/training/run ──────────────────────────

    @app.post("/v1/training/run", dependencies=[_mutation_auth])
    async def run_training():
        """Manually trigger neural training. No-ops (status 'skipped') when the
        neural extra is absent or insufficient signals have accumulated."""
        success = engine.force_train()
        return {
            "status": "success" if success else "skipped",
            "message": "Training completed" if success else "Not enough data or training failed",
            "stats": engine.get_training_stats(),
        }

    # ── POST /v1/nodes/{id}/reinforce ──────────────────

    class ReinforceRequest(BaseModel):
        construct_id: str = ""

    @app.post("/v1/nodes/{node_id}/reinforce", dependencies=[_mutation_auth])
    async def reinforce_node_endpoint(
        node_id: str, request: Optional[ReinforceRequest] = None
    ):
        """Reinforce a node's confidence (+0.05, cap 1.0) after successful use."""
        req = request or ReinforceRequest()
        updated = ops.reinforce_node(node_id, construct_id=req.construct_id)
        if updated is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {"status": "reinforced", "node_id": node_id, "confidence": updated.confidence}

    # ── POST /v1/nodes/{id}/correct ────────────────────

    class CorrectRequest(BaseModel):
        correction_context: str = ""
        construct_id: str = ""

    @app.post("/v1/nodes/{node_id}/correct", dependencies=[_mutation_auth])
    async def correct_node_endpoint(
        node_id: str, request: Optional[CorrectRequest] = None
    ):
        """Mark a node CORRECTED (source_type=CORRECTED, confidence=0.0). Explicit
        removal from ranking — distinct from decay, which only demotes."""
        req = request or CorrectRequest()
        updated = ops.correct_node(
            node_id, correction_context=req.correction_context, construct_id=req.construct_id
        )
        if updated is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {
            "status": "corrected",
            "node_id": node_id,
            "confidence": updated.confidence,
            "source_type": updated.source_type.value,
        }

    # ── POST /v1/nodes/{id}/invalidate ─────────────────
    # Provenance (leg 6a): soft-invalidation. Marks a node stale WITHOUT
    # deleting it. Content is retained; the node drops out of default recall but
    # is still inspectable and recoverable. NOT right-to-forget (that is 6b).

    class InvalidateRequest(BaseModel):
        reason: str = ""
        construct_id: str = ""

    @app.post("/v1/nodes/{node_id}/invalidate", dependencies=[_mutation_auth])
    async def invalidate_node_endpoint(
        node_id: str, request: Optional[InvalidateRequest] = None
    ):
        """Soft-invalidate a node (mark stale, retain content). Non-destructive."""
        req = request or InvalidateRequest()
        updated = ops.invalidate_node(
            node_id, reason=req.reason, construct_id=req.construct_id
        )
        if updated is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {
            "status": "invalidated",
            "node_id": node_id,
            "invalidated_at": (
                updated.invalidated_at.isoformat() if updated.invalidated_at else None
            ),
        }

    # ── GET /v1/nodes/{id}/audit ───────────────────────
    # Provenance (leg 6a): full append-only history for one node, chronological.

    @app.get("/v1/nodes/{node_id}/audit")
    async def node_audit(node_id: str):
        """Full audit history for a node (oldest first)."""
        if store.get_node(node_id) is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {"node_id": node_id, "audit": store.get_node_audit(node_id)}

    # ── GET /v1/audit/recent ───────────────────────────

    @app.get("/v1/audit/recent")
    async def recent_audit(limit: int = Query(50, ge=1, le=1000)):
        """Most recent audit entries across all nodes (newest first)."""
        return {"limit": limit, "audit": store.get_recent_audit(limit=limit)}

    # ── GET /v1/nodes/{id}/lineage ─────────────────────
    # Provenance (leg 6a): derivation trace via DERIVED_FROM edges.

    @app.get("/v1/nodes/{node_id}/lineage")
    async def node_lineage(node_id: str, max_depth: int = Query(10, ge=1, le=100)):
        """Trace a node's source/ancestor chain via DERIVED_FROM edges."""
        if store.get_node(node_id) is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return ops.get_lineage(node_id, max_depth=max_depth)

    # ── Governance Layer (leg 6b): retention / forget / export ─────────
    # "Choose your storage." Nothing auto-deletes by default; only explicit
    # forget, or the opt-in `expire` retention mode, ever deletes.

    # ── POST /v1/retention/sweep ───────────────────────
    class RetentionSweepRequest(BaseModel):
        # Optional per-call overrides; default to the env-resolved policy.
        mode: Optional[str] = None  # keep | archive | expire
        days: Optional[int] = None
        construct_id: str = ""

    @app.post("/v1/retention/sweep", dependencies=[_mutation_auth])
    async def retention_sweep(
        request: Optional[RetentionSweepRequest] = None
    ):
        """Run one retention sweep under the selected storage policy.

        Mode/window resolve from REVIEN_RETENTION / REVIEN_RETENTION_DAYS unless
        overridden in the body. ``keep`` is a no-op; ``archive`` soft-invalidates
        stale unpinned nodes (recoverable); ``expire`` hard-deletes them
        (+tombstone). Pinned nodes are always immune. Returns counts.
        """
        req = request or RetentionSweepRequest()
        return ops.apply_retention(
            mode=req.mode, days=req.days, construct_id=req.construct_id
        )

    # ── GET /v1/nodes/{id}/forget/preview ──────────────
    @app.get("/v1/nodes/{node_id}/forget/preview")
    async def forget_preview(node_id: str, everywhere: bool = Query(False),
                             cascade: Optional[bool] = Query(None)):
        """What a forget WOULD do (nodes removed, cut back, kept; the sources
        the wording is kept out of). Writes nothing."""
        _no_cascade(cascade)
        preview = ops._forget_engine().forget(
            node_id, dry_run=True, everywhere=everywhere)
        if preview.get("status") in ("not_found", "skipped"):
            raise HTTPException(404, f"Node not found: {node_id}")
        preview["exists"] = True
        return preview

    # ── POST /v1/nodes/{id}/forget ─────────────────────
    def _no_cascade(value) -> None:
        if value is not None:
            raise HTTPException(400, detail={
                "status": "refused", "reason": "cascade_removed",
                "message": "cascade was removed: forget names one specific "
                           "thing. To remove a whole conversation or file, "
                           "use delete (DELETE /v1/nodes/{id})."})

    class ForgetRequest(BaseModel):
        cascade: Optional[bool] = None      # removed: any value is a 400
        # Also remove the wording from every other unit that holds it (the
        # default forgets it from this node's source only and names the rest).
        everywhere: bool = False
        # Accepted for older clients and never stored: the verb is recorded
        # by this route, not by free text a caller sends.
        reason: str = ""
        construct_id: str = ""

    @app.post("/v1/nodes/{node_id}/forget", dependencies=[_mutation_auth])
    async def forget_node_endpoint(
        node_id: str, request: Optional[ForgetRequest] = None
    ):
        """The FORGET verb: privacy, for one specific thing (a fact, an
        entity, a decision, a preference). Removes the node, cuts its wording
        out of the stored conversation of the source it came from, scrubs
        every copy, and keeps that wording out of later captures of that
        source. The response says exactly what was removed, cut back and kept,
        what it does and does not stop coming back, and what is left behind
        (docs/FORGET_SPEC.md). A context node is refused: remove a whole
        conversation with delete. A daemon from before the forget gate on the
        same database makes this refuse."""
        from revien.daemon.heartbeat import FORGET_REQUIRES, daemons_lacking

        req = request or ForgetRequest()
        _no_cascade(req.cascade)
        old = daemons_lacking(store, FORGET_REQUIRES, exclude_pid=os.getpid())
        if old:
            raise HTTPException(409, detail={
                "status": "refused", "reason": "restart_required",
                "message": (
                    f"A daemon (pid {old[0]['pid']}) is serving this database "
                    "from before forget protection. It would capture the "
                    "forgotten source again. Restart required: stop it, "
                    "start it again with 'revien start', then forget again. "
                    "Nothing was changed.")})
        result = ops.forget_node(node_id, actor="api:forget",
                                 everywhere=req.everywhere)
        status = result.get("status")
        if status == "not_found":
            raise HTTPException(404, f"Node not found: {node_id}")
        if status == "refused":
            raise HTTPException(409, detail=result)
        if status == "busy":
            raise HTTPException(503, detail=result)
        return result

    # ── POST /v1/forget/wording ────────────────────────
    # FastAPI's own 422 quotes the input it could not parse. For this route
    # the input IS the secret, so the answer names the fields and the kinds of
    # problem and nothing the caller sent.
    from fastapi.exceptions import RequestValidationError
    from fastapi.exception_handlers import request_validation_exception_handler

    @app.exception_handler(RequestValidationError)
    async def _validation_without_the_secret(request, exc):
        if request.url.path.rstrip("/") != "/v1/forget/wording":
            return await request_validation_exception_handler(request, exc)
        problems = [{"field": ".".join(str(p) for p in e.get("loc", ())[1:]) or "body",
                     "problem": e.get("type", "invalid")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "status": "refused", "reason": "invalid_request",
            "problems": problems,
            "message": ("Send a JSON object {\"wording\": \"<the string>\"} "
                        "(optional \"preview\": true). Nothing was changed, "
                        "and nothing you sent is repeated here.")})

    class ForgetWordingRequest(BaseModel):
        wording: str
        # Only count what the string would cut; change nothing.
        preview: bool = False
        # Cut it out of longer words that contain it too ('secret' inside
        # 'secretary'). Refused otherwise: the preview counts them.
        allow_inside_words: bool = False

    @app.post("/v1/forget/wording", dependencies=[_mutation_auth])
    async def forget_wording_endpoint(request: ForgetWordingRequest):
        """Forget a literal string (a key, a token, a password) everywhere:
        every node, the audit trail, recall records, training data and
        snapshots, and keep the same characters out of every later capture.
        The string is never stored, logged or returned (docs/FORGET_SPEC.md
        §2a)."""
        from revien.daemon.heartbeat import FORGET_REQUIRES, daemons_lacking

        old = daemons_lacking(store, FORGET_REQUIRES, exclude_pid=os.getpid())
        if old:
            raise HTTPException(409, detail={
                "status": "refused", "reason": "restart_required",
                "message": (
                    f"A daemon (pid {old[0]['pid']}) is serving this database "
                    "from before forget protection. It would capture the "
                    "string again. Restart required: stop it, start it "
                    "again with 'revien start', then forget again. Nothing "
                    "was changed.")})
        result = ops.forget_wording(request.wording, actor="api:forget",
                                    dry_run=request.preview,
                                    allow_inside_words=request.allow_inside_words)
        status = result.get("status")
        if status == "refused":
            raise HTTPException(400, detail=result)
        if status == "busy":
            raise HTTPException(503, detail=result)
        return result

    # ── GET /v1/export ─────────────────────────────────
    @app.get("/v1/export")
    async def export_everything():
        """Export the FULL graph (nodes + edges) + audit log as JSON.

        "Your data, portable." Superset of /v1/graph — adds the audit trail.
        """
        return ops.export_everything()

    # ── POST /v1/reindex ───────────────────────────────
    # Backfill embeddings for existing nodes (opt-in semantic layer). When the
    # `semantic` extra is absent or REVIEN_SEMANTIC=0, this reports status
    # "disabled" and does nothing — graph retrieval is unaffected either way.
    @app.post("/v1/reindex", dependencies=[_mutation_auth])
    async def reindex():
        """Embed all existing content nodes into the semantic vector index.

        No-op (status 'disabled') when the opt-in `semantic` extra is not
        installed or REVIEN_SEMANTIC=0.
        """
        return semantic.reindex_all()

    # ── GET /v1/semantic/status ────────────────────────
    @app.get("/v1/semantic/status")
    async def semantic_status():
        """Report whether the opt-in semantic/vector layer is active and why."""
        return semantic.status()

    # ── Skills (leg D2: proposals) ─────────────────────
    # GETs are open on loopback like every other read route; the two POSTs
    # (accept/decline) are mutations and gated by require_mutation_auth —
    # identical rule to /v1/ingest's check_capture_auth, mirrored under its
    # own name (Leg C).

    @app.get("/v1/skills")
    async def list_skills_endpoint(
        status: Optional[str] = Query(None),
        origin: Optional[str] = Query(None),
        project_key: Optional[str] = Query(None),
    ):
        """List SKILL nodes (ingested + proposed), user-before-engine
        ordered. `status`/`origin` filter on metadata; `project_key` is a
        SQL prefilter (store.list_nodes)."""
        nodes = store.list_nodes(
            node_type=NodeType.SKILL, limit=1000, project_key=project_key or None,
        )
        if status is not None:
            nodes = [n for n in nodes if (n.metadata or {}).get("status") == status]
        if origin is not None:
            nodes = [n for n in nodes if (n.metadata or {}).get("origin") == origin]
        nodes = sort_user_before_engine(nodes)
        return [_node_to_response(n) for n in nodes]

    @app.get("/v1/skills/{node_id}")
    async def get_skill_endpoint(node_id: str):
        """Get one SKILL node (ingested or proposed) by id."""
        node = store.get_node(node_id)
        if node is None or node.node_type != NodeType.SKILL:
            raise HTTPException(404, f"Skill not found: {node_id}")
        return _node_to_response(node)

    @app.post("/v1/skills/{node_id}/accept", dependencies=[_mutation_auth])
    async def accept_skill_endpoint(node_id: str):
        """Accept a proposal: status -> active, origin stays engine.

        404 when the node doesn't exist at all; 409 when it exists but
        can't be accepted right now (not a SKILL, invalidated, or status
        isn't "proposed") — a real conflict, not a missing resource.
        """
        if store.get_node(node_id) is None:
            raise HTTPException(404, f"Skill not found: {node_id}")
        try:
            updated = accept_proposal(store, node_id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return _node_to_response(updated)

    @app.post("/v1/skills/{node_id}/decline", dependencies=[_mutation_auth])
    async def decline_skill_endpoint(node_id: str):
        """Decline a proposal: declines += 1; the third decline soft-
        invalidates it (GraphOperations.invalidate_node).

        404 when the node doesn't exist at all; 409 when it exists but
        can't be declined right now (not a SKILL, invalidated, or status
        isn't "proposed").
        """
        if store.get_node(node_id) is None:
            raise HTTPException(404, f"Skill not found: {node_id}")
        try:
            updated = decline_proposal(store, node_id)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return _node_to_response(updated)

    return app


def _node_to_response(node: Node) -> NodeResponse:
    return NodeResponse(
        node_id=node.node_id,
        node_type=node.node_type.value,
        label=node.label,
        content=node.content,
        source_id=node.source_id,
        created_at=node.created_at.isoformat(),
        last_accessed=node.last_accessed.isoformat(),
        access_count=node.access_count,
        metadata=node.metadata,
        valid_from=node.valid_from.isoformat() if node.valid_from else None,
        valid_until=node.valid_until.isoformat() if node.valid_until else None,
        origin_runtime=node.origin_runtime,
        origin_source=node.origin_source,
        project_key=node.project_key,
        session_key=node.session_key,
    )


def _edge_to_response(edge: Edge) -> EdgeResponse:
    return EdgeResponse(
        edge_id=edge.edge_id,
        edge_type=edge.edge_type.value,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        weight=edge.weight,
        created_at=edge.created_at.isoformat(),
        metadata=edge.metadata,
        confidence=edge.confidence,
        confidence_set_by=edge.confidence_set_by,
        source_context=edge.source_context,
    )
