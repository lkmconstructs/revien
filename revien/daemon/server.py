"""
Revien REST API Server — FastAPI application exposing all Revien endpoints.
Runs as local daemon on port 7437 or as hosted service.
"""

import time
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import secrets

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.graph.clustering import CommunityDetector
from revien.ingestion.pipeline import IngestionInput, IngestionOutput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine, RetrievalResponse
from revien.semantic.index import SemanticIndex


# ── Request/Response Models ───────────────────────────────

class IngestRequest(BaseModel):
    source_id: str
    content: str
    content_type: str = "conversation"
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Capture path: persist now, embed later — the response returns before any
    # embedding-model load. Queued nodes are keyword-recallable immediately and
    # semantic-recallable once the queue drains (idle sweep or next recall).
    defer_embed: bool = False


class IngestResponse(BaseModel):
    context_node_id: str
    nodes_created: int
    nodes_deduplicated: int
    edges_created: int
    total_nodes_in_graph: int
    total_edges_in_graph: int


class RecallRequest(BaseModel):
    query: str
    top_n: int = 5
    min_score: float = 0.01
    include_invalidated: bool = False
    include_context: bool = False
    include_tensions: bool = False
    # Bi-temporal query time (B2), ISO-8601: "what was true AT this time?"
    as_of: Optional[str] = None


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


class HealthResponse(BaseModel):
    status: str
    node_count: int
    edge_count: int
    uptime_seconds: float
    version: str


class SyncResponse(BaseModel):
    status: str
    message: str


# ── Capture auth (P3: remote capture is opt-in, token-gated) ─────────

# starlette's TestClient reports host "testclient"; it exercises the same
# in-process path as a local adapter, so it counts as loopback.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient", ""}


def check_capture_auth(client_host: Optional[str], auth_header: str) -> None:
    """Gate for /v1/ingest. Raises HTTPException on refusal.

    Loopback callers are never gated — the local adapter path is unchanged
    whether or not a token is configured. A remote caller is refused outright
    unless ``REVIEN_CAPTURE_TOKEN`` is set (remote capture is opt-in), and
    with it set must present ``Authorization: Bearer <token>``. Comparison is
    constant-time.
    """
    host = (client_host or "").strip().lower()
    if host in _LOOPBACK_HOSTS:
        return
    token = os.environ.get("REVIEN_CAPTURE_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            403,
            "Remote capture is disabled. Set REVIEN_CAPTURE_TOKEN on the "
            "daemon and send 'Authorization: Bearer <token>' to enable it.",
        )
    expected = f"Bearer {token}"
    if not secrets.compare_digest(
        (auth_header or "").strip().encode(), expected.encode()
    ):
        raise HTTPException(401, "Invalid or missing capture token.")


# ── App Factory ───────────────────────────────────────────

def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Database path resolution is explicit argument > ``REVIEN_DB_PATH`` env var >
    ``revien.db``. The CLI passes an explicit path, but honoring the env var
    keeps direct ASGI/Docker usage from silently serving a fresh local database.
    """
    db_path = db_path or os.environ.get("REVIEN_DB_PATH", "revien.db")

    app = FastAPI(
        title="Revien",
        description="Graph-based memory engine for AI systems. Memory that returns.",
        version="0.1.0",
    )

    # Shared state
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

    # Store references on app for access by daemon/scheduler
    app.state.store = store
    app.state.ops = ops
    app.state.pipeline = pipeline
    app.state.engine = engine
    app.state.clustering = clustering
    app.state.semantic = semantic

    # ── POST /v1/ingest ───────────────────────────────

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest, http_request: Request):
        """Ingest raw content. Builds graph nodes and edges."""
        check_capture_auth(
            http_request.client.host if http_request.client else "",
            http_request.headers.get("authorization", ""),
        )
        ts = None
        if request.timestamp:
            try:
                ts = datetime.fromisoformat(request.timestamp)
            except ValueError:
                pass

        input_data = IngestionInput(
            source_id=request.source_id,
            content=request.content,
            content_type=request.content_type,
            timestamp=ts,
            metadata=request.metadata,
            defer_embed=request.defer_embed,
        )
        result = pipeline.ingest(input_data)

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
        )

    # ── POST /v1/recall ───────────────────────────────

    @app.post("/v1/recall")
    async def recall(request: RecallRequest):
        """Query memory. Returns ranked nodes by three-factor score."""
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
        )
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
        }

    # ── GET /v1/nodes ─────────────────────────────────

    @app.get("/v1/nodes", response_model=List[NodeResponse])
    async def list_nodes(
        node_type: Optional[str] = Query(None),
        source_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """List all nodes. Supports filtering by type, date, source."""
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

    @app.put("/v1/nodes/{node_id}", response_model=NodeResponse)
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

        updated = store.update_node(node_id, **kwargs)
        if updated is None:
            raise HTTPException(404, f"Node not found: {node_id}")
        return _node_to_response(updated)

    # ── DELETE /v1/nodes/{id} ─────────────────────────

    @app.delete("/v1/nodes/{node_id}")
    async def delete_node(node_id: str):
        """Delete a node and its edges. Permanent."""
        deleted = store.delete_node(node_id)
        if not deleted:
            raise HTTPException(404, f"Node not found: {node_id}")
        return {"status": "deleted", "node_id": node_id}

    # ── POST /v1/edges ─────────────────────────────────

    @app.post("/v1/edges", response_model=EdgeResponse)
    async def create_edge(request: EdgeCreateRequest):
        """Create an explicit typed edge between two existing nodes.

        Useful for human/agent adjudication edges like ``conflicts_with`` where
        both claims should remain live while the tension becomes first-class.
        """
        try:
            edge_type = EdgeType(request.edge_type)
        except ValueError:
            valid = ", ".join(e.value for e in EdgeType)
            raise HTTPException(400, f"Invalid edge_type: {request.edge_type}. Valid: {valid}")

        if store.get_node(request.source_node_id) is None:
            raise HTTPException(404, f"Source node not found: {request.source_node_id}")
        if store.get_node(request.target_node_id) is None:
            raise HTTPException(404, f"Target node not found: {request.target_node_id}")

        edge = store.add_edge(Edge(
            edge_type=edge_type,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            weight=request.weight,
            metadata=request.metadata,
            confidence=request.confidence,
            confidence_set_by=request.confidence_set_by,
            source_context=request.source_context,
        ))
        return _edge_to_response(edge)

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

    @app.post("/v1/graph/import")
    async def import_graph(graph_data: Dict[str, Any]):
        """Import a graph from JSON. For restore/migration."""
        try:
            graph = Graph.model_validate(graph_data)
            store.import_graph(graph, clear_existing=True)
            return {
                "status": "imported",
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
            }
        except Exception as e:
            raise HTTPException(400, f"Invalid graph data: {str(e)}")

    # ── POST /v1/cluster ─────────────────────────────

    @app.post("/v1/cluster")
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
            version="0.1.0",
        )

    # ── POST /v1/sync ─────────────────────────────────

    @app.post("/v1/sync", response_model=SyncResponse)
    async def sync():
        """Trigger manual sync with connected AI systems."""
        # In MVP, sync is a no-op until adapters are connected
        # Phase 4 will wire this to the scheduler
        return SyncResponse(
            status="ok",
            message="Manual sync triggered. Adapters will process on next cycle.",
        )

    # ── POST /v1/mark_used ─────────────────────────────

    class MarkUsedRequest(BaseModel):
        node_id: str
        query: Optional[str] = None

    @app.post("/v1/mark_used")
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

    # ── GET /v1/training/stats ─────────────────────────

    @app.get("/v1/training/stats")
    async def training_stats():
        """Get neural training statistics. Reports neural status even when the
        opt-in `neural` extra (numpy/sklearn) is not installed."""
        return engine.get_training_stats()

    # ── POST /v1/training/run ──────────────────────────

    @app.post("/v1/training/run")
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

    @app.post("/v1/nodes/{node_id}/reinforce")
    async def reinforce_node_endpoint(node_id: str, request: Optional[ReinforceRequest] = None):
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

    @app.post("/v1/nodes/{node_id}/correct")
    async def correct_node_endpoint(node_id: str, request: Optional[CorrectRequest] = None):
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

    @app.post("/v1/nodes/{node_id}/invalidate")
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

    @app.post("/v1/retention/sweep")
    async def retention_sweep(request: Optional[RetentionSweepRequest] = None):
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
    async def forget_preview(node_id: str):
        """Show what a cascade forget WOULD remove (count + ids) — never blind."""
        preview = ops.forget_preview(node_id)
        if not preview.get("exists", False):
            raise HTTPException(404, f"Node not found: {node_id}")
        return preview

    # ── POST /v1/nodes/{id}/forget ─────────────────────
    class ForgetRequest(BaseModel):
        cascade: bool = False
        reason: str = ""
        construct_id: str = ""

    @app.post("/v1/nodes/{node_id}/forget")
    async def forget_node_endpoint(node_id: str, request: Optional[ForgetRequest] = None):
        """Right-to-forget: HARD-delete the node's content (privacy).

        Distinct from invalidate (which retains). Writes a content-free tombstone
        audit entry and re-points children's lineage to a tombstone marker.
        cascade=True forgets the whole DERIVED_FROM subtree.
        """
        req = request or ForgetRequest()
        result = ops.forget_node(
            node_id, cascade=req.cascade, reason=req.reason,
            construct_id=req.construct_id,
        )
        if result.get("status") == "not_found":
            raise HTTPException(404, f"Node not found: {node_id}")
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
    @app.post("/v1/reindex")
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
