"""
Revien REST API Server — FastAPI application exposing all Revien endpoints.
Runs as local daemon on port 7437 or as hosted service.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.ingestion.pipeline import IngestionInput, IngestionOutput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine, RetrievalResponse


# ── Request/Response Models ───────────────────────────────

class IngestRequest(BaseModel):
    source_id: str
    content: str
    content_type: str = "conversation"
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class NodeUpdateRequest(BaseModel):
    label: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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


class HealthResponse(BaseModel):
    status: str
    node_count: int
    edge_count: int
    uptime_seconds: float
    version: str


class SyncResponse(BaseModel):
    status: str
    message: str


# ── App Factory ───────────────────────────────────────────

def create_app(db_path: str = "revien.db") -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Revien",
        description="Graph-based memory engine for AI systems. Memory that returns.",
        version="0.1.0",
    )

    # Shared state
    store = GraphStore(db_path=db_path)
    ops = GraphOperations(store)
    pipeline = IngestionPipeline(store)
    engine = RetrievalEngine(store)
    start_time = time.time()

    # Store references on app for access by daemon/scheduler
    app.state.store = store
    app.state.ops = ops
    app.state.pipeline = pipeline
    app.state.engine = engine

    # ── POST /v1/ingest ───────────────────────────────

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        """Ingest raw content. Builds graph nodes and edges."""
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
        )
        result = pipeline.ingest(input_data)
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
        response = engine.recall(
            query=request.query,
            top_n=request.top_n,
            min_score=request.min_score,
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
                }
                for r in response.results
            ],
            "nodes_examined": response.nodes_examined,
            "retrieval_time_ms": response.retrieval_time_ms,
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
    )
