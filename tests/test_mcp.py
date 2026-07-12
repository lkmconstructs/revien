"""
MCP surface (LEG P5): stdio-capable FastMCP server + optional daemon mount.

Contract under test:
  - The tools are a thin face over the SAME engine/pipeline surface as
    /v1/recall and /v1/ingest — store-then-recall round-trips in-process,
    and the recall payload mirrors the REST serialization exactly.
  - defer_embed rides P3's queue: store queues instead of embedding inline.
  - Bad as_of fails loudly (ToolError), not with a stack trace to the client.
  - REVIEN_MCP_HTTP unset ⇒ NO /mcp route: shipped REST surface unchanged.
  - REVIEN_MCP_HTTP set + SDK present ⇒ /mcp answers MCP traffic, with
    capture-auth semantics re-applied at the mount boundary (loopback
    exempt, remote token-gated — same gate as /v1/ingest, whole mount).

Whole module skips when the mcp extra is not installed (peer-dependency
pattern — core installs must still pass the rest of the suite).
"""

import asyncio
import json
import os
import tempfile

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed (pip install revien[mcp])")

from fastapi.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

from revien.daemon.server import _CaptureAuthASGI, create_app
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionPipeline
from revien.mcp_server import MCP_AVAILABLE, build_mcp_server
from revien.retrieval.engine import RetrievalEngine
from tests.test_capture import _QueueTestIndex


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


@pytest.fixture
def store(db_path):
    s = GraphStore(db_path=db_path)
    yield s
    s.close()


def _call(server, name, arguments):
    """Drive a tool through the FastMCP tool manager (schema validation and
    error wrapping included) and decode the JSON content block back to a dict.

    Dict-annotated tools return (content_blocks, structured) — the content
    block's JSON text is the tool's dict verbatim; structured wraps it in
    {"result": ...}. Parse the content block: that is what a text-only MCP
    client sees."""
    res = asyncio.run(server.call_tool(name, arguments))
    content = res[0] if isinstance(res, tuple) else res
    return json.loads(content[0].text)


STORE_TEXT = (
    "Decision: migrated the billing service to postgres for concurrent writes."
)


# ── Tool round-trip (in-process, graph path — conftest pins semantic off) ──


class TestToolRoundTrip:
    def test_store_then_recall(self, db_path):
        # db_path mode: the factory owns its stack — the `revien mcp` wiring.
        server = build_mcp_server(db_path=db_path)

        stored = _call(server, "revien_store", {"content": STORE_TEXT})
        assert stored["context_node_id"]
        assert stored["total_nodes_in_graph"] >= 1
        # IngestionOutput fields, all of them — the MCP face mirrors /v1/ingest.
        for key in (
            "context_node_id", "nodes_created", "nodes_deduplicated",
            "edges_created", "total_nodes_in_graph", "total_edges_in_graph",
        ):
            assert key in stored

        recalled = _call(
            server,
            "revien_recall",
            {"query": "postgres billing", "include_context": True},
        )
        assert recalled["results"], "stored memory must be recallable"
        assert any("postgres" in r["content"].lower() for r in recalled["results"])
        # Response shape mirrors POST /v1/recall (include_tensions off).
        assert set(recalled.keys()) == {
            "query", "results", "nodes_examined", "retrieval_time_ms",
            "semantic_active", "semantic_note",
        }
        assert set(recalled["results"][0].keys()) == {
            "node_id", "node_type", "label", "content", "score",
            "score_breakdown", "path",
        }
        assert recalled["semantic_active"] is False  # conftest pins graph-only

    def test_bad_as_of_errors_cleanly(self, db_path):
        server = build_mcp_server(db_path=db_path)
        with pytest.raises(ToolError) as e:
            _call(server, "revien_recall", {"query": "x", "as_of": "not-a-time"})
        assert "Invalid as_of" in str(e.value)

    def test_valid_as_of_accepted(self, db_path):
        server = build_mcp_server(db_path=db_path)
        out = _call(
            server,
            "revien_recall",
            {"query": "anything", "as_of": "2026-01-01T00:00:00+00:00"},
        )
        assert "results" in out


# ── defer_embed rides the P3 pending queue ─────────────────


class TestDeferEmbed:
    def test_defer_embed_queues_instead_of_embedding(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        engine = RetrievalEngine(store, semantic=sem)
        server = build_mcp_server(engine=engine, pipeline=pipeline)

        out = _call(
            server,
            "revien_store",
            {"content": STORE_TEXT, "source_id": "phone", "defer_embed": True},
        )
        assert out["context_node_id"]
        assert sem._vectors == {}, "defer_embed must not embed inline"
        assert sem.pending_count() >= 1

    def test_default_store_embeds_inline(self, store):
        sem = _QueueTestIndex(store)
        pipeline = IngestionPipeline(store, semantic=sem)
        engine = RetrievalEngine(store, semantic=sem)
        server = build_mcp_server(engine=engine, pipeline=pipeline)

        _call(server, "revien_store", {"content": STORE_TEXT})
        assert len(sem._vectors) >= 1
        assert sem.pending_count() == 0


# ── Daemon mount gate (REVIEN_MCP_HTTP) ────────────────────


def _route_paths(app):
    return [getattr(r, "path", "") for r in app.routes]


class TestHttpMountGate:
    def test_no_mcp_route_when_unset(self, monkeypatch, db_path):
        monkeypatch.delenv("REVIEN_MCP_HTTP", raising=False)
        app = create_app(db_path=db_path)
        assert not any(p.startswith("/mcp") for p in _route_paths(app)), (
            "REVIEN_MCP_HTTP unset must leave the REST surface without /mcp"
        )
        with TestClient(app) as client:
            assert client.post("/mcp", json={}).status_code == 404

    def test_mcp_mounted_and_responds_when_set(self, monkeypatch, db_path):
        assert MCP_AVAILABLE
        monkeypatch.setenv("REVIEN_MCP_HTTP", "1")
        app = create_app(db_path=db_path)
        assert any(p == "/mcp" for p in _route_paths(app))
        with TestClient(app) as client:  # loopback ⇒ auth-exempt
            resp = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "0"},
                    },
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["result"]["serverInfo"]["name"] == "revien"

    def test_rest_surface_identical_when_unset(self, monkeypatch, db_path):
        """The gate must not perturb anything else: route set with the gate
        unset matches the route set with it explicitly falsy."""
        monkeypatch.delenv("REVIEN_MCP_HTTP", raising=False)
        baseline = _route_paths(create_app(db_path=db_path))
        monkeypatch.setenv("REVIEN_MCP_HTTP", "0")
        assert _route_paths(create_app(db_path=db_path)) == baseline


# ── Capture auth at the mount boundary ─────────────────────


class _Recorder:
    """Minimal ASGI app recording whether the request got through the gate."""

    def __init__(self):
        self.reached = False

    async def __call__(self, scope, receive, send):
        self.reached = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _drive(middleware, client_host, headers=()):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "client": (client_host, 4242) if client_host is not None else None,
        "headers": list(headers),
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


class TestMountAuth:
    def test_loopback_passes_without_token(self, monkeypatch):
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        inner = _Recorder()
        _drive(_CaptureAuthASGI(inner), "127.0.0.1")
        assert inner.reached

    def test_remote_refused_403_when_no_token_configured(self, monkeypatch):
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        inner = _Recorder()
        sent = _drive(_CaptureAuthASGI(inner), "100.64.0.7")
        assert not inner.reached
        assert sent[0]["status"] == 403

    def test_remote_bad_bearer_401(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        inner = _Recorder()
        sent = _drive(
            _CaptureAuthASGI(inner),
            "100.64.0.7",
            headers=[(b"authorization", b"Bearer wrong")],
        )
        assert not inner.reached
        assert sent[0]["status"] == 401

    def test_remote_correct_bearer_passes(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        inner = _Recorder()
        _drive(
            _CaptureAuthASGI(inner),
            "100.64.0.7",
            headers=[(b"authorization", b"Bearer s3cret")],
        )
        assert inner.reached

    # ── Browser-origin gate: the loopback exemption must never be the last
    # line against browser JS. A DNS-rebinding/CSRF page runs AS loopback —
    # the Origin header is what distinguishes a browser from a local process.

    def test_loopback_browser_origin_refused(self, monkeypatch):
        """The rebinding scenario: local browser, no token configured —
        without the origin gate this request would read the whole graph."""
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        monkeypatch.delenv("REVIEN_MCP_ALLOWED_ORIGINS", raising=False)
        inner = _Recorder()
        sent = _drive(
            _CaptureAuthASGI(inner),
            "127.0.0.1",
            headers=[(b"origin", b"https://evil.example")],
        )
        assert not inner.reached
        assert sent[0]["status"] == 403

    def test_allowlisted_origin_passes(self, monkeypatch):
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        monkeypatch.setenv(
            "REVIEN_MCP_ALLOWED_ORIGINS", "https://app.example, https://other.example"
        )
        inner = _Recorder()
        _drive(
            _CaptureAuthASGI(inner),
            "127.0.0.1",
            headers=[(b"origin", b"https://app.example")],
        )
        assert inner.reached

    def test_remote_bearer_with_unlisted_origin_still_refused(self, monkeypatch):
        """Origin gate applies regardless of host — a stolen bearer used from
        a browser page is still refused."""
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        monkeypatch.delenv("REVIEN_MCP_ALLOWED_ORIGINS", raising=False)
        inner = _Recorder()
        sent = _drive(
            _CaptureAuthASGI(inner),
            "100.64.0.7",
            headers=[
                (b"authorization", b"Bearer s3cret"),
                (b"origin", b"https://evil.example"),
            ],
        )
        assert not inner.reached
        assert sent[0]["status"] == 403

    def test_native_client_without_origin_unaffected(self, monkeypatch):
        """Native MCP clients send no Origin — the gate must not touch them."""
        monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
        monkeypatch.delenv("REVIEN_MCP_ALLOWED_ORIGINS", raising=False)
        inner = _Recorder()
        _drive(_CaptureAuthASGI(inner), "127.0.0.1")
        assert inner.reached
