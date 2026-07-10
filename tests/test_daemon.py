"""
Test 4: Daemon and API
Start the daemon (via TestClient). Call all endpoints.
Verify ingest + recall round-trip works through the API.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from revien.daemon.server import create_app


SAMPLE_CONVERSATION = """User: We need to decide on the pricing for the enterprise tier.
Assistant: Based on our analysis, I recommend $499/month with a 20% annual discount.
User: That works. Let's go with that. Also, make sure the deployment uses PostgreSQL, not MySQL. We decided that last week.
Assistant: Confirmed. Enterprise tier at $499/month, 20% annual discount, PostgreSQL for the database layer. I'll update the architecture doc."""


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        yield c
    os.unlink(path)


@pytest.fixture
def seeded_client():
    """Client with sample conversation already ingested."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        c.post("/v1/ingest", json={
            "source_id": "test-session-1",
            "content": SAMPLE_CONVERSATION,
        })
        yield c
    os.unlink(path)


# ── Health Check ──────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/v1/health").json()
        assert data["status"] == "healthy"
        assert "node_count" in data
        assert "edge_count" in data
        assert "uptime_seconds" in data
        assert data["version"] == "0.1.0"

    def test_health_counts_increase_after_ingest(self, client):
        before = client.get("/v1/health").json()
        client.post("/v1/ingest", json={
            "source_id": "test",
            "content": SAMPLE_CONVERSATION,
        })
        after = client.get("/v1/health").json()
        assert after["node_count"] > before["node_count"]
        assert after["edge_count"] > before["edge_count"]

    def test_create_app_honors_revien_db_path_env_when_no_explicit_arg(self, monkeypatch):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            monkeypatch.setenv("REVIEN_DB_PATH", path)
            app = create_app()
            assert app.state.store.db_path == path
        finally:
            os.unlink(path)

    def test_create_app_explicit_db_path_overrides_env(self, monkeypatch):
        fd_env, env_path = tempfile.mkstemp(suffix=".db")
        fd_explicit, explicit_path = tempfile.mkstemp(suffix=".db")
        os.close(fd_env)
        os.close(fd_explicit)
        try:
            monkeypatch.setenv("REVIEN_DB_PATH", env_path)
            app = create_app(db_path=explicit_path)
            assert app.state.store.db_path == explicit_path
        finally:
            os.unlink(env_path)
            os.unlink(explicit_path)


# ── Ingest Endpoint ───────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_returns_200(self, client):
        resp = client.post("/v1/ingest", json={
            "source_id": "test-1",
            "content": "We decided to use Python for the backend.",
        })
        assert resp.status_code == 200

    def test_ingest_returns_counts(self, client):
        data = client.post("/v1/ingest", json={
            "source_id": "test-1",
            "content": SAMPLE_CONVERSATION,
        }).json()
        assert data["nodes_created"] > 0
        assert data["edges_created"] > 0
        assert data["context_node_id"] is not None

    def test_ingest_with_metadata(self, client):
        resp = client.post("/v1/ingest", json={
            "source_id": "test-1",
            "content": "Test content",
            "content_type": "note",
            "metadata": {"tool": "claude-code", "project": "revien"},
        })
        assert resp.status_code == 200


# ── Recall Endpoint ───────────────────────────────────────

class TestRecallEndpoint:
    def test_recall_returns_200(self, seeded_client):
        resp = seeded_client.post("/v1/recall", json={
            "query": "What did we decide about pricing?",
        })
        assert resp.status_code == 200

    def test_recall_returns_results(self, seeded_client):
        data = seeded_client.post("/v1/recall", json={
            "query": "pricing decision",
        }).json()
        assert "results" in data
        assert len(data["results"]) > 0
        assert "nodes_examined" in data
        assert "retrieval_time_ms" in data

    def test_recall_results_have_scores(self, seeded_client):
        data = seeded_client.post("/v1/recall", json={
            "query": "enterprise pricing",
        }).json()
        if data["results"]:
            result = data["results"][0]
            assert "score" in result
            assert "score_breakdown" in result
            assert "recency" in result["score_breakdown"]
            assert "frequency" in result["score_breakdown"]
            assert "proximity" in result["score_breakdown"]

    def test_ingest_recall_roundtrip(self, client):
        """The critical test: ingest content, then recall it via API."""
        # Ingest
        client.post("/v1/ingest", json={
            "source_id": "roundtrip-test",
            "content": SAMPLE_CONVERSATION,
        })
        # Recall
        data = client.post("/v1/recall", json={
            "query": "What did we decide about pricing?",
        }).json()
        assert len(data["results"]) > 0
        all_content = " ".join(
            r["content"].lower() + " " + r["label"].lower()
            for r in data["results"]
        )
        assert "499" in all_content or "pricing" in all_content or "enterprise" in all_content

    def test_recall_can_include_context_nodes_when_requested(self, monkeypatch):
        """API must expose engine.include_context so raw context can surface."""
        monkeypatch.setenv("REVIEN_SEMANTIC", "1")
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            app = create_app(db_path=path)
            with TestClient(app) as c:
                c.post("/v1/ingest", json={
                    "source_id": "context-recall-test",
                    "content": "Atlas is the planning hub. Ledger is the private archive.",
                    "content_type": "note",
                })

                without_context = c.post("/v1/recall", json={
                    "query": "Atlas planning hub Ledger private archive",
                    "top_n": 10,
                }).json()
                assert all(r["node_type"] != "context" for r in without_context["results"])

                with_context = c.post("/v1/recall", json={
                    "query": "Atlas planning hub Ledger private archive",
                    "top_n": 10,
                    "include_context": True,
                }).json()
                assert any(
                    r["node_type"] == "context"
                    and "atlas is the planning hub" in r["content"].lower()
                    for r in with_context["results"]
                )
        finally:
            os.unlink(path)


# ── Nodes Endpoints ───────────────────────────────────────

class TestNodesEndpoints:
    def test_list_nodes_empty(self, client):
        data = client.get("/v1/nodes").json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_nodes_after_ingest(self, seeded_client):
        data = seeded_client.get("/v1/nodes").json()
        assert len(data) > 0

    def test_list_nodes_filter_by_type(self, seeded_client):
        data = seeded_client.get("/v1/nodes?node_type=entity").json()
        for node in data:
            assert node["node_type"] == "entity"

    def test_list_nodes_invalid_type(self, seeded_client):
        resp = seeded_client.get("/v1/nodes?node_type=invalid")
        assert resp.status_code == 400

    def test_get_specific_node(self, seeded_client):
        nodes = seeded_client.get("/v1/nodes").json()
        if nodes:
            node_id = nodes[0]["node_id"]
            data = seeded_client.get(f"/v1/nodes/{node_id}").json()
            assert "node" in data
            assert "edges" in data
            assert "connected_nodes" in data

    def test_get_nonexistent_node(self, client):
        resp = client.get("/v1/nodes/fake-id-123")
        assert resp.status_code == 404

    def test_update_node(self, seeded_client):
        nodes = seeded_client.get("/v1/nodes").json()
        if nodes:
            node_id = nodes[0]["node_id"]
            resp = seeded_client.put(f"/v1/nodes/{node_id}", json={
                "label": "Updated Label",
            })
            assert resp.status_code == 200
            assert resp.json()["label"] == "Updated Label"

    def test_update_nonexistent_node(self, client):
        resp = client.put("/v1/nodes/fake-id", json={"label": "Nope"})
        assert resp.status_code == 404

    def test_delete_node(self, seeded_client):
        nodes = seeded_client.get("/v1/nodes").json()
        if nodes:
            node_id = nodes[0]["node_id"]
            resp = seeded_client.delete(f"/v1/nodes/{node_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"
            # Verify gone
            resp2 = seeded_client.get(f"/v1/nodes/{node_id}")
            assert resp2.status_code == 404

    def test_delete_nonexistent_node(self, client):
        resp = client.delete("/v1/nodes/fake-id")
        assert resp.status_code == 404


# ── Graph Export/Import ───────────────────────────────────

class TestGraphEndpoints:
    def test_export_empty_graph(self, client):
        data = client.get("/v1/graph").json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 0

    def test_export_after_ingest(self, seeded_client):
        data = seeded_client.get("/v1/graph").json()
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0

    def test_import_graph(self, client):
        # First ingest, then export
        client.post("/v1/ingest", json={
            "source_id": "test",
            "content": "We decided to use FastAPI for the REST layer.",
        })
        exported = client.get("/v1/graph").json()
        node_count = len(exported["nodes"])

        # Now import into a "fresh" state
        resp = client.post("/v1/graph/import", json=exported)
        assert resp.status_code == 200
        assert resp.json()["nodes"] == node_count

    def test_export_import_roundtrip(self, seeded_client):
        """Export graph, reimport, verify data survives."""
        exported = seeded_client.get("/v1/graph").json()
        original_count = len(exported["nodes"])

        # Reimport (clears and restores)
        seeded_client.post("/v1/graph/import", json=exported)

        # Verify
        health = seeded_client.get("/v1/health").json()
        assert health["node_count"] == original_count


class TestEdgeEndpoint:
    def test_create_conflicts_with_edge(self, client):
        graph = client.post("/v1/graph/import", json={
            "nodes": [
                {"node_type": "context", "label": "Current claim", "content": "Atlas is the planning hub."},
                {"node_type": "context", "label": "Stale claim", "content": "Ledger is the planning hub."},
            ],
            "edges": [],
        })
        assert graph.status_code == 200
        nodes = client.get("/v1/graph").json()["nodes"]

        resp = client.post("/v1/edges", json={
            "edge_type": "conflicts_with",
            "source_node_id": nodes[0]["node_id"],
            "target_node_id": nodes[1]["node_id"],
            "weight": 0.9,
            "confidence": 0.95,
            "confidence_set_by": "test",
            "source_context": "Hub boundary correction",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["edge_type"] == "conflicts_with"
        assert data["weight"] == 0.9
        assert data["confidence"] == 0.95

        edge_types = {edge["edge_type"] for edge in client.get("/v1/graph").json()["edges"]}
        assert "conflicts_with" in edge_types

    def test_create_edge_rejects_unknown_type(self, client):
        resp = client.post("/v1/edges", json={
            "edge_type": "not_a_real_edge",
            "source_node_id": "missing-a",
            "target_node_id": "missing-b",
        })
        assert resp.status_code == 400
        assert "Invalid edge_type" in resp.json()["detail"]


# ── Sync Endpoint ─────────────────────────────────────────

class TestSyncEndpoint:
    def test_sync_returns_200(self, client):
        resp = client.post("/v1/sync")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
