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


# ── Sync Endpoint ─────────────────────────────────────────

class TestSyncEndpoint:
    def test_sync_returns_200(self, client):
        resp = client.post("/v1/sync")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
