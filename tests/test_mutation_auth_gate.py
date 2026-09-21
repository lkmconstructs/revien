"""
F1 (blocker): every state-changing daemon route must gate on
require_mutation_auth exactly the way /v1/skills/{id}/accept already does —
loopback callers unaffected, remote callers refused without the pairing
token. This file enumerates every mutation route found by grep in
revien/daemon/server.py (@app.post/@app.put/@app.delete, minus /v1/ingest
which has its own equivalent check_capture_auth gate and the read-only
/v1/recall) and drives each one through the daemon_client fixture.

Also covers the second half of F1: PUT /v1/nodes/{id} on a SKILL node must
refuse (409) a metadata change to status/origin/curated — those transitions
belong to accept/decline/ingest only — while every other field (content,
label, other metadata keys) still goes through on 200.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from revien.daemon import server as server_module
from revien.daemon.server import create_app
from revien.graph.schema import Node, NodeType, SourceType


@pytest.fixture
def daemon_client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        yield c
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _force_remote(monkeypatch):
    """Same technique test_skill_proposals.py's own accept/decline gate
    tests use: TestClient always reports host "testclient", which is
    loopback-exempt by default, so emptying _LOOPBACK_HOSTS is what proves
    a route actually calls the gate rather than skipping it by accident."""
    monkeypatch.setattr(server_module, "_LOOPBACK_HOSTS", set())


def _seed_fact(client) -> str:
    resp = client.post("/v1/ingest", json={
        "source_id": "test-mutation-gate",
        "content": "We decided to use PostgreSQL for the Fernweh-Core API.",
    })
    assert resp.status_code == 200
    node_id = resp.json()["context_node_id"]
    return node_id


class TestEveryMutationRouteIsGated:
    """One remote-without-token case per mutation route. Each must be
    refused (403) exactly like /v1/skills/{id}/accept already is — proving
    the route actually calls require_mutation_auth/check_capture_auth,
    not merely that it exists."""

    def test_put_nodes_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"label": "x"})
        assert resp.status_code == 403

    def test_delete_nodes_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.delete(f"/v1/nodes/{node_id}")
        assert resp.status_code == 403

    def test_post_edges_remote_403(self, daemon_client, monkeypatch):
        n1 = _seed_fact(daemon_client)
        n2 = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/edges", json={
            "edge_type": "related_to", "source_node_id": n1, "target_node_id": n2,
        })
        assert resp.status_code == 403

    def test_post_consolidate_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/consolidate", json={})
        assert resp.status_code == 403

    def test_post_graph_import_remote_403(self, daemon_client, monkeypatch):
        exported = daemon_client.get("/v1/graph").json()
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/graph/import", json=exported)
        assert resp.status_code == 403

    def test_post_cluster_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/cluster")
        assert resp.status_code == 403

    def test_post_sync_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/sync")
        assert resp.status_code == 403

    def test_post_mark_used_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/mark_used", json={"node_id": node_id})
        assert resp.status_code == 403

    def test_post_training_run_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/training/run")
        assert resp.status_code == 403

    def test_post_reinforce_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post(f"/v1/nodes/{node_id}/reinforce")
        assert resp.status_code == 403

    def test_post_correct_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post(f"/v1/nodes/{node_id}/correct")
        assert resp.status_code == 403

    def test_post_invalidate_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post(f"/v1/nodes/{node_id}/invalidate")
        assert resp.status_code == 403

    def test_post_retention_sweep_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/retention/sweep")
        assert resp.status_code == 403

    def test_post_forget_remote_403(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        resp = daemon_client.post(f"/v1/nodes/{node_id}/forget")
        assert resp.status_code == 403

    def test_post_reindex_remote_403(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        resp = daemon_client.post("/v1/reindex")
        assert resp.status_code == 403


class TestLoopbackUnaffected:
    """Every mutation route above still returns its normal (non-403/401)
    result on loopback — the pairing gate must not have changed local
    behavior at all."""

    def test_put_nodes_loopback_200(self, daemon_client):
        node_id = _seed_fact(daemon_client)
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"label": "Renamed"})
        assert resp.status_code == 200

    def test_delete_nodes_loopback_200(self, daemon_client):
        node_id = _seed_fact(daemon_client)
        resp = daemon_client.delete(f"/v1/nodes/{node_id}")
        assert resp.status_code == 200

    def test_post_cluster_loopback_200(self, daemon_client):
        resp = daemon_client.post("/v1/cluster")
        assert resp.status_code == 200

    def test_post_sync_loopback_200(self, daemon_client):
        resp = daemon_client.post("/v1/sync")
        assert resp.status_code == 200

    def test_post_reindex_loopback_200(self, daemon_client):
        resp = daemon_client.post("/v1/reindex")
        assert resp.status_code == 200

    def test_post_retention_sweep_loopback_200(self, daemon_client):
        resp = daemon_client.post("/v1/retention/sweep")
        assert resp.status_code == 200

    def test_post_training_run_loopback_200(self, daemon_client):
        resp = daemon_client.post("/v1/training/run")
        assert resp.status_code == 200


class TestRemoteWithTokenPasses:
    """Spot-check (not every route — the gate is one shared function) that
    a correctly-presented pairing token still lets a remote caller through,
    exactly like the skills accept/decline routes already do."""

    def test_put_nodes_remote_with_token_200(self, daemon_client, monkeypatch):
        node_id = _seed_fact(daemon_client)
        _force_remote(monkeypatch)
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret-pairing-token")
        resp = daemon_client.put(
            f"/v1/nodes/{node_id}",
            json={"label": "Renamed"},
            headers={"Authorization": "Bearer s3cret-pairing-token"},
        )
        assert resp.status_code == 200

    def test_post_cluster_remote_with_token_200(self, daemon_client, monkeypatch):
        _force_remote(monkeypatch)
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret-pairing-token")
        resp = daemon_client.post(
            "/v1/cluster", headers={"Authorization": "Bearer s3cret-pairing-token"},
        )
        assert resp.status_code == 200


# ── PUT /v1/nodes/{id} on a SKILL node: status/origin/curated are frozen ──


class TestSkillMetadataFreezeOnPut:
    def _seed_skill(self, daemon_client) -> str:
        store = daemon_client.app.state.store
        node = store.add_node(Node(
            node_type=NodeType.SKILL,
            label="sync fernweh",
            content="## Steps\n\n1. sync\n2. bench\n",
            metadata={
                "description": "sync fernweh branches",
                "triggers": ["fernweh"],
                "version": "1.0",
                "origin": "user",
                "curated": True,
                "status": "active",
                "scope": "project",
                "path": "/tmp/SKILL.md",
            },
            source_type=SourceType.EXTRACTED,
            confidence=1.0,
        ))
        return node.node_id

    def test_changing_status_is_409(self, daemon_client):
        node_id = self._seed_skill(daemon_client)
        store = daemon_client.app.state.store
        md = dict(store.get_node(node_id).metadata)
        md["status"] = "proposed"
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"metadata": md})
        assert resp.status_code == 409

    def test_changing_origin_is_409(self, daemon_client):
        node_id = self._seed_skill(daemon_client)
        store = daemon_client.app.state.store
        md = dict(store.get_node(node_id).metadata)
        md["origin"] = "engine"
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"metadata": md})
        assert resp.status_code == 409

    def test_changing_curated_is_409(self, daemon_client):
        node_id = self._seed_skill(daemon_client)
        store = daemon_client.app.state.store
        md = dict(store.get_node(node_id).metadata)
        md["curated"] = False
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"metadata": md})
        assert resp.status_code == 409

    def test_changing_description_is_200(self, daemon_client):
        node_id = self._seed_skill(daemon_client)
        store = daemon_client.app.state.store
        md = dict(store.get_node(node_id).metadata)
        md["description"] = "an updated description"
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"metadata": md})
        assert resp.status_code == 200
        assert resp.json()["metadata"]["description"] == "an updated description"

    def test_content_only_change_is_200(self, daemon_client):
        node_id = self._seed_skill(daemon_client)
        resp = daemon_client.put(
            f"/v1/nodes/{node_id}", json={"content": "## Steps\n\n1. sync\n"}
        )
        assert resp.status_code == 200

    def test_setting_same_status_value_is_200(self, daemon_client):
        """The 409 fires on an actual CHANGE, not merely on the field's
        presence in the request body."""
        node_id = self._seed_skill(daemon_client)
        store = daemon_client.app.state.store
        md = dict(store.get_node(node_id).metadata)  # status already "active"
        resp = daemon_client.put(f"/v1/nodes/{node_id}", json={"metadata": md})
        assert resp.status_code == 200

    def test_non_skill_node_unaffected(self, daemon_client):
        """The freeze only applies to node_type == SKILL — an ordinary
        node's metadata (even one that happens to have a "status" key) can
        still be freely changed via PUT."""
        store = daemon_client.app.state.store
        node = store.add_node(Node(
            node_type=NodeType.FACT, label="not a skill", content="x",
            metadata={"status": "whatever"},
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))
        resp = daemon_client.put(
            f"/v1/nodes/{node.node_id}",
            json={"metadata": {"status": "something-else"}},
        )
        assert resp.status_code == 200
