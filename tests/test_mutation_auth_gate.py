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


def _mutation_route_cases():
    """(case_id, builder) — builder(client) -> (method, path, json_body).
    One entry per mutation route. The builder runs BEFORE the caller forces
    a remote host, so any seed data it needs (a fact node, two node ids for
    an edge, an exported graph for import) is created over the normal
    loopback path, exactly like the original per-route tests did."""

    def put_nodes(client):
        return "put", f"/v1/nodes/{_seed_fact(client)}", {"label": "x"}

    def delete_nodes(client):
        return "delete", f"/v1/nodes/{_seed_fact(client)}", None

    def post_edges(client):
        n1, n2 = _seed_fact(client), _seed_fact(client)
        return "post", "/v1/edges", {
            "edge_type": "related_to", "source_node_id": n1, "target_node_id": n2,
        }

    def post_consolidate(client):
        return "post", "/v1/consolidate", {}

    def post_graph_import(client):
        exported = client.get("/v1/graph").json()
        return "post", "/v1/graph/import", exported

    def post_cluster(client):
        return "post", "/v1/cluster", None

    def post_sync(client):
        return "post", "/v1/sync", None

    def post_mark_used(client):
        return "post", "/v1/mark_used", {"node_id": _seed_fact(client)}

    def post_training_run(client):
        return "post", "/v1/training/run", None

    def post_reinforce(client):
        return "post", f"/v1/nodes/{_seed_fact(client)}/reinforce", None

    def post_correct(client):
        return "post", f"/v1/nodes/{_seed_fact(client)}/correct", None

    def post_invalidate(client):
        return "post", f"/v1/nodes/{_seed_fact(client)}/invalidate", None

    def post_retention_sweep(client):
        return "post", "/v1/retention/sweep", None

    def post_forget(client):
        return "post", f"/v1/nodes/{_seed_fact(client)}/forget", None

    def post_reindex(client):
        return "post", "/v1/reindex", None

    return {
        "put_nodes": put_nodes,
        "delete_nodes": delete_nodes,
        "post_edges": post_edges,
        "post_consolidate": post_consolidate,
        "post_graph_import": post_graph_import,
        "post_cluster": post_cluster,
        "post_sync": post_sync,
        "post_mark_used": post_mark_used,
        "post_training_run": post_training_run,
        "post_reinforce": post_reinforce,
        "post_correct": post_correct,
        "post_invalidate": post_invalidate,
        "post_retention_sweep": post_retention_sweep,
        "post_forget": post_forget,
        "post_reindex": post_reindex,
    }


_MUTATION_ROUTE_CASES = _mutation_route_cases()


def _call(client, method, path, body):
    """TestClient.delete() (unlike put/post) takes no ``json`` kwarg."""
    if method == "delete":
        return client.delete(path)
    return getattr(client, method)(path, json=body)

# Loopback-200 originally spot-checked a subset of the routes above, not
# every one (the gate is one shared function, per TestRemoteWithTokenPasses)
# — same subset, kept.
_LOOPBACK_CASE_IDS = (
    "put_nodes", "delete_nodes", "post_cluster", "post_sync",
    "post_reindex", "post_retention_sweep", "post_training_run",
)


class TestEveryMutationRouteIsGated:
    """One remote-without-token case per mutation route. Each must be
    refused (403) exactly like /v1/skills/{id}/accept already is — proving
    the route actually calls require_mutation_auth/check_capture_auth,
    not merely that it exists."""

    @pytest.mark.parametrize("case_id", _MUTATION_ROUTE_CASES.keys())
    def test_remote_403(self, daemon_client, monkeypatch, case_id):
        method, path, body = _MUTATION_ROUTE_CASES[case_id](daemon_client)
        _force_remote(monkeypatch)
        resp = _call(daemon_client, method, path, body)
        assert resp.status_code == 403


class TestLoopbackUnaffected:
    """Every mutation route above still returns its normal (non-403/401)
    result on loopback — the pairing gate must not have changed local
    behavior at all."""

    @pytest.mark.parametrize("case_id", _LOOPBACK_CASE_IDS)
    def test_loopback_200(self, daemon_client, case_id):
        method, path, body = _MUTATION_ROUTE_CASES[case_id](daemon_client)
        resp = _call(daemon_client, method, path, body)
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
