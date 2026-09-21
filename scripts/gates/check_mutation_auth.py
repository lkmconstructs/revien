"""G4: every state-changing daemon route refuses a remote caller without the
pairing token and accepts one with it.

CHECK: python scripts/gates/check_mutation_auth.py
EXPECT: mutation auth verification passed
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TMPHOME = tempfile.mkdtemp()
os.environ["REVIEN_HOME"] = TMPHOME

from fastapi.testclient import TestClient  # noqa: E402

from revien.daemon import server as server_module  # noqa: E402
from revien.daemon.server import create_app  # noqa: E402

TOKEN = "mara-gate-pass"


def fail(msg):
    print(f"check_mutation_auth: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=db_path)
    client = TestClient(app)

    # Seed a fact node + get a skill proposal id to exercise path-param
    # routes (auth is checked BEFORE the node lookup in every gated route,
    # so a real id isn't strictly required, but using one keeps this
    # honest against a route that might reorder the checks later).
    seed = client.post("/v1/ingest", json={
        "source_id": "gate-check-seed",
        "content": "We decided to use PostgreSQL for the Fernweh-Core API.",
    })
    if seed.status_code != 200:
        fail(f"seed ingest failed: {seed.status_code} {seed.text}")
    node_id = seed.json()["context_node_id"]

    edges_resp = client.get("/v1/graph")
    if edges_resp.status_code != 200:
        fail("could not fetch /v1/graph for import payload")
    graph_payload = edges_resp.json()

    # Build a skill proposal to exercise /v1/skills/{id}/accept and /decline.
    store = app.state.store
    from datetime import datetime, timedelta, timezone
    from revien.graph.schema import Node, NodeType, SourceType
    from revien.skills.proposals import propose_skills

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 0
    for sess in ("s1", "s2", "s3"):
        for step in ("I'll ping Theo", "I'll run the migration"):
            store.add_node(Node(
                node_type=NodeType.ACTION, label=step, content=step,
                source_id=f"claude-code:fernweh:{sess}", source_type=SourceType.EXTRACTED,
                confidence=0.8, project_key="fernweh", session_key=sess,
                recorded_at=base + timedelta(minutes=n),
            ))
            n += 1
    proposal_summary = propose_skills(store)
    skill_node_id = proposal_summary["proposals"][0].node_id

    # ── enumerate every POST/PUT/DELETE route on the app, minus the two
    # documented carve-outs: /v1/recall (read-only despite being POST) and
    # /v1/ingest (its own check_capture_auth gate, exercised separately
    # below).
    ROUTE_SUBSTITUTIONS = {
        "{node_id}": node_id,
    }
    mutating_routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in ("POST", "PUT", "DELETE"):
            if method not in methods:
                continue
            if path in ("/v1/recall", "/v1/ingest"):
                continue
            resolved_path = path
            for placeholder, value in ROUTE_SUBSTITUTIONS.items():
                resolved_path = resolved_path.replace(placeholder, value)
            if "{" in resolved_path:
                # skills/{node_id}/accept etc. share the same placeholder;
                # anything still unresolved falls back to the skill node id.
                resolved_path = resolved_path.replace("{node_id}", skill_node_id)
            mutating_routes.append((method, resolved_path))

    if not mutating_routes:
        fail("route enumeration found zero POST/PUT/DELETE routes -- broken enumeration")

    expected_min_routes = 15
    if len(mutating_routes) < expected_min_routes:
        fail(
            f"only found {len(mutating_routes)} mutating routes "
            f"(expected at least {expected_min_routes}) -- enumeration is probably wrong"
        )

    # Force the remote-caller branch: TestClient reports client host
    # "testclient", which is loopback-exempt by default. Emptying
    # _LOOPBACK_HOSTS is the same technique the real test suite uses to
    # prove a route actually calls the gate.
    server_module._LOOPBACK_HOSTS = set()
    try:
        # A handful of mutation routes require a body that passes pydantic
        # validation before require_mutation_auth is ever reached -- an
        # empty {} 422s there, which would wrongly read as "accepted the
        # unauthenticated call". Supply a minimally valid body for those so
        # the auth gate is what actually gets exercised.
        BODY_BY_PATH = {
            "/v1/edges": {
                "edge_type": "related_to",
                "source_node_id": node_id,
                "target_node_id": node_id,
            },
            "/v1/mark_used": {"node_id": node_id},
        }

        unauth_failures = []
        for method, path in mutating_routes:
            kwargs = {}
            if method in ("POST", "PUT"):
                kwargs["json"] = BODY_BY_PATH.get(path, {})
            resp = client.request(method, path, **kwargs)
            if resp.status_code not in (401, 403):
                unauth_failures.append((method, path, resp.status_code))
        if unauth_failures:
            fail(f"routes accepted an unauthenticated remote call: {unauth_failures}")

        # ── remote caller WITH the correct pairing token must pass ──
        os.environ["REVIEN_CAPTURE_TOKEN"] = TOKEN
        try:
            resp = client.post(
                f"/v1/skills/{skill_node_id}/accept",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            if resp.status_code in (401, 403):
                fail(f"remote call WITH correct token was refused: {resp.status_code} {resp.text}")
        finally:
            del os.environ["REVIEN_CAPTURE_TOKEN"]
    finally:
        server_module._LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient", ""}

    # ── loopback caller with NO header must NOT be refused (intentional
    # bypass per the auth model -- _LOOPBACK_HOSTS restored above) ──
    loopback_resp = client.post(f"/v1/skills/{skill_node_id}/decline")
    if loopback_resp.status_code in (401, 403):
        fail(
            f"loopback caller with no auth header was refused "
            f"({loopback_resp.status_code}) -- loopback bypass is intentional"
        )

    try:
        os.unlink(db_path)
    except PermissionError:
        pass
    import shutil
    shutil.rmtree(TMPHOME, ignore_errors=True)

    print("mutation auth verification passed")


if __name__ == "__main__":
    main()
