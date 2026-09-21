"""
Origin Layer (WS0) tests — Leg B: recall/CLI/daemon/TOON surfaces over the
origin_runtime/origin_source/project_key/session_key fields Leg A stamped
on every node.

Covers: store.list_nodes / search_nodes_keyword SQL prefilters, engine.recall
source= isolating one or more runtimes (including that a filtered recall
never leaks a different runtime's node reached only through graph-walk
expansion off an in-filter anchor), None-origin nodes never matching a
filter, RetrievalResult/response dict carrying origin fields on every
result, TOON round-trip with the new fields, the daemon's RecallRequest.source
and GET /v1/nodes origin_runtime/project_key params, and the CLI's
--source/status surfaces. Fictional names only (Mara/Theo/Sam, Fernweh-Core).
"""

import os
import tempfile

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.toon import parse_recall, serialize_recall


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _node(label, content, node_type=NodeType.FACT, origin_runtime=None,
          origin_source=None, project_key=None, session_key=None,
          source_id=""):
    return Node(
        node_type=node_type,
        label=label,
        content=content,
        source_id=source_id,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        origin_runtime=origin_runtime,
        origin_source=origin_source,
        project_key=project_key,
        session_key=session_key,
    )


# ── store.list_nodes / search_nodes_keyword SQL prefilters ─────────────────


class TestStoreOriginFilter:
    def test_list_nodes_filters_single_runtime(self, store):
        a = store.add_node(_node("Mara pricing note", "Mara set enterprise "
                                  "pricing at $499.", origin_runtime="claude-code"))
        store.add_node(_node("Theo pricing note", "Theo migrated the DB to "
                              "Postgres.", origin_runtime="codex"))

        results = store.list_nodes(origin_runtime="claude-code")
        assert [n.node_id for n in results] == [a.node_id]

    def test_list_nodes_filters_list_of_runtimes(self, store):
        a = store.add_node(_node("Mara note", "content a", origin_runtime="claude-code"))
        b = store.add_node(_node("Theo note", "content b", origin_runtime="codex"))
        store.add_node(_node("Sam note", "content c", origin_runtime="hermes"))

        results = store.list_nodes(origin_runtime=["claude-code", "codex"])
        ids = {n.node_id for n in results}
        assert ids == {a.node_id, b.node_id}

    def test_list_nodes_project_key_filter(self, store):
        a = store.add_node(_node("Mara note", "content a",
                                  origin_runtime="claude-code",
                                  project_key="Fernweh-Core"))
        store.add_node(_node("Theo note", "content b",
                              origin_runtime="claude-code",
                              project_key="OtherProject"))

        results = store.list_nodes(project_key="Fernweh-Core")
        assert [n.node_id for n in results] == [a.node_id]

    def test_list_nodes_excludes_none_origin_when_filtered(self, store):
        store.add_node(_node("unknown origin", "no provenance", origin_runtime=None))
        a = store.add_node(_node("Mara note", "content", origin_runtime="claude-code"))

        results = store.list_nodes(origin_runtime="claude-code")
        assert [n.node_id for n in results] == [a.node_id]

    def test_search_nodes_keyword_prefilters_by_runtime(self, store):
        a = store.add_node(_node("Mara decision", "We chose Postgres for "
                                  "Fernweh-Core.", origin_runtime="claude-code"))
        store.add_node(_node("Theo decision", "We chose Postgres for the "
                              "sibling repo.", origin_runtime="codex"))

        matches = store.search_nodes_keyword(
            {"postgres"}, origin_runtime="claude-code", exclude_context=False,
        )
        assert [n.node_id for n in matches] == [a.node_id]


# ── engine.recall(source=...) ───────────────────────────────────────────────


class TestRecallSourceFilter:
    def test_filter_isolates_single_runtime(self, store):
        store.add_node(_node("Mara pricing", "Mara decided the enterprise "
                              "tier is $499 per month.", origin_runtime="claude-code"))
        store.add_node(_node("Theo pricing", "Theo decided the enterprise "
                              "tier is $399 per month.", origin_runtime="codex"))
        engine = RetrievalEngine(store)

        response = engine.recall("enterprise tier pricing", source="claude-code")
        assert response.results
        assert all(r.origin_runtime == "claude-code" for r in response.results)
        assert not any(r.origin_runtime == "codex" for r in response.results)

    def test_filter_accepts_list_of_runtimes(self, store):
        store.add_node(_node("Mara pricing", "Mara decided the enterprise "
                              "tier is $499 per month.", origin_runtime="claude-code"))
        store.add_node(_node("Theo pricing", "Theo decided the enterprise "
                              "tier is $399 per month.", origin_runtime="codex"))
        store.add_node(_node("Sam pricing", "Sam decided the enterprise "
                              "tier is $299 per month.", origin_runtime="hermes"))
        engine = RetrievalEngine(store)

        response = engine.recall(
            "enterprise tier pricing", source=["claude-code", "codex"],
        )
        runtimes = {r.origin_runtime for r in response.results}
        assert runtimes <= {"claude-code", "codex"}
        assert "hermes" not in runtimes
        assert runtimes  # at least one of the two matched

    def test_none_origin_nodes_excluded_when_filter_set(self, store):
        store.add_node(_node("mystery pricing", "An unattributed note about "
                              "enterprise tier pricing.", origin_runtime=None))
        store.add_node(_node("Mara pricing", "Mara decided the enterprise "
                              "tier is $499 per month.", origin_runtime="claude-code"))
        engine = RetrievalEngine(store)

        response = engine.recall("enterprise tier pricing", source="claude-code")
        assert all(r.origin_runtime is not None for r in response.results)

    def test_no_filter_is_byte_identical_unfiltered(self, store):
        store.add_node(_node("Mara pricing", "Mara decided the enterprise "
                              "tier is $499 per month.", origin_runtime="claude-code"))
        store.add_node(_node("Theo pricing", "Theo decided the enterprise "
                              "tier is $399 per month.", origin_runtime="codex"))
        engine = RetrievalEngine(store)

        filtered = engine.recall("enterprise tier pricing")
        runtimes = {r.origin_runtime for r in filtered.results}
        assert "claude-code" in runtimes and "codex" in runtimes

    def test_result_carries_origin_fields_always(self, store):
        store.add_node(_node("Mara pricing", "Mara decided the enterprise "
                              "tier is $499 per month.", origin_runtime="claude-code",
                              origin_source="live", project_key="Fernweh-Core"))
        engine = RetrievalEngine(store)

        response = engine.recall("enterprise tier pricing")
        assert response.results
        r = response.results[0]
        assert r.origin_runtime == "claude-code"
        assert r.origin_source == "live"
        assert r.project_key == "Fernweh-Core"

    def test_filter_survives_graph_walk_expansion(self, store):
        """A node reached ONLY through graph-walk expansion off an anchor
        that matched the filter must NOT leak into results if its own
        origin_runtime is a different runtime — the final per-node gate,
        not just the anchor-source SQL prefilter, has to hold."""
        anchor = store.add_node(_node(
            "Fernweh-Core", "Fernweh-Core is Mara's project.",
            node_type=NodeType.ENTITY, origin_runtime="claude-code",
            project_key="Fernweh-Core",
        ))
        # A DIFFERENT runtime's node, graph-connected to the SAME anchor.
        leaker = store.add_node(_node(
            "Theo's migration note", "Theo migrated Fernweh-Core's queue to "
            "Redis.", node_type=NodeType.FACT, origin_runtime="codex",
            project_key="Fernweh-Core",
        ))
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=anchor.node_id,
            target_node_id=leaker.node_id,
        ))
        engine = RetrievalEngine(store)

        response = engine.recall(
            "Fernweh-Core", source="claude-code", include_context=True,
        )
        returned_ids = {r.node_id for r in response.results}
        assert leaker.node_id not in returned_ids
        assert all(r.origin_runtime == "claude-code" for r in response.results)


# ── TOON round-trip with origin fields ──────────────────────────────────────


class TestToonOriginFields:
    def test_round_trips_origin_fields(self):
        payload = {
            "query": "pricing",
            "results": [{
                "node_id": "n-1",
                "node_type": "fact",
                "label": "Mara pricing",
                "content": "enterprise tier is $499/month",
                "score": 0.87,
                "score_breakdown": {"recency": 0.9},
                "path": ["n-1"],
                "origin_runtime": "claude-code",
                "origin_source": "live",
                "project_key": "Fernweh-Core",
            }],
            "nodes_examined": 1,
            "retrieval_time_ms": 1.1,
            "semantic_active": False,
            "semantic_note": None,
        }
        toon = serialize_recall(payload)
        back = parse_recall(toon)
        assert back == payload

    def test_round_trips_none_origin_fields(self):
        payload = {
            "query": "pricing",
            "results": [{
                "node_id": "n-1",
                "node_type": "fact",
                "label": "unattributed",
                "content": "enterprise tier is $499/month",
                "score": 0.5,
                "score_breakdown": {"recency": 0.5},
                "path": [],
                "origin_runtime": None,
                "origin_source": None,
                "project_key": None,
            }],
            "nodes_examined": 1,
            "retrieval_time_ms": 0.5,
            "semantic_active": False,
            "semantic_note": None,
        }
        toon = serialize_recall(payload)
        back = parse_recall(toon)
        assert back == payload


# ── Daemon: /v1/recall source, GET /v1/nodes filters ────────────────────────


@pytest.fixture
def daemon_client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        # Deliberately DIFFERENT vocabulary per runtime — shared entities
        # (e.g. both saying "enterprise pricing") would dedup onto ONE
        # entity node owned by whichever ingest ran first, which would
        # test dedup, not the origin filter.
        c.post("/v1/ingest", json={
            "source_id": "claude-code:Fernweh-Core:sess-1",
            "content": "User: Fernweh-Core will use PostgreSQL for the "
                       "pricing service.\nAssistant: Noted, PostgreSQL for "
                       "the pricing service.",
            "origin_runtime": "claude-code",
            "origin_source": "live",
            "project_key": "Fernweh-Core",
            "session_key": "sess-1",
        })
        c.post("/v1/ingest", json={
            "source_id": "codex:Waypost-Relay:sess-2",
            "content": "User: Waypost-Relay will migrate the queue to "
                       "Redis.\nAssistant: Confirmed, Redis for the "
                       "Waypost-Relay queue.",
            "origin_runtime": "codex",
            "origin_source": "live",
            "project_key": "Waypost-Relay",
            "session_key": "sess-2",
        })
        yield c
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


class TestDaemonOriginFilter:
    def test_recall_source_filters_results(self, daemon_client):
        resp = daemon_client.post("/v1/recall", json={
            "query": "Fernweh-Core PostgreSQL pricing", "source": "claude-code",
            "include_context": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"]
        assert all(r["origin_runtime"] == "claude-code" for r in body["results"])

    def test_recall_source_list_filters_results(self, daemon_client):
        resp = daemon_client.post("/v1/recall", json={
            "query": "Waypost-Relay Redis queue", "source": ["codex"],
            "include_context": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"]
        assert all(r["origin_runtime"] == "codex" for r in body["results"])

    def test_recall_results_carry_origin_fields(self, daemon_client):
        resp = daemon_client.post("/v1/recall", json={
            "query": "Fernweh-Core PostgreSQL pricing", "include_context": True,
        })
        body = resp.json()
        assert body["results"]
        for r in body["results"]:
            assert "origin_runtime" in r
            assert "origin_source" in r
            assert "project_key" in r

    def test_get_nodes_origin_runtime_filter(self, daemon_client):
        resp = daemon_client.get("/v1/nodes", params={"origin_runtime": "codex"})
        assert resp.status_code == 200
        nodes = resp.json()
        assert nodes
        assert all(n["origin_runtime"] == "codex" for n in nodes)

    def test_get_nodes_project_key_filter(self, daemon_client):
        resp = daemon_client.get("/v1/nodes", params={"project_key": "Fernweh-Core"})
        assert resp.status_code == 200
        nodes = resp.json()
        assert nodes
        assert all(n["project_key"] == "Fernweh-Core" for n in nodes)


# ── CLI: --source, status ───────────────────────────────────────────────────


class TestCliOriginFilter:
    def _db_with_two_runtimes(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = GraphStore(db_path=path)
        s.add_node(_node("Mara pricing", "Mara decided the enterprise tier "
                          "is $499 per month.", origin_runtime="claude-code"))
        s.add_node(_node("Theo pricing", "Theo decided the enterprise tier "
                          "is $399 per month.", origin_runtime="codex"))
        s.close()
        return path

    def test_recall_source_option_filters(self):
        db_path = self._db_with_two_runtimes()
        try:
            runner = CliRunner()
            result = runner.invoke(main, [
                "recall", "enterprise tier pricing",
                "--db", db_path, "--source", "claude-code", "--json-output",
            ])
            assert result.exit_code == 0, result.output
            import json as _json
            output = _json.loads(result.output)
            assert output["results"]
            assert all(
                r["origin_runtime"] == "claude-code" for r in output["results"]
            )
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:  # pragma: no cover
                pass

    def test_recall_tabular_output_shows_runtime(self):
        db_path = self._db_with_two_runtimes()
        try:
            runner = CliRunner()
            result = runner.invoke(main, [
                "recall", "enterprise tier pricing", "--db", db_path,
            ])
            assert result.exit_code == 0, result.output
            assert "Runtime:" in result.output
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:  # pragma: no cover
                pass

    def test_status_prints_per_runtime_table(self):
        db_path = self._db_with_two_runtimes()
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["status", "--db", db_path])
            assert result.exit_code == 0, result.output
            assert "Nodes by runtime:" in result.output
            assert "claude-code:" in result.output
            assert "codex:" in result.output
            # Leg C's pairing-token line must still be there.
            assert "Pairing token:" in result.output
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:  # pragma: no cover
                pass
