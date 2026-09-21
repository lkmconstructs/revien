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
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner
from fastapi import HTTPException
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import (
    check_capture_auth,
    create_app,
    _normalize_origin_runtime_param,
)
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.skills.proposals import matching_proposals, propose_skills
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


# ── G1: tensions must respect the source filter ─────────────────────────────


class TestTensionsSourceFilter:
    def test_foreign_tension_partner_is_dropped(self, store):
        """leak1.py scenario, cut down: a codex claim CONFLICTS_WITH a
        claude-code claim. A recall filtered to codex must return the
        codex claim WITHOUT a claude-code tension riding along on it."""
        c_codex = store.add_node(_node(
            "Fernweh ships Tuesday", "Fernweh ships Tuesday",
            origin_runtime="codex",
        ))
        c_cc = store.add_node(_node(
            "Fernweh slips to Friday", "Fernweh slips to Friday",
            origin_runtime="claude-code",
        ))
        store.add_edge(Edge(
            edge_type=EdgeType.CONFLICTS_WITH,
            source_node_id=c_codex.node_id, target_node_id=c_cc.node_id,
            weight=1.0, source_context="conflict",
        ))
        engine = RetrievalEngine(store)

        response = engine.recall(
            "Fernweh ships", source="codex", include_tensions=True,
        )
        assert response.results
        for r in response.results:
            for t in r.tensions:
                assert t["node_id"] != c_cc.node_id

    def test_none_origin_tension_partner_is_dropped(self, store):
        c_codex = store.add_node(_node(
            "Fernweh ships Tuesday", "Fernweh ships Tuesday",
            origin_runtime="codex",
        ))
        c_unknown = store.add_node(_node(
            "unattributed conflicting note", "unattributed conflicting note",
            origin_runtime=None,
        ))
        store.add_edge(Edge(
            edge_type=EdgeType.CONFLICTS_WITH,
            source_node_id=c_codex.node_id, target_node_id=c_unknown.node_id,
            weight=1.0,
        ))
        engine = RetrievalEngine(store)

        response = engine.recall(
            "Fernweh ships", source="codex", include_tensions=True,
        )
        assert response.results
        for r in response.results:
            for t in r.tensions:
                assert t["node_id"] != c_unknown.node_id

    def test_same_runtime_tension_partner_still_surfaces(self, store):
        """Not a leak regression on its own: filtering must not zero out
        an IN-FILTER tension partner."""
        c1 = store.add_node(_node(
            "Fernweh ships Tuesday", "Fernweh ships Tuesday",
            origin_runtime="codex",
        ))
        c2 = store.add_node(_node(
            "Fernweh ships Wednesday instead", "Fernweh ships Wednesday instead",
            origin_runtime="codex",
        ))
        store.add_edge(Edge(
            edge_type=EdgeType.CONFLICTS_WITH,
            source_node_id=c1.node_id, target_node_id=c2.node_id, weight=1.0,
        ))
        engine = RetrievalEngine(store)

        response = engine.recall(
            "Fernweh ships", source="codex", include_tensions=True,
        )
        found = {t["node_id"] for r in response.results for t in r.tensions}
        assert c2.node_id in found


# ── G2: path labels must respect the source filter ──────────────────────────


class TestPathLabelSourceFilter:
    def test_foreign_hop_label_is_filtered_not_leaked(self, store):
        """leak2.py scenario: anchor(codex) -> mid(claude-code) -> leaf(codex).
        A recall filtered to codex must never spell mid's claude-code label
        into the response, but the path's hop COUNT must stay honest."""
        anchor = store.add_node(_node(
            "Fernweh", "Fernweh", node_type=NodeType.ENTITY,
            origin_runtime="codex",
        ))
        mid = store.add_node(_node(
            "PRIVATE claude-code secret topic", "PRIVATE claude-code secret topic",
            node_type=NodeType.TOPIC, origin_runtime="claude-code",
        ))
        leaf = store.add_node(_node(
            "codex leaf reached via foreign hop",
            "codex leaf reached via foreign hop", origin_runtime="codex",
        ))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                             source_node_id=anchor.node_id, target_node_id=mid.node_id))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                             source_node_id=mid.node_id, target_node_id=leaf.node_id))
        engine = RetrievalEngine(store)

        response = engine.recall("Fernweh", source="codex", debug=True)
        for r in response.results:
            assert "PRIVATE claude-code secret topic" not in r.path
            if r.node_id == leaf.node_id:
                assert "[filtered]" in r.path
                # Hop count preserved: the path is still 3 entries long,
                # the middle one is just redacted, not dropped.
                assert len(r.path) == 3

    def test_unfiltered_path_labels_unchanged(self, store):
        anchor = store.add_node(_node(
            "Fernweh", "Fernweh", node_type=NodeType.ENTITY,
            origin_runtime="codex",
        ))
        mid = store.add_node(_node(
            "public topic", "public topic", node_type=NodeType.TOPIC,
            origin_runtime="claude-code",
        ))
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                             source_node_id=anchor.node_id, target_node_id=mid.node_id))
        engine = RetrievalEngine(store)

        response = engine.recall("Fernweh")
        labels = {lbl for r in response.results for lbl in r.path}
        assert "[filtered]" not in labels


# ── G3: skill_proposals must respect the source filter ──────────────────────


def _proposal_node(store, label, steps, origin_runtime, pattern_hash,
                    status="proposed", draft=True):
    return store.add_node(Node(
        node_type=NodeType.SKILL, label=label, content="body",
        source_id=f"skill-proposal:{pattern_hash}",
        metadata={
            "origin": "engine", "status": status, "pattern_hash": pattern_hash,
            "occurrences": 4, "sessions": 2, "declines": 0,
            "steps": steps, "draft": draft,
        },
        source_type=SourceType.INFERRED, confidence=0.5,
        origin_runtime=origin_runtime, origin_source="live",
        project_key=f"p-{origin_runtime}",
        recorded_at=datetime.now(timezone.utc),
    ))


class TestSkillProposalsSourceFilter:
    def test_matching_proposals_filters_by_source(self, store):
        codex_prop = _proposal_node(
            store, "proposed: sync fernweh branches",
            ["sync fernweh branches"], "codex", "hash-codex",
        )
        cc_prop = _proposal_node(
            store, "proposed: sync fernweh branches too",
            ["sync fernweh branches"], "claude-code", "hash-cc",
        )
        out = matching_proposals(store, "sync fernweh branches", ["codex"])
        ids = {p["node_id"] for p in out}
        assert codex_prop.node_id in ids
        assert cc_prop.node_id not in ids

    def test_matching_proposals_unfiltered_returns_both(self, store):
        codex_prop = _proposal_node(
            store, "proposed: sync fernweh branches",
            ["sync fernweh branches"], "codex", "hash-codex",
        )
        cc_prop = _proposal_node(
            store, "proposed: sync fernweh branches too",
            ["sync fernweh branches"], "claude-code", "hash-cc",
        )
        out = matching_proposals(store, "sync fernweh branches", None)
        ids = {p["node_id"] for p in out}
        assert codex_prop.node_id in ids
        assert cc_prop.node_id in ids

    def test_recall_skill_proposals_respect_source(self, store):
        _proposal_node(
            store, "proposed: sync fernweh branches",
            ["sync fernweh branches"], "codex", "hash-codex",
        )
        cc_prop = _proposal_node(
            store, "proposed: sync fernweh branches too",
            ["sync fernweh branches"], "claude-code", "hash-cc",
        )
        engine = RetrievalEngine(store)
        response = engine.recall(
            "sync fernweh branches", source="codex", min_score=0.0,
        )
        ids = {p["node_id"] for p in response.skill_proposals}
        assert cc_prop.node_id not in ids

    def test_none_origin_proposal_excluded_when_filtered(self, store):
        unattributed = _proposal_node(
            store, "proposed: sync fernweh branches",
            ["sync fernweh branches"], None, "hash-none",
        )
        out = matching_proposals(store, "sync fernweh branches", ["codex"])
        ids = {p["node_id"] for p in out}
        assert unattributed.node_id not in ids


# ── G4: an empty (but present) source filter fails closed ───────────────────


class TestEmptyFilterFailsClosed:
    def test_engine_recall_source_empty_list(self, store):
        store.add_node(_node("Mara note", "Mara note content",
                              origin_runtime="claude-code"))
        engine = RetrievalEngine(store)
        response = engine.recall("Mara note", source=[])
        assert response.results == []
        assert response.skill_proposals == []

    def test_engine_recall_source_empty_string(self, store):
        store.add_node(_node("Mara note", "Mara note content",
                              origin_runtime="claude-code"))
        engine = RetrievalEngine(store)
        response = engine.recall("Mara note", source="")
        assert response.results == []

    def test_engine_recall_source_list_of_empty_string(self, store):
        store.add_node(_node("Mara note", "Mara note content",
                              origin_runtime="claude-code"))
        engine = RetrievalEngine(store)
        response = engine.recall("Mara note", source=[""])
        assert response.results == []

    def test_engine_recall_source_none_is_unfiltered(self, store):
        store.add_node(_node("Mara note", "Mara note content",
                              origin_runtime="claude-code"))
        engine = RetrievalEngine(store)
        response = engine.recall("Mara note", source=None)
        assert response.results

    def test_engine_recall_empty_filter_with_tensions(self, store):
        c1 = store.add_node(_node("Fernweh ships Tuesday",
                                   "Fernweh ships Tuesday", origin_runtime="codex"))
        c2 = store.add_node(_node("Fernweh ships Friday",
                                   "Fernweh ships Friday", origin_runtime="codex"))
        store.add_edge(Edge(edge_type=EdgeType.CONFLICTS_WITH,
                             source_node_id=c1.node_id, target_node_id=c2.node_id,
                             weight=1.0))
        engine = RetrievalEngine(store)
        response = engine.recall("Fernweh ships", source=[], include_tensions=True)
        assert response.results == []

    def test_store_list_nodes_origin_runtime_empty_list(self, store):
        store.add_node(_node("Mara note", "content", origin_runtime="claude-code"))
        assert store.list_nodes(origin_runtime=[]) == []

    def test_store_list_nodes_origin_runtime_none_unfiltered(self, store):
        store.add_node(_node("Mara note", "content", origin_runtime="claude-code"))
        assert store.list_nodes(origin_runtime=None)

    def test_store_search_nodes_keyword_origin_runtime_empty_list(self, store):
        store.add_node(_node("Mara pricing note", "enterprise pricing details",
                              origin_runtime="claude-code"))
        matches = store.search_nodes_keyword(
            {"pricing"}, origin_runtime=[], exclude_context=False,
        )
        assert matches == []


# ── G5: GET /v1/nodes origin_runtime accepts multiple shapes ────────────────


class TestNormalizeOriginRuntimeParam:
    def test_none_stays_none(self):
        assert _normalize_origin_runtime_param(None) is None

    def test_repeated_params(self):
        assert _normalize_origin_runtime_param(
            ["claude-code", "codex"]
        ) == ["claude-code", "codex"]

    def test_comma_separated_single_item(self):
        assert _normalize_origin_runtime_param(
            ["claude-code,codex"]
        ) == ["claude-code", "codex"]

    def test_single_value(self):
        assert _normalize_origin_runtime_param(["claude-code"]) == ["claude-code"]

    def test_empty_string_value_supplied(self):
        assert _normalize_origin_runtime_param([""]) == []

    def test_blank_comma_segments_dropped(self):
        assert _normalize_origin_runtime_param(
            ["claude-code,,codex, "]
        ) == ["claude-code", "codex"]


class TestGetNodesOriginRuntimeShapes:
    def test_repeated(self, daemon_client):
        resp = daemon_client.get(
            "/v1/nodes",
            params=[("origin_runtime", "claude-code"), ("origin_runtime", "codex")],
        )
        assert resp.status_code == 200
        nodes = resp.json()
        assert nodes
        assert all(n["origin_runtime"] in ("claude-code", "codex") for n in nodes)

    def test_comma_separated(self, daemon_client):
        resp = daemon_client.get(
            "/v1/nodes", params={"origin_runtime": "claude-code,codex"},
        )
        assert resp.status_code == 200
        nodes = resp.json()
        assert nodes
        assert all(n["origin_runtime"] in ("claude-code", "codex") for n in nodes)

    def test_single(self, daemon_client):
        resp = daemon_client.get(
            "/v1/nodes", params={"origin_runtime": "codex"},
        )
        assert resp.status_code == 200
        nodes = resp.json()
        assert nodes
        assert all(n["origin_runtime"] == "codex" for n in nodes)

    def test_empty_string_fails_closed(self, daemon_client):
        resp = daemon_client.get(
            "/v1/nodes", params={"origin_runtime": ""},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_absent_param_is_unfiltered(self, daemon_client):
        resp = daemon_client.get("/v1/nodes")
        assert resp.status_code == 200
        nodes = resp.json()
        runtimes = {n["origin_runtime"] for n in nodes}
        assert "claude-code" in runtimes and "codex" in runtimes


# ── G6: composite indexes avoid a temp B-tree sort ──────────────────────────


class TestOriginIndexPlan:
    def test_list_nodes_query_plan_has_no_temp_btree_sort(self, store):
        for i in range(5):
            store.add_node(_node(f"note {i}", f"content {i}",
                                  origin_runtime="codex" if i % 2 else "claude-code"))
        conn = store._get_conn()
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM nodes WHERE origin_runtime IN (?) "
            "ORDER BY created_at DESC, node_id LIMIT ? OFFSET ?",
            ("codex", 100, 0),
        ).fetchall()
        plan_text = " ".join(row[-1] for row in plan)
        assert "USE TEMP B-TREE FOR ORDER BY" not in plan_text

    def test_project_key_query_plan_has_no_temp_btree_sort(self, store):
        for i in range(5):
            store.add_node(_node(f"note {i}", f"content {i}", project_key="p1"))
        conn = store._get_conn()
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM nodes WHERE project_key = ? "
            "ORDER BY created_at DESC, node_id LIMIT ? OFFSET ?",
            ("p1", 100, 0),
        ).fetchall()
        plan_text = " ".join(row[-1] for row in plan)
        assert "USE TEMP B-TREE FOR ORDER BY" not in plan_text


# ── G7: list_nodes paging is stable under created_at ties ───────────────────


class TestListNodesOrderingStable:
    def test_node_id_tiebreaker_prevents_dupes_and_skips(self, store):
        conn = store._get_conn()
        now = "2026-01-01T00:00:00+00:00"
        ids = []
        for i in range(6):
            n = store.add_node(_node(f"tie {i}", f"content {i}"))
            ids.append(n.node_id)
            # Force an identical created_at across all rows.
            conn.execute("UPDATE nodes SET created_at = ? WHERE node_id = ?",
                         (now, n.node_id))
        conn.commit()

        page1 = store.list_nodes(limit=3, offset=0)
        page2 = store.list_nodes(limit=3, offset=3)
        seen = [n.node_id for n in page1] + [n.node_id for n in page2]
        assert len(seen) == len(set(seen)) == 6
        assert set(seen) == set(ids)
        # Deterministic across repeated calls.
        again = [n.node_id for n in store.list_nodes(limit=3, offset=0)]
        assert again == [n.node_id for n in page1]


# ── G8: propose_skills O(n) refactor stays correct + reports progress ───────


class TestProposeSkillsRefactor:
    def _seed_actions(self, store, n_sessions=2, steps_per_session=3):
        now = datetime.now(timezone.utc)
        for s in range(n_sessions):
            for i in range(steps_per_session):
                store.add_node(Node(
                    node_type=NodeType.ACTION,
                    label="i will sync branch " + str(i),
                    content="i will sync branch " + str(i),
                    source_id="x", source_type=SourceType.EXTRACTED,
                    confidence=1.0, project_key="p1", session_key=f"s{s}",
                    origin_runtime="claude-code", recorded_at=now,
                ))

    def test_propose_skills_correct_after_refactor(self, store):
        self._seed_actions(store, n_sessions=2, steps_per_session=3)
        summary = propose_skills(store, min_occurrences=2, min_sessions=2)
        assert summary["detected"] >= 1
        assert summary["created"] >= 1

    def test_propose_skills_idempotent_after_refactor(self, store):
        self._seed_actions(store, n_sessions=2, steps_per_session=3)
        propose_skills(store, min_occurrences=2, min_sessions=2)
        second = propose_skills(store, min_occurrences=2, min_sessions=2)
        assert second["created"] == 0

    def test_propose_skills_progress_callback(self, store):
        self._seed_actions(store, n_sessions=2, steps_per_session=3)
        lines = []
        propose_skills(store, min_occurrences=2, min_sessions=2, progress=lines.append)
        joined = " | ".join(lines)
        assert "patterns found" in joined
        assert "proposals written" in joined

    def test_cli_skills_propose_prints_progress(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = GraphStore(db_path=db_path)
            self._seed_actions(store, n_sessions=2, steps_per_session=3)
            store.close()
            runner = CliRunner()
            result = runner.invoke(main, [
                "skills", "propose", "--db", db_path,
                "--min-occurrences", "2", "--min-sessions", "2",
            ])
            assert result.exit_code == 0, result.output
            assert "patterns found" in result.output
            assert "proposals written" in result.output
        finally:
            try:
                os.unlink(db_path)
            except PermissionError:  # pragma: no cover
                pass


# ── G9: matching_proposals SQL prefilter ─────────────────────────────────────


class TestMatchingProposalsPrefilter:
    def test_list_draft_proposed_skills_excludes_non_draft_and_non_proposed(self, store):
        draft = _proposal_node(store, "proposed: sync fernweh branches",
                                ["sync fernweh branches"], "codex", "hash-draft")
        _proposal_node(store, "proposed: not drafted", ["step"], "codex",
                        "hash-nodraft", draft=False)
        _proposal_node(store, "proposed: accepted already", ["step"], "codex",
                        "hash-active", status="active")
        rows = store.list_draft_proposed_skills(limit=1000)
        ids = {n.node_id for n in rows}
        assert draft.node_id in ids
        assert len(ids) == 1

    def test_matching_proposals_finds_needle_among_many_non_draft_skills(self, store):
        for i in range(50):
            _proposal_node(store, "proposed: unrelated " + str(i), ["step " + str(i)],
                            "codex", "hash-bulk-" + str(i), draft=False)
        needle = _proposal_node(store, "proposed: sync fernweh branches",
                                 ["sync fernweh branches"], "codex", "hash-needle")
        out = matching_proposals(store, "sync fernweh branches", None)
        ids = {p["node_id"] for p in out}
        assert needle.node_id in ids
        assert len(ids) == 1


# ── G10: capture-token scheme comparison is case-insensitive ────────────────


class TestAuthSchemeCaseInsensitive:
    def test_lowercase_bearer_scheme_accepted(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        check_capture_auth("100.64.0.7", "bearer s3cret")  # must not raise

    def test_mixed_case_bearer_scheme_accepted(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        check_capture_auth("100.64.0.7", "BeArEr s3cret")  # must not raise

    def test_wrong_token_still_401_regardless_of_scheme_case(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        with pytest.raises(HTTPException) as e:
            check_capture_auth("100.64.0.7", "bearer wrong")
        assert e.value.status_code == 401

    def test_non_bearer_scheme_still_401(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        with pytest.raises(HTTPException) as e:
            check_capture_auth("100.64.0.7", "Basic s3cret")
        assert e.value.status_code == 401

    def test_existing_trailing_space_case_still_401(self, monkeypatch):
        """Pin: G10 must not accidentally make a padded token pass just
        because the scheme comparison went case-insensitive."""
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret")
        with pytest.raises(HTTPException) as e:
            check_capture_auth("100.64.0.7", "bearer s3cret ")
        assert e.value.status_code == 401
