"""An extension's node and edge types through every surface that parses or
shows a type: the store, export/import, REST, MCP, the CLI, TOON, recall,
forget, delete and the maintenance passes.

The toy extension is the one in tests/ext_support.py: a "recipe" node type, a
"meal_plan" node type and a "pairs_with" edge type. Each surface is loaded the
way a real install loads it: REVIEN_EXTENSIONS names the function, the store
opens, and nothing in the test registers a type by hand except where the
test is about the registry itself.
"""
import asyncio
import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien import extensions
from revien.cli import main
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore, ImportValidationError
from revien.retrieval.engine import RetrievalEngine
from revien.toon import parse_recall, serialize_recall
from tests.ext_support import (  # noqa: F401  (clean_registry is autouse)
    clean_registry, make_ext, raw_types,
)

EXT = '''
from revien.extensions import register_edge_type, register_node_type


def register():
    register_node_type("recipe", description="a dish and how to make it")
    register_node_type("meal_plan", description="what is cooked on which day", dedup=False)
    register_edge_type("pairs_with", description="two dishes that suit each other")
'''

SOUP = "Simmer ripe tomatoes with basil and cream for twenty minutes."


@pytest.fixture
def ext_spec(monkeypatch, make_ext):
    """Configure the toy extension the way an install does."""
    spec = f"{make_ext(EXT)}:register"
    monkeypatch.setenv(extensions.ENV_VAR, spec)
    return spec


def seed(db):
    """A database holding a recipe, a meal plan, a fact and the edges between
    them, written with the extension loaded."""
    extensions.load_extensions()
    s = GraphStore(db)
    soup = s.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup", content=SOUP))
    plan = s.add_node(Node(node_type=NodeType("meal_plan"), label="Week one",
                           content="Monday soup, Tuesday bread."))
    bread = s.add_node(Node(node_type=NodeType.FACT, label="Sourdough bread",
                            content="Sourdough bread keeps for three days."))
    pairs = s.add_edge(Edge(edge_type=EdgeType("pairs_with"), source_node_id=soup.node_id,
                            target_node_id=bread.node_id))
    s.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=plan.node_id,
                    target_node_id=soup.node_id))
    s.close()
    return soup, plan, bread, pairs


def unregister():
    """What a process without the extension looks like."""
    extensions._reset_for_tests()


# ── store ─────────────────────────────────────────────────────────────────


class TestStore:
    def test_an_extension_node_and_edge_round_trip_with_their_values(self, ext_spec, tmp_path):
        db = str(tmp_path / "s.db")
        soup, plan, bread, pairs = seed(db)
        s = GraphStore(db)
        try:
            got = s.get_node(soup.node_id)
            assert got.node_type is NodeType("recipe") and got.node_type.value == "recipe"
            assert got.content == SOUP
            edge = [e for e in s.get_edges_for_node(soup.node_id)
                    if e.edge_type.value == "pairs_with"][0]
            assert edge.edge_type is EdgeType("pairs_with")
            assert edge.edge_id == pairs.edge_id
        finally:
            s.close()
        assert raw_types(db) == {"nodes": ["fact", "meal_plan", "recipe"],
                                 "edges": ["pairs_with", "related_to"]}

    def test_listing_filters_and_bulk_reads(self, ext_spec, tmp_path):
        db = str(tmp_path / "s.db")
        soup, plan, bread, _ = seed(db)
        s = GraphStore(db)
        try:
            assert [n.node_id for n in s.list_nodes(node_type=NodeType("recipe"))] == [soup.node_id]
            assert [n.node_id for n in s.list_nodes(node_type=NodeType("meal_plan"))] == [plan.node_id]
            assert len(s.list_nodes()) == 3
            bulk = s.get_nodes_bulk([soup.node_id, plan.node_id, bread.node_id])
            assert {k: v.node_type.value for k, v in bulk.items()} == {
                soup.node_id: "recipe", plan.node_id: "meal_plan", bread.node_id: "fact"}
            hits = s.search_nodes_keyword(["tomatoes"])
            assert [h.node_type.value for h in hits] == ["recipe"]
            assert set(s.get_neighbors(soup.node_id)) == {bread.node_id, plan.node_id}
        finally:
            s.close()

    def test_update_and_delete_keep_the_type(self, ext_spec, tmp_path):
        db = str(tmp_path / "s.db")
        soup, *_ = seed(db)
        s = GraphStore(db)
        try:
            s.update_node(soup.node_id, content="Simmer for half an hour.")
            assert s.get_node(soup.node_id).node_type.value == "recipe"
            assert raw_types(db)["nodes"].count("recipe") == 1
            assert s.delete_node(soup.node_id) is True
            assert s.get_node(soup.node_id) is None
            assert [e.edge_type.value for e in s.get_edges_for_node(soup.node_id)] == []
        finally:
            s.close()


# ── export / import ───────────────────────────────────────────────────────


class TestExportImport:
    def exported(self, tmp_path):
        db = str(tmp_path / "src.db")
        seed(db)
        s = GraphStore(db)
        try:
            return json.loads(s.export_graph().model_dump_json())
        finally:
            s.close()

    def test_the_export_carries_the_values(self, ext_spec, tmp_path):
        data = self.exported(tmp_path)
        assert sorted(n["node_type"] for n in data["nodes"]) == ["fact", "meal_plan", "recipe"]
        assert sorted(e["edge_type"] for e in data["edges"]) == ["pairs_with", "related_to"]

    @pytest.mark.parametrize("mode", ["refuse", "merge", "replace"])
    def test_import_graph_keeps_every_type_in_every_mode(self, ext_spec, tmp_path, mode):
        data = self.exported(tmp_path)
        graph = Graph.model_validate(data)
        dst = str(tmp_path / f"dst-{mode}.db")
        s = GraphStore(dst)
        try:
            if mode != "refuse":
                s.add_node(Node(node_type=NodeType.FACT, label="already here", content="x"))
            result = s.import_graph(graph, mode=mode)
            assert result["nodes_added"] == 3 and result["edges_added"] == 2
            kinds = sorted(n.node_type.value for n in s.list_nodes())
            assert kinds == (["fact", "fact", "meal_plan", "recipe"] if mode == "merge"
                             else ["fact", "meal_plan", "recipe"])
        finally:
            s.close()
        assert sorted(raw_types(dst)["edges"]) == ["pairs_with", "related_to"]

    def test_a_second_export_of_the_import_is_the_same_graph(self, ext_spec, tmp_path):
        data = self.exported(tmp_path)
        dst = str(tmp_path / "dst.db")
        s = GraphStore(dst)
        s.import_graph(Graph.model_validate(data))
        again = json.loads(s.export_graph().model_dump_json())
        s.close()

        def key(d):
            return (sorted((n["node_id"], n["node_type"], n["label"], n["content"])
                           for n in d["nodes"]),
                    sorted((e["edge_id"], e["edge_type"], e["source_node_id"],
                            e["target_node_id"]) for e in d["edges"]))

        assert key(again) == key(data)

    def test_without_the_extension_the_file_does_not_validate_and_says_why(
            self, ext_spec, tmp_path, monkeypatch):
        data = self.exported(tmp_path)
        monkeypatch.delenv(extensions.ENV_VAR)
        unregister()
        with pytest.raises(Exception):
            Graph.model_validate(data)
        hint = extensions.describe_unregistered(data)
        assert "'recipe'" in hint and "'pairs_with'" in hint and "REVIEN_EXTENSIONS" in hint

    def test_import_graph_still_validates_ids(self, ext_spec, tmp_path):
        data = self.exported(tmp_path)
        data["edges"][0]["target_node_id"] = "missing"
        s = GraphStore(str(tmp_path / "dst.db"))
        try:
            with pytest.raises(ImportValidationError):
                s.import_graph(Graph.model_validate(data))
        finally:
            s.close()


# ── REST ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rest(ext_spec, tmp_path):
    db = str(tmp_path / "rest.db")
    soup, plan, bread, pairs = seed(db)
    unregister()                                   # the app must load it itself
    with TestClient(create_app(db_path=db)) as client:
        client.ids = (soup.node_id, plan.node_id, bread.node_id, pairs.edge_id)
        client.db = db
        yield client


class TestRest:
    def test_the_app_loads_the_extension_when_it_starts(self, rest):
        assert list(extensions.registered_node_types()) == ["recipe", "meal_plan"]

    def test_list_nodes_by_extension_type(self, rest):
        soup, plan, _, _ = rest.ids
        r = rest.get("/v1/nodes", params={"node_type": "recipe"})
        assert r.status_code == 200
        assert [(n["node_id"], n["node_type"]) for n in r.json()] == [(soup, "recipe")]
        r = rest.get("/v1/nodes", params={"node_type": "meal_plan"})
        assert [n["node_type"] for n in r.json()] == ["meal_plan"]
        assert sorted(n["node_type"] for n in rest.get("/v1/nodes").json()) == [
            "fact", "meal_plan", "recipe"]

    def test_an_unregistered_node_type_filter_is_still_a_400(self, rest):
        r = rest.get("/v1/nodes", params={"node_type": "bogus"})
        assert r.status_code == 400 and r.json()["detail"] == "Invalid node_type: bogus"

    def test_get_one_node_shows_its_type_and_its_edges(self, rest):
        soup, _, bread, _ = rest.ids
        r = rest.get(f"/v1/nodes/{soup}")
        assert r.status_code == 200
        body = r.json()
        assert body["node"]["node_type"] == "recipe"
        assert sorted(e["edge_type"] for e in body["edges"]) == ["pairs_with", "related_to"]
        assert {c["node_type"] for c in body["connected_nodes"]} == {"fact", "meal_plan"}

    def test_create_an_edge_of_an_extension_type(self, rest):
        soup, plan, _, _ = rest.ids
        r = rest.post("/v1/edges", json={
            "edge_type": "pairs_with", "source_node_id": plan, "target_node_id": soup})
        assert r.status_code == 200 and r.json()["edge_type"] == "pairs_with"
        assert raw_types(rest.db)["edges"].count("pairs_with") == 2

    def test_a_bad_edge_type_lists_the_valid_ones_extension_types_included(self, rest):
        soup, plan, _, _ = rest.ids
        r = rest.post("/v1/edges", json={
            "edge_type": "bogus", "source_node_id": plan, "target_node_id": soup})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert detail.startswith("Invalid edge_type: bogus. Valid: related_to, decided_in,")
        assert detail.endswith(", alias_of, pairs_with")

    def test_update_a_node_keeps_the_type(self, rest):
        soup = rest.ids[0]
        r = rest.put(f"/v1/nodes/{soup}", json={"content": "Simmer for half an hour."})
        assert r.status_code == 200 and r.json()["node_type"] == "recipe"

    def test_recall_returns_the_node_with_its_type(self, rest):
        r = rest.post("/v1/recall", json={"query": "ripe tomatoes basil"})
        assert r.status_code == 200
        types = {x["label"]: x["node_type"] for x in r.json()["results"]}
        assert types["Tomato soup"] == "recipe"

    def test_recall_in_toon_carries_the_type(self, rest):
        r = rest.post("/v1/recall", json={"query": "ripe tomatoes basil", "format": "toon"})
        assert r.status_code == 200
        parsed = parse_recall(r.text)
        assert {x["label"]: x["node_type"] for x in parsed["results"]}["Tomato soup"] == "recipe"

    def test_graph_export_and_import_over_rest(self, rest, tmp_path):
        exported = rest.get("/v1/graph").json()
        assert sorted(n["node_type"] for n in exported["nodes"]) == [
            "fact", "meal_plan", "recipe"]
        assert sorted(e["edge_type"] for e in exported["edges"]) == [
            "pairs_with", "related_to"]
        r = rest.post("/v1/graph/import", params={"mode": "replace"}, json=exported)
        assert r.status_code == 200 and r.json()["nodes"] == 3 and r.json()["edges"] == 2
        again = rest.get("/v1/graph").json()
        assert sorted(n["node_type"] for n in again["nodes"]) == [
            "fact", "meal_plan", "recipe"]

    def test_importing_a_type_nobody_registered_is_a_400_that_says_so(self, rest):
        payload = rest.get("/v1/graph").json()
        payload["nodes"][0]["node_type"] = "not_registered"
        r = rest.post("/v1/graph/import", params={"mode": "replace"}, json=payload)
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert detail.startswith("Invalid graph data: ")
        assert "node types 'not_registered' (1 node)" in detail
        assert "REVIEN_EXTENSIONS" in detail

    def test_the_full_export_and_forget_work_on_an_extension_node(self, rest):
        soup = rest.ids[0]
        full = rest.get("/v1/export").json()
        assert any(n["node_type"] == "recipe" for n in full["graph"]["nodes"])
        r = rest.post(f"/v1/nodes/{soup}/forget")
        assert r.status_code == 200 and r.json()["status"] == "forgotten"
        assert rest.get(f"/v1/nodes/{soup}").status_code == 404

    def test_delete_works_on_an_extension_node(self, rest):
        plan = rest.ids[1]
        assert rest.delete(f"/v1/nodes/{plan}").json()["status"] == "deleted"
        assert "meal_plan" not in raw_types(rest.db)["nodes"]

    def test_health_and_other_reads_are_unaffected(self, rest):
        h = rest.get("/v1/health").json()
        assert h["node_count"] == 3 and h["edge_count"] == 2
        assert rest.get("/v1/tensions").status_code == 200
        assert rest.get("/v1/communities").status_code == 200

    def test_starting_on_a_database_with_unregistered_types_fails_clearly(
            self, ext_spec, tmp_path, monkeypatch):
        db = str(tmp_path / "other.db")
        seed(db)
        monkeypatch.delenv(extensions.ENV_VAR)
        unregister()
        with pytest.raises(extensions.UnknownTypeError, match="'recipe'"):
            create_app(db_path=db)


# ── MCP ───────────────────────────────────────────────────────────────────


class TestMcp:
    @pytest.fixture(autouse=True)
    def need_mcp(self):
        pytest.importorskip("mcp", reason="mcp extra not installed")

    def call(self, server, name, arguments):
        res = asyncio.run(server.call_tool(name, arguments))
        content = res[0] if isinstance(res, tuple) else res
        return json.loads(content[0].text)

    def test_recall_through_the_mcp_server_returns_the_extension_type(self, ext_spec, tmp_path):
        from revien.mcp_server import build_mcp_server, close_mcp_server
        db = str(tmp_path / "mcp.db")
        seed(db)
        unregister()
        server = build_mcp_server(db_path=db)
        try:
            out = self.call(server, "revien_recall",
                            {"query": "ripe tomatoes basil", "top_n": 5})
            types = {r["label"]: r["node_type"] for r in out["results"]}
            assert types["Tomato soup"] == "recipe"
        finally:
            close_mcp_server(server)

    def test_the_mcp_server_refuses_a_database_with_unregistered_types(
            self, ext_spec, tmp_path, monkeypatch):
        from revien.mcp_server import build_mcp_server
        db = str(tmp_path / "mcp.db")
        seed(db)
        monkeypatch.delenv(extensions.ENV_VAR)
        unregister()
        with pytest.raises(extensions.UnknownTypeError, match="'meal_plan'|'recipe'"):
            build_mcp_server(db_path=db)

    def test_a_note_stored_over_mcp_is_unaffected(self, ext_spec, tmp_path):
        from revien.mcp_server import build_mcp_server, close_mcp_server
        server = build_mcp_server(db_path=str(tmp_path / "mcp2.db"))
        try:
            out = self.call(server, "revien_store", {
                "content": "We decided to use Postgres for billing."})
            assert out["nodes_created"] >= 1
        finally:
            close_mcp_server(server)


# ── CLI ───────────────────────────────────────────────────────────────────


@pytest.fixture
def cli_db(ext_spec, tmp_path):
    db = str(tmp_path / "cli.db")
    ids = seed(db)
    unregister()
    return db, ids


def run(*args, **kw):
    return CliRunner().invoke(main, list(args), **kw)


class TestCli:
    def test_export_then_import_keeps_the_types(self, cli_db, tmp_path):
        db, _ = cli_db
        out = str(tmp_path / "graph.json")
        res = run("export", out, "--db", db)
        assert res.exit_code == 0, res.output
        data = json.loads(open(out, encoding="utf-8").read())
        assert sorted(n["node_type"] for n in data["nodes"]) == ["fact", "meal_plan", "recipe"]
        unregister()
        dst = str(tmp_path / "dst.db")
        res = run("import", out, "--db", dst)
        assert res.exit_code == 0, res.output
        assert "Imported 3 nodes, 2 edges" in res.output
        assert raw_types(dst) == {"nodes": ["fact", "meal_plan", "recipe"],
                                  "edges": ["pairs_with", "related_to"]}

    def test_import_of_a_file_with_unregistered_types_says_what_to_register(
            self, cli_db, tmp_path, monkeypatch):
        db, _ = cli_db
        out = str(tmp_path / "graph.json")
        assert run("export", out, "--db", db).exit_code == 0
        monkeypatch.delenv(extensions.ENV_VAR)
        unregister()
        dst = str(tmp_path / "dst.db")
        res = run("import", out, "--db", dst)
        assert res.exit_code == 1
        assert "Invalid graph data" in res.output
        assert "node types 'meal_plan' (1 node), 'recipe' (1 node)" in res.output
        assert "edge types 'pairs_with' (1 edge)" in res.output
        assert "REVIEN_EXTENSIONS" in res.output
        assert not (tmp_path / "dst.db").exists() or raw_types(dst)["nodes"] == []

    def test_recall_in_every_output_shape(self, cli_db):
        db, _ = cli_db
        res = run("recall", "ripe tomatoes basil", "--db", db, "--json-output")
        assert res.exit_code == 0, res.output
        types = {r["label"]: r["node_type"] for r in json.loads(res.output)["results"]}
        assert types["Tomato soup"] == "recipe"
        res = run("recall", "ripe tomatoes basil", "--db", db, "--format", "toon")
        assert res.exit_code == 0, res.output
        parsed = parse_recall(res.output)
        assert {r["label"]: r["node_type"] for r in parsed["results"]}["Tomato soup"] == "recipe"
        res = run("recall", "ripe tomatoes basil", "--db", db)
        assert res.exit_code == 0 and "Type: recipe" in res.output

    def test_status_counts_and_other_read_commands_work(self, cli_db):
        db, _ = cli_db
        res = run("status", "--db", db)
        assert res.exit_code == 0 and "Nodes: 3" in res.output and "Edges: 2" in res.output
        assert run("tensions", "--db", db).exit_code == 0
        assert run("aliases", "--db", db).exit_code == 0

    def test_forget_and_delete_work_on_an_extension_node(self, cli_db):
        db, (soup, plan, bread, _) = cli_db
        res = run("forget", soup.node_id, "--db", db, "--yes")
        assert res.exit_code == 0, res.output
        assert "recipe" not in raw_types(db)["nodes"]
        res = run("delete", plan.node_id, "--db", db, "--yes")
        assert res.exit_code == 0, res.output
        assert raw_types(db)["nodes"] == ["fact"]

    def test_a_database_with_unregistered_types_is_a_clear_error(self, cli_db, monkeypatch):
        db, _ = cli_db
        monkeypatch.delenv(extensions.ENV_VAR)
        for args in (("status",), ("recall", "tomatoes"), ("export",), ("tensions",),
                     ("dream",), ("reindex",)):
            res = run(*args, "--db", db)
            assert res.exit_code == 1, (args, res.output)
            assert res.output.startswith("Error: Cannot open the database"), (args, res.output)
            assert "node types 'meal_plan' (1 node), 'recipe' (1 node)" in res.output
            assert "REVIEN_EXTENSIONS" in res.output
            assert "Traceback" not in res.output
            unregister()
        # and nothing was touched
        assert raw_types(db) == {"nodes": ["fact", "meal_plan", "recipe"],
                                 "edges": ["pairs_with", "related_to"]}

    def test_forget_and_delete_on_such_a_database_change_nothing(self, cli_db, monkeypatch):
        db, (soup, *_) = cli_db
        monkeypatch.delenv(extensions.ENV_VAR)
        for args in (("forget", soup.node_id, "--yes"), ("delete", soup.node_id, "--yes")):
            res = run(*args, "--db", db)
            assert res.exit_code == 1 and "Cannot open the database" in res.output, res.output
            unregister()
        assert raw_types(db)["nodes"] == ["fact", "meal_plan", "recipe"]


# ── TOON ──────────────────────────────────────────────────────────────────


class TestToon:
    def payload(self, node_type):
        row = {"node_id": "n1", "node_type": node_type, "label": "Tomato soup",
               "content": SOUP, "score": 0.5,
               "score_breakdown": {"recency": 0.1, "frequency": 0.2},
               "path": ["Tomato soup"], "origin_runtime": None,
               "origin_source": None, "project_key": None, "recorded_at": None,
               "recorded_at_source": None}
        return {"query": "q", "results": [row], "nodes_examined": 1,
                "retrieval_time_ms": 1.0, "semantic_active": False,
                "semantic_note": "", "skill_proposals": []}

    @pytest.mark.parametrize("value", ["recipe", "meal_plan", "a1_b2_c3"])
    def test_a_result_of_an_extension_type_round_trips(self, value):
        member = extensions.register_node_type(value)
        payload = self.payload(NodeType(value).value)
        assert type(payload["results"][0]["node_type"]) is str and member.value == value
        text = serialize_recall(payload)
        assert value in text
        assert parse_recall(text) == payload

    def test_a_real_recall_of_an_extension_node_round_trips(self, ext_spec, tmp_path):
        db = str(tmp_path / "t.db")
        seed(db)
        s = GraphStore(db)
        try:
            resp = RetrievalEngine(s).recall("ripe tomatoes basil", top_n=5)
        finally:
            s.close()
        payload = {
            "query": resp.query,
            "results": [{
                "node_id": r.node_id, "node_type": r.node_type, "label": r.label,
                "content": r.content, "score": r.score,
                "score_breakdown": r.score_breakdown, "path": r.path,
                "origin_runtime": r.origin_runtime, "origin_source": r.origin_source,
                "project_key": r.project_key, "recorded_at": r.recorded_at,
                "recorded_at_source": r.recorded_at_source} for r in resp.results],
            "nodes_examined": resp.nodes_examined,
            "retrieval_time_ms": resp.retrieval_time_ms,
            "semantic_active": resp.semantic_active, "semantic_note": resp.semantic_note,
            "skill_proposals": resp.skill_proposals,
        }
        assert "recipe" in {r["node_type"] for r in payload["results"]}
        assert parse_recall(serialize_recall(payload)) == payload


# ── recall, forget and the maintenance passes ────────────────────────────


@pytest.fixture
def store(ext_spec, tmp_path):
    extensions.load_extensions()
    s = GraphStore(str(tmp_path / "g.db"))
    yield s
    s.close()


class TestRecall:
    @pytest.mark.parametrize("ntype", ["recipe", "fact"])
    def test_an_extension_node_is_recalled_like_a_generic_claim(self, store, ntype):
        store.add_node(Node(node_type=NodeType(ntype), label="Tomato soup", content=SOUP))
        resp = RetrievalEngine(store).recall("ripe tomatoes basil", top_n=5)
        hit = [r for r in resp.results if r.label == "Tomato soup"]
        assert len(hit) == 1 and hit[0].node_type == ntype and hit[0].content == SOUP

    def test_it_scores_the_same_as_a_fact_of_the_same_text(self, store):
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP, source_id="same"))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="Tomato soups",
                                   content=SOUP, source_id="same"))
        resp = RetrievalEngine(store).recall("ripe tomatoes basil", top_n=5)
        score = {r.node_id: r.score for r in resp.results}
        assert score[soup.node_id] == pytest.approx(score[fact.node_id], rel=0.05)

    def test_the_per_type_weight_setting_names_it_like_any_other_type(self, store, monkeypatch):
        from revien.retrieval.scorer import ScoringConfig
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP))
        plain = RetrievalEngine(store).recall("ripe tomatoes basil", top_n=5)
        monkeypatch.setenv("REVIEN_TYPE_WEIGHTS", "recipe:0.5")
        weighted = RetrievalEngine(store, scoring_config=ScoringConfig.from_env()).recall(
            "ripe tomatoes basil", top_n=5)
        a = [r.score for r in plain.results if r.node_id == soup.node_id][0]
        b = [r.score for r in weighted.results if r.node_id == soup.node_id][0]
        assert b == pytest.approx(a * 0.5)

    def test_the_prefer_types_hint_names_it_like_any_other_type(self, store):
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="Tomato soups",
                                   content=SOUP))
        engine = RetrievalEngine(store)
        plain = {r.node_id: r.score for r in engine.recall("ripe tomatoes basil", top_n=5).results}
        hinted = {r.node_id: r.score for r in engine.recall(
            "ripe tomatoes basil", top_n=5, prefer_types=["recipe"]).results}
        assert hinted[soup.node_id] > plain[soup.node_id]
        assert hinted[fact.node_id] == pytest.approx(plain[fact.node_id])

    def test_the_per_type_recency_setting_names_it_like_any_other_type(self, store, monkeypatch):
        from datetime import datetime, timedelta, timezone
        from revien.retrieval.scorer import ScoringConfig
        old = datetime.now(timezone.utc) - timedelta(days=400)
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP, recorded_at=old, created_at=old))
        decaying = RetrievalEngine(store).recall("ripe tomatoes basil", top_n=5)
        monkeypatch.setenv("REVIEN_TYPE_HALF_LIFE_DAYS", "recipe:0")
        pinned = RetrievalEngine(store, scoring_config=ScoringConfig.from_env()).recall(
            "ripe tomatoes basil", top_n=5)
        a = [r.score_breakdown["recency"] for r in decaying.results if r.node_id == soup.node_id][0]
        b = [r.score_breakdown["recency"] for r in pinned.results if r.node_id == soup.node_id][0]
        assert b == pytest.approx(1.0) and a < 0.9

    def test_a_walk_crosses_an_extension_edge(self, store):
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP))
        bread = store.add_node(Node(node_type=NodeType.FACT, label="Sourdough",
                                    content="Keeps for three days."))
        store.add_edge(Edge(edge_type=EdgeType("pairs_with"), source_node_id=soup.node_id,
                            target_node_id=bread.node_id, weight=0.9))
        resp = RetrievalEngine(store).recall("ripe tomatoes basil", top_n=5)
        assert {r.label for r in resp.results} >= {"Tomato soup", "Sourdough"}


class TestForget:
    SECRET = "ZEBRA77"

    def build(self, store, ntype):
        owner = ("session:claude-code:A", 1, 1)
        ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="a",
                                  content=f"A: the code {self.SECRET} opens it"), owner=owner)
        target = store.add_node(Node(node_type=NodeType(ntype), label="code",
                                     content=f"the code {self.SECRET}"), owner=owner)
        return ctx, target

    @pytest.mark.parametrize("ntype", ["fact", "recipe", "meal_plan"])
    def test_forget_removes_the_node_and_cuts_its_wording_from_its_source(self, store, ntype):
        ctx, target = self.build(store, ntype)
        res = GraphOperations(store).forget_node(target.node_id)
        assert res["status"] == "forgotten"
        assert store.get_node(target.node_id) is None
        assert self.SECRET not in store.get_node(ctx.node_id).content
        assert store.list_suppressions(None)

    def test_the_result_has_the_same_shape_as_for_a_builtin_type(self, store, tmp_path):
        ctx, target = self.build(store, "recipe")
        ext = GraphOperations(store).forget_node(target.node_id)
        other = GraphStore(str(tmp_path / "o.db"))
        try:
            ctx2, target2 = self.build(other, "fact")
            base = GraphOperations(other).forget_node(target2.node_id)
        finally:
            other.close()
        assert set(ext) == set(base)
        for key in ("status", "removed_nodes", "nodes_before", "nodes_after",
                    "edges_before", "edges_after"):
            assert ext[key] == base[key], key

    def test_forgetting_a_wording_cuts_it_from_an_extension_node_as_from_a_builtin_one(
            self, store, tmp_path):
        results = {}
        for ntype in ("fact", "recipe"):
            g = GraphStore(str(tmp_path / f"w-{ntype}.db"))
            try:
                g.add_node(Node(node_type=NodeType(ntype), label="note",
                                content=f"keep {self.SECRET} out of it please"))
                res = GraphOperations(g).forget_wording(self.SECRET)
                left = g.list_nodes()
                assert [n.node_type.value for n in left] == [ntype]
                assert self.SECRET not in left[0].content + left[0].label
                results[ntype] = (res["status"], res["nodes_cut"], res["occurrences_cut"],
                                  res["removed_nodes"], left[0].content)
            finally:
                g.close()
        assert results["recipe"] == results["fact"]
        assert results["recipe"][0] == "forgotten" and results["recipe"][1] == 1

    def test_a_forgotten_node_meets_a_graph_import_as_a_builtin_one_does(self, store, tmp_path):
        """What the suppression gate does with an imported copy of a forgotten
        node depends on what the file proves about it, never on its type."""
        outcomes = {}
        for ntype in ("fact", "recipe"):
            g = GraphStore(str(tmp_path / f"{ntype}.db"))
            try:
                ctx, target = self.build(g, ntype)
                graph = g.export_graph()
                GraphOperations(g).forget_node(target.node_id)
                result = g.import_graph(graph, mode="merge")
                outcomes[ntype] = (
                    result["nodes_added"], result["skipped_suppressed"],
                    sorted(n.node_type.value for n in g.list_nodes()),
                    sorted(n.content for n in g.list_nodes()))
            finally:
                g.close()
        fact, recipe = outcomes["fact"], outcomes["recipe"]
        assert fact[:2] == recipe[:2]
        assert [t.replace("recipe", "fact") for t in recipe[2]] == fact[2]
        assert recipe[3] == fact[3]

    def test_delete_leaves_a_content_free_audit_entry(self, store):
        node = store.add_node(Node(node_type=NodeType("recipe"), label="Soup", content=SOUP))
        assert GraphOperations(store).delete_node(node.node_id) is True
        entries = [a for a in store.get_all_audit() if a["op"] == "delete"]
        assert len(entries) == 1 and SOUP not in json.dumps(entries[0])


class TestMaintenancePasses:
    def test_consolidation_reports_an_extension_orphan_with_its_type(self, store):
        from revien.consolidate import Consolidator
        node = store.add_node(Node(node_type=NodeType("recipe"), label="Lonely",
                                   content="No edges at all."))
        report = Consolidator(store, GraphOperations(store)).run(
            decay=False, recluster=False, reindex=False, invalidate_orphans=False,
            alias=True).to_dict()
        assert report["orphans"]["found"] == 1
        assert report["orphans"]["sample"] == [
            {"node_id": node.node_id, "node_type": "recipe", "label": "Lonely"}]
        assert report["alias"]["ran"] is True

    def test_the_alias_pass_runs_over_a_graph_with_extension_types_and_pairs_only_entities(
            self, store):
        from revien.alias import run_alias_pass
        e1 = store.add_node(Node(node_type=NodeType.ENTITY, label="Tomato Soup", content="e"))
        store.add_node(Node(node_type=NodeType.ENTITY, label="tomato soup", content="e"))
        a = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup", content="x"))
        b = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato Soup.", content="y"))
        store.add_edge(Edge(edge_type=EdgeType("pairs_with"), source_node_id=a.node_id,
                            target_node_id=e1.node_id))
        result = run_alias_pass(store, GraphOperations(store))
        assert result.ran
        aliased = {side["node_id"] for pair in store.list_alias_pairs(live_only=False)
                   for side in (pair["a"], pair["b"])}
        assert a.node_id not in aliased and b.node_id not in aliased
        assert store.get_node(a.node_id).node_type.value == "recipe"
        assert store.get_node(b.node_id).node_type.value == "recipe"

    def test_the_vault_note_of_an_entity_does_not_render_or_lose_extension_claims(
            self, store, tmp_path):
        from revien.distill import VaultDistiller
        entity = store.add_node(Node(node_type=NodeType.ENTITY, label="Tomato",
                                     content="Entity: Tomato"))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="Tomatoes ripen",
                                   content="Tomatoes ripen in late summer."))
        soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                                   content=SOUP))
        for other in (fact, soup):
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=entity.node_id,
                                target_node_id=other.node_id))
        vault = tmp_path / "vault"
        vault.mkdir()
        VaultDistiller(store, str(vault)).distill()
        texts = [p.read_text(encoding="utf-8") for p in (vault / "Revien").glob("*.md")]
        note = [t for t in texts if "# Tomato" in t]
        assert len(note) == 1
        assert "Tomatoes ripen" in note[0] and SOUP not in note[0]
        assert store.get_node(soup.node_id).node_type.value == "recipe"
