"""Behaviour hooks of the extension registry: leaf node types, asymmetric edge
types, node types whose metadata rides in recall, and exclude_node_types.

All sets are empty by default (stock behaviour); the type names here are
arbitrary examples."""
import pytest
from fastapi.testclient import TestClient

from revien import extensions
from revien.daemon.server import create_app
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.retrieval.walker import GraphWalker
from tests.ext_support import clean_registry, make_ext  # noqa: F401


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "hooks.db"))
    yield s
    s.close()


def _node(store, ntype, label, **kw):
    return store.add_node(Node(node_type=ntype, label=label,
                               content=f"about {label}", **kw))


def _edge(store, etype, a, b):
    return store.add_edge(Edge(edge_type=etype, source_node_id=a.node_id,
                               target_node_id=b.node_id, weight=0.8))


# ── registry ──────────────────────────────────────────────

def test_sets_are_empty_by_default():
    assert not extensions.LEAF_NODE_TYPES
    assert not extensions.ASYMMETRIC_EDGE_TYPES
    assert not extensions.RECALL_METADATA_NODE_TYPES


def test_register_functions_fill_the_sets():
    extensions.register_node_type("reference")
    extensions.register_edge_type("one_way")
    extensions.register_leaf_node_type("reference")
    extensions.register_leaf_node_type("reference")          # idempotent
    extensions.register_asymmetric_edge_type("one_way")
    extensions.register_recall_metadata_node_type("topic")   # built-in is fine
    assert extensions.LEAF_NODE_TYPES == {"reference"}
    assert extensions.ASYMMETRIC_EDGE_TYPES == {"one_way"}
    assert extensions.RECALL_METADATA_NODE_TYPES == {"topic"}


def test_register_unknown_type_is_refused():
    with pytest.raises(extensions.ExtensionRegistrationError):
        extensions.register_leaf_node_type("nope")
    with pytest.raises(extensions.ExtensionRegistrationError):
        extensions.register_asymmetric_edge_type("nope")
    with pytest.raises(extensions.ExtensionRegistrationError):
        extensions.register_recall_metadata_node_type("nope")
    assert not extensions.LEAF_NODE_TYPES


def test_env_vars_fill_the_sets(monkeypatch):
    monkeypatch.setenv("REVIEN_LEAF_NODE_TYPES", "topic, decision")
    monkeypatch.setenv("REVIEN_ASYMMETRIC_EDGE_TYPES", "depends_on")
    monkeypatch.setenv("REVIEN_RECALL_METADATA_NODE_TYPES", "entity")
    extensions.load_extensions()
    assert extensions.LEAF_NODE_TYPES == {"topic", "decision"}
    assert extensions.ASYMMETRIC_EDGE_TYPES == {"depends_on"}
    assert extensions.RECALL_METADATA_NODE_TYPES == {"entity"}


def test_env_naming_an_unknown_type_is_an_error(monkeypatch):
    monkeypatch.setenv("REVIEN_LEAF_NODE_TYPES", "nope")
    with pytest.raises(extensions.ExtensionRegistrationError, match="REVIEN_LEAF"):
        extensions.load_extensions()


def test_env_may_name_a_type_an_extension_registers(make_ext, monkeypatch):
    name = make_ext('''
        from revien.extensions import register_node_type
        def register():
            register_node_type("reference")
    ''')
    monkeypatch.setenv(extensions.ENV_VAR, f"{name}:register")
    monkeypatch.setenv("REVIEN_LEAF_NODE_TYPES", "reference")
    extensions.load_extensions()
    assert extensions.LEAF_NODE_TYPES == {"reference"}


def test_an_extension_can_register_the_sets(make_ext, monkeypatch):
    name = make_ext('''
        from revien import extensions as x
        def register():
            x.register_node_type("reference")
            x.register_edge_type("one_way")
            x.register_leaf_node_type("reference")
            x.register_asymmetric_edge_type("one_way")
            x.register_recall_metadata_node_type("reference")
    ''')
    monkeypatch.setenv(extensions.ENV_VAR, f"{name}:register")
    extensions.load_extensions()
    assert extensions.LEAF_NODE_TYPES == {"reference"}
    assert extensions.ASYMMETRIC_EDGE_TYPES == {"one_way"}
    assert extensions.RECALL_METADATA_NODE_TYPES == {"reference"}


# ── leaf node types ───────────────────────────────────────

def _chain(store, middle_type):
    a = _node(store, NodeType.ENTITY, "Alpha")
    m = _node(store, middle_type, "Middle")
    z = _node(store, NodeType.ENTITY, "Zulu")
    _edge(store, EdgeType.RELATED_TO, a, m)
    _edge(store, EdgeType.RELATED_TO, m, z)
    return a, m, z


def test_walk_expands_through_a_node_by_default(store):
    a, m, z = _chain(store, NodeType.TOPIC)
    dist = GraphWalker(store).walk([a.node_id])
    assert set(dist) == {a.node_id, m.node_id, z.node_id}


def test_leaf_node_is_reached_but_not_expanded(store):
    extensions.register_node_type("reference")
    extensions.register_leaf_node_type("reference")
    a, m, z = _chain(store, NodeType("reference"))
    dist = GraphWalker(store).walk([a.node_id])
    assert m.node_id in dist                 # reachable
    assert z.node_id not in dist             # but not walked through


def test_leaf_anchor_does_not_expand_either(store):
    extensions.register_node_type("reference")
    extensions.register_leaf_node_type("reference")
    a, m, z = _chain(store, NodeType("reference"))
    assert set(GraphWalker(store).walk([m.node_id])) == {m.node_id}


# ── asymmetric edge types ─────────────────────────────────

def test_asymmetric_edge_is_invisible_from_the_target_side(store):
    extensions.register_edge_type("one_way")
    extensions.register_asymmetric_edge_type("one_way")
    a = _node(store, NodeType.ENTITY, "Alpha")
    b = _node(store, NodeType.ENTITY, "Bravo")
    _edge(store, EdgeType("one_way"), a, b)
    assert store.get_neighbors(a.node_id) == [b.node_id]
    assert store.get_neighbors(b.node_id) == []
    assert store.get_neighbors_bulk([a.node_id, b.node_id]) == {
        a.node_id: [b.node_id], b.node_id: []}
    weighted = store.get_neighbors_weighted_bulk([a.node_id, b.node_id])
    assert [n for n, _ in weighted[a.node_id]] == [b.node_id]
    assert weighted[b.node_id] == []


def test_symmetric_edges_are_unchanged(store):
    extensions.register_edge_type("one_way")
    extensions.register_asymmetric_edge_type("one_way")
    a = _node(store, NodeType.ENTITY, "Alpha")
    b = _node(store, NodeType.ENTITY, "Bravo")
    _edge(store, EdgeType.RELATED_TO, a, b)
    assert store.get_neighbors(b.node_id) == [a.node_id]
    assert store.get_neighbors_bulk([b.node_id])[b.node_id] == [a.node_id]
    assert store.get_neighbors_weighted_bulk([b.node_id])[b.node_id][0][0] == a.node_id


def test_edge_is_two_way_when_not_declared(store):
    extensions.register_edge_type("one_way")
    a = _node(store, NodeType.ENTITY, "Alpha")
    b = _node(store, NodeType.ENTITY, "Bravo")
    _edge(store, EdgeType("one_way"), a, b)
    assert store.get_neighbors(b.node_id) == [a.node_id]


# ── the exact lane's neighbour read honours both hooks ───

def test_exact_neighbors_stock_reads_both_ways(store):
    extensions.register_edge_type("one_way")
    extensions.register_node_type("reference")
    a, m, z = _chain(store, NodeType("reference"))
    assert set(store.exact_neighbors([m.node_id])[m.node_id]) == {a.node_id, z.node_id}
    b = _node(store, NodeType.ENTITY, "Bravo")
    _edge(store, EdgeType("one_way"), a, b)
    assert store.exact_neighbors([b.node_id])[b.node_id] == [a.node_id]


def test_exact_neighbors_do_not_expand_a_leaf(store):
    extensions.register_node_type("reference")
    extensions.register_leaf_node_type("reference")
    a, m, z = _chain(store, NodeType("reference"))
    got = store.exact_neighbors([a.node_id, m.node_id])
    assert got[a.node_id] == [m.node_id]     # a leaf is still reachable
    assert got[m.node_id] == []              # but gives no neighbours


def test_exact_neighbors_follow_a_one_way_edge_one_way(store):
    extensions.register_edge_type("one_way")
    extensions.register_asymmetric_edge_type("one_way")
    a = _node(store, NodeType.ENTITY, "Alpha")
    b = _node(store, NodeType.ENTITY, "Bravo")
    _edge(store, EdgeType("one_way"), a, b)
    got = store.exact_neighbors([a.node_id, b.node_id])
    assert got[a.node_id] == [b.node_id]
    assert got[b.node_id] == []


# ── recall: metadata and exclude_node_types ───────────────

def _app(tmp_path):
    db = str(tmp_path / "wire.db")
    s = GraphStore(db_path=db)
    _node(s, NodeType.ENTITY, "Quasar", metadata={"k": "plain"})
    _node(s, NodeType("reference"), "Quasar reference", metadata={"k": "ref"})
    s.close()
    return create_app(db_path=db)


@pytest.fixture
def registered():
    extensions.register_node_type("reference")


def _recall(c, **body):
    r = c.post("/v1/recall", json={"query": "Quasar", "top_n": 10,
                                   "include_context": True, **body})
    return r


def test_metadata_is_absent_by_default(tmp_path, registered):
    with TestClient(_app(tmp_path)) as c:
        results = _recall(c).json()["results"]
        assert {r["node_type"] for r in results} == {"entity", "reference"}
        assert all("metadata" not in r for r in results)


def test_metadata_only_for_listed_types(tmp_path, registered):
    extensions.register_recall_metadata_node_type("reference")
    with TestClient(_app(tmp_path)) as c:
        by_type = {r["node_type"]: r for r in _recall(c).json()["results"]}
        assert by_type["reference"]["metadata"]["k"] == "ref"
        assert "metadata" not in by_type["entity"]


def test_exclude_node_types_drops_the_type(tmp_path, registered):
    with TestClient(_app(tmp_path)) as c:
        results = _recall(c, exclude_node_types=["reference"]).json()["results"]
        assert results and {r["node_type"] for r in results} == {"entity"}


def test_exclude_node_types_unknown_is_422(tmp_path, registered):
    with TestClient(_app(tmp_path)) as c:
        r = _recall(c, exclude_node_types=["nope"])
        assert r.status_code == 422
        assert "nope" in r.json()["detail"]


def test_engine_exclude_node_types(store):
    _node(store, NodeType.ENTITY, "Quasar")
    _node(store, NodeType.TOPIC, "Quasar topic")
    engine = RetrievalEngine(store)
    kinds = {r.node_type for r in engine.recall("Quasar", top_n=10).results}
    assert kinds == {"entity", "topic"}
    kinds = {r.node_type for r in
             engine.recall("Quasar", top_n=10, exclude_node_types=["topic"]).results}
    assert kinds == {"entity"}
    with pytest.raises(ValueError):
        engine.recall("Quasar", exclude_node_types=["nope"])


# ── leaf anchors: community boost and alias expansion ─────

def test_community_boost_ignores_leaf_anchor(tmp_path):
    """A leaf-type anchor must not lift personal nodes clustered with it."""
    from revien.graph.clustering import CommunityDetector
    extensions.register_node_type("reference")
    extensions.register_leaf_node_type("reference")
    db = str(tmp_path / "c.db")
    s = GraphStore(db_path=db)
    leaf = _node(s, NodeType("reference"), "Baseline Heron")
    x = _node(s, NodeType.ENTITY, "Secret Xylo")
    y = _node(s, NodeType.ENTITY, "Yonder")
    _edge(s, EdgeType.RELATED_TO, leaf, x)
    _edge(s, EdgeType.RELATED_TO, y, x)
    for i in range(4):
        _edge(s, EdgeType.RELATED_TO, y, _node(s, NodeType.ENTITY, f"Yfriend{i}"))
    cl = CommunityDetector(db_path=db)
    cl.run()
    eng = RetrievalEngine(s, clustering=cl)
    eng._find_anchors = (lambda q, sf=None: [leaf.node_id, y.node_id]
                         if "Heron" in q else [y.node_id])
    with_leaf = {r.node_id: r for r in eng.recall("Yonder and Heron", top_n=20).results}
    without = {r.node_id: r for r in eng.recall("Yonder", top_n=20).results}
    s.close()
    assert (with_leaf[x.node_id].score_breakdown["community_boost"]
            == without[x.node_id].score_breakdown["community_boost"])


def test_leaf_anchor_does_not_expand_aliases(store, monkeypatch):
    monkeypatch.setenv("REVIEN_ALIAS", "1")
    extensions.register_node_type("reference")
    extensions.register_leaf_node_type("reference")
    leaf = _node(store, NodeType("reference"), "Baseline Heron")
    ent = _node(store, NodeType.ENTITY, "Heron Alias")
    stock = _node(store, NodeType.ENTITY, "Quasar Station")
    other = _node(store, NodeType.ENTITY, "Quasar Alias")
    _edge(store, EdgeType.ALIAS_OF, leaf, ent)
    _edge(store, EdgeType.ALIAS_OF, stock, other)
    eng = RetrievalEngine(store)
    from revien.retrieval import engine as engine_mod
    assert engine_mod._alias_expansion_enabled_by_env()
    anchors = eng._find_anchors("Baseline Heron")
    assert leaf.node_id in anchors and ent.node_id not in anchors
    # Control: a non-leaf anchor still expands.
    anchors = eng._find_anchors("Quasar Station")
    assert other.node_id in anchors


# ── MCP / CLI / Hermes parity ─────────────────────────────

def test_recall_metadata_field_helper():
    extensions.register_node_type("reference")
    assert extensions.recall_metadata_field("reference", {"a": 1}) == {}
    extensions.register_recall_metadata_node_type("reference")
    assert extensions.recall_metadata_field("reference", {"a": 1}) == {"metadata": {"a": 1}}
    assert extensions.recall_metadata_field("entity", {"a": 1}) == {}


def test_mcp_recall_metadata_and_exclude(tmp_path, registered):
    from revien import mcp_server
    if not mcp_server.MCP_AVAILABLE:
        pytest.skip("mcp not installed")
    extensions.register_recall_metadata_node_type("reference")
    db = str(tmp_path / "m.db")
    s = GraphStore(db_path=db)
    _node(s, NodeType.ENTITY, "Quasar", metadata={"k": "plain"})
    _node(s, NodeType("reference"), "Quasar reference", metadata={"k": "ref"})
    s.close()
    srv = mcp_server.build_mcp_server(db_path=db)
    try:
        fn = srv._tool_manager._tools["revien_recall"].fn
        by = {r["node_type"]: r for r in fn(query="Quasar", top_n=10,
                                            include_context=True)["results"]}
        assert by["reference"]["metadata"]["k"] == "ref"
        assert "metadata" not in by["entity"]
        out = fn(query="Quasar", top_n=10, include_context=True,
                 exclude_node_types=["reference"])
        assert {r["node_type"] for r in out["results"]} == {"entity"}
        with pytest.raises(ValueError, match="Unknown node type in exclude_node_types"):
            fn(query="Quasar", exclude_node_types=["nope"])
    finally:
        mcp_server.close_mcp_server(srv)
