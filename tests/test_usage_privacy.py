"""Every way a node leaves the graph takes its usage rows with it: a judgment
holds the memory's own words (shared_terms) and a receipt holds its id.

Main's newer removal paths are covered too: the scoped LangChain clear and
the replace import, which wipes nodes with direct SQL.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Graph, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.usage import UsageObserver, UseVerdict

NOW = datetime.now(timezone.utc)


def _node(node_id, label, session=None):
    return Node(node_id=node_id, node_type=NodeType.FACT, label=label,
                content=label + ".", source_id="test", session_key=session,
                source_type=SourceType.EXTRACTED, confidence=1.0)


def _judge(store, node_id):
    assert store.record_use_judgment(
        episode_key=f"e#{node_id}", node_id=node_id, used=True, reason="r",
        shared_terms=["postgresql"], similarity=0.7, query="q", mode="shadow",
        applied=False)


def _receipt(store, rid, ids):
    store.add_recall_receipt(recall_id=rid, runtime="api", query="q",
                             node_ids=ids, issued_at=NOW,
                             expires_at=NOW + timedelta(hours=24))


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "revien.db"))
    for nid, label in (("a", "Billing uses PostgreSQL"), ("b", "Cache is Redis")):
        s.add_node(_node(nid, label))
        _judge(s, nid)
    _receipt(s, "r1", ["a", "b"])
    yield s
    s.close()


def _left(store):
    return ({r["node_id"] for r in store.list_use_judgments()},
            store.get_recall_receipt("r1")["node_ids"])


def test_delete_node_removes_judgments_and_scrubs_receipts(store):
    assert store.delete_node("a")
    assert _left(store) == ({"b"}, ["b"])


def test_forget_node_does_too(store):
    GraphOperations(store).forget_node("a", reason="test")
    assert _left(store) == ({"b"}, ["b"])


def test_rest_delete_does_too(tmp_path):
    path = str(tmp_path / "rest.db")
    s = GraphStore(db_path=path)
    s.add_node(_node("a", "Billing uses PostgreSQL"))
    _judge(s, "a")
    _receipt(s, "r1", ["a"])
    s.close()
    with TestClient(create_app(db_path=path)) as client:
        assert client.delete("/v1/nodes/a").status_code == 200
    s = GraphStore(db_path=path)
    try:
        assert s.list_use_judgments() == []
        assert s.get_recall_receipt("r1")["node_ids"] == []
    finally:
        s.close()


def test_scoped_langchain_clear_does_too(tmp_path):
    path = str(tmp_path / "lc.db")
    s = GraphStore(db_path=path)
    s.add_node(_node("mine", "Billing uses PostgreSQL", session="s1"))
    s.add_node(_node("theirs", "Cache is Redis", session="s2"))
    _judge(s, "mine")
    _judge(s, "theirs")
    _receipt(s, "r1", ["mine", "theirs"])
    s.close()
    import types

    from revien.adapters import langchain_adapter as LA

    st = GraphStore(path)
    mem = types.SimpleNamespace(session_scope="s1", _store=st)
    LA.RevienMemory.clear(mem)
    st.close()
    s = GraphStore(db_path=path)
    try:
        assert {r["node_id"] for r in s.list_use_judgments()} == {"theirs"}
        assert s.get_recall_receipt("r1")["node_ids"] == ["theirs"]
    finally:
        s.close()


def test_replace_import_wipes_judgments_and_receipt_ids(store):
    result = store.import_graph(Graph(nodes=[_node("c", "Fresh node")], edges=[]),
                                mode="replace")
    assert result["removed_nodes"] == 2
    assert store.list_use_judgments() == []
    assert store.get_recall_receipt("r1")["node_ids"] == []
    assert store.get_recall_receipt("r1")["recall_id"] == "r1"  # receipt kept


def test_merge_import_leaves_usage_rows_alone(store):
    store.import_graph(Graph(nodes=[_node("c", "Fresh node")], edges=[]),
                       mode="merge")
    assert _left(store) == ({"a", "b"}, ["a", "b"])


def test_observer_skips_a_removed_node_so_nothing_is_written_back(store):
    store.delete_node("a")
    result = UsageObserver(store, judge=lambda e: UseVerdict(used=True, reason="stand-in")).observe_answer(
        "r1", "Billing runs on PostgreSQL.")
    assert "a" not in {j.node_id for j in result.judgments}
    assert "a" not in {r["node_id"] for r in store.list_use_judgments()}
