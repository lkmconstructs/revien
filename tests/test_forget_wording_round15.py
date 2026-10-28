"""Round 15: the record's own words are evidence, so a stretch that stands
alone in them is not cut when the other session's text is gone."""
import pytest

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import forget_wording
from tests.test_forget_wording_verify2 import pieces_reported

KEY = "staging-credentials-77x1"
LABEL = "rotate the staging-credentials"


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r15.db")


def _label(db, node_id):
    s = GraphStore(db_path=db)
    try:
        return s.get_node(node_id).label
    finally:
        s.close()


def _sessions(db):
    s = GraphStore(db_path=db)
    a = s.add_node(Node(node_type=NodeType.CONTEXT, label="session A",
                        content=f"User: We decided to {LABEL}."), owner=("unitA", 1, 1))
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label=LABEL,
                        content=f"We decided to {LABEL}."), owner=("unitA", 1, 1))
    s.record_provenance(d.node_id, "unitB")
    return s, a, d


def test_V2_the_words_survive_when_the_other_sessions_record_was_deleted(db):
    s, a, d = _sessions(db)
    s.delete_node(a.node_id)                       # a delete, or a retention sweep
    s.close()
    res = forget_wording(db, KEY)
    assert _label(db, d.node_id) == LABEL
    assert res["scope"] != "gone everywhere"
    assert pieces_reported(res)


def test_V3_a_hand_added_fact_whose_content_continues_is_not_cut(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    f = s.add_node(Node(node_type=NodeType.FACT, label=LABEL,
                        content=f"{LABEL} monthly."), owner=("unitB", 1, 1))
    s.close()
    res = forget_wording(db, KEY)
    assert _label(db, f.node_id) == LABEL
    assert res["scope"] != "gone everywhere"


def test_a_fact_truncated_where_the_key_starts_is_still_cut(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    f = s.add_node(Node(node_type=NodeType.FACT, label=LABEL,
                        content=f"We decided to {LABEL}"), owner=("unitB", 1, 1))
    s.close()
    forget_wording(db, KEY)
    assert not _label(db, f.node_id).endswith("credentials")
