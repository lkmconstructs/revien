"""Round 16: a copy that runs to the cut edge is a truncation only when the
create row proves the capture that wrote it; every other record is reported."""
import sqlite3

import pytest

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import cc, forget_wording
from tests.test_forget_wording_ash2 import fake_key, pieces_reported

KEY = "staging-credentials-77x1"
LABEL = "rotate the staging-credentials"


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "r16.db")


def _node(db, node_id):
    s = GraphStore(db_path=db)
    try:
        return s.get_node(node_id)
    finally:
        s.close()


def _decision(db):
    s = GraphStore(db_path=db)
    try:
        return [n for n in s.list_nodes(limit=100) if n.node_type == NodeType.DECISION][0]
    finally:
        s.close()


def test_T3_a_sentence_without_a_period_is_not_a_truncation_once_its_context_is_gone(db):
    cc(db, "A", f"User: We decided to {LABEL}")
    cc(db, "B", f"User: We decided to rotate the {KEY} today.")
    s = GraphStore(db_path=db)
    ctx_a = [n for n in s.list_nodes(limit=100) if n.node_type == NodeType.CONTEXT
             and KEY not in n.content][0]
    s.delete_node(ctx_a.node_id)
    s.close()
    d = _decision(db)
    assert d.label == LABEL
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).label == LABEL
    assert _node(db, d.node_id).content == f"We decided to {LABEL}"      # no field is cut
    assert res["scope"] != "gone everywhere" and pieces_reported(res)


def test_T4_a_record_made_outside_a_capture_is_not_cut(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label=LABEL, content=LABEL))   # hand-added
    s.record_provenance(d.node_id, "unitB")                                      # B's capture deduped into it
    s.close()
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).label == LABEL and _node(db, d.node_id).content == LABEL
    assert res["scope"] != "gone everywhere" and pieces_reported(res)


def test_T5_a_record_whose_create_row_is_gone_is_not_cut(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label=LABEL, content=LABEL),
                   owner=("unitB", 1, 1))
    s.close()
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM audit_log WHERE node_id = ? AND op = 'create'", (d.node_id,))
    conn.commit()
    conn.close()
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).label == LABEL and _node(db, d.node_id).content == LABEL
    assert res["scope"] != "gone everywhere"


def test_T6_a_context_that_is_kept_and_ends_without_a_period_stays_reported(db):
    cc(db, "A", f"User: We decided to {LABEL}")
    cc(db, "B", f"User: We decided to rotate the {KEY} today.")
    d = _decision(db)
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).label == LABEL
    assert res["scope"] != "gone everywhere"


def test_a_capture_made_record_truncated_where_the_key_starts_is_cut(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    f = s.add_node(Node(node_type=NodeType.FACT, label=LABEL, content=LABEL),
                   owner=("unitB", 1, 1))                     # create row: capture:unitB
    s.close()
    forget_wording(db, KEY)
    assert not _node(db, f.node_id).label.endswith("credentials")


def test_a_field_of_the_same_node_that_continues_makes_the_stretch_ambiguous(db):
    s = GraphStore(db_path=db)
    n = s.add_node(Node(node_type=NodeType.FACT, label=LABEL,
                        content=f"We decided to rotate the {KEY} today",
                        source_context=f"{LABEL} monthly"))
    s.close()
    res = forget_wording(db, KEY)
    assert _node(db, n.node_id).label == LABEL
    assert res["scope"] != "gone everywhere"


def test_a_continuation_line_that_starts_with_the_tail_is_cut(db):
    key = fake_key(seed=71)
    head, tail = key[:20], key[20:]
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to use {head}\r\n\t{tail} as the deploy key."),
               owner=("unitB", 1, 1))
    f = s.add_node(Node(node_type=NodeType.FACT, label=f"{tail} as the deploy key",
                        content=f"{tail} as the deploy key"), owner=("unitB", 1, 1))
    s.close()
    forget_wording(db, key)
    cut = not _node(db, f.node_id).label.startswith(tail[:10])
    assert cut


def test_T3_with_a_content_that_is_not_the_label(db):
    """The content is itself a stretch of the other session's text: it is
    not cut either unless the create row proves the capture."""
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label="a rotation decision",
                        content=f"We decided to {LABEL}"), owner=("unitA", 1, 1))
    s.record_provenance(d.node_id, "unitB")
    s.close()
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).content == f"We decided to {LABEL}"
    assert res["scope"] != "gone everywhere"


def test_a_preview_counts_what_the_forget_cuts(db, tmp_path):
    """A record that holds the string itself is judged once, by the main cut:
    the preview and the forget agree on the count."""
    import shutil
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    s.add_node(Node(node_type=NodeType.DECISION, label=LABEL,
                    content=f"We decided to rotate the {KEY} today."), owner=("unitB", 1, 1))
    s.close()
    copy = str(tmp_path / "copy.db")
    shutil.copy(db, copy)
    pre = forget_wording(copy, KEY, dry_run=True)
    res = forget_wording(db, KEY)
    assert pre["would_cut_occurrences"] == res["occurrences_cut"], (pre, res["occurrences_cut"])


def test_T6_a_kept_context_without_a_period_blocks_the_cut_of_a_capture_made_record(db):
    s = GraphStore(db_path=db)
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session A",
                    content=f"User: We decided to {LABEL}"), owner=("unitA", 1, 1))
    s.add_node(Node(node_type=NodeType.CONTEXT, label="session B",
                    content=f"User: We decided to rotate the {KEY} today."),
               owner=("unitB", 1, 1))
    d = s.add_node(Node(node_type=NodeType.DECISION, label="a rotation decision",
                        content=f"We decided to {LABEL}"), owner=("unitB", 1, 1))
    s.record_provenance(d.node_id, "unitA")
    s.close()
    res = forget_wording(db, KEY)
    assert _node(db, d.node_id).content == f"We decided to {LABEL}"
    assert res["scope"] != "gone everywhere"
