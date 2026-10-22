"""`forget --wording` cuts exactly the string the user named, and nothing less
and nothing made up: a word that merely overlaps it survives, two lines that
are not a wrapped line are not joined into a match, a string that is absent
cuts nothing and says so."""
import pytest

from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests.test_forget_wording import cc, forget_wording, leaks, new_key


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "x.db")


def _add(db, *texts):
    s = GraphStore(db_path=db)
    ids = [s.add_node(Node(node_type=NodeType.FACT, label=t[:60], content=t)).node_id
           for t in texts]
    s.close()
    return ids


def _texts(db):
    s = GraphStore(db_path=db)
    try:
        return sorted(n.content for n in s.list_nodes(limit=200))
    finally:
        s.close()


@pytest.mark.parametrize("named,innocent", [
    ("credentials-prod-7781", "rotate the credentials before the audit next week"),
    ("password123456", "reset the password for the shared mailbox on Monday"),
])
def test_a_string_that_is_absent_cuts_nothing_and_says_so(db, named, innocent):
    _add(db, innocent, "another unrelated note about the venue booking")
    before = _texts(db)
    res = forget_wording(db, named)
    assert res["status"] == "forgotten"
    assert res["occurrences_cut"] == 0 and res["nodes_cut"] == 0
    assert res["scope"] == "nothing found"
    assert res["message"].startswith("Nothing was found and nothing was changed")
    assert "gone everywhere" not in res["message"].lower()
    assert _texts(db) == before


def test_a_partial_overlap_survives_a_node_that_holds_the_whole_string_too(db):
    key = "credentials-prod-7781"
    _add(db, f"the deploy login was {key} last spring", "rotate the credentials soon",
         "the password rules changed")
    res = forget_wording(db, key)
    assert res["nodes_cut"] == 1
    left = _texts(db)
    assert "rotate the credentials soon" in left and "the password rules changed" in left
    assert not any(key in t for t in left)


@pytest.mark.parametrize("text", [
    "the word is token\n   abc and that is all",
    "first token\nabc second",
    "short\nabc",
])
def test_two_lines_that_are_not_one_wrapped_line_are_not_joined_into_a_match(text):
    assert scrub.cut_text(text, [scrub.Literal("tokenabc")]) == (text, 0)
    assert not scrub.contains(text, [scrub.Literal("tokenabc")])


def test_the_same_holds_at_capture_time(db):
    from revien.forget import gate
    forget_wording(db, "tokenabc")
    text = "User: We decided to read the channel every morning.\ntoken\n   abc"
    st = GraphStore(db_path=db)
    try:
        assert gate.strip_literals(st, text) == (text, 0)
        joined = "User: the secret is tokenabc, keep it close."
        assert gate.strip_literals(st, joined)[1] == 1
    finally:
        st.close()


def test_a_key_that_really_is_wrapped_is_still_cut_and_only_its_characters(db):
    key = new_key()
    half = len(key) // 2
    line = f"User: the key for the deploy pipeline is {key[:half]}"
    cut, n = scrub.cut_text(f"{line}\n  {key[half:]} and it is live.", [scrub.Literal(key)])
    assert n == 1
    assert cut == "User: the key for the deploy pipeline is  and it is live."
    cc(db, "s2", f"{line}\n  {key[half:]} and it is live.\nUser: The venue is booked.")
    res = forget_wording(db, key)
    assert res["occurrences_cut"] >= 1
    assert leaks(db, key) == {}
    assert any("The venue is booked." in t for t in _texts(db))
    # and the same wrapped key pasted again is kept out
    cc(db, "s3", f"{line}\n  {key[half:]} again.")
    assert leaks(db, key) == {}
