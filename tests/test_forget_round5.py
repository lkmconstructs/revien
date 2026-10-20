"""The owner's ruling on the copy rule (option 1): a capture that is only
recognised as a copy of a forgotten source by its CONTENT (no id of its own,
and no path or inode that matches) loses nothing but the sentences that held
what was forgotten. An identity match (id, real path, inode) is unchanged.

Every test here fails at aab8974 unless it says it is a lock."""
import json
from pathlib import Path

import pytest

from revien.forget import gate
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests import forget_seed as fs

README = [
    "This tool keeps a log of every delivery that arrives at the depot.",
    "Each morning the log is sorted by the time the driver checked in.",
    "Late deliveries are marked so that the manager can follow them up.",
    "The weekly summary is printed on Friday afternoon for the office.",
    "Old entries are archived once they are more than ninety days old.",
    "Questions about the log should go to the front desk first.",
]


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "r5.db"))
    yield s
    s.close()


def put(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fs._age(path)
    return path


def capture(pipe, path, extra=None):
    text = path.read_text(encoding="utf-8")
    return pipe.ingest(IngestionInput(
        source_id=f"file:{path.name}", origin_runtime="file", origin_source="watch",
        content=text, metadata={"path": str(path)}))


def stored_text(store):
    return "\n".join(n.content for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=100)
                     if not n.node_id.startswith("tombstone:"))


def context_ids(store):
    return [n.node_id for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=100)]


class TestAcceptance:
    def test_1_a_readme_copy_keeps_every_readme_line_and_its_own_lines_but_not_the_secret(self, store, tmp_path):
        secret = fs.new_secret()
        secret_line = f"Remember that my safe code is {secret}."
        a = put(tmp_path, "a.md", README + [secret_line])
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        ctx = context_ids(store)
        GraphOperations(store).forget_node(ctx[0], cascade=True)
        assert secret not in stored_text(store)
        b = put(tmp_path, "elsewhere-b.md", README + [secret_line,
                                                      "The printer on the second floor needs new toner.",
                                                      "Visitors sign the book at the entrance."])
        out = capture(pipe, b)
        text = stored_text(store)
        for line in README:
            assert line in text, line
        assert "printer on the second floor" in text and "Visitors sign the book" in text
        assert secret not in text
        assert out.left_out["matched_by"] == "content" and out.left_out["left_out"] == 1

    def test_2_a_node_level_forget_then_a_renamed_copy_strips_only_the_sentence_with_that_wording(self, store, tmp_path):
        secret = fs.new_secret()
        keeps = f"Remember that my safe code is {secret}."          # (no fact was made of this line)
        wording = f"We decided to use {secret} as the project codename."
        a = put(tmp_path, "a2.md", README + [keeps, wording])
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        fact = [n for n in store.list_nodes(node_type=NodeType.FACT, limit=20) if secret in n.content][0]
        assert "project codename" in fact.content
        GraphOperations(store).forget_node(fact.node_id)
        b = put(tmp_path, "renamed-a2.md", README + [keeps, wording, "And one more line of its own."])
        out = capture(pipe, b)
        text = stored_text(store)
        copy = [n.content for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=20)
                if "one more line of its own" in n.content][0]
        assert "project codename" not in copy                       # the sentence carrying that fact's wording
        assert keeps in copy                                        # every other sentence lands
        for line in README:
            assert line in copy
        assert out.left_out["matched_by"] == "content" and out.left_out["left_out"] == 1

    def test_3_a_cascade_then_a_renamed_copy_strips_what_held_the_forgotten_nodes_and_nothing_else(self, store, tmp_path):
        secret = fs.new_secret()
        shared = "We decided to use postgres for the billing service."
        boiler = "Let me run the tests to verify the change."
        plain = "The weather in Lisbon was lovely during the whole weekend."
        lines_a = [f"Remember that my safe code is {secret}.",
                   f"We decided to use {secret} as the project codename.",
                   plain, boiler, shared]
        pipe = IngestionPipeline(store)
        # another live source that already holds the shared decision
        pipe.ingest(IngestionInput(source_id="chatgpt:conversation:C", ingest_key="chatgpt:conversation:C",
                                   origin_runtime="chatgpt", session_key="C",
                                   content=f"User: {shared}\nUser: Something else entirely about parking permits."))
        a = put(tmp_path, "a3.md", lines_a)
        capture(pipe, a)
        a_ctx = [n.node_id for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=20)
                 if secret in n.content][0]
        GraphOperations(store).forget_node(a_ctx, cascade=True)
        b = put(tmp_path, "copy-of-a3.txt", lines_a + ["A line only the copy has, about the quarterly budget."])
        out = capture(pipe, b)
        text = stored_text(store)
        assert secret not in text                                  # the sentences that held the forgotten nodes
        for kept in (plain, boiler, shared, "quarterly budget"):
            assert kept in text, kept
        assert out.left_out["matched_by"] == "content"

    def test_3b_wording_that_another_live_unit_holds_in_its_own_record_is_not_held(self, store, tmp_path):
        shared = "We chose the blue pelican route for every night delivery."
        a_lines = README[:4] + [shared]
        store.add_node(Node(node_type=NodeType.CONTEXT, label="A", content="\n".join(a_lines)),
                       owner=("session:claude-code:HA", 1, 1))
        a_ctx = store.list_nodes(node_type=NodeType.CONTEXT, limit=5)[0].node_id
        store.add_node(Node(node_type=NodeType.FACT, label="route", content="chose the blue pelican route for every night delivery"),
                       owner=("session:claude-code:HA", 1, 1))
        # another live unit says the same thing in its own record (no node of it)
        store.add_node(Node(node_type=NodeType.CONTEXT, label="C",
                            content=f"{shared}\nAnd some other business of its own, entirely."),
                       owner=("session:claude-code:HC", 1, 1))
        GraphOperations(store).forget_node(a_ctx, cascade=True)
        pipe = IngestionPipeline(store)
        out = capture(pipe, put(tmp_path, "copy3b.md", a_lines + ["A line only the copy has."]))
        assert out.left_out is None                                 # nothing was held
        assert shared in stored_text(store)

    def test_4_boilerplate_and_templates_are_never_stripped_from_a_content_matched_capture(self, store, tmp_path):
        boiler = ["Let me run the tests to verify the change.",
                  "Here is the standard template for the weekly report.",
                  "Thanks, that looks good to me, please go ahead."]
        secret = fs.new_secret()
        a = put(tmp_path, "a4.md", README[:3] + boiler + [f"Remember that my safe code is {secret}."])
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        GraphOperations(store).forget_node(context_ids(store)[0], cascade=True)
        b = put(tmp_path, "b4-copy.md", README[:3] + boiler + [f"Remember that my safe code is {secret}.", "Own line."])
        capture(pipe, b)
        text = stored_text(store)
        for line in boiler + README[:3]:
            assert line in text, line
        assert secret not in text

    def test_an_identical_renamed_copy_is_not_refused_whole(self, store, tmp_path):
        secret = fs.new_secret()
        lines = README + [f"Remember that my safe code is {secret}."]
        a = put(tmp_path, "a8.md", lines)
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        GraphOperations(store).forget_node(context_ids(store)[0], cascade=True)
        out = capture(pipe, put(tmp_path, "a8-renamed.md", lines))
        assert out.suppressed is False
        text = stored_text(store)
        for line in README:
            assert line in text
        assert secret not in text

    def test_a_secret_shaped_token_another_live_unit_also_uses_is_not_held(self, store, tmp_path):
        code = "BUILD20261020X"
        a = put(tmp_path, "a9.md", README + [f"The nightly build is tagged {code} for the release."])
        pipe = IngestionPipeline(store)
        pipe.ingest(IngestionInput(source_id="chatgpt:conversation:Z", ingest_key="chatgpt:conversation:Z",
                                   origin_runtime="chatgpt", session_key="Z",
                                   content=f"User: Our other project is also tagged {code} on Fridays."))
        capture(pipe, a)
        a_ctx = [n.node_id for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=20) if "nightly build" in n.content][0]
        GraphOperations(store).forget_node(a_ctx, cascade=True)
        out = capture(pipe, put(tmp_path, "a9-copy.md", README + [f"The nightly build is tagged {code} for the release.", "Own."]))
        assert out.left_out is None                                # nothing held: the token is common to a live unit
        assert code in stored_text(store)

    def test_an_identity_match_is_unchanged_and_loses_every_forgotten_sentence(self, store, tmp_path):
        """Lock: the same file (same path), changed since the forget: every
        sentence that was forgotten from it stays out, README lines too."""
        secret = fs.new_secret()
        a = put(tmp_path, "same.md", README + [f"Remember that my safe code is {secret}."])
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        GraphOperations(store).forget_node(context_ids(store)[0], cascade=True)
        a.write_text(a.read_text() + "A brand new line appended later.\n", encoding="utf-8")
        fs._age(a)
        out = capture(pipe, a)
        text = stored_text(store)
        assert "A brand new line appended later." in text
        assert secret not in text and README[0] not in text
        assert out.left_out["matched_by"] == "identity"

    def test_the_response_names_what_was_left_out_without_saying_it(self, store, tmp_path):
        secret = fs.new_secret()
        a = put(tmp_path, "a6.md", README + [f"Remember that my safe code is {secret}."])
        pipe = IngestionPipeline(store)
        capture(pipe, a)
        GraphOperations(store).forget_node(context_ids(store)[0], cascade=True)
        b = put(tmp_path, "b6-copy.md", README + [f"Remember that my safe code is {secret}.", "Own line."])
        out = capture(pipe, b)
        note = out.left_out_note
        assert "recognised by its content and not by an identity" in note
        assert "The rest of it was kept" in note
        assert "line 7" in note                                   # where, not what
        assert secret not in note and secret not in json.dumps(out.left_out)
        assert out.left_out["lines"] == [7] and out.left_out["sources"]

    def test_the_rest_api_and_cli_carry_the_note(self, tmp_path):
        from click.testing import CliRunner
        from fastapi.testclient import TestClient

        from revien.daemon.server import create_app
        db = str(tmp_path / "rest5.db")
        secret = fs.new_secret()
        a = put(tmp_path, "a7.md", README + [f"Remember that my safe code is {secret}."])
        s = GraphStore(db_path=db)
        capture(IngestionPipeline(s), a)
        GraphOperations(s).forget_node(context_ids(s)[0], cascade=True)
        s.close()
        b = put(tmp_path, "b7-copy.md", README + [f"Remember that my safe code is {secret}.", "Own line."])
        with TestClient(create_app(db_path=db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "file:b7", "origin_runtime": "file", "content": b.read_text(),
                "metadata": {"path": str(b)}})
        body = r.json()
        assert body["left_out"]["matched_by"] == "content"
        assert "recognised by its content and not by an identity" in body["left_out_note"]


class TestTheCopyRuleAtItsThresholds:
    """The numbers of the rule, at their edges (revien/forget/gate.py)."""

    def sig(self, n):
        return {f"s{i}" for i in range(n)}

    def test_four_fifths_of_the_origin(self):
        assert gate.is_copy_of(self.sig(5), {"s0", "s1", "s2", "s3"})                   # 4 of 5: 80%
        assert not gate.is_copy_of(self.sig(6), {"s0", "s1", "s2", "s3"})              # 4 of 6: 67%
        assert gate.is_copy_of(self.sig(10), {f"s{i}" for i in range(8)})              # 8 of 10
        assert not gate.is_copy_of(self.sig(10), {f"s{i}" for i in range(7)})          # 7 of 10

    def test_half_of_the_capture(self):
        cap_ok = self.sig(4) | {"n1", "n2", "n3", "n4"}          # 4 of 8: 50%
        cap_no = self.sig(4) | {"n1", "n2", "n3", "n4", "n5"}    # 4 of 9
        assert gate.is_copy_of(self.sig(4), cap_ok)
        assert not gate.is_copy_of(self.sig(4), cap_no)

    def test_at_least_two_sentences(self):
        assert gate.is_copy_of(self.sig(2), self.sig(2))
        assert not gate.is_copy_of(self.sig(1), self.sig(1))
        assert not gate.is_copy_of(set(), {"x"}) and not gate.is_copy_of({"x", "y"}, set())

    def test_a_pipeline_capture_at_the_edge_of_the_rule(self, store, tmp_path):
        """A has 5 long sentences. A copy carrying 4 of them (80%) is a copy:
        the held one is cut, the rest lands. A copy carrying 3 is not."""
        secret = fs.new_secret()
        lines = README[:4] + [f"Remember that my safe code is {secret}."]
        pipe = IngestionPipeline(store)
        capture(pipe, put(tmp_path, "edge.md", lines))
        GraphOperations(store).forget_node(context_ids(store)[0], cascade=True)
        # carries 3 of A's 5 (60%): not a copy, nothing is cut
        capture(pipe, put(tmp_path, "edge-3.md", README[:2] + [f"Remember that my safe code is {secret}."]
                          + ["Another line of its own, number one.", "Another line of its own, number two."]))
        assert secret in stored_text(store)
        # carries 4 of A's 5 (80%): a copy, the held sentence is cut, the rest lands
        capture(pipe, put(tmp_path, "edge-4.md", README[:3] + [f"Remember that my safe code is {secret}."] + ["Own."]))
        copies = [n.content for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=20)
                  if README[2] in n.content and "Own." in n.content]
        assert copies and secret not in copies[0]


class TestWordingOfTheSpecAndChangelog:
    def test_the_sentence_that_depended_on_the_ruling_is_final(self):
        log = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
        spec = (Path(__file__).resolve().parents[1] / "docs" / "FORGET_SPEC.md").read_text(encoding="utf-8")
        for text in (log, spec):
            assert "REVISE" not in text
            assert "loses nothing but the sentences that held what was forgotten" in text
            assert "by content and not by an identity" in text
