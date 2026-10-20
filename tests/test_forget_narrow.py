"""Forget is for ONE specific thing (docs/FORGET_SPEC.md §1, §2, §6).

The owner's narrowing ruling, one test each: the wording of an entity is its
label (never the content string the extractor gives it); the denylist is bound
to the identity of the origin (a copy at a new path with the same id stays
out, a copy under a different identity lands, and the response says so);
`--everywhere` keeps that wording out of every source; a conversation is not
one thing to forget; a shared README is not touched by a forget in another
source.
"""
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.cli import main
from revien.daemon.server import create_app
from revien.forget import scrub
from revien.forget.engine import WHOLE_CONVERSATION, node_wording, stated
from revien.forget.keys import fingerprint
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests import forget_seed as fs

NAME = "Brigitte Vandermeer"
LINES = [
    ("user", f"{NAME} signed the contract on Friday."),
    ("user", "The venue is booked for the whole of June."),
    ("assistant", f"Noted, I will tell {NAME} about the schedule."),
]


def write_session(root: Path, project: str, stem: str, session_id: str, lines=LINES) -> Path:
    d = root / "projects" / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}.jsonl"
    path.write_text("".join(json.dumps({
        "type": role, "sessionId": session_id, "timestamp": "2026-07-01T10:00:00Z",
        "message": {"role": role, "content": [{"type": "text", "text": text}]}}) + "\n"
        for role, text in lines), encoding="utf-8")
    fs._age(path)
    return path


def sync(db: str, root: Path) -> None:
    fs._resync(db, "claude-code", ClaudeCodeAdapter(session_dir=str(root / "projects")))


def nodes(db: str):
    s = GraphStore(db_path=db)
    try:
        return s.list_nodes(limit=1000)
    finally:
        s.close()


def holders_of(db: str, text: str):
    return [n for n in nodes(db) if text.casefold() in (n.label + n.content).casefold()]


def contexts_holding(db: str, text: str):
    return [n for n in holders_of(db, text) if n.node_type == NodeType.CONTEXT]


def entity(db: str):
    return [n for n in nodes(db) if n.node_type == NodeType.ENTITY and n.label == NAME][0]


@pytest.fixture
def forgotten_name(tmp_path):
    """Brigitte Vandermeer, forgotten from Claude Code session S1."""
    root = tmp_path / "claude"
    db = str(tmp_path / "b.db")
    write_session(root, "proj-a", "s1", "uuid-S1")
    sync(db, root)
    assert contexts_holding(db, NAME)
    s = GraphStore(db_path=db)
    ent = [n for n in s.list_nodes(limit=100)
           if n.node_type == NodeType.ENTITY and n.label == NAME][0]
    res = GraphOperations(s).forget_node(ent.node_id)
    s.close()
    assert res["status"] == "forgotten", res
    return root, db, res


# ── the wording of a node is what was said, not what Revien made of it ─────

class TestTheWordingIsTheLabel:
    def test_an_entity_and_a_topic_are_their_label(self):
        ent = Node(node_type=NodeType.ENTITY, label=NAME, content=f"Entity: {NAME}")
        assert stated(ent) == [NAME] and node_wording(ent) == [NAME]
        topic = Node(node_type=NodeType.TOPIC, label="venue booking",
                     content="Topic: venue booking (mentioned 3 times)")
        assert stated(topic) == ["venue booking"]
        fact = Node(node_type=NodeType.FACT, label="booked", content="The venue is booked.")
        assert stated(fact) == ["The venue is booked."]
        assert node_wording(fact) == ["The venue is booked.", "booked"]

    def test_the_name_is_cut_from_the_conversation_and_the_content_string_is_not_the_wording(self, forgotten_name):
        _root, db, res = forgotten_name
        assert not holders_of(db, NAME)                         # every sentence that said it is gone
        assert "The venue is booked for the whole of June" in contexts_holding(db, "venue")[0].content
        s = GraphStore(db_path=db)
        try:
            (rec,) = s.list_suppressions("active")
        finally:
            s.close()
        assert fingerprint("Entity: " + NAME) not in rec["sentence_fps"] + rec["short_fps"] + rec["wording_fps"]
        assert fingerprint(NAME) in rec["wording_fps"] + rec["short_fps"] + rec["sentence_fps"]
        assert res["wording_cut"] and res["wording_cut"][0]["occurrences"] >= 2


# ── (1) a copy at a new path: the same identity stays out, another lands ──

class TestTheDenylistIsBoundToTheOrigin:
    def test_a_copy_at_a_new_path_with_the_same_session_id_stays_out(self, forgotten_name, tmp_path):
        root, db, _res = forgotten_name
        copy_root = tmp_path / "elsewhere"
        write_session(copy_root, "proj-z", "renamed-copy", "uuid-S1",
                      LINES + [("user", "A later remark about the parking permits.")])
        sync(db, copy_root)
        assert "parking permits" in " ".join(n.content for n in nodes(db))   # the rest is captured
        assert not holders_of(db, NAME)

    def test_a_copy_under_a_different_identity_lands_and_the_response_said_so(self, forgotten_name, tmp_path):
        root, db, res = forgotten_name
        said = " ".join(res["reentry"]["does_not_stop"])
        assert "a copy of this source under a different identity" in said
        assert "is not recognised" in said and "--everywhere" in said
        assert "different identity" in res["message"]
        other = tmp_path / "other"
        write_session(other, "proj-b", "s2", "uuid-S2")
        sync(db, other)
        landed = contexts_holding(db, NAME)
        assert len(landed) == 1 and NAME in landed[0].content

    def test_a_new_sentence_from_the_origin_that_says_the_name_stays_out(self, forgotten_name):
        root, db, _res = forgotten_name
        write_session(root, "proj-a", "s1", "uuid-S1",
                      LINES + [("user", f"Please call {NAME} again tomorrow morning."),
                               ("user", "The catering is confirmed for Friday.")])
        sync(db, root)
        assert not holders_of(db, NAME)
        assert "catering is confirmed" in " ".join(n.content for n in nodes(db))

    def test_a_facts_extractor_label_is_not_wording_a_later_sentence_may_not_say(self, tmp_path):
        """Forgetting "the API through Kong." must not keep out an unrelated
        sentence that only shares the extractor's label ("through Kong"); a
        new sentence with the whole wording is still left out."""
        root = tmp_path / "kong"
        db = str(tmp_path / "kong.db")
        base = [("user", "We decided to expose the API through Kong."),
                ("user", "The venue is booked for the whole of June.")]
        write_session(root, "p", "s1", "uuid-K", base)
        sync(db, root)
        fact = [n for n in nodes(db) if n.node_type == NodeType.FACT and n.content == "the API through Kong."]
        assert fact and fact[0].label == "through Kong"
        s = GraphStore(db_path=db)
        GraphOperations(s).forget_node(fact[0].node_id)
        s.close()
        write_session(root, "p", "s1", "uuid-K", base + [
            ("user", "Later we routed billing through Kong too."),
            ("user", "Someone asked about the API through Kong again.")])
        sync(db, root)
        body = " ".join(n.content for n in nodes(db) if n.node_type == NodeType.CONTEXT)
        assert "Later we routed billing through Kong too" in body        # unrelated: lands
        assert "the API through Kong" not in body                         # the wording: out
        assert "venue is booked" in body

    def test_the_name_inside_a_longer_word_is_not_the_name(self, tmp_path):
        root = tmp_path / "c"
        db = str(tmp_path / "w.db")
        write_session(root, "p", "s1", "uuid-W", [
            ("user", "The API is stable and the Api docs are done."),
            ("user", "The rapid rollout needs capital."),
            ("user", "Therapists agreed to the plan.")])
        sync(db, root)
        s = GraphStore(db_path=db)
        api = s.add_node(Node(node_type=NodeType.ENTITY, label="API", content="Entity: API"),
                         owner=("session:claude-code:uuid-W", 1, 1))
        GraphOperations(s).forget_node(api.node_id)
        s.close()
        write_session(root, "p", "s1", "uuid-W", [
            ("user", "The API is stable and the Api docs are done."),
            ("user", "The rapid rollout needs capital."),
            ("user", "Therapists agreed to the plan."),
            ("user", "A newer remark: the rapid pace suits the API users."),
            ("user", "The capital budget covers Therapists too.")])
        sync(db, root)
        body = " ".join(n.content for n in nodes(db) if n.node_type == NodeType.CONTEXT)
        assert "rapid rollout needs capital" in body and "Therapists agreed" in body
        assert "capital budget covers Therapists" in body
        assert "API is stable" not in body and "suits the API users" not in body


# ── (2) --everywhere on that wording: every source ───────────────────────

class TestEverywhereKeepsTheNameOutOfEverySource:
    def test_the_name_stays_out_of_every_source_and_every_holder_is_listed(self, tmp_path):
        root = tmp_path / "claude"
        db = str(tmp_path / "ev.db")
        write_session(root, "proj-a", "s1", "uuid-S1")
        write_session(root, "proj-b", "s2", "uuid-S2")
        sync(db, root)
        assert len(contexts_holding(db, NAME)) == 2
        s = GraphStore(db_path=db)
        hand = s.add_node(Node(node_type=NodeType.FACT, label="note",
                               content=f"{NAME} owns the guest list. The caterer is Delmonico."))
        ent = [n for n in s.list_nodes(limit=100)
               if n.node_type == NodeType.ENTITY and n.label == NAME]
        assert len(ent) == 1
        before = {n.node_id for n in s.list_nodes(limit=1000)}
        res = GraphOperations(s).forget_node(ent[0].node_id, everywhere=True)
        s.close()
        assert res["status"] == "forgotten" and res["everywhere"] is True
        assert not holders_of(db, NAME), "the name must be cut out of every holder"
        listed = {c["node_id"] for c in res["cut_in_place"]}
        ctx_ids = {n.node_id for n in nodes(db) if n.node_type == NodeType.CONTEXT}
        assert ctx_ids <= listed and hand.node_id in listed            # every holder, in place
        survivor = [n for n in nodes(db) if n.node_id == hand.node_id][0]
        assert "The caterer is Delmonico" in survivor.content            # in place, not cut back
        now = {n.node_id for n in nodes(db)}
        assert ent[0].node_id in before - now and hand.node_id in now and ctx_ids <= now
        # a source that was never seen, a source that grows, and a shared channel
        third = tmp_path / "third"
        write_session(third, "proj-c", "s3", "uuid-S3", LINES + [
            ("user", f"Ask {NAME} for the final guest list.")])
        sync(db, third)
        write_session(root, "proj-a", "s1", "uuid-S1", LINES + [
            ("user", f"{NAME} replied to the invitation.")])
        sync(db, root)
        from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
        st = GraphStore(db_path=db)
        IngestionPipeline(st).ingest(IngestionInput(
            source_id="app-notes", content=f"User: Remember that {NAME} prefers the window table."
                                           "\nUser: The wine list is final."))
        st.close()
        assert not holders_of(db, NAME)
        assert "wine list is final" in " ".join(n.content for n in nodes(db))

    def test_the_exact_wording_is_refused_whole_and_a_restore_lifts_it(self, tmp_path):
        root = tmp_path / "claude"
        db = str(tmp_path / "ev2.db")
        write_session(root, "proj-a", "s1", "uuid-S1")
        sync(db, root)
        s = GraphStore(db_path=db)
        ent = [n for n in s.list_nodes(limit=100)
               if n.node_type == NodeType.ENTITY and n.label == NAME][0]
        GraphOperations(s).forget_node(ent.node_id, everywhere=True)
        from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
        out = IngestionPipeline(s).ingest(IngestionInput(source_id="app-notes", content=NAME))
        assert out.suppressed is True and "revien restore forget:" in out.suppression_note
        rec = [r for r in s.list_suppressions("active") if r["scope"] == "global"][0]
        from revien.forget.restore import restore
        assert restore(s, rec["unit_id"])["status"] == "restored"
        out = IngestionPipeline(s).ingest(IngestionInput(source_id="app-notes", content=NAME))
        assert out.suppressed is False
        s.close()


# ── (3) a conversation is not one thing to forget ─────────────────────────

class TestAConversationIsRefused:
    def _seed(self, tmp_path):
        root = tmp_path / "claude"
        db = str(tmp_path / "ctx.db")
        write_session(root, "proj-a", "s1", "uuid-S1")
        sync(db, root)
        ctx = contexts_holding(db, NAME)[0]
        return db, ctx

    def _snapshot(self, db):
        s = GraphStore(db_path=db)
        try:
            return (s.count_nodes(), s.count_edges(), len(s.get_all_audit()),
                    len(s.list_suppressions(None)))
        finally:
            s.close()

    def test_the_operation_refuses_and_changes_nothing(self, tmp_path):
        db, ctx = self._seed(tmp_path)
        before = self._snapshot(db)
        s = GraphStore(db_path=db)
        res = GraphOperations(s).forget_node(ctx.node_id)
        s.close()
        assert res["status"] == "refused" and res["reason"] == "whole_conversation"
        assert res["message"] == ("To remove a whole conversation or file, use delete; "
                                  "to keep a specific detail out, forget that detail.")
        assert res["message"] == WHOLE_CONVERSATION
        assert self._snapshot(db) == before
        assert holders_of(db, NAME)                      # the conversation is still there

    def test_rest_cli_and_preview_say_the_same(self, tmp_path, monkeypatch):
        db, ctx = self._seed(tmp_path)
        monkeypatch.setattr("revien.cli._load_config", lambda: {"adapters": {}, "db_path": db})
        before = self._snapshot(db)
        with TestClient(create_app(db_path=db)) as client:
            r = client.post(f"/v1/nodes/{ctx.node_id}/forget", json={})
            assert r.status_code == 409 and r.json()["detail"]["message"] == WHOLE_CONVERSATION
            r = client.get(f"/v1/nodes/{ctx.node_id}/forget/preview")
            assert r.json()["status"] == "refused" and r.json()["message"] == WHOLE_CONVERSATION
        out = CliRunner().invoke(main, ["forget", ctx.node_id, "--yes", "--db", db,
                                        "--daemon-url", "http://127.0.0.1:1"])
        assert out.exit_code == 1 and WHOLE_CONVERSATION in out.output
        assert self._snapshot(db) == before

    def test_cascade_is_gone_everywhere(self, tmp_path):
        db, ctx = self._seed(tmp_path)
        before = self._snapshot(db)
        out = CliRunner().invoke(main, ["forget", "--help"])
        assert "--cascade" not in out.output and "--everywhere" in out.output
        s = GraphStore(db_path=db)
        with pytest.raises(TypeError):
            GraphOperations(s).forget_node(ctx.node_id, cascade=True)
        s.close()
        with TestClient(create_app(db_path=db)) as client:
            r = client.get(f"/v1/nodes/{ctx.node_id}/forget/preview", params={"cascade": "true"})
            assert r.status_code == 400 and "use delete" in r.json()["detail"]["message"]
            for body in ({"cascade": True}, {"cascade": False}):
                r = client.post(f"/v1/nodes/{ctx.node_id}/forget", json=body)
                assert r.status_code == 400
                assert r.json()["detail"]["reason"] == "cascade_removed"
                assert "cascade was removed" in r.json()["detail"]["message"]
                assert "use delete" in r.json()["detail"]["message"]
        assert self._snapshot(db) == before

    def test_delete_is_how_a_conversation_goes(self, tmp_path):
        db, ctx = self._seed(tmp_path)
        with TestClient(create_app(db_path=db)) as client:
            assert client.delete(f"/v1/nodes/{ctx.node_id}").status_code == 200
        assert not contexts_holding(db, NAME)


# ── (4) a shared README is not touched by a forget in another source ──────

class TestASharedReadmeLandsUntouched:
    README = [
        "This tool targets Python3.12 and later releases.",
        "Install it from the v2.1.0 tag of the repository.",
        "Run the test suite before you open a pull request.",
        "Configuration lives in the settings file next to the binary.",
    ]

    def _file(self, d: Path, name: str, lines):
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fs._age(p)
        return p

    DECISION = "We decided to freeze the release branch on Friday."

    def test_python312_and_v210_land_in_b_after_a_forget_in_a(self, tmp_path):
        """A forgets one decision. B holds the same README and the same
        decision: a different source, so every line of it lands, "Python3.12"
        and "v2.1.0" included (B is not taken for a copy of A, and nothing in
        it is held)."""
        from revien.adapters.file_watcher import FileWatcherAdapter
        db = str(tmp_path / "readme.db")
        a_dir, b_dir = tmp_path / "a", tmp_path / "b"
        self._file(a_dir, "README.md", self.README + [self.DECISION])
        fs._sync(db, "files-a", FileWatcherAdapter(watch_dir=str(a_dir)))
        s = GraphStore(db_path=db)
        decision = [n for n in s.list_nodes(limit=100)
                    if n.node_type == NodeType.DECISION and "freeze the release branch" in n.content]
        assert decision, [(n.node_type.value, n.content) for n in s.list_nodes(limit=100)]
        res = GraphOperations(s).forget_node(decision[0].node_id)
        s.close()
        assert res["status"] == "forgotten"
        a_ctx = contexts_holding(db, "Configuration lives")
        assert a_ctx and self.DECISION not in a_ctx[0].content       # A lost the sentence
        self._file(b_dir, "README.md", self.README + [self.DECISION, "B has a line of its own about packaging."])
        fs._sync(db, "files-b", FileWatcherAdapter(watch_dir=str(b_dir)))
        b = contexts_holding(db, "packaging")
        assert len(b) == 1
        for line in self.README + [self.DECISION]:
            assert line in b[0].content, line                       # every line, untouched
        assert "Python3.12" in b[0].content and "v2.1.0" in b[0].content
        assert not [r for r in sqlite3.connect(db).execute(
            "SELECT 1 FROM audit_log WHERE op = 'forget_cut' AND node_id = ?", (b[0].node_id,))]
