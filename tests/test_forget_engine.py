"""The forget verb: what it removes, cuts back, keeps and refuses
(docs/FORGET_SPEC.md §2, §3, §5; invariant 2)."""
import json
import sqlite3
from pathlib import Path

import pytest

from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "e.db"))
    yield s
    s.close()


@pytest.fixture
def pipe(store):
    return IngestionPipeline(store)


def session(pipe, key, text, path=None, runtime="claude-code"):
    return pipe.ingest(IngestionInput(
        source_id=f"{runtime}:proj:{key}", ingest_key=f"{runtime}:proj:{key}",
        origin_runtime=runtime, session_key=key, content=text,
        metadata={"path": str(path)} if path else {}))


def conversation(pipe, key, text):
    return pipe.ingest(IngestionInput(
        source_id=f"chatgpt:conversation:{key}",
        ingest_key=f"chatgpt:conversation:{key}", origin_runtime="chatgpt",
        session_key=key, content=text))


def nodes_by_type(store, t):
    return [n for n in store.list_nodes(limit=1000) if n.node_type == t]


def audit_ops(store, node_id):
    return [a["op"] for a in store.get_node_audit(node_id)]


def fact_with(store, needle):
    return [n for n in store.list_nodes(limit=1000)
            if n.node_type != NodeType.CONTEXT and needle in n.label + n.content][0]


class TestOneThingInOneSource:
    def test_removes_the_node_and_cuts_its_wording_from_its_own_source_only(self, store, pipe):
        a = session(pipe, "A", "User: We decided to use postgres for billing. We decided to use ZEBRA77 as the codename.")
        b = session(pipe, "B", "User: We decided to use sqlite for the demo.")
        target = fact_with(store, "ZEBRA77")
        before_b = {n.node_id for n in store.list_nodes(limit=1000)
                    if any(r["unit_id"].endswith(":B") for r in store.get_provenance(n.node_id))}
        res = GraphOperations(store).forget_node(target.node_id)
        assert res["status"] == "forgotten"
        assert store.get_node(target.node_id) is None
        ctx_a = store.get_node(a.context_node_id)
        assert ctx_a is not None and "ZEBRA77" not in ctx_a.content
        assert "postgres for billing" in ctx_a.content
        assert before_b <= {n.node_id for n in store.list_nodes(limit=1000)}
        assert store.get_node(b.context_node_id) is not None

    def test_never_touches_another_units_context_node(self, store, pipe):
        """Invariant 2: even with edges between the two units' nodes."""
        a = session(pipe, "A", "User: We decided to use postgres for billing. We decided to use ZEBRA77 as the codename.")
        b = session(pipe, "B", "User: We decided to use postgres for the demo.")
        for edge_type in (EdgeType.RELATED_TO, EdgeType.DERIVED_FROM, EdgeType.FOLLOWED_BY):
            store.add_edge(Edge(edge_type=edge_type, source_node_id=a.context_node_id,
                                target_node_id=b.context_node_id))
            store.add_edge(Edge(edge_type=edge_type, source_node_id=b.context_node_id,
                                target_node_id=a.context_node_id))
        GraphOperations(store).forget_node(fact_with(store, "ZEBRA77").node_id)
        assert store.get_node(b.context_node_id).content.startswith("User: We decided to use postgres for the demo")
        assert store.get_node(a.context_node_id) is not None

    def test_a_carrier_another_unit_vouched_for_in_full_is_kept(self, store):
        a_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="a", content="A: the code ZEBRA77 opens it"),
                               owner=("session:claude-code:A", 1, 1))
        store.add_node(Node(node_type=NodeType.CONTEXT, label="b", content="B: it was opened with the code ZEBRA77 again"),
                       owner=("session:claude-code:B", 1, 1))
        target = store.add_node(Node(node_type=NodeType.FACT, label="code", content="the code ZEBRA77"),
                                owner=("session:claude-code:A", 1, 1))
        carrier = store.add_node(Node(node_type=NodeType.FACT, label="opens", content="the code ZEBRA77 opens it"),
                                 owner=("session:claude-code:A", 1, 1))
        store.record_provenance(carrier.node_id, "session:claude-code:B", True, True)
        res = GraphOperations(store).forget_node(target.node_id)
        assert store.get_node(carrier.node_id) is not None
        assert [r["unit_id"] for r in store.get_provenance(carrier.node_id)] == ["session:claude-code:B"]
        assert carrier.node_id in [k["node_id"] for k in res["kept"]]
        assert "ZEBRA77" not in store.get_node(a_ctx.node_id).content

    def test_hand_added_nodes_are_never_removed(self, store, pipe):
        session(pipe, "A", "User: We chose Postgres for orders. We decided to use ZEBRA77 as the codename.")
        hand = store.add_node(Node(node_type=NodeType.FACT, label="hand", content="typed by hand"))
        GraphOperations(store).forget_node(fact_with(store, "ZEBRA77").node_id)
        assert store.get_node(hand.node_id) is not None

    def test_labels_compare_unicode_aware(self, store):
        """'Jörg' and 'Jürg' stay distinct: forgetting one name never cuts or
        removes the other, and a decomposed spelling still matches its own."""
        a_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="a",
                                    content='We met "Jo\u0308rg Meyer" about the launch.'),
                               owner=("session:claude-code:A", 1, 1))
        b_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="b",
                                    content='We met "Jürg Meyer" about the launch.'),
                               owner=("session:claude-code:B", 1, 1))
        a_ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Jörg Meyer",
                                    content="Entity: Jörg Meyer"),
                               owner=("session:claude-code:A", 1, 1))
        b_ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Jürg Meyer",
                                    content="Entity: Jürg Meyer"),
                               owner=("session:claude-code:B", 1, 1))
        res = GraphOperations(store).forget_node(a_ent.node_id)
        assert res["status"] == "forgotten"
        assert store.get_node(a_ent.node_id) is None
        assert "Jörg" not in store.get_node(a_ctx.node_id).content.replace("o\u0308", "ö")
        assert store.get_node(b_ent.node_id).label == "Jürg Meyer"
        assert "Jürg Meyer" in store.get_node(b_ctx.node_id).content

    def test_forgetting_twice_is_not_found(self, store, pipe):
        session(pipe, "A", "User: We chose Postgres for orders. We decided to use ZEBRA77 as the codename.")
        target = fact_with(store, "ZEBRA77").node_id
        ops = GraphOperations(store)
        assert ops.forget_node(target)["status"] == "forgotten"
        assert ops.forget_node(target)["status"] == "not_found"


class TestImportedConversations:
    """A forget of one detail in one imported conversation touches that
    conversation and no other."""

    def test_importer_conversations_are_independent_units(self, store, pipe):
        c1 = conversation(pipe, "c1", "User: We chose Postgres for orders. We decided to use ZEBRA77 as the codename.\nAssistant: Noted.")
        c2 = conversation(pipe, "c2", "User: We chose SQLite for the demo.\nAssistant: Noted, SQLite.")
        n2_before = {n.node_id for n in store.list_nodes(limit=1000)
                     if any(r["unit_id"].endswith("c2") for r in store.get_provenance(n.node_id))}
        res = GraphOperations(store).forget_node(fact_with(store, "ZEBRA77").node_id)
        assert res["status"] == "forgotten"
        assert "ZEBRA77" not in store.get_node(c1.context_node_id).content
        assert n2_before and n2_before <= {n.node_id for n in store.list_nodes(limit=1000)}

    def test_openai_adapter_conversations_are_cut_one_at_a_time(self, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        from tests.forget_seed import _chatgpt_export
        db = str(tmp_path / "oa.db")
        with OpenAIAdapter(graph_path=db) as a:
            for cid, secret in (("conv-1", "ALPHA11"), ("conv-2", "BRAVO22")):
                f = tmp_path / f"{cid}.json"
                _chatgpt_export(f, secret)
                conv = json.loads(f.read_text())[0]
                conv["id"] = cid
                f.write_text(json.dumps(conv))
                a.ingest_conversation(str(f))
            n2 = {n.node_id: n.content for n in a.store.list_nodes(source_id="openai:conversation:conv-2", limit=50)}
            target = fact_with(a.store, "ALPHA11")
            res = GraphOperations(a.store).forget_node(target.node_id)
            assert res["status"] == "forgotten"
            assert a.store.get_node(target.node_id) is None
            assert all(n.content != target.content for n in a.store.list_nodes(
                source_id="openai:conversation:conv-1", limit=50))
            assert {n.node_id: n.content for n in a.store.list_nodes(
                source_id="openai:conversation:conv-2", limit=50)} == n2


class TestSingleNodeForget:
    TEXT = ("User: We decided to use postgres for billing.\n"
            "User: My locker code is ZEBRA77XQ and I decided to use ZEBRA77XQ as the codename.\n"
            "Assistant: Noted, we also decided to use redis for caching.")

    def test_cuts_exactly_the_wording_out_of_the_stored_conversation(self, store, pipe):
        a = session(pipe, "A", self.TEXT)
        target = [n for n in store.list_nodes(limit=100)
                  if n.node_type == NodeType.DECISION and "ZEBRA77XQ" in n.content][0]
        res = GraphOperations(store).forget_node(target.node_id)
        assert res["status"] == "forgotten"
        ctx = store.get_node(a.context_node_id)
        assert "ZEBRA77XQ" not in ctx.content and "ZEBRA77XQ" not in ctx.label
        assert "postgres for billing" in ctx.content and "redis for caching" in ctx.content
        assert res["wording_cut"] and res["wording_cut"][0]["context_node_id"] == a.context_node_id
        # snapshots of the context node were cut too
        for row in store.get_node_audit(a.context_node_id):
            assert "ZEBRA77XQ" not in json.dumps(row)
        assert res["bound_to"] == [{"unit_id": "session:claude-code:A"}]
        assert "keeps its wording out of later captures of claude-code session A" in res["blast_radius"]

    def test_every_node_carrying_the_wording_goes_too(self, store, pipe):
        session(pipe, "A", self.TEXT)
        target = [n for n in store.list_nodes(limit=100)
                  if n.node_type == NodeType.DECISION and "ZEBRA77XQ" in n.content][0]
        GraphOperations(store).forget_node(target.node_id)
        for n in store.list_nodes(limit=100):
            assert "ZEBRA77XQ" not in n.label + n.content

    def test_unicode_secret_is_found_in_escaped_audit_json(self, store, pipe, tmp_path):
        secret = "Zoë-密码-ß9"
        session(pipe, "U", f"User: We decided to use {secret} as codename.")
        ops = GraphOperations(store)
        while True:
            held = [n for n in store.list_nodes(limit=100)
                    if n.node_type != NodeType.CONTEXT and secret in n.label + n.content]
            if not held:
                break
            ops.forget_node(held[0].node_id)
        store.close()
        conn = sqlite3.connect(store.db_path)
        try:
            assert scrub.find_residue(conn, [secret]) == {}
        finally:
            conn.close()
        raw = b"".join(open(store.db_path + s, "rb").read() for s in ("", "-wal") if Path(store.db_path + s).exists())
        assert secret.encode() not in raw
        assert json.dumps(secret).strip('"').encode() not in raw   # the \u-escaped form

    def test_preview_changes_nothing(self, store, pipe):
        session(pipe, "A", self.TEXT)
        target = fact_with(store, "ZEBRA77XQ")
        before = (store.count_nodes(), store.count_edges(), len(store.get_all_audit()))
        res = GraphOperations(store).forget_preview(target.node_id)
        assert res["status"] == "preview" and res["exists"]
        assert (store.count_nodes(), store.count_edges(), len(store.get_all_audit())) == before
        assert store.list_suppressions(None) == []

    def test_forget_audit_row_is_content_free_and_the_verb_is_the_code_path(self, store, pipe):
        session(pipe, "A", self.TEXT)
        GraphOperations(store).forget_node(
            fact_with(store, "ZEBRA77XQ").node_id, reason="my password is hunter2",
            construct_id="hunter2")
        rows = [r for r in store.get_all_audit() if r["op"] == "forget"]
        assert rows
        blob = json.dumps(rows)
        assert "hunter2" not in blob and "ZEBRA77XQ" not in blob
        assert all(r["actor"] == "api:forget" for r in rows)


class TestSharedNodes:
    def _shared(self, store, same_words):
        a_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="A convo", content="A said: we deploy on fridays ZEBRA77"),
                               owner=("session:claude-code:A", 1, 1))
        b_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="B convo", content="B said: deploys happen"),
                               owner=("session:claude-code:B", 1, 1))
        shared = store.add_node(Node(node_type=NodeType.FACT, label="deploys on fridays",
                                     content="we deploy on fridays ZEBRA77"),
                                owner=("session:claude-code:A", 1, 1))
        store.record_provenance(shared.node_id, "session:claude-code:B", True, same_words)
        return a_ctx, b_ctx, shared

    def test_name_only_vouching_cuts_back_to_the_label_and_logs_it(self, store):
        a_ctx, b_ctx, shared = self._shared(store, same_words=False)
        res = GraphOperations(store).forget_node(shared.node_id)
        node = store.get_node(shared.node_id)
        assert node is not None and node.content == node.label == "deploys on fridays"
        assert "ZEBRA77" not in json.dumps(node.model_dump(mode="json"))
        assert shared.node_id in res["cut_back"]
        assert "cut_back" in audit_ops(store, shared.node_id)
        entry = [a for a in store.get_node_audit(shared.node_id) if a["op"] == "cut_back"][0]
        assert "ZEBRA77" not in json.dumps(entry) and entry["before"] is None
        assert "cut_back_at" in node.metadata
        assert store.get_node(b_ctx.node_id) is not None
        assert "ZEBRA77" not in store.get_node(a_ctx.node_id).content

    def test_a_node_written_in_full_by_two_units_is_forgotten_from_both(self, store):
        a_ctx, b_ctx, shared = self._shared(store, same_words=True)
        res = GraphOperations(store).forget_node(shared.node_id)
        assert store.get_node(shared.node_id) is None
        assert sorted(b["unit_id"] for b in res["bound_to"]) == [
            "session:claude-code:A", "session:claude-code:B"]

    def test_pipeline_merge_records_what_each_unit_vouched_for(self, store, pipe):
        session(pipe, "A", "User: We chose Postgres for orders.")
        session(pipe, "B", "User: We chose Postgres for billing.")
        pg = [n for n in nodes_by_type(store, NodeType.ENTITY) if n.label == "Postgres"][0]
        rows = {r["unit_id"]: r for r in store.get_provenance(pg.node_id)}
        assert rows["session:claude-code:A"]["wording_ok"] and rows["session:claude-code:B"]["wording_ok"]
        fact = [n for n in nodes_by_type(store, NodeType.FACT)]
        # facts whose label matched but whose words differ vouch for the name only
        shared = [n for n in store.list_nodes(limit=200)
                  if len(store.get_provenance(n.node_id)) == 2 and n.node_type != NodeType.ENTITY]
        for n in shared:
            assert any(not r["wording_ok"] for r in store.get_provenance(n.node_id))


class TestRefusal:
    def test_paraphrased_wording_is_refused_naming_the_context_node(self, store):
        ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="session",
                                  content="User: my locker code is 4471"),
                             owner=("session:claude-code:A", 1, 1))
        para = store.add_node(Node(node_type=NodeType.FACT, label="locker number",
                                   content="The user's combination is four four seven one"),
                              owner=("session:claude-code:A", 1, 1))
        before = (store.count_nodes(), store.count_edges(), len(store.get_all_audit()))
        res = GraphOperations(store).forget_node(para.node_id)
        assert res["status"] == "refused" and res["reason"] == "wording_not_found"
        assert res["context_node_id"] == ctx.node_id and ctx.node_id in res["message"]
        assert (store.count_nodes(), store.count_edges(), len(store.get_all_audit())) == before
        assert store.list_suppressions(None) == []
        assert store.get_node(para.node_id) is not None
        # a whole conversation is removed with delete, not forget
        refused = GraphOperations(store).forget_node(ctx.node_id)
        assert refused["status"] == "refused"
        assert GraphOperations(store).delete_node(ctx.node_id) is True


class TestUnprovenOwnership:
    def test_a_legacy_node_a_unit_later_merged_into_is_still_unproven(self, store):
        """A node from before provenance has unknown owners. A unit that
        merges into it later is one of them, not the only one: forgetting that
        unit must not delete it."""
        legacy = store.add_node(Node(node_type=NodeType.FACT, label="shared fact",
                                     content="we deploy on fridays"))
        store.record_provenance(legacy.node_id, "unproven:", False, False)
        store.add_node(Node(node_type=NodeType.CONTEXT, label="a", content="A: we deploy on fridays"),
                       owner=("session:claude-code:A", 1, 1))
        store.record_provenance(legacy.node_id, "session:claude-code:A", True, False)
        target = store.add_node(Node(node_type=NodeType.FACT, label="schedule", content="we deploy on fridays"),
                                owner=("session:claude-code:A", 1, 1))
        res = GraphOperations(store).forget_node(target.node_id)
        assert store.get_node(legacy.node_id) is not None
        assert legacy.node_id in res["unproven"] or not res["forgotten"].count(legacy.node_id)
        assert legacy.node_id not in res["forgotten"]

    def test_legacy_nodes_are_kept_listed_and_cut_back(self, store, tmp_path):
        """Nodes from before provenance: the forgotten unit's likely nodes are
        kept and listed, never deleted on a guess."""
        ctx = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="legacy", content="User: code ZEBRA77 here",
            source_id="claude-code:proj:old", origin_runtime="claude-code", session_key="old",
            metadata={"ingest_key": "claude-code:proj:old"}))
        fact = store.add_node(Node(
            node_type=NodeType.FACT, label="code fact", content="code ZEBRA77 here",
            source_id="claude-code:proj:old", origin_runtime="claude-code", session_key="old"))
        other = store.add_node(Node(node_type=NodeType.FACT, label="unrelated", content="nothing to see"))
        conn = store._get_conn()
        conn.execute("DELETE FROM node_provenance")          # as if written before 006
        conn.commit()
        store._backfill_provenance(conn)
        conn.commit()
        assert [r["unit_id"] for r in store.get_provenance(ctx.node_id)] == ["session:claude-code:old"]
        assert store.get_provenance(fact.node_id)[0]["unit_id"].startswith("unproven:")
        target = store.add_node(Node(node_type=NodeType.DECISION, label="the code", content="code ZEBRA77 here"),
                                owner=("session:claude-code:old", 1, 1))
        res = GraphOperations(store).forget_node(target.node_id)
        assert res["status"] == "forgotten"
        assert store.get_node(target.node_id) is None
        assert "ZEBRA77" not in store.get_node(ctx.node_id).content
        assert store.get_node(fact.node_id) is not None            # kept, not deleted
        assert store.get_node(other.node_id) is not None
        assert fact.node_id in res["cut_back"] and fact.node_id not in res["unproven"]
        kept = store.get_node(fact.node_id)
        assert "ZEBRA77" not in kept.label + kept.content           # but scrubbed
