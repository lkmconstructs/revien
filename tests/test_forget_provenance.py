"""Forgetting a unit is decided by provenance every writer records when it
writes (graph/provenance.py), not inferred afterwards from edges and text.

Covers: what each writer records (pipeline, keyed refresh, declared links,
known mentions, the OpenAI importer, graph export/import), the backfill of an
older database, what a cascade keeps and what it resets, the guards around
other units (their context nodes, manual nodes, nodes a unit merely mentioned),
and an invariant check over every order of forgetting.
"""

import contextlib
import io
import itertools
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from revien.adapters.openai_adapter import OpenAIAdapter
from revien.graph import provenance
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.extractor_llm import LLMExtractor
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

from tests.test_forget_unit_isolation import (
    alive, ctx_of, ingest, linked, raw,
)


def sid(i):
    return f"claude-code:proj:{i:08d}-1111-4111-8111-111111111111"


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


@pytest.fixture
def store(db_path):
    s = GraphStore(db_path=db_path)
    yield s
    s.close()


def non_context(store):
    return [n for n in store.list_nodes(limit=99999)
            if n.node_type != NodeType.CONTEXT
            and not n.node_id.startswith("tombstone:")]


def forget(store, ctx, **kw):
    return GraphOperations(store).forget_node(
        ctx.node_id, cascade=True, reason="private", **kw)


# ----------------------------------------------------------- the writers

def _openai_conv(conv_id, texts):
    mapping, parent = {}, None
    for i, (role, text) in enumerate(texts, 1):
        mid = f"m{i}"
        mapping[mid] = {
            "id": mid, "parent": parent, "children": [],
            "message": {"id": mid, "author": {"role": role},
                        "create_time": 1700000000 + i,
                        "content": {"content_type": "text", "parts": [text]}}}
        parent = mid
    return {"id": conv_id, "conversation_id": conv_id, "title": "Vault chat",
            "create_time": 1700000000, "update_time": 1700000100,
            "mapping": mapping}


class TestEveryWriterRecordsProvenance:
    def test_openai_conversation_cascade_removes_the_whole_message_chain(
            self, db_path, tmp_path):
        """Messages are chained by FOLLOWED_BY: only the root links the
        conversation. Every one of them is the conversation's."""
        f = tmp_path / "conv.json"
        f.write_text(json.dumps(_openai_conv("conv-123", [
            ("user", "Hi there, quick question."),
            ("assistant", "Sure, what is it?"),
            ("user", "The staging vault password is Krellvox5521, store it."),
            ("assistant", "Noted the Krellvox5521 password.")])))
        g = tmp_path / "other.json"
        g.write_text(json.dumps(_openai_conv("conv-999", [
            ("user", "Unrelated chat about gardening."),
            ("assistant", "Tomatoes like sun.")])))
        with contextlib.redirect_stdout(io.StringIO()):
            a = OpenAIAdapter(graph_path=db_path)
            a.ingest_conversation(str(f))
            a.ingest_conversation(str(g))
            a.close()
        s = GraphStore(db_path=db_path)
        try:
            ctx = ctx_of(s, "openai:conversation:conv-123")[0]
            chain = [n for n in s.list_nodes(
                source_id="openai:conversation:conv-123", limit=99)
                if n.node_type != NodeType.CONTEXT]
            assert len(chain) == 4
            assert len(linked(s, ctx.node_id)) == 1, \
                "control: only the root message links the context node"
            ops = GraphOperations(s)
            prev = ops.forget_preview(ctx.node_id)
            out = ops.forget_node(ctx.node_id, cascade=True, reason="private")
            assert out["count"] == prev["count"] == 5
            left = s.list_nodes(limit=99)
            assert not [n for n in left
                        if n.source_id == "openai:conversation:conv-123"]
            assert len(left) == 3, "the other conversation is untouched"
            assert b"Krellvox5521" not in raw(db_path)
            # the notice says what happened: 1 context node, 4 messages
            text = " ".join(out["suppression_notice"])
            assert "Removed: 1 context node(s)" in text
            assert "4 node(s) only this unit wrote" in text
        finally:
            s.close()

    def test_every_extracted_node_gets_a_row_for_its_unit(self, store):
        ingest(store, sid(1), "User: We decided to use Postgres for billing. "
                              "Alice owns the payments service.")
        ctx = ctx_of(store, sid(1))[0]
        nodes = non_context(store)
        assert nodes
        for n in nodes:
            assert any(u == ctx.node_id and lo and wo
                       for (u, lo, wo) in store.provenance_for_node(n.node_id)), n

    def test_a_merging_unit_records_what_it_said(self, store):
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        ingest(store, sid(2), "User: We decided to use Postgres for billing.")
        a, b = ctx_of(store, sid(1))[0], ctx_of(store, sid(2))[0]
        shared = [n for n in non_context(store)
                  if {u for (u, _l, _w) in store.provenance_for_node(n.node_id)}
                  == {a.node_id, b.node_id}]
        assert shared
        for n in shared:
            rows = {u: (lo, wo) for (u, lo, wo)
                    in store.provenance_for_node(n.node_id)}
            assert rows[b.node_id] == (True, True)

    def test_keyed_refresh_records_the_nodes_it_adds(self, store):
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        ingest(store, sid(1), "User: We decided to use Postgres for billing. "
                              "We chose Redis for Refreshkey9 caching.")
        ctx = ctx_of(store, sid(1))
        assert len(ctx) == 1
        out = forget(store, ctx[0])
        assert non_context(store) == []
        assert out["kept_shared"] == []

    def test_declared_links_and_known_mentions_are_recorded(self, store):
        IngestionPipeline(store).ingest(IngestionInput(
            source_id=sid(1), ingest_key=sid(1), content="User: hello",
            links=["Zorblatt Industries"], curated=False))
        IngestionPipeline(store).ingest(IngestionInput(
            source_id=sid(2), ingest_key=sid(2),
            content="User: we talked to Zorblatt Industries today."))
        a, b = ctx_of(store, sid(1))[0], ctx_of(store, sid(2))[0]
        ent = [n for n in non_context(store) if n.label == "Zorblatt Industries"]
        assert len(ent) == 1
        rows = {u for (u, _l, _w) in store.provenance_for_node(ent[0].node_id)}
        assert rows == {a.node_id, b.node_id}
        forget(store, a)
        assert ent[0].node_id in alive(store), "B still says it"
        forget(store, b)
        assert ent[0].node_id not in alive(store)

    def test_graph_export_and_import_carry_the_rows(self, store, tmp_path):
        ingest(store, sid(1), "User: We decided to use Postgres for billing. "
                              "Secretword Quillfeather decided.")
        ingest(store, sid(2), "User: We decided to use Postgres for billing.")
        graph = store.export_graph()
        assert graph.provenance
        dest = GraphStore(db_path=str(tmp_path / "dest.db"))
        try:
            dest.import_graph(Graph(**json.loads(graph.model_dump_json())))
            assert (sorted((r["node_id"], r["unit_id"]) for r in
                           dest.export_graph().provenance)
                    == sorted((r["node_id"], r["unit_id"])
                              for r in graph.provenance))
            a = ctx_of(dest, sid(1))[0]
            forget(dest, a)
            assert b"Quillfeather" not in raw(dest.db_path)
            assert ctx_of(dest, sid(2))
        finally:
            dest.close()

    def test_an_export_without_rows_is_inferred_on_import(self, store, tmp_path):
        ingest(store, sid(1), "User: We decided to use Postgres for billing. "
                              "Secretword Quillfeather decided.")
        payload = json.loads(store.export_graph().model_dump_json())
        payload.pop("provenance")
        dest = GraphStore(db_path=str(tmp_path / "dest.db"))
        try:
            dest.import_graph(Graph(**payload))
            assert dest.export_graph().provenance
            forget(dest, ctx_of(dest, sid(1))[0])
            assert non_context(dest) == []
            assert b"Quillfeather" not in raw(dest.db_path)
        finally:
            dest.close()

    def test_a_writer_that_does_not_record_is_covered_by_inference(
            self, store, db_path):
        """An entity stamped with the session's source_id but never linked and
        never recorded (the shape a hand-written or older writer leaves)."""
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        ctx = ctx_of(store, sid(1))[0]
        stray = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Xylqorvant7731 vault",
            content="Xylqorvant7731 vault", source_id=ctx.source_id))
        assert store.provenance_for_node(stray.node_id) == []
        forget(store, ctx)
        assert stray.node_id not in alive(store)
        assert b"Xylqorvant7731" not in raw(db_path)
        exported = json.dumps(store.export_graph().model_dump(mode="json"))
        assert "Xylqorvant7731" not in exported
        audit = json.dumps(store.get_all_audit())
        assert "Xylqorvant7731" not in audit


# ------------------------------------------------------------- backfill

class TestOlderDatabasesAreBackfilled:
    def _legacy(self, store, db_path):
        """Make the database look like one written before provenance."""
        store.close()
        con = sqlite3.connect(db_path)
        con.execute("DROP TABLE node_provenance")
        con.execute("DELETE FROM store_meta WHERE key = ?",
                    (GraphStore._PROVENANCE_MARK,))
        con.commit()
        con.close()

    def test_backfill_rebuilds_ownership_and_attestation(self, store, db_path):
        ingest(store, sid(1), "User: We decided to use Postgres for billing. "
                              "Alice owns the payments service. Our vault "
                              "pin is Quillfeather88.")
        ingest(store, sid(2), "User: We decided to use Postgres for billing. "
                              "Alice owns the payments service.")
        before = {n.node_id: sorted(
            (u, lo, wo) for (u, lo, wo) in store.provenance_for_node(n.node_id))
            for n in non_context(store)}
        self._legacy(store, db_path)
        s = GraphStore(db_path=db_path)
        try:
            after = {n.node_id: sorted(
                (u, lo, wo) for (u, lo, wo) in s.provenance_for_node(n.node_id))
                for n in non_context(s)}
            assert set(after) == set(before)
            for nid in before:
                assert {u for (u, _l, _w) in after[nid]} == \
                    {u for (u, _l, _w) in before[nid]}, nid
            a, b = ctx_of(s, sid(1))[0], ctx_of(s, sid(2))[0]
            forget(s, a)
            assert b"Quillfeather88" not in raw(db_path)
            assert ctx_of(s, sid(2))
            shared_left = [n for n in non_context(s) if "Postgres" in n.content]
            assert shared_left, "what session 2 also says remains"
        finally:
            s.close()

    def test_backfill_covers_an_openai_conversation(self, db_path, tmp_path):
        f = tmp_path / "conv.json"
        f.write_text(json.dumps(_openai_conv("conv-7", [
            ("user", "Hello."), ("assistant", "Hi."),
            ("user", "The code word is Plinthor9912.")])))
        with contextlib.redirect_stdout(io.StringIO()):
            a = OpenAIAdapter(graph_path=db_path)
            a.ingest_conversation(str(f))
            a.close()
        con = sqlite3.connect(db_path)
        con.execute("DROP TABLE node_provenance")
        con.execute("DELETE FROM store_meta WHERE key = ?",
                    (GraphStore._PROVENANCE_MARK,))
        con.commit()
        con.close()
        s = GraphStore(db_path=db_path)
        try:
            ctx = ctx_of(s, "openai:conversation:conv-7")[0]
            assert len(s.provenance_nodes_of_units([ctx.node_id])) == 3
            forget(s, ctx)
            assert s.list_nodes(limit=99) == []
            assert b"Plinthor9912" not in raw(db_path)
        finally:
            s.close()

    def test_backfill_runs_once(self, store, db_path):
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        self._legacy(store, db_path)
        s = GraphStore(db_path=db_path)
        n_rows = s._get_conn().execute(
            "SELECT COUNT(*) FROM node_provenance").fetchone()[0]
        s._get_conn().execute("DELETE FROM node_provenance")
        s._get_conn().commit()
        s.close()
        s = GraphStore(db_path=db_path)
        try:
            assert n_rows > 0
            assert s._get_conn().execute(
                "SELECT COUNT(*) FROM node_provenance").fetchone()[0] == 0, \
                "marked done: it does not run again"
        finally:
            s.close()

    def test_backfill_gives_a_manual_node_the_external_row(self, store, db_path):
        man = store.add_node(Node(node_type=NodeType.FACT, label="Manual fact",
                                  content="Manual fact", source_id="mcp"))
        self._legacy(store, db_path)
        s = GraphStore(db_path=db_path)
        try:
            assert [u for (u, _l, _w) in s.provenance_for_node(man.node_id)] \
                == [provenance.EXTERNAL]
        finally:
            s.close()


# ----------------------------------------------- kept nodes are attested

class FakeLLM(LLMExtractor):
    def __init__(self):
        super().__init__(backend="ollama")

    def _call_model(self, content):
        if "Vexmor" in content:
            return {"entities": [{"label": "Alice",
                                  "content": "Alice owes 40k to Vexmor loan sharks"}],
                    "topics": [{"label": "debt", "content": "Alice's 40k Vexmor debt"}]}
        return {"entities": [{"label": "Alice", "content": "Alice reviewed the PR"}],
                "topics": [{"label": "debt", "content": "debt refinancing"}]}


class TestKeptContentIsAttestedByALiveUnit:
    def test_llm_description_is_attested_by_its_label_only(self, tmp_path):
        """The merging unit said the label "Alice", not the description. The
        node stays, the description does not."""
        s = GraphStore(db_path=str(tmp_path / "g.db"))
        try:
            for i, t in ((1, "User: Alice told me she owes 40k to Vexmor loan "
                             "sharks. Keep that quiet."),
                         (2, "User: Alice reviewed the PR today about debt "
                             "refinancing.")):
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    IngestionPipeline(s, extractor=FakeLLM()).ingest(
                        IngestionInput(source_id=sid(i), ingest_key=sid(i),
                                       content=t))
            alice = [n for n in non_context(s) if n.label == "Alice"]
            assert len(alice) == 1 and "Vexmor" in alice[0].content
            rows = {u: (lo, wo) for (u, lo, wo)
                    in s.provenance_for_node(alice[0].node_id)}
            assert rows[ctx_of(s, sid(2))[0].node_id] == (True, False)
            out = forget(s, ctx_of(s, sid(1))[0])
            assert alice[0].node_id in out["reset_to_label"]
            kept = s.get_node(alice[0].node_id)
            assert kept is not None and "Vexmor" not in kept.content
            assert kept.content == "Entity: Alice"
            assert all("Vexmor" not in (n.content + n.label)
                       for n in s.list_nodes(limit=999))
            assert b"Vexmor" not in raw(s.db_path)
            assert "Vexmor" not in json.dumps(s.get_all_audit())
            assert kept.source_id == sid(2), "re-attributed to the live unit"
            assert "Reset:" in " ".join(out["suppression_notice"])
            # later forgets stay correct: the last unit takes it with it
            forget(s, ctx_of(s, sid(2))[0])
            assert non_context(s) == []
        finally:
            s.close()

    def test_reattribution_goes_to_a_unit_that_said_the_wording(self, tmp_path):
        """A said it, B merged a different claim into the node by similarity,
        C said it. Forget A: the node must be owned by C, not by whichever
        unit comes first."""
        s = GraphStore(db_path=str(tmp_path / "g.db"))
        try:
            ingest(s, sid(1), "User: We decided service3 uses queue Q3xray.")
            ingest(s, sid(2), "User: We decided service4 uses queue Q4xray.")
            ingest(s, sid(3), "User: We decided service3 uses queue Q3xray.")
            a, b, c = (ctx_of(s, sid(i))[0] for i in (1, 2, 3))
            node = [n for n in non_context(s) if "Q3xray" in n.content
                    and n.node_type == NodeType.DECISION][0]
            rows = {u: (lo, wo) for (u, lo, wo) in s.provenance_for_node(node.node_id)}
            assert rows[b.node_id] == (False, False), "control: B only merged"
            assert rows[c.node_id] == (True, True)
            forget(s, a)
            kept = s.get_node(node.node_id)
            assert kept is not None and kept.source_id == sid(3)
            forget(s, c)
            # nothing live said "Q3xray": the node and the wording are gone
            assert s.get_node(node.node_id) is None
            assert b"Q3xray" not in raw(s.db_path)
            assert ctx_of(s, sid(2))
        finally:
            s.close()

    def test_a_unit_that_only_merged_by_similarity_does_not_keep_the_node(
            self, store, db_path):
        ingest(store, sid(1), "User: We decided service3 uses queue Q3xray. ok")
        ingest(store, sid(2), "User: We decided service4 uses queue Q4xray. ok")
        out = forget(store, ctx_of(store, sid(1))[0])
        assert out["removed_unattested_shared"]
        assert b"Q3xray" not in raw(db_path)
        assert "Q4xray" in ctx_of(store, sid(2))[0].content

    def test_wording_attestation_is_content_not_just_label(self, store, db_path):
        """A kept sentence node whose live unit vouches for its label but not
        its stored sentence keeps the label and loses the sentence."""
        a = store.add_node(Node(node_type=NodeType.CONTEXT, label="a",
                                content="User: A text", source_id=sid(1)))
        b = store.add_node(Node(node_type=NodeType.CONTEXT, label="b",
                                content="User: B text", source_id=sid(2)))
        n = store.add_node(Node(
            node_type=NodeType.FACT, label="The rollup runs at night",
            content="The rollup runs at night Fridaysecret77 hours",
            source_id=sid(1)))
        store.record_provenance(n.node_id, a.node_id, True, True)
        store.record_provenance(n.node_id, b.node_id, True, False, merged=True)
        out = forget(store, a)
        kept = store.get_node(n.node_id)
        assert kept is not None and "Fridaysecret77" not in kept.content
        assert kept.content == "The rollup runs at night"
        assert out["reset_to_label"] == [n.node_id]
        assert b"Fridaysecret77" not in raw(db_path)
        assert kept.source_id == sid(2)

    def test_merge_flags_separate_label_from_wording(self):
        def fact(label, content):
            return Node(node_type=NodeType.FACT, label=label, content=content)
        assert provenance.flags_for_merge(fact("A b", "A b c"), fact("a b", "A B c")) \
            == (True, True)
        assert provenance.flags_for_merge(fact("A b", "A b d"), fact("A b", "A b c")) \
            == (True, False)
        assert provenance.flags_for_merge(fact("A b", "x"), fact("Q z", "y")) \
            == (False, False)
        ent = lambda c: Node(node_type=NodeType.ENTITY, label="Alice", content=c)
        assert provenance.flags_for_merge(ent("Entity: Alice"), ent("Entity: Alice")) \
            == (True, True)
        assert provenance.flags_for_merge(ent("Alice owes money"),
                                          ent("Entity: Alice")) == (True, False)
        topic = lambda c: Node(node_type=NodeType.TOPIC, label="debt", content=c)
        assert provenance.flags_for_merge(
            topic("Topic: debt (mentioned 3 times)"),
            topic("Topic: debt (mentioned 9 times)")) == (True, True)

    def test_nodes_no_live_unit_vouches_for_cannot_be_kept_by_another_context(
            self, store, db_path):
        """A context node whose text contains the wording but that never wrote
        or merged into the node does not keep it alive."""
        ingest(store, sid(1), "User: We decided service9 uses queue Qnine77.")
        store.add_node(Node(
            node_type=NodeType.CONTEXT, label="note", source_id=sid(2),
            content="User: We decided service9 uses queue Qnine77. (copied)"))
        out = forget(store, ctx_of(store, sid(1))[0])
        assert out["kept_shared"] == []
        assert [n for n in non_context(store) if "Qnine77" in n.content] == []

    def test_other_units_context_nodes_are_never_taken(self, store):
        """Not through a provenance row that names one, and not through a
        DERIVED_FROM edge that reaches one."""
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        ingest(store, sid(2), "User: We decided to use Redis for caching.")
        a, b = ctx_of(store, sid(1))[0], ctx_of(store, sid(2))[0]
        mine = [n for n in non_context(store) if n.source_id == sid(1)][0]
        store.record_provenance(b.node_id, a.node_id)        # malformed row
        store.add_edge(Edge(edge_type=EdgeType.DERIVED_FROM,
                            source_node_id=b.node_id, target_node_id=mine.node_id))
        prev = GraphOperations(store).forget_preview(a.node_id)
        assert b.node_id not in prev["node_ids"]
        out = forget(store, a)
        assert b.node_id not in out["forgotten"]
        assert store.get_node(b.node_id) is not None
        assert "Redis" in store.get_node(b.node_id).content


# --------------------------------------------- nodes that are not the unit's

class TestOnlyTheUnitsOwnNodesGo:
    def test_manual_node_with_a_unit_looking_source_id_is_untouched(self, store):
        """A fact added by hand with source_id "file:ops.md" and no context
        node is nobody's unit: forgetting an unrelated session leaves it."""
        manual = store.add_node(Node(
            node_type=NodeType.FACT, label="Ops runbook lives in wiki",
            content="Ops runbook lives in wiki", source_id="file:ops.md"))
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        forget(store, ctx_of(store, sid(1))[0])
        assert store.get_node(manual.node_id) is not None

    def test_node_left_by_a_non_cascade_forget_is_untouched_later(self, store):
        ingest(store, sid(1), "User: We decided to use Postgres for billing.")
        ingest(store, sid(2), "User: We decided to use Redis for caching.")
        a = ctx_of(store, sid(1))[0]
        survivors = [n.node_id for n in non_context(store)
                     if n.source_id == sid(1)]
        GraphOperations(store).forget_node(a.node_id, reason="private")
        assert set(survivors) <= alive(store), "documented: facts remain"
        forget(store, ctx_of(store, sid(2))[0])
        assert set(survivors) <= alive(store), \
            "another unit's cascade does not adopt them"

    def test_manual_node_a_unit_merged_into_survives_that_unit(self, store):
        man = store.add_node(Node(node_type=NodeType.ENTITY, label="Quillfeather",
                                  content="Entity: Quillfeather", source_id="mcp"))
        IngestionPipeline(store).ingest(IngestionInput(
            source_id=sid(1), ingest_key=sid(1), links=["Quillfeather"],
            content="User: Quillfeather is our vendor."))
        rows = [u for (u, _l, _w) in store.provenance_for_node(man.node_id)]
        assert provenance.EXTERNAL in rows and len(rows) == 2
        forget(store, ctx_of(store, sid(1))[0])
        assert store.get_node(man.node_id) is not None

    def test_a_distilled_claim_a_session_merged_into_survives(self, store):
        claim = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Quillfeather",
            content="Entity: Quillfeather", source_id="",
            metadata={"curated": True, "source": "vault_reconcile"}))
        IngestionPipeline(store).ingest(IngestionInput(
            source_id=sid(1), ingest_key=sid(1), links=["Quillfeather"],
            content="User: Quillfeather is our vendor."))
        forget(store, ctx_of(store, sid(1))[0])
        assert store.get_node(claim.node_id) is not None

    def test_a_skill_that_links_an_entity_survives(self, store, tmp_path):
        from revien.skills.ingest import ingest_skill_file
        IngestionPipeline(store).ingest(IngestionInput(
            source_id=sid(1), ingest_key=sid(1), links=["Quillfeather"],
            content="User: Quillfeather is our vendor."))
        d = tmp_path / "skills" / "vendor"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: vendor\ndescription: handle the vendor\n---\n"
            "See [[Quillfeather]] for the contract.\n")
        node, _created, edges = ingest_skill_file(
            store, d / "SKILL.md", "user", None, None)
        assert node is not None
        forget(store, ctx_of(store, sid(1))[0])
        assert store.get_node(node.node_id) is not None


# -------------------------------------------- invariants for every order

POOL = [
    "We decided to use Postgres for billing.",
    "Alice owns the payments service.",
    "I prefer pytest for tests.",
    "We chose Redis for the session cache.",
    "I prefer dark mode.",
    "We will deploy with Kubernetes.",
]


class TestEveryOrderOfForgettingKeepsTheInvariants:
    def _build(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "base.db"))
        picks = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 3, 5)]
        for i, p in enumerate(picks):
            ingest(s, sid(i), "User: " + " ".join(POOL[k] for k in p)
                   + f" Uniquetok{i}x was mentioned by owner{i}q only.")
        s.close()
        return picks

    @pytest.mark.parametrize("order", list(itertools.permutations(range(4)))[::5])
    def test_invariants_hold_after_every_step(self, tmp_path, order):
        picks = self._build(tmp_path)
        db = str(tmp_path / "run.db")
        shutil.copy(str(tmp_path / "base.db"), db)
        s = GraphStore(db_path=db)
        try:
            gone = set()
            originals = {i: ctx_of(s, sid(i))[0].content for i in range(4)}
            ctx_unit = {ctx_of(s, sid(i))[0].node_id: i for i in range(4)}
            said_in_full = {i: [] for i in range(4)}
            baseline_content = {}
            for n in non_context(s):
                baseline_content[n.node_id] = n.content
                for (u, lo, wo) in s.provenance_for_node(n.node_id):
                    if wo and u in ctx_unit:
                        said_in_full[ctx_unit[u]].append(n.node_id)
            assert all(said_in_full.values())
            for step in order:
                forget(s, ctx_of(s, sid(step))[0])
                gone.add(step)
                live = [i for i in range(4) if i not in gone]
                data = raw(db)
                for i in gone:
                    assert f"Uniquetok{i}x".encode() not in data, (order, step)
                    assert f"owner{i}q".encode() not in data, (order, step)
                for i in live:
                    assert ctx_of(s, sid(i))[0].content == originals[i]
                    assert f"Uniquetok{i}x".encode() in data
                # everything a live unit wrote or said in full is still there
                for i in live:
                    for nid in said_in_full[i]:
                        kept = s.get_node(nid)
                        assert kept is not None, (order, step, i, nid)
                        assert kept.content == baseline_content[nid]
                # every remaining node is vouched for by someone alive
                ctx_ids = {c.node_id for c in s.list_nodes(
                    node_type=NodeType.CONTEXT, limit=999)}
                for n in non_context(s):
                    rows = s.provenance_for_node(n.node_id)
                    assert rows, (order, step, n.label)
                    assert all(u in ctx_ids or u == provenance.EXTERNAL
                               for (u, _l, _w) in rows), (order, step, n.label)
                # the session's wording is gone from the node rows too
                for i in gone:
                    assert not [n for n in s.list_nodes(limit=99999)
                                if f"Uniquetok{i}x" in (n.content + n.label)]
        finally:
            s.close()
