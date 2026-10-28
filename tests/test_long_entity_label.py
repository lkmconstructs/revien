"""
A run of capitalised words longer than any name (an attendee list, a
title-cased line) crashed ingest and recall: the rule extractor took the
whole run as ONE entity, and its label broke the 200-character limit on a
node. The memory was not saved (``revien ingest``, REST, MCP), and REST
recall answered 500 from about 220 characters of query.

Now such a run is not an entity at all (a wrong extraction is worse than
none) and the rest of the text is extracted and saved as usual. Other labels
taken from text (topics, declared links) are cut at 200 like every other
label taken from text. A node that still does not fit the schema is a 422
that repeats nothing the caller sent, never a 500.
"""

import asyncio
import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic import BaseModel

from revien.cli import main
from revien.daemon.server import create_app
from revien.forget.engine import stated
from revien.graph.operations import GraphOperations
from revien.graph.schema import LABEL_CUT_KEY, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion import extractor as extractor_module
from revien.ingestion.extractor import MAX_ENTITY_LABEL, RuleBasedExtractor
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine

NAMES = "Dana Reyes Marco Bell Priya Shah Tomas Lind "
# The reported text: 275 characters, one unbroken run of capitalised words.
ATTENDEES = "Attendees: " + NAMES * 6
# A declared link (a book title, a note title, a [[link]]) over 200 characters.
LONG_TITLE = ("The Long Afternoon: Notes on Tides, Harbours and the People Who "
              "Keep Them, " * 4).strip()


def test_fixture_text_is_the_reported_one():
    assert len(ATTENDEES) == 275
    assert len(LONG_TITLE) > 200


def _labels(nodes, node_type):
    return [n.label for n in nodes if n.node_type == node_type]


def _run(length):
    """One unbroken run of capitalised words, exactly ``length`` characters."""
    pool, words = NAMES.split(), []
    while len(" ".join(words + [pool[len(words) % len(pool)]])) <= length - 3:
        words.append(pool[len(words) % len(pool)])
    run = " ".join(words)
    return run + " Q" + "a" * (length - len(run) - 2)


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "revien.db"))
    yield s
    s.close()


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "revien.db"))
    with TestClient(app) as c:
        yield c


# ── The extractor ─────────────────────────────────────────


class TestExtractor:
    def test_a_long_run_is_not_an_entity(self):
        result = RuleBasedExtractor().extract(ATTENDEES, source_id="t")
        assert _labels(result.nodes, NodeType.ENTITY) == []
        assert result.context_node.content == ATTENDEES
        # The rest of the text is extracted as before.
        assert "dana reyes" in _labels(result.nodes, NodeType.TOPIC)
        assert all(len(n.label) <= 200 for n in result.nodes)

    def test_names_still_extract(self):
        result = RuleBasedExtractor().extract(
            "Dana Reyes met Marco Bell and Priya Shah in Lisbon. "
            "We use PostgreSQL for the \"Harbour Ledger\" service.",
            source_id="t")
        entities = _labels(result.nodes, NodeType.ENTITY)
        for name in ("Dana Reyes", "Marco Bell", "Priya Shah", "Harbour Ledger",
                     "PostgreSQL"):
            assert name in entities

    def test_the_bound_is_the_label_limit(self):
        # The crash started where a label stops fitting: 200 characters.
        limit = next(m.max_length for m in Node.model_fields["label"].metadata
                     if hasattr(m, "max_length"))
        assert MAX_ENTITY_LABEL == limit == 200

    @pytest.mark.parametrize("length", [82, 93, 150, 199, 200])
    def test_a_run_up_to_200_characters_extracts_as_before(self, length):
        run = _run(length)
        assert len(run) == length
        entities = _labels(RuleBasedExtractor().extract(run + " joined.").nodes,
                           NodeType.ENTITY)
        assert run in entities

    def test_a_run_of_201_characters_is_not_an_entity(self):
        run = _run(201)
        assert len(run) == 201
        result = RuleBasedExtractor().extract(run + " joined.")
        # Not cut down into a shorter, wrong name either.
        assert _labels(result.nodes, NodeType.ENTITY) == []

    @pytest.mark.parametrize("text", [
        "the " + "a" * 250 + " tier is ready",
        "the big" + " " * 250 + "service is ready",
    ])
    def test_a_tier_phrase_round_a_pasted_token_is_not_an_entity(self, text):
        result = RuleBasedExtractor().extract(text, source_id="t")
        assert all(len(n.label) <= MAX_ENTITY_LABEL
                   for n in result.nodes if n.node_type == NodeType.ENTITY)

    def test_a_topic_from_a_very_long_word_is_cut_at_200(self):
        word = "z" * 220
        result = RuleBasedExtractor().extract(
            f"{word} alpha. {word} bravo. {word} delta.", source_id="t")
        topics = [n for n in result.nodes if n.node_type == NodeType.TOPIC]
        assert [t.label for t in topics] == [word[:200]]
        assert word in topics[0].content

    def test_a_bigram_topic_of_long_words_is_cut_at_200(self):
        bigram = "q" * 120 + " " + "r" * 120
        result = RuleBasedExtractor().extract((bigram + " ") * 2, source_id="t")
        assert bigram[:200] in _labels(result.nodes, NodeType.TOPIC)


# ── Ingest and recall: pipeline, CLI, REST, MCP ───────────


class TestIngestAndRecall:
    def test_the_attendee_list_is_saved_and_recallable(self, store):
        out = IngestionPipeline(store).ingest(
            IngestionInput(source_id="notes", content=ATTENDEES))
        ctx = store.get_node(out.context_node_id)
        assert ctx is not None and ctx.content == ATTENDEES

        response = RetrievalEngine(store).recall("Priya Shah", include_context=True)
        assert any(r.node_id == out.context_node_id for r in response.results)

    def test_cli_ingest_saves_it(self, tmp_path):
        db = str(tmp_path / "cli.db")
        res = CliRunner().invoke(main, ["ingest", ATTENDEES, "--db", db])
        assert res.exit_code == 0, res.output
        s = GraphStore(db_path=db)
        try:
            contexts = s.list_nodes(node_type=NodeType.CONTEXT)
            assert [n.content for n in contexts] == [ATTENDEES]
        finally:
            s.close()

    def test_rest_ingest_then_recall(self, client):
        r = client.post("/v1/ingest", json={"source_id": "notes", "content": ATTENDEES})
        assert r.status_code == 200, r.text
        ctx_id = r.json()["context_node_id"]
        r = client.post("/v1/recall", json={"query": "Priya Shah", "include_context": True})
        assert r.status_code == 200, r.text
        assert ctx_id in [x["node_id"] for x in r.json()["results"]]

    @pytest.mark.parametrize("repeat", [20, 600])
    def test_rest_recall_with_a_long_run_answers_200(self, client, repeat):
        client.post("/v1/ingest", json={"source_id": "notes",
                                        "content": "Dana Reyes owns the launch."})
        r = client.post("/v1/recall", json={"query": "Dana Reyes " * repeat})
        assert r.status_code == 200, r.text
        assert "results" in r.json()

    def test_mcp_store_then_recall(self, tmp_path):
        pytest.importorskip("mcp", reason="mcp extra not installed")
        from revien.mcp_server import build_mcp_server, close_mcp_server

        def call(server, name, arguments):
            res = asyncio.run(server.call_tool(name, arguments))
            content = res[0] if isinstance(res, tuple) else res
            return json.loads(content[0].text)

        server = build_mcp_server(db_path=str(tmp_path / "mcp.db"))
        try:
            stored = call(server, "revien_store", {"content": ATTENDEES})
            assert stored["context_node_id"]
            recalled = call(server, "revien_recall",
                            {"query": "Priya Shah", "include_context": True})
            assert stored["context_node_id"] in [
                r["node_id"] for r in recalled["results"]]
            long_query = call(server, "revien_recall", {"query": "Dana Reyes " * 20})
            assert "results" in long_query
        finally:
            close_mcp_server(server)


# ── Declared links (vault titles, [[links]], book titles) ──


class TestLongLinks:
    def _entities(self, store):
        return store.list_nodes(node_type=NodeType.ENTITY)

    def test_a_long_link_is_cut_at_200_and_kept_whole_in_content(self, store):
        pipeline = IngestionPipeline(store)
        out = pipeline.ingest(IngestionInput(
            source_id="readwise:h1", content="A highlight about tides.",
            content_type="document", links=[LONG_TITLE]))
        [entity] = self._entities(store)
        assert entity.label == LONG_TITLE[:200]
        assert entity.content == LONG_TITLE
        assert any(e.edge_type == EdgeType.RELATED_TO
                   and e.source_node_id == out.context_node_id
                   for e in store.get_edges_for_node(entity.node_id))

        # Another capture with the same link finds the same entity.
        pipeline.ingest(IngestionInput(
            source_id="readwise:h2", content="Another highlight.",
            content_type="document", links=[LONG_TITLE]))
        assert [n.node_id for n in self._entities(store)] == [entity.node_id]

    def test_a_keyed_refresh_with_a_long_link(self, store):
        pipeline = IngestionPipeline(store)
        for text in ("First version of the note.", "Second version of the note."):
            pipeline.ingest(IngestionInput(
                source_id="vault:Tides.md", content=text, content_type="note",
                ingest_key="vault:Tides.md#1", links=[LONG_TITLE]))
        [entity] = self._entities(store)
        assert entity.label == LONG_TITLE[:200]
        contexts = store.list_nodes(node_type=NodeType.CONTEXT)
        assert [n.content for n in contexts] == ["Second version of the note."]

    def test_a_refresh_that_adds_a_long_link_marks_the_cut(self, store):
        pipeline = IngestionPipeline(store)
        for text, links in (("First version.", []), ("Second version.", [LONG_TITLE])):
            pipeline.ingest(IngestionInput(
                source_id="vault:Tides.md", content=text, content_type="note",
                ingest_key="vault:Tides.md#1", links=links))
        [entity] = self._entities(store)
        assert entity.label == LONG_TITLE[:200] and entity.content == LONG_TITLE
        assert entity.metadata[LABEL_CUT_KEY] == len(LONG_TITLE)
        assert stated(entity) == [LONG_TITLE]

    def test_the_pipeline_marks_only_a_cut(self, store):
        pipeline = IngestionPipeline(store)
        pipeline.ingest(IngestionInput(source_id="readwise:h1", content="A highlight.",
                                       content_type="document", links=[LONG_TITLE]))
        pipeline.ingest(IngestionInput(source_id="readwise:h2", content="Another.",
                                       content_type="document", links=["Short Title"]))
        by_label = {n.label: n for n in self._entities(store)}
        assert by_label[LONG_TITLE[:200]].metadata[LABEL_CUT_KEY] == len(LONG_TITLE)
        assert LABEL_CUT_KEY not in by_label["Short Title"].metadata

    def test_only_a_marked_cut_link_is_its_whole_name(self):
        def entity(content, mark=len(LONG_TITLE), node_type=NodeType.ENTITY):
            return Node(node_type=node_type, label=LONG_TITLE[:200], content=content,
                        metadata={} if mark is None else {LABEL_CUT_KEY: mark})

        assert stated(entity(LONG_TITLE)) == [LONG_TITLE]
        # Without the mark an entity is its label, as on main, even when its
        # content starts with the label and runs on.
        assert stated(entity(LONG_TITLE, mark=None)) == [LONG_TITLE[:200]]
        # The mark holds only while the content is the name it was cut from.
        assert stated(entity("X" + LONG_TITLE[1:])) == [LONG_TITLE[:200]]
        assert stated(entity(LONG_TITLE + " and more")) == [LONG_TITLE[:200]]
        assert stated(entity(LONG_TITLE, mark=str(len(LONG_TITLE)))) == [LONG_TITLE[:200]]
        assert stated(entity(LONG_TITLE, node_type=NodeType.TOPIC)) == [LONG_TITLE[:200]]

    def test_an_edit_drops_the_mark_and_cannot_add_it(self, tmp_path):
        app = create_app(db_path=str(tmp_path / "edit.db"))
        with TestClient(app) as c:
            app.state.pipeline.ingest(IngestionInput(
                source_id="readwise:h1", content="A highlight.", content_type="document",
                links=[LONG_TITLE, "Short Title"]))
            ents = {e["label"]: e for e in c.get(
                "/v1/nodes", params={"node_type": "entity", "limit": 100}).json()}
            cut, short = ents[LONG_TITLE[:200]], ents["Short Title"]
            assert cut["metadata"][LABEL_CUT_KEY] == len(LONG_TITLE)

            # Sending the metadata back unchanged keeps it.
            r = c.put(f"/v1/nodes/{cut['node_id']}", json={"metadata": cut["metadata"]})
            assert r.json()["metadata"][LABEL_CUT_KEY] == len(LONG_TITLE)
            # A content edit drops it, also when the metadata is sent along.
            r = c.put(f"/v1/nodes/{cut['node_id']}",
                      json={"content": LONG_TITLE + ", second edition",
                            "metadata": cut["metadata"]})
            assert r.status_code == 200
            assert LABEL_CUT_KEY not in r.json()["metadata"]
            # And an edit cannot put a mark on a node the pipeline did not mark.
            r = c.put(f"/v1/nodes/{short['node_id']}",
                      json={"metadata": {LABEL_CUT_KEY: len("Short Title")}})
            assert LABEL_CUT_KEY not in r.json()["metadata"]

    def test_the_mark_round_trips_export_and_import(self, tmp_path):
        source = GraphStore(db_path=str(tmp_path / "a.db"))
        try:
            IngestionPipeline(source).ingest(IngestionInput(
                source_id="readwise:h1", content="A highlight.", content_type="document",
                links=[LONG_TITLE]))
            exported = json.loads(source.export_graph().model_dump_json())
        finally:
            source.close()
        app = create_app(db_path=str(tmp_path / "b.db"))
        with TestClient(app) as c:
            assert c.post("/v1/graph/import", json=exported).status_code == 200
            [cut] = [n for n in app.state.store.list_nodes(node_type=NodeType.ENTITY)
                     if n.label == LONG_TITLE[:200]]
            assert cut.metadata[LABEL_CUT_KEY] == len(LONG_TITLE)
            assert stated(cut) == [LONG_TITLE]

    @pytest.mark.parametrize("edited", [False, True])
    def test_an_uncut_200_character_entity_forgets_as_on_main(self, client, edited):
        name = _run(200)
        r = client.post("/v1/ingest", json={
            "source_id": "conv:1", "content": f"We met the {name} today. Also lunch."})
        ctx_id = r.json()["context_node_id"]
        [hit] = [e for e in client.get("/v1/nodes", params={
            "node_type": "entity", "limit": 100}).json() if e["label"] == name]
        if edited:
            r = client.put(f"/v1/nodes/{hit['node_id']}",
                           json={"content": name + ", the society that keeps the ledger"})
            assert r.status_code == 200
        r = client.post(f"/v1/nodes/{hit['node_id']}/forget", json={})
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "forgotten"
        assert client.get(f"/v1/nodes/{hit['node_id']}").status_code == 404
        ctx = client.get(f"/v1/nodes/{ctx_id}").json()["node"]
        assert name not in ctx["content"] and "Also lunch." in ctx["content"]

    @pytest.mark.parametrize("title", ["The Long Afternoon", LONG_TITLE])
    def test_a_forget_uses_the_whole_link_text(self, store, title):
        """Two notes link the same title. The second finds the entity the
        first made and vouches for its whole wording (the title it wrote is
        the entity's content), whether or not the label had to be cut. A
        forget of the entity then uses the whole title, as for a short link:
        it is found word for word, and it is cut from both notes."""
        pipeline = IngestionPipeline(store)
        for sid in ("vault:Tides.md", "vault:Harbours.md"):
            pipeline.ingest(IngestionInput(
                source_id=sid, content=f"Lent out. Reading {title} again.",
                content_type="note", ingest_key=f"{sid}#1", links=[title]))
        [entity] = [n for n in self._entities(store) if n.content == title]
        assert entity.label == title[:200]
        assert stated(entity) == [title]
        rows = store.get_provenance(entity.node_id)
        assert sorted((r["unit_id"], bool(r["label_ok"]), bool(r["wording_ok"]))
                      for r in rows) == [("vault:Harbours.md", True, True),
                                         ("vault:Tides.md", True, True)]

        result = GraphOperations(store).forget_node(entity.node_id)
        assert result["status"] == "forgotten", result.get("message")
        assert store.get_node(entity.node_id) is None
        contexts = store.list_nodes(node_type=NodeType.CONTEXT)
        assert [n.content for n in contexts] == ["Lent out. "] * 2

        # A later capture of the same note, link and all, does not bring the
        # title back: not in its text, not as an entity.
        pipeline.ingest(IngestionInput(
            source_id="vault:Tides.md", content=f"Lent out. I finished {title} today.",
            content_type="note", ingest_key="vault:Tides.md#1", links=[title]))
        assert [n for n in self._entities(store) if n.content == title] == []
        assert all(title not in n.content
                   for n in store.list_nodes(node_type=NodeType.CONTEXT))


# ── REST: a node that does not fit the schema is a 422 ────


class TestSchemaRefusal:
    """Belt and braces: lift the extractor's bound so the old crash comes
    back, and check what REST answers when a node does not fit."""

    def _no_text(self, body, *words):
        for w in words:
            assert w not in body

    def test_ingest_is_422_and_repeats_nothing(self, client, monkeypatch):
        monkeypatch.setattr(extractor_module, "MAX_ENTITY_LABEL", 10 ** 6)
        before = client.get("/v1/health").json()["node_count"]
        r = client.post("/v1/ingest", json={"source_id": "notes", "content": ATTENDEES})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["reason"] == "schema_limit"
        assert detail["problems"] == [{"field": "label", "problem": "string_too_long"}]
        self._no_text(r.text, "Dana", "Reyes", "Priya", "Tomas")
        assert client.get("/v1/health").json()["node_count"] == before

    def test_recall_is_422_not_500(self, client, monkeypatch):
        monkeypatch.setattr(extractor_module, "MAX_ENTITY_LABEL", 10 ** 6)
        r = client.post("/v1/recall", json={"query": "Dana Reyes " * 20})
        assert r.status_code == 422
        assert r.json()["detail"]["reason"] == "schema_limit"
        self._no_text(r.text, "Dana", "Reyes")

    def test_any_other_model_error_is_still_a_500(self, tmp_path, monkeypatch):
        class Unrelated(BaseModel):
            count: int

        def broken_recall(**kwargs):
            Unrelated(count="not a number")

        app = create_app(db_path=str(tmp_path / "other.db"))
        with TestClient(app, raise_server_exceptions=False) as c:
            monkeypatch.setattr(app.state.engine, "recall", broken_recall)
            r = c.post("/v1/recall", json={"query": "Dana Reyes"})
        assert r.status_code == 500

    def test_put_label_over_200_is_422_and_changes_nothing(self, client):
        r = client.post("/v1/ingest", json={"source_id": "notes",
                                            "content": "Dana Reyes owns the launch."})
        node_id = r.json()["context_node_id"]
        before = client.get(f"/v1/nodes/{node_id}").json()["node"]["label"]
        r = client.put(f"/v1/nodes/{node_id}", json={"label": "Tomas Lind " * 30})
        assert r.status_code == 422
        assert r.json()["detail"]["problems"] == [
            {"field": "label", "problem": "string_too_long"}]
        self._no_text(r.text, "Tomas", "Lind")
        assert client.get(f"/v1/nodes/{node_id}").json()["node"]["label"] == before
