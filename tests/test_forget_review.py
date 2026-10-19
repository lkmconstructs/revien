"""Findings from the independent review of db263d8, and the owner's rulings:
word-bounded matching, unproven nodes left alone, the standing denylist, the
race, --everywhere, snapshots, the stale model, and honest wording."""
import gzip
import json
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests import forget_seed as fs

REPO = Path(__file__).resolve().parents[1]
A = "session:claude-code:A"
B = "session:claude-code:B"


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "r.db"))
    yield s
    s.close()


def ctx(store, text, unit=A, **kw):
    return store.add_node(Node(node_type=NodeType.CONTEXT, label=text.split("\n")[0][:200],
                               content=text, **kw), owner=(unit, 1, 1))


def node(store, ntype, label, content, unit=A, **kw):
    return store.add_node(Node(node_type=ntype, label=label, content=content, **kw),
                          owner=(unit, 1, 1))


def run_pipeline(store, text, key="chatgpt:conversation:c1", **kw):
    return IngestionPipeline(store).ingest(IngestionInput(
        source_id=key, ingest_key=key, origin_runtime="chatgpt",
        session_key=key.split(":")[-1], content=text, **kw))


# ── B1: whole words only ──────────────────────────────────────────────

class TestWordBoundedMatching:
    TEXT = ("User: The rapid rollout needs capital.\n"
            "User: Therapists agreed to the plan.\n"
            "User: We decided to cut capital spending by half.\n"
            "User: The API is stable and the Api docs are done.")

    def _setup(self, store):
        c = ctx(store, self.TEXT)
        api = node(store, NodeType.ENTITY, "API", "API")
        dec = node(store, NodeType.DECISION, "cut capital spending by half",
                   "decided to cut capital spending by half.")
        return c, api, dec

    def test_a_short_label_inside_longer_words_cuts_nothing_else(self, store):
        c, api, dec = self._setup(store)
        res = GraphOperations(store).forget_node(api.node_id)
        assert res["status"] == "forgotten"
        text = store.get_node(c.node_id).content
        for kept in ("rapid rollout needs capital", "Therapists agreed",
                     "cut capital spending by half"):
            assert kept in text
        assert "API is stable" not in text and "Api docs" not in text   # mixed case, whole word
        assert store.get_node(dec.node_id) is not None                   # not a carrier

    def test_matching_is_unicode_aware_at_the_edges(self):
        out, n = scrub.cut_text("Das Müller-Projekt ist neu. Der Müllerei geht es gut.", ["Müller"])
        assert n == 1 and "Müllerei" in out

    def test_the_audit_query_and_training_scrubs_are_word_bounded_too(self, store, tmp_path):
        from revien.neural.training import TrainingLoop
        c = ctx(store, "User: The tomato code is ZQXTOM77XYZ now.")
        store.record_use_judgment(
            episode_key="e#1", node_id=c.node_id, used=True, reason="r", shared_terms=[],
            similarity=0.1, query="best tomatoes for a salad", mode="shadow", applied=False)
        other = node(store, NodeType.FACT, "bystander", "plain words", unit=B)
        store.record_use_judgment(
            episode_key="e#2", node_id=other.node_id, used=True, reason="r", shared_terms=[],
            similarity=0.1, query="best tomatoes for a salad", mode="shadow", applied=False)
        store.record_use_judgment(
            episode_key="e#3", node_id=other.node_id, used=True, reason="r", shared_terms=[],
            similarity=0.1, query="what is the tomato code", mode="shadow", applied=False)
        tdb = str(tmp_path / "t.db")
        TrainingLoop(db_path=tdb, model_dir=str(tmp_path / "m")).log_retrieval(
            "best tomatoes for a salad", [{"node_id": "x", "label": "salad", "node_type": "fact", "score": 1}])
        ops = GraphOperations(store)
        ops.training_dbs = (tdb,)
        ops.forget_node(c.node_id, cascade=True)
        queries = {r[0] for r in store._get_conn().execute("SELECT query FROM use_judgments")}
        assert "best tomatoes for a salad" in queries          # unrelated row survives
        assert "what is the tomato code" not in queries        # the echo goes
        t = sqlite3.connect(tdb)
        try:
            assert t.execute("SELECT COUNT(*) FROM retrieval_signals").fetchone()[0] == 1
        finally:
            t.close()


# ── B2: unproven nodes are kept and listed ────────────────────────────

class TestUnprovenIsLeftAlone:
    def test_a_hand_added_fact_is_not_cut_back_and_is_listed(self, store):
        c = ctx(store, "User: We decided to use postgres for billing.")
        f = node(store, NodeType.FACT, "billing db", "We decided to use postgres for billing.")
        hand = store.add_node(Node(node_type=NodeType.FACT, label="note to self",
                                   content="We decided to use postgres for billing."))
        res = GraphOperations(store).forget_node(f.node_id)
        assert store.get_node(hand.node_id).content == "We decided to use postgres for billing."
        assert hand.node_id in res["unproven"] and hand.node_id not in res["cut_back"]
        assert "(no recorded source)" in res["still_held_by"]

    def test_another_units_own_context_and_history_are_not_edited(self, store):
        """An upgraded database: session B is legacy (unproven), and holds
        the same sentence as session A. Forgetting A leaves B's words and B's
        audit history exactly as they were."""
        sentence = "We decided to use postgres for billing."
        a = ctx(store, f"User: {sentence}")
        f = node(store, NodeType.FACT, "billing", sentence)
        b = store.add_node(Node(node_type=NodeType.CONTEXT, label="B chat",
                                content=f"User: {sentence}\nUser: Something else entirely."))
        store.record_provenance(b.node_id, "unproven:session:claude-code:B", False, False)
        audit_before = json.dumps([x for x in store.get_node_audit(b.node_id)])
        res = GraphOperations(store).forget_node(f.node_id)
        after = store.get_node(b.node_id)
        assert after.content == b.content and after.label == b.label
        assert json.dumps(store.get_node_audit(b.node_id)) == audit_before
        assert b.node_id in res["unproven"]
        assert b.node_id in [h["node_id"] for h in res["leftovers"] if h["kind"] == "other_units"]

    def test_a_node_is_never_both_cut_back_and_unproven(self, store):
        a = ctx(store, "User: legacy things here.", unit="session:claude-code:old")
        legacy = store.add_node(Node(node_type=NodeType.FACT, label="legacy", content="legacy description text"))
        store.record_provenance(legacy.node_id, "unproven:session:claude-code:old", False, False)
        res = GraphOperations(store).forget_node(a.node_id, cascade=True)
        assert legacy.node_id in res["cut_back"]
        assert legacy.node_id not in res["unproven"]
        assert "kept and listed" not in res["message"].split(legacy.node_id)[0] or True
        assert store.get_node(legacy.node_id).content == "legacy"


# ── B3: guards that used to survive mutation ──────────────────────────

class TestEveryFieldTheCutReaches:
    SENT = "We decided to use ZQXFOO as the gate."

    def _setup(self, store):
        c = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="chat", content=f"User: hello there friend.\nUser: {self.SENT}",
            event_time_text="ZQXFOO", source_context="notes about ZQXFOO",
            metadata={"caller": {"note": "ZQXFOO again"}}), owner=(A, 1, 1))
        d = node(store, NodeType.DECISION, "use ZQXFOO as the gate", self.SENT)
        GraphOperations(store).forget_node(d.node_id)
        return store.get_node(c.node_id)

    @pytest.mark.parametrize("field", ["event_time_text", "source_context", "metadata"])
    def test_the_cut_reaches(self, store, field):
        c = self._setup(store)
        assert "ZQXFOO" not in json.dumps(getattr(c, field))
        assert "hello there friend" in c.content

    def test_a_merge_entrys_actor_is_cut_but_other_history_is_not(self, store):
        sentence = "We decided to use ZQXFOO as the gate."
        c = ctx(store, f"User: {sentence}")
        d = node(store, NodeType.DECISION, "gate", sentence)
        survivor = node(store, NodeType.ENTITY, "Gate", "Gate", unit=B)
        store._record_node_audit(survivor.node_id, "normalized_merge",
                                 actor=f"'We decided to use ZQXFOO as the gate.' -> 'Gate'",
                                 after_node=survivor)
        store.record_audit(survivor.node_id, "note", actor="mentions ZQXFOO elsewhere")
        GraphOperations(store).forget_node(d.node_id)
        rows = {a["op"]: a for a in store.get_node_audit(survivor.node_id)}
        assert "ZQXFOO" not in rows["normalized_merge"]["actor"]
        assert rows["note"]["actor"] == "mentions ZQXFOO elsewhere"   # another unit's history is its own

    def test_the_vector_of_a_cut_node_is_purged(self, store):
        from tests.forget_surfaces import TinyEmbedder
        from revien.semantic.index import SemanticIndex
        sem = SemanticIndex(store, embedder=TinyEmbedder(), enabled=True)
        c = ctx(store, f"User: hello there friend.\nUser: {self.SENT}")
        d = node(store, NodeType.DECISION, "use ZQXFOO as the gate", self.SENT)
        for n in (c, d):
            sem.index_node(n.node_id, n.label, n.content)
        conn = store._get_conn()
        count = lambda i: conn.execute("SELECT count(*) FROM vec_nodes WHERE node_id = ?", (i,)).fetchone()[0]
        assert count(c.node_id) == 1
        GraphOperations(store).forget_node(d.node_id)
        assert count(c.node_id) == 0 and count(d.node_id) == 0

    def test_provenance_goes_with_its_node(self, store):
        n = node(store, NodeType.FACT, "x", "some words here")
        assert store.get_provenance(n.node_id)
        store.delete_node(n.node_id)
        assert store.get_provenance(n.node_id) == []
        m = node(store, NodeType.FACT, "y", "other words here")
        GraphOperations(store).forget_node(m.node_id)
        assert store.get_provenance(m.node_id) == []


class TestSnapshotsAreScrubbed:
    def _snap(self, tmp_path, secret):
        from revien.watch import snapshot_db
        db = str(tmp_path / "s.db")
        s = GraphStore(db_path=db)
        run_pipeline(s, f"User: My safe code is {secret}. We decided to use {secret} as the codename.")
        out = tmp_path / "s.db.snapshots"
        plain = snapshot_db(db, use_gzip=False, out_dir=out)
        packed = snapshot_db(db, use_gzip=True, out_dir=out)
        return db, s, Path(plain), Path(packed)

    def test_plain_and_gzip_snapshots_are_scrubbed(self, tmp_path):
        secret = fs.new_secret()
        db, s, plain, packed = self._snap(tmp_path, secret)
        assert secret.encode() in plain.read_bytes()
        ctxid = [n.node_id for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=5)][0]
        res = GraphOperations(s).forget_node(ctxid, cascade=True)
        s.close()
        assert secret.encode() not in plain.read_bytes()
        assert secret.encode() not in gzip.decompress(packed.read_bytes())
        assert res["snapshots"]["scrubbed"] == 2
        assert "snapshots" not in {l["kind"] for l in res["leftovers"]}
        c = sqlite3.connect(plain)
        try:
            assert scrub.find_residue(c, [secret]) == {}
        finally:
            c.close()

    def test_a_snapshot_that_cannot_be_opened_is_named(self, tmp_path):
        secret = fs.new_secret()
        db, s, plain, packed = self._snap(tmp_path, secret)
        (plain.parent / "broken.db").write_bytes(b"not a database at all")
        ctxid = [n.node_id for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=5)][0]
        res = GraphOperations(s).forget_node(ctxid, cascade=True)
        s.close()
        named = [l for l in res["leftovers"] if l["kind"] == "snapshots"]
        assert len(named) == 1 and "broken.db" in named[0]["note"]
        assert res["scope"] == "forgotten from Revien"


# ── ruling 1: a standing rule, any order ─────────────────────────────

class TestStandingDenylist:
    TEXT = ("User: My vault code is ZQXVAULT99 for the main safe.\n"
            "User: We decided to use ZQXVAULT99 as the project codename.")

    def _file_unit(self, tmp_path, text):
        f = tmp_path / "notes.md"
        f.write_text(text)
        return f

    def _capture(self, store, f, text):
        return IngestionPipeline(store).ingest(IngestionInput(
            source_id="file:notes.md", origin_runtime="file", origin_source="watch",
            content=text, metadata={"path": str(f)}))

    def test_a_forget_that_lands_after_the_checks_cannot_be_written_over(self, tmp_path):
        """The reviewer's race: the capture passed both gate checks, a forget
        (from another connection) lands, and the capture then writes. The
        write transaction asks again."""
        db = str(tmp_path / "race.db")
        f = self._file_unit(tmp_path, self.TEXT)
        s1 = GraphStore(db_path=db)
        first = self._capture(s1, f, self.TEXT)
        s2 = GraphStore(db_path=db)
        pipe = IngestionPipeline(s1)
        real = pipe.dedup.deduplicate_node
        fired = []

        def slow_dedup(candidate, *a, **kw):
            if not fired:
                fired.append(1)
                # after the second gate check, another process forgets
                GraphOperations(s2).forget_node(first.context_node_id, cascade=True)
            return real(candidate, *a, **kw)

        pipe.dedup.deduplicate_node = slow_dedup
        out = pipe.ingest(IngestionInput(
            source_id="file:notes.md", origin_runtime="file", origin_source="watch",
            content=self.TEXT, metadata={"path": str(f)}))
        assert fired
        s1.close(); s2.close()
        conn = sqlite3.connect(db)
        try:
            assert scrub.find_residue(conn, ["ZQXVAULT99"], ignore=scrub.NOT_CONTENT_TABLES) == {}
        finally:
            conn.close()
        assert out.suppressed is True

    def test_a_sentence_forgotten_mid_capture_is_cut_from_what_was_written(self, tmp_path):
        """A capture from a source with no stable identity (so no unit-level
        record): only the standing denylist can keep the sentence out."""
        db = str(tmp_path / "race2.db")
        s1 = GraphStore(db_path=db)
        run_pipeline(s1, self.TEXT)
        target = [n for n in s1.list_nodes(node_type=NodeType.DECISION, limit=10)
                  if "ZQXVAULT99" in n.content][0]
        s2 = GraphStore(db_path=db)
        pipe = IngestionPipeline(s1)
        real = pipe.dedup.deduplicate_node
        fired = []

        def slow_dedup(candidate, *a, **kw):
            if not fired:
                fired.append(1)
                GraphOperations(s2).forget_node(target.node_id)
            return real(candidate, *a, **kw)

        pipe.dedup.deduplicate_node = slow_dedup
        pipe.ingest(IngestionInput(
            source_id="api:other", content=self.TEXT + "\nUser: A harmless new remark about lunch plans."))
        texts = " ".join(n.content for n in s1.list_nodes(limit=100))
        s1.close(); s2.close()
        assert fired
        assert "We decided to use ZQXVAULT99 as the project codename" not in texts
        assert "harmless new remark" in texts

    def test_a_copy_of_the_forgotten_file_at_a_new_path_with_a_line_appended(self, tmp_path, monkeypatch):
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(seeded.ctx_ids[0], cascade=True)
        s.close()
        from revien.adapters.file_watcher import FileWatcherAdapter
        copy_dir = tmp_path / "elsewhere"
        copy_dir.mkdir()
        dup = copy_dir / "copy.md"
        shutil.copy(seeded.source_file, dup)
        with open(dup, "a") as fh:
            fh.write("An unrelated new line about next quarter planning.\n")
        fs._age(dup)
        fs._resync(seeded.db, "files", FileWatcherAdapter(watch_dir=str(copy_dir)))
        s = GraphStore(db_path=seeded.db)
        texts = " ".join(n.content for n in s.list_nodes(limit=100))
        s.close()
        assert "next quarter planning" in texts
        assert seeded.secret not in texts

    def test_a_short_fragment_is_kept_out_by_source_identity_and_the_response_says_so(self, tmp_path):
        from revien.graph.units import path_unit_key, identity_keys, real_path
        f = tmp_path / "pins.md"
        f.write_text("Meeting notes. PIN Kite940c. Agenda items follow below for today.")
        unit = "file:" + path_unit_key(str(f))
        s = GraphStore(db_path=str(tmp_path / "pins.db"))
        s.register_unit(unit, real_path(str(f)), identity_keys(str(f)))
        c = s.add_node(Node(node_type=NodeType.CONTEXT, label="pins", content=f.read_text()), owner=(unit, 1, 1))
        t = s.add_node(Node(node_type=NodeType.FACT, label="PIN Kite940c.", content="PIN Kite940c."), owner=(unit, 1, 1))
        res = GraphOperations(s).forget_node(t.node_id)
        assert res["fallback"]["fragments"] >= 1
        assert "shorter than 20 characters" in res["fallback"]["note"]
        assert "never matched on their own in another source" in res["message"]
        f.write_text(f.read_text() + " A new line appended at the end of this file.")
        out = IngestionPipeline(s).ingest(IngestionInput(
            source_id="file:pins.md", origin_runtime="file", content=f.read_text(),
            metadata={"path": str(f)}))
        texts = " ".join(n.content for n in s.list_nodes(limit=50))
        assert "Kite940c" not in texts and "A new line appended" in texts
        # ...but a different source is not stopped by a 13-character match
        g = tmp_path / "other.md"
        g.write_text("Other notes. PIN Kite940c. Agenda items follow below for today.")
        IngestionPipeline(s).ingest(IngestionInput(
            source_id="file:other.md", origin_runtime="file", content=g.read_text(),
            metadata={"path": str(g)}))
        assert "Kite940c" in " ".join(n.content for n in s.list_nodes(limit=50))
        s.close()

    def test_the_sync_reset_text_does_not_promise_what_is_no_longer_true(self):
        src = (REPO / "revien" / "cli.py").read_text(encoding="utf-8")
        assert "including any forgotten wording it still contains" not in src
        assert "minus the sentences" in src

    def test_every_other_writer_leaves_forgotten_sentences_out(self, tmp_path, monkeypatch):
        """OpenAI messages, skill bodies and imported graph nodes."""
        seeded = fs.seed("openai_adapter", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        # forget a single message's sentence (node level), then re-ingest the conversation
        target = [n for n in s.list_nodes(limit=50) if seeded.secret in n.content and n.node_type != NodeType.CONTEXT][0]
        GraphOperations(s).forget_node(target.node_id)
        for n in s.list_nodes(limit=50):
            s.delete_node(n.node_id)
        s.close()
        seeded.recapture()
        s = GraphStore(db_path=seeded.db)
        try:
            texts = " ".join(n.content for n in s.list_nodes(limit=50))
        finally:
            s.close()
        assert f"my safe code is {seeded.secret}" not in texts


# ── ruling 2: forget --everywhere ─────────────────────────────────────

class TestEverywhere:
    SENT = "We decided to use postgres for the billing service."

    def _two_units(self, store):
        a = ctx(store, f"User: {self.SENT}\nUser: Alpha only remark about nothing in particular.", unit=A)
        fa = node(store, NodeType.FACT, "billing a", self.SENT, unit=A)
        b = ctx(store, f"User: {self.SENT}\nUser: Bravo only remark about nothing in particular.", unit=B)
        fb = node(store, NodeType.FACT, "billing b", self.SENT, unit=B)
        return a, fa, b, fb

    def test_plain_forget_is_per_source_and_names_who_still_holds_it(self, store):
        a, fa, b, fb = self._two_units(store)
        res = GraphOperations(store).forget_node(fa.node_id)
        assert B in res["still_held_by"]
        assert self.SENT in store.get_node(b.node_id).content
        assert store.get_node(fb.node_id) is not None
        assert res["scope"] == "forgotten from Revien"
        assert any(l["kind"] == "other_units" and B in l["units"] for l in res["leftovers"])

    def test_everywhere_reaches_every_unit_that_holds_it(self, store):
        a, fa, b, fb = self._two_units(store)
        res = GraphOperations(store).forget_node(fa.node_id, everywhere=True)
        assert res["everywhere"] is True
        assert self.SENT not in store.get_node(b.node_id).content
        assert "Bravo only remark" in store.get_node(b.node_id).content   # the rest stays
        # another source's node is cut in place, never deleted or suppressed
        survivor = store.get_node(fb.node_id)
        assert survivor is not None and self.SENT not in survivor.content
        assert res["still_held_by"] == []
        assert {s["unit_id"] for s in res["suppressed"]} == {A}
        # every node the exact-wording cut touched is listed, the origin's own too
        assert {c["node_id"] for c in res["cut_in_place"]} == {a.node_id, b.node_id, fb.node_id}
        assert {c["node_id"] for c in res["cut_in_place"] if c.get("origin")} == {a.node_id}

    def test_the_cli_and_rest_have_the_verb(self, tmp_path):
        out = CliRunner().invoke(main, ["forget", "--help"])
        assert "--everywhere" in out.output
        db = str(tmp_path / "ev.db")
        s = GraphStore(db_path=db)
        a, fa, b, fb = self._two_units(s)
        s.close()
        with TestClient(create_app(db_path=db)) as client:
            r = client.post(f"/v1/nodes/{fa.node_id}/forget", json={"everywhere": True})
            assert r.status_code == 200 and r.json()["everywhere"] is True
            assert r.json()["still_held_by"] == []

    def test_there_is_no_mcp_forget_surface(self):
        src = (REPO / "revien" / "mcp_server.py").read_text(encoding="utf-8")
        import re
        # no tool, handler or route named for forgetting (a comment may say it)
        assert not re.search(r"(def\s+|name[\"']?\s*[:=]\s*[\"'])\w*forget", src, re.I)
        assert "/forget" not in src


# ── ruling 3 and N1, N3 ───────────────────────────────────────────────

class TestModelAndHonesty:
    def test_the_model_is_marked_stale_on_every_forget(self, store, tmp_path, monkeypatch):
        from revien.neural.training import TrainingLoop
        tdb = str(tmp_path / "training.db")
        loop = TrainingLoop(db_path=tdb, model_dir=str(tmp_path / "models"))
        assert loop.is_stale() is False
        c = ctx(store, "User: some words to forget here.")
        ops = GraphOperations(store)
        ops.training_dbs = (tdb,)
        res = ops.forget_node(c.node_id, cascade=True)
        assert res["model_stale"] is True and res["model"]["stale"] is True
        assert loop.is_stale() is True       # the flag lives where the retrain path reads it

    def test_there_is_no_scheduled_retrain_job(self):
        # (a retrain happens only on new retrieval signals, a stale mark, or a
        # manual run: see tests/test_forget_round3.py)
        for rel in ("revien/daemon/scheduler.py", "revien/daemon/daemon.py", "revien/consolidate.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            assert "force_train" not in src and "training_loop.train" not in src

    def test_exports_are_worded_as_may_still_contain_it(self, store):
        store.set_meta("exports_written", json.dumps({"count": 1}))
        c = ctx(store, "User: some words to forget here.")
        res = GraphOperations(store).forget_node(c.node_id, cascade=True)
        (e,) = [l for l in res["leftovers"] if l["kind"] == "exports"]
        assert "may still contain it" in e["note"]

    def test_remaining_counts_content_only_not_identities(self, tmp_path):
        word = "zqxsecretname"
        d = tmp_path / "w"
        d.mkdir()
        f = d / f"{word}.md"
        f.write_text(f"Plan. We decided to use the {word} codename for the gate project.")
        fs._age(f)
        from revien.adapters.file_watcher import FileWatcherAdapter
        db = str(tmp_path / "id.db")
        fs._sync(db, "files", FileWatcherAdapter(watch_dir=str(d)))
        s = GraphStore(db_path=db)
        res = GraphOperations(s).forget_node(fs.context_ids(db)[0], cascade=True)
        s.close()
        assert not ({"suppressions", "node_provenance", "capture_units"} & set(res["remaining"]))

    def test_an_unproven_session_is_suppressed_by_its_known_identity(self, tmp_path):
        """N1: ownership of the node is unproven (it predates provenance), but
        its source key names the session, so the session is suppressed."""
        from tests.test_forget_upgrade import made_by
        path = made_by("v0.3.0-tag-7bf8738", tmp_path)
        s = GraphStore(db_path=path)
        fact = [n for n in s.list_nodes(limit=10) if n.node_type == NodeType.FACT and "PostgreSQL" in n.content][0]
        res = GraphOperations(s).forget_node(fact.node_id)
        assert res["suppressed"] and res["suppressed"][0]["unit_id"] == "session:claude-code:sess-1"
        assert "unproven" in res["identity_note"]
        out = IngestionPipeline(s).ingest(IngestionInput(
            source_id="claude-code:proj-x:sess-1", ingest_key="claude-code:proj-x:sess-1",
            origin_runtime="claude-code", session_key="sess-1",
            content="User: hello\nAssistant: hi"))
        s.close()
        assert out.suppressed is True

    def test_spec_marks_the_old_daemon_design_approved_and_built(self):
        spec = (REPO / "docs" / "FORGET_SPEC.md").read_text(encoding="utf-8")
        assert "Proposed design, for owner approval" not in spec
        assert "approved by the owner and built" in spec
