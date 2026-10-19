"""Round 3 of the independent review (owner ruling: scope the denylist by origin
identity, not by path and not globally), and the bugs found with it.

Every test here fails at 76aa7a0 except the few marked as locks, which pin a
behaviour that was already right and that a mutation in
scripts/forget_mutation_check.py now turns red."""
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from revien.forget import gate, scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests import forget_seed as fs

REPO = Path(__file__).resolve().parents[1]
SHARED = "We decided to use ZQXPG for the billing service."
BOIL = "Let me run the tests to verify the change."
A_ONLY = "Alpha only remark about the weather in Lisbon."
B_ONLY = "Session B also covered the Denver office lease."
UA = "item:chatgpt:conversation:A"
UB = "item:chatgpt:conversation:B"


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "r3.db"))
    yield s
    s.close()


def cap(pipe, key, text, runtime="chatgpt", **kw):
    return pipe.ingest(IngestionInput(
        source_id=key, ingest_key=key, origin_runtime=runtime,
        session_key=key.split(":")[-1], content=text, **kw))


def contexts(store):
    return {tuple(sorted(r["unit_id"] for r in store.get_provenance(n.node_id))): n.content
            for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=50)
            if not n.node_id.startswith("tombstone:")}


def two_sessions(store):
    """A (pipeline capture), B (pipeline capture); A and B share two sentences."""
    pipe = IngestionPipeline(store)
    a = cap(pipe, "chatgpt:conversation:A", f"User: {SHARED}\nUser: {BOIL}\nUser: {A_ONLY}")
    b = cap(pipe, "chatgpt:conversation:B", f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}")
    return pipe, a, b


def b_grows(pipe):
    return cap(pipe, "chatgpt:conversation:B",
               f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}\nUser: Newer B line about parking permits.")


def a_only_fact(store, text=SHARED, label="pg"):
    """A node that only A wrote (so B is not a forgotten unit), holding a
    sentence B's own conversation also holds."""
    store.add_node(Node(node_type=NodeType.CONTEXT, label="A",
                        content=f"User: {SHARED}\nUser: {BOIL}\nUser: {A_ONLY}",
                        metadata={"ingest_key": "chatgpt:conversation:A"}), owner=(UA, 1, 1))
    return store.add_node(Node(node_type=NodeType.FACT, label=label, content=text), owner=(UA, 1, 1))


# ── the owner's ruling: bound to the origin, not to the world ─────────────

class TestOriginScope:
    def test_a_plain_forget_of_a_leaves_bs_own_copy_at_its_next_sync(self, store):
        pipe = IngestionPipeline(store)
        cap(pipe, "chatgpt:conversation:B", f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}")
        fact = a_only_fact(store)
        res = GraphOperations(store).forget_node(fact.node_id)
        assert UB in res["still_held_by"] and res["status"] == "forgotten"
        b_grows(pipe)
        text = contexts(store)[(UB,)]
        assert SHARED in text and BOIL in text and "Newer B line" in text

    def test_a_cascade_of_a_leaves_bs_own_sentences_and_boilerplate(self, store):
        pipe, a, b = two_sessions(store)
        res = GraphOperations(store).forget_node(a.context_node_id, cascade=True)
        assert res["status"] == "forgotten"
        assert UB in res["still_held_by"]
        b_grows(pipe)
        text = contexts(store)[(UB,)]
        assert SHARED in text and BOIL in text and B_ONLY in text
        assert (UA,) not in contexts(store)

    def test_on_an_upgraded_database_b_keeps_its_copy_too(self, tmp_path):
        db = str(tmp_path / "up.db")
        s = GraphStore(db_path=db)
        pipe = IngestionPipeline(s)
        for sid, tail in (("sA", A_ONLY), ("sB", B_ONLY)):
            pipe.ingest(IngestionInput(
                source_id=f"claude-code:proj:{sid}", ingest_key=f"claude-code:proj:{sid}",
                origin_runtime="claude-code", session_key=sid, project_key="proj",
                content=f"User: {SHARED}\nUser: {BOIL}\nUser: {tail}"))
        s.close()
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM node_provenance")
        conn.execute("DELETE FROM capture_units")
        conn.execute("DELETE FROM suppressions")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        conn.close()
        s = GraphStore(db_path=db)
        try:
            fact = [n for n in s.list_nodes(node_type=NodeType.FACT, limit=10)
                    if "ZQXPG" in n.content][0]
            res = GraphOperations(s).forget_node(fact.node_id)
            assert [x["unit_id"] for x in res["suppressed"]] == ["session:claude-code:sA"]
            IngestionPipeline(s).ingest(IngestionInput(
                source_id="claude-code:proj:sB", ingest_key="claude-code:proj:sB",
                origin_runtime="claude-code", session_key="sB", project_key="proj",
                content=f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}\nUser: One more line."))
            ctx = [n for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=10)
                   if "sB" in str(s.get_provenance(n.node_id))][0]
            assert SHARED in ctx.content and BOIL in ctx.content
        finally:
            s.close()

    def test_a_renamed_claude_code_session_keeps_its_session_id_and_stays_forgotten(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        GraphOperations(s).forget_node(seeded.ctx_ids[0], cascade=True)
        s.close()
        copy = seeded.source_file.parent / "renamed-copy.jsonl"
        copy.write_text(seeded.source_file.read_text() + json.dumps({
            "type": "user", "sessionId": "uuid-claude-1", "timestamp": "2026-07-01T10:01:00Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "An unrelated later remark about parking."}]}}) + "\n")
        fs._age(copy)
        seeded.recapture()
        s = GraphStore(db_path=seeded.db)
        try:
            blob = " ".join(n.label + n.content for n in s.list_nodes(limit=200))
        finally:
            s.close()
        assert seeded.secret not in blob

    def test_a_codex_rollout_has_no_internal_id_so_a_renamed_copy_is_recognised_by_content(self, tmp_path, monkeypatch):
        seeded = fs.seed("codex", tmp_path, monkeypatch)
        s = GraphStore(db_path=seeded.db)
        res = GraphOperations(s).forget_node(seeded.ctx_ids[0], cascade=True)
        s.close()
        said = " ".join(res["reentry"]["stops"])
        assert "file name" in said and "content" in said
        copy = seeded.source_file.parent / "rollout-copy-renamed.jsonl"
        copy.write_text(seeded.source_file.read_text() + json.dumps({
            "timestamp": "2026-07-01T10:02:00Z", "type": "response_item",
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "An unrelated later remark about parking."}]}}) + "\n")
        fs._age(copy)
        seeded.recapture()
        s = GraphStore(db_path=seeded.db)
        try:
            blob = " ".join(n.label + n.content for n in s.list_nodes(limit=200))
        finally:
            s.close()
        assert seeded.secret not in blob
        assert "parking" in blob

    def test_the_copy_rule(self):
        sig = {f"s{i}" for i in range(10)}
        assert gate.is_copy_of(sig, sig | {"new1", "new2"})            # a copy with lines appended
        assert gate.is_copy_of(sig, {f"s{i}" for i in range(9)})       # 90% of it, an edit or two
        assert not gate.is_copy_of(sig, {"s0", "s1", "s2", "x", "y"})  # shares three
        assert not gate.is_copy_of(sig, sig | {f"n{i}" for i in range(12)})  # buried in a much longer source
        assert not gate.is_copy_of({"only"}, {"only", "x"})            # one sentence is never enough
        assert not gate.is_copy_of({"a", "b", "c"}, {"a", "b", "z", "y"})  # two of three: boilerplate, not a copy
        assert gate.is_copy_of({"a", "b"}, {"a", "b", "z"})            # a short file copied whole, one line added

    def test_every_response_says_what_stops_coming_back_and_what_does_not(self, store):
        fact = a_only_fact(store)
        res = GraphOperations(store).forget_node(fact.node_id)
        re_ = res["reentry"]
        assert re_["scope"] == "this source"
        assert any("id its source gave it" in s for s in re_["stops"])
        joined = " ".join(re_["does_not_stop"])
        assert "another source" in joined and "forget --everywhere" in joined
        assert "Scope: this source." in res["message"] and "Does not stop:" in res["message"]
        node = store.add_node(Node(node_type=NodeType.FACT, label="x", content="Some other fact to forget now."),
                              owner=(UB, 1, 1))
        res = GraphOperations(store).forget_node(node.node_id, everywhere=True)
        assert res["reentry"]["scope"] == "everywhere"
        assert "exact forgotten wording, in every source" in " ".join(res["reentry"]["stops"])


# ── --everywhere: the exact wording, never a deletion, never a guess ──────

class TestEverywhereDoesNotOverreach:
    def test_a_cascade_with_everywhere_does_not_touch_bs_context_or_suppress_b(self, store):
        pipe, a, b = two_sessions(store)
        before = contexts(store)[(UB,)]
        res = GraphOperations(store).forget_node(a.context_node_id, cascade=True, everywhere=True)
        assert [x["unit_id"] for x in res["suppressed"]] == [UA]
        assert res["cut_in_place"] == []
        assert contexts(store)[(UB,)] == before
        assert B_ONLY in contexts(store)[(UB,)]
        assert "names no wording" in " ".join(res["reentry"]["stops"])
        # B is not suppressed: its next capture lands, nothing of it is stripped
        b_grows(pipe)
        text = contexts(store)[(UB,)]
        assert SHARED in text and BOIL in text and "Newer B line" in text

    def test_a_derived_node_with_everywhere_cuts_the_sentence_in_place_and_lists_every_touch(self, store):
        pipe = IngestionPipeline(store)
        cap(pipe, "chatgpt:conversation:B", f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}")
        fact = a_only_fact(store)
        hand = store.add_node(Node(node_type=NodeType.FACT, label="note",
                                   content=f"{SHARED} Also the hand-written reminder about tulips."))
        res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        assert res["status"] == "forgotten"
        # nothing of another source was deleted or suppressed
        assert [x["unit_id"] for x in res["suppressed"]] == [UA]
        survivor = store.get_node(hand.node_id)
        assert survivor is not None
        assert survivor.content == "Also the hand-written reminder about tulips."   # not cut back to its label
        assert SHARED not in contexts(store)[(UB,)]
        assert BOIL in contexts(store)[(UB,)] and B_ONLY in contexts(store)[(UB,)]
        listed = {c["node_id"] for c in res["cut_in_place"]}
        assert hand.node_id in listed
        assert all(store.get_node(i) is not None for i in listed)
        for c in res["cut_in_place"]:
            assert c["node_id"] in res["message"]
        # ...and the exact wording is banned in every source from now on
        b_grows(pipe)
        text = contexts(store)[(UB,)]
        assert SHARED not in text and BOIL in text and "Newer B line" in text

    def test_everywhere_on_an_upgraded_database_does_not_suppress_on_a_guess(self, tmp_path):
        db = str(tmp_path / "upe.db")
        s = GraphStore(db_path=db)
        pipe = IngestionPipeline(s)
        for sid, tail in (("sA", A_ONLY), ("sB", B_ONLY)):
            pipe.ingest(IngestionInput(
                source_id=f"claude-code:proj:{sid}", ingest_key=f"claude-code:proj:{sid}",
                origin_runtime="claude-code", session_key=sid, project_key="proj",
                content=f"User: {SHARED}\nUser: {BOIL}\nUser: {tail}"))
        s.close()
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM node_provenance")
        conn.execute("DELETE FROM capture_units")
        conn.execute("DELETE FROM suppressions")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        conn.close()
        s = GraphStore(db_path=db)
        try:
            fact = [n for n in s.list_nodes(node_type=NodeType.FACT, limit=10)
                    if "ZQXPG" in n.content][0]
            res = GraphOperations(s).forget_node(fact.node_id, everywhere=True)
            assert [x["unit_id"] for x in res["suppressed"]] == ["session:claude-code:sA"]
            assert not any(r["unit_id"] == "session:claude-code:sB" and r["kind"] != "wording"
                           for r in s.list_suppressions("active"))
            ctx = [n for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=10)
                   if "sB" in str(s.get_provenance(n.node_id))][0]
            assert B_ONLY in ctx.content and BOIL in ctx.content
        finally:
            s.close()


# ── "Gone everywhere" is said only when it is true ────────────────────────

class TestGoneEverywhereIsTrue:
    def test_a_graph_import_cascade_that_leaves_an_unproven_action_node_is_not_gone_everywhere(self, tmp_path):
        secret = fs.new_secret()
        src = GraphStore(db_path=str(tmp_path / "src.db"))
        IngestionPipeline(src).ingest(IngestionInput(source_id="elsewhere", content=fs.sentence(secret)))
        action = src.add_node(Node(node_type=NodeType.ACTION, label="open safe",
                                   content=f"Use the code {secret} to open the safe now."))
        graph = src.export_graph()
        src.close()
        db = str(tmp_path / "dst.db")
        s = GraphStore(db_path=db)
        s.import_graph(graph, mode="merge")
        ctx_node = s.list_nodes(node_type=NodeType.CONTEXT, limit=5)[0]
        res = GraphOperations(s).forget_node(ctx_node.node_id, cascade=True)
        still = s.get_node(action.node_id)
        s.close()
        assert still is not None and secret in still.content      # untouched: its owner is not proven
        assert res["scope"] == "forgotten from Revien"
        assert "gone everywhere" not in res["message"].lower()
        held = [l for l in res["leftovers"] if l["kind"] == "other_units"]
        assert action.node_id in {l["node_id"] for l in held}
        assert res["still_held_by"] == ["(ownership unproven)"]
        assert res["remaining"].get("nodes")      # counts content hits, so it says so

    def test_the_snapshots_get_the_same_treatment_as_the_live_database(self, tmp_path, monkeypatch):
        from revien.watch import snapshot_db
        from tests import forget_surfaces as surf
        seeded = fs.seed("pipeline", tmp_path, monkeypatch)
        ids = list(seeded.ctx_ids) + [seeded.node_with_secret]
        planted = surf.plant(seeded.db, seeded.secret, tmp_path, ids)
        s = GraphStore(db_path=seeded.db)
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        # a recall that surfaced the node, whose query has none of its words
        s.add_recall_receipt(recall_id="rcpt-plain", runtime="claude-code",
                             query="what did I say about the combination",
                             node_ids=[planted["target"]], issued_at=now,
                             expires_at=now + timedelta(hours=24))
        s.record_use_judgment(episode_key="sess#plain", node_id=planted["target"], used=True,
                              reason="r", shared_terms=["combination"], similarity=0.9,
                              query="what did I say about the combination", mode="shadow", applied=False)
        s.close()
        snaps = Path(seeded.db + ".snapshots")
        plain = snapshot_db(seeded.db, use_gzip=False, out_dir=snaps)
        packed = snapshot_db(seeded.db, use_gzip=True, out_dir=snaps)

        def vec_rows(conn):
            conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(conn)
            marks = ",".join("?" * len(planted["ids"]))
            return conn.execute(
                f"SELECT count(*) FROM vec_nodes WHERE node_id IN ({marks})", planted["ids"]).fetchone()[0]

        c = sqlite3.connect(plain)
        assert vec_rows(c) > 0
        assert c.execute("SELECT query FROM recall_receipts WHERE recall_id = 'rcpt-plain'").fetchone()[0]
        c.close()

        s = GraphStore(db_path=seeded.db)
        ops = GraphOperations(s)
        ops.training_dbs = (planted["training_db"],)
        res = ops.forget_node(seeded.ctx_ids[0], cascade=True)
        s.close()
        assert res["snapshots"]["unreadable"] == [] and res["snapshots"]["scrubbed"] == 2

        import gzip
        import shutil
        unpacked = tmp_path / "unpacked.db"
        with gzip.open(packed, "rb") as src, open(unpacked, "wb") as dst:
            shutil.copyfileobj(src, dst)
        for path in (Path(plain), unpacked):
            c = sqlite3.connect(path)
            try:
                assert c.execute("SELECT query FROM recall_receipts WHERE recall_id = 'rcpt-plain'").fetchone()[0] == ""
                assert c.execute("SELECT count(*) FROM use_judgments WHERE query != ''").fetchone()[0] == \
                    c.execute("SELECT count(*) FROM use_judgments WHERE query != '' AND episode_key != 'sess#plain' AND episode_key != 'sess#tool-1'").fetchone()[0]
                assert vec_rows(c) == 0
                assert scrub.find_residue(c, [seeded.secret]) == {}
            finally:
                c.close()

    def test_graph_import_does_not_drop_a_context_node_because_its_label_equals_a_forgotten_label(self, tmp_path):
        label = "We decided to use ZQXPG for the billing service."
        src = GraphStore(db_path=str(tmp_path / "gsrc.db"))
        d = src.add_node(Node(node_type=NodeType.DECISION, label=label, content=label))
        k = src.add_node(Node(node_type=NodeType.CONTEXT, label=label,
                              content=f"{label}\nAn unrelated sentence that must survive the import."))
        graph = src.export_graph()
        src.close()
        s = GraphStore(db_path=str(tmp_path / "gdst.db"))
        s.import_graph(graph, mode="merge")
        res = GraphOperations(s).forget_node(d.node_id)
        assert res["status"] == "forgotten"
        s.delete_node(k.node_id)                       # (so that the same file can come in again)
        out = s.import_graph(graph, mode="merge")
        back = s.get_node(k.node_id)
        gone = s.get_node(d.node_id)
        s.close()
        assert gone is None                             # the forgotten node itself is not brought back
        assert back is not None and "An unrelated sentence that must survive" in back.content
        assert out["skipped_suppressed"] == 1

    def test_revien_own_actor_strings_are_not_residue(self, store):
        c = ctx(store, "User: We use the API for the billing service today.\nUser: Another line stays here.")
        api = node(store, NodeType.ENTITY, "API", "API")
        res = GraphOperations(store).forget_node(api.node_id)
        assert res["status"] == "forgotten"
        assert "audit_log" not in res["remaining"], res["remaining"]
        assert "Still found in: audit_log" not in res["message"]


def ctx(store, text, unit=UA, **kw):
    return store.add_node(Node(node_type=NodeType.CONTEXT, label=text.split("\n")[0][:200],
                               content=text, **kw), owner=(unit, 1, 1))


def node(store, ntype, label, content, unit=UA, **kw):
    return store.add_node(Node(node_type=ntype, label=label, content=content, **kw),
                          owner=(unit, 1, 1))


# ── the reason names what changes, by unit type ───────────────────────────

class TestSuppressionReasonFitsTheUnit:
    @pytest.mark.parametrize("unit_id,noun", [
        ("session:claude-code:abc", "session file"),
        ("session:codex:abc", "session file"),
        ("vault:Notes/a.md#top", "note"),
        ("skill:/x/SKILL.md", "skill file"),
        ("file:/x/notes.md", "file"),
    ])
    def test_no_evidence_reason(self, store, unit_id, noun):
        store.add_suppression(unit_id, "file", "/nowhere/else.md", ["path:/nowhere/else.md"],
                              [], [], "fp", "test")
        block = gate.check(store, gate.UnitRef(unit_id, None), "some text")
        assert block is not None
        assert block.reason == f"no evidence the {noun} changed"

    def test_unchanged_reason_for_a_session(self, store, tmp_path):
        from revien.forget.keys import file_fingerprint
        from revien.graph.units import identity_keys, real_path
        f = tmp_path / "s.jsonl"
        f.write_text("x")
        store.add_suppression("session:claude-code:abc", "file", real_path(str(f)),
                              identity_keys(str(f)), [], [], file_fingerprint(real_path(str(f))), "test")
        block = gate.check(store, gate.UnitRef("session:claude-code:abc", str(f)), "x")
        assert block.reason == "the session file has not changed"


# ── the trained ranking model: stale mark, a reader, a retrain ────────────

class TestModelRetrain:
    @staticmethod
    def _loop(tmp_path):
        from revien.neural.training import TrainingLoop
        return TrainingLoop(db_path=str(tmp_path / "t.db"), model_dir=str(tmp_path / "models"))

    @staticmethod
    def _signals(loop, n, label="plain label"):
        conn = sqlite3.connect(loop.db_path)
        for i in range(n):
            conn.execute(
                "INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                "VALUES (?,?,?,?,?,?,?)", (f"query {i}", f"n{i}", label, "fact", 0.5, i % 2, "2026-01-01T00:00:00+00:00"))
        conn.commit()
        conn.close()

    def test_the_retrain_path_reads_the_stale_mark(self, tmp_path, monkeypatch):
        loop = self._loop(tmp_path)
        self._signals(loop, 120)
        loop._last_train_count = 120                      # trained on all of them already
        assert loop.is_ready_for_training() is False      # nothing new: no retrain due
        loop.mark_stale()
        assert loop.is_stale() is True
        assert loop.is_ready_for_training() is True       # a forget made one due
        # ...but not with too little left to train on
        small = self._loop(tmp_path / "small") if (tmp_path / "small").mkdir() is None else None
        self._signals(small, 30)
        small.mark_stale()
        assert small.is_ready_for_training() is False

    def test_a_training_run_clears_the_mark_and_tells_the_process_to_reload(self, tmp_path, monkeypatch):
        loop = self._loop(tmp_path)
        self._signals(loop, 120)
        loop.mark_stale()
        monkeypatch.setattr(type(loop), "_neural_available", lambda self: True)
        monkeypatch.setattr("revien.neural.scorer_model.NeuralScorer.train", lambda self, data: True)
        seen = []
        loop.on_trained = lambda: seen.append("reloaded")
        assert loop.train() is True
        assert loop.is_stale() is False and seen == ["reloaded"]

    def test_a_retrain_that_cannot_run_is_not_retried_on_every_recall(self, tmp_path, monkeypatch):
        loop = self._loop(tmp_path)
        self._signals(loop, 120)
        loop._last_train_count = 120
        loop.mark_stale()
        calls = []
        monkeypatch.setattr(type(loop), "train", lambda self: calls.append(1) or False)
        loop._maybe_train()
        loop._maybe_train()
        assert len(calls) == 1 and loop.is_stale() is True
        self._signals(loop, 5)
        loop._maybe_train()
        assert len(calls) == 2           # new signals are new reason to try again

    def test_the_live_scorer_is_reloaded_after_a_retrain(self, tmp_path, monkeypatch):
        from revien.retrieval.engine import RetrievalEngine
        s = GraphStore(db_path=str(tmp_path / "rl.db"))
        eng = RetrievalEngine(s, model_dir=str(tmp_path / "m2"), training_db=str(tmp_path / "t2.db"))
        self._signals(eng.training_loop, 120)
        monkeypatch.setattr(type(eng.training_loop), "_neural_available", lambda self: True)
        monkeypatch.setattr("revien.neural.scorer_model.NeuralScorer.train", lambda self, data: True)
        old = eng.neural_scorer
        assert eng.force_train() is True
        assert eng.neural_scorer is not old
        assert eng.get_training_stats()["training"]["model_stale"] is False
        s.close()

    def test_a_forget_retrains_from_the_scrubbed_signals_when_neural_ranking_is_on(self, store, tmp_path):
        loop = self._loop(tmp_path)
        c = ctx(store, "User: remember the gate code ZQXRETRAIN7.\nUser: another plain line stays here.")
        n = node(store, NodeType.FACT, "gate", "the gate code is ZQXRETRAIN7.")
        conn = sqlite3.connect(loop.db_path)
        conn.execute("INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                     "VALUES ('where is ZQXRETRAIN7', ?, 'gate', 'fact', 0.5, 1, 't')", (n.node_id,))
        conn.commit()
        conn.close()
        seen = {}

        def retrain():
            c2 = sqlite3.connect(loop.db_path)
            seen["rows_at_retrain"] = c2.execute("SELECT count(*) FROM retrieval_signals").fetchone()[0]
            c2.close()
            loop._clear_stale()
            return True

        ops = GraphOperations(store)
        ops.training_dbs = (loop.db_path,)
        ops.retrain = retrain
        res = ops.forget_node(c.node_id, cascade=True)
        assert seen["rows_at_retrain"] == 0                # the retrain saw the scrubbed signals
        assert res["model"]["retrained"] is True and res["model_stale"] is False
        assert not [l for l in res["leftovers"] if l["kind"] == "neural_model"]
        assert loop.is_stale() is False

    def test_a_retrain_that_cannot_run_leaves_the_model_stale_and_says_so(self, store, tmp_path):
        loop = self._loop(tmp_path)
        models = Path.home() / ".revien" / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "vectorizer.pkl").write_bytes(b"model")
        c = ctx(store, "User: remember the gate code ZQXRETRAIN8.")
        ops = GraphOperations(store)
        ops.training_dbs = (loop.db_path,)
        ops.retrain = lambda: False
        res = ops.forget_node(c.node_id, cascade=True)
        assert res["model_stale"] is True and res["model"]["neural_ranking"] is True
        (left,) = [l for l in res["leftovers"] if l["kind"] == "neural_model"]
        assert "could not run" in left["note"] and "POST /v1/training/run" in left["note"]
        assert loop.is_stale() is True

    def test_with_neural_ranking_off_nothing_retrains_and_the_command_is_named(self, store, tmp_path):
        loop = self._loop(tmp_path)
        models = Path.home() / ".revien" / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "vectorizer.pkl").write_bytes(b"model")
        c = ctx(store, "User: remember the gate code ZQXRETRAIN9.")
        ops = GraphOperations(store)
        ops.training_dbs = (loop.db_path,)
        res = ops.forget_node(c.node_id, cascade=True)
        assert res["model"] == {"neural_ranking": False, "retrained": False, "stale": True,
                                "marked": 1, "note": ""}
        (left,) = [l for l in res["leftovers"] if l["kind"] == "neural_model"]
        assert "Neural ranking is off, so nothing retrains it: run POST /v1/training/run" in left["note"]

    def test_the_daemon_retrains_after_a_forget_only_when_neural_ranking_is_on(self, tmp_path, monkeypatch):
        from revien.neural.training import TrainingLoop
        from fastapi.testclient import TestClient
        from revien.daemon.server import create_app
        ran = []
        monkeypatch.setattr(TrainingLoop, "train", lambda self: ran.append(1) or True)
        for flag, expect in (("1", True), ("0", False)):
            monkeypatch.setenv("REVIEN_NEURAL", flag)
            db = str(tmp_path / f"d{flag}.db")
            s = GraphStore(db_path=db)
            c = ctx(s, "User: remember the gate code ZQXRETRAIN1.")
            s.close()
            ran.clear()
            with TestClient(create_app(db_path=db)) as client:
                r = client.post(f"/v1/nodes/{c.node_id}/forget", json={"cascade": True})
            body = r.json()
            assert body["model"]["neural_ranking"] is expect
            assert (len(ran) == 1) is expect and body["model"]["retrained"] is expect

    def test_a_forget_from_the_cli_retrains_too_when_neural_ranking_is_on(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_NEURAL", "1")
        assert GraphOperations(store)._forget_engine().retrain is not None
        monkeypatch.setenv("REVIEN_NEURAL", "0")
        assert GraphOperations(store)._forget_engine().retrain is None


# ── every capture path decides under the write lock ───────────────────────

def write_locked(db_path: str) -> bool:
    """True when another connection holds SQLite's write lock right now."""
    c = sqlite3.connect(db_path, timeout=0)
    try:
        c.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as e:
        return "locked" in str(e)
    else:
        c.rollback()
        return False
    finally:
        c.close()


@pytest.fixture
def lock_log(monkeypatch):
    """Records, for every gate decision, whether the write lock was held."""
    log = []
    real = gate.check

    def spy(store, unit, content):
        log.append(write_locked(store.db_path))
        return real(store, unit, content)

    monkeypatch.setattr(gate, "check", spy)
    return log


class TestEveryPathDecidesUnderTheWriteLock:
    def test_the_refresh_path(self, tmp_path, lock_log):
        s = GraphStore(db_path=str(tmp_path / "w1.db"))
        pipe = IngestionPipeline(s)
        cap(pipe, "chatgpt:conversation:W", "User: First line of the conversation here.")
        lock_log.clear()
        cap(pipe, "chatgpt:conversation:W", "User: First line of the conversation here.\nUser: A second, newer line follows.")
        s.close()
        assert lock_log and lock_log[-1] is True      # the decision that counts is the one inside the transaction

    def test_the_append_path_undoes_inside_a_write_transaction(self, tmp_path, lock_log):
        s = GraphStore(db_path=str(tmp_path / "w2.db"))
        pipe = IngestionPipeline(s)
        cap(pipe, "chatgpt:conversation:W2", "User: First line of the conversation here.")
        s.close()
        assert lock_log[-1] is True                   # enforce_after_write

    def test_graph_import(self, tmp_path, lock_log):
        src = GraphStore(db_path=str(tmp_path / "w3s.db"))
        src.add_node(Node(node_type=NodeType.FACT, label="x", content="Some fact to import here."))
        graph = src.export_graph()
        src.close()
        s = GraphStore(db_path=str(tmp_path / "w3.db"))
        s.import_graph(graph, mode="merge")
        s.close()
        assert lock_log and all(lock_log)

    def test_the_openai_adapter(self, tmp_path, lock_log):
        from revien.adapters.openai_adapter import OpenAIAdapter
        export = tmp_path / "oa.json"
        fs._chatgpt_export(export, fs.new_secret())
        export.write_text(json.dumps(json.loads(export.read_text())[0]))
        with OpenAIAdapter(graph_path=str(tmp_path / "w4.db")) as a:
            a.ingest_conversation(str(export))
        assert lock_log and all(lock_log)

    def test_skills(self, tmp_path, lock_log):
        from revien.skills.ingest import ingest_roots
        root = tmp_path / "skills" / "deploy"
        root.mkdir(parents=True)
        (root / "SKILL.md").write_text(
            "---\nname: deploy\ndescription: Deploy the thing\ntriggers: [deploy]\n---\nRun the release script now.\n")
        s = GraphStore(db_path=str(tmp_path / "w5.db"))
        ingest_roots(s, paths=[str(tmp_path / "skills")])
        s.close()
        assert lock_log and all(lock_log)

    def test_the_vault_reconciler(self, tmp_path, lock_log):
        s = GraphStore(db_path=str(tmp_path / "w6.db"))
        from revien.distill import VaultReconciler
        from revien.semantic.index import SemanticIndex
        ent = s.add_node(Node(node_type=NodeType.ENTITY, label="Atlas", content="Atlas"))
        rec = VaultReconciler(s, str(tmp_path), semantic=SemanticIndex(s, enabled=False))
        assert rec._new_claim_node(ent.node_id, "A brand new claim typed into a note.", None) is not None
        s.close()
        assert lock_log and all(lock_log)

    def test_skill_proposals(self, tmp_path, lock_log):
        from revien.skills.proposals import propose_skills
        s = GraphStore(db_path=str(tmp_path / "w7.db"))
        for sess in ("s1", "s2", "s3"):
            for step in ("open the gate", "run the nightly job"):
                s.add_node(Node(node_type=NodeType.ACTION, label=step, content=step,
                                session_key=sess, project_key="p", origin_runtime="claude-code",
                                source_id=f"claude-code:p:{sess}"),
                           owner=(f"session:claude-code:{sess}", 1, 1))
        assert propose_skills(s, min_occurrences=3, min_sessions=2)["detected"] >= 1
        s.close()
        assert lock_log and all(lock_log)

    def test_the_denylist_is_committed_under_the_write_lock_from_its_first_statement(self, store, monkeypatch):
        flags = []
        real = store.add_suppression

        def spy(*a, **kw):
            flags.append(write_locked(store.db_path))
            return real(*a, **kw)

        monkeypatch.setattr(store, "add_suppression", spy)
        fact = a_only_fact(store)
        GraphOperations(store).forget_node(fact.node_id)
        assert flags and all(flags)


# ── the races, one test for each guard ────────────────────────────────────

class TestRaces:
    def test_a_capture_that_lands_between_the_plan_and_the_denylist_is_found_by_the_replan(self, tmp_path, monkeypatch):
        from revien.forget.engine import ForgetEngine
        db = str(tmp_path / "rp.db")
        s = GraphStore(db_path=db)
        pipe = IngestionPipeline(s)
        cap(pipe, "chatgpt:conversation:RP", "User: We decided to use ZQXREPLAN as the gate.\nUser: Another plain line stays here.")
        target = [n for n in s.list_nodes(node_type=NodeType.DECISION, limit=10) if "ZQXREPLAN" in n.content][0]
        late = []
        real = ForgetEngine._record_denylist

        def injected(self, *a, **kw):
            # a capture writes into the same unit after the first plan was made
            late.append(self.store.add_node(
                Node(node_type=NodeType.FACT, label="late",
                     content="We will use ZQXREPLAN as the gate again next week."),
                owner=("item:chatgpt:conversation:RP", 1, 1)))
            return real(self, *a, **kw)

        monkeypatch.setattr(ForgetEngine, "_record_denylist", injected)
        res = GraphOperations(s).forget_node(target.node_id)
        assert late and late[0].node_id in res["forgotten"]
        assert s.get_node(late[0].node_id) is None
        s.close()

    def test_a_forget_landing_during_the_extraction_of_a_refresh_is_not_written_back(self, tmp_path):
        """The refresh path: the pre-write strip ran before the forget, and the
        file changed again afterwards (so the unit-level record lets the
        capture through as new evidence). Only the last word, said inside a
        write transaction after the write, keeps the sentence out."""
        db = str(tmp_path / "rr.db")
        f = tmp_path / "notes.md"
        base = ("Meeting notes for the week ahead. We decided to use ZQXREFRESH as the gate for the project.\n"
                "Another plain line about parking permits stays here.")
        f.write_text(base)
        s1 = GraphStore(db_path=db)
        s2 = GraphStore(db_path=db)
        pipe = IngestionPipeline(s1)

        def capture(text):
            return pipe.ingest(IngestionInput(
                source_id="file:notes.md", ingest_key="file:notes.md", origin_runtime="file",
                origin_source="watch", content=text, metadata={"path": str(f)}))

        capture(base)
        target = [n for n in s1.list_nodes(node_type=NodeType.DECISION, limit=10) if "ZQXREFRESH" in n.content][0]
        grown = base + "\nA newer line about the budget review."
        f.write_text(grown)
        real = pipe.extractor.extract
        fired = []

        def extract(*a, **kw):
            out = real(*a, **kw)
            if not fired:
                fired.append(1)
                GraphOperations(s2).forget_node(target.node_id)
                f.write_text(grown + "\nAnd one more line, after the forget.")   # new evidence
            return out

        pipe.extractor.extract = extract
        out = capture(grown)
        assert fired and out.suppressed is False
        blob = " ".join(n.content for n in s1.list_nodes(limit=100))
        s1.close(); s2.close()
        assert "ZQXREFRESH" not in blob
        assert "budget review" in blob

    def test_a_context_removed_under_a_refresh_is_captured_as_new_instead_of_raising(self, tmp_path):
        db = str(tmp_path / "gone.db")
        s1 = GraphStore(db_path=db)
        s2 = GraphStore(db_path=db)
        pipe = IngestionPipeline(s1)
        first = cap(pipe, "chatgpt:conversation:G", "User: Plain first line of the conversation.")
        real = pipe.extractor.extract
        fired = []

        def extract(*a, **kw):
            out = real(*a, **kw)
            if not fired:
                fired.append(1)
                GraphOperations(s2).delete_node(first.context_node_id)     # the delete verb: no suppression
            return out

        pipe.extractor.extract = extract
        out = cap(pipe, "chatgpt:conversation:G",
                  "User: Plain first line of the conversation.\nUser: A later line arrives.")
        assert fired and out.suppressed is False and out.context_node_id
        assert s1.get_node(out.context_node_id) is not None
        s1.close(); s2.close()

    def test_a_forget_under_a_refresh_suppresses_it_cleanly(self, tmp_path):
        db = str(tmp_path / "gone2.db")
        s1 = GraphStore(db_path=db)
        s2 = GraphStore(db_path=db)
        pipe = IngestionPipeline(s1)
        first = cap(pipe, "chatgpt:conversation:G2", "User: Plain first line of the conversation.")
        real = pipe.extractor.extract
        fired = []

        def extract(*a, **kw):
            out = real(*a, **kw)
            if not fired:
                fired.append(1)
                GraphOperations(s2).forget_node(first.context_node_id, cascade=True)
            return out

        pipe.extractor.extract = extract
        out = cap(pipe, "chatgpt:conversation:G2",
                  "User: Plain first line of the conversation.\nUser: A later line arrives.")
        assert fired and out.suppressed is True
        assert s1.list_nodes(node_type=NodeType.CONTEXT, limit=10) == []
        s1.close(); s2.close()

    def test_an_adapter_sync_does_not_abort_when_one_item_hits_a_foreign_key_error(self):
        import asyncio
        from datetime import datetime, timezone

        from revien.daemon.scheduler import SyncScheduler

        class Pipe:
            def __init__(self):
                self.calls = []

            def ingest(self, item):
                self.calls.append(item.source_id)
                if item.source_id == "flaky" and self.calls.count("flaky") == 1:
                    raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
                if item.source_id == "broken":
                    raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
                from types import SimpleNamespace
                return SimpleNamespace(suppressed=False)

            store = None

        class Adapter:
            async def fetch_new_content(self, since):
                return [{"content": "x", "source_id": sid} for sid in ("flaky", "broken", "fine")]

            async def health_check(self):
                return True

        # a store double with just what the loop touches
        from revien.graph.store import GraphStore as GS
        import tempfile
        d = tempfile.mkdtemp()
        store = GS(db_path=str(Path(d) / "sch.db"))
        pipe = Pipe()
        pipe.store = store
        sched = SyncScheduler(pipe)
        sched.register_adapter("fake", Adapter())
        result = asyncio.new_event_loop().run_until_complete(sched.sync_one("fake"))
        store.close()
        assert result["status"] == "ok"
        assert result["items_ingested"] == 2          # flaky (on retry) and fine
        assert result["items_skipped"] == 1           # broken, twice
        assert pipe.calls.count("flaky") == 2 and pipe.calls.count("broken") == 2


# ── each capture path: the unit-level check AND the sentence-level strip ──
# (a mutation that removes either one alone must turn a test red)

def _conversation(conv_id, secret, extra=()):
    parts = [("user", f"remember that my safe code is {secret}."),
             ("assistant", f"Noted. We decided to use {secret} as the codename.")] + list(extra)
    mapping = {"root": {"id": "root", "message": None, "parent": None, "children": ["m0"]}}
    for i, (role, text) in enumerate(parts):
        mid = f"m{i}"
        mapping[mid] = {"id": mid, "parent": "root" if i == 0 else f"m{i - 1}",
                        "children": [f"m{i + 1}"] if i + 1 < len(parts) else [],
                        "message": {"id": mid, "author": {"role": role}, "create_time": 1_750_000_001 + i,
                                    "content": {"content_type": "text", "parts": [text]}}}
    return {"id": conv_id, "title": "Plans", "create_time": 1_750_000_000,
            "current_node": f"m{len(parts) - 1}", "mapping": mapping}


class TestOpenAIAdapterPaths:
    def test_a_forgotten_conversation_is_refused_by_its_id(self, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        db = str(tmp_path / "oa1.db")
        secret = fs.new_secret()
        export = tmp_path / "c1.json"
        export.write_text(json.dumps(_conversation("conv-1", secret)))
        with OpenAIAdapter(graph_path=db) as a:
            a.ingest_conversation(str(export))
        s = GraphStore(db_path=db)
        ctxs = [n.node_id for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=10)]
        GraphOperations(s).forget_node(ctxs[0], cascade=True)
        s.close()
        with OpenAIAdapter(graph_path=db) as a:
            out = a.ingest_conversation(str(export))
        assert out.get("suppressed") is True and out["node_count"] == 0
        s = GraphStore(db_path=db)
        assert s.list_nodes(limit=50) == []
        s.close()

    def test_a_copy_under_another_conversation_id_is_recognised_by_the_whole_conversation(self, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        db = str(tmp_path / "oa2.db")
        secret = fs.new_secret()
        first = tmp_path / "c1.json"
        first.write_text(json.dumps(_conversation("conv-1", secret)))
        with OpenAIAdapter(graph_path=db) as a:
            a.ingest_conversation(str(first))
        s = GraphStore(db_path=db)
        ctxs = [n.node_id for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=10)]
        GraphOperations(s).forget_node(ctxs[0], cascade=True)
        s.close()
        copy = tmp_path / "c2.json"
        copy.write_text(json.dumps(_conversation("conv-2", secret, extra=[
            ("user", "An unrelated later remark about parking permits.")])))
        with OpenAIAdapter(graph_path=db) as a:
            out = a.ingest_conversation(str(copy))
        assert not out.get("suppressed")
        s = GraphStore(db_path=db)
        blob = " ".join(n.label + n.content for n in s.list_nodes(limit=100))
        s.close()
        assert secret not in blob and "parking permits" in blob

    def test_another_conversation_that_shares_a_line_keeps_it(self, tmp_path):
        from revien.adapters.openai_adapter import OpenAIAdapter
        db = str(tmp_path / "oa3.db")
        secret = fs.new_secret()
        first = tmp_path / "c1.json"
        first.write_text(json.dumps(_conversation("conv-1", secret)))
        with OpenAIAdapter(graph_path=db) as a:
            a.ingest_conversation(str(first))
        s = GraphStore(db_path=db)
        ctxs = [n.node_id for n in s.list_nodes(node_type=NodeType.CONTEXT, limit=10)]
        GraphOperations(s).forget_node(ctxs[0], cascade=True)
        s.close()
        other = tmp_path / "c3.json"
        other.write_text(json.dumps(_conversation("conv-3", fs.new_secret(), extra=[
            ("user", "A long unrelated discussion about the quarterly budget."),
            ("assistant", "Understood, the quarterly budget is on the agenda."),
            ("user", f"Also remember that my safe code is {secret}.")])))
        with OpenAIAdapter(graph_path=db) as a:
            a.ingest_conversation(str(other))
        s = GraphStore(db_path=db)
        blob = " ".join(n.content for n in s.list_nodes(limit=100))
        s.close()
        assert "quarterly budget" in blob and f"my safe code is {secret}" in blob


class TestImportGraphPaths:
    def _forgotten_session(self, tmp_path):
        secret = fs.new_secret()
        src = GraphStore(db_path=str(tmp_path / "gs.db"))
        pipe = IngestionPipeline(src)
        out = cap(pipe, "chatgpt:conversation:GI", f"User: remember that my safe code is {secret}.\nUser: We decided to use {secret} as the codename.")
        graph = src.export_graph()
        src.close()
        dst = GraphStore(db_path=str(tmp_path / "gd.db"))
        dst.import_graph(graph, mode="merge")
        ctx_node = [n for n in dst.list_nodes(node_type=NodeType.CONTEXT, limit=10)][0]
        return secret, graph, dst, ctx_node

    def test_a_node_of_a_forgotten_unit_is_not_imported_even_when_its_text_changed(self, tmp_path):
        secret, graph, dst, ctx_node = self._forgotten_session(tmp_path)
        res = GraphOperations(dst).forget_node(ctx_node.node_id, cascade=True)
        assert any(x["unit_id"] == "item:chatgpt:conversation:GI" for x in res["suppressed"])
        for n in graph.nodes:
            if n.node_type == NodeType.CONTEXT:
                n.content = n.content + "\nNew text written after the forget."
        out = dst.import_graph(graph, mode="merge")
        ctxs = dst.list_nodes(node_type=NodeType.CONTEXT, limit=10)
        dst.close()
        assert ctxs == [] and out["skipped_suppressed"] >= 1

    def test_a_copy_without_an_owner_is_cut_by_content_and_the_rest_comes_in(self, tmp_path):
        secret, graph, dst, ctx_node = self._forgotten_session(tmp_path)
        GraphOperations(dst).forget_node(ctx_node.node_id, cascade=True)
        copy = [n for n in graph.nodes if n.node_type == NodeType.CONTEXT][0]
        copy.node_id = "copy-of-the-session"
        copy.metadata = {}                          # no ingest key: whose it is cannot be said
        copy.content = copy.content + "\nA later unrelated remark about parking permits."
        keep = [copy]
        graph.nodes = keep
        graph.edges = []
        dst.import_graph(graph, mode="merge")
        back = dst.get_node("copy-of-the-session")
        dst.close()
        assert back is not None
        assert secret not in back.content and "parking permits" in back.content


class TestSkillPaths:
    def _skill(self, tmp_path, body):
        root = tmp_path / "skills" / "deploy"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "SKILL.md"
        path.write_text("---\nname: deploy\ndescription: Deploy the thing\ntriggers: [deploy]\n---\n" + body)
        fs._age(path)
        return path

    def test_a_forgotten_skill_is_refused_until_its_file_changes(self, tmp_path):
        from revien.skills.ingest import ingest_roots
        secret = fs.new_secret()
        self._skill(tmp_path, f"Use the code {secret} to unlock the release.\n")
        s = GraphStore(db_path=str(tmp_path / "sk1.db"))
        ingest_roots(s, paths=[str(tmp_path / "skills")])
        node = s.list_nodes(node_type=NodeType.SKILL, limit=5)[0]
        GraphOperations(s).forget_node(node.node_id, cascade=True)
        again = ingest_roots(s, paths=[str(tmp_path / "skills")])
        left = s.list_nodes(node_type=NodeType.SKILL, limit=5)
        s.close()
        assert again["suppressed"] == 1 and left == []

    def test_a_changed_skill_comes_back_without_the_forgotten_sentence(self, tmp_path):
        from revien.skills.ingest import ingest_roots
        secret = fs.new_secret()
        path = self._skill(tmp_path, f"Use the code {secret} to unlock the release.\n")
        s = GraphStore(db_path=str(tmp_path / "sk2.db"))
        ingest_roots(s, paths=[str(tmp_path / "skills")])
        node = s.list_nodes(node_type=NodeType.SKILL, limit=5)[0]
        GraphOperations(s).forget_node(node.node_id, cascade=True)
        self._skill(tmp_path, f"Use the code {secret} to unlock the release.\nThen tag the build and notify the team.\n")
        again = ingest_roots(s, paths=[str(tmp_path / "skills")])
        skills = s.list_nodes(node_type=NodeType.SKILL, limit=5)
        s.close()
        assert again["suppressed"] == 0 and skills
        assert secret not in skills[0].content and "notify the team" in skills[0].content


class TestVaultReconcilePaths:
    def test_the_exact_wording_of_an_everywhere_forget_cannot_be_typed_back_into_a_note(self, tmp_path):
        from revien.distill import VaultReconciler
        from revien.semantic.index import SemanticIndex
        s = GraphStore(db_path=str(tmp_path / "vr.db"))
        ent = s.add_node(Node(node_type=NodeType.ENTITY, label="Atlas", content="Atlas"))
        sentence = "We decided to use ZQXTYPED for the billing service."
        fact = s.add_node(Node(node_type=NodeType.FACT, label="pg", content=sentence), owner=(UA, 1, 1))
        GraphOperations(s).forget_node(fact.node_id, everywhere=True)
        rec = VaultReconciler(s, str(tmp_path), semantic=SemanticIndex(s, enabled=False))
        # a different shape of the same sentence (not the node's whole text), typed back
        assert rec._new_claim_node(ent.node_id, sentence, None) is None
        s.close()


class TestTheGlobalBanCanBeLifted:
    def test_restore_lifts_the_ban_on_the_exact_wording(self, store):
        pipe = IngestionPipeline(store)
        cap(pipe, "chatgpt:conversation:B", f"User: {SHARED}\nUser: {BOIL}\nUser: {B_ONLY}")
        fact = a_only_fact(store)
        GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        from revien.forget.restore import liftable, plan_restore, restore
        (ban,) = [r for r in liftable(store) if r["kind"] == "wording"]
        assert ban["scope"] == "global" and ban["unit_id"].startswith("forget:")
        plan = plan_restore(store, ban["unit_id"])
        assert plan["status"] == "ok" and plan["lifts"][0]["id"] == ban["id"]
        done = restore(store, ban["unit_id"])
        assert done["status"] == "restored"
        b_grows(pipe)
        assert SHARED in contexts(store)[(UB,)]
