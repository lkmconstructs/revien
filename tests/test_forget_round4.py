"""Round 4 of the independent review: --everywhere keeps its own contract,
"Gone everywhere" is never said while the wording survives, Codex identity,
a busy database, the real reason a retrain did not run, and a test per
claim in the CHANGELOG and the spec.

Every test here fails at 481e5ad."""
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from revien.forget import gate, scrub
from revien.forget.engine import ForgetEngine
from revien.forget.keys import fingerprint
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests import forget_seed as fs

REPO = Path(__file__).resolve().parents[1]
TARGET = "We decided to use Kubernetes with ZQXHARBOR as the deployment target."
A_TAIL = "Alpha tail about the weather in Lisbon today."
B_TAIL = "Session B also covered the Denver office lease."


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "r4.db"))
    yield s
    s.close()


def cc(pipe, sid, text):
    return pipe.ingest(IngestionInput(
        source_id=f"claude-code:proj:{sid}", ingest_key=f"claude-code:proj:{sid}",
        origin_runtime="claude-code", session_key=sid, project_key="proj", content=text))


def two_claude_sessions(store):
    pipe = IngestionPipeline(store)
    cc(pipe, "sA", f"User: {TARGET}\nUser: {A_TAIL}")
    cc(pipe, "sB", f"User: {TARGET}\nUser: {B_TAIL}")
    return pipe


def nodes_by_id(store):
    return {n.node_id: n for n in store.list_nodes(limit=500)}


def upgrade(db):
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM node_provenance")
    conn.execute("DELETE FROM capture_units")
    conn.execute("DELETE FROM suppressions")
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()


# ── A. --everywhere keeps its own contract ───────────────────────────────

class TestEverywhereContract:
    def test_a_node_two_sources_wrote_loses_the_wording_but_nothing_else_is_deleted_or_suppressed(self, store):
        two_claude_sessions(store)
        before = nodes_by_id(store)
        fact = [n for n in before.values() if n.node_type == NodeType.FACT][0]
        entity = [n for n in before.values() if n.label == "Kubernetes"][0]
        assert len(store.get_provenance(entity.node_id)) == 2       # co-owned by sA and sB
        res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        after = nodes_by_id(store)
        assert set(before) - set(after) == {fact.node_id}          # only the node that was named
        assert after[entity.node_id].content == entity.content      # the co-owned entity is whole
        assert {r["unit_id"] for r in store.get_provenance(entity.node_id)} == {
            "session:claude-code:sA", "session:claude-code:sB"}
        assert res["suppressed"] == [] and res["cut_back"] == []
        assert "session:claude-code:sB" not in json.dumps(res["suppressed"])
        # both conversations were cut in place, and both are listed there
        cut = {c["node_id"]: c for c in res["cut_in_place"]}
        for sid in ("sA", "sB"):
            ctx_node = [n for n in after.values() if n.node_type == NodeType.CONTEXT
                        and f"session:claude-code:{sid}" in str(store.get_provenance(n.node_id))][0]
            assert ctx_node.node_id in cut and "ZQXHARBOR" not in ctx_node.content
        assert not res["wording_cut"] or all(w["context_node_id"] in cut for w in res["wording_cut"])
        assert "several sources" in res["identity_note"] and "none of them is suppressed" in res["identity_note"]
        # the label that was a piece of the cut wording went with it, so nothing is left to find
        assert not any("ZQXHARBOR" in n.label + n.content for n in after.values())
        assert res["scope"] == "gone everywhere" and res["remaining"] == {}

    def test_the_message_says_what_happened_not_a_template(self, store):
        two_claude_sessions(store)
        fact = [n for n in store.list_nodes(node_type=NodeType.FACT, limit=10)][0]
        res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        msg = res["message"]
        assert "suppresses future imports" not in msg          # nothing was suppressed
        assert "suppresses no source" in msg
        assert "cut back" not in msg.lower().replace("never cut back", "")
        assert "nothing deleted, nothing suppressed" not in msg
        for c in res["cut_in_place"]:
            assert c["node_id"] in msg

    def test_on_an_upgraded_database_an_unproven_node_is_neither_cut_back_nor_leads_to_a_suppression(self, tmp_path):
        db = str(tmp_path / "up4.db")
        s = GraphStore(db_path=db)
        two_claude_sessions(s)
        s.close()
        upgrade(db)
        s = GraphStore(db_path=db)
        try:
            before = nodes_by_id(s)
            fact = [n for n in before.values() if n.node_type == NodeType.FACT][0]
            entity = [n for n in before.values() if n.label == "Kubernetes"][0]
            res = GraphOperations(s).forget_node(fact.node_id, everywhere=True)
            after = nodes_by_id(s)
            assert [x["unit_id"] for x in res["suppressed"]] == ["session:claude-code:sA"]    # the target's own source key
            assert not any(r["unit_id"] == "session:claude-code:sB" and r["kind"] != "wording"
                           for r in s.list_suppressions("active"))
            assert res["cut_back"] == [] and "cut back" not in res["message"].lower()
            assert set(before) - set(after) == {fact.node_id}
            assert after[entity.node_id].content == entity.content and after[entity.node_id].label == "Kubernetes"
        finally:
            s.close()

    def test_a_node_another_source_vouched_for_by_name_is_cut_in_place_not_cut_back(self, store):
        a = "session:claude-code:sA"
        store.add_node(Node(node_type=NodeType.CONTEXT, label="A", content=f"User: {TARGET}\nUser: {A_TAIL}",
                            metadata={"ingest_key": "claude-code:proj:sA"}), owner=(a, 1, 1))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="f", content=TARGET), owner=(a, 1, 1))
        shared = store.add_node(Node(node_type=NodeType.DECISION, label="Deployment decision",
                                     content=f"{TARGET} Also a different reminder about tulips."), owner=(a, 1, 1))
        store.record_provenance(shared.node_id, "session:claude-code:sB", label_ok=True, wording_ok=False)
        res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        survivor = store.get_node(shared.node_id)
        assert survivor is not None and survivor.label == "Deployment decision"
        assert survivor.content == "Also a different reminder about tulips."       # not reduced to its label
        assert res["cut_back"] == [] and shared.node_id in {c["node_id"] for c in res["cut_in_place"]}
        assert {r["unit_id"] for r in store.get_provenance(shared.node_id)} >= {"session:claude-code:sB"}
        # (a plain forget of the same node does cut it back to its label)
        store2 = GraphStore(db_path=store.db_path + ".plain")
        try:
            store2.add_node(Node(node_type=NodeType.CONTEXT, label="A", content=f"User: {TARGET}\nUser: {A_TAIL}"), owner=(a, 1, 1))
            f2 = store2.add_node(Node(node_type=NodeType.FACT, label="f", content=TARGET), owner=(a, 1, 1))
            s2 = store2.add_node(Node(node_type=NodeType.DECISION, label="Deployment decision",
                                      content=f"{TARGET} Also a different reminder about tulips."), owner=(a, 1, 1))
            store2.record_provenance(s2.node_id, "session:claude-code:sB", label_ok=True, wording_ok=False)
            plain = GraphOperations(store2).forget_node(f2.node_id)
            assert s2.node_id in plain["cut_back"]
        finally:
            store2.close()

    def test_a_single_known_writer_is_the_only_source_suppressed(self, store):
        sentence = TARGET
        store.add_node(Node(node_type=NodeType.CONTEXT, label="A", content=f"User: {sentence}\nUser: {A_TAIL}",
                            metadata={"ingest_key": "claude-code:proj:sA"}),
                       owner=("session:claude-code:sA", 1, 1))
        fact = store.add_node(Node(node_type=NodeType.FACT, label="f", content=sentence),
                              owner=("session:claude-code:sA", 1, 1))
        b_ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="B", content=f"User: {sentence}\nUser: {B_TAIL}"),
                               owner=("session:claude-code:sB", 1, 1))
        res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
        assert [x["unit_id"] for x in res["suppressed"]] == ["session:claude-code:sA"]
        assert store.get_node(b_ctx.node_id) is not None and B_TAIL in store.get_node(b_ctx.node_id).content
        assert {c["node_id"] for c in res["cut_in_place"]} >= {b_ctx.node_id}

    @pytest.mark.parametrize("mode", ["plain", "everywhere", "cascade", "cascade_everywhere"])
    def test_no_sentence_of_the_message_is_false(self, store, mode):
        two_claude_sessions(store)
        ops = GraphOperations(store)
        if mode.startswith("cascade"):
            a_ctx = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=10)
                     if "session:claude-code:sA" in str(store.get_provenance(n.node_id))][0]
            res = ops.forget_node(a_ctx.node_id, cascade=True, everywhere=mode.endswith("everywhere"))
        else:
            fact = [n for n in store.list_nodes(node_type=NodeType.FACT, limit=10)][0]
            res = ops.forget_node(fact.node_id, everywhere=(mode == "everywhere"))
        msg = res["message"]
        assert "nothing deleted, nothing suppressed" not in msg      # (a template, true or not)
        if not res["suppressed"]:
            assert "suppresses future imports" not in msg
        else:
            assert "suppresses no source" not in msg
        if not res["cut_in_place"]:
            assert "cut the exact wording in place" not in msg
        else:
            assert "cut the exact wording in place" in msg
        if not res["cut_back"]:
            assert "cut back to their label" not in msg
        if res["scope"] == "gone everywhere":
            assert res["leftovers"] == [] and res["remaining"] == {} and res["still_held_by"] == []
            assert msg.startswith("Gone everywhere.")
        else:
            assert msg.startswith("Forgotten from Revien.")


# ── B. "Gone everywhere" while the wording survives ───────────────────────

class TestGoneEverywhereNeverWhileTheWordingSurvives:
    SECRET = "Mallard0xHarbor"

    def _merged_fact(self, store, others=11):
        a = "session:claude-code:A"
        ctx_a = store.add_node(Node(
            node_type=NodeType.CONTEXT, label="A",
            content=f"User: We picked {self.SECRET} as its harbor code for the project.\nUser: Alpha only remark about the weather."),
            owner=(a, 1, 1))
        fact = store.add_node(Node(
            node_type=NodeType.FACT, label=f"{self.SECRET} as its harbor code",
            content=f"We picked {self.SECRET} as its harbor code for the project."), owner=(a, 1, 1))
        for i in range(others):
            store.record_provenance(fact.node_id, f"session:claude-code:O{i}", label_ok=True, wording_ok=False)
        return ctx_a, fact

    def test_a_label_that_is_the_claim_is_residue_and_its_holders_are_named(self, store):
        ctx_a, fact = self._merged_fact(store)
        res = GraphOperations(store).forget_node(ctx_a.node_id, cascade=True)
        survivor = store.get_node(fact.node_id)
        assert survivor is not None and self.SECRET in survivor.label        # the name stays: others vouched for it
        assert res["scope"] == "forgotten from Revien"
        assert res["message"].startswith("Forgotten from Revien.")
        assert len(res["still_held_by"]) == 11 and "session:claude-code:O3" in res["still_held_by"]
        held = {l["node_id"] for l in res["leftovers"] if l["kind"] == "other_units"}
        assert fact.node_id in held

    def test_the_training_signals_of_a_cut_back_node_are_scrubbed_so_a_retrain_cannot_learn_the_label(self, store, tmp_path):
        from revien.neural.training import TrainingLoop
        loop = TrainingLoop(db_path=str(tmp_path / "t4.db"), model_dir=str(tmp_path / "m4"))
        ctx_a, fact = self._merged_fact(store)
        conn = sqlite3.connect(loop.db_path)
        for i in range(5):
            conn.execute(
                "INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                "VALUES (?,?,?,?,?,?,?)", (f"harbor question {i}", fact.node_id,
                                           f"{self.SECRET} as its harbor code", "fact", 0.5, 1, "t"))
        conn.execute("INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                     "VALUES ('unrelated', 'other', 'plain label', 'fact', 0.5, 0, 't')")
        conn.commit()
        conn.close()
        ops = GraphOperations(store)
        ops.training_dbs = (loop.db_path,)
        seen = {}

        def retrain():
            seen["data"] = json.dumps(loop.get_training_data())
            return True

        ops.retrain = retrain
        ops.forget_node(ctx_a.node_id, cascade=True)
        assert self.SECRET not in seen["data"]                  # what the retrain read
        assert "plain label" in seen["data"]                    # ...and it kept what is unrelated
        assert loop.get_signal_count() == 1

    @pytest.mark.parametrize("hand_nodes", [1, 9, 25])
    def test_any_number_of_hand_nodes_holding_the_word_keeps_it_from_being_gone_everywhere(self, store, hand_nodes):
        word = "ZQXMALLARDWORD"
        ctx_a = store.add_node(Node(node_type=NodeType.CONTEXT, label="A",
                                    content=f"User: Remember that the dock code is {word}.\nUser: Another plain line stays."),
                               owner=("session:claude-code:A", 1, 1))
        ids = [store.add_node(Node(node_type=NodeType.FACT, label=f"hand {i}",
                                   content=f"Hand-written note number {i} mentions {word} again.")).node_id
               for i in range(hand_nodes)]
        res = GraphOperations(store).forget_node(ctx_a.node_id, cascade=True)
        held = {l["node_id"] for l in res["leftovers"] if l["kind"] == "other_units"}
        assert set(ids) <= held
        assert res["scope"] == "forgotten from Revien" and res["remaining"].get("nodes")

    def test_one_other_unit_using_the_word_and_one_hand_node_is_still_not_gone_everywhere(self, store):
        word = "ZQXSHAREDWORD"
        ctx_a = store.add_node(Node(node_type=NodeType.CONTEXT, label="A",
                                    content=f"User: The dock code is {word} for now.\nUser: Another plain line stays."),
                               owner=("session:claude-code:A", 1, 1))
        store.add_node(Node(node_type=NodeType.CONTEXT, label="B", content=f"User: Bravo also saw {word} on a label."),
                       owner=("session:claude-code:B", 1, 1))
        hand = store.add_node(Node(node_type=NodeType.FACT, label="hand", content=f"A note: {word} was mentioned."))
        res = GraphOperations(store).forget_node(ctx_a.node_id, cascade=True)
        assert hand.node_id in {l["node_id"] for l in res["leftovers"] if l["kind"] == "other_units"}
        assert res["scope"] == "forgotten from Revien"

    def test_a_word_everyone_uses_does_not_make_every_forget_unclean(self, store):
        for i in range(6):
            store.add_node(Node(node_type=NodeType.CONTEXT, label=f"c{i}",
                                content=f"User: The deployment pipeline number {i} runs fine."),
                           owner=(f"session:claude-code:C{i}", 1, 1))
        hand = store.add_node(Node(node_type=NodeType.FACT, label="hand", content="Hand note about the deployment pipeline."))
        ctx_a = store.add_node(Node(node_type=NodeType.CONTEXT, label="A",
                                    content="User: The deployment pipeline is down.\nUser: Another plain line stays."),
                               owner=("session:claude-code:A", 1, 1))
        res = GraphOperations(store).forget_node(ctx_a.node_id, cascade=True)
        assert hand.node_id not in {l["node_id"] for l in res["leftovers"] if l["kind"] == "other_units"}

    @pytest.mark.parametrize("path", sorted(fs.ALL_PATHS))
    def test_gone_everywhere_is_never_said_while_the_secret_is_in_the_database(self, path, tmp_path, monkeypatch):
        if path == "langchain":
            from revien.adapters.langchain_adapter import LANGCHAIN_AVAILABLE
            if not LANGCHAIN_AVAILABLE:
                pytest.skip("langchain-core is not installed")
        seeded = fs.seed(path, tmp_path, monkeypatch)
        handle = (seeded.ctx_ids or [seeded.node_with_secret])[0]
        s = GraphStore(db_path=seeded.db)
        res = GraphOperations(s).forget_node(handle, cascade=True)
        s.close()
        conn = sqlite3.connect(seeded.db)
        try:
            in_db = scrub.find_residue(conn, [seeded.secret])
        finally:
            conn.close()
        if res["scope"] == "gone everywhere":
            assert in_db == {}, (path, in_db)
        else:
            assert res["leftovers"] or res["remaining"]


# ── Codex identity ───────────────────────────────────────────────────────

class TestCodexIdentity:
    def test_the_adapter_reads_the_session_id_and_the_unit_id_does_not_move(self, tmp_path, monkeypatch):
        from revien.adapters.codex import CodexAdapter
        seeded = fs.seed("codex", tmp_path, monkeypatch)
        items = fs.run_async(CodexAdapter(session_dir=str(seeded.source_file.parents[3])).fetch_new_content(
            __import__("datetime").datetime(2000, 1, 1, tzinfo=__import__("datetime").timezone.utc)))
        assert items and items[0]["metadata"]["session_id"] == "0197-codex-forgotten"
        s = GraphStore(db_path=seeded.db)
        try:
            units = {r["unit_id"] for n in s.list_nodes(limit=100) for r in s.get_provenance(n.node_id)}
            keys = s.get_unit(next(u for u in units if u.startswith("session:codex:")))["path_keys"]
        finally:
            s.close()
        assert any(u.startswith("session:codex:rollout-") for u in units)      # still the file name
        assert "sid:codex:0197-codex-forgotten" in keys

    def test_an_old_database_without_the_key_keeps_working(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "o.db"))
        unit = gate.UnitRef("session:codex:rollout-x", None)        # no sid: how every earlier capture looked
        gate.register(s, unit)
        assert s.get_unit("session:codex:rollout-x")["path_keys"] == []
        s.close()


# ── a busy database ─────────────────────────────────────────────────────

class TestBusyDatabase:
    def _hold(self, db):
        other = sqlite3.connect(db, timeout=0, check_same_thread=False)
        other.execute("BEGIN IMMEDIATE")
        return other

    def test_a_forget_answers_busy_and_changes_nothing_while_a_capture_holds_the_write_lock(self, store, tmp_path, monkeypatch):
        monkeypatch.setattr(ForgetEngine, "BUSY_WAIT_SECONDS", 0.3)
        store._get_conn().execute("PRAGMA busy_timeout = 50")
        c = store.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: some words to forget here."),
                           owner=("session:claude-code:Z", 1, 1))
        before = nodes_by_id(store)
        other = self._hold(store.db_path)
        try:
            res = GraphOperations(store).forget_node(c.node_id, cascade=True)
        finally:
            other.rollback()
            other.close()
        assert res["status"] == "busy" and res["reason"] == "database_busy"
        assert res["denylist_written"] is False
        assert "Revien is busy writing; nothing was changed" in res["message"]
        assert nodes_by_id(store).keys() == before.keys()
        assert store.list_suppressions("active", include_wording=True) == []
        # ...and once the capture is done, the same forget works
        assert GraphOperations(store).forget_node(c.node_id, cascade=True)["status"] == "forgotten"

    def test_when_only_the_removal_is_blocked_the_response_says_the_denylist_is_already_in_force(self, store, monkeypatch):
        monkeypatch.setattr(ForgetEngine, "BUSY_WAIT_SECONDS", 0.3)
        store._get_conn().execute("PRAGMA busy_timeout = 50")
        c = store.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: some words to forget here."),
                           owner=("session:claude-code:Z", 1, 1))
        real = ForgetEngine._record_denylist
        held = []

        def then_busy(self_, *a, **kw):
            out = real(self_, *a, **kw)
            held.append(self._hold(store.db_path))
            return out

        monkeypatch.setattr(ForgetEngine, "_record_denylist", then_busy)
        try:
            res = GraphOperations(store).forget_node(c.node_id, cascade=True)
        finally:
            for h in held:
                h.rollback()
                h.close()
        assert res["status"] == "busy" and res["denylist_written"] is True
        assert "already kept out of new captures" in res["message"]
        assert "nothing was changed" not in res["message"]      # the denylist was written, so that would be false
        assert store.get_node(c.node_id) is not None            # nothing was removed yet

    def test_the_retry_waits_for_a_short_lock(self, store, monkeypatch):
        monkeypatch.setattr(ForgetEngine, "BUSY_WAIT_SECONDS", 5.0)
        store._get_conn().execute("PRAGMA busy_timeout = 50")
        c = store.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: some words to forget here."),
                           owner=("session:claude-code:Z", 1, 1))
        other = self._hold(store.db_path)
        timer = threading.Timer(0.4, lambda: (other.rollback(), other.close()))
        timer.start()
        res = GraphOperations(store).forget_node(c.node_id, cascade=True)
        timer.join()
        assert res["status"] == "forgotten"

    def test_rest_and_cli_refuse_cleanly(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from fastapi.testclient import TestClient

        from revien.cli import main
        from revien.daemon.server import create_app
        db = str(tmp_path / "busy.db")
        s = GraphStore(db_path=db)
        c = s.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: some words to forget here."),
                       owner=("session:claude-code:Z", 1, 1))
        s.close()
        busy = ForgetEngine._busy(c.node_id, committed=False)
        monkeypatch.setattr(GraphOperations, "forget_node", lambda self, *a, **k: busy)
        with TestClient(create_app(db_path=db)) as client:
            r = client.post(f"/v1/nodes/{c.node_id}/forget", json={"cascade": True})
        assert r.status_code == 503 and r.json()["detail"]["status"] == "busy"
        out = CliRunner().invoke(main, ["forget", c.node_id, "--db", db, "--yes", "--cascade"])
        assert out.exit_code == 1 and "Revien is busy writing; nothing was changed" in out.output


# ── the real reason a retrain did not run ───────────────────────────────

class TestRetrainReason:
    def test_the_training_loop_keeps_the_reason(self, tmp_path, monkeypatch):
        from revien.neural.training import TrainingLoop
        loop = TrainingLoop(db_path=str(tmp_path / "t.db"), model_dir=str(tmp_path / "m"))
        assert loop.train() is False and "not installed" in loop.last_failure     # (no sklearn in this environment)
        monkeypatch.setattr(type(loop), "_neural_available", lambda self: True)
        assert loop.train() is False and "only 0 retrieval signals are left, 100 are needed" == loop.last_failure
        conn = sqlite3.connect(loop.db_path)
        for i in range(240):
            conn.execute("INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                         "VALUES (?,?,?,?,?,?,?)", (f"q{i}", f"n{i}", "label", "fact", 0.5, 1, "t"))
        conn.commit()
        conn.close()

        def one_class(self, data):
            self.last_error = "ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1"
            return False

        monkeypatch.setattr("revien.neural.scorer_model.NeuralScorer.train", one_class)
        assert loop.train() is False
        assert "only one class" in loop.last_failure
        monkeypatch.setattr("revien.neural.scorer_model.NeuralScorer.train", lambda self, data: True)
        assert loop.train() is True and loop.last_failure is None

    def test_the_scorer_says_why(self):
        from revien.neural.scorer_model import NeuralScorer
        import tempfile
        scorer = NeuralScorer(model_dir=tempfile.mkdtemp())
        assert scorer.train([]) is False and scorer.last_error

    def test_the_forget_response_names_that_reason_not_a_guess(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from revien.daemon.server import create_app
        from revien.neural.training import TrainingLoop
        monkeypatch.setenv("REVIEN_NEURAL", "1")
        monkeypatch.setattr(TrainingLoop, "_neural_available", lambda self: True)

        def one_class(self, data):
            self.last_error = "ValueError: needs samples of at least 2 classes, only one class: 1"
            return False

        monkeypatch.setattr("revien.neural.scorer_model.NeuralScorer.train", one_class)
        db = str(tmp_path / "rr.db")
        s = GraphStore(db_path=db)
        c = s.add_node(Node(node_type=NodeType.CONTEXT, label="c", content="User: remember the gate code ZQXREASON1."),
                       owner=("session:claude-code:Z", 1, 1))
        s.close()
        app = create_app(db_path=db)
        with TestClient(app) as client:
            conn = sqlite3.connect(app.state.engine.training_loop.db_path)
            for i in range(240):
                conn.execute("INSERT INTO retrieval_signals (query, node_id, node_label, node_type, score, was_used, timestamp) "
                             "VALUES (?,?,?,?,?,?,?)", (f"q{i}", f"n{i}", "label", "fact", 0.5, 1, "t"))
            conn.commit()
            conn.close()
            body = client.post(f"/v1/nodes/{c.node_id}/forget", json={"cascade": True}).json()
        assert body["model"]["retrained"] is False
        assert "only one class" in body["model"]["note"]
        assert "too few" not in body["model"]["note"]


# ── gate bindings, one test each ────────────────────────────────────────

def _put(store, path_keys, sentence, scope="origin", signature=()):
    store.add_suppression("forget:test-" + str(len(store.list_suppressions('active', include_wording=True))),
                          "wording", None, list(path_keys), [], [], None, "test",
                          sentence_fps=[fingerprint(sentence)], signature_fps=list(signature), scope=scope)


class TestBindings:
    S = "A sentence that was forgotten from one origin only."

    def test_the_unit_binding(self, store):
        _put(store, ["unit:session:codex:X"], self.S)
        long_fps, _ = gate.known(store, gate.UnitRef("session:codex:X", None), "unrelated text")
        assert fingerprint(self.S) in long_fps
        long_fps, _ = gate.known(store, gate.UnitRef("session:codex:Y", None), "unrelated text")
        assert long_fps == set()

    def test_the_session_id_binding(self, store):
        _put(store, ["sid:codex:abc"], self.S)
        long_fps, _ = gate.known(store, gate.UnitRef("session:codex:renamed", None, sid="sid:codex:abc"), "x")
        assert fingerprint(self.S) in long_fps
        long_fps, _ = gate.known(store, gate.UnitRef("session:codex:renamed", None, sid="sid:codex:other"), "x")
        assert long_fps == set()

    def test_the_file_binding(self, store, tmp_path):
        from revien.graph.units import identity_keys
        f = tmp_path / "a.md"
        f.write_text("x")
        _put(store, identity_keys(str(f)), self.S)
        long_fps, _ = gate.known(store, gate.UnitRef(None, str(f)), "x")
        assert fingerprint(self.S) in long_fps
        g = tmp_path / "b.md"
        g.write_text("y")
        long_fps, _ = gate.known(store, gate.UnitRef(None, str(g)), "y")
        assert long_fps == set()

    def test_a_global_entry_applies_everywhere(self, store):
        _put(store, [], self.S, scope="global")
        long_fps, _ = gate.known(store, gate.UnitRef(None, None), "x")
        assert fingerprint(self.S) in long_fps
