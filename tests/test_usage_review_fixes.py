"""Review findings on the usage signal: off mode on the receipt path, a receipt
for an empty recall, apply-mode atomicity, and SKILL evidence."""
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.daemon.server import create_app
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.usage import ReceiptError, UsageObserver, UseVerdict
from tests.test_usage import ANSWER, DB_MEMORY, _session, _write_session

ANSWER_TEXT = "Billing runs on PostgreSQL because concurrent writes ruled out SQLite."


def _used(evidence):
    return UseVerdict(used=True, reason="stand-in: used")


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "r.db"))
    s.add_node(Node(node_id="n-db", node_type=NodeType.FACT, label=DB_MEMORY[1],
                    content=DB_MEMORY[2], source_id="test",
                    source_type=SourceType.EXTRACTED, confidence=1.0))
    yield s
    s.close()


def _receipt(store, ids=("n-db",)):
    return UsageObserver(store).issue_receipt(
        query="billing database", node_ids=list(ids), runtime="api")


# 1. off mode ---------------------------------------------------------------

class TestOffMode:
    def test_receipt_redemption_under_off_writes_no_judgment(self, store):
        r = _receipt(store)
        res = UsageObserver(store, judge=_used, mode="off").observe_answer(
            r.recall_id, ANSWER_TEXT)
        assert store.list_use_judgments() == []
        assert res.mode == "off" and res.judgments == []
        assert res.status == "signal_off" and "off" in res.note

    def test_it_is_closed_not_flagged_and_cannot_be_redeemed_again(self, store):
        r = _receipt(store)
        UsageObserver(store, mode="off").observe_answer(r.recall_id, ANSWER_TEXT)
        assert store.get_recall_receipt(r.recall_id)["status"] == "closed_off"
        store.expire_recall_receipts(now=datetime.now(timezone.utc) + timedelta(days=2))
        assert store.list_receipt_flags() == []
        with pytest.raises(ReceiptError) as err:
            UsageObserver(store, mode="shadow", judge=_used).observe_answer(
                r.recall_id, ANSWER_TEXT)
        assert err.value.code == "already_closed"
        assert store.list_use_judgments() == []

    def test_rest_says_so_and_writes_nothing(self, monkeypatch):
        monkeypatch.setenv("REVIEN_USAGE", "off")
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with TestClient(create_app(db_path=path)) as c:
            c.post("/v1/ingest", json={"source_id": "s", "content_type": "note",
                   "content": "We chose Postgres over SQLite for billing."})
            body = c.post("/v1/recall", json={"query": "Postgres billing",
                                              "receipt": True}).json()
            r = c.post("/v1/usage/observe", json={
                "recall_id": body["recall_id"], "answer": "Postgres for billing."})
            assert r.status_code == 200, r.text
            out = r.json()
            assert out["status"] == "signal_off" and out["judgments"] == []
            again = c.post("/v1/usage/observe", json={
                "recall_id": body["recall_id"], "answer": "x"})
            assert again.status_code == 409
        s = GraphStore(db_path=path)
        assert s.list_use_judgments() == []
        s.close()

    def test_scan_path_under_off_does_nothing(self, store, tmp_path):
        _write_session(tmp_path, _session())
        ad = ClaudeCodeAdapter(session_dir=str(tmp_path / "projects"))
        rep = UsageObserver(store, judge=_used, mode="off").run_claude_code(ad)
        assert rep.judged == 0 and store.list_use_judgments() == []

    def test_observe_file_directly_under_off_also_writes_nothing(self, store, tmp_path):
        path = _write_session(tmp_path, _session())
        from revien.usage.observer import UsageReport
        obs = UsageObserver(store, judge=_used, mode="off")
        obs.observe_file(path, "claude-code:billing:sess-1",
                         UsageReport(mode="off"))
        assert store.list_use_judgments() == []


# 2. empty recall -------------------------------------------------------------

class TestEmptyRecallReceipt:
    def test_no_receipt_for_a_recall_with_no_results(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with TestClient(create_app(db_path=path)) as c:
            body = c.post("/v1/recall", json={"query": "zzqx nothing here",
                                              "receipt": True}).json()
            assert body["results"] == []
            assert body["recall_id"] is None
            assert body["receipt_expires_at"] is None
            assert "no receipt" in body["receipt_note"]
        s = GraphStore(db_path=path)
        try:
            assert s.expire_recall_receipts(
                now=datetime.now(timezone.utc) + timedelta(days=2)) == 0
            assert s.list_receipt_flags() == []
        finally:
            s.close()

    def test_a_recall_with_results_still_gets_one(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with TestClient(create_app(db_path=path)) as c:
            c.post("/v1/ingest", json={"source_id": "s", "content_type": "note",
                   "content": "We chose Postgres over SQLite for billing."})
            body = c.post("/v1/recall", json={"query": "Postgres billing",
                                              "receipt": True}).json()
            assert body["results"] and body["recall_id"]
            assert "receipt_note" not in body


# 3. apply atomicity ----------------------------------------------------------

class _Flaky:
    def __init__(self, real, fail_times=1):
        self.real, self.fail, self.calls = real, fail_times, 0

    def mark_used(self, node_id, query=None):
        self.calls += 1
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("mark_used failed")
        return self.real.mark_used(node_id, query)


def _count(store):
    return store.get_node("n-db").access_count


class TestApplyAtomicity:
    def test_a_failed_mark_used_is_not_recorded_as_applied_and_is_retried_once(
            self, store):
        engine = _Flaky(RetrievalEngine(store))
        r = _receipt(store)
        before = _count(store)
        obs = UsageObserver(store, engine=engine, judge=_used, mode="apply")
        with pytest.raises(RuntimeError):
            obs.observe_answer(r.recall_id, ANSWER_TEXT)
        (row,) = store.list_use_judgments()
        assert row["used"] and not row["applied"]
        assert _count(store) == before
        assert store.use_judgment_counts()["applied"] == 0
        assert store.get_recall_receipt(r.recall_id)["status"] == "open"
        # retry: the used-but-unapplied claim is applied exactly once
        res = obs.observe_answer(r.recall_id, ANSWER_TEXT)
        assert [j.applied for j in res.judgments] == [True]
        assert _count(store) == before + 1
        (row,) = store.list_use_judgments()
        assert row["applied"]
        assert store.use_judgment_counts()["applied"] == 1

    def test_scan_path_retries_a_failed_apply_once(self, store, tmp_path):
        _write_session(tmp_path, _session())
        ad = ClaudeCodeAdapter(session_dir=str(tmp_path / "projects"))
        engine = _Flaky(RetrievalEngine(store))
        before = _count(store)
        obs = UsageObserver(store, engine=engine, judge=_used, mode="apply")
        with pytest.raises(RuntimeError):
            obs.run_claude_code(ad)
        rep = obs.run_claude_code(ad, full=True)
        assert rep.applied == 1 and _count(store) == before + 1
        again = obs.run_claude_code(ad, full=True)
        assert again.applied == 0 and _count(store) == before + 1

    def test_shadow_rows_are_never_applied_later(self, store, tmp_path):
        _write_session(tmp_path, _session())
        ad = ClaudeCodeAdapter(session_dir=str(tmp_path / "projects"))
        UsageObserver(store, judge=_used, mode="shadow").run_claude_code(ad)
        before = _count(store)
        UsageObserver(store, engine=RetrievalEngine(store), judge=_used,
                      mode="apply").run_claude_code(ad, full=True)
        assert _count(store) == before

    def test_only_one_of_many_concurrent_claims_wins(self, tmp_path):
        path = str(tmp_path / "c.db")
        s0 = GraphStore(db_path=path)
        s0.record_use_judgment(episode_key="e", node_id="n", used=True, reason="",
                               shared_terms=[], similarity=None, query="q",
                               mode="apply", applied=False)
        s0.close()
        wins, barrier = [], threading.Barrier(8)

        def worker():
            s = GraphStore(db_path=path)
            try:
                barrier.wait()
                wins.append(s.claim_use_application("e", "n"))
            finally:
                s.close()

        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert wins.count(True) == 1

    def test_two_concurrent_retries_apply_once(self, tmp_path):
        path = str(tmp_path / "c2.db")
        s0 = GraphStore(db_path=path)
        s0.add_node(Node(node_id="n-db", node_type=NodeType.FACT,
                         label=DB_MEMORY[1], content=DB_MEMORY[2], source_id="t",
                         source_type=SourceType.EXTRACTED, confidence=1.0))
        r = UsageObserver(s0).issue_receipt(query="q", node_ids=["n-db"], runtime="api")
        engine = _Flaky(RetrievalEngine(s0))
        with pytest.raises(RuntimeError):
            UsageObserver(s0, engine=engine, judge=_used, mode="apply"
                          ).observe_answer(r.recall_id, ANSWER_TEXT)
        s0.close()
        calls, barrier = [], threading.Barrier(2)

        class Counting:
            def __init__(self, real):
                self.real = real

            def mark_used(self, nid, q=None):
                calls.append(nid)
                return self.real.mark_used(nid, q)

        def worker():
            s = GraphStore(db_path=path)
            try:
                obs = UsageObserver(s, engine=Counting(RetrievalEngine(s)),
                                    judge=_used, mode="apply")
                barrier.wait()
                try:
                    obs.observe_answer(r.recall_id, ANSWER_TEXT)
                except ReceiptError:
                    pass
            finally:
                s.close()

        ts = [threading.Thread(target=worker) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert calls == ["n-db"]
        s = GraphStore(db_path=path)
        assert s.get_node("n-db").access_count == 1
        assert s.use_judgment_counts()["applied"] == 1
        s.close()


# 4. SKILL evidence ----------------------------------------------------------

class TestSkillEvidence:
    BODY = "Step one: rotate the zebrafish credential. Step two: flush quokka caches."

    @pytest.fixture
    def skill_store(self, tmp_path):
        s = GraphStore(db_path=str(tmp_path / "sk.db"))
        s.add_node(Node(node_id="sk", node_type=NodeType.SKILL, label="deploy-runbook",
                        content=self.BODY, source_id="skill:deploy",
                        metadata={"description": "Deploy the billing service",
                                  "triggers": ["deploy", "release"]},
                        source_type=SourceType.EXTRACTED, confidence=1.0))
        yield s
        s.close()

    def test_evidence_uses_the_index_row_recall_returned(self, skill_store):
        seen = []

        def judge(ev):
            seen.append(ev)
            return UseVerdict(used=True, reason="x")

        res = RetrievalEngine(skill_store).recall("deploy billing release")
        shown = next(r.content for r in res.results if r.node_id == "sk")
        assert "zebrafish" not in shown
        rc = UsageObserver(skill_store).issue_receipt(
            query="deploy billing release", node_ids=["sk"], runtime="api")
        UsageObserver(skill_store, judge=judge).observe_answer(
            rc.recall_id, "Rotate the zebrafish credential then flush quokka caches "
                          "to deploy the billing service release.")
        (row,) = skill_store.list_use_judgments()
        assert not {"zebrafish", "quokka"} & set(row["shared_terms"])
        assert all("zebrafish" not in t and "quokka" not in t for t in row["shared_terms"])
        assert seen and not {"zebrafish", "quokka"} & set(seen[0].shared_terms)

    def test_the_projection_is_the_engines_own_function(self):
        from revien.retrieval import engine
        from revien.usage import observer
        assert observer.recalled_content is engine.recalled_content
