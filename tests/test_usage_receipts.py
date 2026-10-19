"""
Usage receipts (N1, any runtime): a recall issued with a receipt is judged
when its answer comes back, through the observer, the REST endpoint, Hermes
or Ollama. A receipt not redeemed within 24h becomes a flag that stays until
a person dismisses it.

The judgment rule is replaced with stand-ins here; tests/test_usage_judge.py
covers the real rule.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex
from revien.usage import ReceiptError, UsageObserver, UseVerdict

ANSWER = "Billing runs on PostgreSQL because the worker pool writes concurrently."


def _used(evidence):
    return UseVerdict(used=True, reason="stand-in: used")


def _boom(evidence):
    raise RuntimeError("rule crashed")


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "revien.db"))
    for node_id, label in (("n-db", "Billing runs on PostgreSQL"),
                           ("n-cache", "Sessions are cached in Redis")):
        s.add_node(Node(node_id=node_id, node_type=NodeType.FACT, label=label,
                        content=label + ".", source_id="test",
                        source_type=SourceType.EXTRACTED, confidence=1.0))
    yield s
    s.close()


def _issue(store, node_ids=("n-db", "n-cache"), runtime="api"):
    return UsageObserver(store).issue_receipt(
        query="billing database", node_ids=list(node_ids), runtime=runtime)


class TestReceipts:
    def test_observe_returns_a_verdict_per_memory(self, store):
        receipt = _issue(store)
        result = UsageObserver(store, judge=_used).observe_answer(
            receipt.recall_id, ANSWER, prompt="what backs billing?")
        assert result.runtime == "api" and result.mode == "shadow"
        assert {j.node_id for j in result.judgments} == {"n-db", "n-cache"}
        db = next(j for j in result.judgments if j.node_id == "n-db")
        assert db.verdict == "used" and not db.applied
        assert db.label == "Billing runs on PostgreSQL"
        assert "postgresql" in db.shared_terms
        assert store.get_recall_receipt(receipt.recall_id)["status"] == "redeemed"
        rows = store.list_use_judgments()
        assert {r["episode_key"] for r in rows} == {f"receipt:api#{receipt.recall_id}"}

    def test_a_receipt_redeems_once(self, store):
        receipt = _issue(store)
        UsageObserver(store, judge=_used).observe_answer(receipt.recall_id, ANSWER)
        with pytest.raises(ReceiptError) as err:
            UsageObserver(store, judge=_used).observe_answer(receipt.recall_id, ANSWER)
        assert err.value.code == "already_redeemed"
        assert store.use_judgment_counts()["judged"] == 2  # not 4

    def test_unknown_and_empty(self, store):
        with pytest.raises(ReceiptError) as err:
            UsageObserver(store, judge=_used).observe_answer("r_nope", ANSWER)
        assert err.value.code == "unknown"
        receipt = _issue(store)
        with pytest.raises(ReceiptError) as err:
            UsageObserver(store, judge=_used).observe_answer(receipt.recall_id, "  ")
        assert err.value.code == "empty_answer"
        assert store.get_recall_receipt(receipt.recall_id)["status"] == "open"

    def test_apply_counts_uses(self, store):
        receipt = _issue(store, node_ids=("n-db",))
        before = store.get_node("n-db").access_count
        result = UsageObserver(store, engine=RetrievalEngine(store), judge=_used,
                               mode="apply").observe_answer(receipt.recall_id, ANSWER)
        assert result.judgments[0].applied
        assert store.get_node("n-db").access_count == before + 1

    def test_a_failed_judgment_leaves_the_receipt_open(self, store):
        receipt = _issue(store)
        with pytest.raises(RuntimeError):
            UsageObserver(store, judge=_boom).observe_answer(receipt.recall_id, ANSWER)
        assert store.get_recall_receipt(receipt.recall_id)["status"] == "open"


class TestFlags:
    def _expire(self, store, receipt):
        later = receipt.expires_at + timedelta(seconds=1)
        return store.expire_recall_receipts(now=later)

    def test_an_unjudged_receipt_becomes_a_flag_not_a_deletion(self, store):
        receipt = _issue(store)
        assert store.list_receipt_flags() == []
        assert self._expire(store, receipt) == 1
        (flag,) = store.list_receipt_flags()
        assert flag["recall_id"] == receipt.recall_id
        assert flag["node_ids"] == ["n-db", "n-cache"]
        assert flag["query"] == "billing database"
        with pytest.raises(ReceiptError) as err:
            UsageObserver(store, judge=_used).observe_answer(receipt.recall_id, ANSWER)
        assert err.value.code == "expired"

    def test_dismissing_keeps_the_record(self, store):
        receipt = _issue(store)
        self._expire(store, receipt)
        assert store.dismiss_receipt_flag(receipt.recall_id)
        assert store.list_receipt_flags() == []
        (kept,) = store.list_receipt_flags(include_dismissed=True)
        assert kept["status"] == "dismissed"
        assert not store.dismiss_receipt_flag(receipt.recall_id)

    def test_receipts_inside_the_window_are_left_alone(self, store):
        receipt = _issue(store)
        assert store.expire_recall_receipts(
            now=datetime.now(timezone.utc) + timedelta(hours=23)) == 0
        assert store.get_recall_receipt(receipt.recall_id)["status"] == "open"

    def test_forgetting_a_node_removes_its_id_from_receipts(self, store):
        receipt = _issue(store)
        self._expire(store, receipt)
        GraphOperations(store).forget_node("n-db", reason="test")
        (flag,) = store.list_receipt_flags()
        assert flag["node_ids"] == ["n-cache"]

    def test_cli_lists_and_dismisses_flags(self, store):
        receipt = _issue(store)
        self._expire(store, receipt)
        db_path = store.db_path
        store.close()
        runner = CliRunner()
        result = runner.invoke(main, ["usage", "flags", "--db", db_path])
        assert result.exit_code == 0, result.output
        assert receipt.recall_id in result.output
        assert "Billing runs on PostgreSQL" in result.output
        result = runner.invoke(main, ["usage", "flags", "--db", db_path,
                                      "--dismiss", receipt.recall_id])
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["usage", "flags", "--db", db_path])
        assert "No usage flags" in result.output


class TestRestEndpoints:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setattr("revien.usage.observer.judge_use", _used)
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        app = create_app(db_path=path)
        with TestClient(app) as c:
            c.post("/v1/ingest", json={
                "source_id": "seed", "content_type": "note",
                "content": "We chose Postgres over SQLite for billing because "
                           "of concurrent writes.",
            })
            yield c

    def test_recall_without_receipt_is_unchanged(self, client):
        body = client.post("/v1/recall", json={"query": "Postgres billing"}).json()
        assert "recall_id" not in body and "receipt_expires_at" not in body

    def test_receipted_recall_then_observe(self, client):
        body = client.post("/v1/recall", json={
            "query": "Postgres billing", "receipt": True, "runtime": "langchain",
        }).json()
        assert body["results"], body
        recall_id = body["recall_id"]

        observed = client.post("/v1/usage/observe", json={
            "recall_id": recall_id,
            "answer": "Billing uses Postgres: SQLite could not take the "
                      "concurrent writes.",
        })
        assert observed.status_code == 200, observed.text
        result = observed.json()
        assert result["runtime"] == "langchain"
        assert {j["node_id"] for j in result["judgments"]} == {
            r["node_id"] for r in body["results"]}
        assert all(j["verdict"] == "used" for j in result["judgments"])

        again = client.post("/v1/usage/observe",
                            json={"recall_id": recall_id, "answer": "x"})
        assert again.status_code == 409
        assert client.post("/v1/usage/observe", json={
            "recall_id": "r_nope", "answer": "x"}).status_code == 404

    def test_flags_endpoint(self, client):
        assert client.get("/v1/usage/flags").json() == {"flags": []}


class TestLiveRuntimes:
    def _seed(self, store):
        IngestionPipeline(store).ingest(IngestionInput(
            source_id="seed", content_type="note",
            content="We chose Postgres over SQLite because of concurrent writes."))

    def test_hermes_prefetch_then_sync_turn(self, store, monkeypatch):
        from tests.test_hermes_provider import _bypass_provider

        monkeypatch.setenv("REVIEN_USAGE", "shadow")
        monkeypatch.setattr("revien.usage.observer.judge_use", _used)
        self._seed(store)
        prov = _bypass_provider(store, SemanticIndex(store))
        assert prov.prefetch("Postgres database decision", session_id="s1")
        prov.sync_turn("Which database?", "Postgres, because SQLite could not "
                       "take the concurrent writes.", session_id="s1")
        prov._flush_sync()
        rows = store.list_use_judgments()
        assert rows and all(r["episode_key"].startswith("receipt:hermes#")
                            for r in rows)

    def test_hermes_issues_no_receipts_unless_opted_in(self, store, monkeypatch):
        from tests.test_hermes_provider import _bypass_provider

        monkeypatch.delenv("REVIEN_USAGE", raising=False)
        self._seed(store)
        prov = _bypass_provider(store, SemanticIndex(store))
        prov.prefetch("Postgres database decision", session_id="s1")
        prov.sync_turn("Which database?", "Postgres.", session_id="s1")
        prov._flush_sync()
        assert store.list_use_judgments() == []
        assert store.expire_recall_receipts(
            now=datetime.now(timezone.utc) + timedelta(days=2)) == 0

    def test_ollama_chat_redeems_its_receipt(self, tmp_path, monkeypatch):
        from revien.adapters.ollama_adapter import OllamaAdapter

        monkeypatch.setenv("REVIEN_USAGE", "shadow")
        monkeypatch.setattr("revien.usage.observer.judge_use", _used)
        adapter = OllamaAdapter(graph_path=str(tmp_path / "o.db"))
        try:
            self._seed(adapter.store)
            monkeypatch.setattr(adapter, "_call_ollama", lambda **kw: (
                "Postgres, since SQLite could not take concurrent writes."))
            adapter.chat("Which database did we choose, Postgres?")
            rows = adapter.store.list_use_judgments()
            assert rows and all(r["episode_key"].startswith("receipt:ollama#")
                                for r in rows)
        finally:
            adapter.close()
