"""
Smoke test for the headline benchmark path. OFFLINE: no network, no real
dataset. Uses a 3-QA SYNTHETIC conversation, default (graph_only) config, the
zero-LLM ExtractiveAnswerer. Must pass WITHOUT the `semantic` extra installed.

Asserts:
  * F1 computes (official normalization + Porter stemming) and is in [0,1].
  * recall@k / MRR / nDCG compute against gold evidence dia_ids.
  * the append-only audit_log records rows (>= one create per node).
  * network egress == 0 on the default path.
  * the adversarial refusal path is exercised and scores correctly.
"""

import os
import tempfile

import pytest

from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine

from revien_bench import answerers as A
from revien_bench import metrics as M
from revien_bench import sovereignty as S
from revien_bench.ingest_locomo import ingest_conversation
from revien_bench.loader import QA, Conversation, Turn


# ── 3-QA synthetic conversation (no network, no real dataset) ─────────────────
def _synthetic_conv() -> Conversation:
    conv = Conversation(conv_id="synthetic", speaker_a="Alice", speaker_b="Bob")
    conv.session_dates = {1: "7 May 2023"}
    conv.turns = [
        Turn(dia_id="D1:1", speaker="Alice", session=1, session_date="7 May 2023",
             text="We decided to deploy the backend on PostgreSQL, not MySQL."),
        Turn(dia_id="D1:2", speaker="Bob", session=1, session_date="7 May 2023",
             text="I prefer JWT tokens for authentication, with refresh tokens in Redis."),
        Turn(dia_id="D1:3", speaker="Alice", session=1, session_date="7 May 2023",
             text="The staging server runs on Docker and Kubernetes for orchestration."),
    ]
    conv.qa = [
        # single-hop (cat 4): answerable from D1:1
        QA(question="What database did we deploy the backend on?",
           answer="PostgreSQL", category=4, evidence=["D1:1"]),
        # single-hop (cat 4): answerable from D1:2
        QA(question="Where are the refresh tokens stored?",
           answer="Redis", category=4, evidence=["D1:2"]),
        # adversarial (cat 5): nothing in the conversation supports this -> refuse
        QA(question="What is Alice's home phone number?",
           answer="not mentioned", category=5, evidence=[], is_adversarial=True),
    ]
    return conv


@pytest.fixture
def synth_store():
    fd, path = tempfile.mkstemp(suffix="_smoke.db")
    os.close(fd)
    store = GraphStore(db_path=path)
    yield store, path
    store.close()
    try:
        os.unlink(path)
    except OSError:
        pass


def test_f1_metric_basic():
    assert M.f1_score("PostgreSQL", "postgresql") == 1.0
    assert M.f1_score("the red car", "a red truck") == pytest.approx(0.5, abs=1e-6)
    assert 0.0 <= M.f1_score("completely different", "PostgreSQL database") <= 1.0


def test_retrieval_metrics_basic():
    retrieved = ["D1:2", "D1:1", "D1:5"]
    gold = {"D1:1"}
    assert M.recall_at_k(retrieved, gold, 1) == 0.0
    assert M.recall_at_k(retrieved, gold, 3) == 1.0
    assert M.mrr(retrieved, gold) == pytest.approx(0.5)
    assert 0.0 < M.ndcg_at_k(retrieved, gold, 10) <= 1.0


def test_smoke_end_to_end(synth_store):
    store, _ = synth_store
    conv = _synthetic_conv()

    # Ingest via the real pipeline (semantic self-disables without the extra).
    summary = ingest_conversation(conv, store)
    assert summary["turns_ingested"] == 3
    assert summary["total_nodes"] > 0
    # dia_id tagging must have stamped at least the context nodes per turn.
    assert summary["nodes_tagged"] > 0

    engine = RetrievalEngine(store)
    answerer = A.ExtractiveAnswerer()

    f1s = []
    refusal_seen = False
    for qa in conv.qa:
        resp = engine.recall(qa.question, top_n=10)
        ctx = A.RetrievedContext(
            query=qa.question,
            contents=[r.content for r in resp.results],
            labels=[r.label for r in resp.results],
        )
        pred = answerer.answer(ctx)
        if qa.is_adversarial:
            score = M.adversarial_score(pred)
            # Nothing supports the phone-number question -> must refuse.
            assert M.is_refusal(pred), f"expected refusal, got {pred!r}"
            refusal_seen = True
        else:
            score = M.f1_score(pred, qa.answer)
        assert 0.0 <= score <= 1.0
        f1s.append(score)

    # F1 computed for every QA.
    assert len(f1s) == 3
    assert refusal_seen

    # recall@k computes against gold evidence for an evidence-bearing question.
    qa0 = conv.qa[0]
    resp0 = engine.recall(qa0.question, top_n=10)
    retrieved = []
    seen = set()
    for r in resp0.results:
        node = store.get_node(r.node_id)
        dia = (node.metadata or {}).get("dia_id") if node else None
        if dia and dia not in seen:
            seen.add(dia)
            retrieved.append(dia)
    rec3 = M.recall_at_k(retrieved, set(qa0.evidence), 3)
    assert 0.0 <= rec3 <= 1.0


def test_audit_rows_written(synth_store):
    store, _ = synth_store
    conv = _synthetic_conv()
    ingest_conversation(conv, store)
    audit = store.get_all_audit()
    creates = sum(1 for r in audit if r["op"] == "create")
    # At least one create row per surviving node.
    assert creates >= store.count_nodes()
    # Append-only: ids strictly increasing.
    ids = [r["id"] for r in audit]
    assert all(b > a for a, b in zip(ids, ids[1:]))


def test_network_egress_zero():
    # Default headline env: local extractor, local/absent embedder, 0 cloud calls.
    prev_emb = os.environ.get("REVIEN_EMBEDDER")
    prev_ext = os.environ.get("REVIEN_EXTRACTOR")
    try:
        os.environ["REVIEN_EMBEDDER"] = "fastembed"
        os.environ["REVIEN_EXTRACTOR"] = "rule"
        check = S.network_egress_zero(cloud_calls=0)
        assert check.passed, check.detail
    finally:
        for k, v in (("REVIEN_EMBEDDER", prev_emb), ("REVIEN_EXTRACTOR", prev_ext)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_consent_subtests_offline():
    # Both consent sub-tests run on in-memory stores; no network, no dataset.
    deny = S.consent_deny_subtest()
    assert deny.passed, deny.detail
    soft = S.consent_soft_invalidate_subtest()
    assert soft.passed, soft.detail
