"""
revien_bench.runner — Orchestrate the §3 benchmark pipeline (headline track).

Per conversation: a FRESH isolated GraphStore (temp file, no cross-conv leakage)
-> ingest every turn (dia_id-tagged) -> optional cluster -> per QA:
recall(q, top_n=K, now=last_session_date) -> extractive answer -> score
(F1 / retrieval-hit recall@k,MRR,nDCG / latency). Writes a full results JSON.

Configs (via env vars, design §3):
  graph_only : REVIEN_SEMANTIC=0, no neural, no cluster   (HEADLINE, default)
  semantic   : REVIEN_SEMANTIC=1 (needs the `semantic` extra; degrades if absent)
  neural     : graph + community clustering + neural rerank (needs extras; degrades)

Answerer: only `extractive` (zero-LLM) is implemented in this build.

Output results/<timestamp>_<config>.json carries: full config, dataset SHA,
revien version, per-category + overall F1, recall@k / MRR / nDCG, latency
p50/p90/p99, cost ($0), network_calls (0), sovereignty pass/fail, per-question rows.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import revien
from revien.graph.clustering import CommunityDetector
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex

from . import answerers as A
from . import metrics as M
from . import sovereignty as S
from .fetch_locomo import DATA_PATH, read_locked_hash
from .ingest_locomo import ingest_conversation, parse_session_date
from .loader import CATEGORY_NAMES, Conversation, QA, load_locomo

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent

RECALL_KS = (1, 3, 5, 10)
RECALL_TOP_N = 10  # retrieve this many for retrieval-quality scoring


def _load_config(name: str) -> Dict:
    cfg_path = _PKG_DIR / "configs" / f"{name}.json"
    if not cfg_path.exists():
        raise SystemExit(f"unknown config {name!r} (no {cfg_path})")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _apply_env(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply config env vars; return the previous values for restoration."""
    prev: Dict[str, Optional[str]] = {}
    for k, v in env.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def _restore_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _retrieved_dia_ids(store: GraphStore, results) -> List[str]:
    """Map ranked retrieval results -> ordered, de-duplicated dia_ids (gold space)."""
    seen = set()
    ordered: List[str] = []
    for r in results:
        node = store.get_node(r.node_id)
        if node is None:
            continue
        dia = (node.metadata or {}).get("dia_id")
        if dia and dia not in seen:
            seen.add(dia)
            ordered.append(dia)
    return ordered


def _score_qa(
    store: GraphStore,
    engine: RetrievalEngine,
    qa: QA,
    conv: Conversation,
    answerer: A.Answerer,
) -> Dict:
    """Run one QA through recall -> extractive answer -> score."""
    now_dt = parse_session_date(conv.last_session_date) or datetime.now(timezone.utc)

    t0 = time.perf_counter()
    # Surface verbatim turns (CONTEXT nodes): for conversational QA the answer
    # lives in the turn itself, not only in distilled extract nodes.
    resp = engine.recall(qa.question, top_n=RECALL_TOP_N, now=now_dt, include_context=True)
    recall_ms = (time.perf_counter() - t0) * 1000.0

    ctx = A.RetrievedContext(
        query=qa.question,
        contents=[r.content for r in resp.results],
        labels=[r.label for r in resp.results],
    )
    t1 = time.perf_counter()
    prediction = answerer.answer(ctx)
    answer_ms = (time.perf_counter() - t1) * 1000.0

    # F1 / adversarial scoring.
    if qa.is_adversarial:
        f1 = M.adversarial_score(prediction)
    else:
        f1 = M.f1_score(prediction, qa.answer)

    # Retrieval quality vs gold evidence dia_ids.
    retrieved = _retrieved_dia_ids(store, resp.results)
    gold = set(qa.evidence)
    recalls = {f"recall@{k}": M.recall_at_k(retrieved, gold, k) for k in RECALL_KS}
    rr = M.mrr(retrieved, gold)
    ndcg = M.ndcg_at_k(retrieved, gold, 10)

    # Supporting node ids (for provenance check): top results that contributed.
    supporting_ids = [r.node_id for r in resp.results[:5]]

    return {
        "conv": conv.conv_id,
        "category": qa.category,
        "category_name": qa.category_name,
        "question": qa.question,
        "gold": qa.answer,
        "prediction": prediction,
        "f1": round(f1, 4),
        "is_adversarial": qa.is_adversarial,
        "refused": A.REFUSAL == prediction or M.is_refusal(prediction),
        "gold_evidence": sorted(gold),
        "retrieved_dia_ids": retrieved,
        **{k: round(v, 4) for k, v in recalls.items()},
        "mrr": round(rr, 4),
        "ndcg@10": round(ndcg, 4),
        "recall_latency_ms": round(recall_ms, 3),
        "answer_latency_ms": round(answer_ms, 3),
        "_supporting_ids": supporting_ids,
    }


def run_benchmark(
    config_name: str,
    answerer_name: str,
    dataset_path: Path,
    out_dir: Path,
    limit_convs: Optional[int] = None,
    max_qa: Optional[int] = None,
) -> Dict:
    cfg = _load_config(config_name)
    prev_env = _apply_env(cfg.get("env", {}))

    try:
        answerer = A.build_answerer(answerer_name)
        conversations = load_locomo(dataset_path)
        if limit_convs is not None:
            conversations = conversations[:limit_convs]
        # Cap QA per conversation for fast subset runs. Non-destructive: we copy
        # the truncated list onto each conv so retrieval/answerer paths are
        # identical, just over fewer questions.
        if max_qa is not None:
            for _conv in conversations:
                _conv.qa = _conv.qa[:max_qa]

        per_q: List[Dict] = []
        ingest_rates: List[float] = []
        recall_latencies: List[float] = []
        supporting_for_provenance: List[str] = []
        total_audit_creates_expected = 0
        # Network/cost accounting is read off the answerer AFTER the run loop
        # (cloud readers self-count their calls + accumulate a cost estimate;
        # local readers stay at 0 / $0.0). Defaults here in case the loop is empty.
        cloud_calls = 0
        cost_usd_estimate = 0.0

        # We keep ONE store alive at provenance-check time, so we sample the
        # first conversation's store for the provenance/audit assertions (each
        # store is structurally identical; checking one is representative and
        # avoids holding 10 stores open).
        provenance_store: Optional[GraphStore] = None
        provenance_supporting: List[str] = []

        for ci, conv in enumerate(conversations):
            fd, db_path = tempfile.mkstemp(suffix=f"_{config_name}_{conv.conv_id}.db")
            os.close(fd)
            store = GraphStore(db_path=db_path)
            try:
                semantic = SemanticIndex(store)  # self-disables without the extra

                t0 = time.perf_counter()
                summary = ingest_conversation(conv, store, semantic=semantic)
                ingest_s = time.perf_counter() - t0
                if ingest_s > 0 and summary["turns_ingested"]:
                    ingest_rates.append(summary["turns_ingested"] / ingest_s)
                total_audit_creates_expected += summary["nodes_created"]

                clustering = None
                if cfg.get("cluster"):
                    detector = CommunityDetector(db_path)
                    detector.run()
                    detector.load_from_db()
                    clustering = detector

                engine = RetrievalEngine(store, clustering=clustering, semantic=semantic)

                for qa in conv.qa:
                    row = _score_qa(store, engine, qa, conv, answerer)
                    recall_latencies.append(row["recall_latency_ms"])
                    per_q.append(row)

                # Sample the first conversation for provenance/audit checks.
                if ci == 0:
                    provenance_supporting = []
                    for row in per_q:
                        if row["conv"] == conv.conv_id:
                            provenance_supporting.extend(row["_supporting_ids"])
                    # Keep this store open for the assertions below.
                    provenance_store = store
                    provenance_db_path = db_path
                    provenance_expected = summary["nodes_created"]
                    continue  # don't close/delete yet
            finally:
                if ci != 0:
                    store.close()
                    try:
                        os.unlink(db_path)
                    except OSError:
                        pass

        # ── Sovereignty assertions (on the sampled conversation 0 store) ──────
        checks: List[S.Check] = []
        if provenance_store is not None:
            checks.append(
                S.provenance_completeness(provenance_store, provenance_supporting)
            )
            checks.append(
                S.audit_integrity(provenance_store, expected_min_creates=provenance_expected)
            )
            provenance_store.close()
            try:
                os.unlink(provenance_db_path)
            except OSError:
                pass
        # Read network/cost accounting off the answerer (cloud readers count
        # their own calls + accumulate a labelled cost ESTIMATE; local readers
        # report 0 / $0.0).
        cloud_calls = int(getattr(answerer, "network_calls", 0))
        cost_usd_estimate = float(getattr(answerer, "cost_usd_estimate", 0.0))

        # Honest egress check: config-derived, names any cloud backend.
        checks.append(
            S.network_egress_zero(cloud_calls=cloud_calls, answerer=answerer_name)
        )
        checks.extend(S.run_consent_subtests())

        # ── Aggregate metrics ─────────────────────────────────────────────────
        report = _aggregate(per_q, ingest_rates, recall_latencies)
        report["sovereignty"] = S.checks_to_dict(checks)
        report["config"] = {
            "name": config_name,
            "env": cfg.get("env", {}),
            "cluster": bool(cfg.get("cluster")),
            "answerer": answerer.name,
            "recall_top_n": RECALL_TOP_N,
            "recall_ks": list(RECALL_KS),
        }
        report["dataset"] = {
            "path": str(dataset_path),
            "sha256": read_locked_hash(),
            "conversations": len(conversations),
        }
        # All-local headline run: cost_usd_estimate is 0.0 and cloud_calls is 0,
        # so this preserves the honest $0 / 0-calls local default. Cloud answerer
        # runs surface a non-zero, clearly-labelled cost ESTIMATE + real call count.
        report["cost_usd"] = round(cost_usd_estimate, 6)
        report["cost_usd_is_estimate"] = True
        report["network_calls"] = cloud_calls
        report["environment"] = {
            "revien_version": revien.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        }
        report["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Drop the internal _supporting_ids from the persisted per-question rows.
        for row in per_q:
            row.pop("_supporting_ids", None)
        report["per_question"] = per_q

        # ── Write results ─────────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}_{config_name}.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["_out_path"] = str(out_path)
        return report
    finally:
        _restore_env(prev_env)


def _aggregate(
    per_q: List[Dict], ingest_rates: List[float], recall_latencies: List[float]
) -> Dict:
    overall_f1 = M.mean([r["f1"] for r in per_q]) if per_q else 0.0

    by_cat: Dict[int, List[Dict]] = {}
    for r in per_q:
        by_cat.setdefault(r["category"], []).append(r)

    per_category = {}
    for cat, rows in sorted(by_cat.items()):
        per_category[CATEGORY_NAMES.get(cat, str(cat))] = {
            "n": len(rows),
            "f1": round(M.mean([x["f1"] for x in rows]), 4),
        }

    # Retrieval metrics: only meaningful for Qs that carry gold evidence.
    evid_rows = [r for r in per_q if r["gold_evidence"]]
    retr = {}
    for k in RECALL_KS:
        key = f"recall@{k}"
        retr[key] = round(M.mean([r[key] for r in evid_rows]), 4) if evid_rows else None
    retr["mrr"] = round(M.mean([r["mrr"] for r in evid_rows]), 4) if evid_rows else None
    retr["ndcg@10"] = round(M.mean([r["ndcg@10"] for r in evid_rows]), 4) if evid_rows else None
    retr["n_with_evidence"] = len(evid_rows)

    return {
        "n_questions": len(per_q),
        "overall_f1": round(overall_f1, 4),
        "per_category_f1": per_category,
        "retrieval": retr,
        "latency_ms": {
            "recall": M.latency_percentiles(recall_latencies),
        },
        "ingest_turns_per_sec": round(M.mean(ingest_rates), 2) if ingest_rates else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Revien LoCoMo benchmark (headline track)")
    ap.add_argument("--config", default="graph_only",
                    choices=["graph_only", "semantic", "neural"])
    ap.add_argument("--answerer", default="extractive",
                    help="reader spec: extractive | ollama:<model> | "
                         "openai:<model> | openrouter:<model> | together:<model> | "
                         "claude:<model>")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results"))
    ap.add_argument("--dataset", default=str(DATA_PATH))
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of conversations (debug/subset)")
    ap.add_argument("--max-qa", type=int, default=None, dest="max_qa",
                    help="cap QA per conversation (debug/subset)")
    args = ap.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. "
              f"Run: python -m revien_bench.fetch_locomo")
        return 2

    report = run_benchmark(
        config_name=args.config,
        answerer_name=args.answerer,
        dataset_path=dataset_path,
        out_dir=Path(args.out),
        limit_convs=args.limit,
        max_qa=args.max_qa,
    )
    _print_summary(report)
    return 0


def _print_summary(report: Dict) -> None:
    print("\n=== Revien LoCoMo benchmark ===")
    print(f"config        : {report['config']['name']} / {report['config']['answerer']}")
    print(f"questions     : {report['n_questions']}")
    print(f"overall F1    : {report['overall_f1']}")
    print("per-category F1:")
    for cat, v in report["per_category_f1"].items():
        print(f"   {cat:14s} n={v['n']:<4d} F1={v['f1']}")
    r = report["retrieval"]
    print(f"retrieval     : recall@1={r['recall@1']} @5={r['recall@5']} "
          f"@10={r['recall@10']} MRR={r['mrr']} nDCG@10={r['ndcg@10']} "
          f"(n_evid={r['n_with_evidence']})")
    lat = report["latency_ms"]["recall"]
    print(f"recall latency: p50={lat['p50']}ms p90={lat['p90']}ms p99={lat['p99']}ms")
    print(f"ingest        : {report['ingest_turns_per_sec']} turns/sec")
    _est = " (est.)" if report.get("cost_usd") else ""
    print(f"cost          : ${report['cost_usd']}{_est}   network_calls={report['network_calls']}")
    sov = report["sovereignty"]
    print(f"sovereignty   : {'PASS' if sov['all_passed'] else 'FAIL'}")
    for c in sov["checks"]:
        print(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    print(f"results JSON  : {report.get('_out_path')}")


if __name__ == "__main__":
    raise SystemExit(main())
