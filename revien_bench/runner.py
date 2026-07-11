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
import hashlib
import json
import os
import platform
import re
import shutil
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
from . import failure_analysis as FA
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


# ── Per-conversation checkpoint / resume ──────────────────────────────────────
# A full LoCoMo run is ~20 min and ~2000 QA. Writing results only at the very
# end means any interruption (laptop sleep, a single hung HTTP call) loses
# everything and re-burns the API calls on retry. We append ONE JSON line per
# COMPLETED conversation to a checkpoint file and flush it immediately, so an
# interruption loses at most the in-progress conversation. On startup we load it
# and SKIP conversations already done (keyed by conv_id), guarded by the dataset
# SHA so a changed dataset never falsely resumes.


def _sanitize(text: str) -> str:
    """Make an answerer/config spec safe for a filename (e.g. 'openai:gpt-4o')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


# ── Run-identity fingerprints ──────────────────────────────────────────────────
# Two silent-staleness traps got walked into on July 10-11 2026 and are closed
# here at the root: (1) the checkpoint resumed rows recorded under DIFFERENT
# env knobs (`ran=0 resumed=10` replayed a pre-rerank baseline into a rerank
# "confirm"); (2) the db cache served ingests built by an OLDER extractor
# after a code fix. Neither cache identity included the state that actually
# produced the data. Now it does: env + code fingerprints are part of the
# checkpoint FILENAME (a knob/code change simply never sees the old file;
# crash + relaunch with identical state resumes as before) and of the cache
# META (mismatch = rebuild, same as a dataset-SHA change). Stale files are
# left behind rather than deleted — they are small, and history is history.

def _env_fingerprint(prefix_allowlist: Optional[tuple] = None) -> str:
    """Fingerprint of the REVIEN_* environment (or an allowlisted subset)."""
    items = sorted(
        (k, v) for k, v in os.environ.items()
        if k.startswith("REVIEN_")
        and (prefix_allowlist is None or k in prefix_allowlist)
    )
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()[:8]


def _code_fingerprint(*subpackages: str) -> str:
    """Content hash of revien source subpackages (e.g. 'retrieval'). Catches
    dev-tree edits that a version number cannot — the extractor-regex trap."""
    import revien
    root = Path(revien.__file__).resolve().parent
    h = hashlib.sha256()
    for sub in sorted(subpackages):
        pkg = root / sub
        if not pkg.is_dir():
            continue
        for py in sorted(pkg.rglob("*.py")):
            h.update(str(py.relative_to(root)).encode("utf-8"))
            h.update(py.read_bytes())
    return h.hexdigest()[:8]


# Env vars that change what an INGEST produces (extraction, embeddings, CSL).
# Retrieval-only knobs (REVIEN_RERANK*, ranking weights, top-K) are DELIBERATELY
# excluded: the whole point of the db cache is reusing identical ingests across
# ranking-knob sweep variants. The code fingerprint is the backstop for
# anything this list misses.
_INGEST_ENV_KEYS = (
    "REVIEN_CSL", "REVIEN_SEMANTIC", "REVIEN_EMBEDDER", "REVIEN_EMBED_MODEL",
    "REVIEN_EXTRACTOR", "REVIEN_SENSITIVITY_BACKEND", "REVIEN_SENSITIVITY_MODEL",
    "REVIEN_TENSION_BACKEND", "REVIEN_TENSION_MODEL", "REVIEN_INGEST_DENY",
)


def _ingest_fingerprint() -> str:
    return (_env_fingerprint(_INGEST_ENV_KEYS)
            + _code_fingerprint("ingestion", "graph", "semantic", "adapters"))


def _run_fingerprint() -> str:
    """Full run identity for checkpoints: ALL REVIEN_* env (ranking knobs
    included — that's trap #1) + retrieval-affecting code."""
    return (_env_fingerprint()
            + _code_fingerprint("retrieval", "semantic", "graph",
                                "ingestion", "neural"))


def _checkpoint_path(out_dir: Path, config_name: str, answerer_name: str) -> Path:
    """Checkpoint path for this exact (config, answerer, env, code) identity.
    A knob or code change yields a different filename — the old checkpoint
    can never falsely resume; an identical relaunch resumes exactly as before."""
    return out_dir / (
        f".checkpoint_{_sanitize(config_name)}_{_sanitize(answerer_name)}"
        f"_{_run_fingerprint()}.jsonl"
    )


def _load_checkpoint(
    ckpt_path: Path, dataset_sha: Optional[str]
) -> Dict[str, Dict]:
    """Load completed-conversation records, keyed (de-duplicated) by conv_id.

    Only records whose stored dataset_sha matches the CURRENT dataset SHA are
    honored — a changed dataset must NOT falsely resume. Records for a different
    SHA (or malformed lines) are skipped. Later lines win on duplicate conv_id
    (an append-only re-run of the same conv supersedes the earlier one).
    """
    done: Dict[str, Dict] = {}
    if not ckpt_path.exists():
        return done
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn final line from a hard kill
        if not isinstance(rec, dict) or "conv_id" not in rec:
            continue
        # Dataset-SHA guard: skip records that don't match the current dataset.
        if dataset_sha is not None and rec.get("dataset_sha") != dataset_sha:
            continue
        done[str(rec["conv_id"])] = rec
    return done


def _append_checkpoint(ckpt_path: Path, record: Dict) -> None:
    """Append one completed-conversation record and flush+fsync to disk.

    fsync so a crash/sleep immediately after a conversation can't lose the line
    that was already 'written' to a buffer.
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


# ── Pristine ingest cache ─────────────────────────────────────────────────────
# Ingest (extraction + embedding every turn) dominates bench wall-clock (~10 min
# for the full dataset) and is IDENTICAL across scoring-knob sweeps. --db-cache
# keeps one pristine post-ingest DB per (config, conversation), SHA-guarded, and
# every run works on a TEMP COPY — recall's touch_node writes and clustering
# never contaminate the cache, so sweep variants stay comparable.


def _cache_paths(db_cache: Path, config_name: str, conv_id: str) -> tuple:
    base = db_cache / f"{_sanitize(config_name)}_{_sanitize(conv_id)}.db"
    return base, Path(str(base) + ".meta.json")


def _cache_load_meta(meta_path: Path, dataset_sha: Optional[str]) -> Optional[Dict]:
    """Meta for a cached DB, or None when absent/SHA-mismatched (never falsely
    reuse a cache built from a different dataset) or ingest-identity-mismatched
    (never falsely reuse an ingest built by different code or ingest env —
    the extractor-regex trap). Old metas without the fingerprint rebuild once."""
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if dataset_sha is not None and meta.get("dataset_sha") != dataset_sha:
        return None
    if meta.get("ingest_fp") != _ingest_fingerprint():
        return None
    return meta


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
    dia_map: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Run one QA through recall -> extractive answer -> score."""
    now_dt = parse_session_date(conv.last_session_date) or datetime.now(timezone.utc)

    t0 = time.perf_counter()
    # Surface verbatim turns (CONTEXT nodes): for conversational QA the answer
    # lives in the turn itself, not only in distilled extract nodes.
    # debug=True: per-node scores/filters/anchors feed the miss classification
    # below. Overhead is a few dict copies — negligible vs the ~350ms p50, and
    # it's applied to EVERY config equally so latency comparisons stay fair.
    resp = engine.recall(
        qa.question, top_n=RECALL_TOP_N, now=now_dt, include_context=True, debug=True
    )
    recall_ms = (time.perf_counter() - t0) * 1000.0

    ctx = A.RetrievedContext(
        query=qa.question,
        contents=[r.content for r in resp.results],
        labels=[r.label for r in resp.results],
    )
    t1 = time.perf_counter()
    # A single hung/failing answerer call (socket timeout, HTTP error, malformed
    # response) must NOT kill the whole run. Catch it here, record the QA as
    # unanswered (empty prediction -> F1 0), note the error, and continue. The
    # extractive reader never raises; this guard matters for the LLM readers.
    answer_error: Optional[str] = None
    try:
        prediction = answerer.answer(ctx)
    except Exception as e:  # noqa: BLE001 - intentional: one bad call ≠ dead run
        prediction = ""
        answer_error = f"{type(e).__name__}: {e}"
    answer_ms = (time.perf_counter() - t1) * 1000.0

    # F1 / adversarial scoring. An empty prediction scores F1 0 for a normal
    # question; for an adversarial question an empty/failed answer is NOT a valid
    # refusal, so adversarial_score("") is 0 too — a failed call is never rewarded.
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

    # Per-query failure taxonomy: WHERE did each missed gold item die —
    # never_extracted / no_anchors / walk_depth_miss / disconnected /
    # filtered_out / outranked. The raw diagnostics dict is NOT persisted
    # (it's the whole walked frontier); only the classification is.
    gold_miss_causes: Optional[Dict[str, Dict]] = None
    if dia_map is not None and gold:
        gold_miss_causes = FA.classify_misses(
            store, resp.diagnostics, gold, retrieved, dia_map
        )

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
        "answer_error": answer_error,
        "refused": A.REFUSAL == prediction or M.is_refusal(prediction),
        "gold_evidence": sorted(gold),
        "retrieved_dia_ids": retrieved,
        "gold_miss_causes": gold_miss_causes,
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
    fresh: bool = False,
    db_cache: Optional[Path] = None,
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

        # ── Checkpoint / resume setup ─────────────────────────────────────────
        dataset_sha = read_locked_hash()
        ckpt_path = _checkpoint_path(out_dir, config_name, answerer_name)
        if fresh:
            # --fresh: ignore + delete any prior checkpoint and start over.
            try:
                ckpt_path.unlink()
            except OSError:
                pass
            done_records: Dict[str, Dict] = {}
        else:
            # Resume: load completed conversations (SHA-guarded), skip them below.
            done_records = _load_checkpoint(ckpt_path, dataset_sha)
        resumed_conv_ids = set(done_records)

        per_q: List[Dict] = []
        ingest_rates: List[float] = []
        recall_latencies: List[float] = []
        total_audit_creates_expected = 0
        n_resumed = 0
        n_ran = 0
        # Network/cost accounting is read off the answerer AFTER the run loop
        # (cloud readers self-count their calls + accumulate a cost estimate;
        # local readers stay at 0 / $0.0). Defaults here in case the loop is empty.
        cloud_calls = 0
        cost_usd_estimate = 0.0
        # Cost/calls carried in from resumed conversations (summed with the live
        # answerer's counters at the end so the aggregate covers resumed+new).
        resumed_cloud_calls = 0
        resumed_cost_usd = 0.0

        # ── Replay resumed conversations into the accumulators FIRST ──────────
        # The final aggregate (F1, per-category, recall@k, latency, cost) must
        # cover ALL conversations — resumed + newly run. We fold each completed
        # record's rows + stats back in here, then run only the not-yet-done
        # conversations below.
        for conv in conversations:
            rec = done_records.get(conv.conv_id)
            if rec is None:
                continue
            n_resumed += 1
            for row in rec.get("rows", []):
                per_q.append(row)
                recall_latencies.append(row.get("recall_latency_ms", 0.0))
            ir = rec.get("ingest_rate")
            if ir:
                ingest_rates.append(ir)
            total_audit_creates_expected += rec.get("nodes_created", 0)
            resumed_cloud_calls += int(rec.get("conv_network_calls", 0) or 0)
            resumed_cost_usd += float(rec.get("conv_cost_usd", 0.0) or 0.0)

        # We keep ONE store alive at provenance-check time, so we sample the
        # first FRESHLY-RUN conversation's store for the provenance/audit
        # assertions (each store is structurally identical; checking one is
        # representative and avoids holding 10 stores open). On a full resume
        # (all conversations already done) there is no live store to sample, so
        # the provenance/audit checks are skipped — noted in the report.
        provenance_store: Optional[GraphStore] = None
        provenance_supporting: List[str] = []
        provenance_db_path: Optional[str] = None
        provenance_expected = 0

        for conv in conversations:
            if conv.conv_id in resumed_conv_ids:
                continue  # already completed in a prior run — skip (resume)
            # Snapshot the answerer's counters so we can attribute per-conv
            # cost/calls to THIS conversation's checkpoint record.
            calls_before = int(getattr(answerer, "network_calls", 0))
            cost_before = float(getattr(answerer, "cost_usd_estimate", 0.0))
            is_first_fresh = provenance_store is None

            fd, db_path = tempfile.mkstemp(suffix=f"_{config_name}_{conv.conv_id}.db")
            os.close(fd)

            # Pristine-cache lookup: on a hit, the working temp DB starts as a
            # COPY of the post-ingest snapshot and ingest is skipped entirely.
            cached_meta: Optional[Dict] = None
            cache_db = cache_meta_path = None
            if db_cache is not None:
                cache_db, cache_meta_path = _cache_paths(db_cache, config_name, conv.conv_id)
                cached_meta = _cache_load_meta(cache_meta_path, dataset_sha)
                if cached_meta is not None and cache_db.exists():
                    shutil.copyfile(cache_db, db_path)
                else:
                    cached_meta = None

            store = GraphStore(db_path=db_path)
            keep_store_open = False
            try:
                semantic = SemanticIndex(store)  # self-disables without the extra

                conv_ingest_rate = 0.0
                if cached_meta is not None:
                    summary = {
                        "turns_ingested": cached_meta["turns_ingested"],
                        "nodes_created": cached_meta["nodes_created"],
                    }
                    # Fold the ORIGINAL measured rate so the aggregate stays an
                    # honest ingest number, not a cache-copy artifact.
                    conv_ingest_rate = float(cached_meta.get("ingest_rate", 0.0))
                    if conv_ingest_rate:
                        ingest_rates.append(conv_ingest_rate)
                else:
                    t0 = time.perf_counter()
                    summary = ingest_conversation(conv, store, semantic=semantic)
                    ingest_s = time.perf_counter() - t0
                    if ingest_s > 0 and summary["turns_ingested"]:
                        conv_ingest_rate = summary["turns_ingested"] / ingest_s
                        ingest_rates.append(conv_ingest_rate)
                    if db_cache is not None:
                        # Snapshot the pristine post-ingest state via SQLite's
                        # backup API (safe on a live connection, WAL included),
                        # then the meta sidecar with the SHA guard.
                        db_cache.mkdir(parents=True, exist_ok=True)
                        import sqlite3 as _sq
                        dst = _sq.connect(str(cache_db))
                        try:
                            store._get_conn().backup(dst)
                        finally:
                            dst.close()
                        cache_meta_path.write_text(json.dumps({
                            "dataset_sha": dataset_sha,
                            "ingest_fp": _ingest_fingerprint(),
                            "conv_id": conv.conv_id,
                            "turns_ingested": summary["turns_ingested"],
                            "nodes_created": summary["nodes_created"],
                            "ingest_rate": round(conv_ingest_rate, 4),
                        }), encoding="utf-8")
                total_audit_creates_expected += summary["nodes_created"]

                clustering = None
                if cfg.get("cluster"):
                    detector = CommunityDetector(db_path)
                    detector.run()
                    detector.load_from_db()
                    clustering = detector

                engine = RetrievalEngine(store, clustering=clustering, semantic=semantic)

                # dia_id -> node_ids, built once per conversation; feeds the
                # per-query miss classification in _score_qa.
                dia_map = FA.build_dia_map(store)

                conv_rows: List[Dict] = []
                for qa in conv.qa:
                    row = _score_qa(store, engine, qa, conv, answerer, dia_map=dia_map)
                    recall_latencies.append(row["recall_latency_ms"])
                    per_q.append(row)
                    conv_rows.append(row)

                n_ran += 1

                # Per-conv cost/calls delta (cloud readers self-count globally).
                conv_calls = int(getattr(answerer, "network_calls", 0)) - calls_before
                conv_cost = float(getattr(answerer, "cost_usd_estimate", 0.0)) - cost_before

                # ── Persist this conversation's checkpoint line (flush+fsync) ──
                # Written AFTER the conversation fully completes, so an
                # interruption loses at most the in-progress conversation.
                _append_checkpoint(
                    ckpt_path,
                    {
                        "conv_id": conv.conv_id,
                        "dataset_sha": dataset_sha,
                        "rows": conv_rows,
                        "ingest_rate": round(conv_ingest_rate, 4),
                        "nodes_created": summary["nodes_created"],
                        "conv_network_calls": conv_calls,
                        "conv_cost_usd": round(conv_cost, 6),
                    },
                )

                # Sample the FIRST freshly-run conversation for provenance/audit.
                if is_first_fresh:
                    provenance_supporting = []
                    for row in conv_rows:
                        provenance_supporting.extend(row["_supporting_ids"])
                    # Keep this store open for the assertions below.
                    provenance_store = store
                    provenance_db_path = db_path
                    provenance_expected = summary["nodes_created"]
                    keep_store_open = True
            finally:
                if not keep_store_open:
                    store.close()
                    try:
                        os.unlink(db_path)
                    except OSError:
                        pass

        # ── Sovereignty assertions (on the sampled conversation 0 store) ──────
        checks: List[S.Check] = []
        norm_merges_sample: Optional[Dict] = None
        if provenance_store is not None:
            # False-merge precision surface, sampled from the same store the
            # provenance checks use (one conversation is representative for
            # eyeballing pairs; the vault eval reports its full list).
            norm_merges_sample = FA.normalization_merge_report(provenance_store)
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
        # report 0 / $0.0). The live counters reflect only THIS session's
        # freshly-run conversations, so we add back the cost/calls carried in
        # from resumed conversations — the aggregate must cover resumed + new.
        cloud_calls = int(getattr(answerer, "network_calls", 0)) + resumed_cloud_calls
        cost_usd_estimate = (
            float(getattr(answerer, "cost_usd_estimate", 0.0)) + resumed_cost_usd
        )

        # Honest egress check: config-derived, names any cloud backend.
        checks.append(
            S.network_egress_zero(cloud_calls=cloud_calls, answerer=answerer_name)
        )
        checks.extend(S.run_consent_subtests())

        # ── Aggregate metrics ─────────────────────────────────────────────────
        report = _aggregate(per_q, ingest_rates, recall_latencies)
        # Retrieval failure taxonomy: where the missed gold evidence died.
        # Rows from pre-taxonomy checkpoints are counted as unclassified.
        report["retrieval_failure_analysis"] = FA.aggregate_failures(per_q)
        report["normalization_merges_sample"] = norm_merges_sample
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
        # Resume bookkeeping: how many conversations were replayed from the
        # checkpoint vs. freshly run this session, and where the checkpoint lives.
        report["resume"] = {
            "checkpoint_path": str(ckpt_path),
            "conversations_resumed": n_resumed,
            "conversations_ran": n_ran,
            "fresh": bool(fresh),
            "provenance_checked": provenance_store is not None,
        }
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
    ap.add_argument("--fresh", action="store_true",
                    help="ignore + delete any existing checkpoint and start over "
                         "(default: resume from the checkpoint, skipping completed "
                         "conversations)")
    ap.add_argument("--db-cache", default=None, dest="db_cache",
                    help="directory of pristine post-ingest DB snapshots (per "
                         "config+conversation, SHA-guarded). On a hit, ingest is "
                         "skipped and recall runs against a temp COPY — the cache "
                         "is never mutated. Cuts sweep iterations from ~10min to "
                         "recall-only time.")
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
        fresh=args.fresh,
        db_cache=Path(args.db_cache) if args.db_cache else None,
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
    fa = report.get("retrieval_failure_analysis") or {}
    if fa.get("gold_items_missed"):
        causes = ", ".join(f"{c}={n}" for c, n in fa["by_cause"].items())
        print(f"miss taxonomy : {fa['gold_items_missed']} gold items missed — {causes}")
        od = fa.get("outranked_detail")
        if od:
            print(f"   outranked   : median best rank {od['median_best_rank']}, "
                  f"{od['within_20']}/{od['n']} within top-20")
        if fa.get("rows_unclassified"):
            print(f"   (unclassified rows from old checkpoint: {fa['rows_unclassified']})")
    lat = report["latency_ms"]["recall"]
    print(f"recall latency: p50={lat['p50']}ms p90={lat['p90']}ms p99={lat['p99']}ms")
    print(f"ingest        : {report['ingest_turns_per_sec']} turns/sec")
    _est = " (est.)" if report.get("cost_usd") else ""
    print(f"cost          : ${report['cost_usd']}{_est}   network_calls={report['network_calls']}")
    sov = report["sovereignty"]
    print(f"sovereignty   : {'PASS' if sov['all_passed'] else 'FAIL'}")
    for c in sov["checks"]:
        print(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    res = report.get("resume")
    if res:
        print(f"resume        : ran={res['conversations_ran']} "
              f"resumed={res['conversations_resumed']} "
              f"(checkpoint: {res['checkpoint_path']})")
    print(f"results JSON  : {report.get('_out_path')}")


if __name__ == "__main__":
    raise SystemExit(main())
