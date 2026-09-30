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

Optional end-to-end LLM-judge track (--judge, default 'f1' = off): a SEPARATE
binary CORRECT/WRONG accuracy score from an LLM comparing each prediction to
the gold answer (revien_bench.judges), added as report["judge"]/["reader"] and
per-question judge_correct/judge_error — NEVER blended with F1 above. The two
publishable rows (model names are placeholders — picking them is the owner's
call):
  local : --answerer ollama:<model> --judge ollama:<model>          (egress PASS)
  cloud : --answerer openrouter:<model> --judge openrouter:<model>  (egress FAIL,
          by design — a cloud reader/judge is data leaving the machine; the
          sovereignty check labels this honestly rather than hiding it)
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
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
from revien.semantic.index import SemanticIndex, build_embedder, embed_context_mode
from revien.semantic.rerank import CrossEncoderReranker

from . import answerers as A
from . import failure_analysis as FA
from . import judges as J
from . import decompose as D
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


# ── Alias-inference measurement hook (opt-in, DEFAULT OFF) ────────────────────
# revien.alias.run_alias_pass draws ALIAS_OF edges but nothing in the bench
# pipeline ever calls it — recall's anchor expansion (REVIEN_ALIAS, on by
# default) can only union an alias's neighborhood in if the edge already
# exists, so measuring the recall claim needs the pass run BEFORE recall,
# per conversation, right after ingest completes. REVIEN_BENCH_ALIAS=1 turns
# it on; unset/0 is a byte-identical no-op (the hook is never even imported).


def _bench_alias_enabled() -> bool:
    """Read per-call, not cached — matches REVIEN_ALIAS / REVIEN_FENCE's
    on-demand env convention elsewhere in this codebase."""
    return os.environ.get("REVIEN_BENCH_ALIAS", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _run_bench_alias_pass(store: GraphStore, semantic) -> Dict:
    """Run the alias-inference pass on `store` (post-ingest, pre-recall) and
    shape a compact stats dict for the checkpoint/report. Lazy import so an
    unset REVIEN_BENCH_ALIAS never even touches revien.alias."""
    from revien.alias import run_alias_pass
    t0 = time.perf_counter()
    result = run_alias_pass(store, semantic=semantic)
    return {
        "ran": result.ran,
        "entities_considered": result.entities_considered,
        "candidates_considered": result.candidates_considered,
        "edges_created": result.edges_created,
        "edges_by_method": dict(result.edges_by_method),
        "sample": result.sample,
        "note": result.note,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


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

# How LLM readers render retrieved memories: "dated" = each memory prefixed
# with the day it was said. Recorded in the results JSON and folded into the
# checkpoint fingerprint.
READER_CONTEXT = "dated-resolved"


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


def _run_fingerprint(decompose: str = "none") -> str:
    """Full run identity for checkpoints: ALL REVIEN_* env (ranking knobs
    included — that's trap #1) + retrieval-affecting code. A decompose spec
    other than 'none' is part of the identity (sub-query recalls change every
    row); 'none' adds nothing, so pre-existing checkpoints keep resuming."""
    dec = (decompose or "none").strip()
    dec_fp = ("" if dec.lower() == "none"
              else hashlib.sha256(dec.encode("utf-8")).hexdigest()[:4])
    return (_env_fingerprint()
            + _code_fingerprint("retrieval", "semantic", "graph",
                                "ingestion", "neural")
            # How the LLM readers see context is part of run identity: an
            # old undated checkpoint must never resume into a dated run.
            + hashlib.sha256(READER_CONTEXT.encode("utf-8")).hexdigest()[:4]
            + dec_fp)


def _checkpoint_path(
    out_dir: Path, config_name: str, answerer_name: str, judge_name: str = "f1",
    decompose_name: str = "none",
) -> Path:
    """Checkpoint path for this exact (config, answerer, judge, env, code)
    identity. A knob or code change yields a different filename — the old
    checkpoint can never falsely resume; an identical relaunch resumes exactly
    as before. The judge spec is part of the identity (F5): resuming a
    judge-less checkpoint under a NEW --judge would otherwise silently reuse
    rows that were never judged, corrupting the judge accuracy denominator."""
    return out_dir / (
        f".checkpoint_{_sanitize(config_name)}_{_sanitize(answerer_name)}"
        f"_{_sanitize(judge_name)}_{_run_fingerprint(decompose_name)}.jsonl"
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


_TRUTHY = ("1", "true", "yes", "on", "require", "required", "strict")


def _semantic_requested(cfg: Dict) -> bool:
    """Did the run's CONFIG ask for the semantic layer (REVIEN_SEMANTIC truthy)?"""
    return str((cfg.get("env") or {}).get("REVIEN_SEMANTIC", "0")).strip().lower() in _TRUTHY


def _layer_status(semantic, reranker=None) -> Dict:
    """Real layer state, read off the live objects (never env vars):
    SemanticIndex.is_enabled/inactive_reason()/status() (revien/semantic/index.py
    -- flips False in `_safe_disable` after a runtime error) and
    CrossEncoderReranker.is_enabled (revien/semantic/rerank.py)."""
    active = bool(getattr(semantic, "is_enabled", False))
    reason = None if active else (
        semantic.inactive_reason() if hasattr(semantic, "inactive_reason")
        else "semantic layer absent")
    embed_model = embed_dim = embed_context = None
    try:
        sem_status = semantic.status()
        embedder = str(sem_status.get("embedder", "unknown"))
        if active:
            embed_model = sem_status.get("embed_model")
            embed_dim = sem_status.get("embed_dim")
            embed_context = sem_status.get("embed_context")
    except Exception:
        embedder = "unknown"
    rr = bool(getattr(reranker, "is_enabled", False)) if reranker is not None else False
    # Resolved depth/model live on CrossEncoderReranker.top_k / .model_name
    # (rerank.py __init__); None when there is no reranker or it is disabled.
    return {
        "semantic_active": active,
        "rerank_active": rr,
        "rerank_top_k": getattr(reranker, "top_k", None) if rr else None,
        "rerank_model": getattr(reranker, "model_name", None) if rr else None,
        "embedder": embedder,
        "embed_model": embed_model,
        "embed_dim": embed_dim,
        "embed_context": embed_context,
        "semantic_inactive_reason": reason,
    }


def _resolve_embed_model() -> Optional[str]:
    """Model name the CURRENT run's semantic index will use, read off the same
    provider factory SemanticIndex uses (revien/semantic/index.py build_embedder).
    Provider construction is lazy -- no model load. Dim is NOT resolved here:
    it is a pre-load default until the first embed, so only the name is trusted."""
    try:
        return getattr(build_embedder(), "model_name", None)
    except Exception:
        return None


def _cache_load_meta(meta_path: Path, dataset_sha: Optional[str],
                     semantic_requested: bool = False,
                     embed_model: Optional[str] = None,
                     embed_dim: Optional[int] = None,
                     embed_context: Optional[str] = None) -> Optional[Dict]:
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
    ls = meta.get("layer_status")
    if not isinstance(ls, dict) or (semantic_requested and not ls.get("semantic_active")):
        why = ("no layer_status recorded" if not isinstance(ls, dict)
               else "semantic layer was inactive during its ingest")
        print(f"[bench] db-cache: ignoring stale snapshot {meta_path.name} - {why}")
        return None
    if semantic_requested:
        # Snapshot vectors belong to the model that embedded them.
        snap_model, snap_dim = ls.get("embed_model"), ls.get("embed_dim")
        why = None
        if snap_model is None:
            why = "no embed_model recorded (legacy snapshot)"
        elif embed_model is not None and snap_model != embed_model:
            why = f"embedded with {snap_model}, this run uses {embed_model}"
        elif embed_dim is not None and snap_dim is not None and snap_dim != embed_dim:
            why = f"embedded at dim {snap_dim}, this run uses dim {embed_dim}"
        elif (embed_context is not None
              and (ls.get("embed_context") or "off") != embed_context):
            # A snapshot from before the knob existed was built with "off".
            why = (f"embedded with REVIEN_EMBED_CONTEXT={ls.get('embed_context') or 'off'}, "
                   f"this run uses {embed_context}")
        if why:
            print(f"[bench] db-cache: ignoring stale snapshot {meta_path.name} - {why}")
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


def _fuse_responses(resps: List, top_n: int):
    """UNION sub-query recalls by node_id, keeping each node's best score,
    sorted by score descending (stable: the original query's recall is first,
    so ties favour it), truncated to top_n. Returns (results, diagnostics).

    Diagnostics are merged for the miss taxonomy: per-node scores take the
    max across recalls, filter reasons come from the first recall that saw the
    node, anchor ids are unioned. Only the keys failure_analysis reads are
    kept. Caveat: scores from different queries are pooled, so the taxonomy's
    best_rank under decomposition is a pooled-score rank."""
    best: Dict[str, object] = {}
    for resp in resps:
        for r in resp.results:
            cur = best.get(r.node_id)
            if cur is None or r.score > cur.score:
                best[r.node_id] = r
    results = sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_n]

    diags = [r.diagnostics for r in resps if r.diagnostics]
    merged: Optional[Dict] = None
    if diags:
        scores: Dict[str, float] = {}
        filtered: Dict[str, str] = {}
        anchors: Dict[str, List[str]] = {}
        for d in diags:
            for nid, sc in (d.get("scores") or {}).items():
                if nid not in scores or sc > scores[nid]:
                    scores[nid] = sc
            for nid, why in (d.get("filtered") or {}).items():
                filtered.setdefault(nid, why)
            for kind, ids in (d.get("anchors") or {}).items():
                seen = anchors.setdefault(kind, [])
                seen.extend(i for i in ids if i not in seen)
        merged = {"scores": scores, "filtered": filtered, "anchors": anchors,
                  "decompose_merged": True}
    return results, merged


def _score_qa(
    store: GraphStore,
    engine: RetrievalEngine,
    qa: QA,
    conv: Conversation,
    answerer: A.Answerer,
    dia_map: Optional[Dict[str, List[str]]] = None,
    judge=None,
    decomposer=None,
) -> Dict:
    """Run one QA through recall -> extractive answer -> score."""
    now_dt = parse_session_date(conv.last_session_date) or datetime.now(timezone.utc)

    # Benchmark-only decompose row: split the question first (an LLM call for a
    # cloud spec, NOT counted in recall latency), then recall every part.
    queries = [qa.question]
    decompose_error: Optional[str] = None
    decompose_ms = 0.0
    if decomposer is not None:
        try:
            dec = decomposer.decompose(qa.question)
            queries = list(dec.queries) or [qa.question]
            decompose_error = dec.error
            decompose_ms = dec.latency_ms
        except Exception as e:  # noqa: BLE001 - fall back to the original query
            decompose_error = f"{type(e).__name__}: {e}"

    t0 = time.perf_counter()
    # Surface verbatim turns (CONTEXT nodes): for conversational QA the answer
    # lives in the turn itself, not only in distilled extract nodes.
    # debug=True: per-node scores/filters/anchors feed the miss classification
    # below. Overhead is a few dict copies — negligible vs the ~350ms p50, and
    # it's applied to EVERY config equally so latency comparisons stay fair.
    resps = [
        engine.recall(q, top_n=RECALL_TOP_N, now=now_dt, include_context=True, debug=True)
        for q in queries
    ]
    recall_ms = (time.perf_counter() - t0) * 1000.0  # wall time across ALL sub-recalls
    if len(resps) == 1:
        resp = resps[0]
    else:
        f_results, f_diag = _fuse_responses(resps, RECALL_TOP_N)
        resp = SimpleNamespace(results=f_results, diagnostics=f_diag)

    ctx = A.RetrievedContext(
        query=qa.question,
        contents=[r.content for r in resp.results],
        labels=[r.label for r in resp.results],
        dates=[r.recorded_at for r in resp.results],
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

    # End-to-end LLM-judge track — SEPARATE from F1, never blended (see
    # judges.py header). None/None when no judge is configured (the default
    # F1-only path); an older resumed checkpoint row simply lacks these keys,
    # which .get() downstream tolerates.
    judge_correct: Optional[bool] = None
    judge_error: Optional[str] = None
    if judge is not None:
        try:
            verdict = judge.judge(qa.question, qa.answer, prediction, qa.category_name)
            judge_correct = verdict.correct
            judge_error = verdict.error
        except Exception as e:  # noqa: BLE001 - one bad judge call ≠ dead run
            judge_correct = False
            judge_error = f"{type(e).__name__}: {e}"

    dec_fields: Dict = {}
    if decomposer is not None:
        dec_fields = {
            "n_subqueries": len(queries),
            "decompose_error": decompose_error,
            "decompose_latency_ms": round(decompose_ms, 3),
        }
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
        "judge_correct": judge_correct,
        "judge_error": judge_error,
        **dec_fields,
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
    judge_name: str = "f1",
    allow_degraded: bool = False,
    decompose_name: str = "none",
) -> Dict:
    cfg = _load_config(config_name)
    want_semantic = _semantic_requested(cfg)
    layer_statuses: List[Dict] = []
    # Out-of-config knobs: REVIEN_* already in the process env that the config
    # file does not set (snapshot BEFORE the config env is applied).
    env_overrides = {k: v for k, v in sorted(os.environ.items())
                     if k.startswith("REVIEN_") and k not in (cfg.get("env") or {})}
    prev_env = _apply_env(cfg.get("env", {}))

    try:
        answerer = A.build_answerer(answerer_name)
        judge_obj = J.build_judge(judge_name)
        decomposer = D.build_decomposer(decompose_name)
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
        ckpt_path = _checkpoint_path(
            out_dir, config_name, answerer_name, judge_name, decompose_name)
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
        alias_pass_stats: List[Dict] = []
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
        # Same bookkeeping, for the judge track (0/0.0 when no judge is active
        # or every resumed record predates the judge fields).
        resumed_judge_calls = 0
        resumed_judge_cost = 0.0
        resumed_dec_calls = 0
        resumed_dec_cost = 0.0

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
            if rec.get("alias") is not None:
                alias_pass_stats.append(rec["alias"])
            resumed_cloud_calls += int(rec.get("conv_network_calls", 0) or 0)
            resumed_cost_usd += float(rec.get("conv_cost_usd", 0.0) or 0.0)
            resumed_judge_calls += int(rec.get("conv_judge_network_calls", 0) or 0)
            resumed_judge_cost += float(rec.get("conv_judge_cost_usd", 0.0) or 0.0)
            resumed_dec_calls += int(rec.get("conv_decompose_network_calls", 0) or 0)
            resumed_dec_cost += float(rec.get("conv_decompose_cost_usd", 0.0) or 0.0)

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
            judge_calls_before = int(getattr(judge_obj, "network_calls", 0)) if judge_obj else 0
            judge_cost_before = float(getattr(judge_obj, "cost_usd_estimate", 0.0)) if judge_obj else 0.0
            dec_calls_before = int(getattr(decomposer, "network_calls", 0))
            dec_cost_before = float(getattr(decomposer, "cost_usd_estimate", 0.0))
            is_first_fresh = provenance_store is None

            fd, db_path = tempfile.mkstemp(suffix=f"_{config_name}_{conv.conv_id}.db")
            os.close(fd)

            # Pristine-cache lookup: on a hit, the working temp DB starts as a
            # COPY of the post-ingest snapshot and ingest is skipped entirely.
            cached_meta: Optional[Dict] = None
            cache_db = cache_meta_path = None
            if db_cache is not None:
                cache_db, cache_meta_path = _cache_paths(db_cache, config_name, conv.conv_id)
                cached_meta = _cache_load_meta(
                    cache_meta_path, dataset_sha, want_semantic,
                    embed_model=_resolve_embed_model() if want_semantic else None,
                    embed_context=embed_context_mode() if want_semantic else None)
                if cached_meta is not None and cache_db.exists():
                    shutil.copyfile(cache_db, db_path)
                else:
                    cached_meta = None

            store = GraphStore(db_path=db_path)
            keep_store_open = False
            try:
                semantic = SemanticIndex(store)  # self-disables without the extra

                if cached_meta is None and want_semantic and not allow_degraded:
                    _st = _layer_status(semantic)
                    if not _st["semantic_active"]:
                        _degraded_exit(conv.conv_id, _st, "before ingest")

                conv_ingest_rate = 0.0
                if cached_meta is not None:
                    # Rerank is retrieval-time: report THIS run's depth/model,
                    # not whatever the snapshot was ingested under.
                    layer_statuses.append({
                        **cached_meta["layer_status"],
                        **{k: v for k, v in
                           _layer_status(semantic, CrossEncoderReranker()).items()
                           if k.startswith("rerank_")},
                    })
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
                    status = _layer_status(semantic, CrossEncoderReranker())
                    layer_statuses.append(status)
                    degraded = want_semantic and not status["semantic_active"]
                    if degraded:
                        if db_cache is not None:
                            print(f"[bench] db-cache: NOT caching {conv.conv_id} — "
                                  f"semantic layer inactive during ingest "
                                  f"({status['semantic_inactive_reason']})")
                        if not allow_degraded:
                            _degraded_exit(conv.conv_id, status, "during ingest")
                    if db_cache is not None and not degraded:
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
                            "layer_status": status,
                        }), encoding="utf-8")
                total_audit_creates_expected += summary["nodes_created"]

                # Optional alias-inference measurement hook — after ingest,
                # before recall (see _bench_alias_enabled's header). Runs on
                # BOTH cache-hit and freshly-ingested conversations alike, so
                # --db-cache reuse never silently skips it.
                alias_stats: Optional[Dict] = None
                if _bench_alias_enabled():
                    alias_stats = _run_bench_alias_pass(store, semantic)
                    alias_pass_stats.append(alias_stats)

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
                    row = _score_qa(
                        store, engine, qa, conv, answerer,
                        dia_map=dia_map, judge=judge_obj,
                        decomposer=decomposer,
                    )
                    recall_latencies.append(row["recall_latency_ms"])
                    per_q.append(row)
                    conv_rows.append(row)

                n_ran += 1

                # Per-conv cost/calls delta (cloud readers self-count globally).
                conv_calls = int(getattr(answerer, "network_calls", 0)) - calls_before
                conv_cost = float(getattr(answerer, "cost_usd_estimate", 0.0)) - cost_before
                conv_judge_calls = (
                    int(getattr(judge_obj, "network_calls", 0)) - judge_calls_before
                    if judge_obj else 0
                )
                conv_judge_cost = (
                    float(getattr(judge_obj, "cost_usd_estimate", 0.0)) - judge_cost_before
                    if judge_obj else 0.0
                )

                conv_dec_calls = int(getattr(decomposer, "network_calls", 0)) - dec_calls_before
                conv_dec_cost = float(getattr(decomposer, "cost_usd_estimate", 0.0)) - dec_cost_before

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
                        "conv_judge_network_calls": conv_judge_calls,
                        "conv_judge_cost_usd": round(conv_judge_cost, 6),
                        "conv_decompose_network_calls": conv_dec_calls,
                        "conv_decompose_cost_usd": round(conv_dec_cost, 6),
                        "alias": alias_stats,
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
        judge_calls_total = (
            (int(getattr(judge_obj, "network_calls", 0)) if judge_obj else 0)
            + resumed_judge_calls
        )
        judge_cost_total = (
            (float(getattr(judge_obj, "cost_usd_estimate", 0.0)) if judge_obj else 0.0)
            + resumed_judge_cost
        )

        dec_calls_total = int(getattr(decomposer, "network_calls", 0)) + resumed_dec_calls
        dec_cost_total = float(getattr(decomposer, "cost_usd_estimate", 0.0)) + resumed_dec_cost

        # Honest egress check: config-derived, names any cloud backend. The
        # measured-call secondary signal covers BOTH the reader and the judge —
        # a cloud judge with reader calls at 0 must still trip the counter.
        checks.append(
            S.network_egress_zero(
                cloud_calls=cloud_calls + judge_calls_total + dec_calls_total,
                answerer=answerer_name,
                judge=judge_name,
                decompose=decompose_name,
            )
        )
        checks.extend(S.run_consent_subtests())

        # ── Aggregate metrics ─────────────────────────────────────────────────
        report = _aggregate(per_q, ingest_rates, recall_latencies)
        # Retrieval failure taxonomy: where the missed gold evidence died.
        # Rows from pre-taxonomy checkpoints are counted as unclassified.
        report["retrieval_failure_analysis"] = FA.aggregate_failures(per_q)
        report["alias"] = _aggregate_alias(alias_pass_stats, enabled=_bench_alias_enabled())
        report["normalization_merges_sample"] = norm_merges_sample
        report["sovereignty"] = S.checks_to_dict(checks)
        report["layer_status"] = _aggregate_layer_status(layer_statuses, want_semantic)
        report["layer_status"]["allow_degraded"] = bool(allow_degraded)
        report["env_overrides"] = env_overrides
        report["config"] = {
            "name": config_name,
            "env": cfg.get("env", {}),
            "cluster": bool(cfg.get("cluster")),
            "answerer": answerer.name,
            "judge": judge_obj.name if judge_obj is not None else "f1",
            "decompose": decomposer.name if decomposer is not None else "none",
            "recall_top_n": RECALL_TOP_N,
            "recall_ks": list(RECALL_KS),
        }
        # End-to-end LLM-judge track — SEPARATE from F1 above, never blended.
        # Present only when an LLM judge is actually configured (spec != 'f1');
        # the default F1-only path's report shape is unchanged.
        if judge_obj is not None:
            reader_provider, _ = A.parse_provider(answerer_name)
            report["reader"] = {
                "spec": answerer_name,
                "model": answerer.name,
                "prompt_sha256": (
                    None if reader_provider == "extractive" else A.ANSWER_PROMPT_SHA256
                ),
            }
            report["judge"] = _aggregate_judge(
                per_q, judge_name, judge_obj.name, judge_calls_total, judge_cost_total
            )
        if decomposer is not None:
            dec_rows = [r for r in per_q if r.get("n_subqueries") is not None]
            report["decompose"] = {
                "spec": decompose_name,
                "model": decomposer.name,
                "prompt_sha256": D.DECOMPOSE_PROMPT_SHA256,
                "network_calls": dec_calls_total,
                "cost_usd": round(dec_cost_total, 6),
                "cost_usd_is_estimate": True,
                "decompose_errors": sum(1 for r in dec_rows if r.get("decompose_error")),
                # Sub-queries include the original question, so this is >= 1.
                "mean_subqueries": (
                    round(sum(r["n_subqueries"] for r in dec_rows) / len(dec_rows), 3)
                    if dec_rows else None
                ),
                "n_questions": len(dec_rows),
                "taxonomy_basis": "merged_subrecall_diagnostics",
            }
        report["reader_context"] = READER_CONTEXT
        report["dataset"] = {
            "path": str(dataset_path),
            "sha256": read_locked_hash(),
            "conversations": len(conversations),
        }
        # All-local headline run: cost_usd_estimate is 0.0 and cloud_calls is 0,
        # so this preserves the honest $0 / 0-calls local default. Cloud answerer
        # runs surface a non-zero, clearly-labelled cost ESTIMATE + real call count.
        # Top-level cost_usd/network_calls (F1): must be the HONEST total across
        # both the reader AND the judge — a report that only counted the reader
        # while a cloud judge silently made calls understated both. The per-block
        # breakdown (report["judge"]) still carries the judge-only figures.
        report["cost_usd"] = round(cost_usd_estimate + judge_cost_total + dec_cost_total, 6)
        report["cost_usd_is_estimate"] = True
        report["network_calls"] = cloud_calls + judge_calls_total + dec_calls_total
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


def _aggregate_judge(
    per_q: List[Dict],
    judge_spec: str,
    judge_model_name: str,
    network_calls: int,
    cost_usd: float,
) -> Dict:
    """Fold per-question judge_correct/judge_error rows into one run-level
    judge block. SEPARATE from F1 (report["overall_f1"]/["per_category_f1"]) —
    never blended, see judges.py header. Only rows carrying a non-None
    judge_correct are counted, so an older resumed row from a pre-judge
    checkpoint (or a run mixing a judge-less resume with a judged rerun) never
    silently corrupts the accuracy denominator."""
    rows = [r for r in per_q if r.get("judge_correct") is not None]
    n = len(rows)
    error_rows = [r for r in rows if r.get("judge_error")]
    errors = len(error_rows)
    # F5: judge errors are excluded from the accuracy DENOMINATOR — an
    # unparseable/failed verdict is neither a correct nor a wrong answer, it's
    # a missing measurement, and folding it into the denominator as an
    # implicit "wrong" understated accuracy. accuracy_denominator is the
    # actually-judged (non-error) count; accuracy_overall is None (not 0.0)
    # when nothing was judged, so a report reader can't mistake "no data" for
    # "0% accuracy".
    denom_rows = [r for r in rows if not r.get("judge_error")]
    accuracy_denominator = len(denom_rows)
    n_correct = sum(1 for r in denom_rows if r["judge_correct"])
    accuracy_overall = (
        round(n_correct / accuracy_denominator, 4)
        if accuracy_denominator else None
    )

    by_cat: Dict[int, List[Dict]] = {}
    for r in denom_rows:
        by_cat.setdefault(r["category"], []).append(r)
    per_category: Dict[str, Dict] = {}
    for cat, crows in sorted(by_cat.items()):
        per_category[CATEGORY_NAMES.get(cat, str(cat))] = {
            "n": len(crows),
            "accuracy": round(
                sum(1 for x in crows if x["judge_correct"]) / len(crows), 4
            ),
        }

    # LoCoMo category 5 (adversarial) gold is the WRONG answer by construction;
    # the right behaviour is a refusal, which a gold-comparison judge marks
    # WRONG. Published LLM-judge accuracies exclude the category for that
    # reason, so the comparable figure is reported alongside, never instead.
    non_adv = [r for r in denom_rows if not r.get("is_adversarial")]
    n_correct_excl_adv = sum(1 for r in non_adv if r["judge_correct"])
    accuracy_excl_adversarial = (
        round(n_correct_excl_adv / len(non_adv), 4) if non_adv else None
    )

    return {
        "spec": judge_spec,
        "model": judge_model_name,
        "prompt_sha256": J.JUDGE_PROMPT_SHA256,
        "max_tokens": J.JUDGE_MAX_TOKENS,
        "accuracy_overall": accuracy_overall,
        "accuracy_denominator": accuracy_denominator,
        "n_correct": n_correct,
        "accuracy_excl_adversarial": accuracy_excl_adversarial,
        "n_excl_adversarial": len(non_adv),
        "per_category_accuracy": per_category,
        "judge_errors": errors,
        "n_judged": n,
        "network_calls": network_calls,
        "cost_usd": round(cost_usd, 6),
        "cost_usd_is_estimate": True,
    }


def _degraded_exit(conv_id: str, status: Dict, when: str) -> None:
    print(f"[bench] ERROR: config requests the semantic layer but it is inactive "
          f"{when} ({conv_id}): {status.get('semantic_inactive_reason')}. "
          f"Refusing to produce a degraded number; pass --allow-degraded to override.")
    raise SystemExit(3)


def _embedder_label(ls: Dict) -> str:
    """`<provider>:<model>` for the Layers line (provider alone when unknown)."""
    emb, model = ls.get("embedder"), ls.get("embed_model")
    return f"{emb}:{model}" if model else f"{emb}"


def _aggregate_layer_status(statuses: List[Dict], requested: bool) -> Dict:
    """Run-level layer status: active only if EVERY observed conversation was."""
    if not statuses:
        return {"semantic_requested": requested, "semantic_active": None,
                "rerank_active": None, "rerank_top_k": None, "rerank_model": None,
                "embedder": None, "embed_model": None, "embed_dim": None,
                "embed_context": None,
                "semantic_inactive_reason": None}
    reasons = sorted({s["semantic_inactive_reason"] for s in statuses
                      if s.get("semantic_inactive_reason")})
    return {
        "semantic_requested": requested,
        "semantic_active": all(s.get("semantic_active") for s in statuses),
        "rerank_active": all(s.get("rerank_active") for s in statuses),
        "rerank_top_k": statuses[-1].get("rerank_top_k"),
        "rerank_model": statuses[-1].get("rerank_model"),
        "embedder": statuses[-1].get("embedder"),
        "embed_model": statuses[-1].get("embed_model"),
        "embed_dim": statuses[-1].get("embed_dim"),
        "embed_context": statuses[-1].get("embed_context"),
        "semantic_inactive_reason": "; ".join(reasons) or None,
    }


def _aggregate_alias(stats: List[Dict], enabled: bool) -> Dict:
    """Fold per-conversation alias-pass stats (REVIEN_BENCH_ALIAS=1) into one
    run-level report section. Present-but-empty when the hook never ran (env
    unset, or a resumed checkpoint predates the hook) so a report diff always
    shows whether the measurement fired, never a silently missing key."""
    if not enabled or not stats:
        return {"enabled": enabled, "ran": False}
    from revien.alias import ALIAS_SAMPLE_CAP

    edges_by_method: Dict[str, int] = {}
    for s in stats:
        for method, n in (s.get("edges_by_method") or {}).items():
            edges_by_method[method] = edges_by_method.get(method, 0) + n
    sample: List[Dict] = []
    for s in stats:
        sample.extend(s.get("sample") or [])
    return {
        "enabled": True,
        "ran": True,
        "conversations": len(stats),
        "entities_considered": sum(s.get("entities_considered", 0) for s in stats),
        "candidates_considered": sum(s.get("candidates_considered", 0) for s in stats),
        "edges_created": sum(s.get("edges_created", 0) for s in stats),
        "edges_by_method": edges_by_method,
        "duration_ms_total": round(sum(s.get("duration_ms", 0.0) for s in stats), 3),
        "notes": sorted({s["note"] for s in stats if s.get("note")}),
        "sample": sample[:ALIAS_SAMPLE_CAP],
    }


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
                    choices=["graph_only", "semantic", "neural", "semantic_rrf"])
    ap.add_argument("--answerer", default="extractive",
                    help="reader spec: extractive | ollama:<model> | "
                         "openai:<model> | openrouter:<model> | together:<model> | "
                         "claude:<model>")
    ap.add_argument("--judge", default="f1",
                    help="end-to-end LLM-judge spec (SEPARATE from F1, never "
                         "blended): f1 (no LLM judge, default) | ollama:<model> "
                         "(local, egress PASS) | openai:<model> | "
                         "openrouter:<model> | together:<model> | claude:<model> "
                         "(cloud, egress FAIL by design, labeled)")
    ap.add_argument("--decompose", default="none",
                    help="BENCHMARK-ONLY query decomposition row: none (default) | "
                         "ollama:<model> (local, egress PASS if loopback) | "
                         "openai|openrouter|together|claude:<model> (cloud, egress "
                         "FAIL by design, labeled). Splits each question into "
                         "sub-questions, recalls each, unions before the reader.")
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
    ap.add_argument("--allow-degraded", action="store_true", dest="allow_degraded",
                    help="continue even when the config requests the semantic "
                         "layer but it is inactive (result is labelled degraded). "
                         "Default: exit 3 rather than emit a degraded number.")
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
        judge_name=args.judge,
        allow_degraded=args.allow_degraded,
        decompose_name=args.decompose,
    )
    _print_summary(report)
    return 0


def _print_summary(report: Dict) -> None:
    print("\n=== Revien LoCoMo benchmark ===")
    print(f"config        : {report['config']['name']} / {report['config']['answerer']}")
    print(f"questions     : {report['n_questions']}")
    ls = report.get("layer_status") or {}
    print(f"layers        : semantic={ls.get('semantic_active')} "
          f"rerank={ls.get('rerank_active')} embedder={_embedder_label(ls)} "
          f"rerank_top_k={ls.get('rerank_top_k')}")
    if report.get("env_overrides"):
        print("Env overrides : " + " ".join(
            f"{k}={v}" for k, v in report["env_overrides"].items()))
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
    dec = report.get("decompose")
    if dec:
        print(f"decompose     : {dec['model']} calls={dec['network_calls']} "
              f"cost=${dec['cost_usd']} mean_subqueries={dec['mean_subqueries']} "
              f"errors={dec['decompose_errors']}")
    judge = report.get("judge")
    if judge:
        print(f"judge (LLM)   : {judge['spec']} accuracy={judge['accuracy_overall']} "
              f"(excl. adversarial: {judge.get('accuracy_excl_adversarial')}, "
              f"n={judge.get('n_excl_adversarial')}) "
              f"n={judge['n_judged']} errors={judge['judge_errors']} "
              f"cost=${judge['cost_usd']} calls={judge['network_calls']} "
              f"— NOT comparable to the F1/retrieval numbers above")
    print(f"results JSON  : {report.get('_out_path')}")


if __name__ == "__main__":
    raise SystemExit(main())
