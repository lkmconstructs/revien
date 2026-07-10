"""
revien_bench.sweep — ranking-knob sweep over cached ingests.

The at-scale miss taxonomy says 72% of semantic-path misses are `outranked`
(gold node walked AND scored, median best rank 26 vs the top-10 cutoff). This
harness turns the ranking knobs one at a time and reads recall + the taxonomy
per variant, against IDENTICAL pristine ingests (--db-cache), so a delta means
the knob — not ingest noise.

Knobs (env, read by RetrievalEngine / ScoringConfig.from_env):
    REVIEN_SEMANTIC_TOP_K          how many nearest neighbours get the
                                   similarity-primary score (default 30)
    REVIEN_SEMANTIC_SIM_FLOOR      admission floor for semantic anchors (0.30)
    REVIEN_GRAPH_REFINE            graph-composite weight in the semantic
                                   blend (0.25)
    REVIEN_RECENCY_WEIGHT /        three-factor weights (0.35/0.30/0.35) —
    REVIEN_FREQUENCY_WEIGHT /      pass a full set summing to 1.0
    REVIEN_PROXIMITY_WEIGHT
    REVIEN_RECENCY_HALF_LIFE_DAYS  recency decay half-life (7.0)
    REVIEN_EDGE_WEIGHT_BLEND       path-strength share of the proximity term
                                   (default 0.0 = hop-only, shipped)
    REVIEN_EDGE_CONFIDENCE_IN_WALK multiply edge confidence into walk
                                   strength (default 0)

Usage:
    python -m revien_bench.sweep --limit 2 --db-cache <dir> --out <dir>
    python -m revien_bench.sweep --variants baseline,topk100 ...

Each variant runs fresh (its own out subdir, no checkpoint cross-talk) and the
comparison table prints at the end. Extractive answerer only — retrieval is
what the knobs move; a reader would just add noise and cost.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .fetch_locomo import DATA_PATH
from .runner import run_benchmark

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent

# One hypothesis per variant, stated where the numbers will land.
#
# ROUND 1 (access-time recency, July 2 2026 AM) is settled — results in
# results/sweep history + HANDOFF: no_freq won big (+19% recall@10, +89% @1,
# all from the outranked bucket); topk100/refine10 hurt; floor15 inert
# (top-30 all clear 0.30); halflife365 inert because access-time recency was
# CONSTANT in-bench — which exposed the recency-semantics bug.
#
# ROUND 2 (below) runs on CONTENT-time recency (recorded_at): the half-life
# knob is now live, and REVIEN_TOUCH_ON_RECALL=0 makes "honest frequency"
# (only mark_used feeds access_count) measurable against "no frequency".
VARIANTS: Dict[str, Dict[str, str]] = {
    # Control: the SHIPPED defaults. As of the July 2 2026 ship call that's
    # the round-2 champion itself — content recency at 365d half-life, no
    # self-touch (mark_used-only frequency). Round-2 rows measured before the
    # flip had baseline = content recency + 7d + self-touch.
    "baseline": {},
    # Round-1 champion re-run under content recency: freq amputated outright.
    "no_freq": {
        "REVIEN_RECENCY_WEIGHT": "0.5",
        "REVIEN_FREQUENCY_WEIGHT": "0.0",
        "REVIEN_PROXIMITY_WEIGHT": "0.5",
    },
    # Honest frequency: keep the 0.30 weight but recall() stops touching its
    # own results — only mark_used() (real usage) increments access_count.
    # In-bench nobody calls mark_used, so freq contributes a constant 0: this
    # measures the "keep the feature, fix the signal" option.
    "no_touch": {"REVIEN_TOUCH_ON_RECALL": "0"},
    # Content recency at 7d half-life zeroes anything said >1 month ago.
    # LoCoMo gold spans months of sessions — flatter decay may recover it.
    "hl90": {"REVIEN_RECENCY_HALF_LIFE_DAYS": "90"},
    "hl365": {"REVIEN_RECENCY_HALF_LIFE_DAYS": "365"},
    # Combo candidates: cleaner frequency signal x flatter content decay.
    "no_touch_hl90": {
        "REVIEN_TOUCH_ON_RECALL": "0",
        "REVIEN_RECENCY_HALF_LIFE_DAYS": "90",
    },
    "no_freq_hl90": {
        "REVIEN_RECENCY_WEIGHT": "0.5",
        "REVIEN_FREQUENCY_WEIGHT": "0.0",
        "REVIEN_PROXIMITY_WEIGHT": "0.5",
        "REVIEN_RECENCY_HALF_LIFE_DAYS": "90",
    },
    # Round-2 refinement: no_touch won the frequency fork outright; the
    # half-life ladder showed 7d crushes @1 and 90d crushes @10, while 365d
    # lifts @1. Near-flat decay + honest frequency: recency becomes a gentle
    # tiebreak instead of a burial.
    "no_touch_hl365": {
        "REVIEN_TOUCH_ON_RECALL": "0",
        "REVIEN_RECENCY_HALF_LIFE_DAYS": "365",
    },
    # Same idea via weight instead of half-life: keep 7d decay but shrink
    # recency's share so it can't bury months-old gold.
    "no_touch_lowrec": {
        "REVIEN_TOUCH_ON_RECALL": "0",
        "REVIEN_RECENCY_WEIGHT": "0.15",
        "REVIEN_FREQUENCY_WEIGHT": "0.30",
        "REVIEN_PROXIMITY_WEIGHT": "0.55",
    },
    # ROUND 3 (weighted walk, July 10 2026). Edges have carried weight
    # (author-drawn vault links 0.8, extractor co-occurrence guesses 0.3) and
    # mark_used() reinforcement since leg 1 — and the BFS read none of it:
    # hop count was the only proximity signal. The blend ladder mixes path
    # strength (product of edge weights along the shortest-hop path) into
    # proximity: (1-b)*hop_decay + b*strength. Hypothesis: wins come out of
    # `outranked` (hub nodes reached over weak guess-edges lose rank to nodes
    # reached over strong ones); `disconnected` must not move — strength
    # shapes ranking, never reachability. ew*_conf multiplies edge confidence
    # in: mostly-0.5 defaults predict it inert, but if CSL-set confidences
    # separate at scale it's a free second signal.
    "ew25": {"REVIEN_EDGE_WEIGHT_BLEND": "0.25"},
    "ew50": {"REVIEN_EDGE_WEIGHT_BLEND": "0.5"},
    "ew75": {"REVIEN_EDGE_WEIGHT_BLEND": "0.75"},
    "ew100": {"REVIEN_EDGE_WEIGHT_BLEND": "1.0"},
    "ew50_conf": {
        "REVIEN_EDGE_WEIGHT_BLEND": "0.5",
        "REVIEN_EDGE_CONFIDENCE_IN_WALK": "1",
    },
    # ROUND 3b (cross-encoder rerank, July 10 2026). The ew ladder came back
    # IDENTICAL to baseline on both corpora — measured mechanism: the entire
    # top-20 is distance-0 semantic anchors, so proximity (hop OR strength)
    # never decides ranks; the outranked bucket is anchor-vs-anchor bi-encoder
    # misranking. The lever that CAN reorder the head is a cross-encoder
    # reading query+candidate together: rescore the top-K base-ranked results
    # (median outrank 26 -> a 30-head covers it), tail untouched.
    # Hypothesis: outranked shrinks, recall@1 moves most (head reordering),
    # recall_p50_ms pays the model cost — the sweep prices that trade.
    "rerank": {"REVIEN_RERANK": "1"},
    "rerank_k50": {"REVIEN_RERANK": "1", "REVIEN_RERANK_TOP_K": "50"},
}


def _apply(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    prev = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    return prev


def _restore(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _row(name: str, env: Dict[str, str], report: Dict) -> Dict:
    r = report["retrieval"]
    fa = report.get("retrieval_failure_analysis") or {}
    causes = fa.get("by_cause", {})
    return {
        "variant": name,
        "env": env,
        "recall@1": r["recall@1"],
        "recall@5": r["recall@5"],
        "recall@10": r["recall@10"],
        "mrr": r["mrr"],
        "ndcg@10": r["ndcg@10"],
        "missed": fa.get("gold_items_missed", 0),
        "outranked": causes.get("outranked", 0),
        "disconnected": causes.get("disconnected", 0),
        "other_miss": sum(
            n for c, n in causes.items() if c not in ("outranked", "disconnected")
        ),
        "median_outrank": (fa.get("outranked_detail") or {}).get("median_best_rank"),
        "recall_p50_ms": report["latency_ms"]["recall"]["p50"],
        "results_json": report.get("_out_path"),
    }


def _print_table(rows: List[Dict]) -> None:
    cols = ["variant", "recall@1", "recall@5", "recall@10", "mrr", "ndcg@10",
            "missed", "outranked", "disconnected", "other_miss",
            "median_outrank", "recall_p50_ms"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n=== ranking-knob sweep ===")
    print(header)
    print("-" * len(header))
    base = rows[0] if rows else None
    for r in rows:
        line = "  ".join(str(r[c]).ljust(widths[c]) for c in cols)
        if base is not None and r is not base and base["recall@10"]:
            delta = r["recall@10"] - base["recall@10"]
            line += f"  ({'+' if delta >= 0 else ''}{delta:.4f} vs baseline)"
        print(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Revien ranking-knob sweep")
    ap.add_argument("--config", default="semantic",
                    help="bench config every variant runs under (default: semantic)")
    ap.add_argument("--variants", default=None,
                    help=f"comma list from: {', '.join(VARIANTS)} (default: all)")
    ap.add_argument("--limit", type=int, default=2,
                    help="conversations per variant (default 2 — exploration; "
                         "confirm winners on the full set)")
    ap.add_argument("--max-qa", type=int, default=None, dest="max_qa")
    ap.add_argument("--dataset", default=str(DATA_PATH))
    ap.add_argument("--db-cache", required=True, dest="db_cache",
                    help="pristine ingest cache dir (built on first run)")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results" / "sweep"))
    args = ap.parse_args(argv)

    names = list(VARIANTS) if args.variants is None else [
        v.strip() for v in args.variants.split(",") if v.strip()
    ]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; valid: {list(VARIANTS)}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found at {dataset_path}")

    out_root = Path(args.out)
    rows: List[Dict] = []
    for name in names:
        env = VARIANTS[name]
        print(f"\n--- variant {name!r} {env or '(defaults)'} ---")
        prev = _apply(env)
        try:
            report = run_benchmark(
                config_name=args.config,
                answerer_name="extractive",
                dataset_path=dataset_path,
                out_dir=out_root / name,
                limit_convs=args.limit,
                max_qa=args.max_qa,
                fresh=True,  # each variant is self-contained; never resume across knobs
                db_cache=Path(args.db_cache),
            )
        finally:
            _restore(prev)
        rows.append(_row(name, env, report))

    _print_table(rows)

    summary_path = out_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_sweep.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nsweep summary : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
