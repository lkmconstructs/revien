"""
revien_bench.failure_analysis — classify WHY each gold evidence item was missed.

recall@10 says HOW MUCH gold evidence recall misses (~53% on the last semantic
run). This module says WHERE each miss died in the pipeline, per query, so
tuning effort goes to the stage that's actually losing it:

    never_extracted  — NO node in the store carries this dia_id. Ingestion
                       never produced (or dedup swallowed) a surviving node
                       for the gold turn. Fix lives in ingestion.
    no_anchors       — anchor selection found NOTHING for the query, so the
                       walk never started. Total seed miss. Fix lives in
                       anchor selection (semantic seeds / query parsing).
    walk_depth_miss  — a gold node exists and IS reachable from the anchors,
                       but beyond the engine's max_depth. Fix: walk radius or
                       better (closer) anchors.
    disconnected     — a gold node exists but has NO path from any anchor even
                       at DEEP_DEPTH. The anchors landed in the wrong region
                       of the graph (seed miss) or the edges to reach it were
                       never created (ingestion edge gap).
    filtered_out     — a gold node was walked but excluded by a result filter
                       (invalidated / context_excluded).
    outranked        — a gold node was walked AND scored, but ranked below
                       top-k. The ONLY class where the scoring formula itself
                       is the culprit.

When a dia_id maps to several nodes (verbatim context + extracted nodes), the
most favorable node decides the class: if ANY node was scored it's outranked;
else filtered; else depth/disconnected. 'never_extracted' requires NO node.

Needs engine.recall(..., debug=True) — the diagnostics dict (anchor sets,
walked distances, per-node scores incl. sub-threshold, filter reasons) is the
input to this classification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from revien.graph.store import GraphStore
from revien.retrieval.walker import GraphWalker

# How far the "was it reachable at all?" probe walks. Deliberately generous:
# past this, 'walk_depth_miss' and 'disconnected' are operationally identical
# (no realistic walk radius reaches it).
DEEP_DEPTH = 6


def normalization_merge_report(store: GraphStore) -> Dict:
    """The false-merge precision surface: every dedup merge that happened ONLY
    because of label normalization, as '"candidate" -> "existing"' pairs from
    the audit log. Recall gains from normalization are only trustworthy next
    to this list — a greedy matcher shows up here before it shows up as a
    wrong answer."""
    pairs = sorted({
        row["actor"]
        for row in store.get_all_audit()
        if row.get("op") == "normalized_merge" and row.get("actor")
    })
    return {"count": len(pairs), "pairs": pairs}


def build_dia_map(store: GraphStore) -> Dict[str, List[str]]:
    """Map dia_id -> [node_ids] for every node in the store. Built ONCE per
    conversation (one full node scan), then shared across all its QA rows."""
    dia_map: Dict[str, List[str]] = {}
    for node in store.list_nodes(limit=999999):
        dia = (node.metadata or {}).get("dia_id")
        if dia:
            dia_map.setdefault(dia, []).append(node.node_id)
    return dia_map


def classify_misses(
    store: GraphStore,
    diagnostics: Optional[Dict],
    gold: Set[str],
    retrieved: Sequence[str],
    dia_map: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Classify every gold dia_id absent from `retrieved`. Returns
    {dia_id: {"cause": ..., ...detail}}; empty dict when nothing was missed.

    `diagnostics` is RetrievalResponse.diagnostics from a debug=True recall.
    A None/empty diagnostics (defensive) classifies conservatively using only
    the store: never_extracted vs unknown.
    """
    diag = diagnostics or {}
    scores: Dict[str, float] = diag.get("scores", {})
    filtered: Dict[str, str] = diag.get("filtered", {})
    anchors: List[str] = (diag.get("anchors") or {}).get("all", [])

    retrieved_set = set(retrieved)
    out: Dict[str, Dict] = {}
    deep_reach: Optional[Dict[str, int]] = None  # lazy — deep walk only if needed

    for dia in sorted(gold):
        if dia in retrieved_set:
            continue

        node_ids = dia_map.get(dia, [])
        if not node_ids:
            out[dia] = {"cause": "never_extracted"}
            continue

        if not diag:
            out[dia] = {"cause": "unknown_no_diagnostics"}
            continue

        scored = [(nid, scores[nid]) for nid in node_ids if nid in scores]
        if scored:
            best_nid, best_score = max(scored, key=lambda x: x[1])
            # 1-based rank of the best gold node among everything scored.
            rank = 1 + sum(1 for s in scores.values() if s > best_score)
            out[dia] = {
                "cause": "outranked",
                "best_rank": rank,
                "best_score": round(best_score, 4),
            }
            continue

        reasons = sorted({filtered[nid] for nid in node_ids if nid in filtered})
        if reasons:
            out[dia] = {"cause": "filtered_out", "reasons": reasons}
            continue

        if not anchors:
            out[dia] = {"cause": "no_anchors"}
            continue

        if deep_reach is None:
            deep_reach = GraphWalker(store, max_depth=DEEP_DEPTH).walk(list(anchors))
        if any(nid in deep_reach for nid in node_ids):
            out[dia] = {"cause": "walk_depth_miss"}
        else:
            out[dia] = {"cause": "disconnected"}

    return out


def aggregate_failures(per_q_rows: List[Dict]) -> Dict:
    """Fold per-row `gold_miss_causes` into overall + per-category counts.

    Tolerates rows without the key (e.g. conversations resumed from a
    checkpoint written before this analysis existed) — they're counted as
    unclassified, never guessed.
    """
    by_cause: Dict[str, int] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    outranked_ranks: List[int] = []
    total_missing = 0
    rows_classified = 0
    rows_unclassified = 0

    for row in per_q_rows:
        if not row.get("gold_evidence"):
            continue  # no gold to miss — retrieval metrics skip these too
        causes = row.get("gold_miss_causes")
        if causes is None:
            rows_unclassified += 1
            continue
        rows_classified += 1
        cat = row.get("category_name", "?")
        cat_bucket = by_category.setdefault(cat, {})
        for _dia, info in causes.items():
            cause = info.get("cause", "?")
            total_missing += 1
            by_cause[cause] = by_cause.get(cause, 0) + 1
            cat_bucket[cause] = cat_bucket.get(cause, 0) + 1
            if cause == "outranked" and isinstance(info.get("best_rank"), int):
                outranked_ranks.append(info["best_rank"])

    outranked_summary = None
    if outranked_ranks:
        ranks = sorted(outranked_ranks)
        outranked_summary = {
            "n": len(ranks),
            "median_best_rank": ranks[len(ranks) // 2],
            "within_20": sum(1 for r in ranks if r <= 20),
        }

    return {
        "gold_items_missed": total_missing,
        "by_cause": dict(sorted(by_cause.items(), key=lambda kv: -kv[1])),
        "by_category": by_category,
        "outranked_detail": outranked_summary,
        "rows_classified": rows_classified,
        "rows_unclassified": rows_unclassified,
    }
