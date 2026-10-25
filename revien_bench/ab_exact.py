"""
revien_bench.ab_exact - offline A/B of the exact lane (REVIEN_EXACT) against
the same engine with the lane off, on YOUR graph and YOUR questions.

    python -m revien_bench.ab_exact --db /path/to/revien.db \\
        --queries /path/to/questions.json [--floor 0.458] [--top-n 10] \\
        [--fork-ranks fork.json] [--rerank on|off] [--out report.json]

What it does
  * Works on a COPY of the database (SQLite backup into a temp dir). The
    original is opened read-only and never written. Opening the copy runs the
    normal connect-time migrations (including 007's indexes) on the copy only.
  * Sets REVIEN_SEMANTIC=1 and, when --floor is given, REVIEN_SEMANTIC_SIM_FLOOR
    (otherwise whatever is already in the environment). Every other REVIEN_*
    variable is left as the caller set it and is printed in the report, so the
    settings are visible. REVIEN_EXACT itself is not read: the two modes are
    two engines built with exact=False and exact=True over the SAME store,
    SAME semantic index and SAME reranker, so nothing differs but the lane.
  * Runs each query 3 times per mode (default --runs 3). Ranks come from run 1
    (run 1 also carries debug diagnostics); latency is the median over runs.
    A mode whose rankings differ between runs is flagged `unstable`.
  * Reports Recall@N (mean fraction of gold ids found in the top N), Any@N
    (queries with at least one gold in the top N, the "23/30" count), Hit@1 and
    MRR (first gold, rank within top N else 0) PER CATEGORY, plus per-query
    ranks for both modes. Paraphrase and exact/date categories are separate
    groups (see group_of).
  * Refuses to run when the semantic layer is not active (exit 2) unless
    --allow-graph-only is given: an A/B without the semantic layer is not the
    target configuration.

Query file: a JSON list (or {"queries": [...]}) of objects with the question
text (text | query | question | q), the gold node ids (gold | gold_ids |
gold_node_ids | gold_nodes | expected | answer_ids; a string or a list) and a
category label (category | cat | label | type | kind).

The gate (see AB_EXACT.md): on the same snapshot, the lane must tie or beat
switch-off on paraphrases and on EVERY category (name-plus-topic included),
AND tie or beat the comparison system's ("fork") numbers on names/titles and
dates (together, names alone, dates alone and each name/date category), or
it stays off. Date questions whose gold the lane cannot see are reported
apart, with a count of the ones the lane pushed down. Every comparison with
the fork is over the same questions on both sides: the exact/date questions
that have fork ranks. Supply the fork's rankings with --fork-ranks (JSON object
{"<query text>": ["<node id>", ...]} ranked best first, or a list of
{"query"|"text": ..., "results"|"ranked"|"fork": [id | {"node_id": id}, ...]});
without it the exact/date half of the gate is reported as "not evaluated".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

TEXT_KEYS = ("text", "query", "question", "q")
GOLD_KEYS = ("gold", "gold_ids", "gold_node_ids", "gold_nodes", "expected",
             "answer_ids")
CAT_KEYS = ("category", "cat", "label", "type", "kind")
RUNS = 3

EXACT_WORDS = ("exact", "name", "title", "baseline", "id", "code", "quote")
DATE_WORDS = ("date", "time", "day")


def group_of(category: str) -> str:
    """paraphrase | date | exact | other. Paraphrase is always its own group;
    a category naming a date is a date; anything naming a name/title/exact
    form is exact; everything else stays 'other' and is reported on its own."""
    c = (category or "").strip().lower()
    if "paraphras" in c:
        return "paraphrase"
    if any(w in c for w in DATE_WORDS):
        return "date"
    if any(w in c for w in EXACT_WORDS):
        return "exact"
    return "other"


def _first(d: Dict[str, Any], keys: Tuple[str, ...]):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("queries") or raw.get("items") or []
    out = []
    for i, item in enumerate(raw):
        text = _first(item, TEXT_KEYS)
        gold = _first(item, GOLD_KEYS)
        cat = _first(item, CAT_KEYS) or "uncategorized"
        if text is None or gold is None:
            raise ValueError(
                f"query {i}: needs question text ({'/'.join(TEXT_KEYS)}) and "
                f"gold ids ({'/'.join(GOLD_KEYS)}); got keys {sorted(item)}")
        if isinstance(gold, (str, int)):
            gold = [gold]
        gold = [g["node_id"] if isinstance(g, dict) else str(g) for g in gold]
        out.append({"text": str(text), "gold": gold, "category": str(cat)})
    return out


def load_fork(path: str) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def ids(seq):
        return [x["node_id"] if isinstance(x, dict) else str(x) for x in seq]

    if isinstance(raw, dict) and "results" not in raw and "queries" not in raw:
        return {k: ids(v) for k, v in raw.items() if isinstance(v, list)}
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("queries") or []
    out = {}
    for item in raw:
        text = _first(item, TEXT_KEYS)
        seq = _first(item, ("results", "ranked", "fork"))
        if text is not None and isinstance(seq, list):
            out[str(text)] = ids(seq)
    return out


# ── metrics ───────────────────────────────────────────────────────────────────
def rank_of(ranked: List[str], gold: List[str], n: int) -> Optional[int]:
    g = set(gold)
    for i, nid in enumerate(ranked[:n], start=1):
        if nid in g:
            return i
    return None


def score_one(ranked: List[str], gold: List[str], n: int) -> Dict[str, Any]:
    g = set(gold)
    found = len(g & set(ranked[:n]))
    r = rank_of(ranked, gold, n)
    return {"rank": r, "recall": found / len(g) if g else 0.0,
            "any": 1 if r else 0, "hit1": 1 if r == 1 else 0,
            "rr": 1.0 / r if r else 0.0}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "recall_at_n": round(sum(r["recall"] for r in rows) / n, 4),
        "any_at_n": sum(r["any"] for r in rows),
        "hit_at_1": sum(r["hit1"] for r in rows),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
    }


def tie_or_beat(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[bool]:
    """a >= b on Recall@N, Hit@1 and MRR (None when either side is empty)."""
    if not a.get("n") or not b.get("n"):
        return None
    return (a["recall_at_n"] >= b["recall_at_n"] and a["hit_at_1"] >= b["hit_at_1"]
            and a["mrr"] >= b["mrr"])


# ── the gate ──────────────────────────────────────────────────────────────────
def compute_gate(per_query: List[Dict[str, Any]], by_group: Dict[str, Any],
                 have_fork: bool) -> Dict[str, Any]:
    """The verdict (see AB_EXACT.md). Every comparison is over ONE set of rows
    for both sides.

    * switch-off: on must tie or beat off on the paraphrase group AND on every
      category (a name-plus-topic category sits in the exact group, which is
      otherwise only compared with the fork: the old build regressed there).
      Both modes run on every question, so the rows are the same by
      construction.
    * the fork: on must tie or beat it on exact + date together, exact alone,
      date alone, and each exact/date category, over the questions that HAVE
      fork ranks. Totalling "on" over every question and the fork over only
      the ranked ones would compare different numbers of questions (and raw
      Hit@1 / any@N counts with them).

    ELIGIBLE needs the paraphrase group and exact + date to tie or beat (a
    group or category with no ranked questions is skipped, but exact + date
    must have some), and every exact/date question must have fork ranks. A
    single False anywhere is STAYS OFF."""
    para_ok = tie_or_beat(by_group.get("paraphrase", {}).get("on", {"n": 0}),
                          by_group.get("paraphrase", {}).get("off", {"n": 0}))

    def vs_fork(rows):
        paired = [r for r in rows if "fork" in r]
        if not have_fork or not paired:
            return None, len(rows), len(paired)
        return (tie_or_beat(aggregate([r["on"] for r in paired]),
                            aggregate([r["fork"] for r in paired])),
                len(rows), len(paired))

    def in_groups(*groups):
        return [r for r in per_query if r["group"] in groups]

    exact_ok, ed_rows, ed_paired = vs_fork(in_groups("exact", "date"))
    exact_only_ok, _, _ = vs_fork(in_groups("exact"))
    date_ok, _, _ = vs_fork(in_groups("date"))

    cats_off: Dict[str, Optional[bool]] = {}
    cats_fork: Dict[str, Optional[bool]] = {}
    for c in sorted({r["category"] for r in per_query}):
        rows = [r for r in per_query if r["category"] == c]
        cats_off[c] = tie_or_beat(aggregate([r["on"] for r in rows]),
                                  aggregate([r["off"] for r in rows]))
        if rows[0]["group"] in ("exact", "date"):
            cats_fork[c] = vs_fork(rows)[0]

    unpaired = ed_rows - ed_paired
    checks = [para_ok, exact_ok, exact_only_ok, date_ok,
              *cats_off.values(), *cats_fork.values()]
    if any(c is False for c in checks):
        verdict = "STAYS OFF"
    elif para_ok and exact_ok and unpaired == 0:
        verdict = "ELIGIBLE"
    elif para_ok and exact_ok:
        verdict = ("NOT EVALUATED (%d of %d exact/date questions have no fork "
                   "ranks)" % (unpaired, ed_rows))
    else:
        verdict = "NOT EVALUATED (a half of the gate had no data)"
    return {
        "paraphrase_on_ties_or_beats_off": para_ok,
        "exact_and_date_on_ties_or_beats_fork": exact_ok,
        "exact_only_on_ties_or_beats_fork": exact_only_ok,
        "date_only_on_ties_or_beats_fork": date_ok,
        "categories_on_ties_or_beats_off": cats_off,
        "categories_on_ties_or_beats_fork": cats_fork,
        "exact_date_questions": ed_rows,
        "exact_date_questions_with_fork_ranks": ed_paired,
        "verdict": verdict,
    }


def _worse(on_rank: Optional[int], off_rank: Optional[int]) -> bool:
    """Is rank ``on_rank`` worse than ``off_rank``? None (not in the top N) is
    worse than any rank."""
    if off_rank is None:
        return False
    return on_rank is None or on_rank > off_rank


def unseen_gold_summary(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Date questions whose gold the lane CANNOT see (no day-level event_time,
    no trusted recorded_at; a created_at-keyed gold is the usual case). The
    lane can still boost and admit OTHER nodes for such a question, so it can
    push the unseen gold down: count it. The lane has no way to help these."""
    rows = [r for r in per_query if r.get("lane_can_see_gold_date") is False]
    displaced = [r for r in rows if _worse(r["on"]["rank"], r["off"]["rank"])]
    return {
        "questions": len(rows),
        "displaced": len(displaced),
        "dropped_out_of_top_n": sum(
            1 for r in displaced if r["on"]["rank"] is None),
        "improved": sum(1 for r in rows if _worse(r["off"]["rank"], r["on"]["rank"])),
        "unchanged": sum(1 for r in rows if r["on"]["rank"] == r["off"]["rank"]),
        "hit_at_1_off": sum(r["off"]["hit1"] for r in rows),
        "hit_at_1_on": sum(r["on"]["hit1"] for r in rows),
        "displaced_questions": [
            {"text": r["text"], "rank_off": r["off"]["rank"],
             "rank_on": r["on"]["rank"]} for r in displaced],
    }


# ── database copy ─────────────────────────────────────────────────────────────
def copy_database(src: str, dst: str) -> None:
    """Consistent copy via the SQLite backup API; source opened read-only."""
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    try:
        s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    except sqlite3.Error:
        s = sqlite3.connect(src)
    try:
        d = sqlite3.connect(dst)
        try:
            s.backup(d)
        finally:
            d.close()
    finally:
        s.close()


# ── can the lane see the gold node's date at all? ─────────────────────────────
TRUSTED_SAID = ("content", "capture", "import")


def date_visibility(store, gold: List[str]) -> List[Dict[str, Any]]:
    """For each gold node: does it carry a day-level event_time, or a
    recorded_at from a trusted source (content/capture/import)? The lane
    matches dates on nothing else (never created_at). `visible` False means
    the lane CANNOT see this memory by date, which is not the same as the lane
    having missed it."""
    out = []
    conn = store._get_conn()
    for nid in gold:
        row = conn.execute(
            "SELECT event_time_start, event_time_granularity, recorded_at, "
            "json_extract(metadata, '$.recorded_at_source') FROM nodes "
            "WHERE node_id = ?", (nid,)).fetchone()
        if row is None:
            out.append({"node_id": nid, "found": False, "visible": False})
            continue
        ev, gran, rec, src = row
        day_level = bool(ev) and gran == "day"
        said = bool(rec) and src in TRUSTED_SAID
        out.append({
            "node_id": nid, "found": True,
            "event_time_granularity": gran if ev else None,
            "event_time_day_level": day_level,
            "recorded_at_present": bool(rec), "recorded_at_source": src,
            "recorded_at_usable": said,
            "coarse_event_time_only": bool(ev) and not day_level and not said,
            "visible": day_level or said,
        })
    return out


# ── the run ───────────────────────────────────────────────────────────────────
def run_ab(
    db_path: str,
    queries: List[Dict[str, Any]],
    *,
    top_n: int = 10,
    runs: int = RUNS,
    fork: Optional[Dict[str, List[str]]] = None,
    allow_graph_only: bool = False,
    semantic_factory: Optional[Callable[[Any], Any]] = None,
    workdir: Optional[str] = None,
) -> Dict[str, Any]:
    from revien.graph.store import GraphStore
    from revien.retrieval.engine import RetrievalEngine
    from revien.retrieval.exact import parse_signals
    from revien.semantic.index import SemanticIndex
    from revien.semantic.rerank import CrossEncoderReranker

    tmp = tempfile.mkdtemp(prefix="ab_exact_", dir=workdir)
    copy = os.path.join(tmp, "revien_copy.db")
    try:
        src_stat = os.stat(db_path)
        copy_database(db_path, copy)
        store = GraphStore(db_path=copy)
        try:
            semantic = (semantic_factory(store) if semantic_factory
                        else SemanticIndex(store))
            reranker = CrossEncoderReranker()
            sem_active = bool(semantic.is_enabled)
            rerank_active = bool(reranker.is_enabled)
            if not sem_active and not allow_graph_only:
                raise SystemExit(
                    "semantic layer is NOT active ("
                    f"{semantic.inactive_reason()}); this A/B is only "
                    "meaningful with REVIEN_SEMANTIC=1 working. Fix that, or "
                    "pass --allow-graph-only to run anyway.")
            engines = {
                "off": RetrievalEngine(store, semantic=semantic,
                                       reranker=reranker, exact=False),
                "on": RetrievalEngine(store, semantic=semantic,
                                      reranker=reranker, exact=True),
            }
            nodes = store._get_conn().execute(
                "SELECT COUNT(*) FROM nodes").fetchone()[0]
            per_query = []
            for q in queries:
                row = {"text": q["text"], "category": q["category"],
                       "group": group_of(q["category"]), "gold": q["gold"]}
                for mode, eng in engines.items():
                    lat, first, stable = [], None, True
                    diag = None
                    for i in range(runs):
                        resp = eng.recall(q["text"], top_n=top_n, debug=(i == 0))
                        ranked = [r.node_id for r in resp.results]
                        lat.append(resp.retrieval_time_ms)
                        if i == 0:
                            first = ranked
                            diag = (resp.diagnostics or {}).get("exact")
                            row[f"boosts_{mode}"] = {
                                r.node_id: r.score_breakdown["exact_boost"]
                                for r in resp.results
                                if "exact_boost" in r.score_breakdown}
                        elif ranked != first:
                            stable = False
                    s = score_one(first, q["gold"], top_n)
                    row[mode] = s
                    row[f"latency_ms_{mode}"] = round(statistics.median(lat), 2)
                    if not stable:
                        row[f"unstable_{mode}"] = True
                    if mode == "on" and diag is not None:
                        row["exact_detected"] = {
                            k: v for k, v in diag["detected"].items()
                            if v and k != "entity_anchors"}
                        row["exact_injected"] = diag["injected"]
                if mode == "on" and parse_signals(q["text"]).dates:
                    vis = date_visibility(store, q["gold"])
                    row["date_visibility"] = vis
                    row["lane_can_see_gold_date"] = any(v["visible"] for v in vis)
                if fork is not None and q["text"] in fork:
                    row["fork"] = score_one(fork[q["text"]], q["gold"], top_n)
                per_query.append(row)
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Source untouched?
    after = os.stat(db_path)
    untouched = (after.st_mtime_ns == src_stat.st_mtime_ns
                 and after.st_size == src_stat.st_size)

    cats = sorted({r["category"] for r in per_query})
    groups = sorted({r["group"] for r in per_query})
    modes = ["off", "on"] + (["fork"] if fork else [])

    def agg(rows, mode):
        return aggregate([r[mode] for r in rows if mode in r])

    by_category = {c: {m: agg([r for r in per_query if r["category"] == c], m)
                       for m in modes} for c in cats}
    by_group = {g: {m: agg([r for r in per_query if r["group"] == g], m)
                    for m in modes} for g in groups}
    overall = {m: agg(per_query, m) for m in modes}

    gate = compute_gate(per_query, by_group, bool(fork))

    return {
        "settings": {
            "db": db_path, "nodes": nodes, "top_n": top_n, "runs": runs,
            "semantic_active": sem_active,
            "rerank_active": rerank_active,
            "semantic_note": semantic.inactive_reason(),
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith("REVIEN_")},
            "exact_config": {k: v for k, v in os.environ.items()
                             if k.startswith("REVIEN_EXACT")},
            "source_untouched": untouched,
        },
        "overall": overall, "by_group": by_group, "by_category": by_category,
        "latency_ms_median": {
            m: round(statistics.median(
                [r[f"latency_ms_{m}"] for r in per_query]), 2)
            for m in ("off", "on")} if per_query else {},
        "gate": gate, "date_unseen_gold": unseen_gold_summary(per_query),
        "queries": per_query,
    }


# ── rendering ─────────────────────────────────────────────────────────────────
def _fmt(a: Dict[str, Any]) -> str:
    if not a.get("n"):
        return "n=0"
    return (f"n={a['n']:<3} R@N={a['recall_at_n']:.3f} any={a['any_at_n']}/{a['n']} "
            f"hit@1={a['hit_at_1']} mrr={a['mrr']:.3f}")


def render(rep: Dict[str, Any]) -> str:
    s = rep["settings"]
    L = [f"exact-lane A/B  db={s['db']} ({s['nodes']} nodes, copy)  top_n={s['top_n']} "
         f"runs={s['runs']}  semantic_active={s['semantic_active']}  "
         f"rerank_active={s['rerank_active']}",
         f"settings: {json.dumps(s['env'])}",
         f"source database untouched: {s['source_untouched']}", ""]
    L.append("overall")
    for m, a in rep["overall"].items():
        L.append(f"  {m:<5} {_fmt(a)}")
    L.append("by group")
    for g, ms in rep["by_group"].items():
        L.append(f" {g}")
        for m, a in ms.items():
            L.append(f"  {m:<5} {_fmt(a)}")
    L.append("by category")
    for c, ms in rep["by_category"].items():
        L.append(f" {c}")
        for m, a in ms.items():
            L.append(f"  {m:<5} {_fmt(a)}")
    L.append(f"median latency ms: {rep['latency_ms_median']}")
    L.append("")
    L.append("per-query ranks (off -> on; '-' = not in top N; * = changed)")
    for r in rep["queries"]:
        a, b = r["off"]["rank"], r["on"]["rank"]
        fk = f"  fork={r['fork']['rank'] or '-'}" if "fork" in r else ""
        L.append(f"  [{r['group'][:5]:<5}] {a or '-'} -> {b or '-'}"
                 f"{'*' if a != b else ' '}{fk}  {r['text'][:70]}")
        if "date_visibility" in r:
            for v in r["date_visibility"]:
                if not v["found"]:
                    why = "gold node not in this database"
                elif v["visible"]:
                    why = ("day-level event_time" if v["event_time_day_level"]
                           else f"recorded_at ({v['recorded_at_source']})")
                else:
                    why = ("CANNOT SEE: " + ("event_time is " + str(v["event_time_granularity"])
                           + "-level only; " if v["coarse_event_time_only"] else
                           "no event_time; ")
                           + ("recorded_at source is "
                              f"{v['recorded_at_source']!r}, not trusted"
                              if v["recorded_at_present"] else "no recorded_at"))
                L.append(f"          gold date visibility: {why}")
    u = rep["date_unseen_gold"]
    L += ["", "date questions whose gold the lane cannot see "
          "(no day-level event_time, no trusted recorded_at)",
          f"  {u['questions']} questions; the lane pushed the gold DOWN on "
          f"{u['displaced']} ({u['dropped_out_of_top_n']} out of the top N), "
          f"up on {u['improved']}, unchanged on {u['unchanged']}; "
          f"Hit@1 {u['hit_at_1_off']} -> {u['hit_at_1_on']}"]
    for d in u["displaced_questions"]:
        L.append(f"    displaced: {d['rank_off'] or '-'} -> {d['rank_on'] or '-'}"
                 f"  {d['text'][:70]}")
    g = rep["gate"]
    L += ["", "gate", f"  paraphrase: on ties-or-beats off: {g['paraphrase_on_ties_or_beats_off']}",
          "  every category, on ties-or-beats off: " + ", ".join(
              f"{c} {v}" for c, v in g["categories_on_ties_or_beats_off"].items()),
          f"  exact+date: on ties-or-beats fork: {g['exact_and_date_on_ties_or_beats_fork']}"
          f" (over the {g['exact_date_questions_with_fork_ranks']} of "
          f"{g['exact_date_questions']} questions that have fork ranks)",
          f"  exact only vs fork: {g['exact_only_on_ties_or_beats_fork']}",
          f"  date only vs fork: {g['date_only_on_ties_or_beats_fork']}",
          "  each exact/date category vs fork: " + ", ".join(
              f"{c} {v}" for c, v in g["categories_on_ties_or_beats_fork"].items()),
          f"  verdict: {g['verdict']}"]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--floor", type=float, default=None,
                    help="REVIEN_SEMANTIC_SIM_FLOOR for both modes")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--fork-ranks", default=None)
    ap.add_argument("--rerank", choices=("on", "off"), default=None,
                    help="set REVIEN_RERANK for both modes (default: whatever "
                         "the environment says; the cross-encoder is on unless "
                         "REVIEN_RERANK=0)")
    ap.add_argument("--allow-graph-only", action="store_true")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    args = ap.parse_args(argv)

    os.environ["REVIEN_SEMANTIC"] = "1"
    os.environ.setdefault("REVIEN_TOUCH_ON_RECALL", "0")
    if args.floor is not None:
        os.environ["REVIEN_SEMANTIC_SIM_FLOOR"] = repr(args.floor)
    if args.rerank is not None:
        os.environ["REVIEN_RERANK"] = "1" if args.rerank == "on" else "0"
    # The two modes are chosen by constructor argument, never by env: a
    # stray REVIEN_EXACT must not leak into the baseline engine.
    os.environ.pop("REVIEN_EXACT", None)

    queries = load_queries(args.queries)
    fork = load_fork(args.fork_ranks) if args.fork_ranks else None
    rep = run_ab(args.db, queries, top_n=args.top_n, runs=args.runs, fork=fork,
                 allow_graph_only=args.allow_graph_only)
    print(render(rep))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
