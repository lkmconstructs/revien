"""
revien_bench.ab_exact - offline A/B of the exact lane (REVIEN_EXACT) against
the same engine with the lane off, on YOUR graph and YOUR questions.

    python -m revien_bench.ab_exact --db /path/to/revien.db \\
        --queries /path/to/questions.json [--floor 0.458] [--top-n 10] \\
        [--fork-ranks fork.json] [--rerank on|off] [--out report.json]

What it does
  * Works on a COPY of the database in a temp dir (copy_database). The
    original is never written. Opening the copy runs the normal connect-time
    migrations (including 007's indexes) on the copy only. The report says
    whether the original's size and mtime stayed the same and no -wal/-shm
    appeared beside it during the run.
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
  * Refuses (exit 2, no numbers, no verdict) when the semantic layer is not
    active, before the first recall OR at any point during the run (the
    embedder can fail at first use and leave every recall graph-only), unless
    --allow-graph-only is given: an A/B without the semantic layer is not the
    target configuration. With --allow-graph-only a run that is graph-only
    throughout is reported as such; a run that is partly semantic and partly
    graph-only is refused all the same. Likewise for the cross-encoder: with
    --rerank on it must be active on every recall, and a reranker that goes
    inactive part way through a run is refused whatever was asked.
    Each response's own semantic_active is read, and both layers are checked
    again after the last recall.

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
    """Date questions whose gold the lane CANNOT see (its own date rule,
    ExactLane.match_dates, matches no gold node: no event_time and no trusted
    recorded_at on the question's date; a created_at-keyed gold is the usual
    case). The lane can still boost and admit OTHER nodes for such a
    question, so it can push the unseen gold down: count it. The lane has no
    way to help these."""
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


# ── refusal ───────────────────────────────────────────────────────────────────
class ABRefused(SystemExit):
    """This run gives no numbers and no verdict. The command line prints the
    message and exits 2 (``code``); str() is the message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = 2


class _Layer:
    """One optional layer (semantic index, cross-encoder) over a whole run.

    ``check_start`` refuses before any recall when the layer must be active
    and is not. ``observe`` is called once per recall (and once after the
    last one) with whether the layer was active. It refuses the moment the
    layer is inactive when it must be active, or changes state between
    recalls (a run that is partly with and partly without it compares
    nothing). Enabled at the start and inactive from the first recall on (a
    model that fails at first use) is not a change: no recall had it."""

    def __init__(self, name: str, active_at_start: bool, must_be_active: bool,
                 reason: Callable[[], Optional[str]], fix: str):
        self.name, self.at_start, self.must = name, active_at_start, must_be_active
        self._reason, self._fix = reason, fix
        self.first: Optional[bool] = None
        self.inactive = 0           # observations with the layer inactive
        self.inactive_recalls = 0   # recalls that ran without it

    @property
    def active(self) -> bool:
        """Active on every observation (and at the start)."""
        return self.at_start and self.inactive == 0

    def check_start(self) -> None:
        if not self.at_start:
            self.inactive += 1
            self._refuse_if_required(False, "before the first recall")

    def _refuse_if_required(self, active: bool, where: str) -> None:
        if self.must and not active:
            raise ABRefused(
                f"REFUSED: the {self.name} is not active ({where}): "
                f"{self._reason() or 'inactive'}. Part or all of this A/B "
                f"would run without it, so it is not the configuration being "
                f"compared, and no numbers or verdict are given. {self._fix}")

    def observe(self, active: bool, where: str, recall: bool = False) -> None:
        if not active:
            self.inactive += 1
            self.inactive_recalls += 1 if recall else 0
        self._refuse_if_required(active, where)
        if self.first is None:
            self.first = active
        elif active != self.first:
            raise ABRefused(
                f"REFUSED: the {self.name} went "
                f"{'inactive' if self.first else 'active'} part way through "
                f"the run ({where}): {self._reason() or 'active again'}. Some "
                f"recalls ran with it and some without, so the two modes are "
                f"not comparable, and no numbers or verdict are given. "
                f"{self._fix}")


# ── database copy ─────────────────────────────────────────────────────────────
_SIDECARS = ("-wal", "-shm")


def _sidecar_state(path: str) -> Tuple[bool, ...]:
    real = os.path.realpath(path)
    return tuple(os.path.exists(real + s) for s in _SIDECARS)


def _source_state(path: str) -> Tuple[Any, ...]:
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size) + _sidecar_state(path)


def _is_wal_file(path: str) -> bool:
    """The header says WAL (file format bytes 18 and 19 are 2)."""
    with open(path, "rb") as f:
        head = f.read(20)
    return (len(head) == 20 and head[:16] == b"SQLite format 3\x00"
            and head[18] == 2 and head[19] == 2)


def copy_database(src: str, dst: str) -> str:
    """Copy ``src`` to ``dst`` without writing to ``src`` or leaving files
    beside it. Returns the method used: "file" or "sqlite_backup".

    A WAL database that no connection has open (no -wal, no -shm) holds all
    of its content in the main file, so the file is copied as bytes: opening
    it with SQLite, even read-only, would create a -wal and a -shm beside it
    that a read-only connection cannot remove. The copy is kept only if,
    after it, the file's mtime and size are unchanged and still no -wal/-shm
    exists (nothing opened it meanwhile); otherwise, and for a database that
    is open (a live daemon) or not in WAL mode, the SQLite backup API makes
    the copy through a read-only connection, which reads the -wal a live
    writer keeps and adds no file of its own when one is already there.
    ``immutable=1`` is never used: it is unsafe on a live database."""
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if _is_wal_file(src) and not any(_sidecar_state(src)):
        before = _source_state(src)
        shutil.copyfile(src, dst)
        if _source_state(src) == before:
            return "file"
        os.remove(dst)
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
    return "sqlite_backup"


# ── can the lane see the gold node's date at all? ─────────────────────────────
TRUSTED_SAID = ("content", "capture", "import")


def date_visibility(lane, query: str,
                    gold: List[str]) -> Optional[List[Dict[str, Any]]]:
    """For each gold node, did the lane's own date rule match it? None when
    the question holds no date (as the lane reads it).

    This calls ExactLane.parse and ExactLane.match_dates, the functions the
    lane boosts from, with the scope the A/B's recalls use (live nodes, no
    context nodes, every origin): event_time at whatever granularity the node
    has (day, week, month, year, fuzzy), then trusted recorded_at at half
    weight, never created_at. ``visible`` False means the lane CANNOT see
    this memory by date, which is not the same as the lane having missed it.
    ``matched`` lists each match (date, field, granularity, strength);
    ``matched_field`` / ``matched_granularity`` are the strongest one."""
    sig = lane.parse(query)
    if not sig.dates:
        return None
    hits: Dict[str, List[Dict[str, Any]]] = {}
    for dm in lane.match_dates(sig.dates):
        for nid, gran, strength in dm.event:
            hits.setdefault(nid, []).append(
                {"date": dm.signal.text, "field": "event_time",
                 "granularity": gran, "strength": strength})
        for nid, strength in dm.recorded:
            hits.setdefault(nid, []).append(
                {"date": dm.signal.text, "field": "recorded_at",
                 "granularity": dm.signal.granularity, "strength": strength})
    out = []
    conn = lane.store._get_conn()
    for nid in gold:
        row = conn.execute(
            "SELECT event_time_start, event_time_end, event_time_granularity, "
            "recorded_at, json_extract(metadata, '$.recorded_at_source'), "
            "node_type, invalidated_at FROM nodes WHERE node_id = ?",
            (nid,)).fetchone()
        if row is None:
            out.append({"node_id": nid, "found": False, "visible": False,
                        "matched": []})
            continue
        ev_s, ev_e, gran, rec, src, ntype, inval = row
        mine = hits.get(nid, [])
        best = max(mine, key=lambda h: h["strength"]) if mine else None
        out.append({
            "node_id": nid, "found": True, "visible": bool(mine),
            "matched": mine,
            "matched_field": best["field"] if best else None,
            "matched_granularity": best["granularity"] if best else None,
            "event_time_granularity": gran if ev_s else None,
            "event_time": [ev_s, ev_e or ev_s] if ev_s else None,
            "recorded_at_present": bool(rec), "recorded_at": rec,
            "recorded_at_source": src,
            "recorded_at_trusted": src in TRUSTED_SAID,
            "in_recall_scope": inval is None and ntype != "context",
        })
    return out


def _cannot_see_why(v: Dict[str, Any]) -> str:
    """Why the lane's date rule did not reach this gold node, field by field."""
    why = []
    if not v["in_recall_scope"]:
        why.append("outside recall's scope (invalidated, or a context node)")
    if v["event_time"] is None:
        why.append("no event_time")
    else:
        lo, hi = (str(x)[:10] for x in v["event_time"])
        why.append(f"event_time is {v['event_time_granularity']}-level "
                   f"({lo}..{hi}), not on the question's date")
    if not v["recorded_at_present"]:
        why.append("no recorded_at")
    elif not v["recorded_at_trusted"]:
        why.append(f"recorded_at source is {v['recorded_at_source']!r}, "
                   "not trusted")
    else:
        why.append(f"recorded_at {str(v['recorded_at'])[:10]} "
                   f"({v['recorded_at_source']}) not on the question's date")
    return "; ".join(why)


# ── the run ───────────────────────────────────────────────────────────────────
def run_ab(
    db_path: str,
    queries: List[Dict[str, Any]],
    *,
    top_n: int = 10,
    runs: int = RUNS,
    fork: Optional[Dict[str, List[str]]] = None,
    allow_graph_only: bool = False,
    require_rerank: bool = False,
    semantic_factory: Optional[Callable[[Any], Any]] = None,
    reranker_factory: Optional[Callable[[], Any]] = None,
    workdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the A/B. Raises ABRefused (exit status 2 from the command line)
    when the semantic layer is not active on every recall (unless
    ``allow_graph_only``, which still refuses a run that is only partly
    graph-only), when ``require_rerank`` and the cross-encoder is not active
    on every recall, or when the cross-encoder goes inactive part way."""
    from revien.graph.store import GraphStore
    from revien.retrieval.engine import RetrievalEngine
    from revien.semantic.index import SemanticIndex
    from revien.semantic.rerank import CrossEncoderReranker

    tmp = tempfile.mkdtemp(prefix="ab_exact_", dir=workdir)
    copy = os.path.join(tmp, "revien_copy.db")
    try:
        src_before = _source_state(db_path)
        copy_method = copy_database(db_path, copy)
        store = GraphStore(db_path=copy)
        try:
            semantic = (semantic_factory(store) if semantic_factory
                        else SemanticIndex(store))
            reranker = (reranker_factory() if reranker_factory
                        else CrossEncoderReranker())
            sem = _Layer(
                "semantic layer", bool(semantic.is_enabled),
                not allow_graph_only, semantic.inactive_reason,
                "Fix the semantic layer (REVIEN_SEMANTIC=1 with a working "
                "embedder; for the local one, the model must be in "
                "FASTEMBED_CACHE_PATH), or pass --allow-graph-only to run "
                "graph-only on purpose.")
            rr = _Layer(
                "cross-encoder reranker", bool(reranker.is_enabled),
                require_rerank, reranker.inactive_reason,
                "Make the reranker model available, or run with --rerank off.")
            total = len(queries) * 2 * runs
            sem.check_start()
            rr.check_start()
            engines = {
                "off": RetrievalEngine(store, semantic=semantic,
                                       reranker=reranker, exact=False),
                "on": RetrievalEngine(store, semantic=semantic,
                                      reranker=reranker, exact=True),
            }
            nodes = store._get_conn().execute(
                "SELECT COUNT(*) FROM nodes").fetchone()[0]
            per_query = []
            done = 0
            for qi, q in enumerate(queries):
                row = {"text": q["text"], "category": q["category"],
                       "group": group_of(q["category"]), "gold": q["gold"]}
                for mode, eng in engines.items():
                    lat, first, stable = [], None, True
                    diag = None
                    for i in range(runs):
                        resp = eng.recall(q["text"], top_n=top_n, debug=(i == 0))
                        done += 1
                        where = (f"recall {done} of {total}: mode {mode}, run "
                                 f"{i + 1}, question {qi + 1} "
                                 f"{q['text'][:60]!r}")
                        # The response's own flag: set after the recall ran,
                        # so a layer that failed inside it reads False.
                        sem.observe(bool(resp.semantic_active), where, True)
                        rr.observe(bool(reranker.is_enabled), where, True)
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
                vis = date_visibility(engines["on"].exact_lane, q["text"],
                                      q["gold"])
                if vis is not None:
                    row["date_visibility"] = vis
                    row["lane_can_see_gold_date"] = any(v["visible"] for v in vis)
                if fork is not None and q["text"] in fork:
                    row["fork"] = score_one(fork[q["text"]], q["gold"], top_n)
                per_query.append(row)
            # And once more after the last recall.
            sem.observe(bool(semantic.is_enabled), "after the last recall")
            rr.observe(bool(reranker.is_enabled), "after the last recall")
            sem_note = semantic.inactive_reason()
            rr_note = reranker.inactive_reason()
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Source untouched: same mtime and size, and no -wal/-shm appeared.
    src_after = _source_state(db_path)
    untouched = src_after == src_before

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
            "semantic_active": sem.active,
            "semantic_active_at_start": sem.at_start,
            "semantic_inactive_recalls": sem.inactive_recalls,
            "semantic_note": sem_note,
            "rerank_active": rr.active,
            "rerank_active_at_start": rr.at_start,
            "rerank_note": rr_note,
            "rerank_inactive_recalls": rr.inactive_recalls,
            "copy_method": copy_method,
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
         f"source database untouched: {s['source_untouched']}"]
    if not s["semantic_active"]:
        L.append(f"GRAPH-ONLY run (--allow-graph-only): the semantic layer was "
                 f"inactive throughout ({s['semantic_note']}). These numbers "
                 f"are not the target configuration.")
    if s["rerank_active_at_start"] and not s["rerank_active"]:
        L.append(f"cross-encoder reranker inactive throughout ({s['rerank_note']}).")
    L.append("")
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
                    why = (f"seen by date on {v['matched_field']}"
                           + (f" ({v['recorded_at_source']}, half weight)"
                              if v["matched_field"] == "recorded_at" else "")
                           + f" at {v['matched_granularity']} granularity")
                else:
                    why = "CANNOT SEE: " + _cannot_see_why(v)
                L.append(f"          gold date visibility: {why}")
    u = rep["date_unseen_gold"]
    L += ["", "date questions whose gold the lane cannot see "
          "(no event_time and no trusted recorded_at on the question's date)",
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
          f"  verdict: {g['verdict']}"
          + ("  [graph-only run: not the target configuration]"
             if not s["semantic_active"] else "")]
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
                         "REVIEN_RERANK=0). With 'on' the run is refused "
                         "unless the cross-encoder is active on every recall")
    ap.add_argument("--allow-graph-only", action="store_true",
                    help="run even though the semantic layer is inactive "
                         "throughout (still refused if it is active for only "
                         "part of the run)")
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
    try:
        rep = run_ab(args.db, queries, top_n=args.top_n, runs=args.runs,
                     fork=fork, allow_graph_only=args.allow_graph_only,
                     require_rerank=(args.rerank == "on"))
    except ABRefused as refused:
        print(str(refused), file=sys.stderr)
        return refused.code
    print(render(rep))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
