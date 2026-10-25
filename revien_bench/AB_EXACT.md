# Exact-lane A/B (`revien_bench/ab_exact.py`)

Offline comparison of recall with `REVIEN_EXACT` off and on, over your own
graph and your own questions. It needs no network and no model beyond the ones
your recall already uses.

## Run

```
python -m revien_bench.ab_exact \
    --db /path/to/snapshot/revien.db \
    --queries /path/to/questions.json \
    --floor 0.458 \
    --rerank on \
    --fork-ranks /path/to/fork-ranks.json \
    --out ab-exact-report.json
```

- `--db` is never written. It is copied into a temp directory and every
  recall runs on the copy; opening the copy applies the normal migrations
  (007's three indexes) to the copy only. The report says whether the
  original's size and mtime are unchanged and no `-wal` / `-shm` file
  appeared beside it during the run (`source_untouched`), and how it was
  copied (`copy_method`):
  - `file`: a WAL database that nothing has open (no `-wal`, no `-shm`
    beside it) keeps all its content in the main file, which is copied as
    bytes. Opening it with SQLite, even read-only, would leave an empty
    `-wal` and a 32 KB `-shm` beside it that a read-only connection cannot
    remove. The copy is kept only if the file's mtime and size did not change
    and no `-wal` / `-shm` appeared while it was copied.
  - `sqlite_backup`: otherwise (a live daemon has the database open, or it
    is not in WAL mode, or something opened it during the file copy), the
    SQLite backup API copies it through a read-only connection, which reads
    the live writer's `-wal` and adds no file of its own where one is already
    there. `immutable=1` is never used: it is unsafe on a live database.
  The copy is consistent either way, but a database a daemon is writing to
  is a moving target: run the A/B on a snapshot (stop the daemon, or copy the
  database with `sqlite3 revien.db ".backup snapshot.db"`) so both modes and
  any comparison system score the same graph.
- The script sets `REVIEN_SEMANTIC=1`, and `REVIEN_SEMANTIC_SIM_FLOOR` when
  `--floor` is given (omit it to keep whatever is in your environment). It
  prints every `REVIEN_*` variable it ran with. `--rerank on|off` sets
  `REVIEN_RERANK` for both modes (the cross-encoder is on by default; the
  report says `rerank_active`).
- It refuses (exit status 2, a `REFUSED: ...` line on stderr, no numbers and
  no verdict) when the semantic layer is not active, checked before the first
  recall, on every recall (each response's own `semantic_active`) and again
  after the last one. An embedding model that is missing fails at first use,
  not at start-up, so a run can begin "active" and recall graph-only
  throughout: that is refused too. `--allow-graph-only` runs anyway when the
  layer is inactive for the whole run (the report then says
  `semantic_active=False`, `GRAPH-ONLY run`, and marks the verdict as not the
  target configuration); a run where it is active for some recalls and not
  others is refused even with it. With `--rerank on` the cross-encoder must
  be active on every recall or the run is refused; a cross-encoder that goes
  inactive part way through a run is refused whatever was asked, and one
  that fails at its first use when `--rerank on` was not given is reported
  as `rerank_active=False` with the reason.
- The two modes are two engines over the same store, the same semantic index
  and the same reranker, built with `exact=False` and `exact=True`. Nothing
  differs but the lane. A stray `REVIEN_EXACT` in your shell is ignored.
- Each query runs 3 times per mode (`--runs`). Ranks come from run 1; latency
  is the median over the runs; a mode whose ranking changes between runs is
  flagged `unstable_off` / `unstable_on` in the JSON.
- Query file: a JSON list (or `{"queries": [...]}`) of objects with the question
  (`text`, or `query` / `question` / `q`), the gold node ids (`gold`, or
  `gold_ids` / `gold_node_ids` / `gold_nodes` / `expected` / `answer_ids`; a
  string or a list) and the category (`category`, or `cat` / `label` / `type`
  / `kind`). A query missing text or gold stops the run with its index. Check
  the field names of your question file against this list before the first
  run.
- Categories are grouped by name: a category containing "paraphras" is the
  **paraphrase** group; one containing "date", "time" or "day" is **date**; one
  containing "exact", "name", "title", "baseline", "id", "code" or "quote" is
  **exact**; anything else is **other** and is reported but not gated.
- `--fork-ranks` is optional: the ranked node ids of a comparison system (the
  "fork" in the flag and report names) per question, either
  `{"<question text>": ["<id>", ...]}` or a list of `{"query"|"text": ...,
  "results"|"ranked"|"fork": [id | {"node_id": id}, ...]}`. Convert your own
  results file to one of those shapes if it differs.

## What it reports

Per category, per group and overall, for off, on (and the fork when given):

- **Recall@N** (N = `--top-n`, default 10): mean fraction of a question's gold
  ids found in the top N. With one gold id per question this is the hit rate.
- **any@N**: questions with at least one gold id in the top N (the "23/30"
  style count).
- **Hit@1**: the top result is a gold id.
- **MRR**: mean of 1/rank of the first gold id (0 when it is not in the top N).
- For every question that contains a date (as the lane reads it): whether
  the lane's own date rule matches the gold node. The report calls the same
  function the lane boosts from (`ExactLane.match_dates`), with the scope the
  recalls use, so the two cannot disagree: `event_time` at whatever
  granularity the node has (a month-level gold is seen by "April 2026" and
  by "on April 14, 2026", at month weight 0.4; week 0.6, year 0.15, fuzzy
  0.1), then `recorded_at` from a trusted source (`content`, `capture`,
  `import`) at half weight, never `created_at`. A seen gold is printed with
  the field and the granularity it matched at (`matched_field`,
  `matched_granularity`, and every match in `matched`, in the JSON). "CANNOT
  SEE" means the lane had nothing to match, which is different from the lane
  missing, and says why field by field (no `event_time`, or one that is not
  on the question's date; no `recorded_at`, an untrusted source, or a date
  that does not match; outside recall's scope); `lane_can_see_gold_date` is
  in the JSON.
- **Date questions whose gold the lane cannot see** are summed up separately:
  how many, in how many of them the lane pushed that gold down (and how many
  fell out of the top N), in how many it moved up, Hit@1 before and after,
  and each displaced question with its two ranks. The lane cannot help these
  but can still boost and admit other nodes, so this is its cost on questions
  keyed on a date it has no field for (for instance a question keyed on
  `created_at`). It is reported, not gated.
- Per-query ranks, off then on (and the fork), with a mark where they differ,
  and for the "on" mode what the lane detected and injected.

## The gate

On the same snapshot, with the same floor, the lane stays off unless **all** of:

1. on **ties or beats** off on the paraphrase group AND on every category
   (Recall@N, Hit@1 and MRR each greater than or equal). A category such as
   `name_plus_topic` is grouped as exact and is otherwise compared only with
   the fork, so each category is checked against switch-off on its own: the
   case where a named entity hijacks the answer shows up there; and
2. on **ties or beats** the fork on the exact + date groups together, on the
   exact group alone, on the date group alone and on each exact/date category
   (the same three metrics each). A group or category with no fork-ranked
   questions is skipped; the exact + date check must have some. A name gain
   cannot pay for a date loss.

Every comparison with the fork is made **over the same questions on both
sides**: the exact/date questions that have fork ranks. "On" is not totalled
over questions the fork did not rank (that would compare Hit@1 and any@N raw
counts over different numbers of questions). If some exact/date questions have
no fork ranks the verdict cannot be `ELIGIBLE`; the report says how many.

The report prints each check and a verdict: `ELIGIBLE`, `STAYS OFF` (any check
is False), or `NOT EVALUATED` when a half had no data (for instance no
`--fork-ranks`, or no paraphrase questions) or some exact/date questions have
no fork ranks. `ELIGIBLE` means the numbers did not say no; it does not turn
anything on.

## Read the numbers with care

- A small question set is thin: at thirty questions one question is 3 points
  of Hit@1. Grow it (roughly 15 paraphrases, 10 exact names/titles, 10 dates
  in mixed formats), keep an early set as a regression floor, and check that
  the date questions have gold nodes with an `event_time` or a trusted
  `recorded_at` on that date at all: the report marks the ones the lane
  cannot see.
- A bare `3/4`-style date needs a marker to be read as one (`on 3/4`,
  `03/04`, `3/4/2026`; see `exact.py`): "the week of 3/10" or "from 3/4 to
  3/8" are read as ratios, not dates, and no date reads across a slash or a
  hyphen into another number ("May 5-10", "21 September 3/4" is September
  21). Write date questions the way your users would, and check the
  report's detected dates.
- The lane reads at most 4,096 characters of a question: a longer one is
  read at its first and last 2,048 (cut at whitespace), and its
  `exact_detected` says `truncated` with `scanned_chars` and `query_chars`.
  At most 16 codes per question are looked for inside labels, longest
  first; when more were asked, `exact_detected.ids_capped` lists the ones
  left out (`REVIEN_EXACT_MAX_CONTAINED_CODES` raises the cap).
- The reranker is on by default in production. It re-sorts the top 20
  candidates by its own score, on a scale of its own. With it on the lane
  works twice: before reranking its boost decides which nodes enter that head,
  and after reranking the boost is added again on the engine's scale (the
  head keeps the reranker's order, the head's own scores are dealt back out
  down that order, the boost is added, the head is re-sorted), capped at the
  same 0.10. Run the A/B with the same `REVIEN_RERANK` you serve with, and
  run it both ways: the numbers differ, and the report prints which one it
  used.
- A question that names an entity and asks about something else ("what did X
  write about Y") does not lift X's own node; it lifts the nodes connected to
  X. A question about the entity ("who is X") lifts X. See `exact.py` for the
  rules and their words.
- With `REVIEN_LEXICAL=bm25` the lane scales its label boosts by 0.5, since
  BM25 already counted the label's words. That combination has not been
  measured.
- Re-snapshot before each comparison. The lane and the fork must score the
  same graph.
