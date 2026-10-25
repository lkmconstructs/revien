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

- `--db` is opened read-only and copied (SQLite backup) into a temp directory;
  every recall runs on the copy and the report says whether the original's
  size and mtime are unchanged. Opening the copy applies the normal migrations
  (007's two indexes) to the copy only.
- The script sets `REVIEN_SEMANTIC=1`, and `REVIEN_SEMANTIC_SIM_FLOOR` when
  `--floor` is given (omit it to keep whatever is in your environment). It
  prints every `REVIEN_*` variable it ran with. `--rerank on|off` sets
  `REVIEN_RERANK` for both modes (the cross-encoder is on by default; the
  report says `rerank_active`). It refuses to run (exit 2) if the semantic
  layer is not active, unless you pass `--allow-graph-only`.
- The two modes are two engines over the same store, the same semantic index
  and the same reranker, built with `exact=False` and `exact=True`. Nothing
  differs but the lane. A stray `REVIEN_EXACT` in your shell is ignored.
- Each query runs 3 times per mode (`--runs`). Ranks come from run 1; latency
  is the median over the runs; a mode whose ranking changes between runs is
  flagged `unstable_off` / `unstable_on` in the JSON.
- Query file: a JSON list (or `{"queries": [...]}`) of objects with the question
  (`text`, or `query` / `question` / `q`), the gold node ids (`gold`, or
  `gold_ids` / `gold_node_ids` / `gold_nodes` / `expected`; a string or a list)
  and the category (`category`, or `cat` / `label` / `type` / `kind`). A query
  missing text or gold stops the run with its index. Check the field names of
  your question file against this list before the first run.
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
- For every question that contains a date: whether the gold node carries a
  day-level `event_time`, or a `recorded_at` from a trusted source
  (`content`, `capture`, `import`), the only two things the lane matches dates
  on (never `created_at`). "CANNOT SEE" means the lane had nothing to match,
  which is different from the lane missing; `lane_can_see_gold_date` is in the
  JSON. Matches on `event_time` carry full weight, on `recorded_at` half.
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
  the date questions have gold nodes with a day-level `event_time` or a
  trusted `recorded_at` at all: the report marks the ones the lane cannot see.
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
