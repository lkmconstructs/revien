# Revien Flagship — Session Handoff (2026-06-18)

Resume point for a fresh thread. Memory files (auto-loaded) hold the deep history:
`revien-flagship-direction.md`, `api-cost-projection-gate.md`, `sovereign-loop-harness.md`,
`spine-gate-non-enforcing.md`. This is the 60-second "you are here."

## CSL Leg B — WIP as of 2026-06-29 (resume here)
The Claim Sovereignty Layer is now WIRED into live ingest and proven end-to-end
(governor + candidate queue + pipeline hook, opt-in via `REVIEN_CSL`; acceptance 9/9,
0 safety breaches). Trigger 2 recognizer locked to **qwen via OpenRouter** (`qwen/qwen3.6-max-preview`);
adversarial-fresh battery ~2–4% residual on deep-coded, 0 on clear.
**Fact-update fix (the current front):** measured baseline was 0/68 ordinary fact-changes
holding current truth; built `revien/fact_change.py` (dimension/marker/value detection) as a
recognizer-gated fallback → **~76% now, 0 sensitive breaches, 0 wrongful erases** (186 unit
tests pass). Measure with `python -m revien_bench.measure_fact_update` (+ qwen env).
**NEXT:** the residual ~24% is no-shared-vocabulary cross-phrasing ("looking for work"→"got
hired") that rules can't link — decision pending with Lissa: build an **LLM contradiction-judge**
(mirror the recognizer infra, ~95%+, per-ingest cloud cost) vs push rules to ~85%. See
memory `leg-b-csl-wired.md`.

## Where things stand
- **Work lives in this sandbox:** `C:\Users\Melissa\Revien\revien-build`, branch
  `feat/advanced-core-port`. Latest commit `6c9506f`. **Nothing pushed** to the OSS remote
  (`github.com/lkmconstructs/revien.git`) — push is Lissa's call.
- **Flagship is built + verified** (legs 1–6, committed): confidence layer, clustering
  (Louvain; Leiden opt-in), neural reranker (opt-in), hybrid semantic search, provenance
  (audit/lineage/soft-invalidate), choose-your-storage governance (retention/forget/export),
  pluggable local-first extraction (cloud opt-in + disclosed). Everything cloud/heavy is opt-in.
- **Benchmark harness built** (`revien_bench/`, dev-only): LoCoMo loader, ingest, official
  token-F1 + recall@k metrics, ExtractiveAnswerer + LLM answerers (Ollama/OpenAI/Anthropic),
  sovereignty asserts, runner, report. Design in `BENCHMARK_DESIGN.md`.
- **The big fix:** benchmark exposed near-zero recall (1.5% F1). Root cause was UPSTREAM —
  CONTEXT nodes (verbatim turns) were excluded from semantic indexing AND results, so the
  answer-bearing content was invisible to retrieval. Fixed (`6c9506f`): **recall@10 0.04 → 0.47
  (~12×)** on free local bge-small. 1.5% was a pipeline bug, NOT a sovereignty tradeoff.

## IMMEDIATE NEXT (paid — project & confirm first, per api-cost-projection-gate)
Get the headline F1 with the **cost-smart** config (retrieval works on the free rule
extractor + local embeddings now, so only the *answerer* needs the API):

```
# SUBSET smoke first (~250 calls, ~$0.02–0.15):
OPENAI_API_KEY=<FRESH key> REVIEN_SEMANTIC=1 python -m revien_bench.runner \
    --config semantic --answerer openai:gpt-4.1-mini --limit 2 --max-qa 80 --out results/
# then FULL (~2,000 calls, ~$0.10–1.30):
OPENAI_API_KEY=<FRESH key> REVIEN_SEMANTIC=1 python -m revien_bench.runner \
    --config semantic --answerer openai:gpt-4.1-mini --out results/
```
- Key lives in this machine's local session logs (Claude Code, not a cloud chat), so reuse is
  fine; Lissa disables it after the benchmarks are done. Pass it inline on the command.
- Do NOT use `REVIEN_EXTRACTOR=openai` (the ~5,882-call per-turn extraction hog, ~$3–6) —
  unnecessary now that context nodes are surfaced.
- Anchor: ~$0.000047 / gpt-4.1-mini answer call (measured). Lissa is OpenAI Tier 4 ($140 bal).

## Benchmark result (directional — for the README)
Cost-smart config (rule extractor + local bge-small + gpt-4.1-mini reader), **304 QA / 2 convs, $0.03**:
**overall token-F1 0.367**, recall@10 0.43. Per-cat: **adversarial 0.83 (hallucination resistance —
refuses when it lacks info)**, single-hop 0.32, multi-hop 0.19, open-domain 0.15, **temporal 0.09**
(answerer "yesterday"→date gap). Recall-bound; better embeddings/extraction lift it. 2/10 convs =
directional, not leaderboard-definitive. Checkpoint at `results/.checkpoint_semantic_openai_gpt-4.1-mini.jsonl`
— resume to extend (no re-burn); compute aggregate from banked rows offline. NOTE: the full 1,986-QA
run kept dying on a flaky background-process executor (not Revien); each `python -m revien_bench.runner
--config semantic --answerer openai:gpt-4.1-mini` resumes from checkpoint.

## OPEN items
1. **(optional) Extend the benchmark** past 304 QA by resuming the checkpoint on a stable executor.
2. **Recall latency ~250ms** (pre-existing): `_keyword_search` does `list_nodes(limit=999999)`
   full-table-scan + redundant `get_node` in the recall path. The 2 red `test_retrieval_time_under_100ms*`
   tests assert an aspirational <100ms the code never met. Fix the scan OR set honest thresholds (best-of-N).
3. **Silent extractor fallback** — `LLMExtractor` silently falls back to rule on failure; make it
   loud (it masked the leak under quota 429s).
4. **L8 launch** — README leading with the **sovereignty wedge + honest numbers** (recall + the
   $0/zero-egress/audit/consent metrics where Revien wins; state the recall-vs-cloud gap openly),
   a design writeup **in Lissa's voice** (the credibility artifact), zero-telemetry statement, CI,
   Apache license + a one-line Letta courtesy ("recursive summarization inspired by Letta's
   partial-evict pattern").

## Notes
- Strategy is settled: **pure OSS**, goal = credibility/portfolio for consulting. Path A
  (local-first port of RCE *concepts*, clean-room — RCE itself is internal IP, never ported).
- The A-vs-B "chase recall vs sovereignty-only" fork is DEAD — recall was a bug, now fixed.
- `godot-bash-plugin/vendor/godot-mcp` (in the Thornwood repo, unrelated) was committed as an
  embedded git repo — convert to submodule or un-vendor someday.
