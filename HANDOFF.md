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

## Semantic-as-spine + miss taxonomy (July 1, 2026)
- **Semantic layer promoted from extra to SPINE**: sqlite-vec + fastembed are core deps
  (setup.py install_requires + requirements.txt). Rationale: graph-only recall has no
  query-relevance term (recall@10 0.05 vs 0.47 hybrid). `revien[semantic]` extra kept as alias.
- **Degrade is LOUD now**: engine warns once per process on graph-only recall; every
  RetrievalResponse (and `/v1/recall`) carries `semantic_active` + `semantic_note` (why it's off,
  incl. runtime-failure reason). `REVIEN_SEMANTIC=require` makes missing/broken semantic FATAL.
  The warning immediately caught a real silent degrade: a corrupted fastembed model cache in
  %TEMP% that had this machine silently on graph-only.
- **Stale-vector bug fixed**: SemanticIndex registers a content listener on GraphStore —
  node label/content updates re-embed immediately, deletes drop the vector. Metadata/access
  updates do NOT trigger embeds (hot path). Manual `/v1/reindex` still exists for backfill.
- **Per-query retrieval miss taxonomy** in the bench (`revien_bench/failure_analysis.py`,
  needs `engine.recall(debug=True)` diagnostics): each missed gold dia_id classified as
  never_extracted / no_anchors / walk_depth_miss / disconnected / filtered_out / outranked.
  Aggregated in results JSON (`retrieval_failure_analysis`) + summary print. Old checkpoints
  fold in as "unclassified" (no false resume).
- **At-scale taxonomy (FULL 1,986-QA run, July 2, 2026)** — the 8-QA micro-run's
  "disconnected dominates" read did NOT hold at scale. Full numbers
  (`results/20260702T032834Z_semantic.json` / `..._graph_only.json`, extractive reader, $0):
  * semantic recall@10 **0.4209** vs graph_only **0.0488** (8.6x) — spine promotion justified.
  * semantic misses (1,768): **outranked=1276 (72%)**, disconnected=479 (27%),
    never_extracted=9, walk_depth_miss=4. Outranked median best rank **26**; 480/1276 already
    within top-20. Dominant in EVERY category.
  * graph_only misses (2,688): disconnected=1342, outranked=1128, no_anchors=108 — semantic
    anchors fixed most connectivity; what remains is RANKING.
  * ingestion is near-lossless (never_extracted=9/2,688) and depth-3 walk is fine
    (walk_depth=4) — do NOT spend effort there.
  * Caveat: graph_only "wins" overall F1 (0.089 vs 0.063) ONLY via adversarial — it retrieves
    nothing, the extractive reader refuses, refusal scores 1.0. Broken-by-default is not
    hallucination resistance; per-category non-adversarial F1 improves 2-20x with semantic.

## OPEN items
1. **(optional) Extend the benchmark** past 304 QA by resuming the checkpoint on a stable executor.
2. **Recall latency ~250ms** (pre-existing): `_keyword_search` does `list_nodes(limit=999999)`
   full-table-scan + redundant `get_node` in the recall path. The 2 red `test_retrieval_time_under_100ms*`
   tests assert an aspirational <100ms the code never met. Fix the scan OR set honest thresholds (best-of-N).
3. **Silent extractor fallback** — `LLMExtractor` silently falls back to rule on failure; make it
   loud (it masked the leak under quota 429s). (Semantic-layer silent degrade is FIXED — same
   pattern still open for the extractor.)
4. **L8 launch** — README leading with the **sovereignty wedge + honest numbers** (recall + the
   $0/zero-egress/audit/consent metrics where Revien wins; state the recall-vs-cloud gap openly),
   a design writeup **in Lissa's voice** (the credibility artifact), zero-telemetry statement, CI,
   Apache license + a one-line Letta courtesy ("recursive summarization inspired by Letta's
   partial-evict pattern").
5. **Ranking: frequency term is self-poisoning — DECISION NEEDED** (sweep-confirmed at full
   scale, July 2 2026 PM). The ranking-knob sweep (`revien_bench/sweep.py`, cached ingests via
   `--db-cache`, env knobs `REVIEN_*_WEIGHT` / `REVIEN_SEMANTIC_TOP_K` / etc.) found:
   * **no_freq (recency 0.5 / freq 0.0 / proximity 0.5) full-scale: recall@10 0.4209 → 0.5004,
     recall@1 0.1083 → 0.2052 (+89%), MRR +53%.** All recovered misses came out of `outranked`
     (1276→1092); disconnected untouched at 479 — mechanism confirmed by control variant
     (prox_only ≡ no_freq exactly; in-bench recency is constant).
   * Cause: `recall()` touches its own results (access_count++), so previously-retrieved hubs
     outrank query-relevant nodes. Popularity prior contaminated by the engine's own behavior.
   * Losers: topk100 (−0.04, floods ranking with weak sims), refine10 (crushes recall@1),
     floor15 + halflife365 (inert), no_freq+refine10 combo (tanks).
   * **ROUND 2 (content recency approved + landed, July 2 2026 PM):** recency now scores
     `recorded_at` (content time, created_at fallback) — SHIPPED as default per Lissa.
     REVIEN_TOUCH_ON_RECALL gate added (default ON). Sweep under new semantics:
     **no_touch beats no_freq on every metric** — fork (a) resolved: keep the frequency
     term, stop recall() self-touching. Half-life ladder: 7d crushes @1, 90d crushes @10,
     365d (gentle tiebreak) wins. **Champion no_touch_hl365 FULL-SCALE: recall@10 0.5101,
     recall@1 0.1763, MRR 0.3046** (vs shipped-default baseline 0.4209/0.1083/0.2146 =
     +21%/+63%/+42%). All signals honest. Round-1 no_freq full numbers were measured under
     the old access-time recency — not comparable.
   * **SHIPPED (Lissa approved, July 2 2026):** both defaults flipped —
     REVIEN_TOUCH_ON_RECALL default OFF (mark_used()-only frequency; =1 restores old
     behavior) + recency_half_life_days 7 → 365 (recency = content-age tiebreak). Both
     env-overridable. **Verified: stock defaults, zero overrides, full 1,986 QA reproduce
     the champion exactly — recall@10 0.5101 / @1 0.1763 / MRR 0.3046**
     (`results/20260702T220901Z_semantic.json`). This is the new official defaults number
     (was 0.4209/0.1083/0.2146 this morning). Tests updated (defaults asserted; decay
     mechanics tested with explicit 7d configs).
   * Sweep tooling kept: `revien_bench/sweep.py` + `--db-cache` pristine-ingest cache
     (ingest once, sweep ranking knobs in recall-only time).
6. **Entity normalization leg: SHIPPED with a measured verdict (July 6 2026).**
   * Prong A — `revien/graph/normalize.py`: ONE definition of label equivalence (case,
     separators, possessives, LEADING ARTICLES — the extractor captures "The Atlas Server";
     deliberately NOT plurals/aliases). Applied at dedup, find_node_by_label, fuzzy,
     anchors, link resolution. Everywhere, free, correct.
   * Prong B — gazetteer mention pass in the pipeline: known entities get CONTEXT->ENTITY
     edges (weight 0.6) when a turn mentions them in ANY surface form (word-boundary on
     normalized text, min-length 4 guard, per-pipeline cache). **GATED TO CURATED ENTITIES
     ONLY** after the measured verdict below.
   * **Verdict, two corpora:** cross-corpus DECISIVE — attachment 0.625→1.0 clean,
     0→0.75 fragile (the one holdout is semantic aliasing: "offline mode"→Roadmap 2026 —
     vocabulary work, correctly out of scope). Pure conversation: ZERO recall gain
     (0.5141 identical to 4 decimals) — ungated gazetteer moved 25 items from
     `disconnected` to `outranked` (reachable, then buried by ranking) at −19% ingest and
     +60% recall latency. Hence curated gating: keep the whole win, drop the whole cost.
     Cross-type MERGING (Asher's framing) deliberately not built — collapsing FACT into
     ENTITY destroys type semantics; the mention edge achieves walkability with distinct
     nodes, and the identical-recall result shows merging wouldn't have bought recall either.
   * **False-merge precision surface (Asher's guard):** every normalization-only merge is
     audit-logged with both labels and surfaced in both benches
     (`normalization_merges` in vault eval, `normalization_merges_sample` in runner).
     Vault corpus first reading: 2 merges, both correct article variants, zero false.
     Substring merges impossible by construction (exact canonical equality; word-boundary
     gazetteer) — pinned by tests. Same-name-different-referent residue is what the audit
     list exists for.
   * Conversational `disconnected` (now ~454) is NOT surface-form fragmentation — the
     remaining bottleneck stays RANKING (1,094 outranked, median rank 35 → reranker /
     top-N headroom) and extraction coverage. Still open: entity regex matches across
     newlines (junk like "Deployment\nRuns").

## Obsidian vault legs (July 2 2026 — Lissa's scoping)
**Framing (hers, binding): AND-not-OR.** The vault is a SECOND corpus with its OWN eval and
OWN number — never blended with the conversational LoCoMo figures (0.51 recall@10). The
prize isn't purity, it's structure: the author already drew the graph.

**Leg 1 — vault ingest: SHIPPED (built + wired + tested, 15 tests).**
- `revien/adapters/obsidian.py`: chunk per heading (kills note-blobs), `[[wikilinks]]` +
  tags + note-title → `links`, frontmatter date → recorded_at (beats mtime), all curated.
- Pipeline: `IngestionInput.links` → ENTITY nodes (found case-insensitively or created) +
  CONTEXT→ENTITY RELATED_TO edges at weight 0.8 (author-drawn beats the extractor's 0.3
  co-occurrence guesses). `curated=True` → confidence 1.0 + `metadata.curated`.
- **CSL curated shield** (`supersession_ingest.py`): a machine claim that would
  AUTO_SUPERSEDE a curated claim is downgraded to the candidate queue — human-curated
  memory is never silently overwritten; a curated claim superseding a machine claim passes
  through (the human's word outranks ours). Consent Is Law, mechanized. Tested both ways.
- CLI: `revien connect obsidian --path <vault>` + one-shot `revien sync-vault [--full]`.
- Scheduler passthrough fixed (timestamp was dropped for ALL adapters; links/curated added).
- Known v1 trades: re-edited notes re-ingest as new CONTEXT units (mtime-gated; chunk
  reconciliation is future work). Daemon config→adapter auto-sync factory is a PRE-EXISTING
  gap for every adapter; sync-vault is the guaranteed path.

**Leg 2 — distill-OUT: SHIPPED (built + wired + tested, 12 tests + live-verified).**
"Your AI's memory is files you can open." `revien/distill.py` (VaultDistiller) + CLI
`revien distill-vault [--folder Revien] [--min-claims N]`:
- One markdown note per entity holding machine-side memory: claims grouped by type
  (Decisions/Facts/Preferences/Events), each line carrying provenance (source, content
  date, confidence, vault-vs-conversation), related entities as [[wikilinks]] — Revien's
  memory threads into Obsidian's OWN graph pane. Plus a `_Revien Index` note.
- Hard rails, each with its own test: writes ONLY inside the distill folder; overwrite AND
  prune are marker-gated (`revien: derived` frontmatter) so user files are inviolate even
  inside the folder; ingest adapter skips marked notes (echo loop closed — live-verified
  with a --full re-sync); deterministic content, idempotent regeneration (no phantom churn
  for sync tools); pure read — graph never mutated; vault-echo guard (entities with only
  vault-origin claims are not echoed back); junk graph labels degrade to not-linked, never
  broken markdown.
- Live finding for the eval leg: cross-corpus entity attachment is fragile — a conversation
  saying "Providence-Core" (hyphen) never attached to vault entity "Providence Core", so
  the claim distilled nowhere. Entity normalization (ingest-edge leg, OPEN 6) is what
  closes this; the vault eval should measure attachment rate explicitly.

**GATE (hers, binding): the vault eval lands BEFORE any public Obsidian claim.** The
honest-numbers wedge is the one thing we don't spend.

**Vault eval v1: BUILT + FIRST NUMBERS (July 6 2026).** `revien_bench/vault_eval.py` +
fixture corpus (`fixtures/vault/` — 18 interlinked notes, fictional "Fernweh Labs" vault;
`vault_conversations.json` — 12 turns w/ attachment ground truth incl. deliberate fragile
label variants; `vault_qa.json` — 43 QA across single_note / cross_note / attachment).
SHA-locked, $0, zero egress, miss-taxonomy instrument transferred (gold space = note paths
+ turn ids), results in their own namespace `results/vault/` with no `overall_f1` key so
the number can't be pasted into the conversational table. 7 smoke tests incl. gold-id
integrity + isolation assertions.
- **Numbers (post verbatim-confidence fix): recall@10 0.8837 overall — single_note 0.9333,
  cross_note 0.7333, attachment 1.0. MRR 0.7417. Misses: 9, all outranked.**
- **Attachment rate (its own line, always): 0.625 on clean-label turns, 0.0 on fragile
  variants (hyphen/lowercase).** Baseline-with-known-gap; entity normalization (OPEN 6)
  is the fix and this line is its scorecard.
- **THE VERBATIM-CONFIDENCE BUG (bank this for the pitch — story, not changelog):** the
  eval's first run caught the rule extractor storing verbatim CONTEXT turns as
  inferred/0.5 — the system literally half-believing the user's own words vs curated
  content, AND putting them on the INFERRED decay path. The LLM extractor had it right;
  the default didn't. Fixed (extractor.py: EXTRACTED/1.0). Vault attachment retrieval
  went 0.0 → 1.0 on the fix. Lissa's framing, binding: "we built a mixed-corpus eval and
  it immediately caught our own system disbelieving the user's literal words" — the
  honest-numbers wedge demonstrating itself (the counter-position to Mem0-style junk
  metrics). Do NOT bury it.
- **Gate status: measurement side SATISFIED (July 6 2026).** Post-fix conversational
  confirm landed (`results/20260706T140042Z_semantic.json`, full 1,986 QA, stock defaults):
  **recall@10 0.5141 / recall@1 0.1974 / MRR 0.3231** (was 0.5101/0.1763/0.3046).
  Effect-size read of the verbatim fix on episodic recall: modest and consistently
  positive, concentrated at the top of the ranking (@1 +12% rel, MRR +6% rel) — exactly
  the mechanism's prediction: answer-bearing verbatim turns now outrank extraction-noise
  nodes (still 0.5), and verbatim memory is off the decay path. Mixed corpus: existential
  (attachment 0→1.0). Pure conversation: precision bump. One bug, two corpora, coherent
  story. **The three publishable numbers: conversational 0.5141, vault 0.8837, attachment
  0.625-clean/0.0-fragile (own line, known gap).** The publish CALL itself remains Lissa's.

## Notes
- Strategy is settled: **pure OSS**, goal = credibility/portfolio for consulting. Path A
  (local-first port of RCE *concepts*, clean-room — RCE itself is internal IP, never ported).
- The A-vs-B "chase recall vs sovereignty-only" fork is DEAD — recall was a bug, now fixed.
- `godot-bash-plugin/vendor/godot-mcp` (in the Thornwood repo, unrelated) was committed as an
  embedded git repo — convert to submodule or un-vendor someday.
