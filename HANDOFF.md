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
2. **Recall latency: FIXED (July 6 2026 late) — the <100ms tests pass for the first time.**
   Graph-only recall 903ms -> 40-47ms. Three separate rots, each found by an instrument:
   * **Training doom loop (the big one):** TrainingLoop._maybe_train ran on EVERY recall,
     exporting the ENTIRE signal table to attempt training that could never succeed
     (sklearn absent -> _last_train_count never advanced -> permanently "ready"). ~750ms
     of every recall once the table grew. NOTE: this variably inflated EVERY latency
     number recorded July 2-6 (the 349->949ms "creep" was mostly this table growing).
     Fixed: availability check before any data export + failed-attempt backoff.
   * Structural: double BFS per recall -> single walk_full(); SELECT-per-node -> bulk
     IN() queries (get_nodes_bulk/get_neighbors_bulk, one round-trip per BFS level);
     Python full-table keyword scan -> SQL-side search_nodes_keyword (same semantics);
     normalize_label lru_cached.
   * **onnxruntime import hangs on WMI** (machine-level: wedged winmgmt made
     platform.system() take 21 MINUTES; found via faulthandler stack dump after three
     25-min network-theory probes came back wrong). Revien-side hardening that stays:
     fastembed availability via find_spec (no import at module load — pytest collection
     went 26min -> 15s on the wedged box), model load offline-first (HF_HUB_OFFLINE with
     first-install fallback — also stops fastembed's silent 65MB re-download loop and
     closes a real zero-network gap). Machine cure: elevated
     `Restart-Service winmgmt -Force` or reboot (was PENDING at handoff).
   * **CLOSED (July 7 2026, post-reboot):** full-bench identity VERIFIED — every
     retrieval figure exact to 4 decimals (0.1974/0.4128/0.5141/0.3231/0.3562) and the
     taxonomy exact to the item (1072/479/9/4, median rank 33) vs the pre-perf run on
     the same cache. **Official post-fix latency: recall p50 85ms / p90 250ms** (was
     p50 950ms) — 11x, zero retrieval change, sovereignty PASS
     (`results/20260707T151942Z_semantic.json`). Suite: 463 passed, 0 failed. The
     <100ms latency tests pass. WMI was healed by reboot; the find_spec/offline-first
     hardening stays regardless.
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

**Leg 2b — EDITABLE distilled memory: SHIPPED (commit 8f3d040, 23 tests + live).**
Distilled notes are now editable, not read-only: correct a claim (curated node
CORRECTS-supersedes), delete a line (soft-invalidate), add a line (new curated claim),
reconciled on `revien sync-vault` / `reconcile-vault`. Safety rests on a per-note manifest
(distill_manifest table = last-rendered snapshot; the ONLY safe delete referent — Mentat
caught this dependency pre-build). Hardened through TWO adversarial passes (28 findings:
8 original data-loss bugs fixed, 1 fix-introduced regression (global-redistill wiped user
free-text) replaced with surgical anchor-writeback, 4 refuted). This makes the "edit, own"
half of the hero literally true. Known v1 limits (documented, not bugs): dup-edited-line
winner is order-dependent (but stable, no oscillation); shared-claim delete from one note
drops that note's row but doesn't invalidate globally; wikilinks inside reconcile-added
claims aren't parsed into edges until next full distill.

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

## LAUNCH PLAN (July 7 2026 — the finish-out order)
Written at 87% Fable budget; anything below marked [mech] is executable by any instance
from this document alone. [Lissa] items need her hands or voice. Work top to bottom.

**Phase 0 — close the verification debt (in flight now)**
- [x] WMI healed post-reboot (platform 0.26s, model load 2.91s offline-first).
- [ ] [mech] Full-bench identity + official p50: the running post-perf full bench must
  reproduce recall@10 0.5141 / @1 0.1974 / MRR 0.3231 and taxonomy 1072/479/9/4 EXACTLY
  (same cache as the pre-perf July 6 run). If yes: OPEN 2 CLOSED, record the new p50 as
  the official latency number (slice preview: ~105ms p50 with semantic on, was ~500ms+).
  If retrieval numbers differ AT ALL: stop, bisect commit 370917e — do not rationalize.

**Phase 1 — last code item (small)**
- [ ] [mech] OPEN 3: LLMExtractor silent fallback -> loud. One stderr warning per process
  when the LLM path falls back to rule (mirror the semantic-degrade pattern in
  engine._warn_if_semantic_inactive). Test: force a failing backend, assert the warning.
  ~30 min including tests. Commit alone.

**Phase 2 — the launch surface (docs, no code) — [mech] items DONE (commit ac5f08d)**
STATUS (July 7): README/CHANGELOG/TELEMETRY/CI all written + committed, adversarially
verified (64 claims, 0 false/fabricated/wrong-number). Version 0.1.0->0.2.0 (3 sites).
STILL NEEDS LISSA: (1) voice-pass on the README — it's honest and structured but written
in my register, not hers; (2) the design writeup (her voice, the two bug stories); (3) two
judgment calls surfaced by the verification pass, below.
  - LETTA LINE REMOVED: the "recursive summarization inspired by Letta" courtesy had no
    honest home — Revien ships NO summarization feature (thesis is the opposite). If a
    future eviction/summarization feature ships, attach the courtesy there. Not in the
    launch README.
  - CI CAVEAT: 30 test "errors" on Windows are environmental (daemon socket-bind + SQLite
    temp-file permission races), not product failures — a fresh full run is 467 passed /
    0 real failures. Linux CI should be clean; if the daemon/integration tests error on
    CI too, mark those fixtures xfail-on-CI with the known-issue note (do NOT chase green
    by deleting coverage).
- [x] ~~[mech draft, Lissa voice-pass] README rewrite.~~ DONE — voice-pass pending. Structure: sovereignty wedge first
  ($0 / zero egress / zero telemetry / audit log / consent / curated shield), then HONEST
  NUMBERS exactly as banked: conversational recall@10 0.5141 (LoCoMo 1,986 QA, local
  extractive, $0), vault 0.8837, attachment 0.625->1.0 clean / 0.75 fragile with the
  aliasing holdout named. RULES: AND-not-OR (vault + conversational are separate tables,
  never blended); state the recall-vs-cloud gap openly (gpt-4.1-mini reader lifts F1 to
  0.367 — reader quality, not retrieval); the adversarial-F1 caveat (graph_only "wins"
  adversarial only by retrieving nothing) goes IN the README, not a footnote. Kill the
  stale comparison table's unverifiable checkmarks. Update install/quickstart for
  `revien connect obsidian` / `sync-vault` / `distill-vault`.
- [ ] [Lissa] Design writeup in her voice — the credibility artifact. The two banked
  stories carry it: (1) the verbatim-confidence bug ("our eval caught our own system
  disbelieving the user's literal words"), (2) the training doom loop ("the learning
  loop was the latency bug; three network theories lost to one stack dump"). Both are
  the honest-numbers wedge demonstrating itself — the Mem0 contrast writes itself.
- [ ] [mech] Zero-telemetry statement — now fully true incl. model loads (offline-first).
  One paragraph in README + a TELEMETRY.md if she wants it standalone.
- [ ] [mech] Letta courtesy line ("recursive summarization inspired by Letta's
  partial-evict pattern") wherever the summarization docs land.

**Phase 3 — release mechanics**
- [ ] [mech] CI: GitHub Actions, ubuntu-latest, `pip install -e .[dev] && pytest tests
  revien_bench/tests`. Linux likely dodges the Windows teardown errors; if not, mark
  those fixtures xfail-on-windows with the known-issue comment. Badge in README.
- [ ] [mech] Version 0.1.0 -> 0.2.0 in setup.py (+ revien.__version__); CHANGELOG.md
  summarizing the seven commits' arc.
- [ ] [Lissa] PyPI publish decision + credentials (README's `pip install revien` is a
  claim until then). Build: `python -m build`, wheel excludes revien_bench (already
  configured).
- [ ] [Lissa] MERGE GATE: feat/advanced-core-port -> main is HER hands (or Julien's).
  Never an instance's. Then tag v0.2.0.

## NEXT LEGS (planned July 7 2026, post-PyPI — Lissa + Asher's gap analysis, corrected
## against the code and merged with the instruments' findings)

Two tracks; every item names which master it serves. Standing rules bind: both corpora
re-run after any ingest/scoring change; new identity-memory features need their OWN eval
track (LoCoMo/vault fixtures don't measure them); no public number without its JSON.

**Track A — recall numbers (the public wedge). Ordered by measured leverage:**
- A1. **Weighted walk + reranker** (lever #1 per taxonomy: 1,072 outranked, median rank
  33, 316 already in top-20). Includes wiring the DEAD edge-weight code — edges already
  have weight + confidence (schema.py:140) and mark_used() reinforcement; the walker does
  unweighted BFS and reads none of it. Then train/replace the neural scorer (signals
  accumulate; force_train never called). This subsumes the doc's item 4.
- A2. **Extraction coverage** (disconnected=479) + the newline entity-regex fix
  ("Deployment\nRuns" junk — documented first task).
- A3. **Per-NodeType recency half-life** — small ScoringConfig change (identity/preference
  long, event/observation short). NOTE: confidence decay is ALREADY type-gated (INFERRED
  only, pinned immune); this is recency ranking only, and the sweep says recency is a weak
  recall lever — do it for identity-modeling correctness, expect no headline movement.

**Track B — identity memory (VesselOS-class use). Ordered by leverage-per-effort:**
- B1. **Tension/COEXIST** — CONTRADICTS edge type EXISTS (schema.py:30), never drawn.
  Add a COEXIST gate outcome for preference/identity-class claims: draw CONTRADICTS, both
  nodes stay live, neither queued nor superseded ("I want closeness" + "I want space").
  CLAIM_TAXONOMY §7.2 (sentiment ≠ contradiction) is the guardrail. Needs its own small
  eval/tests — invisible to existing benches. (Engram-harvest task a6c924dc relates.)
- B2. **Bi-temporal validity** — event_time_start/end columns EXIST (CSL Leg 2); schema
  comment says valid_from/valid_until were deliberately deferred to L3. The leg: (a)
  supersession closes the OLD fact's validity window instead of just invalidating,
  (b) recall(..., as_of=) query path, (c) tests. "Where did she live in March?" becomes
  answerable. This is our own L3, not a Graphiti import.
- B3. **Dream mode (autonomous consolidation)** — a periodic daemon sweep that also mops
  the KNOWN dangling threads: _apply_decay is never called automatically (verified),
  neural training never fires, reindex/clustering refresh are manual, orphan cleanup.
  Any pass that MERGES existing nodes must feed the false-merge audit surface.
- B4 (defer). **FSRS-6 decay** — measure per-type recency (A3) first; FSRS is
  tuning-heavy and the sweep says recency isn't the lever. Revisit with data.
- B5 (far). **Knowledge-gap detection** — product feature atop clustering + density;
  not a memory-engine primitive. Park it.

**Corrections to the gap analysis (so nobody re-litigates):** CONTRADICTS edge type,
edge confidence/weights, and event-time interval columns ALL already exist — the gaps are
that nothing draws/reads/closes them. The candidate queue already preserves both sides of
a contradiction. Confidence decay is already per-type-gated. The doc's items 1/2/4 are
therefore smaller than specced; what it omitted (reranker) is the largest measured lever.

## A1 VERDICT (July 10 2026) — weighted walk NULL, cross-encoder rerank is the lever
Both halves built, swept, and full-scale confirmed in one session. Read this before
touching ranking again.

**Weighted walk: built, measured INERT — and the null is the finding.** The walker now
tracks path strength (product of edge weights along the shortest-hop path, strongest
same-level parent wins, same one-round-trip-per-level perf shape) and the scorer blends
it into proximity behind REVIEN_EDGE_WEIGHT_BLEND (default 0.0 = byte-identical;
REVIEN_EDGE_CONFIDENCE_IN_WALK multiplies edge confidence in). The full ew ladder
(0.25/0.5/0.75/1.0/conf) came back IDENTICAL to baseline to 4 decimals on BOTH corpora.
Measured mechanism (hand-verified against the cached bench db): with semantic-as-spine
the entire top-20 is distance-0 semantic anchors, where path strength is definitionally
1.0 — proximity (hop OR strength) is candidate-generation, not ranking. The `outranked`
bucket is anchor-vs-anchor BI-ENCODER misranking; no graph-side knob can reach it. The
knob ships (graph-only fallback + Track B edge-heavy flows want it); no default flip.

**Cross-encoder head rerank: the champion.** revien/semantic/rerank.py — local ONNX
cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2, 80MB, fastembed TextCrossEncoder), OPT-IN
via REVIEN_RERANK=1, rescores the top-K (REVIEN_RERANK_TOP_K, default 30) base-ranked
results BEFORE the top_n slice, tail untouched, raw score in
score_breakdown[rerank_score], base score never overwritten. Same discipline as the
semantic spine: find_spec guard, offline-first load, self-disabling, loud degrade.
- **FULL SCALE (1,986 QA, fresh checkpoint, `results/20260710T171258Z_semantic.json`):
  recall@1 0.1974 -> 0.4036 (2.04x), recall@5 0.4128 -> 0.5851, recall@10 0.5141 ->
  0.6370 (+24%), MRR 0.3231 -> 0.5341, nDCG@10 0.3562 -> 0.5397.** Taxonomy: outranked
  1072 -> 733; disconnected EXACTLY 479, never_extracted 9, walk_depth 4 — all pinned
  (mechanism confirmed to the item). Sovereignty PASS, $0, network_calls=0.
- Vault corpus same knob: recall@1 0.5233 -> 0.7674, MRR 0.7378 -> 0.9593, recall@10
  0.8837 -> 0.9419, single_note 1.0, attachment MRR 1.0, misses 9 -> 5 (all outranked).
- k50 vs k30: +0.005 recall@10 at 2x rerank cost — k30 is the shape.
- **The open trade — LATENCY (Lissa's call, default-flip gate):** recall p50 85ms ->
  593ms, p90 250ms -> 1,097ms at full scale (cross-encoder cost is token-bound; vault's
  shorter notes ran p50 ~573ms). Options if default-on is wanted: int8 quantized variant
  of the same model (onnx/model_quantized.onnx exists upstream), content truncation cap,
  or ship as documented "quality mode". Default stays OFF until her verdict.
- Residual outranked (733) has median best rank 94 (was 33) — the reranker consumed its
  own headroom; what remains is buried too deep for a bigger head. Next levers there are
  extraction coverage (A2) and aliasing/vocabulary, NOT more reranking.
- The old TF-IDF NeuralScorer stays as-is (real-usage signals, never fires in-bench);
  "replace the neural scorer" is hereby satisfied by the cross-encoder path.

**Bench trap logged:** the runner RESUMES from `.checkpoint_semantic_extractive.jsonl`
even when env knobs changed — a first confirm run came back `ran=0 resumed=10` and
silently replayed July's pre-rerank rows. ALWAYS read the `resume:` line before trusting
a run; stale checkpoint moved to `.pre-rerank.jsonl.bak`. Env knobs are not part of the
checkpoint identity — worth fixing someday (hash the knob env into the checkpoint name).

**Also this session:** conflicts_with edge type + POST /v1/edges + recall
include_context passthrough + REVIEN_DB_PATH env fallback (Lissa's server patch,
reviewed + pushed, `8164f28`) — B1's edge primitive now has an API.

## A2 VERDICT (July 10 2026) — wide net under the rerank guard; extraction leg retired
Same session as A1; read A1's mechanism first — A2's answer builds on it.

**Newline entity-regex fix (`8fa95b4`): shipped, measured recall-INERT.** \\s in the
multi-word entity pattern fused line-boundary words into phantom entities
("Deployment\\nRuns") — now horizontal-only whitespace, quoted patterns single-line,
regression-tested. Fixed-extractor rebuild reproduced every conversational number to 4
decimals: the junk never sat on gold paths. Hygiene, not headline — do NOT expect recall
from extractor cosmetics.

**The `disconnected` bucket fell to a knob, not an extraction leg.** Definition
(failure_analysis): gold unreachable from anchors even at depth 6 — i.e. below the
semantic top-K cutoff AND no graph path. Every turn IS embedded, so net WIDTH is the
direct lever: REVIEN_SEMANTIC_TOP_K 30->100. That was round-1's measured loser (weak
sims flooded the ranking) — but that failure mode is exactly what the cross-encoder head
now guards. Old loser, new bodyguard:
- rerank_wide (k30 head): disconnected 71->33 subset at NO latency cost, but converts
  land in `outranked` (the gazetteer lesson: reachability without ranking moves buckets).
- **rerank_wide_k50 (the quality-mode shape): REVIEN_RERANK=1 REVIEN_SEMANTIC_TOP_K=100
  REVIEN_RERANK_TOP_K=50. FULL SCALE (1,986 QA, fresh checkpoint, fixed extractor,
  `results/20260710T202404Z_semantic.json`): recall@1 0.4175 (2.11x baseline), recall@5
  0.6073, recall@10 0.6607 (+28.5% vs 0.5141), MRR 0.5514, nDCG 0.5583. Disconnected
  479 -> 246 (-49%), walk_depth 4 -> 1. Latency p50 1,156ms / p90 1,902ms. Sovereignty
  PASS, $0, network_calls=0.**
- Vault same knobs + fixed extractor: recall@10 0.9419 -> 0.9535, cross_note 0.8667,
  misses 4 (all outranked, ZERO disconnected). Both-corpora rule satisfied.
- SIM_FLOOR 0.30->0.20 bound nothing (identical rows) — the floor is not the constraint
  at top-100.

**The ranking-mode menu (all env, all measured, defaults untouched pending Lissa):**
  default        85ms p50   recall@10 0.5141   (shipped behavior, byte-identical)
  rerank         593ms p50  recall@10 0.6370   (REVIEN_RERANK=1)
  quality mode   1.16s p50  recall@10 0.6607   (+ SEMANTIC_TOP_K=100, RERANK_TOP_K=50)

**What A2 retires and what it leaves:** the planned extraction-coverage leg for
conversational recall is RETIRED — the wide net does the reaching cheaper than better
extraction would (second measurement today to kill planned work; cheaper than building
it). Still real: residual 246 disconnected + deep-buried outranked (median rank 69) are
the aliasing/vocabulary class ("offline mode"->Roadmap 2026); never_extracted stays 9
(ingest near-lossless, as banked July 2). Extraction work now only serves graph QUALITY
(Track B / distill), not recall numbers.

**Second cache trap logged:** the bench db cache is keyed by config+conv, NOT extractor
version — post-regex measurements used a fresh dir (results/db_cache2; db_cache is
old-extractor). Same class as the checkpoint trap: code version isn't part of either
cache identity. Fold both into one fix when touched next (hash extractor/knob env into
cache + checkpoint names).

## B1 SHIPPED (July 10 2026) — COEXIST: tension as first-class memory (`702826e`)
The identity-memory leg. Two AFFIRMATIVE claims pulling opposite directions ("I want
closeness" / "I want space") now COEXIST: both live, nothing queued, nothing
invalidated, tension drawn as a CONFLICTS_WITH edge (the type the morning patch
`8164f28` introduced; edge is non-mutating by contract, idempotent per pair).

- **Detection is recognizer territory, not rules** — the rule gate literally cannot see
  affirmative-affirmative opposition (no flip, no negation, often no shared tokens).
  `revien/tension.py` LLMTensionRecognizer mirrors + subclasses the sensitivity
  recognizer's transport: REVIEN_TENSION_BACKEND (ollama local default), one-time cloud
  disclosure, never raises, abstain on unavailable. ONLY a clean TENSION verdict draws
  the edge; COMPATIBLE/UNSURE fall through to unchanged NO_CONFLICT.
- **§7.2 mechanized as guards, all pinned by tests:** retraction keeps its supersession
  path (a negation never reaches the hook); sentiment/similarity need the recognizer's
  explicit TENSION to fire; single-valued types (identity/relationship/state/health)
  are excluded — a value mismatch there is a real either/or and keeps human review.
- **Human resolution surface:** ClaimGovernor.coexist_candidate(id) — resolve a queued
  candidate as "both true": edge drawn, candidate resolved "coexist", claims intact.
- **Own eval track built, NUMBER PENDING:** revien_bench/tension_eval.py + 24-pair
  fixture (8 tension / 16 §7.2 controls). Reports coexist recall, false-fires (must be
  0), and classifier-blocked separately — the RULE CLASSIFIER is the binding coverage
  limit ("I want space to be alone with my thoughts" -> unclassified -> hook never
  reached), the same known CSL bound, NOT recognizer quality. Needs a live backend
  (ollama on the server, or OpenRouter — key is Lissa's); refuses to print a number
  from a dead transport. **Run it server-side before any B1 claim.**
- **Bench numbers cannot move** (double-gated: REVIEN_CSL off in benches AND tension
  backend unwired by default; unwired gate byte-identical, pinned by test). Both-corpora
  re-run therefore waived for this leg — the waiver reasoning recorded here on purpose.
- Cost profile: the hook consults the recognizer on scoped-but-compatible tension-type
  pairs — broader than Trigger 2's would-be-auto trigger. Wire LOCAL (default) or
  accept per-pair cloud cost knowingly.
- NEXT for identity memory: recall/lineage surfacing of tension edges (a "tensions"
  view — who am I in conflict with myself about?), B2 bi-temporal validity.

**Phase 4 — post-launch roadmap (NOT launch-blocking; keep out of scope creep's reach)**
- Reranker / ranking headroom: 1,072 outranked, median rank 33, 316 in top-20. The
  neural scorer is trained on accumulated signals or replaced. Biggest recall lever left.
- Extraction coverage (conversational disconnected=479 is extraction, not matching) +
  the newline entity-regex fix.
- Aliasing/vocabulary ("offline mode" -> Roadmap 2026 — the attachment holdout).
- Daemon config->adapter auto-sync factory (pre-existing gap, all adapters).
- Vault note-edit chunk reconciliation (re-ingest currently duplicates CONTEXT units).

**Standing rules that survive any instance change:** verify against the running system;
both corpora re-run after ANY ingest-touching change; taxonomy read before any tuning;
the attachment line always reported separately; no public number without its results
JSON; the false-merge pairs list reviewed each run; merge gate is Lissa's.

## Notes
- Strategy is settled: **pure OSS**, goal = credibility/portfolio for consulting. Path A
  (local-first port of RCE *concepts*, clean-room — RCE itself is internal IP, never ported).
- The A-vs-B "chase recall vs sovereignty-only" fork is DEAD — recall was a bug, now fixed.
- `godot-bash-plugin/vendor/godot-mcp` (in the Thornwood repo, unrelated) was committed as an
  embedded git repo — convert to submodule or un-vendor someday.
