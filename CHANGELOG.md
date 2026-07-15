# Changelog

All notable changes to Revien are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [Unreleased]

### Added
- **Persistent adapter-sync cursors** (`sync_cursors` table). The first-ever sync of an
  adapter starts at epoch, so everything from before the daemon existed is ingested; a
  daemon restart resumes from the last successful sync instead of resetting to now()
  and silently skipping the offline window. The cursor is captured BEFORE the fetch and
  persisted only on success — content landing mid-sync is caught by the next window,
  and a failed sync never advances the position.
- **`ingest_key` on `IngestionInput`** — stable re-ingest identity for a unit that is
  re-fetched whole on every change. The claude_code and codex adapters key each session
  file: a grown session now REFRESHES its one existing context node (no-op when
  unchanged, in-place update + re-extraction + dedup when grown) instead of stacking a
  duplicate whole-session context node every sync. No key = append behavior, unchanged.
  Note: refresh only ever adds — it never removes claims extracted from text that was
  later edited away; the key is intended for append-only units like session logs.

### Fixed
- **Auto-sync fires immediately at daemon startup**, then every interval — no more
  silent 6-hour wait before the first sync of connected adapters.
- **Pre-existing duplicate session contexts stop growing.** The first keyed ingest of a
  session that was synced before this release ADOPTS the newest of its old unkeyed
  context nodes (stamps and refreshes it) instead of appending yet another copy. Known
  limitation: the older historical duplicates remain in the graph — retroactive merge
  is out of scope here; the consolidate (dream) pass is the future home for that cleanup.

## [0.3.0] — 2026-07-11

Smarter by default. Retrieval quality jumped for every user with zero config —
and memory learned to hold tension and time.

### Added
- **Cross-encoder reranker, ON by default.** A local 23MB int8 model rescores the
  top-20 candidates reading query and memory together. Full-scale verified:
  conversational recall@10 0.514 → 0.593 (+15%), recall@1 0.197 → 0.386 (+95%),
  MRR +57% at p50 261ms; vault recall@10 0.884 → 0.942, MRR 0.959. `REVIEN_RERANK=0`
  opts out and restores the 85ms path byte-identically; fp32/deeper-head knobs reach
  0.661 for latency-tolerant consumers. The whole latency-quality dial is measured,
  banked, and documented — every point verified at full scale, $0, zero egress.
- **Tension as first-class memory (COEXIST).** Two affirmative claims pulling in
  opposite directions ("I want closeness" / "I want space") now BOTH stay live, with
  the tension drawn as a `conflicts_with` edge instead of one claim silently
  superseding the other. Opt-in recognizer (`REVIEN_TENSION_BACKEND`, local Ollama
  default, cloud disclosed); human queue resolution ("both true") included. Surfaced
  via `revien tensions`, `GET /v1/tensions`, and `include_tensions` on recall.
- **Bi-temporal validity.** Supersession closes the old fact's validity window and
  opens the new one's at the transition instant. `recall(as_of=...)` — also
  `revien recall --as-of` and the REST `as_of` field — answers "what was true THEN":
  a superseded fact whose window covers the queried moment comes back.
- **`POST /v1/edges`** for explicit typed edges, `conflicts_with` edge type,
  `include_context` on the recall API, `REVIEN_DB_PATH` env fallback for direct
  ASGI/Docker use.
- **Weighted graph walk** (path strength from edge weights, `REVIEN_EDGE_WEIGHT_BLEND`)
  — measured inert for semantic-first ranking, shipped default-off for graph-only
  and identity-memory flows.

### Fixed
- Entity extraction no longer fuses words across newlines into phantom entities
  ("Deployment\nRuns").
- Benchmark checkpoint and ingest-cache identity now include the env knobs and code
  that produced them — a knob or code change can never silently resume or reuse
  stale data (two real near-misses closed).

## [0.2.1] — 2026-07-07

### Fixed
- **First-install semantic load.** On a cold model cache — every fresh `pip install`, before
  the model is fetched — the embedding model failed to download and the semantic layer
  silently degraded to graph-only (recall@10 ~0.05 instead of ~0.51). The offline-first
  loader set `HF_HUB_OFFLINE=1` as an env var, but `huggingface_hub` freezes that into a
  module constant at import, locking the process offline so the download fallback could
  never fire. Now uses fastembed's per-call `local_files_only=True` parameter: warm cache
  loads locally (zero network), cold cache downloads once. Caught by CI's cold cache — a
  warm dev machine could not reproduce it.

## [0.2.0] — 2026-07-07

The recall-and-sovereignty release. Retrieval went from a keyword-matching baseline to a
semantic-first hybrid, measured honestly on two separate corpora, with a benchmark harness
that has already caught the project's own bugs.

### Added
- **Semantic retrieval as core spine.** `sqlite-vec` + `fastembed` are now core
  dependencies. Local, on-device embeddings (`bge-small`, 384-dim) — still $0, still zero
  network on the default path. Graph-only recall remains available (`REVIEN_SEMANTIC=0`)
  but the degrade is now **loud**: a warning per recall and a `semantic_note` on every
  response. `REVIEN_SEMANTIC=require` makes a broken layer fatal.
- **Obsidian vault support (second corpus, AND-not-OR).** `revien connect obsidian`,
  `revien sync-vault`, `revien distill-vault`. Ingest chunks notes by heading and
  transcribes `[[wikilinks]]` into graph edges; distill writes memory back into the vault
  as provenance-tagged markdown that never re-ingests itself.
- **Curated shield.** Human-authored vault claims can never be silently auto-superseded by
  machine-extracted ones — contradictions route to a review queue.
- **Benchmark instruments.** Per-query miss taxonomy (`never_extracted` / `no_anchors` /
  `walk_depth_miss` / `disconnected` / `filtered_out` / `outranked`), a ranking-knob sweep
  harness with a pristine-ingest cache, a dedicated vault eval, and a false-merge audit
  surface. Reproducible results JSONs in `results/`.
- **Entity normalization** (case, separators, possessives, leading articles) applied
  everywhere labels meet, plus curated-entity mention linking so a turn saying
  "atlas-server" attaches to the entity "Atlas Server".
- **Loud extractor fallback.** LLM extraction failures now escalate once per outage instead
  of scrolling past — the aggregate signal that a silent regex fallback had masked.
- Env-tunable scoring knobs; `semantic_active` / `semantic_note` on recall responses;
  content-listener hooks that keep embeddings in sync with node edits.

### Changed
- **Recency now scores content time** (`recorded_at`), not last-access time, so "recent"
  means recently *true*, not recently *touched*. Default half-life 7d → **365d**.
- **Frequency is usage-driven, not retrieval-driven.** `recall()` no longer touches its own
  results by default (`REVIEN_TOUCH_ON_RECALL` off) — the old self-reinforcing loop was a
  popularity signal masquerading as relevance. `mark_used()` feeds frequency now.
- **Verbatim turns are stored as ground truth** (`EXTRACTED`, confidence 1.0), matching the
  schema definition — previously the rule extractor stored them as `inferred`/0.5, which
  half-weighted the user's own words and put them on a decay path.

### Fixed
- **Recall latency: ~950ms → 85ms (p50).** The dominant cost was a training-loop bug that
  exported the entire signal history on *every* recall to attempt a training run that could
  never succeed. Also: single-pass graph walk, bulk node/edge queries, and SQL-side keyword
  search replace per-node round-trips and a full-table Python scan. The `<100ms` retrieval
  tests pass for the first time.
- Offline-first model loading — no metadata revalidation, no silent re-downloads, closing a
  real zero-network gap and a startup hang.

### Measured (reproducible, $0, 0 network calls)
- Conversational (LoCoMo, 1,986 QA): recall@10 **0.514**, MRR 0.323, p50 85ms.
- Vault (43 QA): recall@10 **0.884**; attachment rate 1.00 clean-label / 0.75 fragile.

## [0.1.0]

Initial release: graph-based memory engine, three-factor scoring, REST API, Claude Code /
file-watcher / OpenAI / LangChain / Ollama adapters.
