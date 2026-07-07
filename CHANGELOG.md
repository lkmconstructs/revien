# Changelog

All notable changes to Revien are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

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
