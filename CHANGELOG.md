# Changelog

All notable changes to Revien are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [Unreleased]

### Changed (read this: some memories lose their date)
- **Existing memories from Claude Code, Codex, every Obsidian note (the upgrade backfill cannot tell a frontmatter date from a file date, so all `obsidian` rows are labeled `mtime`), and the file watcher will no longer show a date.** Their stored `recorded_at` was a file modification time (when a file was last written), not when anything was said, so showing it as "when said" was false. They keep their `recorded_at` for ordering but are labeled `mtime` (or `unlabeled`/`undated` when no source can be derived) and render with no date and no date note. Nothing is deleted. Re-ingest does NOT restore a date for an unchanged keyed session (the keyed re-ingest is a no-op), and `sync-vault --full` adds a second context node beside the old one rather than relabeling it; a true date comes only from newly ingested content.
- **Session adapters (Claude Code, Codex) now stamp the first message's time, not the file mtime**, labeled `content` (`mtime` only when no message carries a time). Obsidian notes carry `content` only when the note has a frontmatter date, otherwise `mtime`.
- **Hermes and MCP `store` capture now stamp the capture time** (labeled `capture`) instead of leaving `recorded_at` empty. This changes recency scoring for those rows: they now rank as recent-as-of-capture instead of undated.
- **`recorded_at` on the API means "when it was said" only when `recorded_at_source` is `content`, `capture` or `import`.** For `mtime`, unlabeled, or null, treat it as unknown; consumers that render a date should gate on the source (`revien.dates.said_date`).
- **Every ingest path now declares an honest `timestamp_source`:** `sync-vault` passes the adapter's source through (a note without a frontmatter date is `mtime`), `revien ingest` and MCP stamp the capture time as `capture`, `POST /v1/ingest` accepts `timestamp_source` (a request that supplies its own `timestamp` and no source is `content`; with no timestamp at all nothing is stamped), importers are `content`, reconcile claims, skill ingest and skill proposals are `capture`.
- **Distilled vault notes no longer print a date for mtime or unlabeled memories** (the provenance line previously rendered `recorded_at` unconditionally).
- **`revien status` splits the old `unknown` bucket** into `undated` (no `recorded_at`) and `unlabeled` (`recorded_at` set, no source), and marks `mtime`/`unlabeled`/`undated` as `(date not shown)`.
- **Successor lookup on edit/delete runs only under `REVIEN_EMBED_CONTEXT=prev`**, and the previous/next-turn lookup now range-scans `idx_nodes_session (session_key, recorded_at, created_at)` (older databases drop and recreate the narrower index automatically on open).

### Added
- **Pre-existing rows are labeled with `recorded_at_source`; an unknown or mtime date is never shown as "when said".** A `user_version` 4 backfill (and standalone migration 004) derives the source from each row's origin; `said_date()` renders a date and the note only for content/capture/import, so unlabeled and mtime rows are undated; `revien status` counts nodes by source.
- **Recall results carry `recorded_at`.** Every recall result now has `recorded_at` (when the memory was said, ISO-8601 in the speaker's own UTC offset, or null when unknown) across the daemon, MCP, CLI and TOON; memory-context blocks show each memory's date as `[YYYY-MM-DD]` so a consuming model can resolve "yesterday" or "next month".
- **No relative age is shown to a reader or model.** Memory blocks show the absolute `[YYYY-MM-DD]` date of when a memory was said (`recorded_at`), never an age computed from ingest time (`created_at`); the date is omitted when unknown, and the Ollama memory-context line now shows it instead of a misleading "N days ago".
- **TOON recall gains `recorded_at` and `recorded_at_source` columns.** The tabular recall rows carry when each memory was said and where that time came from; `parse_recall` tolerates TOON from an older daemon that has neither column.
- **`REVIEN_EMBED_CONTEXT=prev` (opt-in).** Embeds each conversation turn together with the turn before it in the same session; the recipe is recorded in the store and a mismatch at open warns in `status()` and the recall `semantic_note`, not only on stderr. Editing or deleting a turn re-queues the next turn for embedding so forgotten text cannot stay findable through its successor's vector. New index `idx_nodes_session` backs the previous/next-turn lookup (created automatically on open).
- **The embedder is recorded in the store.** `semantic_meta` keeps `embed_model` and `embed_dim`. Opening with a different model warns (`revien reindex`); a dimension change degrades recall to graph-only with a clear note instead of failing silently, and `revien reindex` now recovers by rebuilding the vector table under the new dimension. `reindex` records the new recipe only when every batch succeeded. While it runs the store holds `embed_state=rebuilding` (the intended model/dim under separate keys); a failure part-way leaves `embed_state=partial`, a warning in `status()["warnings"]` and the recall `semantic_note` on every open until a full reindex succeeds, a loud stderr line, and `revien reindex` exits non-zero.
- **Readers are told what the bracketed date means.** The benchmark answerer prompt gains one rule (the bracketed date is when a memory was said; resolve "yesterday"/"last week"/"next month" against it) and the Hermes, Ollama and LangChain memory blocks carry a one-line note to the same effect when any memory is dated; `READER_CONTEXT` is now `dated-resolved`.
- **Importers — `revien import-chatgpt` / `import-claude` / `import-readwise`.**
  Batch-import a ChatGPT export, a Claude.ai export, or a Readwise
  highlights CSV through the same ingestion pipeline every live adapter
  uses (`revien/importers/`): deny list, context fence, dedup, and
  idempotent `ingest_key` refresh all apply, so re-running an unchanged
  export is a no-op. ChatGPT and Claude each become one unit per
  conversation; for ChatGPT the unit is the thread actually displayed
  (`current_node`'s parent chain back to root) — a message on an edited-
  away branch is never ingested. Readwise becomes one unit per highlight,
  linked to its book via a declared entity edge. Every command accepts
  `--dry-run` (parses and reports counts, writes nothing) and `--limit N`.
  Origin fields are stamped `chatgpt` / `claude` / `readwise` with
  `origin_source="import"`. Zero new dependencies — stdlib `zipfile`/
  `csv`/`json` only, and an export .zip is read straight out of the
  archive, never extracted to disk.
  - Known limits: (a) re-importing an EDITED conversation refreshes the
    verbatim text and adds new claims but does not retract claims extracted
    from the old text — the pipeline's keyed refresh is add-only; (b) an
    unclosed recall-marker (e.g. a truncated export, or a conversation that
    itself quotes Revien's memory-context marker) makes the context fence
    drop the whole remainder of that conversation's import unit from the
    marker line onward, not just the turn the marker appears in.
- **Benchmark: end-to-end LoCoMo LLM-judge track.** `revien_bench/judges.py` —
  a SEPARATE, never-blended binary CORRECT/WRONG accuracy score from an LLM
  comparing each predicted answer to the LoCoMo gold answer, alongside (not
  instead of) the official token-F1 metric. `runner.py --judge {f1|ollama:<m>|
  openai:<m>|openrouter:<m>|together:<m>|claude:<m>}` (default `f1` = no LLM
  judge, report shape unchanged). Mirrors the LLM-reader track's transport,
  once-to-stderr cloud disclosure, cost-ESTIMATE table, and frozen/sha256-gated
  prompt (`prompts/judge.txt`) exactly — same env vars, same egress honesty.
  `sovereignty.network_egress_zero` now judges the `--judge` spec the same way
  it already judged `--answerer`: a cloud judge FAILS and is named, even at 0
  measured calls; `ollama:<model>` is loopback-local and PASSes. Report gains
  `judge` (spec, model, prompt sha256, overall + per-category accuracy, judge
  errors, network_calls, cost_usd) and `reader` blocks, and `report.py` renders
  them as a separate "End-to-end QA (LLM judge)" table explicitly labeled "not
  comparable to the retrieval metrics above." Per-question rows gain
  `judge_correct`/`judge_error`; older checkpoints without these keys resume
  fine (tolerated as absent, not corrupting the aggregate).
- **Origin layer — every node now carries where it came from.** Four
  nullable columns on `nodes` (`revien/graph/schema.py`): `origin_runtime`
  (claude-code/codex/hermes/ollama/openai/langchain/obsidian/file/api/
  chatgpt/claude/readwise/None), `origin_source` (live/import/vault/watch/
  api/None), `project_key`, `session_key`. A shared pure function,
  `revien/graph/origin.py:derive_origin(source_id)`, maps each adapter's
  `source_id` convention (`claude-code:{project}:{session}`,
  `codex:{project}:{session}`, `hermes`, `openai:conversation:{id}`,
  `vault:{path}#{slug}`, `file:{name}`, `api:{url}`, ...) to the four
  fields; unrecognized prefixes come back all-`None`. Every adapter now
  sets these explicitly on `IngestionInput`; `revien/ingestion/pipeline.py`
  stamps them on every produced node (including `context` nodes) in the
  same loop that already stamps `source_modality`/`recorded_at`, falling
  back to `derive_origin(source_id)` when a caller omits them.
  Migration `revien/graph/migrations/003_origin_layer.py` adds the columns
  and three indexes (`idx_nodes_origin_runtime`, `idx_nodes_origin_source`,
  `idx_nodes_project`) and backfills every existing row where
  `origin_runtime IS NULL AND source_id != ''` via the same
  `derive_origin`; idempotent — a second run backfills 0 rows.
  `export_graph`/`import_graph` round-trip all four fields.
  Honesty note, stated plainly because it changed the plan for this leg:
  the adapters' `"adapter"` tag on `IngestionInput.metadata` never reached
  a stored node — `pipeline.py` never reads `IngestionInput.metadata` at
  all (verified: zero references). Backfilling origin from that metadata
  was never possible; the migration and the pipeline fallback both derive
  origin from `source_id` instead, and an in-place guard in
  `store._ensure_db()` runs the same backfill for any database that skips
  the numbered-migration path.
- **`--source` filter on recall, status, the daemon, and TOON.**
  `store.list_nodes` gains `origin_runtime` (str or list) and
  `project_key` filters as a SQL prefilter (not a post-filter on the
  candidate set); the graph-walk's candidate fetch takes the same filter,
  closing a leak where a walk could pull in a node from a runtime the
  caller had just filtered out. `engine.recall(source=...)` filters
  candidates by `origin_runtime`; every `RetrievalResult` now carries
  `origin_runtime`, `origin_source`, `project_key` (`None` allowed, key
  always present so TOON's tabular reshape stays uniform across results).
  `POST /v1/recall` accepts `source` (str or list); `GET /v1/nodes` gains
  `origin_runtime` (repeatable) and `project_key` query params; the recall
  response's `results` and `toon.py`'s `serialize_recall`/`parse_recall`
  carry and round-trip the three fields. CLI: `revien recall --source
  claude-code` (repeatable), `revien recall` prints a runtime column,
  `revien status` prints a per-runtime node-count table.
- **`revien token` + file-backed pairing auth.** `revien/pairing.py`:
  `mint_token()` (32 bytes, `secrets.token_urlsafe`, written 0o600 where
  the OS honors it) at `$REVIEN_HOME/pairing.token` (default
  `~/.revien/pairing.token`); `configured_token()` resolves
  `REVIEN_CAPTURE_TOKEN` (env) first, then the file, then `None`. `revien
  token` prints the token, minting one if absent; `--rotate` mints a
  replacement; `--path` prints only the file path. The full token prints
  exactly once per invocation — never partial, never masked.
  `daemon/server.py`'s `check_capture_auth` now resolves the token through
  `pairing.configured_token()` instead of reading the env directly, so a
  remote caller can pair via the minted file with no env var set. A new
  `require_mutation_auth(client_host, auth_header)` — same rule, separate
  name — gates the skill-mutation endpoints below. Loopback stays trusted
  unconditionally; a remote caller with no token configured gets 403, a
  remote caller with the wrong token gets 401. `revien status` prints
  `pairing token: set (env)` / `set (file)` / `not set`.
- **`revien skills ingest` / `list` / `show`.** A new `NodeType.SKILL`
  node stores a `SKILL.md`'s body verbatim (frontmatter stripped) via a
  minimal non-YAML parser in the `obsidian.py` style
  (`revien/skills/frontmatter.py` — name/description/triggers/version,
  inline or block list; no pyyaml). `revien skills ingest [--path P]...
  [--global]` scans project skill folders (default `./.claude/skills`,
  `./.codex/skills`) or, with `--global`, the global homes
  (`~/.claude/skills`, `~/.codex/skills`, `~/.hermes/skills`) and is
  idempotent by path — re-ingesting an unchanged skill refreshes its node
  in place rather than duplicating it. `origin_runtime` is set from which
  root a skill was found under (`.claude` → claude-code, `.codex` → codex,
  `.hermes` → hermes, else `None`); `origin_source` is `vault`; a project
  skill's `project_key` is its containing project folder's name, a global
  skill's is `None`. `[[wikilink]]`s and trigger words in the body get
  best-effort `RELATED_TO` edges to existing `ENTITY`/`TOPIC` nodes by
  exact case-insensitive label match — no new entities are created.
  `revien skills list [--project P] [--status S] [--format json|toon]` and
  `revien skills show <name>` (prints the body). A user-authored skill is
  `origin=user`, `curated=True`, and sorts before any engine-origin skill
  of the same name everywhere skills are listed or recalled.
- **Skill proposals — the engine drafts a skill from what you actually
  did.** `revien/skills/proposals.py:detect_repeated_sequences` groups
  live `ACTION` nodes by `(project_key, session_key)` (or, when
  `session_key` is `None`, by `(project_key, recorded_at date)` as a
  time-gap fallback), normalizes each label (lowercase, strip punctuation,
  drop leading articles/pronouns, collapse whitespace), and slides 2-4
  gram windows over the ordered sequence to find a step sequence that
  repeats. Detection thresholds are fixed for this release —
  **3 occurrences across at least 2 distinct sessions** — not adaptive;
  adaptive per-user thresholds are v0.5. A qualifying pattern becomes
  exactly one `SKILL` node (`propose_skills`, idempotent by
  `pattern_hash` = sha256 of the normalized steps): `metadata.origin =
  "engine"`, `status = "proposed"`, `confidence = 0.5`, `source_type =
  INFERRED`, with a `DERIVED_FROM` edge to every source `ACTION` node it
  was built from. When an LLM extractor is configured
  (`REVIEN_EXTRACTOR != rule-based`) the body is drafted through it and
  `draft = True`; the rule-based path sets `draft = False` and only ever
  writes a numbered-steps skeleton. A proposal is never created with
  `status = "active"` — a human has to move it there.
  `revien skills accept <node_id>` sets `status = "active"` (origin stays
  `engine`), `revien skills decline <node_id>` increments `declines`; the
  third decline soft-invalidates the node via
  `GraphOperations.invalidate_node`. Both audit before/after snapshots
  (`skill_accept` / `skill_decline`); both take exactly one node id, no
  `--all`. A same-name proposal never overwrites a user-authored skill —
  not because of the `curated` flag (no gate reads it for skills today;
  SKILL nodes never enter the ClaimGovernor's supersession candidates),
  but structurally: every proposal's label carries a "proposed: " prefix
  no hand-written skill would use, and dedup only ever compares same-type
  nodes. `GET /v1/skills` (filters: status, origin, project_key), `GET
  /v1/skills/{node_id}`, and the two mutating routes — `POST
  /v1/skills/{node_id}/accept` / `.../decline` — gated by
  `require_mutation_auth`. Recall's response gains `skill_proposals`
  (always present, may be `[]`): proposals with `status = "proposed"`,
  `draft = True`, not invalidated, whose normalized step labels share at
  least one keyword (length ≥ 4) with the query — a rule-drafted skeleton
  proposal (`draft = False`) never surfaces there, only in `revien skills
  list`. Same shape in JSON and TOON; TOON carries it as a separate
  top-level tabular array and `parse_recall` round-trips it, present as
  `[]` when empty.
- **BM25 lexical lane (REVIEN_LEXICAL=bm25) + entity-anchor union (P1
  follow-up)** — `revien/retrieval/bm25.py` is a production-validated
  overlay ported near-verbatim from a pre-0.3.0 engine where it measured
  recall@10 0.5814 -> 0.6395 (REVIEN_HYBRID=rrf, live graph): a pure-stdlib
  Okapi BM25 ranker over the same node label+content corpus the shipped
  substring keyword lane scans. Where that lane treats every keyword hit
  the same (a document repeating a common word looks as relevant as the
  one document holding the rare, query-specific term), BM25 scores term
  rarity (inverse document frequency) and saturating term frequency, so
  the genuinely distinctive document wins. `_lexical_candidates` dispatches
  both call sites that previously called `_keyword_search` directly — the
  RRF fusion list and the keyword-fallback anchor path — to whichever lane
  `REVIEN_LEXICAL` selects; unset (or anything but `bm25`) is the shipped
  keyword lane, byte-identical. BM25 scores compose with semantic
  similarity by MAX (not sum) at the same seam the overlay used, so a node
  with both signals isn't double-counted; `score_breakdown["bm25_score"]`
  only appears when the lane is actually selected, same pattern as
  `semantic_sim`.
  Also lands the P1 regression fix this lane's production deployment
  exposed: under `REVIEN_HYBRID=rrf`, entity anchors (`_find_anchors`,
  alias expansion included) used to be REPLACED wholesale by the RRF-fused
  keyword/semantic candidate list — measured +65 disconnected results on
  the eval, because any node reachable ONLY through an entity match (never
  surfacing in either fusion list) stopped seeding the walk at all. Entity
  anchors are now PREPENDED onto the fused set unconditionally under
  `REVIEN_HYBRID=rrf` (no separate flag — this corrects a known
  regression, not a new experiment), and because `_find_anchors` already
  runs its `REVIEN_ALIAS` one-hop `ALIAS_OF` expansion before this union,
  an alias-expanded anchor now survives into the RRF path exactly like it
  does on the shipped path. The prepend is a real ordering effect, not
  cosmetic: `diagnostics["anchors"]["all"]` now lists entity anchors
  (deduped, keeping first occurrence) ahead of the fused list, and — since
  the walker seeds every anchor at distance 0 with its own path entry —
  can change which anchor's label leads a shared result's `path` when a
  node is reachable from both an entity anchor and a fused one. Both
  defaults stay off: unset `REVIEN_LEXICAL` is the exact keyword lane;
  unset `REVIEN_HYBRID` is the exact shipped anchor path.
  Honesty notes from review: (1) this port caps the RRF path's lexical
  candidate list at `semantic_top_k`, where the production overlay this
  was validated against ran that list uncapped — the quoted recall@10
  0.5814 -> 0.6395 numbers come from a slightly different configuration;
  a LoCoMo cross-check of the capped shape is still pending, not done.
  (2) `_bm25_candidates` reintroduces the exact
  `list_nodes(limit=999999)`-then-scan-in-Python shape OPEN 2 (see
  `store.py`'s `search_nodes_keyword`) moved OFF of and into SQL, because
  BM25's document-frequency/average-length stats need the whole corpus's
  tokens, not a pre-filtered slice — measured ~2.2x recall latency at 4k
  nodes vs the keyword lane's SQL-side scan when `REVIEN_LEXICAL=bm25` is
  selected; unselected, the cost isn't paid. (3) `REVIEN_RRF_K` stays
  unvalidated by design — a malformed or non-positive value silently falls
  back to the default (60.0), matching every other env-float ranking knob
  in this class; it is never allowed to raise and crash `recall()`.
- **Evidence-backed alias resolution (alias leg)** — "Sam", "Sam R.", and
  "sam@..." land as three separate ENTITY nodes (extraction has no way to
  know they're one person), and a recall anchored to one of them used to
  never reach the others. A new `ALIAS_OF` edge type (`revien/graph/schema.py`)
  joins surface-form variants and conceptual synonyms without ever merging
  the underlying nodes — reversible, audited, evidence-only. `revien/alias.py`
  is the new inference pass: BLOCKED candidate generation (shared normalized
  token, or mutual top-K label-embedding neighbors — never all-pairs),
  precision-first scoring (name_form needs co-occurrence or embedding
  corroboration; conceptual needs BOTH high embedding similarity AND
  co-occurrence), hard guards against aliasing conflicting or
  differently-typed nodes, and idempotent re-runs. Opt-in via
  `revien dream --alias` / `Consolidator.run(alias=True)` / `POST
  /v1/consolidate {"alias": true}` — off by default because, unlike the
  other dream passes, it writes new edges. At recall, `_find_anchors`
  unions in each anchor's LIVE `ALIAS_OF` neighbors, one hop only (no
  transitive chaining — that compounds false-positive risk multiplicatively
  for a gap one hop already closes); gated by `REVIEN_ALIAS` (default on).
  `revien aliases` lists live alias pairs with their evidence;
  `revien aliases --remove <edge_id>` reverses one (soft-invalidate,
  audited, never deleted). `POST /v1/edges` also accepts `alias_of` directly
  for manually-declared aliases (routed through `add_edge_audited` too, so a
  manually-declared alias carries the same create-audit row an inferred one
  does). New audited edge-mutation path (`store.add_edge_audited`,
  `store.update_edge`, `GraphOperations.invalidate_edge`) backing all of
  this — the existing (unaudited) `store.add_edge` is unchanged for its
  other callers.
  Coherent semantics after adversarial review: the graph walk
  (`get_neighbors_bulk` / `get_neighbors_weighted_bulk`) now excludes
  soft-invalidated edges for every edge type, not just ALIAS_OF — an
  invalidated edge never routes a walk again, so `revien aliases --remove`
  actually kills that pair's connection everywhere, not just at the anchor
  step. `REVIEN_ALIAS=0` disables anchor-expansion (distance-0 seeding)
  only — a still-LIVE alias edge legitimately continues to route ordinary
  graph walks at whatever hop distance it sits. Community detection
  (`clustering.py`) excludes ALIAS_OF edges entirely (a recall-routing edge
  reshaping topic communities was never the intent) plus any invalidated
  edge. name_form scoring tightened: the SUBSET shape — one label's
  normalized token set a strict subset of the other's, at ANY token count
  ("sam" is a subset of "sam r", but just as much "new york" of "new york
  times" or "ford" of "ford foundation") — can no longer be aliased on
  co-occurrence alone regardless of how much of it exists, because a
  qualified superset is usually a DIFFERENT entity that merely shares the
  shorter name (a first pass only caught the single-token case, missing
  "New York" / "New York Times"-shaped false positives — fixed to cover
  every token count). Only a real label-embedding similarity draws that
  shape now; every other name_form pair's co-occurrence bar rose from >=1
  to >=2 distinct shared neighbors, matching conceptual's bar.
- `action` node type — committed future work (to-dos, follow-ups, "I'll X"), DECISION's forward-looking sibling. Extracted by both the rule extractor (conservative commitment patterns) and the LLM extractor, and distilled to an "Actions" section in vault notes.
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
- **Context fence (leg 6c)** — recall re-entry no longer becomes new memory. Every
  ingestion route eventually calls `pipeline.ingest()`, and several of them inject
  Revien's own recalled memory back into the text they then hand to that same
  pipeline: Claude Code's harness-wrapped `<system-reminder>` blocks, ollama_adapter's
  `[Revien Memory Context]` fence, hermes_provider's `## Relevant memory (Revien)`
  header, langchain_adapter's `## Relevant Context (from N nodes)` block. Left alone,
  the graph re-learns what it already told you, with confidence compounding on each
  loop. `revien/ingestion/fence.py` strips exactly those marker-bounded spans —
  case-sensitive, pairing-based, no JSON/schema sniffing — before the ingest_key hash
  and before extraction. `REVIEN_FENCE` is ON by default (`REVIEN_FENCE=0` restores
  pre-fence behavior byte-identically); stripped spans are logged with source_id,
  count, chars, and which marker families fired, and content that fences down to
  nothing is skipped rather than stored as an empty husk.

### Fixed
- **Auto-sync fires immediately at daemon startup**, then every interval — no more
  silent 6-hour wait before the first sync of connected adapters.
- **Pre-existing duplicate session contexts stop growing.** The first keyed ingest of a
  session that was synced before this release ADOPTS the newest of its old unkeyed
  context nodes (stamps and refreshes it) instead of appending yet another copy. Known
  limitation: the older historical duplicates remain in the graph — retroactive merge
  is out of scope here; the consolidate (dream) pass is the future home for that cleanup.
- **A date is shown as "when it was said" only when it is.** Session adapters (Claude Code, Codex) stamp the earliest message timestamp in the file instead of the file's mtime; mtime-derived times (file watcher, session files with no message timestamps) are labelled `recorded_at_source=mtime`, never rendered as a date, and never trigger the "dates in brackets" note. A keyed refresh from an mtime source keeps the earlier `recorded_at`. `IngestionInput.timestamp_source` (`content` | `capture` | `mtime` | `import`) is stamped on nodes and carried on recall results (`recorded_at_source`). Rows ingested before the source existed keep their old rendering.
- **The speaker's calendar day is kept.** `recorded_at` is serialized in its own UTC offset (naive = UTC), so a 9:30pm-EDT message renders as that day everywhere.
- **Hermes and MCP stores carry a capture time** (`capture` source) when the caller gives none.
- **The context fence strips only the exact "dates in brackets" note Revien emits,** not any user line that starts the same way.
- **Benchmark layer status is read live on db-cache hits** (a dead embedder at recall time now exits 3 / is labelled, instead of reporting the snapshot's status); the checkpoint fingerprint covers the benchmark's own source; the cache compares embedder dimension.


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
