# Changelog

All notable changes to Revien are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic versioning.

## [Unreleased]

### Added
- **`revien sync reset <adapter>` — re-read a connected adapter's full history.**
  Resets that adapter's persisted sync cursor (`GraphStore.reset_sync_cursor`),
  so the next sync starts at EPOCH; the change is picked up at the daemon's
  next sync (default every 6 hours). It deletes nothing from the graph. Only
  adapters whose items all carry a stable `ingest_key` (claude-code, codex;
  declared by `IDEMPOTENT_REREAD` on the adapter class, and checked against
  what each adapter actually emits in `tests/test_sync_reset.py`, including a
  codex rollout with no known project) can be reset: re-reading a session
  refreshes its one context node instead of duplicating it. Other adapters
  (file watcher, Obsidian, generic API) are refused with exit 1, because a
  full re-read would add everything again as new nodes; a config entry with a
  missing or unknown `type` is refused as a config error instead. The name
  must be a connected adapter, unless `--orphaned` is given (below); `usage:*`
  positions are refused with a pointer to `revien usage scan --full`. All of
  these refusals happen before the database is opened, so a mistyped name
  does not write the current schema into an older Revien database.
  `--orphaned` resets a saved position whose adapter is no longer in the
  config. It works only when a `sync_cursors` row exists for that name (and is
  refused for a connected adapter); the adapter's type cannot be checked, and
  the command says so.
  Known and unchanged: keying is per project + file stem, so same-stem files
  in one project collide; and nodes extracted from an edited session are not
  refreshed (only its context node is).
  A missing, empty, directory, non-SQLite, locked or non-Revien `--db` gives
  a one-line error and exit 1. A foreign file is recognised read-only: it
  must have Revien's `nodes`, `edges` and `audit_log` tables with every
  column the oldest real Revien database (commit 7757085) has (`sync_cursors`,
  added later, is checked when present). The file is opened read-only and
  creates no `-wal`/`-shm` sidecar (immutable when there is no `-wal` file,
  read-only when one exists, since recent data lives there); tests assert
  the bytes and directory listing are unchanged for foreign WAL-mode files.
- **Reset/sync generation fencing.** `sync_cursors` gains a `generation`
  column (added in place to existing databases). A reset keeps the row, blanks
  the cursor and bumps the generation. The scheduler reads the generation
  before it fetches and saves its cursor only if it is unchanged
  (`set_sync_cursor_if_generation`, a single SQL compare-and-set, so it also
  fences a reset made from another process). A sync that was running when you
  reset finishes with status `reset_during_sync` and its newer position is
  discarded; the reset stands. Tested by interleaving sync start, reset and
  sync finish. Limit: a sync that was already running may still have ingested
  items (harmless, ingestion refreshes in place).
- **Suppression tombstones: forgetting a session stops sync recapturing it.**
  A new content-free table `ingest_suppressions` holds one row per stable
  source identity (`identity`, `kind`, `created_at`, `reason`, `actor`, plus a
  skip counter). It is written, in the same transaction as the delete, by an
  explicit user forget (`forget_node` / `POST /v1/nodes/{id}/forget`, with
  any reason text) and by `DELETE /v1/nodes/{id}`. Automatic cleanup (retention
  expiry) writes none: it is identified by an internal argument (`suppress=False`,
  passed by `apply_retention`), never by the reason string, so a user who types
  `retention_expire` as a reason still gets a tombstone and is counted. That
  argument is not itself exposed, but `POST /v1/retention/sweep` (caller-chosen
  `mode` and `days`) and `REVIEN_RETENTION=expire` reach `apply_retention`, so
  an expiry requested through the REST sweep writes no tombstone either.
  OPEN OWNER DECISION: whether a caller-requested expiry should write
  tombstones. Ordinary ingestion of a suppressed identity
  (adapter sync, MCP store, REST ingest, `revien ingest`) is skipped, printed,
  counted (`ingest_suppressions.skipped_count`, the sync result's
  `items_suppressed`, `suppressed` in the REST/MCP ingest response) and never
  silent. `revien suppressions` lists them; `revien restore <identity>` (and
  `POST /v1/suppressions/restore`) lifts one, audited. `revien import` and
  `POST /v1/graph/import` skip nodes of suppressed sources and count them
  unless `--override-suppressions` / `override_suppressions=true`, the
  explicit reimport, which lifts them (audited). A restored session is read
  again when its file next changes or after `revien sync reset`.
  Identity: the node's `source_id`, its context node's `ingest_key`, and for
  Claude Code and Codex sessions the runtime-assigned session UUID
  (`claude-code-session:<uuid>`, `codex-session:<uuid>`), which does not depend
  on the project folder or the dated Codex folder, so a session moved to
  another project, or whose Codex project (`session_meta.cwd`) was renamed,
  stays suppressed. The UUID identity is used only when the session file stem
  really contains a UUID. Tombstones written by the previous build get the
  session row derived from the identity they already hold, when the database
  is opened. `revien restore` lifts all rows of one unit together (for a watched
  file, see the file-path bound under Known gaps).
  Owner decision (confirmed for now): forgetting ONE extracted node
  suppresses that node's whole source session, because the node's content
  cannot be matched without storing it. The forgotten node itself is gone
  (deleted, its audit snapshots blanked, absent from recall and every
  export); SIBLING nodes already captured from that session remain, including
  the session's verbatim context (transcript) node, which still holds the
  original text unless it is forgotten too. Every forget/delete response
  (`suppression_notice`, `context_nodes_remaining`) and `revien suppressions`
  say this: "This forgets this node and suppresses all future imports from
  <identity>. Sibling facts already captured from that source remain." (A
  whole-session cascade from the context node says "Removed:" and "Kept:"
  instead, below.)
  Only source ids that name one unit are tombstoned (`claude-code:`, `codex:`,
  `openai:conversation:`, `file:`, `vault:`, `skill:`, `skill-proposal:`); shared channel ids
  ("mcp", "hermes", "ollama_*", "langchain", `api:<feed>`) are not, because
  that would block everything the channel sends. For those the response says
  plainly: "Deleted now, but eligible for recapture because the source supplied
  no stable item identity" (`recapture_eligible`, `suppression_unavailable`).
  This is an acknowledged failure mode, not a completed right-to-forget (see
  Known gaps). The identity string itself (a project, file or skill path)
  remains in the table; the content does not.
  The `sync reset` forget warning now counts only what is provably not
  covered (`count_unprotected_forgets`: forgets recorded before this change,
  or of a source with no stable identity) and needs `--yes` for those.
- **The OpenAI importer and skill ingest are gated too.** Both write through the
  store, not the pipeline, so each applies the tombstone check itself: a
  forgotten `openai:conversation:<id>` is skipped on single and bulk import
  (`suppressed` / `conversations_suppressed`), and a forgotten skill is skipped
  by `revien skills ingest` (counted, reported).
- **The egress gate pins the one module that opens a connection, by hash.**
  The `revien sync reset` loopback probe now lives in its own module,
  `revien/loopback_probe.py` (about 35 lines; `cli._probe_daemon` only calls
  it). `scripts/gates/check_egress.py` pins the SHA-256 of that WHOLE FILE
  (`PINNED`): any edit to it, a decorator, a default argument, a nested
  function, a second `def`, a lambda, an added import, even a comment, fails
  the gate on every run until the hash is changed on purpose, in a commit
  that also keeps `test_probe_egress_is_loopback_only` passing (it records
  every `http.client` connection, DNS lookup and socket connect and fails if
  one names a non-loopback host). Every OTHER `revien/**/*.py` file changed
  against `main` is parsed (with its `main` version) and any NEW
  network-client use in it fails, wherever it is. The scan reports: every
  name of a multi-name import, `import http.client`, `from http import
  client`, a parent package used through an attribute or an alias made by
  assignment (`h = http`, chains included), `sys.modules[...]` / `.get(...)`,
  `_socket`, `ssl`, `socketserver`, `asyncio.open_connection` and its
  siblings, `subprocess`, `ctypes`, `os.system`/`os.popen`/`exec*`, the
  dynamic-import facility (`__import__`, `importlib.import_module`,
  `getattr`/`vars` on `importlib`, `builtins` or `sys`, a bare reference to
  either, `exec`, `eval`, `__builtins__`: a literal argument names the module,
  anything else is `<dynamic-import>`), and the clients `httpx`, `requests`,
  `urllib.request`, `urllib3`, `aiohttp`, `ftplib`, `smtplib`, `poplib`,
  `imaplib`, `telnetlib`, `xmlrpc.client`, `websockets`, `websocket`,
  `pycurl`, `grpc`, `paramiko`, `boto3`, `botocore`, `anthropic`, `openai`.
  The gate runs its own self-tests on every run (positive and negative
  snippets, that each edit shape above to the pinned file is rejected, and
  that every documented limit still evades). Limits: the shapes known to
  evade the scan, each with a snippet the self-test runs and requires to
  stay unreported (so an entry cannot go stale). The scan is a heuristic, not
  a proof: this list is what has been shown to evade it, not a claim that
  nothing else can:
  - alias through a container, a call or a parameter
  - a method of an object whose type the scanner cannot see
  - a network client the NETWORK_MODULES list does not name
  - a module that is not imported where it is used, reached through a variable set elsewhere
  What the gate does not look at:
  - non-Python files (a .pth file, a shell script, a build hook) and compiled extension modules
  - code inside third-party packages the listed clients pull in
  - any file unchanged against `main` (the scan covers the diff, not the tree), and a repository with no `main` to diff against
  - a runtime effect of data (a URL a config file supplies to an already allowed client): the gate looks for clients, not for destinations
  The gate prints both lists when it passes. (Earlier versions keyed an
  allowlist by file and function; a connection added inside the probe after
  its check, in a decorator or in a default argument was therefore not seen.)
- **Stdio MCP is no longer corrupted by ingest notices.** The suppression,
  deny-list and context-fence notices, and the migration note, go to stderr;
  stdout carries only JSON-RPC under `revien mcp`. Tested against the real
  stdio server process.
- **`revien sync reset` says only what it can prove about a running daemon.**
  `/v1/health` now reports `features` and a `db_id`: a random per-database id
  created once and kept in `store_meta` (`GraphStore.database_id()`). It is not
  derived from the path or the contents, so it reveals neither; an earlier
  draft hashed the path, which a dictionary of likely paths recovers in under
  a second. A copy of the database file carries the same id. The command probes `--daemon-url` (default
  `http://127.0.0.1:7437`): a current daemon on this database is reported as
  protected against the reset being undone; a daemon that does not report the
  features is reported as started before this version (it ignores suppressions
  and the reset fence: restart it); nothing answering is reported as "nothing
  was checked". Older daemons cannot be made to refuse writes from outside the
  process, so restarting any daemon started before the upgrade is required.
- **Privacy fix: forgotten and deleted content survived outside the node row.**
  `GraphStore.delete_node` (the one path under forget, `DELETE /v1/nodes`
  and LangChain `clear()`; `revien import --replace` applies the same audit
  scrub to the graph it replaces) now also: blanks the audit snapshots of the
  node AND of every edge attached to it (ALIAS_OF audit rows carry the endpoint
  labels in `source_context`; op, actor and time are kept); deletes
  supersession candidates that name the node (their reason/trace quote claims),
  the node's distill-manifest hash rows, its embedding-queue row and its vector
  row (sqlite-vec is loaded on demand, so a CLI or plain store delete removes
  it too); runs with `PRAGMA secure_delete` so deleted pages are zeroed; and
  truncates the WAL after a forget/delete. The neural signal log
  (`retrieval_signals`, node labels) is purged in processes running the
  engine with neural logging on.
  The query text of a recall (the user's own words, which can repeat the
  forgotten fact) is purged too. A receipt that surfaced the forgotten node
  has its `query` blanked (the row stays, so an expired flag is not lost). In
  the neural `training.db`, every signal row of each logged recall that
  surfaced the node is deleted, not only the node's own row: a signal with its
  query blanked is no longer training data, and the other rows of that recall
  carry the same query. In the usage ledger (`use_judgments`) the judgment
  rows of the forgotten node are deleted AND the `query` of every other row
  of the same episode or recall is blanked (the other nodes' own evidence
  stays). `training.db` now sets `secure_delete` on every connection and, once
  per file, rewrites it (`VACUUM`): the first open of a file without the
  marker `secure_scrub_v1` in its `training_state` table, whatever its free
  list holds, then records the marker. The free list is not the test: a
  delete that leaves other rows on a page frees no page, so the page keeps the
  deleted row's bytes and `freelist_count` is 0 (an earlier version vacuumed
  only a file with free pages and left that residue). A file written by an
  older build is cleaned on its first open by this version; after that,
  deletes zero what they remove. Bounded gap: a recall
  whose query repeated the fact but did not return the node cannot be linked
  to it without reading the text, and is kept.
  Tests: one seeds a unique secret, forgets everything, then scans every
  table, the raw database and WAL bytes, `/v1/export`, `/v1/graph`,
  `export_everything`, `revien export`, recall and TOON output. A second does
  a SINGLE-node forget and says plainly what remains: the session's verbatim
  context node (and near-duplicate sibling claims) keep the text, a pending
  owner decision, and every other place the text appears is that node's own
  row or audit snapshot; the forgotten node's rows are gone everywhere. A
  third scans the raw bytes of the database and `training.db` for a query.
  One-time migration on open: audit rows still holding the content of nodes
  deleted by EARLIER versions are blanked, narrowly (only rows shaped like a
  node snapshot whose node no longer exists, or an edge snapshot with a missing
  endpoint; forget/suppress rows and rows of live nodes and edges are never
  touched; nothing is deleted). Recorded in `store_meta`, so it runs once.
  Limit: content that left the database cannot be recalled by it (see Known
  gaps: vault notes, backups, copies of earlier exports).

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
- **Usage signal (N1) — did an answer actually draw on what recall handed it?**
  `mark_used` existed (access count +1, an "access" audit row, +0.05 on the
  node's edges) but its only caller was `POST /v1/mark_used`: the MCP server
  exposes only `revien_recall` and `revien_store`, nothing produced the signal,
  and the frequency term counted dedup hits instead of use. New
  `revien/usage/` closes that loop for Claude Code. `episodes.py` reads each
  session transcript for `revien_recall` calls (bare or
  `mcp__<server>__revien_recall`), the memories each returned, and the
  assistant's answer that followed. An episode is emitted only once its turn has
  closed (`stop_reason: "end_turn"`, or a new human prompt), so an answer still
  being written is never judged. Thinking blocks, tool inputs, `isMeta` and
  sidechain entries are never part of the answer. `evidence.py` gathers two
  independent signals per memory: shared distinctive terms, with the prompt's
  and query's own words removed first so restating the question proves
  nothing; and the best cosine between the memory and any answer paragraph,
  from the configured embedder (local by default), or `None` when the
  semantic layer is off. `judge.py:judge_use` is the rule: both signals must
  agree, it abstains when one is missing, and it is precision-first because a
  false "used" compounds. Judgments land in a new `use_judgments` ledger, one
  row per (episode, memory), claimed by `INSERT OR IGNORE`. That makes rescans
  of a grown session no-ops, and two processes can never count one use twice;
  abstentions are not written. `shared_terms` holds the memory's own words, so
  `delete_node`, and with it right-to-forget, removes that node's rows. The
  observer also skips any recalled memory whose node no longer exists, so a
  rescan can't copy forgotten words back in from the transcript.
  `REVIEN_USAGE` picks the mode:
  - `shadow` (default) records judgments only;
  - `apply` also calls `mark_used`;
  - `off` disables the pass.
  Unrecognized values fall back to shadow, so a typo never writes to the graph.
  `revien usage scan [--mode] [--full] [--path]` runs a pass with its own
  cursor (`usage:claude-code` in `sync_cursors`), and `revien usage list
  [--filter used|unused]` shows each judgment with its evidence. The daemon
  schedules the pass only when `REVIEN_USAGE_INTERVAL_HOURS` is set (opt-in,
  like dream mode). New `SemanticIndex.embed_texts()` embeds arbitrary text
  without touching the vector table. Starting bars in `judge.py`: at least
  2 shared terms (or all of a 1-2 word memory's) and similarity >= 0.75;
  it abstains when the memory has no distinctive words.
- **Usage receipts: the same judgment for any runtime, not only Claude
  Code.**
  - `POST /v1/recall` with `"receipt": true` (and an optional `"runtime"`
    name) returns a `recall_id` plus `receipt_expires_at`. The default
    response is unchanged.
  - The client hands the answer back to `POST /v1/usage/observe {recall_id,
    answer, prompt?}`. That call returns, for every memory on the receipt,
    its verdict (used / unused / abstain), shared terms, similarity, the
    rule's reason, and whether it was applied. Errors: 404 unknown receipt,
    409 already redeemed, 410 expired, 400 empty answer. Nothing is written
    on an error.
  - A receipt's row is in the new `recall_receipts` table. It holds ids, the
    runtime and the query (so restated words can be discounted), never memory
    content. Memories are judged as they are now, not as they were recalled.
  - A receipt not redeemed within 24 hours becomes a flag. Flags are never
    deleted: `revien status` shows a count, `revien usage flags` lists them
    and `--dismiss` marks one as seen, and `GET /v1/usage/flags` returns
    them. A judgment that fails leaves its receipt open, so it expires into a
    flag rather than vanishing. Forgetting a node removes its id from
    receipts and flags and blanks the query of a receipt that surfaced it.
  - Issuing a receipt and calling the usage endpoints take the pairing-token
    gate; plain recall stays ungated.
  - The Hermes provider issues a receipt at `prefetch` and redeems it with
    the reply in `sync_turn`, on the existing worker, so turns stay
    non-blocking. The Ollama adapter does the same inside `chat()`. Both run
    only when `REVIEN_USAGE` is explicitly `shadow` or `apply`; unset, they
    issue nothing.
  - Not covered yet: Codex sessions (whose real format needs verifying against
    a live session first), receipts on the MCP `revien_recall` tool (Claude
    Code is covered by its transcripts), and the activation-level probe
    (tier 2, local open-weight models).

### Fixed
- **The Claude Code adapter now reads real Claude Code sessions.** It read
  only a top-level `content` field. Real transcripts nest the text under
  `message.content`, so every real session parsed to nothing and the adapter
  fetched 0 items. Verified against a live transcript: 0 items before, the full
  conversation after. The test fixtures used a flat `{"type": "human",
  "content": ...}` shape Claude Code never writes, so the suite stayed green.
  The adapter now reads `message.content` (the flat shape still works) and
  skips two kinds of entry that aren't the human's words: `isMeta` entries
  (harness-injected skill bodies, command caveats, agent reports) and
  `isSidechain` entries. Sidechain entries make up every subagent transcript
  under `<session>/subagents/`, where the "user" is the parent agent, so those
  files ingest as nothing (an earlier probe fixture omitted the flag; Claude
  Code writes it). As defence in depth, a file whose own stem is not a session
  UUID (a sub-agent file, `agent-<id>.jsonl`) also records the `sessionId` its
  entries carry (a bare UUID, never text) as `parent_session_id`, and the
  forgotten parent session's tombstone matches it, so a sub-agent file that
  did ingest cannot recapture a forgotten session. The id is read from the
  first entry that has one, however far into the file (the file is streamed;
  an earlier version stopped after 200 lines, so a later `sessionId` was
  missed). A forget of the parent session with `cascade` also removes its
  sub-agent context nodes already ingested (those that name it as
  `parent_session_id`), and the forget response lists them
  (`subagent_contexts_forgotten`). Tested with realistic (sidechain) files;
  with the flag removed, ingesting a sub-agent file AFTER only the parent was
  forgotten (the earlier test also forgot the sub-agent's own unit, so
  removing the parent lookup did not fail it); a `sessionId` placed after 450
  other lines; and cascade of already-ingested sub-agent units.
  Limit: a sub-agent file ingested by a build before this one has no
  `parent_session_id` and is not recognised as part of its parent.
  `session_files(since)` and `source_id_for(path)` are
  now public, so anything else that reads sessions sees exactly the files
  ingest reads. Known limitation: an existing install's `claude-code` sync cursor has already
  moved past older sessions, so only sessions that change from now on are
  captured until that cursor is reset; run `revien sync reset claude-code`
  to have the next daemon sync re-read the full history.
- **Auto-sync fires immediately at daemon startup**, then every interval — no more
  silent 6-hour wait before the first sync of connected adapters.
- **Pre-existing duplicate session contexts stop growing.** The first keyed ingest of a
  session that was synced before this release ADOPTS the newest of its old unkeyed
  context nodes (stamps and refreshes it) instead of appending yet another copy. Known
  limitation: the older historical duplicates remain in the graph — retroactive merge
  is out of scope here; the consolidate (dream) pass is the future home for that cleanup.

### Known gaps / required follow-up
- **REQUIRED: stable per-item provenance for shared channels** (REST/MCP default
  ids, hermes, ollama, langchain, `api:<feed>`, especially API-fed items).
  Until sources supply a stable item identity, forgetting such an item deletes
  it now but leaves it eligible for recapture; the response says so. This is
  not a completed right-to-forget for those sources.
- Item-level stable ids should eventually narrow the blast radius of a
  single-node forget, which today suppresses the whole source session.
- Tombstones bind to identities that exist, not to content. A file-watcher
  file is bound to its base name AND, when its context node recorded where it
  was read from (`source_path`, the real path with symlinks resolved), to that
  path (a `file_path` row). The path row blocks that one file. The base-name
  row blocks another file of the same name only while a recorded path no
  longer exists on disk (the forgotten file may have moved there), so an
  unrelated `README.md` elsewhere is captured while the forgotten one is still
  in place. Bounds: after the forgotten file is deleted or moved, a new
  unrelated file with the same name is skipped until you restore the path
  (`revien restore <path>`, which lifts only that file and the name row when
  no other forgotten path of that name remains); a different file written to
  the forgotten path is skipped; `revien restore file:<name>` lifts the name
  and every path under it and says which; a tombstone made before this change,
  or from an extracted node of a name shared by several files, has no
  recorded path and stays name-wide (`revien suppressions` flags it
  "Name-wide"); a graph import has no path for extracted nodes, so they are
  skipped when their name is tombstoned; the path is stored in the
  suppression (as revealing as the name already was) and is listed by
  `revien suppressions`. A renamed file is a new name, a gap as before. A vault
  note is bound to its relative path and heading slug, a skill to its
  resolved path; renaming either (and moving a vault note or skill) creates a
  new identity and the content can be captured again. Content-free stable
  identities were evaluated and rejected: an inode is reused and changes on an
  editor's atomic save; a content hash is not content-free. Claude Code and
  Codex are covered by their session UUID; a session whose stem has no UUID is
  bound to its project folder, and a session the runtime copies into a NEW
  session id (resume/fork) is a different identity.
- Distilled Obsidian vault notes live outside the database: a forget removes
  the distill-manifest row but the note keeps the claim until the next
  `revien distill-vault`, which drops or rewrites it. The forget response now
  says so when a manifest referenced the node (`distilled_notes_remaining`, and
  the notice: "the next `revien distill-vault` removes it"), on `forget` and on
  `DELETE /v1/nodes/{id}` alike. Required
  follow-up: re-distill the configured vault as part of a forget.
- Forgetting an extracted node does not remove the session's verbatim context
  node, nor near-duplicate sibling claims; the response names the context
  node(s). OPEN OWNER DECISION (A): whether a single-node forget should also
  remove the text from the context node.
- **Forgetting a whole session in one call: cascade from its context node,
  without touching any other unit, decided by provenance written at write time.**
  The pipeline never writes `DERIVED_FROM`, so the first cascade (which
  follows only `DERIVED_FROM`) removed just the context node. Two later
  versions inferred "what the unit owns" afterwards from edges, `source_id`
  and a substring check, and each missed a writer's shape: OpenAI messages are
  chained by `FOLLOWED_BY` and only the root links the conversation; an entity
  stamped with the session but not linked; an LLM entity description is
  attested by its label only; a node re-attributed to an arbitrary unit
  outlived every unit that said its wording. So ownership is now recorded
  when the node is written.
  New table `node_provenance(node_id, unit_id, label_ok, wording_ok)`:
  content-free (ids and two booleans). `unit_id` is the context node of the
  unit that wrote a node or merged into it (or `~external`, a writer that is
  not a unit: a node added by hand that a unit later merged into, or a node
  that outlived a deliberate non-cascade forget of its unit). `label_ok`: the
  unit said the label (normalized-label equal). `wording_ok`: it said the
  stored wording; text Revien generates from the label ("Entity: X", "Topic:
  X (mentioned 3 times)") carries no wording, so there the label is enough.
  Writers: the ingestion pipeline records every node it creates or dedups
  into, in `ingest`, in the keyed refresh, for declared links and for
  known-entity mentions (the flags compare the unit's candidate with the
  node it was merged into); the OpenAI importer records every message;
  graph export/import carries the rows (`provenance` in the export; an export
  from an older build has none and gets them inferred on import). Skill
  ingest, skill proposals and the vault reconciler create nodes that are not a
  unit's text and write no rows; the first unit that merges into one records
  `~external` first, so forgetting that unit cannot take it (tested). A
  non-cascade forget of a context node drops its rows and gives the nodes it
  alone wrote the `~external` row, because the documented behaviour is that
  they stay.
  Existing databases are backfilled once on open (`store_meta` key
  `node_provenance_backfill_done`) from edges to context nodes, the
  `source_id` a node carries, and the live context text (label and, for
  decisions, actions, facts and preferences, the sentence found in it). How
  far to trust that: it recovers membership wherever an edge or a unit
  `source_id` survives, and its flags are only ever set from text it can see,
  so it never claims more than the text shows (a creator that is the only unit
  with its `source_id` is trusted); it cannot tell a node a unit created from
  a hand-added node the unit merged into except by `source_id`; an LLM
  description is never `wording_ok` there, so such a node loses its
  description when its first unit is forgotten. A node nothing links and no
  unit's `source_id` claims has no rows and is nobody's: a cascade never
  touches it. The same inference runs at cascade time for a node with no rows
  at all that the unit links or whose `source_id` it carries (a writer that
  does not record, an older build), so a missed writer is covered.
  The rule: the forgotten unit is gone, everything else remains.
  `cascade=true` on a session's context node (`POST /v1/nodes/{id}/forget`,
  `forget_node(..., cascade=True)`; `GET .../forget/preview` returns the same
  lists first) forgets: that context node; the Claude Code sub-agent context
  nodes that name the session as their parent; and every node that only these
  wrote (no other live writer has a row for it): `forgotten`, counted in the
  notice as "node(s) only this unit wrote". A node another live unit also
  wrote or merged into is kept (`kept_shared`, with the live units in
  `kept_shared_linked_from`), minus this unit's rows, and re-attributed
  (`source_id` and origin columns) to a unit that said its wording, so
  forgetting that unit later finds it. What stays is decided by what a live
  unit vouches for: one that said the wording keeps the node as it is; one
  that said only the label (an LLM description, or a sentence whose stored
  text differs from what that unit said) keeps the node with its content reset to the label
  (`reset_to_label`; the old wording is removed from the row, the embedding,
  the audit snapshots, the usage judgments and the supersession candidates
  that quote it); a node no live unit vouches for even by label (dedup also
  merges on edit distance 2 and, opt-in, on semantic similarity, and keeps the
  FIRST unit's wording) is removed with the unit (`removed_unattested_shared`)
  and the other unit's verbatim text is untouched. Another unit's context
  node is never deleted: not through a provenance row naming it, not through a
  `DERIVED_FROM` edge reaching it, not for sharing the `source_id`; those are
  listed (`other_contexts_same_source`). It applies only to per-unit sources;
  a context node of a shared channel (`mcp`, `api:<feed>`) is removed alone
  and the response says the facts extracted from it were kept
  (`extracted_facts_kept`), and cascade from an extracted node still follows
  `DERIVED_FROM` only.
  Tested: every order of forgetting four overlapping sessions keeps each live
  unit's context text and every node a live unit said in full, leaves no
  trace of a forgotten unit's unique token in any row or the raw database and
  WAL bytes, and leaves every remaining node with a row from a live writer.
  Measured with 12 overlapping sessions (28 nodes): across the 12 cascades,
  0 other contexts deleted, 0 links to nodes another session said in full
  lost, and 33 of 1111 other-session links removed, all to nodes merged by
  similarity and worded by the forgotten session; each is reported.
  Not covered (pending owner decisions A and B, unchanged): a single-node
  forget leaves the text in the session's context node, and the REST
  retention sweep writes no tombstones. An earlier file version of a watched
  file is a separate context node and is listed, not removed.
  The tombstone covers only the forgotten unit's identity. For a file-watcher
  file the identity is its path and name (above), so another file of the same
  name keeps being captured while the forgotten path exists; for a vault note
  whose `source_id` is shared with another note, the identity IS shared, the
  forget still writes it (skipping it would let the forgotten note be silently
  recaptured) and the notice names the other context nodes and says that
  future imports of every item with that identity are skipped until `revien
  restore`. The notice and response state what was removed ("Removed:", with
  counts taken from what was deleted), what was kept because another unit also
  vouches for it ("Kept:"), what was reset ("Reset:") and what was removed
  because no other unit vouches for it ("Also removed:"); the "Sibling facts
  already captured ... remain" sentence is only used for a single-node forget.
- Forget notices, corrected: forgetting an extracted node of a shared channel
  (`mcp`) no longer says that cascading from the context node "forgets the whole
  session" (it removes only that node; a channel is not a session); a cascade
  from a shared-channel or identity-less context node says the extracted facts
  were kept; `DELETE /v1/nodes/{id}` carries the distilled-note notice.
- LangChain `RevienMemory.clear()` writes no tombstone (decision: it is
  infrastructure bulk-clearing, i.e. cleanup; Revien does not infer intent from
  a method name). An application that exposes it as a user-facing "forget my
  memories" control must call the explicit forget semantics instead. Its audit
  snapshots are still scrubbed.
- Forgets and `DELETE`s made before this change have no tombstone and are not
  retroactively protected (their old audit snapshots ARE scrubbed by the
  one-time migration); `DELETE` of an unsuppressible source leaves no audit
  row, so the warning cannot count it.
- Suppressions apply to ingestion through the pipeline, the OpenAI importer,
  skill ingest, graph import and the skill proposer (gated on the proposal's
  stable `skill-proposal:<pattern hash>` identity, so a forgotten proposal is
  not recreated; its identity is a hash of the step labels, so a pattern with
  different steps is a new proposal, and the actions it was drawn from stay
  until forgotten). The only other store writers are the LangChain wrapper's
  edge links and the vault reconciler's user-authored claims; neither ingests
  external content.
- A daemon started before the upgrade ignores tombstones and the reset fence
  and can recapture content or undo a reset until it is restarted. `revien sync
  reset` detects one at `--daemon-url` and warns; it cannot stop one it cannot
  reach, and cannot detect one elsewhere. Restart every daemon after upgrading.
- `GET /v1/suppressions` is open like every other read (recall, `/v1/graph`,
  `/v1/export`); `/v1/export` already returns the same identities in its audit
  rows, so gating one route would protect nothing. Keep the daemon on loopback.
- Secure deletion is best effort: `secure_delete` and a WAL truncate cover the
  database file, but not filesystem snapshots, backups, earlier exports, the
  vault, or a WAL that another open connection prevents from checkpointing.

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
