# Hermes Agent Memory Provider — Scoping Note (LEG P5 item 5)

Researched 2026-07-12 against hermes-agent v0.18.2 (2026.7.7.2, released July 8, 2026). MIT license,
Python core. Release cadence: minor every 2–3 weeks, patches within days — active, fast-moving, pre-1.0.

## 1. The provider interface (it's real, public, and documented)

Hermes ships a pluggable external-memory system. Nine provider plugins already exist (Honcho, Mem0,
Supermemory, Hindsight, OpenViking, Holographic, RetainDB, ByteRover, Memori). Only **one** external
provider can be active at a time; the built-in memory (MEMORY.md / USER.md) stays active alongside it.
So the provider slot is a slot — Revien would compete head-to-head for THE memory backend, not sit in a pile.

The contract is a `MemoryProvider` ABC (`from agent.memory_provider import MemoryProvider`), documented at
the developer-guide URL below and in-repo at `website/docs/developer-guide/memory-provider-plugin.md`:

- **Required:** `name` (property), `is_available() -> bool` ("NO network calls" — config presence check),
  `initialize(session_id, **kwargs)` (once at startup), `get_config_schema()` / `save_config(values,
  hermes_home)` (drives `hermes memory setup`), `get_tool_schemas()` / `handle_tool_call(tool_name, args,
  **kwargs)` (tools injected after init).
- **Optional hooks (the automatic-memory part):** `prefetch(query, *, session_id="")` — recalled context
  before each turn, background/non-blocking; `sync_turn(user, assistant, *, session_id="", messages=None)` —
  persist the turn after each response, "MUST be non-blocking"; `on_session_end(messages)`;
  `on_pre_compress(messages)`; `on_memory_write(...)` (mirror built-in writes); `system_prompt_block()`.
- **Registration:** plugin dir `~/.hermes/plugins/<name>/` with `__init__.py` exposing
  `def register(ctx): ctx.register_memory_provider(MyProvider())`, plus `plugin.yaml` (name/version/
  description/hooks) and README. Pip distribution via entry-point group `hermes_agent.plugins`.
  Their AGENTS.md policy: third-party product integrations **must** ship as standalone plugin repos —
  a Revien provider would not land in their tree, which also means no upstream review gate on our cadence.
- **Lifecycle Hermes drives automatically:** prefetch relevant memories pre-turn, sync turns post-response,
  extract on session end, inject provider context into the system prompt.

Verdict on stability: documented, versioned, and load-bearing for nine existing providers — real API, but
pre-1.0 with a hot cadence. Keep any adapter thin and pinned-tested.

## 2. Two routes

**(a) Native provider** — standalone plugin implementing the ABC, backed by the daemon's REST:
`prefetch` → `POST /v1/recall`, `sync_turn` → `POST /v1/ingest` (P3's `defer_embed=true` is what makes
"MUST be non-blocking" honest on a cold model), tools `revien_recall` / `revien_store` via
`get_tool_schemas`, config schema = daemon URL + capture token. Effort: small — a few hundred lines of
HTTP adapter plus plugin.yaml/README/installer; the engine surface already exists. Maintenance: tracking a
pre-1.0 ABC, realistically a small bump every few Hermes minors. Buys: **ambient** memory (recall injected
every turn without the model choosing to ask; every turn persisted), the memory-provider slot, and a line
in their providers list next to Mem0 and Supermemory — distribution to exactly our crowd.

**(b) Hermes as MCP client** — Hermes consumes any MCP server: `mcp_servers.<name>` with `command:`/`url:`
in `config.yaml`; tools auto-registered alongside built-ins. Once P5's MCP server lands this route is a
paragraph of docs, zero new code, zero maintenance. But it buys tools only: recall happens when the model
decides to call it, nothing syncs turns, nothing fires at session end, nothing lands in the system prompt,
and Revien appears nowhere in their memory story. Voluntary memory, not memory.

These aren't competing — (b) is free and immediate, (a) is the actual product placement.

## 3. Recommendation: BUILD — sequenced after the MCP server (P5 item 1)

The API is public and proven by nine implementations; the adapter is thin because the daemon already speaks
the needed verbs; and the strategic fit is the best on the board — Hermes' audience is the local-model,
provider-agnostic, own-your-infra crowd, i.e. Revien's audience with an agent already in hand. The incumbent list
is dominated by hosted/SaaS memory (Honcho, Mem0, Supermemory are the marquee names); a single-SQLite-file,
zero-egress backend is a differentiated entry in that list, not a me-too. The single-active-provider
constraint cuts both ways: winning the slot displaces Mem0/Supermemory for that user entirely.

Risks, stated flat: pre-1.0 ABC churn (mitigate: thin adapter, pinned integration test, their own nine
providers create pressure against breaking changes); and `sync_turn` volume — per-turn ingest is a firehose
relative to curated capture, so the provider should ingest conversationally (`content_type="conversation"`,
`source_id="hermes"`) and lean on defer_embed rather than pretending every turn deserves a synchronous
embedding. Not a NO-BUILD on any axis. Sequencing only: route (b) ships for free with the MCP server doc;
route (a) is its own leg after P5 item 1 and P3's defer_embed both land.

## 4. The leg (post-P5-1, post-P3)

**LEG P6 — Hermes Memory Provider (native)**

**What:** A standalone Hermes plugin (`hermes-revien`) implementing `MemoryProvider` against the daemon's
REST surface, plus a `revien connect hermes` installer that drops it into `~/.hermes/plugins/revien/`.

**Scope:**
1. Provider class: `is_available` (config-file/env check, no network per their rule), `initialize`,
   `prefetch` → `/v1/recall` (top_n small, background), `sync_turn` → `/v1/ingest` with
   `defer_embed=true` on a daemon thread, `on_session_end` → one summary-grade ingest,
   `get_tool_schemas`/`handle_tool_call` for explicit `revien_recall`/`revien_store`,
   `get_config_schema`/`save_config` (daemon URL, `REVIEN_CAPTURE_TOKEN` as `secret: true`),
   `system_prompt_block` (one line: what Revien is, that recall is automatic).
2. `plugin.yaml`, README, packaging for both install paths (dir copy + `hermes_agent.plugins` entry point).
3. `revien connect hermes` — mirrors the codex installer pattern: copy plugin, print the
   `hermes memory setup` step.
4. Integration test against a real Hermes install: plant a memory, new session, prefetch surfaces it;
   run two turns, nodes present in the SQLite file; `hermes memory status` shows revien active.

**Close condition:** the integration round-trip above green against Hermes ≥ the version current at build
time; version pinned in the test; README states the tested Hermes version honestly.

**Not in scope:** implementing `on_pre_compress`/`on_memory_write` (later, if usage shows value);
Honcho-style dialectic user modeling; upstreaming into their providers doc (ask after it works — free
distribution, separate small follow-up); any engine changes.

**Price:** small-medium — one adapter class over existing endpoints, an installer clone, one real-client
integration test. The MCP server leg is bigger; this rides behind it.

## Sources

- https://github.com/NousResearch/hermes-agent (repo, v0.18.2, MIT, releases)
- https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin (ABC, hooks, register())
- https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/memory-provider-plugin.md (same, in-tree)
- https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/ (provider list, lifecycle, single-active rule)
- https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins (mcp_servers config, `hermes_agent.plugins` entry point)
- https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md (standalone-repo policy for third-party integrations)
