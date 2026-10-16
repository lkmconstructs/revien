# Forget, delete and cleanup: specification

Status: **draft for owner approval.** §4 (vault), §8 and §10 are updated with the owner's 2026-10-16 confirmations; only §8's daemon-detection design still awaits approval. No further code is built against these rules until this document is approved. Code is then built once against it, and verified against the invariants in §9.

Sources: owner rulings (Lissa, with Ash's reads, 2026-10-16), and the findings of seven independent verification rounds on branch `claude/project-mnemosyne-2eevt2`.

---

## 1. Three verbs, never folded together

| Verb | Meaning | Leaves |
|---|---|---|
| **forget** | Privacy. The user wants this content gone and not coming back. | A content-free **suppression record**, an audit entry, and a scrub of every copy (§4). |
| **delete** | Operational. Remove this node now. | A content-free **delete audit entry** (node id, actor, time). **No suppression record.** The content may be captured again if its source still holds it. |
| **automatic cleanup** | Retention, decay and other unattended maintenance. | Its own audit entry. No suppression record. |

- `DELETE /v1/nodes/{id}`, LangChain `clear()` and a **user-triggered retention sweep** (`POST /v1/retention/sweep`) are **delete**. Only scheduled, unattended retention is **automatic cleanup**.
- Only an explicit forget call (`forget`, REST `POST /v1/nodes/{id}/forget`, CLI) is **forget**. Revien never infers forget from a method name. An application that offers "forget my memories" must call the forget verb.
- Which verb ran is recorded by the code path itself, never by a caller-supplied reason string.

## 2. What a forget removes

1. **The forgotten node**, everywhere (§4).
2. **The forgotten wording inside the stored conversation.** The exact sentence is cut out of the session's context node: its label, its content, the audit snapshots, the vector and the search index.
   - If the wording cannot be found verbatim (for example, the model paraphrased it), the forget is **refused**, with a message naming the context node. The user can then forget the whole conversation. Revien never claims a removal it could not do.
3. **Shared nodes whose wording only the forgotten unit supplied** are cut back to their label (§5).
4. **Session cascade:** forgetting a context node with `cascade` forgets the whole unit (§3) and everything only it wrote.

Every response states exactly what was removed, cut back and kept, with counts that match the database. Single-node forgets also name the blast radius: "This forgets one node and suppresses future imports from session X."

## 3. Units and ownership

- A **unit** is one source item with a stable identity: a Claude Code or Codex session (its UUID; sub-agent files that carry the parent's `sessionId` belong to it), an OpenAI conversation, a watched file (by real path), a vault section, or a skill.
- **Ownership is recorded when something is written**, in `node_provenance(node_id, unit_id, label_ok, wording_ok)`, by every writer that creates a node or merges into one. Nothing is inferred later from edges or text matching.
- **Upgrade rule:** when an existing database is backfilled, ownership that cannot be proven is **not guessed**. A node with unproven ownership is kept, marked unproven, and listed when a forget touches its likely unit. **Revien never deletes on a guess.**
- Hand-added nodes and nodes outside any unit are never removed by a cascade.
- Labels are compared with Unicode-aware normalization, so "Jörg" and "Jürg" stay distinct.

## 4. "Gone everywhere"

After a forget, the forgotten text must be absent from all of these:
- every table, including serialized audit snapshots (node and edge audit rows), supersession candidates, the distill manifest, the usage ledger (including `query` on every judgment of the same recall), recall receipts, `vec_nodes`, `pending_embeds` and the suppression table;
- the raw database file, the WAL and the freelist (`secure_delete` is on, the WAL is truncated after a forget), and `training.db` (same treatment);
- `/v1/export`, `/v1/graph`, `/v1/nodes`, `/v1/audit/*`, `revien export`, recall and TOON output.

Allowed leftovers, each named in the response when it applies:
- the user's own source files (the session `.jsonl` on disk);
- exports written to disk before the forget;
- distilled Obsidian notes, until the user runs `revien distill-vault`. Revien never rewrites the vault on its own; automatic re-distill stays parked. A forget that leaves a distilled note behind must say so explicitly, naming the note and the manual command that removes the fact. Until that runs, the operation is reported as "forgotten from Revien", never "gone everywhere". The keyed fingerprint (§6) stops the unchanged note from being re-imported in the meantime;
- the echo of a query the user types after the forget.

## 5. Shared nodes

- If another live unit vouched for the **name and the wording**, the node stays as is, and ownership moves to that unit.
- If another live unit vouched for the **name only**, the node is **cut back to its label**. The forgotten wording is scrubbed per §4, and the cut is logged (node id, time; no content) so the description can be regenerated from surviving units on a later index pass. A regenerated description is clean by construction.
- If no live unit vouches for it, it is removed with the forgotten unit.
- Old databases: descriptions saved before provenance existed are treated as unproven wording and are cut back, and logged, the first time a forget touches them. This is a privacy fix, not data loss.

## 6. Coming back: suppression

- A forget writes a **content-free** suppression record bound to the unit's stable identity.
- **File-backed units** (watched files, vault sections, session files) are suppressed **until the file changes**. The record also stores a **keyed fingerprint** of the forgotten content: an HMAC with a secret key kept only in the user's Revien home, so a short secret cannot be recovered by guessing. Text with an identical fingerprint is never recaptured, even if the file's timestamp moves. New or changed content is new evidence and may be captured.
- **Units with no file behind them** (OpenAI imports, manual entries) are suppressed **for good**.
- **Shared channels** (`mcp`, `hermes`, `ollama_*`, `langchain`, `api:<feed>`) cannot be suppressed without blocking the whole channel. Forgetting from them **deletes now**, and the response says the content is "eligible for recapture because the source supplied no stable item identity". Stable per-item ids are **required follow-up work**.
- Every capture path is gated: the pipeline, adapters, MCP, REST, CLI, importers, graph import, skills and skill proposals. A gate that cannot see a stable identity says so; it never claims protection it does not give.
- Path identities use the real path. The verifier attacks to cover are symlinks, hard links (inode where available), a moved file followed by a new file at the old path, made-up `path` values sent through the API, and case changes. A caller-supplied path that does not exist on disk is never trusted to lift a name-level suppression.

## 7. Restore

- `revien restore <path>` lifts **one** file's suppression. Restore by bare name never fans out across paths.
- Bulk restore requires an explicit pattern (`revien restore 'projA/**'`) and lists everything it lifts before lifting.
- Restore is audited, and only an explicit restore or a re-import with an explicit override can lift suppression.

## 8. Sync reset and fencing

- `revien sync reset <adapter>` works only for adapters whose re-read is provably duplicate-free; others are refused. `--orphaned` is allowed only where a saved sync position exists. When explicit forgets exist, `--yes` is required.
- **Generation fencing:** a reset cannot be undone by a sync that started before it (compare-and-set on the cursor generation).
- **Old daemons are detected and refused, not just documented.** A daemon running pre-upgrade code ignores suppression and fencing, so the upgraded CLI and API must refuse operations that depend on them, with a clear "restart required" error. Proposed design, for owner approval:
  - A current daemon writes a heartbeat (code features, process id, time) into the database. Reading it needs no network.
  - Pre-upgrade daemons write no heartbeat, so they are detected through one pinned, loopback-only health check (`/v1/health` on the local machine; an answer without the feature list means old code). Talking to Revien's own daemon on loopback is not egress. The egress gate pins that one module by hash and also forbids every other file from importing it or reaching into it, which closes the re-export evasion found in round 7.
  - Either detection, a stale-code heartbeat or an old health answer, refuses the operation with the restart-required error. No daemon answering means proceed.
  - The rejected alternative is process-table inspection. It is platform-specific (the CI runs on Windows too), and it needs subprocess calls that the egress gate flags.

## 9. Invariants the verifier checks

1. After any forget, a unique secret seeded through every capture path is absent from every surface in §4, except the allowed leftovers, each of which is disclosed in the response.
2. No forget, cascade or upgrade ever deletes another unit's context node, or a node another live unit vouched for in full.
3. No suppressed identity is recaptured by any path without an explicit restore or override, including after a move, rename, symlink, hard link or re-timestamp.
4. Delete, user-triggered sweeps and automatic cleanup write no suppression records. Each writes its own audit entry.
5. Every response sentence and CHANGELOG sentence is true against the code.
6. Every invariant has a test that fails when its guarding code is mutated away.
7. Databases created by v0.2.x, v0.3.0 and every commit on this branch upgrade without error, and without deleting on a guess.

## 10. Out of scope here, tracked separately

- The `list_nodes` crash on databases created by v0.3.0 and 7757085 (seen at 7420060 and 2f1dfb5; its status on `main` is unconfirmed). It is investigated on its own and fixed in its own PR. **But if it breaks the supported upgrade path for this redesign, it is a release blocker for the forget work.**
- Stable per-item identity for shared channels.
- Automatic re-distill of the vault on forget.
- **Confirmed split:** the usage signal, receipts and the Claude Code adapter fix ship first, as their own PR, with a smaller blast radius and independent review. Forget follows as a separate PR built against this spec.
