# Forget, delete and cleanup: specification

Status: **approved by the owner.** §8's old-daemon detection design is also approved, and it is already built on `main` (the first PR). Code is built once against this document and verified against the invariants in §9.

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
4. **Matching is by whole words, and the cut is a sentence.** The wording is matched case-insensitively on word boundaries (Unicode-aware), so forgetting the entity "API" never touches "rapid" or "capital". Only the sentence that holds the wording is cut; the rest of the node stays.
5. **A forget reaches the origin it was forgotten from, and nowhere else by default.** Other units that hold the same sentence are not changed, now or at any later capture; the response names every one of them (`still_held_by`). `forget --everywhere` (CLI, REST) is explicit and exact: it cuts **only the exact wording that was targeted** (the forgotten node's own content, never a label and never every sentence of a forgotten session) out of every other node that holds it, with the same word-bounded matching. The cut is **in place**: the sentence goes and the node stays (a node that was only that sentence stays as `[removed]`, and so does a label that was a piece of it). Apart from the node the user named, `--everywhere` deletes nothing (a node written by several sources keeps all of its owners), never cuts another node back to its label because of the wording, and suppresses **only the forgotten origin's own identity**: the node's single proven writer, or the source in an unproven node's own source key. A node written by several sources has no single origin, so none is suppressed, and the response says so. Every node whose wording was cut is listed with its source (`cut_in_place`), and the message is built from what happened. A whole-session forget with `--everywhere` bans that session's identity everywhere and names no wording, so its sentences are banned nowhere else. There is no MCP forget surface.
6. **Nodes whose owner is not proven are never changed** by another unit's forget (§3). They are kept and listed.
7. **Session cascade:** forgetting a context node with `cascade` forgets the whole unit (§3) and everything only it wrote.

Every response states exactly what was removed, cut back and kept, with counts that match the database. It also names the **scope that applied** and which re-entry it does and does not stop (`reentry`: the scope, `stops`, `does_not_stop`); a plain forget says that a copy saved under a new name, with no id of its own and edited past the copy rule (§6), could come back, and points at `--everywhere`. Single-node forgets also name the blast radius: "This forgets one node and suppresses future imports from session X."

## 3. Units and ownership

- A **unit** is one source item with a stable identity: a Claude Code or Codex session (its UUID; sub-agent files that carry the parent's `sessionId` belong to it), an OpenAI conversation, a watched file (by real path), a vault section, or a skill.
- **Ownership is recorded when something is written**, in `node_provenance(node_id, unit_id, label_ok, wording_ok)`, by every writer that creates a node or merges into one. Nothing is inferred later from edges or text matching.
- **Upgrade rule:** when an existing database is backfilled, ownership that cannot be proven is **not guessed**. A node with unproven ownership is kept, marked unproven, and listed when a forget touches its likely unit. **Revien never deletes on a guess.**
- Hand-added nodes, nodes outside any unit and another unit's context node are never removed or rewritten by a forget; unproven nodes are only listed.
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
- the echo of a query the user types after the forget;
- other units that still hold the wording (plain forget is per source; `--everywhere` removes them), named in the response;
- the trained ranking model, which cannot unlearn. A forget marks it stale where the training loop reads it. With neural ranking on, the forget retrains it from the scrubbed signals; if that cannot run it stays stale and the response says so. With neural ranking off nothing retrains it, and it stays stale until `POST /v1/training/run`. There is no scheduled or nightly retrain job; the training loop retrains on new signals (100, then every 500) and on the stale mark. `revien watch` snapshots are part of the raw store and are scrubbed like the live database; one that could not be scrubbed is named;
- nodes Revien was not allowed to touch (owner not proven) that hold a sentence of the forgotten text, or a word only it used, named under `other_units`.

"Gone everywhere" is printed only when there are no leftovers, nothing remains in a table, and no surviving node holds what was forgotten: a sentence taken out (whatever its owner), a word only the forgotten text used in a node nobody vouches for, or, in a node the forget cut back, cut in place or kept, a label or description of 20 or more characters that is a piece of the forgotten text. Otherwise the response says "Forgotten from Revien" and lists them, naming the holders under `still_held_by`.

## 5. Shared nodes

- If another live unit vouched for the **name and the wording**, the node stays as is, and ownership moves to that unit.
- If another live unit vouched for the **name only**, the node is **cut back to its label**. The forgotten wording is scrubbed per §4, and the cut is logged (node id, time; no content) so the description can be regenerated from surviving units on a later index pass. A regenerated description is clean by construction.
- If no live unit vouches for it, it is removed with the forgotten unit.
- `forget --everywhere` never cuts a shared node back to its label: it cuts the exact targeted sentence out of it in place (§2.5).
- A cut-back applies only to a shared node that is not a context node and whose wording is unproven; the cut is the sentence, never the whole node unless nothing else is left.
- Old databases: descriptions saved before provenance existed are treated as unproven wording and are cut back, and logged, the first time a forget touches them. This is a privacy fix, not data loss.

## 6. Coming back: suppression

- A forget writes a **content-free** suppression record bound to the unit's stable identity.
- **Forget is a standing rule, bound to the origin that was forgotten (owner ruling).** Every forgotten sentence (20 characters or more) is kept as a keyed fingerprint (the same key as above, content-free) together with the identity of the source it was forgotten from, and is left out of any later capture that comes from that origin or is a copy of it, and out of nothing else. Another source that holds the same sentence keeps its own copy, now and at every later sync.<!-- REVISE after the owner's ruling on the copy rule: this sentence is not final --> An origin is recognised, by the code as built, in one of three ways:
  1. **By an id that travels with a copy.** A Claude Code session file is the session named by the `sessionId` inside it (read from its first entries; a sub-agent file carrying its parent's `sessionId` is that session; a renamed, moved or re-saved copy stays forgotten). A Codex rollout is the file named by its stem, with the `id` in its `session_meta` line kept as one more key of that unit (so no unit id changes, and a renamed or moved rollout stays forgotten). A ChatGPT, Claude.ai or OpenAI conversation is its conversation id from the export; a Readwise highlight its id.
  2. **By real path and inode; a copy by content.** Watched files, vault sections and skills.
  3. **By content alone.** A Claude Code file or Codex rollout that carries no id, and every source with no identity (MCP, Hermes, Ollama, LangChain, API feeds, the CLI, a graph-import node without an ingest key).
  A **copy** (cases 2 and 3) is a capture that carries at least 80% of the forgotten source's own sentences of 20 or more characters (and at least two of them), where those make up at least half of the capture's own such sentences. A file copied to a new path with a line appended stays forgotten; a different session that shares a line or two of boilerplate is not taken for a copy. A source with fewer than two such sentences, or a copy edited or padded past the rule, is not recognised by content and can come back under a new name if it has no id of its own (Known limits). Fragments under 20 characters cannot be matched on their own; they are kept out of their own source (and a copy of it) by its identity, never in another source, and the response says so.
- **`forget --everywhere` adds one global entry:** the exact wording that was targeted, kept out of every source. It can be lifted with `revien restore forget:<id>`.
- **Capture decisions are made under the write lock.** The pipeline's keyed-refresh path, graph import, the OpenAI adapter, skills, vault reconcile and skill proposals ask the gate inside a transaction taken with BEGIN IMMEDIATE. The pipeline's append path writes and then undoes what was forgotten inside one IMMEDIATE transaction (`enforce_after_write`). A forget commits its denylist under BEGIN IMMEDIATE first and plans the removal after that commit, so whichever of a forget and a capture comes first, a forgotten sentence is not left.
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
- **Old daemons are detected and refused, not just documented.** A daemon running pre-upgrade code ignores suppression and fencing, so the upgraded CLI and API must refuse operations that depend on them, with a clear "restart required" error. Design approved by the owner and built:
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

- The `list_nodes` crash on databases created by v0.3.0 and 7757085. **Resolved:** it does not reproduce on `main` and is not an upgrade blocker (verified for PR #5).
- Stable per-item identity for shared channels.
- Automatic re-distill of the vault on forget.
- **Confirmed split:** the usage signal, receipts and the Claude Code adapter fix ship first, as their own PR, with a smaller blast radius and independent review. Forget follows as a separate PR built against this spec.
