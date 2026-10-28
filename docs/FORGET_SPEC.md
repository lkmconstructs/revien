# Forget, delete and cleanup: specification

Status: **approved by the owner**, narrowed by the owner's ruling of 2026-10-20: forget is for **one specific thing**, not a whole conversation. §8's old-daemon detection design is also approved, and it is already built on `main` (the first PR). Code is built once against this document and verified against the invariants in §9.

Sources: owner rulings (2026-10-16) and the findings of seven independent verification rounds.

---

## 1. Three verbs, never folded together

| Verb | Meaning | Leaves |
|---|---|---|
| **forget** | Privacy, for **one specific thing**: a fact, an entity, a decision, a preference, or a named piece of wording. The user wants it gone and not coming back. | A content-free **forget record**, an audit entry, and a scrub of every copy (§4). |
| **delete** | Operational. Remove this node now, a whole conversation included. | A content-free **delete audit entry** (node id, actor, time). **No forget record.** The content may be captured again if its source still holds it. |
| **automatic cleanup** | Retention, decay and other unattended maintenance. | Its own audit entry. No forget record. |

- `DELETE /v1/nodes/{id}`, LangChain `clear()` and a **user-triggered retention sweep** (`POST /v1/retention/sweep`) are **delete**. Only scheduled, unattended retention is **automatic cleanup**.
- Only an explicit forget call (`forget`, REST `POST /v1/nodes/{id}/forget`, CLI) is **forget**. Revien never infers forget from a method name. An application that offers "forget my memories" must call the forget verb.
- Which verb ran is recorded by the code path itself, never by a caller-supplied reason string.

## 2. What a forget removes

A forget names **one specific thing**. It does not name a conversation.

1. **A conversation is refused.** Forgetting a context node (the stored text of a session, a conversation or a file), or a skill, is refused with: "A conversation's or file's stored text is not one thing to forget. To remove it, run `revien delete <node id>`; to remove a specific secret wherever it appears, run `revien forget --wording`." Nothing changes. There is no `cascade`: the CLI flag and the operation argument do not exist, and a REST request that still sends a `cascade` field is answered 400 (`cascade_removed`).
2. **The forgotten node**, everywhere (§4).
3. **The forgotten wording inside the stored conversation.** The wording of a node is what was said: a fact, decision or preference is its text; an entity or a topic is its **label** ("Brigitte Vandermeer"), never the content string Revien gives it ("Entity: Brigitte Vandermeer"). Every sentence of the source's stored conversation that holds the wording is cut out of the context node: its label, its content, the audit snapshots, the vector and the search index.
   - If the wording cannot be found verbatim in the stored conversation (for example, the model paraphrased it), the forget is **refused**, with a message naming the context node. To remove the whole conversation, use delete. Revien never claims a removal it could not do.
4. **Other nodes of that source that carry the wording** are removed with it if no other live unit vouched for them; shared nodes are handled by §5.
5. **Matching is by whole words, and the cut is a sentence.** The wording is matched case-insensitively on word boundaries (Unicode-aware), so forgetting the entity "API" never touches "rapid" or "capital". Only the sentence that holds the wording is cut; the rest of the node stays.
6. **A forget reaches the origin it was forgotten from, and nowhere else by default.** Other units that hold the same sentence are not changed, now or at any later capture; the response names every one of them (`still_held_by`). `forget --everywhere` (CLI, REST) is explicit and exact: it cuts **only the exact wording that was targeted** (the forgotten node's own wording, as above) out of every other node that holds it, with the same word-bounded matching. The cut is **in place**: the sentence goes and the node stays (a node that was only that sentence stays as `[removed]`, and so does a label that was a piece of it). Apart from the node the user named, `--everywhere` deletes nothing (a node written by several sources keeps all of its owners) and never cuts another node back to its label because of the wording. Every node whose wording was cut is listed with its source (`cut_in_place`, the origin's own conversation too), and the message is built from what happened. There is no MCP forget surface.
7. **Nodes whose owner is not proven are never changed** by another unit's forget (§3). They are kept and listed.

Every response states exactly what was removed, cut back and kept, with counts that match the database. It also names the **scope that applied** and which re-entry it does and does not stop (`reentry`: the scope, `stops`, `does_not_stop`): a plain forget says that "a copy of this source under a different identity (another id, another path) is not recognised; use --everywhere to keep this wording out of every source". The blast radius is stated too: "This forgets one node and keeps its wording out of later captures of session X."

## 2a. Forget a literal string: a secret

Forget is also the way to remove **a secret that should never have been recorded** (a key, a token, a password). The user names the exact characters; only those characters go. Revien never guesses what a secret looks like and never forgets a whole conversation for one.

- **`revien forget --wording STRING`** (`--wording -` reads the string from standard input, so it is not in the shell history) and **`POST /v1/forget/wording`** `{"wording": ..., "preview": false}`. A string shorter than 6 characters (line wraps ignored) is refused, before anything is cut and without echoing it: a shorter string is common enough to cut ordinary text. `--preview` counts what it would cut first.
- **One column inventory (§2a).** The cut, the residue check behind "gone everywhere", the piece report, the snapshot scrub and the capture gate all read the text columns from the schema (`revien/forget/inventory.py`), not from a list. "Gone everywhere" means: no copy of the whole string or of a reportable piece in any of those columns, in the database file, its write-ahead log and shared-memory file, or in any watch snapshot (columns and raw bytes). Identity columns are renamed with the same cut everywhere they are stored and reported; an identity that cannot be renamed is a leftover.
- **Raw characters, not whole words, case-sensitive.** A secret sits inside `?token=...&x` and `ghp_...`, where there is no word boundary, so the whole-word rule of a named-node forget (§2.5) does not apply here. A different case or encoding is a different string. A line break (and the indent after it) is treated as a wrap when the line before it has 20 or more characters (`scrub.MIN_WRAPPED_LINE`): a key wrapped across two lines is found by its joined characters, and a named string that coincides with such a join is cut across it. After a shorter line the two lines are two texts and are never joined. The half of an occurrence that a derived field kept (a label is the content cut off where the line wrapped) is cut with it, in the same node's label, preview and context line, its audit rows and the watch snapshots: only a prefix at the end or a suffix at the start of the field, at least 8 characters and a quarter of the string, and only where the field is a verbatim stretch of that node's content that stops exactly inside the occurrence that was cut. A piece anywhere else is not cut; it is reported from every surface (node fields, audit rows, plain and gzip watch snapshots) as `partial_copy`: surface, node, field, number of characters, and the answer is then not "gone everywhere".
- **What it cuts:** exactly those characters, nothing around them, out of every node (a conversation's stored text included), every audit row (including those of nodes that were deleted, edited or refreshed away before the forget: the scrub reads every row of `audit_log`, not the rows of nodes it touches), recall queries and receipts, supersession candidates, edges, `training.db` rows and the `revien watch` snapshots. The vectors of the nodes it cut are removed and the nodes are re-embedded from the cut text.
- **What it keeps out:** the string goes on a **global** denylist, per string and not per sentence, as keyed fingerprints only: its length in characters, a keyed rolling hash and a keyed HMAC. It is checked in the pipeline (before anything is extracted or stored, and again inside the write transaction), the OpenAI adapter, skills, vault reconcile, graph import, and every node write (`add_node`, `update_node`, so `PUT /v1/nodes/{id}` cannot put it back). `--override-suppressed` does not lift it; `revien restore forget:<id>` does.
- **It records the string nowhere:** not in the forget record, not in the audit row of the forget (counts only), not in the log, not in the CLI or REST response. The record discloses the string's length.
- **`forget.key`.** Every fingerprint is keyed by it. If it is lost or replaced, earlier records match nothing. The next forget says so (`warnings`: "forget.key is not the key the earlier forgets were made with ... forget those again"), because Revien keeps a keyed check value of the key it last used; there is no way to recover the old records without the old key.
- **What it cannot do:** recall a copy that already left the machine (a cloud extractor saw the sentence at capture time; **rotate a leaked key first, forget second**), edit your own files or exports (listed as leftovers), or find the same secret in another encoding.
- **"Gone everywhere"** is said only when the string is in no table, no raw file (db, -wal, -shm), no snapshot and nothing is listed under `leftovers`; otherwise it is "Forgotten from Revien" and the leftovers are named.

## 3. Units and ownership

- A **unit** is one source item with a stable identity: a Claude Code or Codex session (its UUID; sub-agent files that carry the parent's `sessionId` belong to it), an OpenAI conversation, a watched file (by real path), a vault section, or a skill.
- **Ownership is recorded when something is written**, in `node_provenance(node_id, unit_id, label_ok, wording_ok)`, by every writer that creates a node or merges into one. Nothing is inferred later from edges or text matching.
- **Upgrade rule:** when an existing database is backfilled, ownership that cannot be proven is **not guessed**. A node with unproven ownership is kept, marked unproven, and listed when a forget touches its likely unit. **Revien never deletes on a guess.** The source an unproven node names in its own source key (a session id, a conversation or highlight id, a vault section) is still its identity for the denylist (§6).
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
- distilled Obsidian notes, until the user runs `revien distill-vault`. Revien never rewrites the vault on its own; automatic re-distill stays parked. A forget that leaves a distilled note behind must say so explicitly, naming the note and the manual command that removes the fact. Until that runs, the operation is reported as "forgotten from Revien", never "gone everywhere". With `--everywhere` the keyed fingerprint (§6) also stops the exact wording, retyped into the note, from being captured in the meantime;
- the echo of a query the user types after the forget;
- other units that still hold the wording (plain forget is per source; `--everywhere` removes them), named in the response;
- the trained ranking model, which cannot unlearn. A forget marks it stale where the training loop reads it. With neural ranking on, the forget retrains it from the scrubbed signals; if that cannot run it stays stale and the response says so. With neural ranking off nothing retrains it, and it stays stale until `POST /v1/training/run`. There is no scheduled or nightly retrain job; the training loop retrains on new signals (100, then every 500) and on the stale mark. `revien watch` snapshots are part of the raw store and are scrubbed like the live database; one that could not be scrubbed is named;
- nodes Revien was not allowed to touch (owner not proven) that hold a sentence of the forgotten text, or a word only it used, named under `other_units`.

"Gone everywhere" is printed only when there are no leftovers, nothing remains in a table, and no surviving node holds what was forgotten: a sentence taken out (whatever its owner), a word only the forgotten text used in a node nobody vouches for, or, in a node the forget cut back, cut in place or kept, a label or description of 20 or more characters that is a piece of the forgotten text. Otherwise the response says "Forgotten from Revien" and lists them, naming the holders under `still_held_by`.

## 5. Shared nodes

- If another live unit vouched for the **name and the wording**, the node stays as is, and ownership moves to that unit.
- If another live unit vouched for the **name only**, the node is **cut back to its label**. The forgotten wording is scrubbed per §4, and the cut is logged (node id, time; no content) so the description can be regenerated from surviving units on a later index pass. A regenerated description is clean by construction.
- If no live unit vouches for it, it is removed with the forgotten node.
- `forget --everywhere` never cuts a shared node back to its label: it cuts the exact targeted sentence out of it in place (§2.6).
- A cut-back applies only to a shared node that is not a context node and whose wording is unproven; the cut is the sentence, never the whole node unless nothing else is left. A node never loses the label another unit vouched for.
- Old databases: descriptions saved before provenance existed are treated as unproven wording and are cut back, and logged, the first time a forget touches them. This is a privacy fix, not data loss.

## 6. Coming back: the denylist

- A forget writes a **content-free forget record** (the `suppressions` table). It holds keyed fingerprints and the identity of the origin the wording was forgotten from; it holds no text. The fingerprints are an HMAC with a secret key kept only in the user's Revien home (`$REVIEN_HOME/forget.key`), so a short secret cannot be recovered by guessing.
- **The record is a standing rule bound to the origin's identity.** It keeps out of later captures the sentences that held the forgotten wording, and any new sentence that holds the wording as whole words (the wording is also kept as a keyed fingerprint of its form and its length in words, which is compared with every run of that many words in a sentence). It applies to a capture only if that capture has the **same identity** as the origin:
  1. **An id that travels with a copy.** A Claude Code session file is the session named by the `sessionId` inside it (a sub-agent file carrying its parent's `sessionId` is that session; a renamed, moved or re-saved copy is the same session). A Codex rollout is the file named by its stem, with the `id` in its `session_meta` line kept as one more key of that unit (so a renamed or moved rollout is the same source). A ChatGPT, Claude.ai or OpenAI conversation is its conversation id from the export; a Readwise highlight its id.
  2. **Real path and inode.** Watched files, vault sections and skills (and a Claude Code or Codex file that carries no id): a symlink, a hard link, the same file moved, a case change and a new timestamp are the same file.
  A capture that does not have that identity is not the origin: **it lands as written**. Revien does not guess from content that it is a copy (no percentage of shared sentences, no shared tokens): a copy of the source under a different identity (another id, another path, a file under another name and inode) is not recognised. The forget response says so. A source with no identity of its own (a shared channel: MCP, Hermes, Ollama, LangChain, API feeds, the CLI) cannot be bound at all, and the response says so (`not_bound`).
- **The source itself is not suppressed.** Only the sentences go: a file that changed, grew, was moved or was re-synced is captured, minus the forgotten wording. The response to a forget lists the sources the wording is kept out of (`bound_to`).
- **`forget --everywhere` adds one global record:** the exact wording that was targeted, kept out of every source (the sentences that held it, in any capture, and a capture that is exactly that wording is refused whole). It can be lifted with `revien restore forget:<id>`.
- Fragments under 20 characters cannot be matched as sentences on their own; they are kept out of their own source by its identity, never in another source, and the response says so (`fallback`).
- **Capture decisions are made under the write lock.** The pipeline's keyed-refresh path, graph import, the OpenAI adapter, skills, vault reconcile and skill proposals ask the gate inside a transaction taken with BEGIN IMMEDIATE. The pipeline's append path writes and then undoes what was forgotten inside one IMMEDIATE transaction (`enforce_after_write`). A forget commits its denylist under BEGIN IMMEDIATE first and plans the removal after that commit, so whichever of a forget and a capture comes first, a forgotten sentence is not left.
- **Shared channels** (`mcp`, `hermes`, `ollama_*`, `langchain`, `api:<feed>`, the CLI) have no stable item identity: a forget from them removes now, and the response says the wording is not kept out of a later capture of the same channel ("this capture came through a shared channel, which gives it no identity of its own"). `--everywhere` is the way to keep it out. Stable per-item ids are **required follow-up work**.
- Every capture path is gated: the pipeline, adapters, MCP, REST, CLI, importers, graph import, skills and skill proposals. A gate that cannot see a stable identity says so; it never claims protection it does not give.
- Path identities use the real path. A caller-supplied path that does not exist on disk proves nothing, and neither does a path to some other file.

## 7. Restore

- `revien restore <path>` lifts the forget records bound to **one** file; `revien restore <unit id>` those bound to that unit; `revien restore forget:<id>` one record. Restore by bare name never fans out across paths.
- Bulk restore requires an explicit pattern (`revien restore 'projA/**'`) and lists everything it lifts before lifting. `revien restore --list` shows what is active.
- Restore is audited, and only an explicit restore or a re-import with an explicit override can lift a record. It lets the wording be captured again; it does not bring a removed node back.

## 8. Sync reset and fencing

- `revien sync reset <adapter>` works only for adapters whose re-read is provably duplicate-free; others are refused. `--orphaned` is allowed only where a saved sync position exists. When explicit forgets exist, `--yes` is required.
- **Generation fencing:** a reset cannot be undone by a sync that started before it (compare-and-set on the cursor generation).
- **Old daemons are detected and refused, not just documented.** A daemon running pre-upgrade code ignores suppression and fencing, so the upgraded CLI and API must refuse operations that depend on them, with a clear "restart required" error. Design approved by the owner and built:
  - A current daemon writes a heartbeat (code features, process id, time) into the database. Reading it needs no network.
  - Pre-upgrade daemons write no heartbeat, so they are detected through one pinned, loopback-only health check (`/v1/health` on the local machine; an answer without the feature list means old code). Talking to Revien's own daemon on loopback is not egress. The egress gate pins that one module by hash and also forbids every other file from importing it or reaching into it, which closes the re-export evasion found in round 7.
  - Either detection, a stale-code heartbeat or an old health answer, refuses the operation with the restart-required error. No daemon answering means proceed.
  - The rejected alternative is process-table inspection. It is platform-specific (the CI runs on Windows too), and it needs subprocess calls that the egress gate flags.

## 9. Invariants the verifier checks

1. After a forget of a string (§2a), the string is in no table, no raw file, no snapshot and no vector of a node it cut, whatever happened to the nodes before (deleted, edited, refreshed away), and stays out of every later capture. After a forget of one thing, its wording is absent from every surface of its origin's identity: the stored conversation, every table, the file and the WAL (§4), except the allowed leftovers. The user's other sentences in that source may remain; they are disclosed (`still_held_by`, `other_units`, `remaining`), and "Gone everywhere" appears only when nothing remains. `--everywhere` removes the wording from every source.
2. No forget deletes or rewrites another unit's content beyond the named wording, and no upgrade deletes on a guess. A node that several origins co-wrote is forgotten for each origin that wrote it, and the response states that (`bound_to`, `identity_note`). A forget of a context node is refused.
3. Forgotten wording is not recaptured from the origin it was forgotten from by any path without an explicit restore or override, including after a re-sync, a grown session, a move, rename, symlink, hard link or re-timestamp; and a capture with a different identity lands as written, and the response said so.
4. Delete, user-triggered sweeps and automatic cleanup write no forget records. Each writes its own audit entry.
5. Every response sentence and CHANGELOG sentence is true against the code.
6. Every invariant has a test that fails when its guarding code is mutated away.
7. Databases created by v0.2.x, v0.3.0 and every commit on this branch upgrade without error, and without deleting on a guess.

## 10. Out of scope here, tracked separately

- The `list_nodes` crash on databases created by v0.3.0 and 7757085. **Resolved:** it does not reproduce on `main` and is not an upgrade blocker (verified for PR #5).
- Stable per-item identity for shared channels.
- Automatic re-distill of the vault on forget.
- **Confirmed split:** the usage signal, receipts and the Claude Code adapter fix ship first, as their own PR, with a smaller blast radius and independent review. Forget follows as a separate PR built against this spec.
