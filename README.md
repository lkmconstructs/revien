![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![CI](https://github.com/lkmconstructs/revien/actions/workflows/ci.yml/badge.svg)
![Network](https://img.shields.io/badge/network%20calls-0-blue)

# Revien

**Memory that returns.**

Revien is a local-first, graph-based memory engine for AI systems. It gives any AI tool — local models, Claude Code, API assistants, agent frameworks — persistent memory across sessions. No GPU. No cloud account. No telemetry. Nothing is compacted away, and nothing you feed it leaves your machine.

```bash
pip install revien
revien connect claude-code
revien start
```

That's it. Revien starts building persistent memory on disk, in a single SQLite file you own.

---

## The wedge: sovereignty you can verify

Most memory systems ask you to trust that your data is handled well. Revien is built so you don't have to — every sovereignty claim below is enforced in code and checked by the benchmark suite on every run:

- **$0, zero network egress on the default path.** Local extraction, local embeddings (`bge-small`, on-device), local SQLite. The benchmark asserts `network_calls == 0` and fails if anything phones home. Model files load offline-first — a warm install touches nothing.
- **Zero telemetry.** Revien collects no usage data, no crash reports, no phone-home. See [TELEMETRY.md](TELEMETRY.md).
- **Nothing compacted away.** The full graph is preserved. Retrieval is surgical — it returns only what's relevant — but it never summarizes your history into oblivion to save space.
- **A non-destructive audit trail.** Every node creation, update, supersession, and merge is recorded. You can trace any fact back to the exact turn it came from, and review every automatic decision the engine made.
- **Consent is enforced, not requested.** A per-source deny list stops capture at the door. Soft-invalidation is reversible. Nothing is hard-deleted behind your back.
- **Your curated knowledge outranks the machine's.** If you connect an Obsidian vault, a machine-extracted claim can never silently overwrite something you wrote by hand — contradictions go to a review queue, not a destructive merge.

The discipline behind these claims is the product. Revien ships with a benchmark harness that measures its own retrieval honestly — including where it's weak — and it has already caught its own bugs before they could reach a user.

---

## How it works

### Memory is a graph, not a compaction buffer

When you feed Revien a conversation or a note, it extracts typed nodes — **entities, decisions, facts, preferences, topics, events** — and connects them with typed edges. Every ingestion also stores the verbatim turn as a `context` node, so the original wording is never lost. The graph grows; nothing is thrown away.

### Retrieval is semantic-first, refined by the graph

When you query Revien:

1. **Semantic search** embeds your query and finds the nearest stored memories by meaning — so "what did we pick for the database?" finds a turn about "we went with Postgres" even with no shared keywords.
2. **A graph walk** from those anchors pulls in connected context — the decision, the entity, the reasoning that surrounds the hit.
3. **Three-factor scoring** refines the ranking:
   - **Recency** — how recent is the memory's *content* (when it was actually said), decaying gently so old-but-true facts aren't buried.
   - **Frequency** — how often the memory has been *confirmed useful* (via explicit use, not merely returned — a retrieval popularity loop would just surface whatever it surfaced last).
   - **Proximity** — how many graph hops from the query's anchors.

Only the top results come back. Your AI gets a lean, relevant context window instead of a dump.

The semantic layer is the spine: with it, retrieval finds the right memory by meaning; without it, recall falls back to keyword matching and degrades sharply — so Revien makes that degrade **loud** (a warning on every recall, and a `semantic_note` on every response) rather than silently returning worse results.

---

## Benchmarks

Revien is measured on two separate corpora. **They are reported separately and never blended** — conversational memory (episodic: who said what, when) and vault memory (curated: decisions, facts, reference) are different problems, and averaging them would hide more than it shows.

All numbers below are reproducible from a fresh checkout: local extraction, local `bge-small` embeddings, a zero-LLM extractive reader, **$0 and 0 network calls**. Each has a results JSON in `results/`.

### Conversational recall — LoCoMo, 1,986 QA

| Metric | Value |
|--------|-------|
| Recall@10 | **0.514** |
| Recall@5 | 0.413 |
| Recall@1 | 0.197 |
| MRR | 0.323 |
| nDCG@10 | 0.356 |
| Recall latency (p50 / p90) | **85ms / 250ms** |
| Cost / network calls | **$0 / 0** |
| Sovereignty checks | **PASS** |

### Vault recall — curated Obsidian corpus, 43 QA

| Metric | Value |
|--------|-------|
| Recall@10 (overall) | **0.884** |
| Single-note questions | 0.933 |
| Cross-note (multi-hop) questions | 0.733 |
| MRR | 0.738 |

**Attachment rate** — a vault-specific measure of whether a conversation about a known entity actually connects to it in the graph — is reported on its own line, with its known gap stated openly:

- **1.00** on clean-label mentions (8 turns)
- **0.75** on fragile variants — lowercase, hyphenated, or aliased (4 turns)

The one attachment miss is semantic aliasing ("offline mode" → the roadmap note that plans it): a *concept* mapping to an entity, not a surface form. That's vocabulary work on the roadmap, not a bug we're hiding.

### How to read these numbers honestly

- **These are retrieval numbers, not end-to-end answer quality.** The default reader is a zero-LLM extractive stub, chosen so the benchmark measures *retrieval* cleanly rather than a language model's fluency. End-to-end token-F1 with this stub is low by design (~0.06); swapping in a real LLM reader raises answer quality substantially — but that's the reader's contribution, not Revien's retrieval, so we don't headline it.
- **The adversarial category is a trap for naive scoring.** A system that retrieves *nothing* scores a perfect 1.0 on "refuse to answer" questions, because an empty result correctly produces a refusal. So a broken retriever can post a *higher* adversarial score than a working one. We surface this rather than let it flatter the numbers — it's exactly the kind of metric artifact the honest-numbers discipline exists to catch.
- **The remaining conversational gap is ranking, not coverage.** A per-query miss taxonomy (shipped in the bench) shows the answer is usually *found* — the graph walk reaches it and the scorer scores it — but it lands at a median rank of ~33, outside the top-10 we return. It's in the graph; it just isn't surfaced. Extraction and the walk are near-lossless; ranking is the next lever, and the taxonomy points to exactly where it leaks.

---

## Quick start

### Install

```bash
# From PyPI (semantic layer included as a core dependency)
pip install revien

# From source
git clone https://github.com/lkmconstructs/revien
cd revien
pip install -e .

# Optional extras: LangChain adapter, neural reranker, Leiden clustering
pip install revien[langchain]
pip install revien[all]
```

Semantic retrieval (`sqlite-vec` + `fastembed`) is a **core dependency**, not an extra — graph-only recall is a fraction as good, so it ships on by default. Set `REVIEN_SEMANTIC=0` to force it off, or `REVIEN_SEMANTIC=require` to make a missing/broken layer a hard error instead of a silent degrade.

### Connect Claude Code and start

```bash
revien connect claude-code
revien start
```

The daemon runs on `localhost:7437`, serving the REST API and auto-syncing connected adapters.

### Recall from the terminal

```bash
revien recall "What database did we decide to use?"
```

```
Query: What database did we decide to use?
Found 3 results (85.2ms, 14 nodes examined)

  [1] We decided to deploy the backend on PostgreSQL, not MySQL.
      Type: context | Score: 0.910
  [2] PostgreSQL
      Type: entity | Score: 0.884
  [3] Enterprise tier decision
      Type: decision | Score: 0.803
```

### Use the API

```python
import httpx

httpx.post("http://localhost:7437/v1/ingest", json={
    "source_id": "my-session",
    "content": "We decided to use PostgreSQL for the database layer.",
    "content_type": "conversation",
})

resp = httpx.post("http://localhost:7437/v1/recall", json={
    "query": "What database are we using?",
})
data = resp.json()
if not data["semantic_active"]:
    print("warning: running degraded —", data["semantic_note"])
for r in data["results"]:
    print(f"[{r['node_type']}] {r['label']} ({r['score']:.2f})")
```

---

## Obsidian: a second corpus, in and out

Revien treats an Obsidian vault as a second memory corpus *beside* your conversations — not instead of them. A vault is a knowledge graph a human already drew: `[[wikilinks]]` are edges, headings are chunk boundaries, frontmatter dates are timestamps. Revien reads that structure directly.

```bash
# Connect a vault and ingest it (chunked by heading, wikilinks become edges)
revien connect obsidian --path ~/my-vault
revien sync-vault

# Write Revien's memory BACK into the vault as readable markdown
revien distill-vault
```

- **Ingest** brings your curated notes in as high-confidence, human-authored memory. They outrank machine-extracted claims on conflict.
- **Distill** writes one markdown note per entity into a `Revien/` folder inside your vault — every claim with its provenance, related entities as `[[wikilinks]]`, so Revien's memory threads into your vault's own graph view. It only ever writes inside its own folder, only overwrites files it created, and never re-ingests its own output. Your memory becomes files you can open.

---

## Adapters

| Adapter | What it does | Interface |
|---------|-------------|-----------|
| **Claude Code** | Reads Claude Code session logs (JSONL), auto-syncs on schedule | `revien connect claude-code` |
| **Obsidian** | Ingests a vault chunked by heading; distills memory back out | `revien connect obsidian` |
| **File Watcher** | Watches a directory for new/changed files | `revien connect file-watcher --path DIR` |
| **Generic API** | Pulls conversation data from a REST endpoint | `revien connect api --path URL` |
| **OpenAI / ChatGPT** | Ingests ChatGPT conversation exports | Python: `OpenAIAdapter` |
| **LangChain** | Drop-in `BaseMemory` replacement | Python: `RevienMemory` |
| **Ollama** | Bridges Revien memory to local Ollama models | Python: `OllamaAdapter` |

### Build your own

```python
from revien.adapters.base import RevienAdapter

class MyAdapter(RevienAdapter):
    async def fetch_new_content(self, since):
        # Return a list of {content, content_type, timestamp, metadata}
        ...

    async def health_check(self):
        return True
```

---

## REST API

The daemon exposes a REST API on `localhost:7437`:

| Method | Endpoint | Function |
|--------|----------|----------|
| POST | `/v1/ingest` | Ingest raw content into the graph |
| POST | `/v1/recall` | Query memory (returns results + `semantic_active` / `semantic_note`) |
| GET | `/v1/nodes` | List nodes (filter by type, source) |
| GET | `/v1/nodes/{id}` | Get a node with its edges |
| PUT | `/v1/nodes/{id}` | Update a node |
| DELETE | `/v1/nodes/{id}` | Delete a node and its edges |
| POST | `/v1/sync` | Trigger a manual adapter sync |
| GET | `/v1/health` | Health check |

Interactive docs at `http://localhost:7437/docs` when the daemon is running.

---

## Graph schema

**Node types:** `entity` · `topic` · `decision` · `fact` · `preference` · `event` · `context`

**Edge types:** `related_to` · `decided_in` · `mentioned_by` · `depends_on` · `followed_by` · `contradicts` · `corrects` · `derived_from`

Every ingestion creates a `context` node holding the verbatim interaction; extracted nodes connect back to it. Any fact or decision traces to its origin.

---

## Configuration

Config lives at `~/.revien/config.json`, created on first run. Retrieval is also tunable via environment variables (the scoring knobs the benchmark sweeps):

| Env var | Default | Effect |
|---------|---------|--------|
| `REVIEN_SEMANTIC` | on | `0` disables semantic; `require` makes a broken layer fatal |
| `REVIEN_RECENCY_HALF_LIFE_DAYS` | `365` | Content-recency decay; long by default so old facts aren't buried |
| `REVIEN_TOUCH_ON_RECALL` | off | On restores retrieval-driven frequency (a popularity loop; off by default) |
| `REVIEN_RECENCY_WEIGHT` / `_FREQUENCY_WEIGHT` / `_PROXIMITY_WEIGHT` | `0.35 / 0.30 / 0.35` | Three-factor blend |
| `REVIEN_EXTRACTOR` | `rule` | `ollama` / `openai` / etc. for LLM-based extraction (regex fallback always attached) |
| `REVIEN_INGEST_DENY` | — | Comma-separated source IDs that are never captured |

---

## Architecture

```
Any AI System / Obsidian vault
            │
            ▼
     ┌─────────────┐     ┌──────────────┐
     │  Revien API  │────▶│  Ingestion    │──── extract typed nodes + edges,
     │  (FastAPI)   │     │  Pipeline     │     embed, dedup, govern claims
     └──────┬──────┘     └──────┬───────┘
            │                    ▼
            │            ┌──────────────┐
            │            │  Graph Store  │──── SQLite + sqlite-vec (all local)
            │            │  nodes/edges/ │
            │            │  audit log    │
            ▼            └──────┬───────┘
     ┌─────────────┐           │
     │  Retrieval   │◀──────────┘
     │  semantic-   │──── nearest-by-meaning anchors → graph walk
     │  first +     │      → three-factor refine → top-N
     │  graph walk  │
     └──────┬──────┘
            ▼
   Lean, relevant context  ──▶  distill back to vault (optional)
```

---

## Roadmap

- Reranking to close the ranking gap (the largest remaining recall lever)
- Broader extraction coverage for conversational memory
- Alias/vocabulary resolution (the attachment holdout)
- Note-edit reconciliation for vault re-sync
- Graph visualization and inspection tools

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## About

Revien is the open-source memory layer from [LKM Constructs](https://lkmconstructs.com).

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2026 LKM Constructs LLC.
