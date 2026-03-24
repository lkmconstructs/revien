# Revien

**Memory that returns.**

Revien is a graph-based memory engine for AI systems. It gives any AI tool — local models, Claude Code, API-based assistants, agent frameworks — persistent memory across sessions. No GPU required. No cloud account needed. Nothing is compacted away by default.

```bash
pip install revien
revien connect claude-code
revien start
```

That's it. Revien starts building persistent memory.

---

## Why Revien?

Most AI tools still lose continuity between sessions. Most memory systems either forget too much, hide too much, or force developers to assemble the pipeline themselves.

| Approach | Problem |
|----------|---------|
| **RAG / Vector DBs** | Requires GPU for embeddings. Developer builds the pipeline. |
| **Platform memory** (Claude, ChatGPT) | Vendor-locked. Opaque. Non-portable. |
| **LangChain Memory** | Compacts and summarizes. Irreversible information loss. |
| **Mem0** | Optimized for lightweight memory, but not graph-native relational structure. |
| **Context compaction** | Asks the LLM to decide what to forget. Burns tokens on housekeeping. |

Revien takes a different approach: **store everything as a graph, compact nothing, retrieve surgically.**

Every piece of context becomes a node in a knowledge graph with typed relationships to other nodes. When your AI needs context, Revien walks the graph and returns only what's relevant — scored by recency, frequency, and relationship proximity. The full history remains preserved in the graph. Nothing is ever summarized away.

---

## How It Works

### Memory is a graph, not a vector store

When you feed Revien a conversation, it extracts:
- **Entities** — people, projects, tools, organizations
- **Decisions** — choices that were made and why
- **Facts** — specific data points, configurations, values
- **Topics** — recurring themes across conversations
- **Preferences** — how you like things done
- **Events** — things that happened at specific times

These become nodes. Relationships between them become edges. The graph grows over time but retrieval stays fast because you're walking edges, not scanning embeddings.

### Three-factor retrieval scoring

When you query Revien, every candidate node gets scored on three dimensions:

- **Recency** — when was this last relevant? (exponential decay)
- **Frequency** — how often does this come up? (logarithmic, diminishing returns)
- **Proximity** — how many graph edges from the query anchor? (hop distance)

The composite score determines what gets surfaced. Only the top results are returned — your AI gets a lean, relevant context window instead of a bloated dump.

### Self-reinforcing memory

Every time a node is retrieved, its access count increases. This boosts its frequency score in future queries. Memory that's actually useful becomes easier to find over time — automatically, with no ML model, no training step, no user intervention.

---

## Quick Start

### Install

```bash
pip install revien
```

### Start the daemon

```bash
revien start
```

This launches the Revien daemon on `localhost:7437`. It serves the REST API and runs the auto-sync scheduler.

### Connect to Claude Code

```bash
revien connect claude-code
```

Revien reads local session logs and indexes them into the graph. Every new session gets ingested automatically.

### Query from the terminal

```bash
revien recall "What database did we decide to use?"
```

Returns scored results from your memory graph:

```
1. [decision] Enterprise tier at $499/month, 20% annual discount, PostgreSQL
   Score: 0.89 | Recency: 1.00 | Frequency: 0.63 | Proximity: 1.00

2. [entity] PostgreSQL
   Score: 0.84 | Recency: 1.00 | Frequency: 0.46 | Proximity: 1.00
```

### Use the API

```python
import httpx

# Ingest a conversation
httpx.post("http://localhost:7437/v1/ingest", json={
    "source_id": "my-session",
    "content": "We decided to use PostgreSQL for the database layer.",
    "content_type": "conversation",
})

# Recall relevant memory
response = httpx.post("http://localhost:7437/v1/recall", json={
    "query": "What database are we using?"
})
for result in response.json()["results"]:
    print(f"[{result['node_type']}] {result['label']} (score: {result['score']:.2f})")
```

---

## Adapters

Revien connects to AI systems through adapters. Three ship with the package:

| Adapter | What it does |
|---------|-------------|
| **Claude Code** | Reads Claude Code session logs (JSONL). Auto-syncs on schedule. |
| **File Watcher** | Watches a directory for new/changed files. Ingests on change. |
| **Generic API** | Connects to any REST endpoint returning conversation data. |

### Connect an adapter

```bash
# Claude Code (auto-detects log location)
revien connect claude-code

# Watch a directory
revien connect file-watcher --path /path/to/conversations/

# Generic API endpoint
revien connect api --url https://your-system.com/api/conversations --header "Authorization: Bearer ..."
```

### Build your own adapter

```python
from revien.adapters.base import RevienAdapter

class MyAdapter(RevienAdapter):
    async def fetch_new_content(self, since):
        # Return list of {content, content_type, timestamp, metadata}
        ...

    async def health_check(self):
        return True
```

---

## REST API

The daemon exposes a full REST API on `localhost:7437`:

| Method | Endpoint | Function |
|--------|----------|----------|
| POST | `/v1/ingest` | Ingest raw content into the graph |
| POST | `/v1/recall` | Query memory with three-factor scoring |
| GET | `/v1/nodes` | List nodes (filter by type, date, source) |
| GET | `/v1/nodes/{id}` | Get a specific node with edges |
| PUT | `/v1/nodes/{id}` | Update a node |
| DELETE | `/v1/nodes/{id}` | Delete a node and its edges |
| GET | `/v1/graph` | Export full graph as JSON |
| POST | `/v1/graph/import` | Import graph from JSON |
| POST | `/v1/sync` | Trigger manual sync |
| GET | `/v1/health` | Health check |

Interactive docs at `http://localhost:7437/docs` when the daemon is running.

---

## Graph Schema

### Node Types

`entity` · `topic` · `decision` · `fact` · `preference` · `event` · `context`

### Edge Types

`related_to` · `decided_in` · `mentioned_by` · `depends_on` · `followed_by` · `contradicts`

Every ingestion creates a `context` node representing the full interaction. All extracted nodes connect back to it. You can always trace any fact or decision back to the conversation where it originated.

---

## Architecture

```
Any AI System
     │
     ▼
┌─────────────┐     ┌──────────────┐
│  Revien API  │────▶│  Ingestion    │──── extract nodes + edges
│  (FastAPI)   │     │  Engine       │     from raw content
└──────┬──────┘     └──────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│  Retrieval   │◀───│  Graph Store  │──── SQLite (local)
│  Engine      │     │  (nodes +     │     PostgreSQL (hosted)
│  (3-factor)  │     │   edges)      │
└─────────────┘     └──────────────┘
       │
       ▼
  Scored results
  (top N nodes)
       │
       ▼
  Any AI's context window
  (lean, relevant, surgical)
```

---

## Early Benchmarks

From 5 sample conversations (60 nodes, 147 edges):

| Metric | Value |
|--------|-------|
| Average retrieval time | 38.75ms |
| Queries under 50ms | 67% |
| Queries under 100ms | 100% |
| Hit rate (relevant results) | 87% (13/15) |
| Zero GPU | ✓ |
| Zero cloud dependency | ✓ |

The two misses were intentionally vague queries with no extractable entities ("Tell me about our architecture").

---

## Configuration

Config lives at `~/.revien/config.json`. Created automatically on first run.

```json
{
  "daemon": {
    "host": "127.0.0.1",
    "port": 7437
  },
  "sync": {
    "interval_hours": 6
  },
  "retrieval": {
    "max_results": 5,
    "max_hops": 3,
    "recency_weight": 0.35,
    "frequency_weight": 0.30,
    "proximity_weight": 0.35,
    "recency_half_life_days": 7
  },
  "adapters": []
}
```

All retrieval weights are configurable. Adjust to your use case — boost recency for fast-moving projects, boost frequency for stable knowledge bases.

---

## Roadmap

- Additional adapters for popular AI tools
- Improved vague-query handling
- Hosted deployment option
- Graph visualization and inspection tools
- Performance tuning and scale improvements

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## About

Revien is the open-source memory layer from [LKM Constructs](https://lkmconstructs.com).

## License

Apache 2.0 — see [LICENSE](LICENSE).

Copyright 2026 LKM Constructs LLC.
