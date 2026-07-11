# Telemetry

**Revien collects nothing.**

No usage analytics. No crash reporting. No phone-home. No license check. No "anonymous"
metrics. There is no telemetry to opt out of, because there is none to begin with.

## What runs locally, and only locally

On the default path, Revien makes **zero network calls**. This is not a promise — it is
asserted by the benchmark suite, which fails if any network egress is detected during a run
(`sovereignty: network_egress_zero`). Every published benchmark reports `network_calls: 0`.

- **Extraction** runs on-device (regex by default; a local Ollama model if you opt in).
- **Embeddings** run on-device (`fastembed` / `bge-small`). Model files load offline-first
  from a local cache — a warm install contacts no server, not even for metadata.
- **Storage** is a single local SQLite file you own.

## The only ways text leaves your machine

...are ones you explicitly turn on, and each discloses itself once, to stderr, before the
first byte leaves:

- `REVIEN_EXTRACTOR=openai` (or another cloud backend) — sends content to that provider for
  extraction.
- `REVIEN_EMBEDDER=openai` — sends content to OpenAI for embeddings.
- A cloud reader in the benchmark harness (`--answerer openai:...`) — dev/eval only.

If none of these are set, nothing you feed Revien ever leaves the device.

## First install

On first run, `fastembed` downloads the `bge-small` model (~65MB) once from Hugging Face,
then caches it locally. Every run after that is fully offline. If you need a fully
air-gapped install, pre-seed the model cache and set `HF_HUB_OFFLINE=1`.
