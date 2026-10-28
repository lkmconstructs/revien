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

## ONNX Runtime's own telemetry (a dependency, switched off by Revien)

Revien's local embedder and reranker run on ONNX Runtime (`onnxruntime`, pulled in by
`fastembed`). **onnxruntime 1.29.0 and later (first released 2026-08-17), on Linux and
macOS, contains Microsoft's telemetry SDK (1DS).** Versions up to 1.28 do not. Unless it is
switched off, importing onnxruntime is enough, no model needs to load: about 9 seconds after
the import the process connects to `https://mobile.events.data.microsoft.com/OneCollector/1.0`,
again a few seconds later, and sends:

- a persistent device id (kept in `~/.cache/Microsoft/DeveloperTools/.onnxruntime/deviceid`),
- OS name and version, CPU model and total memory,
- network type (wired, unmetered) and whether it runs in a container or a VM,
- the full path of the Python executable, which on most machines contains your user name.

Events it has not sent yet (for example from a short command that exits quickly) are queued
in an SQLite file in the same folder (`onnxruntime.db`), and the next onnxruntime process
that runs long enough without the switch sends them. Windows builds report through the OS
instead (TraceLogging/ETW).

**What switches it off.** Only the environment variable `ORT_DISABLE_TELEMETRY`, set
*before onnxruntime is first imported*. ONNX Runtime reads it once, at that import; set later,
it is too late. ONNX Runtime treats `1`, `true`, `yes`, `y` and `on` (any case) as off;
`0`, `false`, `no`, `off` and an empty value leave the telemetry on. It ignores spaces, tabs
and line breaks around the value, but no other character: `1` preceded by a no-break space
leaves the telemetry on. Calling
`onnxruntime.disable_telemetry_events()` does **not** stop it.

**What Revien does.** `import revien` sets `ORT_DISABLE_TELEMETRY=1` before it imports
anything else, so the variable is in place before anything Revien loads can import
onnxruntime, and every process Revien starts inherits it. If you set the variable yourself,
your value is kept (an empty value counts as unset), and Revien reports what it means: a value
ONNX Runtime does not treat as off is reported as telemetry on. Revien also still calls
`disable_telemetry_events()`, which covers the Windows ETW path. Measured on Linux with
onnxruntime 1.31.0: with the switch, a long-running `revien ingest` and `revien recall`
makes 0 connections and creates no `~/.cache/Microsoft` folder (it made 2 connections
without it).

You can see the result: `revien status` prints `ONNX Runtime telemetry: disabled`, or
`NOT disabled:` with the reason and the fix; `GET /v1/semantic/status` and the benchmark's
sovereignty report carry `ort_telemetry_disabled` (true, false, or null when onnxruntime is
not installed) and `ort_telemetry_problem`, and the benchmark's `network_egress_zero` check
fails when the telemetry is not off. Gate G21 (`scripts/gates/check_native_egress.py`)
checks it from outside the process, since a C++ library's connection bypasses Python.

**If your application imports onnxruntime before Revien, set the switch yourself.** When
onnxruntime (directly, or through fastembed or another library) is imported before
`import revien`, Revien is too late to stop it in that process. Set
`ORT_DISABLE_TELEMETRY=1` in the environment, or in `os.environ` before the first
`import onnxruntime`. Revien detects this case: it reports `ort_telemetry_disabled: false`
and logs one warning naming the fix.

**Events queued before you upgraded.** If onnxruntime 1.29.0 or later ran on this machine without the
switch (an earlier Revien, or another program), events may be waiting in
`~/.cache/Microsoft/DeveloperTools/.onnxruntime/`. Revien's processes do not send them
(with the switch, the SDK does not run), but any onnxruntime process that lacks the switch
will. You can delete that folder to drop them. Other applications that use onnxruntime share
it, so Revien does not delete it for you.

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
