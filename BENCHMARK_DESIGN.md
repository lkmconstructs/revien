# Revien L7 Benchmark — Design Spec

> Source: research workflow `wf_01b3d4ce-2fe` (5 scouts + adversarial verify + synthesis), 2026-06-18.
> Verified facts only; hallucinated sources rejected (see §0).

## 0. Grounding correction (read first)

The research surfaced two hallucinations this design explicitly rejects:
- A scout claimed "LoCoMo is Anthropic's benchmark, LLM-judged." **False.** LoCoMo = **Snap Research**, Maharana et al., ACL 2024 (arXiv:2402.17753); official QA metric is **token-level F1 with normalization, NOT LLM-as-judge** (`task_eval/evaluate_qa.py`).
- A "verifier" refuted the 7,512 QA-count citing arXiv:2604.11563 ("Synthius-Mem") — **that paper ID does not exist; hallucinated.** The 7,512 total / 2,705 single-hop / 1,871 adversarial figures ARE confirmed against the real 2402.17753 and are what we use.

Mem0's "92.5%" is real but is **Mem0's own paper (arXiv:2504.19413), vendor-run, LLM-judge** — a different metric than official LoCoMo F1. §4 handles this apples-to-oranges.

Decision: build **two scoring tracks** — (A) official LoCoMo F1 (comparable to canonical leaderboard) and (B) optional LLM-judge (comparable to Mem0's number) — report both, disclose which is which. Plus the sovereignty metrics that are Revien's reason to exist.

## 1. Dataset
- Source: `github.com/snap-research/locomo` → `data/locomo10.json` (LoCoMo-10: 10 conversations, ~26k tokens each, ~1.5–1.9k QA total).
- Schema: array of conversations; each `conversation.session_N` = turns `{speaker, text, dia_id, [img_url/blip_caption]}` + `session_N_date_time`; `qa` = `{question, answer, evidence:[dia_id], category}` (1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial→refusal). `event_summary`/`observation` ignored (out of scope).
- Local load: fetch once via `python -m revien_bench.fetch_locomo`, pin by **SHA-256** in `revien_bench/DATASET.lock` (reproducible, detects drift). NOT vendored (licensing). Images dropped (text-only system; keep BLIP caption text only).

## 2. Metrics
**A. End-to-end QA — official LoCoMo F1 (PRIMARY/headline).** Token-F1 vs gold w/ official normalization, per-category + overall. Deterministic, no judge variance, comparable to real leaderboard. Adversarial (cat 5): "correct" = answer normalizes to refusal/"not mentioned" → measures **hallucination resistance** (Revien can refuse from absence-of-node).
**B. End-to-end QA — LLM-judge accuracy (SECONDARY, disclosed, Mem0-comparability only).** Binary correct/incorrect per Mem0-style rubric; two judges — **local (Ollama)** + **cloud (Claude)** — report inter-judge Cohen's κ. **Off by default** (`--judge f1`); default run is pure F1, zero cloud, $0.
**C. Retrieval quality (decoupled from answerer).** Using LoCoMo `evidence` dia_ids as ground-truth → recall@k (1/3/5/10), MRR, nDCG@10. Reported graph-only / +semantic / +neural. Isolates retrieval from answer composition.
**D. Sovereignty + cost/latency (THE DIFFERENTIATOR — equal weight to A).** Latency (ingest turns/sec; recall p50/p90/p99, base vs semantic vs neural); **cost USD/run (default $0)**; **network egress bytes (default 0, asserted)**; provenance completeness (% answer-supporting nodes with traceable lineage, target 100%); audit integrity (append-only verified); consent/retention sub-test (`REVIEN_INGEST_DENY` + soft-invalidation recoverability).

## 3. Harness architecture
Per conversation, fresh isolated `GraphStore` (no cross-conv leakage): ingest each turn (metadata carries `dia_id`+session date; `timestamp`=session date for real temporal signal) → optional cluster/neural → per QA: `engine.recall(q, top_n=K, now=last_session_date)` → assemble top-K context → answer → score (F1, retrieval-hit, latency, tokens, cost).
- **Answerer (the reader), pluggable+disclosed:** `extractive` (default, **zero-LLM**, deterministic span selection — isolates retrieval); `ollama:<model>` (local); `claude` (cloud). Same context to all so the only variable is the reader.
- **Judges (track B only):** local `ollama:qwen2.5:14b` / cloud `claude` (discloses, prints model id); temp=0; frozen rubric in `prompts/judge.txt` (hash recorded).
- **Reproducibility:** pinned dataset SHA + revien git-sha + embedder model/revision + seeds; deterministic default path; per-run `results/<ts>_<config>.json` (full config, hashes, per-category F1, recall@k/MRR/nDCG, latency percentiles, cost, network_calls, sovereignty pass/fail, per-question rows). `--config {graph_only|semantic|neural|full}` via env vars.

## 4. Competitor framing (published numbers only, no re-running)

| System | Number | Metric | Provenance | README note |
|---|---|---|---|---|
| LoCoMo human | ~87.9% | F1 | paper-reported (2402.17753) | upper reference |
| **Mem0** | 92.5% | **LLM-judge**, 300Q | vendor-run (2504.19413) | cloud-embedding; NOT official F1; their judge |
| Mem0^g (graph) | 68.44% | LLM-judge | vendor-run | their graph variant scores *lower* than vector |
| **Letta/MemGPT** | **no LoCoMo #** | — | confirmed absent (2310.08560: DMR/KV/DocQA) | any "Letta on LoCoMo" is third-party |
| **"Hermes"** | **none** | — | not a memory benchmark (NemoClaw runtime) | omit / footnote it doesn't exist |
| **Revien** | *this harness* | official F1 + recall@k + $0/0-egress | self-run, not third-party-audited | our honest claim |

**Revien expected to WIN (state plainly):** zero cloud, $0/run, 0 egress, full provenance/lineage per answer, append-only audit, consent-governed retention + soft-invalidation/recovery, on-device latency, no lock-in; plausibly strong hallucination resistance (adversarial).
**Cloud-embedding (Mem0) likely LEADS:** raw single/multi-hop recall F1. We expect graph-only F1 to **trail** and will NOT hide it — show graph-only vs +semantic to quantify how much the local embedder closes the gap. All categories, both judges, named configs. No cherry-picking.

## 5. Build plan — `revien_bench/` (sibling to `revien/`, dev-only, not in wheel)
```
revien_bench/{__init__, DATASET.lock, fetch_locomo, loader, ingest_locomo,
  answerers, judges, metrics, sovereignty, runner, report}.py
  prompts/{judge.txt, answerer.txt}   configs/{graph_only,semantic,neural,full}.json
tests/test_bench_smoke.py             # 3-QA synthetic conv, default config, no network
data/                                 # gitignored; locomo10.json lands here
BENCHMARK.md                          # public methodology + results + caveats
```
Run: `python -m revien_bench.fetch_locomo` then `python -m revien_bench.runner --config graph_only --answerer extractive --judge f1 --out results/` (headline, local, deterministic, $0, zero-egress). Cloud track: `--answerer claude --judge claude` (discloses + records cost/egress). `python -m revien_bench.report results/<file>.json > BENCHMARK_RESULTS.md`.

## 6. Risks/caveats (disclose in BENCHMARK.md)
1. Self-run, not third-party-audited (publish harness + dataset hash + per-Q results for reproducibility).
2. F1 ≠ LLM-judge — never compare across them; track B exists only for judge-vs-judge.
3. LLM-judge variance (mitigate: temp=0, frozen hashed prompt, two judges, report κ; still indicative not definitive).
4. Reader-LLM confound (mitigate: decoupled recall@k + default zero-LLM extractive answerer).
5. Apples-to-oranges ingestion (competitors chunk/summarize differently; directional comparison).
6. Dataset licensing — fetch+checksum not vendor; honor Snap's research terms; images dropped.
7. Expected to trail on raw recall — stated up front; benchmark quantifies the gap AND shows where local-first wins.
8. Adversarial scoring is a refusal-matcher proxy (publish the matcher).
9. Hallucinated-source warning baked in (LoCoMo=Snap not Anthropic; Letta/Hermes have no LoCoMo numbers).
10. Hardware variance (record CPU/RAM/OS; report percentiles).

**Net:** headline run = local, deterministic, $0, zero-egress, official-F1 + recall@k + sovereignty. LLM-judge track exists only to stand beside Mem0's figure with caveats. Win framed where real (sovereignty/cost/audit/local); expected recall deficit stated openly.
