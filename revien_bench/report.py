"""
revien_bench.report — Render a results JSON into a Markdown report.

HEADLINE build: Revien's OWN numbers only. The competitor-comparison table
(design §4: Mem0 / LoCoMo-human / Letta) is DEFERRED to the LLM-judge track and
is intentionally NOT emitted here — it depends on track-B judge decisions.

Usage:  python -m revien_bench.report results/<file>.json [> BENCHMARK_RESULTS.md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict


def render(report: Dict) -> str:
    lines = []
    cfg = report.get("config", {})
    ds = report.get("dataset", {})
    env = report.get("environment", {})

    lines.append("# Revien LoCoMo Benchmark Results")
    lines.append("")
    lines.append(f"- **Config:** `{cfg.get('name')}` / answerer `{cfg.get('answerer')}` "
                 f"(cluster={cfg.get('cluster')})")
    lines.append(f"- **Timestamp:** {report.get('timestamp')}")
    lines.append(f"- **Dataset:** `{ds.get('path')}` "
                 f"({ds.get('conversations')} conversations) — SHA-256 `{ds.get('sha256')}`")
    lines.append(f"- **Revien:** v{env.get('revien_version')} · Python {env.get('python')} · "
                 f"{env.get('platform')}")
    lines.append(f"- **Questions scored:** {report.get('n_questions')}")
    lines.append(f"- **Cost:** ${report.get('cost_usd')} · **Network calls:** "
                 f"{report.get('network_calls')}")
    lines.append("")

    # ── Headline F1 ───────────────────────────────────────────────────────────
    lines.append("## End-to-end QA — official LoCoMo token-F1 (PRIMARY)")
    lines.append("")
    lines.append(f"**Overall F1: {report.get('overall_f1')}**")
    lines.append("")
    lines.append("| Category | N | F1 |")
    lines.append("|---|---:|---:|")
    for cat, v in report.get("per_category_f1", {}).items():
        lines.append(f"| {cat} | {v['n']} | {v['f1']} |")
    lines.append("")
    lines.append("> Adversarial (category 5) F1 = hallucination-resistance: 1.0 when the "
                 "answerer correctly refuses (no information available / not mentioned).")
    lines.append("")

    # ── Retrieval quality ─────────────────────────────────────────────────────
    r = report.get("retrieval", {})
    lines.append("## Retrieval quality (decoupled from answerer)")
    lines.append("")
    lines.append("| recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@10 | N (w/ evidence) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {r.get('recall@1')} | {r.get('recall@3')} | {r.get('recall@5')} | "
        f"{r.get('recall@10')} | {r.get('mrr')} | {r.get('ndcg@10')} | "
        f"{r.get('n_with_evidence')} |"
    )
    lines.append("")

    # ── Latency + throughput ──────────────────────────────────────────────────
    lat = report.get("latency_ms", {}).get("recall", {})
    lines.append("## Latency & throughput (sovereignty / on-device)")
    lines.append("")
    lines.append("| recall p50 | p90 | p99 | mean | ingest turns/sec |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {lat.get('p50')}ms | {lat.get('p90')}ms | {lat.get('p99')}ms | "
        f"{lat.get('mean')}ms | {report.get('ingest_turns_per_sec')} |"
    )
    lines.append("")

    # ── Sovereignty ───────────────────────────────────────────────────────────
    sov = report.get("sovereignty", {})
    lines.append("## Sovereignty assertions (THE DIFFERENTIATOR)")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if sov.get('all_passed') else 'FAIL'}**")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for c in sov.get("checks", []):
        detail = ", ".join(f"{k}={v}" for k, v in c.get("detail", {}).items()
                           if not isinstance(v, list))
        result = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"| `{c.get('name')}` | {result} | {detail} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Headline track only: zero-LLM, zero-cloud, deterministic. "
                 "Competitor comparison (Mem0 / LoCoMo-human / Letta) and LLM-judge "
                 "accuracy are deferred to track B._")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    # Ensure UTF-8 stdout so `> BENCHMARK_RESULTS.md` is safe on a Windows
    # console whose default codepage (cp1252) can't encode em-dash/middot.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m revien_bench.report results/<file>.json", file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"ERROR: no such results file: {path}", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    sys.stdout.write(render(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
