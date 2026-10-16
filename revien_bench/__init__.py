"""
revien_bench — Dev-only LoCoMo benchmark harness for Revien (NOT shipped in the wheel).

Headline / core track: zero-LLM, zero-cloud, deterministic. The official
LoCoMo token-F1 metric (Snap Research, Maharana et al. ACL 2024,
arXiv:2402.17753) plus retrieval quality (recall@k / MRR / nDCG), sovereignty
assertions ($0 cost, 0 network egress, provenance, audit integrity, consent),
and latency percentiles.

Optional end-to-end LLM-judge track (judges.py, --judge on the runner):
a SEPARATE, never-blended binary CORRECT/WRONG accuracy score from an LLM
comparing each prediction to the LoCoMo gold answer, mirroring the LLM reader
track's transport/disclosure/cost-estimate/frozen-prompt pattern (local
Ollama or a disclosed cloud provider — openai/openrouter/together/claude).
Defaults to 'f1' (no LLM judge; the report shape is then unchanged from the
headline track).

Still DEFERRED (NOT implemented here):
  * the competitor-comparison table in report.py (Mem0 / LoCoMo-human / Letta)
  * Cohen's kappa / inter-judge agreement

This package is a sibling of `revien/` and is intentionally absent from
setup.py's package list, so it never lands in the pip wheel.
"""

__all__ = []
