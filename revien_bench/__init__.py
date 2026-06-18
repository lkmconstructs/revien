"""
revien_bench — Dev-only LoCoMo benchmark harness for Revien (NOT shipped in the wheel).

Headline / core track ONLY in this build: zero-LLM, zero-cloud, deterministic.
The official LoCoMo token-F1 metric (Snap Research, Maharana et al. ACL 2024,
arXiv:2402.17753) plus retrieval quality (recall@k / MRR / nDCG), sovereignty
assertions ($0 cost, 0 network egress, provenance, audit integrity, consent),
and latency percentiles.

Explicitly DEFERRED to the LLM-judge track (NOT implemented here):
  * judges.py LLM judges (local Ollama + cloud Claude, Cohen's kappa)
  * OllamaAnswerer / ClaudeAnswerer
  * the competitor-comparison table in report.py

This package is a sibling of `revien/` and is intentionally absent from
setup.py's package list, so it never lands in the pip wheel.
"""

__all__ = []
