"""
Revien Cross-Encoder Reranker — OPT-IN, LOCAL-FIRST head reranking.

WHY (A1, measured July 10 2026): with semantic-as-spine, the entire ranking
battle happens INSIDE the semantic anchor set — on both corpora the top-20 is
100% distance-0 anchors, so the `outranked` miss bucket (72% of misses,
median best rank 26) is anchor-vs-anchor bi-encoder similarity misranking.
Graph-side levers (hop decay, edge-weight blend) were measured inert there.
The lever that CAN reorder the head is a cross-encoder: it reads query and
candidate TOGETHER instead of comparing independently-computed embeddings,
trading a little per-query latency for a fundamentally stronger relevance
judgment over a small head.

Contract (mirrors the semantic spine's discipline):
  * Imports stay GUARDED — availability is checked via find_spec WITHOUT
    importing fastembed at module load (a wedged hub endpoint once turned an
    import into a 23-minute hang; see semantic/index.py).
  * ``REVIEN_RERANK`` gates activation. DEFAULT ON (July 11 2026 —
    "smarter by default"): the measured default is the int8 model at
    depth 20 — recall@10 0.5141 -> ~0.64, recall@1 doubled, ~200ms p50 —
    per the round-5/5b depth+quantization profile. ``REVIEN_RERANK=0``
    opts out and restores the pre-rerank 85ms path byte-identically.
  * Model load is OFFLINE-FIRST via the per-call ``local_files_only``
    parameter (warm cache = zero network), falling back to a one-time
    download on a cold cache — same rationale as FastEmbedProvider.
  * Runtime failure DISABLES the layer loudly (stderr once) and recall
    continues un-reranked — degraded, never broken, never silent.

Knobs:
    REVIEN_RERANK          0 to opt out (default 1)
    REVIEN_RERANK_MODEL    cross-encoder model (default: int8-quantized
                           ms-marco-MiniLM-L-6-v2, 23MB CPU ONNX; set to
                           Xenova/ms-marco-MiniLM-L-6-v2 for fp32)
    REVIEN_RERANK_TOP_K    how many head results to rescore (default 20;
                           30 + REVIEN_SEMANTIC_TOP_K=100 is quality mode)
"""

import importlib.util
import os
import sys
from typing import Callable, List, Optional, Sequence

_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None

# The fp32 source repo — fastembed's registry entry, and the weights the
# int8 variant quantizes. Selectable explicitly for parity checks.
RERANK_SOURCE_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# int8 sibling: SAME weights repo, quantized ONNX file (23MB vs 87MB).
# Bench-verified parity with fp32 (recall@10 -0.0017, IDENTICAL miss
# taxonomy) at 1.27-1.38x. Not in fastembed's registry, so it is registered
# as a custom model on first use.
INT8_RERANK_MODEL = "revien/ms-marco-MiniLM-L-6-v2-int8"

# Shipped defaults (round-5/5b measured shape): int8 at depth 20 —
# ~200ms p50 for +8pts recall@10 and doubled recall@1 over no-rerank.
DEFAULT_RERANK_MODEL = INT8_RERANK_MODEL
DEFAULT_RERANK_TOP_K = 20


def _register_builtin_variants() -> None:
    """Idempotently register revien's custom reranker variants with fastembed."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from fastembed.common.model_description import ModelSource
    try:
        TextCrossEncoder.add_custom_model(
            model=INT8_RERANK_MODEL,
            sources=ModelSource(hf=RERANK_SOURCE_MODEL),
            model_file="onnx/model_quantized.onnx",
            size_in_gb=0.023,
        )
    except Exception:  # noqa: BLE001 - already registered (re-init) is fine
        pass


def _rerank_enabled_by_env() -> bool:
    # DEFAULT ON — "smarter by default" (July 11 2026). REVIEN_RERANK=0
    # opts out and restores the pre-rerank latency path byte-identically.
    return os.environ.get("REVIEN_RERANK", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


class CrossEncoderReranker:
    """Rescores a small head of candidates with a local cross-encoder.

    Inert by default: is_enabled is False unless REVIEN_RERANK=1 AND
    fastembed is importable. All failure modes degrade to pass-through.

    ``scorer`` is injectable for tests: a callable
    (query, [texts]) -> [scores]. Production default lazily loads
    fastembed's TextCrossEncoder.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        scorer: Optional[Callable[[str, Sequence[str]], List[float]]] = None,
    ):
        self.model_name = model_name or os.environ.get(
            "REVIEN_RERANK_MODEL", DEFAULT_RERANK_MODEL
        )
        try:
            self.top_k = int(top_k if top_k is not None else os.environ.get(
                "REVIEN_RERANK_TOP_K", DEFAULT_RERANK_TOP_K
            ))
        except ValueError:
            self.top_k = DEFAULT_RERANK_TOP_K
        self._scorer = scorer
        self._model = None
        self._runtime_failure: Optional[str] = None
        self._enabled = scorer is not None or (
            _rerank_enabled_by_env() and _FASTEMBED_AVAILABLE
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._runtime_failure is None

    def inactive_reason(self) -> Optional[str]:
        if self.is_enabled:
            return None
        if self._runtime_failure is not None:
            return f"disabled after runtime error: {self._runtime_failure}"
        if not _rerank_enabled_by_env() and self._scorer is None:
            return "opted out (REVIEN_RERANK=0)"
        if not _FASTEMBED_AVAILABLE:
            return "fastembed not installed"
        return "disabled"

    def _ensure_model(self) -> None:
        if self._model is not None or self._scorer is not None:
            return
        _register_builtin_variants()
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        # OFFLINE-FIRST via the per-call parameter, NOT HF_HUB_OFFLINE (which
        # is read into a module constant at import time and would permanently
        # break the cold-cache fallback — see FastEmbedProvider._ensure_model).
        try:
            self._model = TextCrossEncoder(
                model_name=self.model_name, local_files_only=True
            )
        except Exception:
            self._model = TextCrossEncoder(model_name=self.model_name)

    def _score(self, query: str, texts: Sequence[str]) -> List[float]:
        if self._scorer is not None:
            return list(self._scorer(query, texts))
        self._ensure_model()
        return [float(s) for s in self._model.rerank(query, list(texts))]

    def rerank(self, query: str, results: list) -> list:
        """Reorder the head of a scored result list by cross-encoder relevance.

        ``results`` is the engine's score-sorted list of RetrievalResult.
        The top ``top_k`` entries are rescored against ``query`` on their
        CONTENT and re-sorted by the cross-encoder score; the tail keeps its
        original order below the head. Each rescored result gets the raw
        cross-encoder score in score_breakdown["rerank_score"] — the .score
        field is NOT overwritten (the base score stays auditable).

        Any runtime failure disables the layer for the process (loudly) and
        returns the input unchanged.
        """
        if not self.is_enabled or len(results) <= 1:
            return results
        head = results[: self.top_k]
        tail = results[self.top_k:]
        try:
            scores = self._score(query, [r.content for r in head])
        except Exception as exc:  # degrade loudly, never break recall
            self._runtime_failure = repr(exc)
            sys.stderr.write(
                f"[revien.rerank] DISABLED after runtime error: {exc!r}. "
                f"Recall continues un-reranked.\n"
            )
            sys.stderr.flush()
            return results
        for r, s in zip(head, scores):
            r.score_breakdown["rerank_score"] = round(s, 4)
        head.sort(
            key=lambda r: r.score_breakdown["rerank_score"], reverse=True
        )
        return head + tail
