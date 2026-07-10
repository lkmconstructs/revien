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
  * ``REVIEN_RERANK`` gates activation. DEFAULT OFF — this ships as a knob
    for the sweep to measure; the default flips only on a measured win, per
    house rule. ``REVIEN_RERANK=1`` enables; anything else disables.
  * Model load is OFFLINE-FIRST via the per-call ``local_files_only``
    parameter (warm cache = zero network), falling back to a one-time
    download on a cold cache — same rationale as FastEmbedProvider.
  * Runtime failure DISABLES the layer loudly (stderr once) and recall
    continues un-reranked — degraded, never broken, never silent.

Knobs:
    REVIEN_RERANK          1 to enable (default 0)
    REVIEN_RERANK_MODEL    cross-encoder model (default ms-marco-MiniLM-L-6-v2,
                           ~80MB, CPU ONNX)
    REVIEN_RERANK_TOP_K    how many head results to rescore (default 30)
"""

import importlib.util
import os
import sys
from typing import Callable, List, Optional, Sequence

_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 30


def _rerank_enabled_by_env() -> bool:
    return os.environ.get("REVIEN_RERANK", "0").strip().lower() in (
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
            return "not enabled (REVIEN_RERANK unset)"
        if not _FASTEMBED_AVAILABLE:
            return "fastembed not installed"
        return "disabled"

    def _ensure_model(self) -> None:
        if self._model is not None or self._scorer is not None:
            return
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
