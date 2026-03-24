"""
Revien Neural Scorer — Learned relevance scoring model (SCAFFOLDED).
Learns from user retrieval patterns to adjust edge weights.
Interface is stable; training activates post-MVP.
"""

import logging
from typing import Optional

logger = logging.getLogger("revien.neural")


class NeuralScorer:
    """
    Lightweight scoring model that learns from retrieval patterns.
    Observes which nodes are used after retrieval (positive signal)
    and which are ignored (negative signal).

    SCAFFOLDED — logs signals for future training, does not yet adjust scores.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._model_available = False

        if model_path:
            self._try_load_model(model_path)

    def _try_load_model(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(model_path)
            self._model_available = True
            logger.info(f"Neural scorer loaded from {model_path}")
        except (ImportError, Exception) as e:
            logger.info(f"Neural scorer not available: {e}")

    def adjust_score(self, base_score: float, node_id: str, query: str) -> float:
        """
        Adjust a base score using the learned model.
        Returns the base score unmodified if model is not available.

        POST-MVP: Will use the trained model to boost/penalize scores
        based on learned user patterns.
        """
        if self._model_available and self._model is not None:
            # POST-MVP: Run inference to get adjustment factor
            pass
        return base_score

    @property
    def is_neural(self) -> bool:
        return self._model_available
