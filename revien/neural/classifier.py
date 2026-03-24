"""
Revien Neural Classifier — ONNX-based node classifier (SCAFFOLDED).
Replaces rule-based extraction when the optional model file is installed.
Interface is stable; implementation activates post-MVP.
"""

import logging
from typing import Optional

from revien.ingestion.extractor import ExtractionResult, RuleBasedExtractor

logger = logging.getLogger("revien.neural")


class NeuralClassifier:
    """
    Neural node classifier using ONNX runtime.
    Falls back to rule-based extraction when model is not available.

    Usage:
        classifier = NeuralClassifier(model_path="~/.revien/models/classifier.onnx")
        result = classifier.extract(content, source_id)

    The extract() method returns the same ExtractionResult format as
    RuleBasedExtractor, so the graph doesn't know or care which method was used.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._fallback = RuleBasedExtractor()
        self._model_available = False

        if model_path:
            self._try_load_model(model_path)

    def _try_load_model(self, model_path: str) -> None:
        """Attempt to load the ONNX model. Fail silently to rule-based fallback."""
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(model_path)
            self._model_available = True
            logger.info(f"Neural classifier loaded from {model_path}")
        except ImportError:
            logger.info("onnxruntime not installed. Using rule-based extraction.")
        except Exception as e:
            logger.info(f"Could not load model: {e}. Using rule-based extraction.")

    def extract(self, content: str, source_id: str = "") -> ExtractionResult:
        """
        Extract nodes and edges from content.
        Uses neural model if available, otherwise falls back to rule-based.
        """
        if self._model_available and self._model is not None:
            return self._neural_extract(content, source_id)
        return self._fallback.extract(content, source_id)

    def _neural_extract(self, content: str, source_id: str) -> ExtractionResult:
        """
        Neural extraction path. SCAFFOLDED — not implemented in MVP.
        When implemented, this will:
        1. Tokenize content
        2. Run through ONNX classifier
        3. Map output labels to NodeType
        4. Build nodes and edges from classified spans
        """
        # POST-MVP: Replace with actual neural inference
        # For now, fall back to rule-based
        logger.debug("Neural extraction not yet implemented, using rule-based fallback")
        return self._fallback.extract(content, source_id)

    @property
    def is_neural(self) -> bool:
        """Whether the neural model is active."""
        return self._model_available
