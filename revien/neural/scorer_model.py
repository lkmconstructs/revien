"""
Revien Neural Scorer — Learned relevance scoring model.
Learns from user retrieval patterns to adjust edge weights.

Uses a lightweight TF-IDF + linear model trained on retrieval signals.
No GPU required — runs on CPU with numpy/sklearn.

OPT-IN EXTRA: numpy and scikit-learn are NOT base dependencies. They install
via `pip install revien[neural]`. This module guards its heavy imports so the
package imports cleanly when they are absent — in that case the neural scorer
is silently disabled and base three-factor + community + confidence scoring
runs unchanged. See setup.py extras_require["neural"].
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

# Guarded heavy import — numpy is part of the optional `neural` extra.
# If it (or sklearn at train time) is missing, the scorer disables itself
# and adjust_score() becomes a pass-through.
try:
    import numpy as np  # noqa: F401
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    np = None
    _NUMPY_AVAILABLE = False

logger = logging.getLogger("revien.neural")


class NeuralScorer:
    """
    Lightweight scoring model that learns from retrieval patterns.

    Architecture:
    - Query/node labels → TF-IDF vectors (sklearn)
    - Concatenated vectors → Linear layer → score adjustment

    Training signals:
    - Positive: Node was used after retrieval
    - Negative: Node was retrieved but ignored

    The model learns which query-node patterns correlate with actual usage.

    When numpy/sklearn are not installed (the `neural` extra is absent), the
    scorer stays inert: is_neural is False and adjust_score() returns the base
    score unchanged.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or Path.home() / ".revien" / "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._vectorizer = None
        self._weights = None
        self._bias = 0.0
        self._model_available = False

        # Without numpy there is nothing to load or run — stay disabled.
        if _NUMPY_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Load trained model from disk if available."""
        vectorizer_path = self.model_dir / "vectorizer.pkl"
        weights_path = self.model_dir / "weights.npz"

        if vectorizer_path.exists() and weights_path.exists():
            try:
                with open(vectorizer_path, "rb") as f:
                    self._vectorizer = pickle.load(f)

                data = np.load(weights_path)
                self._weights = data["weights"]
                self._bias = float(data["bias"])
                self._model_available = True
                logger.info("Neural scorer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load neural scorer: {e}")
                self._model_available = False

    def adjust_score(self, base_score: float, node_label: str, query: str) -> float:
        """
        Adjust a base score using the learned model.

        Args:
            base_score: The three-factor composite score
            node_label: The label of the candidate node
            query: The retrieval query

        Returns:
            Adjusted score (clamped to [0, 1]). Returns base_score unchanged
            when the neural extra is absent or no model is trained.
        """
        if not self._model_available or self._weights is None:
            return base_score

        try:
            # Vectorize query and node
            combined = f"{query} [SEP] {node_label}"
            vec = self._vectorizer.transform([combined]).toarray()[0]

            # Linear projection to get adjustment
            raw_adjustment = np.dot(vec, self._weights) + self._bias

            # Sigmoid to bound adjustment to [-0.3, 0.3]
            adjustment = 0.3 * (2 / (1 + np.exp(-raw_adjustment)) - 1)

            adjusted = base_score + adjustment
            return float(np.clip(adjusted, 0.0, 1.0))

        except Exception as e:
            logger.debug(f"Score adjustment failed: {e}")
            return base_score

    def train(self, training_data: List[Dict]) -> bool:
        """
        Train the model on accumulated retrieval signals.

        Args:
            training_data: List of dicts with keys:
                - query: str
                - node_label: str
                - base_score: float
                - was_used: bool

        Returns:
            True if training succeeded. Returns False (and logs) if the neural
            extra (numpy/sklearn) is not installed.
        """
        if not _NUMPY_AVAILABLE:
            logger.warning(
                "numpy not installed. Neural training disabled "
                "(install with: pip install revien[neural])."
            )
            return False

        if len(training_data) < 50:
            logger.info(f"Not enough training data: {len(training_data)}/50 minimum")
            return False

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            # Prepare training examples
            texts = []
            labels = []

            for item in training_data:
                combined = f"{item['query']} [SEP] {item.get('node_label', item.get('node_id', ''))}"
                texts.append(combined)
                labels.append(1 if item["was_used"] else 0)

            # Fit vectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
            X = self._vectorizer.fit_transform(texts)
            y = np.array(labels)

            # Train logistic regression
            model = LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
            )
            model.fit(X, y)

            # Extract weights
            self._weights = model.coef_[0]
            self._bias = model.intercept_[0]
            self._model_available = True

            # Save model
            self._save_model()

            logger.info(f"Neural scorer trained on {len(training_data)} examples")
            return True

        except ImportError:
            logger.warning(
                "sklearn not installed. Neural training disabled "
                "(install with: pip install revien[neural])."
            )
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def _save_model(self) -> None:
        """Persist trained model to disk."""
        try:
            vectorizer_path = self.model_dir / "vectorizer.pkl"
            weights_path = self.model_dir / "weights.npz"

            with open(vectorizer_path, "wb") as f:
                pickle.dump(self._vectorizer, f)

            np.savez(weights_path, weights=self._weights, bias=self._bias)

            logger.info(f"Model saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Could not save model: {e}")

    @property
    def is_neural(self) -> bool:
        """Whether the neural model is active."""
        return self._model_available

    def get_stats(self) -> Dict:
        """Return model statistics."""
        if not _NUMPY_AVAILABLE:
            return {"status": "unavailable", "reason": "neural extra not installed", "features": 0}

        if not self._model_available:
            return {"status": "not_trained", "features": 0}

        return {
            "status": "active",
            "features": len(self._weights) if self._weights is not None else 0,
            "bias": float(self._bias),
            "weight_mean": float(np.mean(np.abs(self._weights))) if self._weights is not None else 0,
        }
