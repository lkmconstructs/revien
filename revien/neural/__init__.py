"""
Revien neural package — opt-in learned reranking.

NeuralScorer and TrainingLoop import cleanly without the heavy `neural` extra
(numpy/scikit-learn). When that extra is absent the scorer is inert and base
scoring is used; TrainingLoop is pure stdlib (sqlite3) and always works.
"""

from .scorer_model import NeuralScorer
from .training import TrainingLoop

__all__ = ["NeuralScorer", "TrainingLoop"]
