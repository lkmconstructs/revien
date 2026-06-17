"""
Revien Training Loop — Accumulates retrieval usage signals and trains the neural scorer.
Logging starts from day one so training data accumulates.
Training triggers when buffer reaches threshold.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("revien.neural")


class TrainingLoop:
    """
    Accumulates usage signals from retrieval events and triggers training.
    
    Logs positive signals (node was used after retrieval) and
    negative signals (node was retrieved but ignored).
    
    Training threshold: 100 signals (lowered from 500 for faster cold start).
    Retraining: Every 500 new signals after initial training.
    """

    INITIAL_THRESHOLD = 100
    RETRAIN_INTERVAL = 500

    def __init__(self, db_path: Optional[str] = None, model_dir: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".revien" / "training.db")
        self.model_dir = model_dir or str(Path.home() / ".revien" / "models")
        self._last_train_count = 0
        self._ensure_db()
        self._load_train_state()

    def _ensure_db(self) -> None:
        """Create the training signal table."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_label TEXT,
                node_type TEXT,
                score REAL,
                was_used INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_query ON retrieval_signals(query)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_node ON retrieval_signals(node_id)
        """)
        conn.commit()
        conn.close()

    def _load_train_state(self) -> None:
        """Load last training count from DB."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT value FROM training_state WHERE key = 'last_train_count'"
        ).fetchone()
        if row:
            self._last_train_count = int(row[0])
        conn.close()

    def _save_train_state(self) -> None:
        """Persist training state."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO training_state (key, value) VALUES (?, ?)",
            ("last_train_count", str(self._last_train_count)),
        )
        conn.commit()
        conn.close()

    def log_retrieval(
        self,
        query: str,
        results: List[Dict],
    ) -> None:
        """
        Log a retrieval event. Each result node gets a signal entry.
        Initially logged as was_used=0; call mark_used() when the user
        actually uses a retrieved node.
        """
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        for result in results:
            conn.execute(
                """INSERT INTO retrieval_signals
                   (query, node_id, node_label, node_type, score, was_used, timestamp)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (
                    query,
                    result.get("node_id", ""),
                    result.get("label", result.get("node_label", "")),
                    result.get("node_type", ""),
                    result.get("score", 0.0),
                    now,
                ),
            )
        conn.commit()
        conn.close()
        
        # Check if we should train
        self._maybe_train()

    def mark_used(self, node_id: str, query: Optional[str] = None) -> None:
        """Mark a node as actually used after retrieval (positive signal)."""
        conn = sqlite3.connect(self.db_path)
        if query:
            conn.execute(
                "UPDATE retrieval_signals SET was_used = 1 WHERE node_id = ? AND query = ?",
                (node_id, query),
            )
        else:
            conn.execute(
                "UPDATE retrieval_signals SET was_used = 1 WHERE node_id = ?",
                (node_id,),
            )
        conn.commit()
        conn.close()

    def get_signal_count(self) -> int:
        """Get total number of logged signals."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM retrieval_signals").fetchone()[0]
        conn.close()
        return count

    def is_ready_for_training(self) -> bool:
        """Check if enough new signals have accumulated for training."""
        count = self.get_signal_count()
        
        # Initial training
        if self._last_train_count == 0 and count >= self.INITIAL_THRESHOLD:
            return True
        
        # Retraining
        if count - self._last_train_count >= self.RETRAIN_INTERVAL:
            return True
        
        return False

    def get_training_data(self) -> List[Dict]:
        """Export all signals as training data."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT query, node_id, node_label, node_type, score, was_used, timestamp 
               FROM retrieval_signals"""
        ).fetchall()
        conn.close()
        return [
            {
                "query": r[0],
                "node_id": r[1],
                "node_label": r[2] or r[1],  # Fall back to node_id if no label
                "node_type": r[3],
                "score": r[4],
                "was_used": bool(r[5]),
                "timestamp": r[6],
            }
            for r in rows
        ]

    def _maybe_train(self) -> bool:
        """Check threshold and trigger training if ready."""
        if not self.is_ready_for_training():
            return False
        return self.train()

    def train(self) -> bool:
        """
        Run training loop on accumulated signals.
        """
        count = self.get_signal_count()
        
        if count < self.INITIAL_THRESHOLD:
            logger.info(
                f"Not enough signals for training: "
                f"{count}/{self.INITIAL_THRESHOLD}"
            )
            return False

        try:
            from revien.neural.scorer_model import NeuralScorer
            
            training_data = self.get_training_data()
            
            # Instantiate scorer and train
            scorer = NeuralScorer(model_dir=self.model_dir)
            success = scorer.train(training_data)
            
            if success:
                self._last_train_count = count
                self._save_train_state()
                logger.info(f"Training completed on {count} signals")
            
            return success
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def get_stats(self) -> Dict:
        """Return training loop statistics."""
        count = self.get_signal_count()
        conn = sqlite3.connect(self.db_path)
        used_count = conn.execute(
            "SELECT COUNT(*) FROM retrieval_signals WHERE was_used = 1"
        ).fetchone()[0]
        conn.close()
        
        return {
            "total_signals": count,
            "positive_signals": used_count,
            "negative_signals": count - used_count,
            "last_train_count": self._last_train_count,
            "signals_until_next_train": max(
                0, 
                self.INITIAL_THRESHOLD - count if self._last_train_count == 0 
                else self.RETRAIN_INTERVAL - (count - self._last_train_count)
            ),
            "ready_for_training": self.is_ready_for_training(),
        }
