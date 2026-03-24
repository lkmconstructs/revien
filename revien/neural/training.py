"""
Revien Training Loop — Accumulates retrieval usage signals for future model training.
Logging starts from day one so training data accumulates.
Training activates post-MVP when buffer reaches 500 signals.
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
    Accumulates usage signals from retrieval events.
    Logs positive signals (node was used after retrieval) and
    negative signals (node was retrieved but ignored).

    Training buffer threshold: 500 signals.
    When reached, post-MVP update will trigger fine-tuning.
    """

    BUFFER_THRESHOLD = 500

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".revien" / "training.db")
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the training signal table."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_type TEXT,
                score REAL,
                was_used INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
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
                   (query, node_id, node_type, score, was_used, timestamp)
                   VALUES (?, ?, ?, ?, 0, ?)""",
                (
                    query,
                    result.get("node_id", ""),
                    result.get("node_type", ""),
                    result.get("score", 0.0),
                    now,
                ),
            )
        conn.commit()
        conn.close()

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
        """Check if enough signals have accumulated for training."""
        return self.get_signal_count() >= self.BUFFER_THRESHOLD

    def get_training_data(self) -> List[Dict]:
        """Export all signals as training data."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT query, node_id, node_type, score, was_used, timestamp FROM retrieval_signals"
        ).fetchall()
        conn.close()
        return [
            {
                "query": r[0],
                "node_id": r[1],
                "node_type": r[2],
                "score": r[3],
                "was_used": bool(r[4]),
                "timestamp": r[5],
            }
            for r in rows
        ]

    def train(self) -> bool:
        """
        Run training loop. SCAFFOLDED — returns False in MVP.
        POST-MVP: Will fine-tune the scoring model on accumulated signals.
        """
        if not self.is_ready_for_training():
            logger.info(
                f"Not enough signals for training: "
                f"{self.get_signal_count()}/{self.BUFFER_THRESHOLD}"
            )
            return False

        # POST-MVP: Actual training logic here
        logger.info("Training scaffolded but not implemented in MVP")
        return False
