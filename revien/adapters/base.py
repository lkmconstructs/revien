"""
Revien Adapter Base Class — Interface that all adapters must implement.
Two methods. That's the whole contract.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List


class RevienAdapter(ABC):
    """
    Base class for all Revien adapters.
    Each connected AI system needs an adapter that implements this interface.
    """

    # True only when re-reading the adapter's FULL history (sync cursor at
    # EPOCH) is provably idempotent: every item it emits carries an
    # ``ingest_key`` so the pipeline refreshes the existing node in place.
    # `revien sync reset` refuses adapters that do not declare this.
    IDEMPOTENT_REREAD = False

    @abstractmethod
    async def fetch_new_content(
        self, since: datetime
    ) -> List[Dict]:
        """
        Fetch content created since the given timestamp.

        Returns list of dicts, each containing:
            - content: str (the raw text)
            - content_type: str (conversation | document | note | code)
            - timestamp: str (ISO-8601)
            - metadata: dict (optional, adapter-specific)
            - source_id: str (optional, identifies the source)
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the connected system is reachable."""
        raise NotImplementedError
