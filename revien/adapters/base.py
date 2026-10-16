"""
Revien Adapter Base Class — Interface that all adapters must implement.
Two methods. That's the whole contract.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional


def parse_message_timestamp(value) -> Optional[datetime]:
    """A per-message timestamp from a session log -> aware datetime, or None.
    Accepts ISO-8601 (trailing Z ok) and epoch numbers (seconds, or
    milliseconds when implausibly large). Anything else is None - an adapter
    must fall back to mtime (and say so) rather than guess."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            secs = value / 1000.0 if value > 1e11 else float(value)
            return datetime.fromtimestamp(secs, tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            v = value.strip()
            if v[-1] in ("Z", "z"):
                v = v[:-1] + "+00:00"
            dt = datetime.fromisoformat(v)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return None


class RevienAdapter(ABC):
    """
    Base class for all Revien adapters.
    Each connected AI system needs an adapter that implements this interface.
    """

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
