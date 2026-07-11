"""
Revien Generic API Adapter — Connects to any REST endpoint returning conversation data.
Configurable URL, headers, and response parsing.
Covers Ollama, LM Studio, and custom setups.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .base import RevienAdapter


class GenericAPIAdapter(RevienAdapter):
    """
    Generic REST API adapter. Connects to any system that exposes an endpoint
    returning conversation data.

    Default response format expected:
    {
        "conversations": [
            {
                "content": "conversation text",
                "timestamp": "ISO-8601",
                "metadata": { ... }
            }
        ]
    }

    Custom parsers can be provided for non-standard response formats.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        response_parser: Optional[Callable] = None,
        source_id_prefix: str = "api",
        timeout: float = 30.0,
    ):
        """
        Args:
            url: REST endpoint URL to fetch conversations from
            headers: HTTP headers (e.g., auth tokens)
            params: Default query parameters
            response_parser: Custom function to parse response JSON into content list.
                             Should accept (dict) and return List[Dict] with
                             content, content_type, timestamp, metadata keys.
            source_id_prefix: Prefix for source_id on ingested content
            timeout: Request timeout in seconds
        """
        self.url = url
        self.headers = headers or {}
        self.params = params or {}
        self.response_parser = response_parser or self._default_parser
        self.source_id_prefix = source_id_prefix
        self.timeout = timeout

    async def fetch_new_content(self, since: datetime) -> List[Dict]:
        """Fetch content from the configured REST endpoint."""
        try:
            import httpx

            params = {**self.params, "since": since.isoformat()}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.url,
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

            results = self.response_parser(data)

            # Ensure each result has a source_id
            for r in results:
                if "source_id" not in r:
                    r["source_id"] = f"{self.source_id_prefix}:{self.url}"

            return results

        except ImportError:
            # httpx not installed
            return []
        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.url,
                    headers=self.headers,
                )
                return response.status_code < 500
        except Exception:
            return False

    def _default_parser(self, data: Any) -> List[Dict]:
        """
        Default response parser.
        Handles common response formats:
        1. {"conversations": [...]}
        2. {"data": [...]}
        3. {"messages": [...]}
        4. Direct list [...]
        """
        items = []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = (
                data.get("conversations")
                or data.get("data")
                or data.get("messages")
                or data.get("results")
                or []
            )

        results = []
        for item in items:
            if isinstance(item, str):
                results.append({
                    "content": item,
                    "content_type": "conversation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {"adapter": "generic_api"},
                })
            elif isinstance(item, dict):
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("message")
                    or ""
                )
                if content:
                    results.append({
                        "content": content,
                        "content_type": item.get("content_type", "conversation"),
                        "timestamp": item.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        "metadata": item.get("metadata", {"adapter": "generic_api"}),
                    })

        return results
