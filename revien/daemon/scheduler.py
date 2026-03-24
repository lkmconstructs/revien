"""
Revien Scheduler — Background auto-sync scheduler for connected adapters.
Uses APScheduler to periodically pull new content from connected AI systems.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from revien.adapters.base import RevienAdapter
    from revien.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("revien.scheduler")


class SyncScheduler:
    """
    Manages periodic sync with connected AI system adapters.
    Each adapter is polled at a configurable interval (default: 6 hours).
    """

    def __init__(
        self,
        pipeline: "IngestionPipeline",
        interval_hours: float = 6.0,
    ):
        self.pipeline = pipeline
        self.interval_hours = interval_hours
        self._adapters: Dict[str, "RevienAdapter"] = {}
        self._last_sync: Dict[str, datetime] = {}
        self._scheduler = None
        self._running = False

    def register_adapter(self, name: str, adapter: "RevienAdapter") -> None:
        """Register an adapter for periodic sync."""
        self._adapters[name] = adapter
        self._last_sync[name] = datetime.now(timezone.utc)
        logger.info(f"Registered adapter: {name}")

    def unregister_adapter(self, name: str) -> None:
        """Remove an adapter from sync rotation."""
        self._adapters.pop(name, None)
        self._last_sync.pop(name, None)
        logger.info(f"Unregistered adapter: {name}")

    def list_adapters(self) -> List[str]:
        """List registered adapter names."""
        return list(self._adapters.keys())

    async def sync_all(self) -> Dict[str, dict]:
        """
        Run sync for all registered adapters.
        Returns a dict of adapter_name -> sync result.
        """
        results = {}
        for name, adapter in self._adapters.items():
            try:
                result = await self._sync_adapter(name, adapter)
                results[name] = result
            except Exception as e:
                logger.error(f"Sync failed for {name}: {e}")
                results[name] = {"status": "error", "error": str(e)}
        return results

    async def sync_one(self, name: str) -> dict:
        """Sync a specific adapter by name."""
        adapter = self._adapters.get(name)
        if adapter is None:
            return {"status": "error", "error": f"Unknown adapter: {name}"}
        return await self._sync_adapter(name, adapter)

    async def _sync_adapter(self, name: str, adapter: "RevienAdapter") -> dict:
        """Sync a single adapter: fetch new content and ingest it."""
        from revien.ingestion.pipeline import IngestionInput

        since = self._last_sync.get(name, datetime.now(timezone.utc))

        # Health check first
        healthy = await adapter.health_check()
        if not healthy:
            return {"status": "unhealthy", "adapter": name}

        # Fetch new content
        new_content = await adapter.fetch_new_content(since)
        if not new_content:
            self._last_sync[name] = datetime.now(timezone.utc)
            return {"status": "ok", "adapter": name, "items_ingested": 0}

        # Ingest each piece of content
        ingested = 0
        for item in new_content:
            input_data = IngestionInput(
                source_id=item.get("source_id", f"adapter:{name}"),
                content=item.get("content", ""),
                content_type=item.get("content_type", "conversation"),
                metadata=item.get("metadata", {}),
            )
            if input_data.content.strip():
                self.pipeline.ingest(input_data)
                ingested += 1

        self._last_sync[name] = datetime.now(timezone.utc)
        logger.info(f"Synced {name}: {ingested} items ingested")
        return {"status": "ok", "adapter": name, "items_ingested": ingested}

    def start(self) -> None:
        """Start the background scheduler."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            self._scheduler = AsyncIOScheduler()
            self._scheduler.add_job(
                self.sync_all,
                "interval",
                hours=self.interval_hours,
                id="revien_auto_sync",
            )
            self._scheduler.start()
            self._running = True
            logger.info(f"Scheduler started (interval: {self.interval_hours}h)")
        except ImportError:
            logger.warning("APScheduler not installed. Auto-sync disabled.")
            logger.warning("Install with: pip install apscheduler")

    def stop(self) -> None:
        """Stop the background scheduler."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running
