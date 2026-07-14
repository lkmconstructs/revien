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

    # Deferred-embed idle sweep (capture leg): how often the daemon checks the
    # pending-embed queue. The check is one COUNT on an empty/absent table when
    # idle — the sweep only pays anything when captures are actually waiting.
    EMBED_DRAIN_INTERVAL_SECONDS = 30

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
        # Jobs registered before start() — the daemon wires its extra jobs
        # (dream consolidation) before the lifespan startup starts the
        # scheduler inside uvicorn's running loop. Flushed by start().
        self._pending_jobs: List[tuple] = []

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
            # timestamp was previously dropped here — every synced item
            # ingested with recorded_at=None, so temporal resolution and
            # content-time recency had nothing to anchor to.
            ts = None
            raw_ts = item.get("timestamp")
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts)
                except (ValueError, TypeError):
                    ts = None
            input_data = IngestionInput(
                source_id=item.get("source_id", f"adapter:{name}"),
                content=item.get("content", ""),
                content_type=item.get("content_type", "conversation"),
                timestamp=ts,
                metadata=item.get("metadata", {}),
                links=item.get("links", []) or [],
                curated=bool(item.get("curated", False)),
            )
            if input_data.content.strip():
                self.pipeline.ingest(input_data)
                ingested += 1

        self._last_sync[name] = datetime.now(timezone.utc)
        logger.info(f"Synced {name}: {ingested} items ingested")
        return {"status": "ok", "adapter": name, "items_ingested": ingested}

    async def drain_pending_embeds(self) -> int:
        """Idle sweep for the deferred-embed queue (capture leg): embed
        anything a defer_embed ingest left behind. Runs on the event loop —
        the store's single shared connection stays single-threaded. Returns
        count embedded (0 on empty queue or disabled layer)."""
        semantic = getattr(self.pipeline, "semantic", None)
        if semantic is None or not semantic.is_enabled:
            return 0
        drained = semantic.drain_pending()
        if drained:
            logger.info(f"Embedded {drained} deferred capture(s) (idle sweep)")
        return drained

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
            self._scheduler.add_job(
                self.drain_pending_embeds,
                "interval",
                seconds=self.EMBED_DRAIN_INTERVAL_SECONDS,
                id="revien_embed_drain",
            )
            self._scheduler.start()
            self._running = True
            logger.info(f"Scheduler started (interval: {self.interval_hours}h)")
            # Flush jobs queued before start (the daemon registers the dream
            # job before the lifespan starts us inside uvicorn's loop).
            for job_id, func, hours in self._pending_jobs:
                self._scheduler.add_job(func, "interval", hours=hours, id=job_id)
                logger.info("Registered job %r (interval: %sh)", job_id, hours)
            self._pending_jobs = []
        except ImportError:
            logger.warning("APScheduler not installed. Auto-sync disabled.")
            logger.warning("Install with: pip install apscheduler")

    def add_interval_job(self, job_id: str, func, hours: float) -> bool:
        """Register an extra periodic job (e.g. the dream-mode consolidation
        pass). Called before start(), the job is queued and registered when
        the scheduler starts — the daemon wires its jobs before the lifespan
        startup starts the scheduler inside uvicorn's loop. Returns True in
        both cases; when APScheduler is absent start() logs the miss and
        queued jobs never fire, same as auto-sync itself."""
        if self._scheduler is None or not self._running:
            self._pending_jobs.append((job_id, func, hours))
            logger.info(
                "Queued job %r until scheduler start (interval: %sh)",
                job_id, hours,
            )
            return True
        self._scheduler.add_job(func, "interval", hours=hours, id=job_id)
        logger.info("Registered job %r (interval: %sh)", job_id, hours)
        return True

    def stop(self) -> None:
        """Stop the background scheduler."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running
