"""
Revien Daemon — Main daemon process that runs the API server and scheduler.
The entrypoint for `revien start`.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("revien.daemon")


class RevienDaemon:
    """
    Main daemon process. Manages:
    - FastAPI server (REST API)
    - Sync scheduler (auto-sync with adapters)
    - Graceful shutdown
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7437,
        db_path: Optional[str] = None,
        sync_interval_hours: float = 6.0,
        adapters: Optional[dict] = None,
    ):
        self.host = host
        self.port = port
        self.db_path = db_path or self._default_db_path()
        self.sync_interval_hours = sync_interval_hours
        # config.json's "adapters" dict ({name: {type, ...}}). These get
        # registered LIVE on the sync scheduler at start() — without this,
        # `revien connect` wrote config nothing ever read and the "begin
        # syncing" promise was false.
        self.adapters = adapters or {}
        self._app = None
        self._scheduler = None

    def _default_db_path(self) -> str:
        """Default database location: ~/.revien/revien.db"""
        revien_dir = Path.home() / ".revien"
        revien_dir.mkdir(parents=True, exist_ok=True)
        return str(revien_dir / "revien.db")

    def start(self) -> None:
        """Start the daemon: API server + scheduler."""
        import uvicorn
        from .server import create_app
        from .scheduler import SyncScheduler

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        logger.info(f"Starting Revien daemon on {self.host}:{self.port}")
        logger.info(f"Database: {self.db_path}")

        # Create app — the store opens here, so everything from this point
        # to uvicorn's exit runs under the try/finally below: any failure
        # between open and serve still releases the handle.
        self._app = create_app(db_path=self.db_path)

        try:
            # Create and attach scheduler. NOT started here: the lifespan
            # startup (create_app) starts it inside uvicorn's running event
            # loop — AsyncIOScheduler.start() outside one raises
            # RuntimeError on apscheduler 3.11 + py3.13, and on older
            # combinations bound a loop that never ran, so sync/drain/dream
            # jobs never fired.
            self._scheduler = SyncScheduler(
                pipeline=self._app.state.pipeline,
                interval_hours=self.sync_interval_hours,
            )
            self._app.state.scheduler = self._scheduler

            # Register connected adapters (config.json "adapters") on the
            # scheduler so `connect` -> `start` actually syncs. Malformed or
            # unknown entries are logged and skipped, never fatal.
            from revien.adapters import build_adapter_from_config

            for name, entry in self.adapters.items():
                adapter = build_adapter_from_config(entry)
                if adapter is None:
                    logger.warning(
                        f"Skipping adapter {name!r}: unknown or malformed entry "
                        f"(type={entry.get('type') if isinstance(entry, dict) else entry!r})"
                    )
                    continue
                self._scheduler.register_adapter(name, adapter)
            if self._scheduler.list_adapters():
                logger.info(
                    f"Live adapters: {', '.join(self._scheduler.list_adapters())} "
                    f"(sync every {self.sync_interval_hours}h)"
                )

            # Shutdown rides uvicorn's own signal handling: uvicorn catches
            # SIGINT/SIGTERM and runs the ASGI lifespan shutdown, which stops
            # the scheduler and closes the store (see create_app's lifespan).
            # The old signal.signal handlers here called sys.exit(0) directly
            # — that PREEMPTED uvicorn's handler, so the lifespan teardown
            # never ran and the sqlite handle died with the process instead
            # of being closed.

            # Dream-mode cadence (B3.1) — STRICTLY OPT-IN: unset env means the
            # consolidation pass never runs unattended. Manual-first was the
            # design gate; this is the second step, and every scheduled run
            # logs its full report (never silent). Reindex and orphan
            # invalidation are NOT available on the schedule — those stay
            # manual decisions. The job is queued now and registered when the
            # lifespan startup starts the scheduler.
            dream_hours = os.environ.get("REVIEN_DREAM_INTERVAL_HOURS", "").strip()
            if dream_hours:
                try:
                    interval = float(dream_hours)
                except ValueError:
                    logger.warning(
                        "REVIEN_DREAM_INTERVAL_HOURS=%r is not a number; "
                        "dream mode NOT scheduled", dream_hours)
                    interval = 0.0
                if interval > 0:
                    from revien.consolidate import Consolidator

                    app_state = self._app.state

                    def _dream() -> None:
                        consolidator = Consolidator(
                            app_state.store, app_state.ops,
                            semantic=app_state.semantic,
                            clustering=app_state.clustering,
                        )
                        report = consolidator.run()  # decay + recluster only
                        logger.info("dream pass: %s", report.to_dict())

                    self._scheduler.add_interval_job(
                        "dream_consolidation", _dream, hours=interval,
                    )

            # Start uvicorn (blocking)
            uvicorn.run(
                self._app,
                host=self.host,
                port=self.port,
                log_level="info",
            )
        finally:
            # Normally a no-op — the lifespan teardown already stopped the
            # scheduler and closed the store. Covers the paths where the app
            # never ran (wiring crash, port in use).
            self.close()

    def close(self) -> None:
        """Stop background work and release the database. Idempotent.

        The lifespan teardown (create_app) does the same cleanup on a normal
        shutdown, so calling this after — or instead of — one is a no-op:
        scheduler.stop() checks its running flag, GraphStore.close() checks
        its connection. Safe to call on a daemon that never fully started.
        """
        if self._scheduler is not None:
            self._scheduler.stop()
        if self._app is not None:
            store = getattr(self._app.state, "store", None)
            if store is not None:
                store.close()

    @property
    def app(self):
        return self._app

    @property
    def scheduler(self):
        return self._scheduler
