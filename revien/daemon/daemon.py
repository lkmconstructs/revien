"""
Revien Daemon — Main daemon process that runs the API server and scheduler.
The entrypoint for `revien start`.
"""

import logging
import os
import signal
import sys
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
    ):
        self.host = host
        self.port = port
        self.db_path = db_path or self._default_db_path()
        self.sync_interval_hours = sync_interval_hours
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

        # Create app
        self._app = create_app(db_path=self.db_path)

        # Create and attach scheduler
        self._scheduler = SyncScheduler(
            pipeline=self._app.state.pipeline,
            interval_hours=self.sync_interval_hours,
        )
        self._app.state.scheduler = self._scheduler

        # Register shutdown handler
        def _shutdown(signum, frame):
            logger.info("Shutting down...")
            if self._scheduler:
                self._scheduler.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Start scheduler (non-blocking)
        self._scheduler.start()

        # Start uvicorn (blocking)
        uvicorn.run(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

    @property
    def app(self):
        return self._app

    @property
    def scheduler(self):
        return self._scheduler
