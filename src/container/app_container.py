import asyncio
import logging
import signal
from asyncio import Task
from typing import Any

from src.container.manager import ConnectionManager
from src.settings import AllSettings

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(self, settings: AllSettings):
        self.settings = settings
        self._connection_manager: ConnectionManager | None = None
        self._tasks: list[Task] = []

        self._api: Any = None

        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """
        Handle shutdown signal.

        Idempotent: repeated signals (e.g. double Ctrl+C) are ignored,
        so the shutdown sequence runs exactly once.

        Args:
            sig: Signal that was received (SIGTERM or SIGINT)
        """
        if self._shutdown_event.is_set():
            logger.debug("Ignoring repeated shutdown signal %s", sig.name)
            return
        logger.info("Received shutdown signal %s", sig.name)
        self._shutdown_event.set()

    def setup_signal_handlers(self) -> None:
        """
        Setup graceful shutdown signal handlers for SIGTERM and SIGINT.

        Should be called once during application startup.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),
            )
        logger.debug("Signal handlers SIGTERM and SIGINT registered")

    async def run_app(self):
        self.setup_signal_handlers()
        try:
            logger.info("Starting Connection Manager")
            self._connection_manager = ConnectionManager(self.settings)
            await self._connection_manager.start()
        except Exception:
            logger.error("Failed to start Connection Manager")
            raise
        try:
            logger.info("Starting FastApi")
            # self._api = ...
            # task = asyncio.create_task(self._api.start())
            # self._tasks.append(task)
        except Exception:
            logger.error("Failed to start Bot")
            raise

        await self._shutdown_event.wait()
        await self.stop_app()

    async def stop_app(self):
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._connection_manager is not None:
            await self._connection_manager.stop()
        logger.info("Application stopped gracefully")
