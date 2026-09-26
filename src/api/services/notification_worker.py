from __future__ import annotations

import asyncio
import logging

from src.api.services.user_service import UserService

logger = logging.getLogger(__name__)


class ReminderWorker:
    """Раз в NOTIFICATIONS__CHECK_INTERVAL_SECONDS создаёт напоминания о поисках без избранного."""

    def __init__(self, users: UserService, interval_seconds: float) -> None:
        self.users = users
        self.interval_seconds = interval_seconds

    async def run(self) -> None:
        logger.info("Reminder worker started: interval=%ss", self.interval_seconds)
        while True:
            try:
                await self.users.process_due_reminders()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reminder worker iteration failed")
            await asyncio.sleep(self.interval_seconds)
