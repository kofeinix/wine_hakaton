import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.settings.settings import DatabaseSettings
from src.connections.database.base import Base

logger = logging.getLogger(__name__)

# create_all не меняет существующие таблицы; новые колонки добавляем идемпотентно
SCHEMA_PATCHES = (
    "ALTER TABLE IF EXISTS wine_images ADD COLUMN IF NOT EXISTS is_generated BOOLEAN NOT NULL DEFAULT false",
    "ALTER TABLE IF EXISTS wines ADD COLUMN IF NOT EXISTS serving_temperature VARCHAR(50)",
    "ALTER TABLE IF EXISTS wines ADD COLUMN IF NOT EXISTS shade VARCHAR(255)",
    "ALTER TABLE IF EXISTS wines ALTER COLUMN alcohol TYPE VARCHAR(50) USING alcohol::text",
    "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS avatar_url TEXT",
    "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS review_notification_period_minutes INTEGER NOT NULL DEFAULT 1440",
    "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS review_reactions_notified_at TIMESTAMPTZ",
)


class DatabaseClient:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.engine = create_async_engine(
            settings.url,
            echo=settings.echo,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_pre_ping=True,
            pool_recycle=1200,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._is_connected = False

    @retry(
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, 30),  # int 30 — WARNING
        reraise=True,
    )
    async def _ping_db_connection(self) -> None:
        async with self.engine.begin() as conn:  # type: ignore[misc]
            await conn.execute(text("SELECT 1"))

    async def connect(self) -> None:
        if self._is_connected:
            return
        await self._ping_db_connection()
        self._is_connected = True
        logger.info("Database client connected")

    async def create_tables(self) -> None:
        import src.connections.database.models  # noqa: F401

        async with self.engine.begin() as conn:  # type: ignore[misc]
            await conn.run_sync(Base.metadata.create_all)
            for patch in SCHEMA_PATCHES:
                await conn.execute(text(patch))

    async def drop_tables(self) -> None:
        import src.connections.database.models  # noqa: F401

        async with self.engine.begin() as conn:  # type: ignore[misc]
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    async def close(self) -> None:
        await self.engine.dispose()
        self._is_connected = False
        logger.info("Database client disconnected")
