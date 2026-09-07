import logging

from redis.asyncio import Redis

from src.settings.settings import RedisSettings

logger = logging.getLogger(__name__)


class RedisClient:
    """Thin wrapper providing lifecycle around Redis."""

    def __init__(self, config: RedisSettings):
        self._config = config
        self._client: Redis | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Initialize and verify connection."""
        if self._client is not None:
            return

        common_kwargs = {
            "db": self._config.db,
            "username": self._config.username,
            "password": self._config.password,
            "ssl": self._config.ssl,
            "socket_timeout": self._config.socket_timeout,
            "socket_connect_timeout": self._config.socket_connect_timeout,
            "retry_on_timeout": self._config.retry_on_timeout,
            "max_connections": self._config.max_connections,
            "decode_responses": False,  # Manual JSON handling
        }

        self._client = Redis(
            host=self._config.host, port=self._config.port, **common_kwargs
        )
        logger.info("Direct Redis client initialized")

        try:
            await self._client.ping()
            self._started = True
            logger.info("Redis client ready")
        except Exception:
            logger.exception("Redis connection failed")
            if self._client:
                await self._client.close()
            self._client = None
            raise

    async def stop(self) -> None:
        """Close connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._started = False
            logger.info("Redis client closed")

    @property
    def client(self) -> Redis:
        """Raw Redis client for full Redis API access."""
        if self._client is None:
            raise RuntimeError("Call start() first")
        return self._client
