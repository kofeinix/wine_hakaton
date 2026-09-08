import logging
from typing import Any

from limiters import AsyncSemaphore, AsyncTokenBucket
from redis.asyncio import Redis

from src.connections.database.redis import RedisClient
from src.settings.rate_limiter import LimiterConfig, RateLimiterSettings

logger = logging.getLogger(__name__)

class RateLimiterManager:
    """
    Manager that creates one or more rate limiters (semaphore/token_bucket)
    on top of a shared RedisClient.
    """

    def __init__(self, config: RateLimiterSettings, redis_client: RedisClient):
        self._config = config
        self._redis_client = redis_client
        self._limiters: dict[str, Any] = {}
        self._started = False

    async def start(self) -> None:
        """Bind manager to a Redis client and initialize all limiters."""
        if not self._config:
            logger.warning("Tried to create RateLimiter with empty config!")
            raise Exception("Tried to create RateLimiter with empty config!")

        if self._redis_client.started:
            raw_client = self._redis_client.client
        else:
            logger.warning("Failed to create RateLimiter as Redis is not started!")
            raise Exception("Failed to create RateLimiter as Redis is not started!")
        for item in self._config.limiters:
            limiter = self._create_single_limiter(item, raw_client)
            self._limiters[item.name] = limiter
        logger.info("Redis rate limiters initialized")
        self._started = True

    @property
    def started(self):
        return self._started

    @staticmethod
    def _create_single_limiter(
        item: LimiterConfig, raw_client: Redis
    ) -> AsyncSemaphore | AsyncTokenBucket:
        """Create single limiter instance based on mode."""
        base = {
            "name": item.name,
            "capacity": item.capacity,
            "max_sleep": item.max_sleep,
            "connection": raw_client,
            "corporate_prefix": item.prefix,
        }
        if item.mode == "semaphore":
            if not item.semaphore:
                raise ValueError("semaphore config required for mode 'semaphore'")
            return AsyncSemaphore(
                **base,
                expiry=item.semaphore.expiry,
            )

        elif item.mode == "token_bucket":
            if not item.token_bucket:
                raise ValueError("token_bucket config required for mode 'token_bucket'")
            return AsyncTokenBucket(
                **base,
                refill_frequency=item.token_bucket.refill_frequency,
                refill_amount=item.token_bucket.refill_amount,
            )
        else:
            raise ValueError(f"Unsupported limiter mode: {item.mode}")

    def get_limiter(self, name: str) -> AsyncSemaphore | AsyncTokenBucket:
        """Get a limiter instance by name."""
        if not self.started:
            raise RuntimeError("Rate limiters are not started yet")

        if name not in self._limiters:
            raise KeyError(f"Limiter not found: {name}")
        return self._limiters[name]

    async def stop(self) -> None:
        """Close resources held by limiters and reset state."""
        self._limiters.clear()
        self._started = False
        logger.info("Redis rate limiters closed")
