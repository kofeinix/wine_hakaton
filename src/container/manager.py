from src.connections.database.postgres import DatabaseClient
from src.connections.database.qdrant import QdrantClient
from src.connections.database.redis import RedisClient
from src.connections.rate_limiter import RateLimiterManager
from src.settings.settings import AllSettings


class ConnectionManager:
    def __init__(self, settings):
        self.settings: AllSettings = settings
        self.database: DatabaseClient | None = None
        self.qdrant: QdrantClient | None = None
        self.redis: RedisClient | None = None
        self.rate_limiter: RateLimiterManager | None = None
        self._initialize()

    def _initialize(self):
        self.database = DatabaseClient(self.settings.database)
        self.qdrant = QdrantClient(self.settings.qdrant)
        self.redis = RedisClient(self.settings.redis)
        self.rate_limiter = RateLimiterManager(self.settings.rate_limiter,
                                               self.redis)

    async def start(self):
        await self.database.connect()
        await self.qdrant.connect()
        await self.redis.start()
        await self.rate_limiter.start()

    async def stop(self):
        await self.database.close()
        await self.qdrant.close()
        await self.redis.stop()
        await self.rate_limiter.stop()