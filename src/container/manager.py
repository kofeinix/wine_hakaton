import asyncio
import logging

from src.connections.database.postgres import DatabaseClient
from src.connections.minio import MinioClient
from src.connections.qdrant import QdrantClient
from src.connections.redis import RedisClient
from src.connections.rate_limiter import RateLimiterManager
from src.llm.langchain_openai import ChatOpenAIWrapper
from src.ml.siglip2 import SiglipImageEmbedder
from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import AllSettings

logger = logging.getLogger(__name__)

LLM_START_TIMEOUT_SECONDS = 15


class ConnectionManager:
    def __init__(self, settings):
        self.settings: AllSettings = settings
        self.database: DatabaseClient | None = None
        self.qdrant: QdrantClient | None = None
        self.redis: RedisClient | None = None
        self.rate_limiter: RateLimiterManager | None = None
        self.llm: ChatOpenAIWrapper | None = None
        self.minio: MinioClient | None = None
        self.bottle_yolo: YoloBottleCropper | None = None
        self.label_yolo: YoloLabelCropper | None = None
        self.embeddings: SiglipImageEmbedder | None = None
        self._initialize()

    def _initialize(self):
        self.database = DatabaseClient(self.settings.database)
        self.qdrant = QdrantClient(self.settings.qdrant)
        self.redis = RedisClient(self.settings.redis)
        self.rate_limiter = RateLimiterManager(self.settings.rate_limiter,
                                               self.redis)
        self.llm = ChatOpenAIWrapper(self.settings.llm)
        self.minio = MinioClient(self.settings.minio)
        bottle_yolo_settings = self.settings.yolo.model_copy(
            update={"model_path": self.settings.yolo.bottle_model_path}
        )
        label_yolo_settings = self.settings.yolo.model_copy(
            update={"model_path": self.settings.yolo.label_model_path}
        )
        self.bottle_yolo = YoloBottleCropper(bottle_yolo_settings)
        self.label_yolo = YoloLabelCropper(label_yolo_settings)
        self.embeddings = SiglipImageEmbedder(self.settings.embeddings)

    async def start(self):
        await self.database.connect()
        await self.qdrant.connect()
        await self.redis.start()
        await self.rate_limiter.start()
        await self._start_llm()
        await self.minio.start()
        await self.bottle_yolo.start()
        await self.label_yolo.start()
        await self.embeddings.start()

    async def _start_llm(self) -> None:
        """LLM нужна только для OCR этикеток: без неё поиск идёт по изображению, сервис стартует."""
        if not self.settings.llm.configured:
            logger.warning("LLM__BASE_URL / LLM__MODEL_NAME not set: label OCR is disabled, searching by image only")
            self.llm = None
            return
        try:
            # проверка подключения с ретраями; недоступный адрес не должен надолго задерживать старт
            await asyncio.wait_for(self.llm.start(), timeout=LLM_START_TIMEOUT_SECONDS)
        except Exception:
            logger.warning(
                "LLM %s at %s is unavailable: label OCR is disabled, searching by image only. "
                "Start the LLM and restart the app to enable OCR.",
                self.settings.llm.model_name,
                self.settings.llm.base_url,
                exc_info=True,
            )
            await self.llm.stop()
            self.llm = None

    async def stop(self):
        await self.database.close()
        await self.qdrant.close()
        await self.redis.stop()
        await self.rate_limiter.stop()
        if self.llm is not None:
            await self.llm.stop()
        await self.minio.stop()
        await self.bottle_yolo.stop()
        await self.label_yolo.stop()
        await self.embeddings.stop()
