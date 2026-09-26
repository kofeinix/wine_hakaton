import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.settings.settings import QdrantSettings

logger = logging.getLogger(__name__)


class QdrantClient:
    def __init__(self, settings: QdrantSettings) -> None:
        self.settings = settings
        self.client = AsyncQdrantClient(
            url=settings.url,
            grpc_port=settings.grpc_port,
            prefer_grpc=settings.prefer_grpc,
            https=settings.https,
            api_key=settings.api_key or None,
            prefix=settings.prefix or None,
            timeout=settings.timeout,
        )
        self._is_connected = False
        logger.info('Qdrant client initialized')

    @retry(
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, 30),  # int 30 — WARNING
        reraise=True,
    )
    async def _ping_qdrant_connection(self) -> None:
        await self.client.get_collections()

    async def connect(self) -> None:
        if self._is_connected:
            return
        await self._ping_qdrant_connection()
        self._is_connected = True
        logger.info("Qdrant client connected")

    async def close(self) -> None:
        await self.client.close()
        self._is_connected = False
        logger.info("Qdrant client disconnected")

    async def search(self, vector: list[float], limit: int) -> QueryResponse | None:
        return await self.search_collection(
            collection_name=self.settings.collection_name,
            vector=vector,
            limit=limit,
            with_payload=False,
        )

    async def search_collection(
        self,
        collection_name: str,
        vector: list[float],
        limit: int,
        with_payload: bool = True,
        query_filter: Any | None = None,
    ) -> QueryResponse | None:
        if not self._is_connected:
            raise RuntimeError("Qdrant client not connected")
        try:
            return await self._query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )
        except Exception:
            logger.exception("Vector search failure in collection %s", collection_name)
            return None

    # Повтор на обрыв keep-alive соединения (httpx.ReadError): без него view молча
    # получал 0 точек и поиск возвращал пустой результат.
    @retry(
        wait=wait_exponential_jitter(initial=0.2, max=1),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, 30),  # int 30 — WARNING
        reraise=True,
    )
    async def _query_points(self, **kwargs: Any) -> QueryResponse:
        return await self.client.query_points(**kwargs)

    async def retrieve_vectors(
        self,
        collection_name: str,
        point_ids: list[str],
    ) -> list:
        if not self._is_connected:
            raise RuntimeError("Qdrant client not connected")
        if not point_ids:
            return []
        try:
            return await self.client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            logger.exception("Vector retrieve failure in collection %s", collection_name)
            return []
