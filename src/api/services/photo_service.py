from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import PurePosixPath

from src.api.repositories import WineRepository
from src.api.schemas import WinePhotoListResponse, WinePhotoResponse
from src.connections.database.models import Wine
from src.container.manager import ConnectionManager

logger = logging.getLogger(__name__)

PHOTO_CACHE_TTL_SECONDS = 60 * 60


class WinePhotoService:
    def __init__(
        self,
        connection_manager: ConnectionManager,
        wine_repository: WineRepository | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        if connection_manager.database is None:
            raise RuntimeError("Database connection is not initialized")
        self.wine_repository = wine_repository or WineRepository(connection_manager.database)

    @property
    def redis(self):
        return self.connection_manager.redis

    async def get_wine_photos(self, wine: Wine) -> WinePhotoListResponse:
        wine_id = str(wine.id)
        cache_key = f"wine:{wine_id}:photos:v2"
        cached = await self._get_cached_photos(cache_key)
        if cached is not None:
            return WinePhotoListResponse(wine_id=wine_id, photos=cached)

        prefix = f"{wine_id}/"
        photos: list[WinePhotoResponse] = []
        try:
            objects = await self.connection_manager.minio.list_files(prefix=prefix)
        except Exception:
            logger.exception("Could not list MinIO photos for wine %s", wine_id)
            objects = []
        objects_by_name = {item.object_name: item for item in objects}

        image_paths = [
            (image.minio_path, image.is_main)
            for image in wine.images
            if image.minio_path
        ]
        known_paths = {path for path, _ in image_paths}
        image_paths.extend(
            (item.object_name, self._is_main_photo(PurePosixPath(item.object_name).name))
            for item in objects
            if item.object_name not in known_paths
        )

        for object_name, is_main in image_paths:
            filename = PurePosixPath(object_name).name
            object_info = objects_by_name.get(object_name)
            photos.append(
                WinePhotoResponse(
                    object_name=object_name,
                    filename=filename,
                    url=f"/api/v1/wines/{wine_id}/photos/{filename}",
                    is_main=is_main,
                    size_bytes=object_info.size if object_info is not None else None,
                    content_type=(
                        object_info.content_type if object_info is not None else "image/jpeg"
                    ),
                )
            )

        photos.sort(key=lambda photo: (not photo.is_main, photo.filename))
        await self._set_cached_photos(cache_key, photos)
        return WinePhotoListResponse(wine_id=wine_id, photos=photos)

    async def get_photo_file(self, wine: Wine, filename: str):
        if "/" in filename or filename in {"", ".", ".."}:
            return None

        photos = await self.get_wine_photos(wine)
        object_names_by_filename = {photo.filename: photo.object_name for photo in photos.photos}
        object_name = object_names_by_filename.get(filename)
        if object_name is None:
            return None

        return await self.connection_manager.minio.get_file_with_content_type(object_name)

    async def get_photo_file_by_wine_id(
        self,
        wine_id: str,
        filename: str,
    ) -> tuple[BytesIO, str] | None:
        wine = await self.wine_repository.get_wine(wine_id)
        if wine is None:
            return None
        return await self.get_photo_file(wine, filename)

    async def _get_cached_photos(self, cache_key: str) -> list[WinePhotoResponse] | None:
        if self.connection_manager.redis is None or not self.connection_manager.redis.started:
            return None

        try:
            raw = await self.redis.client.get(cache_key)
        except Exception:
            logger.exception("Could not read Redis photo cache key %s", cache_key)
            return None

        if raw is None:
            return None

        try:
            payload = json.loads(raw.decode("utf-8"))
            return [WinePhotoResponse.model_validate(item) for item in payload]
        except Exception:
            logger.exception("Invalid Redis photo cache payload for key %s", cache_key)
            return None

    async def _set_cached_photos(
        self,
        cache_key: str,
        photos: list[WinePhotoResponse],
    ) -> None:
        if self.connection_manager.redis is None or not self.connection_manager.redis.started:
            return

        try:
            payload = json.dumps(
                [photo.model_dump() for photo in photos],
                ensure_ascii=False,
            )
            await self.redis.client.setex(
                cache_key,
                PHOTO_CACHE_TTL_SECONDS,
                payload.encode("utf-8"),
            )
        except Exception:
            logger.exception("Could not write Redis photo cache key %s", cache_key)

    @staticmethod
    def _is_main_photo(filename: str) -> bool:
        return filename == "main.jpg" or filename.startswith(("main__", "main_"))
