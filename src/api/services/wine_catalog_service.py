from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from uuid import uuid4

from src.api.repositories import WineRepository
from src.api.schemas import (
    CompactSearchMatch,
    CompactSearchResponse,
    ExtractedWineLabel,
    SearchMatch,
    SearchResponse,
    WinePhotoResponse,
    WineResponse,
)
from src.api.services.photo_service import WinePhotoService
from src.connections.database.models import Wine
from src.container.manager import ConnectionManager
from src.llm.models import WineOutput
from src.ml.utils import image_to_jpeg_bytes, open_rgb_image
from src.ml.yolo import LabelCrop

logger = logging.getLogger(__name__)


class WineCatalogService:
    def __init__(
        self,
        connection_manager: ConnectionManager,
        wine_repository: WineRepository | None = None,
        photo_service: WinePhotoService | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        if connection_manager.database is None:
            raise RuntimeError("Database connection is not initialized")
        self.wine_repository = wine_repository or WineRepository(connection_manager.database)
        self.photo_service = photo_service or WinePhotoService(connection_manager)

    @property
    def vector_searcher(self):
        return self.connection_manager.qdrant

    @property
    def cropper(self):
        return self.connection_manager.yolo

    @property
    def embedder(self):
        return self.connection_manager.embeddings

    @property
    def llm(self):
        return self.connection_manager.llm

    async def search_by_image(
        self,
        image_bytes: bytes,
        limit: int,
    ) -> CompactSearchResponse:
        matches = await self._search_image_matches(image_bytes=image_bytes, limit=limit)
        return CompactSearchResponse(
            result=[
                CompactSearchMatch(wine_id=match.wine.id, score=match.score)
                for match in matches.results
            ]
        )

    async def search_by_image_extended(
        self,
        image_bytes: bytes,
        filename: str | None,
        content_type: str,
        limit: int,
    ) -> SearchResponse:
        return await self._search_image_matches(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
            limit=limit,
        )

    async def get_wine(self, wine_id: str) -> WineResponse | None:
        wine = await self.wine_repository.get_wine(wine_id)
        if wine is None:
            return None
        photos = await self.photo_service.get_wine_photos(wine)
        return self._wine_response(wine, photos=photos.photos)

    async def _search_image_matches(
        self,
        image_bytes: bytes,
        limit: int,
        filename: str | None = None,
        content_type: str = "image/jpeg",
    ) -> SearchResponse:
        query_id = uuid4()
        image = open_rgb_image(image_bytes)

        crop = await self._crop_label(image)
        full_jpeg = image_to_jpeg_bytes(image)
        crop_jpeg = image_to_jpeg_bytes(crop.image) if crop is not None else None

        full_extract_task = asyncio.create_task(self._extract_label(full_jpeg))
        crop_extract_task = (
            asyncio.create_task(self._extract_label(crop_jpeg))
            if crop_jpeg is not None
            else None
        )
        vector_task = asyncio.create_task(asyncio.to_thread(self.embedder.embed, image))

        full_extract = await full_extract_task
        crop_extract = await crop_extract_task if crop_extract_task is not None else None
        vector = await vector_task

        qdrant_matches = await self._search_qdrant(vector=vector, limit=limit)
        merged_labels = self.merge_labels(crop_extract, full_extract)
        text_matches = await self._search_database_by_extraction([merged_labels], limit=limit)
        wines_by_id = await self._load_wines_by_ids(
            [*qdrant_matches.keys(), *text_matches.keys()]
        )

        results = self._merge_matches(
            qdrant_scores=qdrant_matches,
            text_scores=text_matches,
            wines_by_id=wines_by_id,
            limit=limit,
        )

        return SearchResponse(
            query_id=query_id,
            filename=filename,
            content_type=content_type,
            image_size_bytes=len(image_bytes),
            recognized_label=self._recognized_label(crop_extract, full_extract),
            label_crop=(
                {"box": crop.box, "confidence": crop.confidence}
                if crop is not None
                else None
            ),
            extracted_full_image=full_extract,
            extracted_label_crop=crop_extract,
            results=results,
        )

    @staticmethod
    def merge_labels(
        primary: ExtractedWineLabel | None,
        fallback: ExtractedWineLabel | None,
    ) -> ExtractedWineLabel:
        if primary is None and fallback is None:
            return ExtractedWineLabel()
        if fallback is None:
            return primary
        if primary is None:
            return fallback
        data = fallback.model_dump()
        data.update(primary.model_dump(exclude_none=True))
        return ExtractedWineLabel(**data)

    async def _crop_label(self, image) -> LabelCrop | None:
        try:
            return self.cropper.crop(image)
        except Exception:
            logger.exception("YOLO label crop failed")
            return None

    async def _extract_label(self, image_bytes: bytes) -> ExtractedWineLabel | None:
        if self.llm is None:
            return None
        try:
            raw = await self.llm.analyze_image(image_bytes)
            return self._extracted_label(raw)
        except Exception:
            logger.exception("NuExtract image analysis failed")
            return None

    async def _search_qdrant(self, vector: list[float], limit: int) -> dict[str, float]:
        if self.connection_manager.qdrant is None:
            return {}

        response = await self.vector_searcher.search(
            vector=vector,
            limit=max(50, limit * 10),
        )
        if not response:
            return {}

        photo_scores: dict[str, float] = {}
        for point in response.points:
            photo_id = str(point.id)
            photo_scores[photo_id] = max(
                photo_scores.get(photo_id, 0.0),
                self._clamp_score(float(point.score)),
            )
        return await self.wine_repository.wine_scores_by_photo_ids(photo_scores)

    async def _search_database_by_extraction(
        self,
        extracted_labels: list[ExtractedWineLabel],
        limit: int,
    ) -> dict[str, float]:
        terms = self._search_terms(extracted_labels)
        if not terms:
            return {}

        wines = await self.wine_repository.search_by_terms(terms=terms, limit=limit)
        return {
            str(wine.id): score
            for wine, score in sorted(
                ((wine, self._text_score(wine, extracted_labels, terms)) for wine in wines),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            if score > 0
        }

    async def _load_wines_by_ids(self, wine_ids: list[str]) -> dict[str, WineResponse]:
        wines = await self.wine_repository.load_wines_by_ids(wine_ids)
        return {str(wine.id): self._wine_response(wine) for wine in wines}

    def _merge_matches(
        self,
        qdrant_scores: dict[str, float],
        text_scores: dict[str, float],
        wines_by_id: dict[str, WineResponse],
        limit: int,
    ) -> list[SearchMatch]:
        merged: list[SearchMatch] = []
        for wine_id in set(qdrant_scores) | set(text_scores):
            wine = wines_by_id.get(wine_id)
            if wine is None:
                continue
            vector_score = qdrant_scores.get(wine_id)
            text_score = text_scores.get(wine_id)
            if vector_score is not None and text_score is not None:
                score = vector_score * 0.65 + text_score * 0.35
                sources = ["vector", "text"]
                reason = "Matched by image vector and extracted label fields"
            elif vector_score is not None:
                score = vector_score
                sources = ["vector"]
                reason = "Matched by image vector in Qdrant"
            else:
                score = text_score or 0.0
                sources = ["text"]
                reason = "Matched by extracted label fields in Postgres"
            merged.append(
                SearchMatch(
                    wine=wine,
                    score=self._clamp_score(score),
                    vector_score=vector_score,
                    text_score=text_score,
                    sources=sources,
                    reason=reason,
                )
            )
        return sorted(merged, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _wine_response(
        wine: Wine,
        photos: list[WinePhotoResponse] | None = None,
    ) -> WineResponse:
        rating = float(wine.rating) if wine.rating is not None else None
        wine_id = str(wine.id)
        photos = photos or []
        image_url = photos[0].url if photos else None
        region = wine.region.name if wine.region is not None else None
        country = wine.region.country if wine.region is not None else None
        grapes = [link.grape.name for link in wine.grape_links if link.grape is not None]
        style = " ".join(item for item in (wine.color, wine.sugar) if item) or None
        return WineResponse(
            id=wine_id,
            name=wine.name,
            producer=wine.producer.name if wine.producer is not None else None,
            country=country,
            region=region,
            vintage=wine.year,
            grapes=grapes,
            style=style,
            average_rating=rating or 0,
            rating=rating,
            price=float(wine.price) if wine.price is not None else None,
            currency="RUB" if wine.price is not None else None,
            image_url=image_url,
            photos=photos,
            description=wine.description,
            color=wine.color,
            wine_type=style,
            url=wine.source_url,
            alcohol=f"{wine.alcohol:g}%" if wine.alcohol is not None else None,
            food_pairings=[],
        )

    @staticmethod
    def _extracted_label(output: WineOutput) -> ExtractedWineLabel:
        return ExtractedWineLabel(
            name=output.name,
            producer=output.producer,
            wine_type=output.wine_type,
            region=output.region,
            alcohol=output.alcohol,
            vintage=output.vintage,
            label_description=output.label_description,
            volume=output.volume,
        )

    @staticmethod
    def _recognized_label(
        crop_extract: ExtractedWineLabel | None,
        full_extract: ExtractedWineLabel | None,
    ) -> str | None:
        label = crop_extract or full_extract
        if label is None:
            return None
        return " ".join(
            value
            for value in (label.producer, label.name, label.wine_type, label.region, label.alcohol)
            if value
        ) or None

    @staticmethod
    def _search_terms(extracted_labels: list[ExtractedWineLabel]) -> list[str]:
        values: list[str] = []
        for label in extracted_labels:
            values.extend(
                [
                    label.name or "",
                    label.producer or "",
                    label.wine_type or "",
                    label.region or "",
                    label.alcohol or "",
                    label.label_description or "",
                ]
            )

        terms = []
        for value in values:
            value = value.strip()
            if len(value) >= 2:
                terms.append(value)
            terms.extend(token for token in re.split(r"\W+", value) if len(token) >= 3)
        return list(dict.fromkeys(terms))[:24]

    @staticmethod
    def _text_score(
        wine: Wine,
        extracted_labels: list[ExtractedWineLabel],
        terms: list[str],
    ) -> float:
        haystack = " ".join(
            item or ""
            for item in (
                wine.name,
                wine.producer.name if wine.producer is not None else None,
                wine.color,
                wine.sugar,
                wine.region.name if wine.region is not None else None,
                wine.region.country if wine.region is not None else None,
                f"{wine.alcohol:g}%" if wine.alcohol is not None else None,
                wine.description,
            )
        ).casefold()
        wine_producer = wine.producer.name if wine.producer is not None else None
        wine_region = wine.region.name if wine.region is not None else None
        wine_type = " ".join(item for item in (wine.color, wine.sugar) if item) or None
        wine_alcohol = f"{wine.alcohol:g}%" if wine.alcohol is not None else None
        field_scores = defaultdict(float)

        for label in extracted_labels:
            checks = {
                "name": (label.name, wine.name, 0.34),
                "producer": (label.producer, wine_producer, 0.28),
                "wine_type": (label.wine_type, wine_type, 0.14),
                "region": (label.region, wine_region, 0.14),
                "alcohol": (label.alcohol, wine_alcohol, 0.10),
            }
            for field, (query_value, wine_value, weight) in checks.items():
                if query_value and wine_value and query_value.casefold() in wine_value.casefold():
                    field_scores[field] = max(field_scores[field], weight)

        token_hits = sum(1 for term in terms if term.casefold() in haystack)
        token_score = min(0.25, token_hits * 0.025)
        return min(1.0, sum(field_scores.values()) + token_score)

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(0.0, min(1.0, score))


def datetime_now_utc():
    from datetime import UTC, datetime

    return datetime.now(UTC)
