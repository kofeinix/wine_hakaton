from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from statistics import mean, pstdev
from typing import Any

from src.api.repositories.wine_repository import WineRepository
from src.api.schemas import (
    CompactSearchMatch,
    CompactSearchResponse,
    SearchMatchResponse,
    SearchResponse,
    WinePhotoResponse,
    WineResponse,
)
from src.api.services.photo_service import WinePhotoService
from src.connections.database.models import Wine


VIEW_COLLECTIONS = {
    "original": "wine_original_siglip2",
    "bottle_crop": "wine_bottle_crop_siglip2",
    "label_crop": "wine_label_crop_siglip2",
}

VIEW_WEIGHTS = {
    "original": 0.25,
    "bottle_crop": 0.25,
    "label_crop": 0.50,
}


@dataclass
class PhotoCandidate:
    photo_id: str
    wine_id: str | None
    slug: str | None
    score: float
    view_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WineCandidate:
    wine_id: str
    slug: str
    score: float
    max_score: float
    mean_score: float
    n_photos: int
    score_std: float


class WineService:
    def __init__(self, connection_manager) -> None:
        self.connection_manager = connection_manager
        self.repository = WineRepository(connection_manager.database)
        self.photo_service = WinePhotoService(
            label_cropper=getattr(connection_manager, "label_yolo", None)
            or getattr(connection_manager, "yolo", None),
            bottle_cropper=getattr(connection_manager, "bottle_yolo", None),
        )

    async def search_by_image(
        self,
        image_bytes: bytes,
        limit: int = 10,
    ) -> CompactSearchResponse:
        candidates, _diagnostics, _crops = await self._search_candidates(
            image_bytes=image_bytes,
            limit=limit,
        )
        return CompactSearchResponse(
            result=[
                CompactSearchMatch(
                    wine_id=candidate.wine_id,
                    slug=candidate.slug,
                    score=candidate.score,
                    max_score=candidate.max_score,
                    mean_score=candidate.mean_score,
                    n_photos=candidate.n_photos,
                    score_std=candidate.score_std,
                )
                for candidate in candidates
            ]
        )

    async def search_by_image_extended(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        limit: int = 10,
    ) -> SearchResponse:
        candidates, diagnostics, crops = await self._search_candidates(
            image_bytes=image_bytes,
            limit=limit,
        )
        wines = await self.repository.load_wines_by_ids([candidate.wine_id for candidate in candidates])
        wine_by_id = {str(wine.id): self._wine_to_response(wine) for wine in wines}

        results = [
            SearchMatchResponse(
                wine_id=candidate.wine_id,
                slug=candidate.slug,
                score=candidate.score,
                max_score=candidate.max_score,
                mean_score=candidate.mean_score,
                n_photos=candidate.n_photos,
                score_std=candidate.score_std,
                wine=wine_by_id.get(candidate.wine_id),
            )
            for candidate in candidates
        ]
        best = results[0] if results else None
        second = results[1] if len(results) > 1 else None
        return SearchResponse(
            status="found" if best else "not_found",
            slug=best.slug if best else None,
            confidence=best.score if best else 0.0,
            gap=(best.score - second.score) if best and second else 0.0,
            wine=best.wine if best else None,
            top5=[
                {"slug": result.slug, "score": result.score}
                for result in results[:5]
            ],
            results=results,
            crops=crops,
            diagnostics={
                **diagnostics,
                "filename": filename,
                "content_type": content_type,
            },
        )

    async def get_wine(self, wine_id: str) -> WineResponse | None:
        wine = await self.repository.get_wine(wine_id)
        if wine is None:
            return None
        return self._wine_to_response(wine)

    async def get_photo_file_by_wine_id(self, wine_id: str, filename: str):
        wine = await self.repository.get_wine(wine_id)
        if wine is None:
            return None
        for image in wine.images:
            object_name = image.minio_path
            if PurePosixPath(object_name).name == filename:
                return await self.connection_manager.minio.get_file_with_content_type(object_name)
        return None

    async def _search_candidates(
        self,
        image_bytes: bytes,
        limit: int,
    ) -> tuple[list[WineCandidate], dict[str, Any], dict]:
        query_views = self.photo_service.build_query_views(image_bytes)
        active_views = [view for view in VIEW_WEIGHTS if view in query_views.images]
        weights = self._normalized_weights(active_views)

        vectors = {
            view: self.connection_manager.embeddings.embed(query_views.images[view])
            for view in active_views
        }
        search_tasks = [
            self.connection_manager.qdrant.search_collection(
                collection_name=VIEW_COLLECTIONS[view],
                vector=vectors[view],
                limit=100,
                with_payload=True,
            )
            for view in active_views
        ]
        responses = await asyncio.gather(*search_tasks)

        photo_candidates: dict[str, PhotoCandidate] = {}
        missing_payload_photo_ids: list[str] = []
        per_view_counts: dict[str, int] = {}

        for view, response in zip(active_views, responses, strict=True):
            points = list(getattr(response, "points", []) or [])
            per_view_counts[view] = len(points)
            for point in points:
                payload = getattr(point, "payload", None) or {}
                point_id = str(getattr(point, "id", ""))
                photo_id = str(payload.get("photo_id") or point_id)
                if not photo_id:
                    continue
                wine_id = str(payload["wine_id"]) if payload.get("wine_id") else None
                slug = str(payload["slug"]) if payload.get("slug") else None
                weighted_score = float(getattr(point, "score", 0.0) or 0.0) * weights[view]
                candidate_key = f"{wine_id}:{photo_id}" if wine_id is not None else point_id
                candidate = photo_candidates.get(candidate_key)
                if candidate is None:
                    candidate = PhotoCandidate(
                        photo_id=photo_id,
                        wine_id=wine_id,
                        slug=slug,
                        score=0.0,
                    )
                    photo_candidates[candidate_key] = candidate
                candidate.score += weighted_score
                candidate.view_scores[view] = float(getattr(point, "score", 0.0) or 0.0)
                candidate.wine_id = candidate.wine_id or wine_id
                candidate.slug = candidate.slug or slug
                if candidate.wine_id is None:
                    missing_payload_photo_ids.append(photo_id)

        if missing_payload_photo_ids:
            wine_ids_by_photo = await self.repository.wine_ids_by_photo_ids(missing_payload_photo_ids)
            for photo_id, wine_id in wine_ids_by_photo.items():
                if photo_id in photo_candidates:
                    photo_candidates[photo_id].wine_id = wine_id

        candidates = self._group_by_wine(photo_candidates.values(), limit=limit)
        diagnostics = {
            "global": {
                "active_views": active_views,
                "weights": weights,
                "collections": {view: VIEW_COLLECTIONS[view] for view in active_views},
                "per_view_top_k": 100,
                "per_view_counts": per_view_counts,
            }
        }
        return candidates, diagnostics, query_views.crops

    @staticmethod
    def _normalized_weights(active_views: list[str]) -> dict[str, float]:
        total = sum(VIEW_WEIGHTS[view] for view in active_views)
        if total <= 0:
            return {}
        return {view: VIEW_WEIGHTS[view] / total for view in active_views}

    @staticmethod
    def _group_by_wine(
        photo_candidates,
        limit: int,
    ) -> list[WineCandidate]:
        scores_by_wine: dict[str, list[PhotoCandidate]] = {}
        for candidate in photo_candidates:
            if candidate.wine_id is None or candidate.score <= 0:
                continue
            scores_by_wine.setdefault(candidate.wine_id, []).append(candidate)

        wine_candidates: list[WineCandidate] = []
        for wine_id, photos in scores_by_wine.items():
            scores = [photo.score for photo in photos if photo.score > 0]
            if not scores:
                continue
            max_score = max(scores)
            slug = next((photo.slug for photo in photos if photo.slug), wine_id)
            wine_candidates.append(
                WineCandidate(
                    wine_id=wine_id,
                    slug=slug,
                    score=max_score,
                    max_score=max_score,
                    mean_score=mean(scores),
                    n_photos=len(scores),
                    score_std=pstdev(scores) if len(scores) > 1 else 0.0,
                )
            )

        wine_candidates.sort(key=lambda item: item.max_score, reverse=True)
        return wine_candidates[: min(limit, 10)]

    def _wine_to_response(self, wine: Wine) -> WineResponse:
        photos = [
            WinePhotoResponse(
                id=str(image.id),
                url=f"/api/v1/wines/{wine.id}/photos/{PurePosixPath(image.minio_path).name}",
                object_name=image.minio_path,
                filename=PurePosixPath(image.minio_path).name,
                is_main=image.is_main,
                source_url=image.source_url,
            )
            for image in wine.images
        ]
        image_url = photos[0].url if photos else None
        grapes = [
            link.grape.name
            for link in wine.grape_links
            if link.grape is not None
        ]
        price = float(wine.price) if wine.price is not None else None
        rating = float(wine.rating) if wine.rating is not None else None
        wine_type = " ".join(part for part in [wine.color, wine.sugar] if part) or None
        return WineResponse(
            id=str(wine.id),
            slug=wine.sku,
            sku=wine.sku,
            name=wine.name,
            producer=wine.producer.name if wine.producer else None,
            region=wine.region.name if wine.region else None,
            country=wine.region.country if wine.region else None,
            vintage=wine.year,
            year=wine.year,
            color=wine.color,
            sugar=wine.sugar,
            style=wine_type,
            wine_type=wine_type,
            alcohol=wine.alcohol,
            price=price,
            stock=wine.stock,
            rating=rating,
            description=wine.description,
            url=wine.source_url,
            source_url=wine.source_url,
            image_url=image_url,
            grapes=grapes,
            photos=photos,
        )
