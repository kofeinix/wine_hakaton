from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from pathlib import PurePosixPath
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from qdrant_client import models as qdrant_models
from rapidfuzz import fuzz

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
from src.connections.database.models import Grape, Wine
from src.ml.catboost_reranker import CatBoostWineReranker, build_candidate_feature_rows
from src.ml.patch_scoring import patch_similarity_score
from src.ml.text_normalization import normalize_match_text

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


PATCH_VIEW_COLLECTIONS = {
    "original": "wine_original_dinov3_patches",
    "label_crop": "wine_label_crop_dinov3_patches",
}

VIEW_WEIGHTS = {
    "original": 0.30,
    "bottle_crop": 0.15,
    "label_crop": 0.55,
}

PATCH_VIEW_WEIGHTS = {
    "original": 0.6,
    "label_crop": 0.4,
}

GLOBAL_CANDIDATE_LIMIT = 50
PER_VIEW_TOP_K = 150
PATCH_RERANK_TOP2_GAP = 0.005
PATCH_SCORE_STRATEGY = "sym20"
PATCH_TOKEN_CACHE_MAX_ITEMS = 512
PATCH_POINT_NAMESPACE = uuid.UUID("92670dbe-559b-55c3-ae90-ac7c7b9e50bd")
LABEL_PHOTO_AREA_THRESHOLD = 0.60
LABEL_PHOTO_CONFIDENCE_THRESHOLD = 0.50
VIEW_SCORE_TOP_K_PHOTOS = 3
VIEW_SCORE_BEST_WEIGHT = 0.75
VIEW_SCORE_MEAN_WEIGHT = 0.25
OCR_VISUAL_WEIGHT = 0.75
OCR_TEXT_WEIGHT = 0.25
OCR_RERANK_LIMIT = 50
OCR_FUZZY_WRATIO_WEIGHT = 0.25
OCR_FUZZY_TOKEN_SET_WEIGHT = 0.75
OCR_DOMAIN_WEIGHT = 0.65
OCR_FUZZY_WEIGHT = 0.35
OCR_GRAPE_WEIGHT = 0.60
OCR_PRODUCER_WEIGHT = 0.25
OCR_NAME_WEIGHT = 0.15
OCR_GRAPE_EXTRA_MATCH_THRESHOLD = 0.55
OCR_GRAPE_EXTRA_MATCH_BONUS = 0.08
DEFAULT_SEARCH_STAGES = ("global", "patches")
VALID_SEARCH_STAGES = {"global", "patches", "llm"}
CATBOOST_RERANKER_MODEL_PATH = PROJECT_ROOT / "models" / "catboost" / "wine_reranker.cbm"

@dataclass(frozen=True)
class ViewMatch:
    score: float
    point_id: str
    photo_id: str


@dataclass(frozen=True)
class WineCandidate:
    wine_id: str
    slug: str
    score: float
    max_score: float
    mean_score: float
    n_photos: int
    score_std: float
    cosine_score: float = 0.0
    view_photo_ids: dict[str, str] | None = None
    patch_score: float | None = None


@dataclass(frozen=True)
class PatchTokenCacheEntry:
    mtime_ns: int
    size: int
    tokens: np.ndarray
    grid: tuple[int, int] | None


class WineService:
    def __init__(self, connection_manager) -> None:
        self.connection_manager = connection_manager
        self.repository = WineRepository(connection_manager.database)
        self.global_encoder = connection_manager.settings.search.global_encoder
        self.collection_encoder = connection_manager.settings.search.collection_encoder
        self.photo_service = WinePhotoService(
            label_cropper=getattr(connection_manager, "label_yolo", None)
            or getattr(connection_manager, "yolo", None),
            bottle_cropper=getattr(connection_manager, "bottle_yolo", None),
        )
        self._patch_token_cache: OrderedDict[str, PatchTokenCacheEntry] = OrderedDict()
        self._catboost_reranker: CatBoostWineReranker | None = None
        self._catboost_load_failed = False

    async def search_by_image(
        self,
        image_bytes: bytes,
        limit: int = 10,
        views: list[str] | None = None,
        stages: int | str | list[str] | None = None,
        main_photos_only: bool = False,
    ) -> CompactSearchResponse:
        candidates, _diagnostics, _crops = await self._search_candidates(
            image_bytes=image_bytes,
            limit=limit,
            views=views,
            stages=stages,
            main_photos_only=main_photos_only,
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
                    cosine_score=candidate.cosine_score,
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
        stages: int | str | list[str] | None = None,
        main_photos_only: bool = False,
    ) -> SearchResponse:
        candidates, diagnostics, crops = await self._search_candidates(
            image_bytes=image_bytes,
            limit=limit,
            stages=stages,
            main_photos_only=main_photos_only,
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
                cosine_score=candidate.cosine_score,
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

    async def search_by_image_catboost(
        self,
        image_bytes: bytes,
        limit: int = 10,
        views: list[str] | None = None,
        main_photos_only: bool = False,
    ) -> CompactSearchResponse:
        candidates, _diagnostics, _crops = await self._search_candidates_catboost(
            image_bytes=image_bytes,
            limit=limit,
            views=views,
            main_photos_only=main_photos_only,
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
                    cosine_score=candidate.cosine_score,
                )
                for candidate in candidates
            ]
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
        views: list[str] | None = None,
        stages: int | str | list[str] | None = None,
        main_photos_only: bool = False,
    ) -> tuple[list[WineCandidate], dict[str, Any], dict]:
        search_stages = self._normalize_search_stages(stages)
        query_views = self.photo_service.build_query_views(image_bytes)
        allowed = set(views) if views else set(VIEW_WEIGHTS)
        label_area_ratio = self._label_crop_area_ratio(query_views.crops)
        label_confidence = self._label_crop_confidence(query_views.crops)
        label_photo_mode = (
            label_area_ratio > LABEL_PHOTO_AREA_THRESHOLD
            and label_confidence > LABEL_PHOTO_CONFIDENCE_THRESHOLD
        )
        # Активны только views, чей кроп реально построился (available=True).
        active_views = [
            view
            for view in VIEW_WEIGHTS
            if view in query_views.images
            and view in allowed
            and query_views.crops.get(view) is not None
            and query_views.crops[view].available
        ]
        if label_photo_mode and "label_crop" in active_views:
            active_views = [view for view in active_views if view == "label_crop"]
            logger.info(
                "Detected label-photo input: label_area_ratio=%.3f confidence=%.3f; using label_crop only.",
                label_area_ratio,
                label_confidence,
            )
        ocr = await self._extract_ocr_text(query_views)

        embeddings = self._embed_global_images([query_views.images[view] for view in active_views])
        vectors = dict(zip(active_views, embeddings, strict=True))
        query_filter = self._main_photo_filter() if main_photos_only else None
        search_tasks = [
            self.connection_manager.qdrant.search_collection(
                collection_name=self._view_collection(view),
                vector=vectors[view],
                limit=PER_VIEW_TOP_K,
                with_payload=True,
                query_filter=query_filter,
            )
            for view in active_views
        ]
        responses = await asyncio.gather(*search_tasks)

        # Собираем cosine score по каждому view: wine_id -> {view: [score по фото]}.
        wine_view_matches: dict[str, dict[str, list[ViewMatch]]] = {}
        wine_slugs: dict[str, str] = {}
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
                if wine_id is None:
                    # Фото без wine_id в payload не можем привязать к вину — пропускаем.
                    continue
                cosine = float(getattr(point, "score", 0.0) or 0.0)
                wine_view_matches.setdefault(wine_id, {}).setdefault(view, []).append(
                    ViewMatch(score=cosine, point_id=point_id, photo_id=photo_id)
                )
                if slug:
                    wine_slugs.setdefault(wine_id, slug)

        # Нормализуем веса только по views, которые реально вернули результаты.
        views_with_results = [view for view in active_views if per_view_counts.get(view, 0) > 0]
        weights = self._normalized_weights(views_with_results)

        global_limit = max(limit, GLOBAL_CANDIDATE_LIMIT)
        candidates = self._group_by_wine(wine_view_matches, wine_slugs, weights, limit=global_limit)
        rerank_applied = False
        rerank_reason = "confident_global_top1"
        llm_applied = False
        llm_reason = "not_requested"
        llm_selection: dict[str, Any] | None = None
        patch_reason = self._patch_rerank_reason(candidates) if "patches" in search_stages else None
        logger.info(
            "Global search found top_%s candidates. %s",
            len(candidates),
            self._format_top_gap(candidates),
        )
        if "patches" in search_stages and patch_reason is not None:
            logger.info("Global candidates are close: %s. Starting DINOv3 patch rerank.", patch_reason)
            reranked = await self._rerank_with_dinov3_patches(
                candidates=candidates[:GLOBAL_CANDIDATE_LIMIT],
                query_views=query_views,
                active_views=active_views,
            )
            if reranked:
                candidates = reranked
                rerank_applied = True
                rerank_reason = "global_candidates_close"
                logger.info("Patch rerank completed. %s", self._format_top_gap(candidates))
            else:
                rerank_reason = "patch_rerank_unavailable"
                logger.info("Patch rerank unavailable; keeping global ranking. %s", self._format_top_gap(candidates))
        elif "patches" not in search_stages:
            rerank_reason = "stage_global_only"
            logger.info("Search stages=%s; skipping patch rerank.", search_stages)
        else:
            logger.info("Global top1 is confident; skipping patch rerank.")

        candidates, ocr_diagnostics = await self._rerank_with_ocr(
            candidates=candidates[:OCR_RERANK_LIMIT],
            ocr_text=ocr["normalized_text"],
        )

        if "llm" in search_stages:
            candidates, llm_selection = await self._rerank_with_llm_choice(
                image_bytes=image_bytes,
                candidates=candidates[:GLOBAL_CANDIDATE_LIMIT],
            )
            llm_applied = llm_selection is not None
            llm_reason = "selected_candidate" if llm_applied else "llm_unavailable"

        candidates = candidates[:limit]
        diagnostics = {
            "global": {
                "active_views": active_views,
                "weights": weights,
                "encoder": self.global_encoder,
                "collection_encoder": self.collection_encoder,
                "collections": {view: self._view_collection(view) for view in active_views},
                "per_view_top_k": PER_VIEW_TOP_K,
                "per_view_counts": per_view_counts,
                "label_area_ratio": label_area_ratio,
                "label_confidence": label_confidence,
                "label_photo_mode": label_photo_mode,
                "label_photo_area_threshold": LABEL_PHOTO_AREA_THRESHOLD,
                "label_photo_confidence_threshold": LABEL_PHOTO_CONFIDENCE_THRESHOLD,
                "fusion": "weighted_cosine",
                "aggregation": "view_best75_top3mean25_then_weighted_max_sum",
                "candidate_limit": global_limit,
                "stages": search_stages,
                "main_photos_only": main_photos_only,
                "patch_rerank_top2_gap": PATCH_RERANK_TOP2_GAP,
            },
            "patch_rerank": {
                "applied": rerank_applied,
                "reason": rerank_reason,
                "mode": "on_the_fly",
                "fusion": f"dinov3_{PATCH_SCORE_STRATEGY}",
                "weights": PATCH_VIEW_WEIGHTS,
            },
            "llm_rerank": {
                "applied": llm_applied,
                "reason": llm_reason,
                "selection": llm_selection,
            },
            "ocr_rerank": {
                **ocr,
                **ocr_diagnostics,
                "visual_weight": OCR_VISUAL_WEIGHT,
                "text_weight": OCR_TEXT_WEIGHT,
            },
        }
        return candidates, diagnostics, query_views.crops

    async def _search_candidates_catboost(
        self,
        image_bytes: bytes,
        limit: int,
        views: list[str] | None = None,
        main_photos_only: bool = False,
    ) -> tuple[list[WineCandidate], dict[str, Any], dict]:
        query_views = self.photo_service.build_query_views(image_bytes)
        allowed = set(views) if views else set(VIEW_WEIGHTS)
        label_area_ratio = self._label_crop_area_ratio(query_views.crops)
        label_confidence = self._label_crop_confidence(query_views.crops)
        label_photo_mode = (
            label_area_ratio > LABEL_PHOTO_AREA_THRESHOLD
            and label_confidence > LABEL_PHOTO_CONFIDENCE_THRESHOLD
        )
        active_views = [
            view
            for view in VIEW_WEIGHTS
            if view in query_views.images
            and view in allowed
            and query_views.crops.get(view) is not None
            and query_views.crops[view].available
        ]
        if label_photo_mode and "label_crop" in active_views:
            active_views = [view for view in active_views if view == "label_crop"]

        embeddings = self._embed_global_images([query_views.images[view] for view in active_views])
        vectors = dict(zip(active_views, embeddings, strict=True))
        query_filter = self._main_photo_filter() if main_photos_only else None
        responses = await asyncio.gather(
            *[
                self.connection_manager.qdrant.search_collection(
                    collection_name=self._view_collection(view),
                    vector=vectors[view],
                    limit=PER_VIEW_TOP_K,
                    with_payload=True,
                    query_filter=query_filter,
                )
                for view in active_views
            ]
        )

        wine_view_matches: dict[str, dict[str, list[ViewMatch]]] = {}
        wine_slugs: dict[str, str] = {}
        per_view_counts: dict[str, int] = {}
        for view, response in zip(active_views, responses, strict=True):
            points = list(getattr(response, "points", []) or []) if response is not None else []
            per_view_counts[view] = len(points)
            for point in points:
                payload = getattr(point, "payload", None) or {}
                point_id = str(getattr(point, "id", ""))
                photo_id = str(payload.get("photo_id") or point_id)
                wine_id = str(payload["wine_id"]) if payload.get("wine_id") else None
                slug = str(payload["slug"]) if payload.get("slug") else None
                if wine_id is None:
                    continue
                cosine = float(getattr(point, "score", 0.0) or 0.0)
                wine_view_matches.setdefault(wine_id, {}).setdefault(view, []).append(
                    ViewMatch(score=cosine, point_id=point_id, photo_id=photo_id)
                )
                if slug:
                    wine_slugs.setdefault(wine_id, slug)

        views_with_results = [view for view in active_views if per_view_counts.get(view, 0) > 0]
        weights = self._normalized_weights(views_with_results)
        feature_rows = build_candidate_feature_rows(
            wine_view_matches=wine_view_matches,
            wine_slugs=wine_slugs,
            weights=weights,
            crops=query_views.crops,
        )
        reranker = self._load_catboost_reranker()
        if reranker is not None and feature_rows:
            model_scores = reranker.predict_rows(feature_rows)
            ranked_rows = sorted(zip(feature_rows, model_scores, strict=True), key=lambda item: item[1], reverse=True)
            fusion = "catboost"
        else:
            ranked_rows = sorted(
                ((row, row.baseline_score) for row in feature_rows),
                key=lambda item: item[1],
                reverse=True,
            )
            fusion = "weighted_cosine_fallback"

        candidates = [
            WineCandidate(
                wine_id=row.wine_id,
                slug=row.slug,
                score=float(score),
                max_score=row.max_score,
                mean_score=row.mean_score,
                n_photos=row.n_photos,
                score_std=row.score_std,
                cosine_score=row.cosine_score,
                view_photo_ids=row.view_photo_ids,
            )
            for row, score in ranked_rows[:limit]
        ]
        diagnostics = {
            "global": {
                "active_views": active_views,
                "weights": weights,
                "encoder": self.global_encoder,
                "collection_encoder": self.collection_encoder,
                "collections": {view: self._view_collection(view) for view in active_views},
                "per_view_top_k": PER_VIEW_TOP_K,
                "per_view_counts": per_view_counts,
                "label_area_ratio": label_area_ratio,
                "label_confidence": label_confidence,
                "label_photo_mode": label_photo_mode,
                "fusion": fusion,
                "main_photos_only": main_photos_only,
            },
            "catboost": {
                "model_path": str(CATBOOST_RERANKER_MODEL_PATH),
                "loaded": reranker is not None,
            },
        }
        return candidates, diagnostics, query_views.crops

    def _embed_global_images(self, images: list[Image.Image]) -> list[list[float]]:
        if self.global_encoder == "dinov3":
            return self.connection_manager.dinov3.embed_many(images)
        return self.connection_manager.embeddings.embed_many(images)

    def _load_catboost_reranker(self) -> CatBoostWineReranker | None:
        if self._catboost_reranker is not None:
            return self._catboost_reranker
        if self._catboost_load_failed:
            return None
        if not CATBOOST_RERANKER_MODEL_PATH.is_file():
            self._catboost_load_failed = True
            logger.warning("CatBoost reranker model not found: %s", CATBOOST_RERANKER_MODEL_PATH)
            return None
        try:
            self._catboost_reranker = CatBoostWineReranker.load(CATBOOST_RERANKER_MODEL_PATH)
        except Exception:
            self._catboost_load_failed = True
            logger.exception("Failed to load CatBoost reranker model: %s", CATBOOST_RERANKER_MODEL_PATH)
            return None
        return self._catboost_reranker

    def _view_collection(self, view: str) -> str:
        return f"wine_{view}_{self.collection_encoder}"

    @staticmethod
    def _label_crop_area_ratio(crops: dict[str, Any]) -> float:
        label = crops.get("label_crop")
        original = crops.get("original")
        if label is None or original is None or not getattr(label, "available", False):
            return 0.0
        if getattr(label, "source_view", "original") not in {"original", "original_fallback"}:
            return 0.0
        label_box = getattr(label, "box", None)
        original_width = getattr(original, "width", None)
        original_height = getattr(original, "height", None)
        if label_box is None or not original_width or not original_height:
            return 0.0
        x1, y1, x2, y2 = label_box
        label_area = max(0, x2 - x1) * max(0, y2 - y1)
        original_area = max(1, int(original_width) * int(original_height))
        return label_area / original_area

    @staticmethod
    def _label_crop_confidence(crops: dict[str, Any]) -> float:
        label = crops.get("label_crop")
        if label is None or not getattr(label, "available", False):
            return 0.0
        if getattr(label, "source_view", "original") not in {"original", "original_fallback"}:
            return 0.0
        try:
            return float(getattr(label, "confidence", None) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _extract_ocr_text(self, query_views) -> dict[str, Any]:
        llm = getattr(self.connection_manager, "llm", None)
        if llm is None:
            return {
                "applied": False,
                "reason": "llm_unavailable",
                "source_view": None,
                "text": "",
                "normalized_text": "",
            }

        source_view = self._ocr_source_view(query_views)
        image = query_views.images.get(source_view)
        if image is None:
            return {
                "applied": False,
                "reason": "no_source_image",
                "source_view": source_view,
                "text": "",
                "normalized_text": "",
            }

        try:
            text = await llm.ocr_image_text(self._image_to_jpeg_bytes(image))
        except Exception:
            logger.exception("OCR extraction failed for source view %s", source_view)
            return {
                "applied": False,
                "reason": "ocr_failed",
                "source_view": source_view,
                "text": "",
                "normalized_text": "",
            }

        normalized = normalize_match_text(text)
        logger.info(
            "OCR source=%s applied=%s raw_text=%r normalized_text=%r",
            source_view,
            bool(normalized),
            text,
            normalized,
        )
        return {
            "applied": bool(normalized),
            "reason": "ok" if normalized else "empty_ocr_text",
            "source_view": source_view,
            "text": text,
            "normalized_text": normalized,
        }

    @staticmethod
    def _ocr_source_view(query_views) -> str:
        for view in ("label_crop", "bottle_crop"):
            crop = query_views.crops.get(view)
            if crop is not None and getattr(crop, "available", False) and view in query_views.images:
                return view
        return "original"

    @staticmethod
    def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=92, optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _main_photo_filter() -> qdrant_models.Filter:
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="photo_id",
                    match=qdrant_models.MatchValue(value="main"),
                )
            ]
        )

    @staticmethod
    def _normalize_search_stages(stages: int | str | list[str] | None) -> list[str]:
        if stages is None:
            return list(DEFAULT_SEARCH_STAGES)
        if isinstance(stages, int):
            return ["global"] if stages <= 1 else ["global", "patches"]

        raw_items: list[str] = []
        if isinstance(stages, str):
            raw_items = [item.strip() for item in stages.split(",")]
        else:
            for item in stages:
                raw_items.extend(part.strip() for part in str(item).split(","))

        normalized: list[str] = []
        for item in raw_items:
            key = item.lower()
            if not key:
                continue
            if key == "1":
                key = "global"
            elif key == "2":
                for legacy_stage in ("global", "patches"):
                    if legacy_stage not in normalized:
                        normalized.append(legacy_stage)
                continue
            elif key in {"patch", "pathes", "patch_tokens"}:
                key = "patches"
            if key not in VALID_SEARCH_STAGES:
                logger.warning("Ignoring unknown search stage: %s", item)
                continue
            if key not in normalized:
                normalized.append(key)

        if "global" not in normalized:
            normalized.insert(0, "global")

        return [stage for stage in ("global", "patches", "llm") if stage in normalized]

    @staticmethod
    def _normalized_weights(active_views: list[str]) -> dict[str, float]:
        total = sum(VIEW_WEIGHTS[view] for view in active_views)
        if total <= 0:
            return {}
        return {view: VIEW_WEIGHTS[view] / total for view in active_views}

    @staticmethod
    def _group_by_wine(
        wine_view_matches: dict[str, dict[str, list[ViewMatch]]],
        wine_slugs: dict[str, str],
        weights: dict[str, float],
        limit: int,
    ) -> list[WineCandidate]:
        wine_candidates: list[WineCandidate] = []
        for wine_id, views in wine_view_matches.items():
            # Для каждого view берём лучший cosine score по фото вина, затем
            # применяем view weight.
            view_scores: list[float] = []
            raw_cosine_scores: list[float] = []
            view_photo_ids: dict[str, str] = {}
            for view, matches in views.items():
                sorted_matches = sorted(matches, key=lambda item: item.score, reverse=True)
                best_match = sorted_matches[0]
                best_cosine = best_match.score
                top_scores = [item.score for item in sorted_matches[:VIEW_SCORE_TOP_K_PHOTOS]]
                mean_top_score = mean(top_scores)
                view_score = (
                    VIEW_SCORE_BEST_WEIGHT * best_cosine
                    + VIEW_SCORE_MEAN_WEIGHT * mean_top_score
                )
                raw_cosine_scores.append(best_cosine)
                view_scores.append(weights.get(view, 0.0) * view_score)
                view_photo_ids[view] = best_match.photo_id
            if not view_scores:
                continue
            max_score = max(view_scores)
            mean_score = mean(view_scores)
            sum_score = sum(view_scores)
            # Дополнительный view не должен снижать итоговый score:
            # лучший сигнал остаётся главным, остальные дают ограниченный бонус.
            score = 0.7 * max_score + 0.3 * sum_score
            # Исходный косинусный score: максимальный по всем фото/views вина.
            cosine_score = max(raw_cosine_scores) if raw_cosine_scores else 0.0
            wine_candidates.append(
                WineCandidate(
                    wine_id=wine_id,
                    slug=wine_slugs.get(wine_id, wine_id),
                    score=score,
                    max_score=max_score,
                    mean_score=mean_score,
                    n_photos=len(view_scores),
                    score_std=pstdev(view_scores) if len(view_scores) > 1 else 0.0,
                    cosine_score=cosine_score,
                    view_photo_ids=view_photo_ids,
                )
            )

        wine_candidates.sort(key=lambda item: item.score, reverse=True)
        return wine_candidates[:limit]

    @staticmethod
    def _patch_rerank_reason(candidates: list[WineCandidate]) -> str | None:
        if len(candidates) < 2:
            return None
        top1 = candidates[0].score
        top2 = candidates[1].score
        top2_gap = top1 - top2
        if top2_gap < PATCH_RERANK_TOP2_GAP:
            return f"top1_top2_gap={top2_gap:.4f} < {PATCH_RERANK_TOP2_GAP:.4f}"
        return None

    @staticmethod
    def _format_top_gap(candidates: list[WineCandidate]) -> str:
        if not candidates:
            return "no candidates"
        top1 = candidates[0]
        if len(candidates) < 2:
            return f"top1 wine_id={top1.wine_id} confidence={top1.score:.4f}; no top2"
        top2 = candidates[1]
        gap = top1.score - top2.score
        return (
            f"top1 wine_id={top1.wine_id} confidence={top1.score:.4f}; "
            f"top2 wine_id={top2.wine_id} confidence={top2.score:.4f}; "
            f"gap={gap:.4f}"
        )

    async def _rerank_with_dinov3_patches(
        self,
        candidates: list[WineCandidate],
        query_views,
        active_views: list[str],
    ) -> list[WineCandidate]:
        started = perf_counter()
        patch_views = [
            view
            for view in active_views
            if PATCH_VIEW_WEIGHTS.get(view, 0.0) > 0 and view in query_views.images
        ]
        if not candidates or not patch_views or getattr(self.connection_manager, "dinov3", None) is None:
            return []

        query_patches: dict[str, tuple[np.ndarray, tuple[int, int] | None]] = {}
        query_started = perf_counter()
        query_results = self.connection_manager.dinov3.embed_patch_token_arrays_many(
            [query_views.images[view] for view in patch_views]
        )
        query_elapsed = perf_counter() - query_started
        for view, (tokens, grid) in zip(patch_views, query_results, strict=True):
            if tokens is not None and tokens.size > 0:
                query_patches[view] = (tokens, grid)
        if not query_patches:
            return []

        patch_weights = self._normalized_patch_weights(list(query_patches))
        candidate_patches, cache_stats = await self._load_candidate_patch_tokens_many(
            candidates,
            list(query_patches),
        )
        score_started = perf_counter()
        reranked: list[WineCandidate] = []
        for candidate in candidates:
            view_scores: list[float] = []
            raw_scores: list[float] = []
            for view, (tokens, grid) in query_patches.items():
                candidate_tokens, candidate_grid = candidate_patches.get((candidate.wine_id, view), (None, None))
                if candidate_tokens is None:
                    continue
                patch_score = patch_similarity_score(
                    tokens,
                    candidate_tokens,
                    grid,
                    candidate_grid,
                    strategy=PATCH_SCORE_STRATEGY,
                )
                raw_scores.append(patch_score)
                view_scores.append(patch_weights.get(view, 0.0) * patch_score)
            if not view_scores:
                continue
            max_score = max(view_scores)
            mean_score = mean(view_scores)
            sum_score = sum(view_scores)
            score = 0.7 * max_score + 0.3 * sum_score
            reranked.append(
                WineCandidate(
                    wine_id=candidate.wine_id,
                    slug=candidate.slug,
                    score=score,
                    max_score=max_score,
                    mean_score=mean_score,
                    n_photos=len(view_scores),
                    score_std=pstdev(view_scores) if len(view_scores) > 1 else 0.0,
                    cosine_score=candidate.cosine_score,
                    view_photo_ids=candidate.view_photo_ids,
                    patch_score=max(raw_scores) if raw_scores else score,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        logger.info(
            "Patch rerank timings: views=%s candidates=%s query_encode=%.2fs "
            "candidate_load=%.2fs score=%.2fs total=%.2fs cache_hits=%s cache_misses=%s qdrant_hits=%s fallback_images=%s images=%s",
            list(query_patches),
            len(candidates),
            query_elapsed,
            cache_stats["elapsed"],
            perf_counter() - score_started,
            perf_counter() - started,
            cache_stats["hits"],
            cache_stats["misses"],
            cache_stats["qdrant_hits"],
            cache_stats["fallback_images"],
            cache_stats["images"],
        )
        return reranked

    @staticmethod
    def _normalized_patch_weights(active_views: list[str]) -> dict[str, float]:
        total = sum(PATCH_VIEW_WEIGHTS.get(view, 0.0) for view in active_views)
        if total <= 0:
            return {}
        return {view: PATCH_VIEW_WEIGHTS.get(view, 0.0) / total for view in active_views}

    @staticmethod
    def _candidate_view_path(wine_id: str, photo_id: str, view: str) -> Path | None:
        relative = Path(wine_id) / photo_id / f"{view}.jpg"
        for root in (Path("/data/images"), PROJECT_ROOT / "data" / "images"):
            path = root / relative
            if path.is_file():
                return path
        return None

    async def _load_candidate_patch_tokens_many(
        self,
        candidates: list[WineCandidate],
        patch_views: list[str],
    ) -> tuple[dict[tuple[str, str], tuple[np.ndarray, tuple[int, int] | None]], dict[str, float | int]]:
        started = perf_counter()
        items: list[tuple[tuple[str, str], str, str, str, Path | None]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            for view in patch_views:
                key = (candidate.wine_id, view)
                if key in seen:
                    continue
                photo_id = (candidate.view_photo_ids or {}).get(view)
                if photo_id is None:
                    continue
                point_id = self._patch_point_id(candidate.wine_id, photo_id, view)
                image_path = self._candidate_view_path(candidate.wine_id, photo_id, view)
                seen.add(key)
                items.append((key, point_id, photo_id, view, image_path))

        if not items:
            return {}, {
                "elapsed": perf_counter() - started,
                "hits": 0,
                "misses": 0,
                "qdrant_hits": 0,
                "fallback_images": 0,
                "images": 0,
            }

        result: dict[tuple[str, str], tuple[np.ndarray, tuple[int, int] | None]] = {}
        qdrant_pending: dict[str, tuple[tuple[str, str], str, str, Path | None]] = {}
        hits = 0
        for key, point_id, photo_id, view, image_path in items:
            cached = self._patch_token_cache.get(point_id)
            if cached is not None:
                self._patch_token_cache.move_to_end(point_id)
                result[key] = (cached.tokens, cached.grid)
                hits += 1
                continue
            qdrant_pending[point_id] = (key, photo_id, view, image_path)

        qdrant_hits = 0
        fallback_items: list[tuple[tuple[str, str], str, Path, int, int]] = []
        for view in patch_views:
            collection_name = PATCH_VIEW_COLLECTIONS.get(view)
            if collection_name is None:
                continue
            point_ids = [
                point_id
                for point_id, (_key, _photo_id, pending_view, _image_path) in qdrant_pending.items()
                if pending_view == view
            ]
            records = await self.connection_manager.qdrant.retrieve_vectors(collection_name, point_ids)
            retrieved_ids: set[str] = set()
            for record in records:
                point_id = str(getattr(record, "id", ""))
                pending = qdrant_pending.get(point_id)
                if pending is None:
                    continue
                vector = getattr(record, "vector", None)
                if vector is None:
                    continue
                key, _photo_id, _view, _image_path = pending
                payload = getattr(record, "payload", None) or {}
                grid = self._patch_grid_from_payload(payload)
                tokens = np.asarray(vector, dtype=np.float32)
                if tokens.ndim != 2 or tokens.size == 0:
                    continue
                result[key] = (tokens, grid)
                self._store_patch_token_cache(point_id, tokens, grid)
                retrieved_ids.add(point_id)
                qdrant_hits += 1

            for point_id in point_ids:
                if point_id in retrieved_ids:
                    continue
                key, _photo_id, _view, image_path = qdrant_pending[point_id]
                if image_path is None:
                    continue
                try:
                    stat = image_path.stat()
                except OSError:
                    continue
                fallback_items.append((key, point_id, image_path, stat.st_mtime_ns, stat.st_size))

        images: list[Image.Image] = []
        keys: list[tuple[str, str]] = []
        point_ids: list[str] = []
        try:
            for key, point_id, image_path, _mtime_ns, _size in fallback_items:
                try:
                    with Image.open(image_path) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                    keys.append(key)
                    point_ids.append(point_id)
                except Exception:
                    logger.exception("Failed to load DINOv3 patch candidate image %s", image_path)
            if not images:
                return result, {
                    "elapsed": perf_counter() - started,
                    "hits": hits,
                    "misses": len(qdrant_pending),
                    "qdrant_hits": qdrant_hits,
                    "fallback_images": 0,
                    "images": len(items),
                }

            encoded = self.connection_manager.dinov3.embed_patch_token_arrays_many(images)
            for key, (tokens, grid) in zip(keys, encoded, strict=True):
                if tokens is not None and tokens.size > 0:
                    result[key] = (tokens, grid)
            for point_id, (tokens, grid) in zip(point_ids, encoded, strict=True):
                if tokens is not None and tokens.size > 0:
                    self._store_patch_token_cache(point_id, tokens, grid)
            return result, {
                "elapsed": perf_counter() - started,
                "hits": hits,
                "misses": len(qdrant_pending),
                "qdrant_hits": qdrant_hits,
                "fallback_images": len(images),
                "images": len(items),
            }
        finally:
            for image in images:
                image.close()

    @staticmethod
    def _patch_point_id(wine_id: str, photo_id: str, view: str) -> str:
        return str(uuid.uuid5(PATCH_POINT_NAMESPACE, f"{wine_id}/{photo_id}/{view}"))

    @staticmethod
    def _patch_grid_from_payload(payload: dict[str, Any]) -> tuple[int, int] | None:
        try:
            height = int(payload["patch_grid_h"])
            width = int(payload["patch_grid_w"])
        except (KeyError, TypeError, ValueError):
            return None
        return height, width

    def _store_patch_token_cache(
        self,
        cache_key: str,
        tokens: np.ndarray,
        grid: tuple[int, int] | None,
    ) -> None:
        self._patch_token_cache[cache_key] = PatchTokenCacheEntry(
            mtime_ns=0,
            size=0,
            tokens=tokens,
            grid=grid,
        )
        self._patch_token_cache.move_to_end(cache_key)
        while len(self._patch_token_cache) > PATCH_TOKEN_CACHE_MAX_ITEMS:
            self._patch_token_cache.popitem(last=False)

    async def _rerank_with_ocr(
        self,
        candidates: list[WineCandidate],
        ocr_text: str,
    ) -> tuple[list[WineCandidate], dict[str, Any]]:
        if not candidates:
            return candidates, {
                "rerank_applied": False,
                "reason": "no_candidates",
                "candidate_scores": [],
            }
        if not ocr_text:
            return candidates, {
                "rerank_applied": False,
                "reason": "empty_ocr_text",
                "candidate_scores": [],
            }

        wines = await self.repository.load_wines_by_ids([candidate.wine_id for candidate in candidates])
        wine_by_id = {str(wine.id): wine for wine in wines}
        reranked: list[WineCandidate] = []
        diagnostics: list[dict[str, Any]] = []
        for candidate in candidates:
            wine = wine_by_id.get(candidate.wine_id)
            candidate_text = self._wine_to_ocr_candidate_text(wine)
            normalized_candidate = normalize_match_text(candidate_text)
            wratio = fuzz.WRatio(ocr_text, normalized_candidate) if normalized_candidate else 0.0
            token_set = fuzz.token_set_ratio(ocr_text, normalized_candidate) if normalized_candidate else 0.0
            fuzzy_score = (
                OCR_FUZZY_WRATIO_WEIGHT * wratio
                + OCR_FUZZY_TOKEN_SET_WEIGHT * token_set
            ) / 100.0
            structured_score = self._structured_ocr_score(ocr_text, wine)
            ocr_score = OCR_DOMAIN_WEIGHT * structured_score["domain_score"] + OCR_FUZZY_WEIGHT * fuzzy_score
            final_score = OCR_VISUAL_WEIGHT * candidate.score + OCR_TEXT_WEIGHT * ocr_score
            reranked.append(
                WineCandidate(
                    wine_id=candidate.wine_id,
                    slug=candidate.slug,
                    score=final_score,
                    max_score=candidate.max_score,
                    mean_score=candidate.mean_score,
                    n_photos=candidate.n_photos,
                    score_std=candidate.score_std,
                    cosine_score=candidate.cosine_score,
                    view_photo_ids=candidate.view_photo_ids,
                    patch_score=candidate.patch_score,
                )
            )
            diagnostics.append(
                {
                    "wine_id": candidate.wine_id,
                    "visual_score": candidate.score,
                    "ocr_score": ocr_score,
                    "domain_score": structured_score["domain_score"],
                    "fuzzy_score": fuzzy_score,
                    "grape_score": structured_score["grape_score"],
                    "producer_score": structured_score["producer_score"],
                    "name_score": structured_score["name_score"],
                    "candidate_grapes": structured_score["candidate_grapes"],
                    "wratio": wratio,
                    "token_set_ratio": token_set,
                    "final_score": final_score,
                    "candidate_text": normalized_candidate,
                }
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        diagnostics.sort(key=lambda item: item["final_score"], reverse=True)
        logger.info(
            "OCR rerank applied for %s candidates. Top candidate text scores: %s",
            len(candidates),
            [
                {
                    "wine_id": item["wine_id"],
                    "visual_score": round(float(item["visual_score"]), 4),
                    "ocr_score": round(float(item["ocr_score"]), 4),
                    "domain_score": round(float(item["domain_score"]), 4),
                    "fuzzy_score": round(float(item["fuzzy_score"]), 4),
                    "grape_score": round(float(item["grape_score"]), 4),
                    "producer_score": round(float(item["producer_score"]), 4),
                    "name_score": round(float(item["name_score"]), 4),
                    "candidate_grapes": item["candidate_grapes"],
                    "wratio": round(float(item["wratio"]), 2),
                    "token_set_ratio": round(float(item["token_set_ratio"]), 2),
                    "final_score": round(float(item["final_score"]), 4),
                    "candidate_text": item["candidate_text"],
                }
                for item in diagnostics[:10]
            ],
        )
        return reranked, {
            "rerank_applied": True,
            "reason": "ok",
            "candidate_scores": diagnostics[:10],
        }

    @classmethod
    def _structured_ocr_score(
        cls,
        ocr_text: str,
        wine: Wine | None,
    ) -> dict[str, Any]:
        if wine is None:
            return {
                "domain_score": 0.0,
                "grape_score": 0.0,
                "producer_score": 0.0,
                "name_score": 0.0,
                "candidate_grapes": [],
            }

        grape_groups = cls._wine_grape_groups(wine)
        grape_score = cls._grape_field_score(ocr_text, grape_groups)
        producer_score = cls._field_score(
            ocr_text,
            [wine.producer.name] if wine.producer is not None else [],
        )
        name_score = cls._field_score(ocr_text, [wine.name])
        domain_score = (
            OCR_GRAPE_WEIGHT * grape_score
            + OCR_PRODUCER_WEIGHT * producer_score
            + OCR_NAME_WEIGHT * name_score
        )
        return {
            "domain_score": domain_score,
            "grape_score": grape_score,
            "producer_score": producer_score,
            "name_score": name_score,
            "candidate_grapes": [
                normalize_match_text(variant)
                for group in grape_groups
                for variant in group
            ],
        }

    @classmethod
    def _wine_grape_groups(
        cls,
        wine: Wine,
    ) -> list[list[str]]:
        groups: list[list[str]] = []
        seen: set[str] = set()
        for link in wine.grape_links:
            grape = link.grape
            if grape is None:
                continue
            normalized = normalize_match_text(grape.name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            groups.append(cls._grape_variants(grape))

        return groups

    @classmethod
    def _grape_variants(cls, grape: Any) -> list[str]:
        variants = [getattr(grape, "name", "")]
        variants.extend(alias.alias for alias in getattr(grape, "aliases", []) if alias.alias)
        return cls._dedupe_text_variants(variants)

    @classmethod
    def _grape_field_score(cls, ocr_text: str, grape_groups: list[list[str]]) -> float:
        if not grape_groups:
            return 0.5
        group_scores = sorted(
            (cls._field_score(ocr_text, group) for group in grape_groups),
            reverse=True,
        )
        best_score = group_scores[0]
        extra_matches = sum(
            1
            for score in group_scores[1:]
            if score >= OCR_GRAPE_EXTRA_MATCH_THRESHOLD
        )
        return min(1.0, best_score + OCR_GRAPE_EXTRA_MATCH_BONUS * extra_matches)

    @staticmethod
    def _field_score(ocr_text: str, variants: list[str]) -> float:
        normalized_variants = [normalize_match_text(variant) for variant in variants if variant]
        if not normalized_variants:
            return 0.5

        scores: list[float] = []
        for variant in normalized_variants:
            scores.append(WineService._single_field_score(ocr_text, variant))
        return max(scores, default=0.0)

    @staticmethod
    def _single_field_score(ocr_text: str, normalized_variant: str) -> float:
        return max(
            fuzz.partial_ratio(ocr_text, normalized_variant),
            fuzz.partial_token_sort_ratio(ocr_text, normalized_variant),
            fuzz.WRatio(ocr_text, normalized_variant),
        ) / 100.0

    @staticmethod
    def _dedupe_text_variants(variants: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = normalize_match_text(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(variant)
        return result

    @staticmethod
    def _wine_to_ocr_candidate_text(wine: Wine | None) -> str:
        if wine is None:
            return ""
        grape_names = [
            link.grape.name
            for link in wine.grape_links
            if link.grape is not None
        ]
        parts = [
            wine.producer.name if wine.producer else None,
            wine.name,
            wine.year,
            " ".join(grape_names),
            wine.color,
            wine.sugar,
        ]
        return " ".join(str(part) for part in parts if part)

    async def _rerank_with_llm_choice(
        self,
        image_bytes: bytes,
        candidates: list[WineCandidate],
    ) -> tuple[list[WineCandidate], dict[str, Any] | None]:
        if not candidates or getattr(self.connection_manager, "llm", None) is None:
            return candidates, None

        started = perf_counter()
        wines = await self.repository.load_wines_by_ids([candidate.wine_id for candidate in candidates])
        wine_by_id = {str(wine.id): wine for wine in wines}
        llm_candidates = [
            self._wine_to_llm_candidate(idx, candidate, wine_by_id.get(candidate.wine_id))
            for idx, candidate in enumerate(candidates, start=1)
        ]
        try:
            selection = await self.connection_manager.llm.choose_wine_from_image(
                image_bytes=image_bytes,
                candidates=llm_candidates,
            )
        except Exception:
            logger.exception("LLM wine rerank failed")
            return candidates, None

        selected_idx = selection.selected_number - 1
        if selected_idx < 0 or selected_idx >= len(candidates):
            logger.warning("LLM selected out-of-range candidate number: %s", selection.selected_number)
            return candidates, {
                "selected_number": selection.selected_number,
                "selected_wine_id": selection.selected_wine_id,
                "confidence": selection.confidence,
                "reason": selection.reason,
                "elapsed": perf_counter() - started,
                "valid": False,
            }

        selected = candidates[selected_idx]
        reordered = [selected] + [candidate for idx, candidate in enumerate(candidates) if idx != selected_idx]
        logger.info(
            "LLM rerank selected number=%s wine_id=%s confidence=%s elapsed=%.2fs",
            selection.selected_number,
            selected.wine_id,
            selection.confidence,
            perf_counter() - started,
        )
        return reordered, {
            "selected_number": selection.selected_number,
            "selected_wine_id": selected.wine_id,
            "model_selected_wine_id": selection.selected_wine_id,
            "confidence": selection.confidence,
            "reason": selection.reason,
            "elapsed": perf_counter() - started,
            "valid": True,
        }

    @staticmethod
    def _wine_to_llm_candidate(index: int, candidate: WineCandidate, wine: Wine | None) -> dict[str, Any]:
        grape_names = []
        if wine is not None:
            grape_names = [
                link.grape.name
                for link in wine.grape_links
                if link.grape is not None
            ]
        return {
            "number": index,
            "wine_id": candidate.wine_id,
            "name": wine.name if wine is not None else candidate.slug,
            "color": wine.color if wine is not None else None,
            "sugar": wine.sugar if wine is not None else None,
            "alcohol": wine.alcohol if wine is not None else None,
            "producer_name": wine.producer.name if wine is not None and wine.producer else None,
            "region_name": wine.region.name if wine is not None and wine.region else None,
            "grape_name": ", ".join(grape_names),
        }

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
