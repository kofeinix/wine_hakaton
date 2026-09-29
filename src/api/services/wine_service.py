"""Поиск вина по фото.

Пайплайн основного поиска:
  1. кропы запроса (YOLO: бутылка, этикетка) и визуальный поиск по Qdrant — visual_search.py;
  2. OCR этикетки (vision-LLM) и реранк: final = visual_score + ocr_bonus — ocr_matching.py.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
from contextlib import suppress
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import structlog

from src.api.repositories.wine_repository import WineRepository
from src.api.schemas import (
    DetectionsResponse,
    OcrInfo,
    SearchInfo,
    SearchMatch,
    SearchResponse,
    SearchTimings,
    WinePhoto,
    WineResponse,
)
from src.api.services.ocr_match_service import OcrMatchService
from src.api.services.photo_service import QueryImageViews, WinePhotoService
from src.api.services.similarity import max_score, similarities
from src.api.services.visual_search import (
    GLOBAL_CANDIDATE_LIMIT,
    LABEL_PHOTO_AREA_THRESHOLD,
    LABEL_PHOTO_CONFIDENCE_THRESHOLD,
    MULTI_WINE_LABEL_THRESHOLD,
    PER_VIEW_TOP_K,
    VisualMatches,
    VisualSearcher,
    WineCandidate,
    group_by_wine,
    image_to_jpeg_bytes,
    main_photos_filter,
    ocr_source_view,
    select_views,
)
from src.connections.database.models import Wine
from src.settings.logging_setup import set_request_log_fields
from src.ml.text_normalization import normalize_match_text
from src.ml.view_adapter import ViewAdapter

logger = structlog.get_logger(__name__)

OCR_RERANK_LIMIT = 50
OCR_CACHE_SIZE = 512  # распознанный текст по хешу кропа: повторный поиск того же фото без LLM

SUGAR_VARIANTS = {
    normalize_match_text(key): [key, *variants]
    for key, variants in {
        "брют": ["brut", "брит", "бриот", "брю", "brut nature", "extra brut", "экстра брют"],
        "экстра брют": ["extra brut", "экстрабрют", "брют экстра"],
        "сухое": ["сухой", "сухая", "dry", "sec"],
        "полусухое": ["полусухой", "полусухая", "semi dry", "semidry", "demi sec"],
        "полусладкое": ["полусладкий", "полусладкая", "semi sweet", "semisweet", "doux"],
        "сладкое": ["сладкий", "сладкая", "sweet"],
    }.items()
}


@dataclass
class SearchOutcome:
    """Результат поиска до сборки HTTP-ответа."""

    candidates: list[WineCandidate]  # итоговый порядок, не больше limit
    crops: dict[str, Any]
    ocr: dict[str, Any]
    matches: VisualMatches
    reranked: int  # сколько кандидатов прошло OCR-реранк
    timings_ms: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    query_views: QueryImageViews | None = None
    # сходство 0–1 для каждого кандидата: скор / максимально возможный в этом поиске
    similarities: list[float] = field(default_factory=list)


class WineService:
    def __init__(self, connection_manager) -> None:
        self.connection_manager = connection_manager
        self.repository = WineRepository(connection_manager.database)
        self.photo_service = WinePhotoService(
            label_cropper=connection_manager.label_yolo,
            bottle_cropper=connection_manager.bottle_yolo,
        )
        self.visual = VisualSearcher(
            embedder=connection_manager.embeddings,
            qdrant=connection_manager.qdrant,
            collection_encoder=connection_manager.settings.search.collection_encoder,
            adapter=ViewAdapter.load(connection_manager.settings.embeddings.adapter_path),
        )
        self.ocr_matcher = OcrMatchService(self.repository, SUGAR_VARIANTS)
        self.ocr_skip_visual_gap = connection_manager.settings.search.ocr_skip_visual_gap
        self.ocr_skip_min_score = connection_manager.settings.search.ocr_skip_min_score
        self.ocr_budget_seconds = connection_manager.settings.search.ocr_budget_seconds
        # YOLO и SigLIP — CPU/GPU-работа: в отдельном потоке, чтобы не блокировать event loop.
        # Один воркер — модели не вызываются параллельно (Ultralytics не потокобезопасен).
        self._cv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv")
        self._ocr_cache: OrderedDict[str, str] = OrderedDict()

    async def _cv(self, fn, *args):
        # copy_context: логи из потока CV тоже с request_id (run_in_executor контекст не переносит)
        context = contextvars.copy_context()
        return await asyncio.get_running_loop().run_in_executor(self._cv_executor, context.run, fn, *args)

    # --- публичные методы (routes.py) ------------------------------------------

    async def build_response(
        self,
        outcome: SearchOutcome,
        search_id: str | None = None,
        debug: bool = False,
    ) -> SearchResponse:
        """Ответ /search/image/extended: результаты с разложением скора и карточками вин."""
        wines = await self.repository.load_wines_by_ids([c.wine_id for c in outcome.candidates])
        cards = {str(wine.id): wine_to_response(wine) for wine in wines}
        selection = outcome.matches.selection
        return SearchResponse(
            search_id=search_id,
            status="found" if outcome.candidates else "not_found",
            results=[
                SearchMatch(
                    rank=rank,
                    slug=candidate.slug,
                    wine_id=candidate.wine_id,
                    visual_score=round(_visual_score(candidate), 4),
                    ocr_score=round(candidate.ocr_score, 4),
                    final_score=round(candidate.score, 4),
                    similarity=round(outcome.similarities[rank - 1], 4) if outcome.similarities else 0.0,
                    wine=cards.get(candidate.wine_id),
                )
                for rank, candidate in enumerate(outcome.candidates, start=1)
            ],
            ocr=OcrInfo(
                applied=outcome.ocr["applied"],
                source_view=outcome.ocr["source_view"],
                text=outcome.ocr["text"],
                cached=outcome.ocr["cached"],
                skipped=outcome.ocr["skipped"],
                reason=outcome.ocr["reason"],
            ),
            crops=outcome.crops,
            image=outcome.query_views.image if outcome.query_views else None,
            detections=outcome.query_views.detections if outcome.query_views else DetectionsResponse(),
            search=SearchInfo(
                active_views=selection.active_views,
                label_photo_mode=selection.label_photo_mode,
                multi_wine_mode=selection.multi_wine_mode,
                labels_detected=selection.labels_detected,
                candidates=outcome.reranked,
                max_score=round(max_score(outcome.matches.weights, bool(outcome.ocr["applied"])), 4),
            ),
            timings_ms=SearchTimings(**outcome.timings_ms),
            diagnostics=outcome.diagnostics if debug else None,
        )

    async def get_wine(self, wine_id: str) -> WineResponse | None:
        wine = await self.repository.get_wine(wine_id)
        if wine is None:
            return None
        return wine_to_response(wine)

    async def get_photo(self, photo_id: str):
        """(BytesIO, content_type) фото из MinIO или None.

        Для главного фото сначала webp (прозрачный фон, в 2–3 раза легче); jpg — если webp нет
        в БД или в MinIO (например, архив webp не загружен).
        """
        image = await self.repository.get_image(photo_id)
        if image is None:
            return None
        minio = self.connection_manager.minio
        if image.webp_minio_path:
            photo = await minio.get_file_with_content_type(image.webp_minio_path, missing_ok=True)
            if photo is not None:
                return photo
        return await minio.get_file_with_content_type(image.minio_path)

    # --- основной поиск ---------------------------------------------------------

    async def search(
        self,
        image_bytes: bytes,
        limit: int = 10,
        views: list[str] | None = None,
        main_photos_only: bool = False,
    ) -> SearchOutcome:
        """Кропы -> (OCR || визуальный поиск) -> OCR-реранк."""
        started = perf_counter()
        timings: dict[str, float] = {}
        logger.info(
            "image_search_started",
            image_bytes=len(image_bytes),
            limit=limit,
            requested_views=views,
            main_photos_only=main_photos_only,
        )

        def ms(since: float) -> float:
            return round((perf_counter() - since) * 1000, 1)

        query_views = await self._cv(self.photo_service.build_query_views, image_bytes)
        timings["crops_ms"] = ms(started)
        logger.info(
            "image_crops",
            image_size=_image_size(query_views),
            bottle_crop=_crop_log(query_views.crops.get("bottle_crop")),
            label_crop=_crop_log(query_views.crops.get("label_crop")),
            detections={
                "bottles": len(query_views.detections.bottles),
                "labels": len(query_views.detections.labels),
            },
            duration_ms=timings["crops_ms"],
        )
        selection = select_views(query_views, views)
        logger.info(
            "image_views_selected",
            views=selection.active_views,
            label_area_ratio=round(selection.label_area_ratio, 4),
            label_confidence=round(selection.label_confidence, 4),
            label_photo_mode=selection.label_photo_mode,
            labels_detected=selection.labels_detected,
            multi_wine_mode=selection.multi_wine_mode,
        )

        async def ocr_stage() -> dict[str, Any]:
            since = perf_counter()
            result = await self._extract_ocr_text(query_views)
            timings["ocr_ms"] = ms(since)
            return result

        async def visual_stage() -> VisualMatches:
            since = perf_counter()
            vectors = await self._cv(self.visual.embed, query_views, selection)
            timings["embedding_ms"] = ms(since)
            since = perf_counter()
            result = await self.visual.search(selection, vectors, main_photos_filter() if main_photos_only else None)
            timings["vector_search_ms"] = ms(since)
            return result

        # OCR ждёт LLM, визуальный поиск — модель и Qdrant: независимы, идут параллельно
        ocr_task = asyncio.create_task(ocr_stage())
        try:
            matches = await visual_stage()
        except BaseException:
            ocr_task.cancel()
            raise
        global_limit = max(limit, GLOBAL_CANDIDATE_LIMIT)
        candidates = group_by_wine(matches, limit=global_limit)
        logger.info("visual_search", candidates=len(candidates), **_top_gap(candidates))

        visual_gap = _relative_gap(candidates)
        top_score = candidates[0].score if candidates else 0.0
        if not ocr_task.done() and 0 < self.ocr_skip_visual_gap < visual_gap and top_score >= self.ocr_skip_min_score:
            # визуальный результат уверенный: top-1 далеко впереди и сам по себе похож — OCR его не меняет
            # (проверено на eval), не ждём LLM. Слабый top-1 с большим отрывом (вина нет в каталоге,
            # сложное фото) — как раз случай, когда нужен текст этикетки
            ocr_task.cancel()
            with suppress(asyncio.CancelledError):
                await ocr_task
            logger.info(
                "ocr_skipped",
                reason="confident_visual",
                visual_gap=round(visual_gap, 4),
                top_score=round(top_score, 4),
                skip_gap=self.ocr_skip_visual_gap,
                skip_min_score=self.ocr_skip_min_score,
            )
            ocr = _ocr_result("skipped_confident_visual", ocr_source_view(query_views), skipped=True)
            timings["ocr_ms"] = 0.0
        else:
            ocr = await self._await_ocr(ocr_task, started, query_views, timings)
        mark = perf_counter()
        candidates, ocr_diagnostics = await self._rerank_with_ocr(candidates[:OCR_RERANK_LIMIT], ocr["text"])
        timings["rerank_ms"] = ms(mark)
        timings["total_ms"] = ms(started)

        diagnostics = {
            "visual": {
                **self._visual_diagnostics(matches),
                "aggregation": "view_best75_top3mean25_then_weighted_max_sum",
                "candidate_limit": global_limit,
                "main_photos_only": main_photos_only,
            },
            "ocr_rerank": {
                **ocr_diagnostics,
                "ocr_status": ocr["reason"],
                "visual_gap": round(visual_gap, 4) if visual_gap != float("inf") else None,
                "ocr_skip_visual_gap": self.ocr_skip_visual_gap,
                "ocr_skip_min_score": self.ocr_skip_min_score,
                "normalized_text": ocr["normalized_text"],
                "formula": "final = visual_score + ocr_bonus (src/ml/ocr_matching.py)",
            },
        }
        top = candidates[0] if candidates else None
        set_request_log_fields(
            search_top1_slug=top.slug if top else None,
            search_top1_score=round(top.score, 4) if top else None,
            search_ocr_status=ocr["reason"],
            search_candidates=len(candidates),
            search_views=selection.active_views,
            search_timings_ms=timings,
        )
        logger.info(
            "image_search",
            image_bytes=len(image_bytes),
            views=selection.active_views,
            top1_slug=top.slug if top else None,
            top1_score=round(top.score, 4) if top else None,
            ocr_status=ocr["reason"],
            candidates=len(candidates),
            timings_ms=timings,
        )
        return SearchOutcome(
            similarities=similarities(
                [candidate.score for candidate in candidates[:limit]], matches.weights, bool(ocr["applied"])
            ),
            candidates=candidates[:limit],
            crops=query_views.crops,
            ocr=ocr,
            matches=matches,
            reranked=min(len(candidates), OCR_RERANK_LIMIT),
            timings_ms=timings,
            diagnostics=diagnostics,
            query_views=query_views,
        )

    def _visual_diagnostics(self, matches: VisualMatches) -> dict[str, Any]:
        selection = matches.selection
        return {
            "active_views": selection.active_views,
            "weights": matches.weights,
            "collection_encoder": self.visual.collection_encoder,
            "collections": {view: self.visual.collection(view) for view in selection.active_views},
            "per_view_top_k": PER_VIEW_TOP_K,
            "per_view_counts": matches.per_view_counts,
            "label_area_ratio": selection.label_area_ratio,
            "label_confidence": selection.label_confidence,
            "label_photo_mode": selection.label_photo_mode,
            "labels_detected": selection.labels_detected,
            "multi_wine_mode": selection.multi_wine_mode,
            "multi_wine_label_threshold": MULTI_WINE_LABEL_THRESHOLD,
            "label_photo_area_threshold": LABEL_PHOTO_AREA_THRESHOLD,
            "label_photo_confidence_threshold": LABEL_PHOTO_CONFIDENCE_THRESHOLD,
        }

    # --- OCR ----------------------------------------------------------------------

    async def _await_ocr(self, ocr_task: asyncio.Task, started: float, query_views, timings: dict) -> dict[str, Any]:
        """Результат OCR, но не дольше бюджета от начала поиска: медленная LLM не должна держать ответ."""
        if self.ocr_budget_seconds <= 0 or ocr_task.done():
            return await ocr_task
        remaining = self.ocr_budget_seconds - (perf_counter() - started)
        try:
            return await asyncio.wait_for(asyncio.shield(ocr_task), timeout=max(remaining, 0.0))
        except TimeoutError:
            await _cancel(ocr_task)
            logger.warning("ocr_timeout", budget_s=self.ocr_budget_seconds)
            timings["ocr_ms"] = round((perf_counter() - started) * 1000, 1)
            return _ocr_result("timeout_budget", ocr_source_view(query_views), skipped=True)

    async def _extract_ocr_text(self, query_views) -> dict[str, Any]:
        llm = self.connection_manager.llm
        source_view = ocr_source_view(query_views)
        if llm is None:
            return _ocr_result("llm_unavailable", None)
        image = query_views.images.get(source_view)
        if image is None:
            return _ocr_result("no_source_image", source_view)
        jpeg = image_to_jpeg_bytes(image)
        key = hashlib.sha1(jpeg).hexdigest()
        cached = self._ocr_cache.get(key)
        if cached is not None:
            self._ocr_cache.move_to_end(key)
            normalized = normalize_match_text(cached)
            return _ocr_result("ok" if normalized else "empty_ocr_text", source_view, cached, normalized, cached=True)
        try:
            text = await llm.ocr_image_text(jpeg)
        except Exception:
            logger.exception("ocr_failed", source_view=source_view)
            return _ocr_result("ocr_failed", source_view)
        self._ocr_cache[key] = text
        while len(self._ocr_cache) > OCR_CACHE_SIZE:
            self._ocr_cache.popitem(last=False)

        normalized = normalize_match_text(text)
        logger.info("ocr_text", source_view=source_view, applied=bool(normalized), raw_text=text, normalized_text=normalized)
        return _ocr_result("ok" if normalized else "empty_ocr_text", source_view, text, normalized)

    async def _rerank_with_ocr(
        self,
        candidates: list[WineCandidate],
        ocr_text: str,
    ) -> tuple[list[WineCandidate], dict[str, Any]]:
        """final = visual_score + ocr_bonus; формула и веса — в src/ml/ocr_matching.py."""
        if not candidates:
            return candidates, {"rerank_applied": False, "reason": "no_candidates", "candidate_scores": []}
        scores = await self.ocr_matcher.score(ocr_text, [candidate.wine_id for candidate in candidates])
        if not scores:
            return candidates, {"rerank_applied": False, "reason": "empty_ocr_text", "candidate_scores": []}

        visual_by_id = {candidate.wine_id: candidate.score for candidate in candidates}
        reranked = sorted(
            (
                replace(
                    candidate,
                    score=candidate.score + scores[candidate.wine_id].bonus,
                    visual_score=candidate.score,
                    ocr_score=scores[candidate.wine_id].bonus,
                )
                for candidate in candidates
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        candidate_scores = [
            {
                **scores[candidate.wine_id].diagnostics(),
                "visual_score": round(visual_by_id[candidate.wine_id], 4),
                "final_score": round(candidate.score, 4),
            }
            for candidate in reranked[:10]
        ]
        logger.info(
            "ocr_rerank",
            candidates=len(candidates),
            top=[
                {key: item[key] for key in ("wine_id", "visual_score", "ocr_bonus", "final_score")}
                for item in candidate_scores[:5]
            ],
        )
        return reranked, {
            "rerank_applied": True,
            "reason": "ok",
            "candidate_scores": candidate_scores,
            # весь пул до реранка — для офлайн-анализа формулы
            "pool": [
                {
                    "wine_id": candidate.wine_id,
                    "visual_score": round(candidate.score, 6),
                    "ocr_bonus": round(scores[candidate.wine_id].bonus, 6),
                }
                for candidate in candidates
            ],
        }

def wine_to_response(wine: Wine) -> WineResponse:
    """Карточка вина для ответов API (поиск, история, избранное, уведомления)."""
    def photo(image) -> WinePhoto:
        # ?v=webp — другой адрес для браузера: фото кешируются как immutable, и без него у тех, кто
        # уже открывал сайт, неделю показывался бы старый jpg
        suffix = "?v=webp2" if image.webp_minio_path else ""  # webp2 — после обрезки полей
        return WinePhoto(id=str(image.id), url=f"/api/v1/photos/{image.id}{suffix}", is_main=image.is_main)

    # main первым, затем остальные настоящие по имени файла (yandex_1, yandex_2, ...)
    real = sorted((i for i in wine.images if not i.is_generated), key=lambda i: (not i.is_main, i.minio_path))
    generated = sorted((i for i in wine.images if i.is_generated), key=lambda i: i.minio_path)
    photos = [photo(image) for image in real]
    return WineResponse(
        id=str(wine.id),
        slug=wine.sku,
        name=wine.name,
        producer=wine.producer.name if wine.producer else None,
        region=wine.region.name if wine.region else None,
        country=wine.region.country if wine.region else None,
        year=wine.year,
        color=wine.color,
        sugar=wine.sugar,
        alcohol=wine.alcohol,
        serving_temperature=wine.serving_temperature,
        shade=wine.shade,
        price=float(wine.price) if wine.price is not None else None,
        rating=float(wine.rating) if wine.rating is not None else None,
        description=wine.description,
        grapes=[link.grape.name for link in wine.grape_links if link.grape is not None],
        food_pairings=[link.food.name for link in wine.food_links if link.food is not None],
        source_url=wine.source_url,
        image_url=photos[0].url if photos else None,
        photos=photos,
        generated_photos=[photo(image) for image in generated],
    )


def _visual_score(candidate: WineCandidate) -> float:
    return candidate.visual_score if candidate.visual_score is not None else candidate.score


def _image_size(query_views: QueryImageViews) -> str | None:
    if query_views.image is None:
        return None
    return f"{query_views.image.width}x{query_views.image.height}"


def _crop_log(crop: Any) -> dict[str, Any]:
    if crop is None or not getattr(crop, "available", False):
        return {"available": False}
    width = getattr(crop, "width", None)
    height = getattr(crop, "height", None)
    return {
        "available": True,
        "source_view": getattr(crop, "source_view", None),
        "confidence": getattr(crop, "confidence", None),
        "size": f"{width}x{height}" if width and height else None,
    }


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def _ocr_result(
    reason: str,
    source_view: str | None,
    text: str = "",
    normalized: str = "",
    cached: bool = False,
    skipped: bool = False,
) -> dict[str, Any]:
    return {
        "applied": bool(normalized),
        "cached": cached,
        "skipped": skipped,
        "reason": reason,
        "source_view": source_view,
        "text": text,
        "normalized_text": normalized,
    }


def _relative_gap(candidates: list[WineCandidate]) -> float:
    """Отрыв визуального top-1 от top-2: (v1 - v2) / v2; один кандидат — бесконечность."""
    if not candidates:
        return 0.0
    if len(candidates) < 2 or candidates[1].score <= 0:
        return float("inf")
    return (candidates[0].score - candidates[1].score) / candidates[1].score


def _top_gap(candidates: list[WineCandidate]) -> dict[str, Any]:
    """top-1, top-2 и отрыв между ними — поля для лога визуального поиска."""
    fields: dict[str, Any] = {}
    if candidates:
        fields.update(top1_wine_id=candidates[0].wine_id, top1_score=round(candidates[0].score, 4))
    if len(candidates) > 1:
        fields.update(
            top2_wine_id=candidates[1].wine_id,
            top2_score=round(candidates[1].score, 4),
            gap=round(candidates[0].score - candidates[1].score, 4),
        )
    return fields
