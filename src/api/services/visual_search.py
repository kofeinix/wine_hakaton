"""Визуальный поиск: кропы запроса -> SigLIP2-эмбеддинги -> Qdrant -> кандидаты-вина."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from io import BytesIO
from statistics import mean, pstdev
from typing import Any

from PIL import Image
from qdrant_client import models as qdrant_models

logger = logging.getLogger(__name__)

# Вес каждого view в итоговом визуальном скоре (нормализуется по views с результатами).
VIEW_WEIGHTS = {
    "original": 0.30,
    "bottle_crop": 0.15,
    "label_crop": 0.55,
}
PER_VIEW_TOP_K = 150  # точек из каждой коллекции
GLOBAL_CANDIDATE_LIMIT = 50  # вин-кандидатов после группировки
# Фото крупным планом этикетки: этикетка занимает большую часть кадра -> ищем только по ней.
LABEL_PHOTO_AREA_THRESHOLD = 0.60
LABEL_PHOTO_CONFIDENCE_THRESHOLD = 0.50
# Полка / витрина: столько этикеток и больше -> весь кадр не описывает нужное вино, ищем без original.
MULTI_WINE_LABEL_THRESHOLD = 3
# Скор view для вина: 0.75 * лучший cosine + 0.25 * среднее top-3 фото.
VIEW_SCORE_TOP_K_PHOTOS = 3
VIEW_SCORE_BEST_WEIGHT = 0.75
VIEW_SCORE_MEAN_WEIGHT = 0.25


@dataclass(frozen=True)
class ViewMatch:
    score: float
    rank: int  # позиция точки в выдаче Qdrant для этого view (1 = лучшая)
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
    view_match_counts: dict[str, int] | None = None
    # после OCR-реранка: score = visual_score + ocr_score
    visual_score: float | None = None
    ocr_score: float = 0.0


@dataclass(frozen=True)
class ViewSelection:
    active_views: list[str]
    label_area_ratio: float
    label_confidence: float
    label_photo_mode: bool
    labels_detected: int = 0
    multi_wine_mode: bool = False


@dataclass(frozen=True)
class VisualMatches:
    selection: ViewSelection
    wine_view_matches: dict[str, dict[str, list[ViewMatch]]]  # wine_id -> view -> совпадения
    wine_slugs: dict[str, str]
    per_view_counts: dict[str, int]
    weights: dict[str, float]


def select_views(query_views, allowed: list[str] | None = None) -> ViewSelection:
    """Views, чей кроп построился; для фото этикетки крупным планом — только label_crop."""
    allowed_set = set(allowed) if allowed else set(VIEW_WEIGHTS)
    active_views = [
        view
        for view in VIEW_WEIGHTS
        if view in query_views.images
        and view in allowed_set
        and query_views.crops.get(view) is not None
        and query_views.crops[view].available
    ]
    label_area_ratio = _label_crop_area_ratio(query_views.crops)
    label_confidence = _label_crop_confidence(query_views.crops)
    label_photo_mode = (
        label_area_ratio > LABEL_PHOTO_AREA_THRESHOLD
        and label_confidence > LABEL_PHOTO_CONFIDENCE_THRESHOLD
    )
    if label_photo_mode and "label_crop" in active_views:
        active_views = ["label_crop"]
        logger.info(
            "Detected label-photo input: label_area_ratio=%.3f confidence=%.3f; using label_crop only.",
            label_area_ratio,
            label_confidence,
        )

    detections = getattr(query_views, "detections", None)
    labels_detected = len(getattr(detections, "labels", None) or [])
    multi_wine_mode = labels_detected >= MULTI_WINE_LABEL_THRESHOLD
    # явно переданные views (эксперименты) не переопределяем
    if multi_wine_mode and allowed is None and "original" in active_views and len(active_views) > 1:
        active_views = [view for view in active_views if view != "original"]
        logger.info("Detected %d labels (shelf photo); searching without the original view.", labels_detected)
    return ViewSelection(
        active_views,
        label_area_ratio,
        label_confidence,
        label_photo_mode,
        labels_detected=labels_detected,
        multi_wine_mode=multi_wine_mode,
    )


def normalized_weights(views: list[str]) -> dict[str, float]:
    total = sum(VIEW_WEIGHTS[view] for view in views)
    if total <= 0:
        return {}
    return {view: VIEW_WEIGHTS[view] / total for view in views}


def group_by_wine(matches: VisualMatches, limit: int) -> list[WineCandidate]:
    """Визуальный скор вина по всем views, лучшие `limit` кандидатов."""
    candidates: list[WineCandidate] = []
    for wine_id, views in matches.wine_view_matches.items():
        view_scores: list[float] = []
        raw_cosine_scores: list[float] = []
        view_photo_ids: dict[str, str] = {}
        view_match_counts: dict[str, int] = {}
        for view, view_matches in views.items():
            ranked = sorted(view_matches, key=lambda item: item.score, reverse=True)
            best = ranked[0]
            top_mean = mean(item.score for item in ranked[:VIEW_SCORE_TOP_K_PHOTOS])
            view_score = VIEW_SCORE_BEST_WEIGHT * best.score + VIEW_SCORE_MEAN_WEIGHT * top_mean
            raw_cosine_scores.append(best.score)
            view_scores.append(matches.weights.get(view, 0.0) * view_score)
            view_photo_ids[view] = best.photo_id
            view_match_counts[view] = len(view_matches)
        if not view_scores:
            continue
        max_score = max(view_scores)
        # Дополнительный view не должен снижать итоговый score: лучший сигнал главный,
        # остальные дают ограниченный бонус.
        score = 0.7 * max_score + 0.3 * sum(view_scores)
        candidates.append(
            WineCandidate(
                wine_id=wine_id,
                slug=matches.wine_slugs.get(wine_id, wine_id),
                score=score,
                max_score=max_score,
                mean_score=mean(view_scores),
                n_photos=len(view_scores),
                score_std=pstdev(view_scores) if len(view_scores) > 1 else 0.0,
                cosine_score=max(raw_cosine_scores),
                view_photo_ids=view_photo_ids,
                view_match_counts=view_match_counts,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:limit]


def main_photos_filter() -> qdrant_models.Filter:
    return qdrant_models.Filter(
        must=[qdrant_models.FieldCondition(key="photo_id", match=qdrant_models.MatchValue(value="main"))]
    )


def ocr_source_view(query_views) -> str:
    """Кроп для OCR: этикетка -> бутылка -> исходное фото."""
    for view in ("label_crop", "bottle_crop"):
        crop = query_views.crops.get(view)
        if crop is not None and getattr(crop, "available", False) and view in query_views.images:
            return view
    return "original"


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue()


class VisualSearcher:
    def __init__(self, embedder, qdrant, collection_encoder: str = "siglip2") -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_encoder = collection_encoder

    def collection(self, view: str) -> str:
        return f"wine_{view}_{self.collection_encoder}"

    def embed(self, query_views, selection: ViewSelection) -> list[list[float]]:
        """Синхронная CPU/GPU-часть — можно вынести в поток."""
        if not selection.active_views:
            return []
        return self.embedder.embed_many([query_views.images[view] for view in selection.active_views])

    async def search(
        self,
        selection: ViewSelection,
        vectors: list[list[float]],
        query_filter: qdrant_models.Filter | None = None,
    ) -> VisualMatches:
        responses = await asyncio.gather(
            *[
                self.qdrant.search_collection(
                    collection_name=self.collection(view),
                    vector=vector,
                    limit=PER_VIEW_TOP_K,
                    with_payload=True,
                    query_filter=query_filter,
                )
                for view, vector in zip(selection.active_views, vectors, strict=True)
            ]
        )
        wine_view_matches: dict[str, dict[str, list[ViewMatch]]] = {}
        wine_slugs: dict[str, str] = {}
        per_view_counts: dict[str, int] = {}
        for view, response in zip(selection.active_views, responses, strict=True):
            points = list(getattr(response, "points", []) or []) if response is not None else []
            per_view_counts[view] = len(points)
            for rank, point in enumerate(points, start=1):
                payload = getattr(point, "payload", None) or {}
                wine_id = payload.get("wine_id")
                if not wine_id:  # фото без wine_id не привязать к вину
                    continue
                point_id = str(getattr(point, "id", ""))
                wine_view_matches.setdefault(str(wine_id), {}).setdefault(view, []).append(
                    ViewMatch(
                        score=float(getattr(point, "score", 0.0) or 0.0),
                        rank=rank,
                        point_id=point_id,
                        photo_id=str(payload.get("photo_id") or point_id),
                    )
                )
                if payload.get("slug"):
                    wine_slugs.setdefault(str(wine_id), str(payload["slug"]))
        # Веса нормализуются только по views, которые реально вернули результаты.
        weights = normalized_weights([view for view in selection.active_views if per_view_counts.get(view)])
        return VisualMatches(selection, wine_view_matches, wine_slugs, per_view_counts, weights)


def _label_crop_area_ratio(crops: dict[str, Any]) -> float:
    label = crops.get("label_crop")
    original = crops.get("original")
    if label is None or original is None or not getattr(label, "available", False):
        return 0.0
    if getattr(label, "source_view", "original") not in {"original", "original_fallback"}:
        return 0.0
    label_box = getattr(label, "box", None)
    width = getattr(original, "width", None)
    height = getattr(original, "height", None)
    if label_box is None or not width or not height:
        return 0.0
    x1, y1, x2, y2 = label_box
    return max(0, x2 - x1) * max(0, y2 - y1) / max(1, int(width) * int(height))


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
