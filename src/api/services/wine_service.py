"""Поиск вина по фото.

Пайплайн основного поиска:
  1. кропы запроса (YOLO: бутылка, этикетка) и визуальный поиск по Qdrant — visual_search.py;
  2. OCR этикетки (vision-LLM) и реранк: final = visual_score + ocr_bonus — ocr_matching.py;
  3. опционально (stage "llm") — выбор кандидата vision-LLM.

Отдельный эндпоинт /search/image/catboost ранжирует тех же кандидатов CatBoost-моделью.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from time import perf_counter
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
from src.api.services.ocr_match_service import OcrMatchService
from src.api.services.photo_service import WinePhotoService
from src.api.services.visual_search import (
    GLOBAL_CANDIDATE_LIMIT,
    LABEL_PHOTO_AREA_THRESHOLD,
    LABEL_PHOTO_CONFIDENCE_THRESHOLD,
    PER_VIEW_TOP_K,
    VisualMatches,
    VisualSearcher,
    WineCandidate,
    group_by_wine,
    image_to_jpeg_bytes,
    main_photos_filter,
    ocr_source_view,
)
from src.connections.database.models import Wine
from src.ml.catboost_reranker import CatBoostWineReranker, build_candidate_feature_rows
from src.ml.text_normalization import normalize_match_text

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

OCR_RERANK_LIMIT = 50
# "global" (визуальный поиск + OCR-реранк) выполняется всегда; "llm" — опциональный выбор LLM.
VALID_SEARCH_STAGES = ("global", "llm")
CATBOOST_RERANKER_MODEL_PATH = PROJECT_ROOT / "models" / "catboost" / "wine_reranker.cbm"

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
        )
        self.ocr_matcher = OcrMatchService(self.repository, SUGAR_VARIANTS)
        self._catboost_reranker: CatBoostWineReranker | None = None
        self._catboost_load_failed = False

    # --- публичные методы (routes.py) ------------------------------------------

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
        return CompactSearchResponse(result=[CompactSearchMatch(**_match_fields(c)) for c in candidates])

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
            SearchMatchResponse(**_match_fields(candidate), wine=wine_by_id.get(candidate.wine_id))
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
            top5=[{"slug": result.slug, "score": result.score} for result in results[:5]],
            results=results,
            crops=crops,
            diagnostics={**diagnostics, "filename": filename, "content_type": content_type},
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
        return CompactSearchResponse(result=[CompactSearchMatch(**_match_fields(c)) for c in candidates])

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

    # --- основной поиск ---------------------------------------------------------

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
        ocr = await self._extract_ocr_text(query_views)
        matches = await self.visual.run(
            query_views,
            allowed=views,
            query_filter=main_photos_filter() if main_photos_only else None,
        )
        global_limit = max(limit, GLOBAL_CANDIDATE_LIMIT)
        candidates = group_by_wine(matches, limit=global_limit)
        logger.info("Global search found top_%s candidates. %s", len(candidates), _format_top_gap(candidates))

        candidates, ocr_diagnostics = await self._rerank_with_ocr(candidates[:OCR_RERANK_LIMIT], ocr["text"])

        llm_selection: dict[str, Any] | None = None
        if "llm" in search_stages:
            candidates, llm_selection = await self._rerank_with_llm_choice(
                image_bytes=image_bytes,
                candidates=candidates[:GLOBAL_CANDIDATE_LIMIT],
            )

        diagnostics = {
            "global": {
                **self._visual_diagnostics(matches),
                "fusion": "weighted_cosine",
                "aggregation": "view_best75_top3mean25_then_weighted_max_sum",
                "candidate_limit": global_limit,
                "stages": search_stages,
                "main_photos_only": main_photos_only,
            },
            "llm_rerank": {
                "applied": llm_selection is not None,
                "reason": (
                    "not_requested" if "llm" not in search_stages
                    else "selected_candidate" if llm_selection is not None
                    else "llm_unavailable"
                ),
                "selection": llm_selection,
            },
            "ocr_rerank": {
                **ocr,
                **ocr_diagnostics,
                "formula": "final = visual_score + ocr_bonus (src/ml/ocr_matching.py)",
            },
        }
        return candidates[:limit], diagnostics, query_views.crops

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
            "label_photo_area_threshold": LABEL_PHOTO_AREA_THRESHOLD,
            "label_photo_confidence_threshold": LABEL_PHOTO_CONFIDENCE_THRESHOLD,
        }

    @staticmethod
    def _normalize_search_stages(stages: int | str | list[str] | None) -> list[str]:
        """Список стадий из query-параметра. Неизвестные (в т.ч. удалённая "patches") игнорируются."""
        if stages is None or isinstance(stages, int):
            return ["global"]
        items = [stages] if isinstance(stages, str) else [str(item) for item in stages]
        requested = {part.strip().lower() for item in items for part in item.split(",") if part.strip()}
        unknown = requested - set(VALID_SEARCH_STAGES) - {"1", "2", "patches"}
        if unknown:
            logger.warning("Ignoring unknown search stages: %s", sorted(unknown))
        return [stage for stage in VALID_SEARCH_STAGES if stage == "global" or stage in requested]

    # --- OCR ----------------------------------------------------------------------

    async def _extract_ocr_text(self, query_views) -> dict[str, Any]:
        llm = self.connection_manager.llm
        source_view = ocr_source_view(query_views)
        if llm is None:
            return _ocr_result("llm_unavailable", None)
        image = query_views.images.get(source_view)
        if image is None:
            return _ocr_result("no_source_image", source_view)
        try:
            text = await llm.ocr_image_text(image_to_jpeg_bytes(image))
        except Exception:
            logger.exception("OCR extraction failed for source view %s", source_view)
            return _ocr_result("ocr_failed", source_view)

        normalized = normalize_match_text(text)
        logger.info(
            "OCR source=%s applied=%s raw_text=%r normalized_text=%r",
            source_view,
            bool(normalized),
            text,
            normalized,
        )
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
            (replace(candidate, score=candidate.score + scores[candidate.wine_id].bonus) for candidate in candidates),
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
            "OCR rerank applied for %s candidates. Top: %s",
            len(candidates),
            [
                (item["wine_id"][:8], item["visual_score"], item["ocr_bonus"], item["final_score"])
                for item in candidate_scores[:5]
            ],
        )
        return reranked, {
            "rerank_applied": True,
            "reason": "ok",
            "candidate_scores": candidate_scores,
            # весь пул до реранка — для офлайн-анализа формулы (scripts/ocr_lab)
            "pool": [
                {
                    "wine_id": candidate.wine_id,
                    "visual_score": round(candidate.score, 6),
                    "ocr_bonus": round(scores[candidate.wine_id].bonus, 6),
                }
                for candidate in candidates
            ],
        }

    # --- CatBoost ----------------------------------------------------------------

    async def _search_candidates_catboost(
        self,
        image_bytes: bytes,
        limit: int,
        views: list[str] | None = None,
        main_photos_only: bool = False,
    ) -> tuple[list[WineCandidate], dict[str, Any], dict]:
        query_views = self.photo_service.build_query_views(image_bytes)
        matches = await self.visual.run(
            query_views,
            allowed=views,
            query_filter=main_photos_filter() if main_photos_only else None,
        )
        feature_args = {
            "wine_view_matches": matches.wine_view_matches,
            "wine_slugs": matches.wine_slugs,
            "weights": matches.weights,
            "crops": query_views.crops,
        }
        base_rows = sorted(build_candidate_feature_rows(**feature_args), key=lambda row: row.baseline_score, reverse=True)
        candidate_ids = [row.wine_id for row in base_rows[:GLOBAL_CANDIDATE_LIMIT]]

        ocr = await self._extract_ocr_text(query_views)
        ocr_scores = await self.ocr_matcher.score(ocr["text"], candidate_ids)
        ocr_features = {wine_id: score.features for wine_id, score in ocr_scores.items()}
        selected = set(candidate_ids)
        feature_rows = [
            row
            for row in build_candidate_feature_rows(**feature_args, ocr_features=ocr_features)
            if row.wine_id in selected
        ]

        reranker = self._load_catboost_reranker()
        if reranker is not None and feature_rows:
            scores = reranker.predict_rows(feature_rows)
            fusion = "catboost"
        else:
            scores = [row.baseline_score for row in feature_rows]
            fusion = "weighted_cosine_fallback"
        ranked = sorted(zip(feature_rows, scores, strict=True), key=lambda item: item[1], reverse=True)

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
                view_match_counts={
                    view: len(view_matches) for view, view_matches in matches.wine_view_matches[row.wine_id].items()
                },
            )
            for row, score in ranked[:limit]
        ]
        diagnostics = {
            "global": {
                **self._visual_diagnostics(matches),
                "fusion": fusion,
                "main_photos_only": main_photos_only,
            },
            "catboost": {
                "model_path": str(CATBOOST_RERANKER_MODEL_PATH),
                "loaded": reranker is not None,
                "candidate_pool": len(feature_rows),
                "ocr_applied": ocr["applied"],
                "ocr_source_view": ocr["source_view"],
                "ocr_feature_candidates": len(ocr_features),
            },
        }
        return candidates, diagnostics, query_views.crops

    def _load_catboost_reranker(self) -> CatBoostWineReranker | None:
        if self._catboost_reranker is not None or self._catboost_load_failed:
            return self._catboost_reranker
        if not CATBOOST_RERANKER_MODEL_PATH.is_file():
            self._catboost_load_failed = True
            logger.warning("CatBoost reranker model not found: %s", CATBOOST_RERANKER_MODEL_PATH)
            return None
        try:
            self._catboost_reranker = CatBoostWineReranker.load(CATBOOST_RERANKER_MODEL_PATH)
        except Exception:
            self._catboost_load_failed = True
            logger.exception("Failed to load CatBoost reranker model: %s", CATBOOST_RERANKER_MODEL_PATH)
        return self._catboost_reranker

    # --- LLM-выбор (stage "llm") --------------------------------------------------

    async def _rerank_with_llm_choice(
        self,
        image_bytes: bytes,
        candidates: list[WineCandidate],
    ) -> tuple[list[WineCandidate], dict[str, Any] | None]:
        if not candidates or self.connection_manager.llm is None:
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

        result = {
            "selected_number": selection.selected_number,
            "selected_wine_id": selection.selected_wine_id,
            "confidence": selection.confidence,
            "reason": selection.reason,
        }
        selected_idx = selection.selected_number - 1
        if not 0 <= selected_idx < len(candidates):
            logger.warning("LLM selected out-of-range candidate number: %s", selection.selected_number)
            return candidates, {**result, "elapsed": perf_counter() - started, "valid": False}

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
            **result,
            "selected_wine_id": selected.wine_id,
            "model_selected_wine_id": selection.selected_wine_id,
            "elapsed": perf_counter() - started,
            "valid": True,
        }

    @staticmethod
    def _wine_to_llm_candidate(index: int, candidate: WineCandidate, wine: Wine | None) -> dict[str, Any]:
        grape_names = [link.grape.name for link in wine.grape_links if link.grape is not None] if wine else []
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

    # --- ответы -------------------------------------------------------------------

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
            price=float(wine.price) if wine.price is not None else None,
            stock=wine.stock,
            rating=float(wine.rating) if wine.rating is not None else None,
            description=wine.description,
            url=wine.source_url,
            source_url=wine.source_url,
            image_url=photos[0].url if photos else None,
            grapes=[link.grape.name for link in wine.grape_links if link.grape is not None],
            photos=photos,
        )


def _match_fields(candidate: WineCandidate) -> dict[str, Any]:
    """Поля кандидата для CompactSearchMatch / SearchMatchResponse."""
    return asdict(candidate)


def _ocr_result(reason: str, source_view: str | None, text: str = "", normalized: str = "") -> dict[str, Any]:
    return {
        "applied": bool(normalized),
        "reason": reason,
        "source_view": source_view,
        "text": text,
        "normalized_text": normalized,
    }


def _format_top_gap(candidates: list[WineCandidate]) -> str:
    if not candidates:
        return "no candidates"
    top1 = candidates[0]
    if len(candidates) < 2:
        return f"top1 wine_id={top1.wine_id} confidence={top1.score:.4f}; no top2"
    top2 = candidates[1]
    return (
        f"top1 wine_id={top1.wine_id} confidence={top1.score:.4f}; "
        f"top2 wine_id={top2.wine_id} confidence={top2.score:.4f}; "
        f"gap={top1.score - top2.score:.4f}"
    )
