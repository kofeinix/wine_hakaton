from __future__ import annotations

import asyncio
import logging

from src.api.repositories.wine_repository import WineRepository
from src.connections.database.models import Wine
from src.ml.ocr_matching import (
    OcrCandidate,
    OcrCandidateScore,
    OcrCatalog,
    OcrQuery,
    score_candidates,
)
from src.ml.text_normalization import normalize_match_text

logger = logging.getLogger(__name__)


class OcrMatchService:
    """OCR-скоры кандидатов по сырому OCR-тексту."""

    def __init__(self, repository: WineRepository, sugar_variants: dict[str, list[str]]) -> None:
        self.repository = repository
        self.sugar_variants = sugar_variants
        self._catalog: OcrCatalog | None = None

    async def catalog(self) -> OcrCatalog:
        if self._catalog is None:
            wine_names, producer_names, grapes = await self.repository.load_ocr_vocabulary()
            raw_colors = await self.repository.load_color_aliases()
            color_variants = {
                normalize_match_text(color): [color, *aliases]
                for color, aliases in raw_colors.items()
                if normalize_match_text(color)
            }
            self._catalog = OcrCatalog.build(
                wine_names=wine_names,
                producer_names=producer_names,
                grapes=grapes,
                color_variants=color_variants,
                sugar_variants=self.sugar_variants,
            )
            logger.info(
                "OCR catalog loaded: wines=%s producers=%s grapes=%s",
                len(wine_names),
                len(self._catalog.producers),
                len(grapes),
            )
        return self._catalog

    async def score(self, raw_ocr_text: str, wine_ids: list[str]) -> dict[str, OcrCandidateScore]:
        """OCR-скор каждого вина из пула. Пустой OCR — пустой результат."""
        if not normalize_match_text(raw_ocr_text) or not wine_ids:
            return {}
        catalog = await self.catalog()
        wines = {str(wine.id): wine for wine in await self.repository.load_wines_by_ids(wine_ids, for_ocr=True)}
        candidates = [to_ocr_candidate(wine_id, wines.get(wine_id)) for wine_id in wine_ids]
        # сопоставление — чистый CPU (~50 мс): в потоке, чтобы не блокировать event loop
        scores = await asyncio.to_thread(_score, raw_ocr_text, catalog, candidates)
        return {score.wine_id: score for score in scores}


# заглушки вместо производителя: для OCR это «производитель не указан» — без бонуса и без штрафа
PLACEHOLDER_PRODUCERS = frozenset({normalize_match_text("Неизвестный производитель")})


def to_ocr_candidate(wine_id: str, wine: Wine | None) -> OcrCandidate:
    if wine is None:  # карточки нет: сравнивать нечего, бонусов не будет
        return OcrCandidate(wine_id, None, None, {}, None, None, None)
    grapes = {
        str(link.grape.id): (link.grape.name, *(alias.alias for alias in link.grape.aliases))
        for link in wine.grape_links
        if link.grape is not None
    }
    return OcrCandidate(
        wine_id=wine_id,
        name=wine.name,
        producer=(
            wine.producer.name
            if wine.producer is not None and normalize_match_text(wine.producer.name) not in PLACEHOLDER_PRODUCERS
            else None
        ),
        grapes=grapes,
        color=wine.color,
        sugar=wine.sugar,
        alcohol=wine.alcohol,
    )


def _score(raw_ocr_text: str, catalog: OcrCatalog, candidates: list[OcrCandidate]) -> list[OcrCandidateScore]:
    return score_candidates(OcrQuery.build(raw_ocr_text, catalog), catalog, candidates)
