from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.connections.database.models import ColorAlias, Grape, Producer, Wine, WineGrape, WineImage
from src.connections.database.postgres import DatabaseClient

logger = logging.getLogger(__name__)


class WineRepository:
    def __init__(self, database: DatabaseClient) -> None:
        self.database = database


    async def get_wine(self, wine_id: str) -> Wine | None:
        try:
            id_ = UUID(wine_id)
        except ValueError:
            return None

        async with self.database.session() as session:
            return await session.scalar(self._with_wine_options(select(Wine)).where(Wine.id == id_))

    async def wine_scores_by_photo_ids(self, photo_scores: dict[str, float]) -> dict[str, float]:
        photo_ids = []
        for photo_id in photo_scores:
            try:
                photo_ids.append(UUID(photo_id))
            except ValueError:
                logger.debug("Skipping non-UUID photo id from Qdrant: %s", photo_id)

        if not photo_ids:
            return {}

        async with self.database.session() as session:
            rows = (
                await session.execute(
                    select(WineImage.id, WineImage.wine_id).where(WineImage.id.in_(photo_ids))
                )
            ).all()

        scores: dict[str, float] = {}
        for photo_id, wine_id in rows:
            score = photo_scores.get(str(photo_id))
            if score is None:
                continue
            wine_id_str = str(wine_id)
            scores[wine_id_str] = max(scores.get(wine_id_str, 0.0), score)
        return scores

    async def wine_ids_by_photo_ids(self, photo_ids: list[str]) -> dict[str, str]:
        ids = []
        for photo_id in dict.fromkeys(photo_ids):
            try:
                ids.append(UUID(photo_id))
            except ValueError:
                logger.debug("Skipping non-UUID photo id from Qdrant: %s", photo_id)

        if not ids:
            return {}

        async with self.database.session() as session:
            rows = (
                await session.execute(
                    select(WineImage.id, WineImage.wine_id).where(WineImage.id.in_(ids))
                )
            ).all()

        return {str(photo_id): str(wine_id) for photo_id, wine_id in rows}

    async def load_wines_by_ids(self, wine_ids: list[str], for_ocr: bool = False) -> list[Wine]:
        """Карточки вин. for_ocr=True — только то, что нужно OCR-матчингу (без фото и региона)."""
        ids = []
        for wine_id in dict.fromkeys(wine_ids):
            try:
                ids.append(UUID(wine_id))
            except ValueError:
                logger.debug("Skipping non-UUID wine id from search backend: %s", wine_id)

        if not ids:
            return []

        async with self.database.session() as session:
            return list(
                (
                    await session.scalars(
                        (self._with_ocr_options if for_ocr else self._with_wine_options)(select(Wine)).where(
                            Wine.id.in_(ids)
                        )
                    )
                ).all()
            )

    async def load_color_aliases(self) -> dict[str, list[str]]:
        async with self.database.session() as session:
            rows = (
                await session.execute(
                    select(ColorAlias.color, ColorAlias.alias).order_by(ColorAlias.color, ColorAlias.alias)
                )
            ).all()

        aliases: dict[str, list[str]] = {}
        for color, alias in rows:
            aliases.setdefault(color, []).append(alias)
        return aliases

    async def get_image(self, photo_id: str) -> WineImage | None:
        try:
            id_ = UUID(photo_id)
        except ValueError:
            return None
        async with self.database.session() as session:
            return await session.get(WineImage, id_)

    async def load_ocr_vocabulary(self) -> tuple[list[str], list[str], dict[str, list[str]]]:
        """Названия вин, производители и сорта (id -> название + синонимы) для OCR-матчинга."""
        async with self.database.session() as session:
            wine_names = list((await session.scalars(select(Wine.name))).all())
            producer_names = list((await session.scalars(select(Producer.name))).all())
            grapes = (await session.scalars(select(Grape).options(selectinload(Grape.aliases)))).all()
        return (
            wine_names,
            producer_names,
            {str(grape.id): [grape.name, *(alias.alias for alias in grape.aliases)] for grape in grapes},
        )

    @staticmethod
    def _with_wine_options(statement):
        return statement.options(
            selectinload(Wine.producer),
            selectinload(Wine.region),
            selectinload(Wine.grape_links).selectinload(WineGrape.grape).selectinload(Grape.aliases),
            selectinload(Wine.images),
        )

    @staticmethod
    def _with_ocr_options(statement):
        return statement.options(
            selectinload(Wine.producer),
            selectinload(Wine.grape_links).selectinload(WineGrape.grape).selectinload(Grape.aliases),
        )
