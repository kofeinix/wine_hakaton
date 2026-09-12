from __future__ import annotations

import logging
from decimal import Decimal
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from src.connections.database.models import Grape, Producer, Region, Wine, WineGrape, WineImage
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
            return await session.get(
                Wine,
                id_,
                options=[
                    selectinload(Wine.producer),
                    selectinload(Wine.region),
                    selectinload(Wine.grape_links).selectinload(WineGrape.grape),
                    selectinload(Wine.images),
                ],
            )

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


    async def search_by_terms(self, terms: list[str], limit: int) -> list[Wine]:
        if not terms:
            return []

        conditions = []
        statement = (
            self._with_wine_options(select(Wine))
            .join(Wine.producer)
            .outerjoin(Wine.region)
            .outerjoin(Wine.grape_links)
            .outerjoin(WineGrape.grape)
        )

        for term in terms:
            pattern = f"%{term}%"
            conditions.extend(
                [
                    Wine.name.ilike(pattern),
                    Producer.name.ilike(pattern),
                    Region.name.ilike(pattern),
                    Region.country.ilike(pattern),
                    Grape.name.ilike(pattern),
                    Wine.color.ilike(pattern),
                    Wine.sugar.ilike(pattern),
                    Wine.description.ilike(pattern),
                ]
            )

        async with self.database.session() as session:
            return list(
                (
                    await session.scalars(
                        statement
                        .where(or_(*conditions))
                        .distinct()
                        .order_by(Wine.rating.desc().nullslast())
                        .limit(max(50, limit * 8))
                    )
                ).all()
            )

    async def load_wines_by_ids(self, wine_ids: list[str]) -> list[Wine]:
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
                        self._with_wine_options(select(Wine)).where(Wine.id.in_(ids))
                    )
                ).all()
            )

    @staticmethod
    def _with_wine_options(statement):
        return statement.options(
            selectinload(Wine.producer),
            selectinload(Wine.region),
            selectinload(Wine.grape_links).selectinload(WineGrape.grape),
            selectinload(Wine.images),
        )
