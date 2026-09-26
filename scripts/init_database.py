import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import delete, select, tuple_
from sqlalchemy.dialects.postgresql import insert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connections.database.models import ColorAlias, Grape, GrapeAlias, Producer, Region, Wine, WineGrape, WineImage
from src.connections.database.postgres import DatabaseClient
from src.connections.database.schemas import (
    ColorAliasCreate,
    GrapeAliasCreate,
    GrapeCreate,
    ProducerCreate,
    RegionCreate,
    WineCreate,
    WineGrapeCreate,
    WineImageCreate,
)
from src.settings.settings import all_settings


DEFAULT_DB_DIR = Path("/data/db")
ASYNC_PG_MAX_QUERY_ARGUMENTS = 32767

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def upsert_rows(session, model, rows: list[dict]) -> None:
    if not rows:
        return

    column_count = len(model.__table__.columns)
    batch_size = max(1, ASYNC_PG_MAX_QUERY_ARGUMENTS // column_count)

    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        statement = insert(model).values(batch)
        update_columns = {
            column.name: statement.excluded[column.name]
            for column in model.__table__.columns
            if not column.primary_key
        }
        if not update_columns:
            await session.execute(
                statement.on_conflict_do_nothing(
                    index_elements=[column.name for column in model.__table__.primary_key.columns],
                )
            )
            continue

        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[column.name for column in model.__table__.primary_key.columns],
                set_=update_columns,
            )
        )


async def prune_rows(session, model, rows: list[dict]) -> int:
    """Удалить строки таблицы, первичного ключа которых нет в rows."""
    pk_columns = list(model.__table__.primary_key.columns)
    keep = {tuple(str(row[column.name]) for column in pk_columns) for row in rows}
    existing = (await session.execute(select(*pk_columns))).all()
    stale = [tuple(pk) for pk in existing if tuple(str(value) for value in pk) not in keep]
    batch_size = max(1, ASYNC_PG_MAX_QUERY_ARGUMENTS // len(pk_columns))
    for offset in range(0, len(stale), batch_size):
        batch = stale[offset : offset + batch_size]
        await session.execute(delete(model).where(tuple_(*pk_columns).in_(batch)))
    return len(stale)


async def import_db(db_dir: Path, create_tables: bool, drop_existing: bool, prune: bool = False) -> None:
    database = DatabaseClient(all_settings.database)
    await database.connect()

    try:
        if drop_existing:
            await database.drop_tables()
        if create_tables:
            await database.create_tables()

        producers = [
            ProducerCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "producers.json")
        ]
        regions = [
            RegionCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "regions.json")
        ]
        grapes = [
            GrapeCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "grapes.json")
        ]
        grape_aliases = [
            GrapeAliasCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "grape_aliases.json")
        ]
        color_aliases = [
            ColorAliasCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "color_aliases.json")
        ]
        wines = [
            WineCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "wines.json")
        ]
        wine_grapes = [
            WineGrapeCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "wine_grapes.json")
        ]
        wine_images = [
            WineImageCreate.model_validate(row).model_dump()
            for row in load_json(db_dir / "wine_images.json")
        ]

        async with database.session() as session:
            await upsert_rows(session, Producer, producers)
            await upsert_rows(session, Region, regions)
            await upsert_rows(session, Grape, grapes)
            await upsert_rows(session, GrapeAlias, grape_aliases)
            await upsert_rows(session, ColorAlias, color_aliases)
            await upsert_rows(session, Wine, wines)
            await upsert_rows(session, WineGrape, wine_grapes)
            await upsert_rows(session, WineImage, wine_images)
            if prune:
                # от дочерних таблиц к родительским; таблицы пользователей не трогаем
                for model, rows in (
                    (WineImage, wine_images),
                    (WineGrape, wine_grapes),
                    (GrapeAlias, grape_aliases),
                    (ColorAlias, color_aliases),
                    (Wine, wines),
                    (Grape, grapes),
                    (Region, regions),
                    (Producer, producers),
                ):
                    removed = await prune_rows(session, model, rows)
                    if removed:
                        logger.info("Pruned %s stale rows from %s", removed, model.__tablename__)
            await session.commit()

        logger.info(
            (
                "Imported %s producers, %s regions, %s grapes, %s grape_aliases, %s color_aliases, "
                "%s wines, %s wine_grapes, %s wine_images."
            ),
            len(producers),
            len(regions),
            len(grapes),
            len(grape_aliases),
            len(color_aliases),
            len(wines),
            len(wine_grapes),
            len(wine_images),
        )
    finally:
        await database.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import normalized wine JSON files into Postgres.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--no-create-tables", action="store_true")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Удалить ВСЕ таблицы (включая пользователей, историю, избранное) перед импортом.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Удалить из таблиц каталога строки, которых нет в JSON (пользовательские данные не трогаются).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(
        import_db(
            db_dir=args.db_dir,
            create_tables=not args.no_create_tables,
            drop_existing=args.drop_existing,
            prune=args.prune,
        )
    )
