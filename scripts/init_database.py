import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connections.database.postgres import DatabaseClient
from src.connections.database.models import Wine, Winery
from src.connections.database.schemas import WineCreate, WineryCreate
from src.settings.settings import all_settings


DEFAULT_EXPORT_DIR = Path("/data/export")
ASYNC_PG_MAX_QUERY_ARGUMENTS = 32767

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def wine_payload(row: dict) -> dict:
    return WineCreate.model_validate(row).model_dump()


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
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[column.name for column in model.__table__.primary_key.columns],
                set_=update_columns,
            )
        )


async def import_export(
    export_dir: Path,
    create_tables: bool,
) -> None:
    database = DatabaseClient(all_settings.database)
    await database.connect()

    try:
        if create_tables:
            await database.create_tables()

        wineries = [
            WineryCreate.model_validate(row).model_dump()
            for row in load_json(export_dir / "wineries.json")
        ]
        wines = [
            wine_payload(row)
            for row in load_json(export_dir / "wines.json")
        ]

        async with database.session() as session:
            await upsert_rows(session, Winery, wineries)
            await upsert_rows(session, Wine, wines)
            await session.commit()
        logger.info(
            "Imported %s wineries and %s wines.",
            len(wineries),
            len(wines),
        )
    finally:
        await database.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import wine export JSON files into Postgres.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--no-create-tables", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(
        import_export(
            export_dir=args.export_dir,
            create_tables=not args.no_create_tables,
        )
    )
