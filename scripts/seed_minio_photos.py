#!/usr/bin/env python3
"""Загрузить фото вин из data/images_extended в MinIO и (опционально) синхронизировать wine_images.

Источник: data/images_extended/<wine_id>/<photo_dir>/original.jpg: настоящие фото (main,
yandex_*, vivino_*) и сгенерированные сцены flux_* (is_generated — фронт показывает их по кнопке).
Объект в бакете: <wine_id>/<photo_dir>.jpg — так его ищет API (wine_images.minio_path).

  # загрузить (уже загруженные пропускаются)
  uv run python scripts/seed_minio_photos.py
  # показать, как изменится wine_images, ничего не меняя
  uv run python scripts/seed_minio_photos.py --skip-upload --sync-db --dry-run
  # привести wine_images в БД и data/db/wine_images.json к загруженным фото,
  # удалить из бакета объекты, на которые не ссылается ни одно фото
  uv run python scripts/seed_minio_photos.py --skip-upload --sync-db --write-json --prune-minio

Для запуска с хоста: MINIO__HOST=localhost DATABASE__HOST=localhost.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import io
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, select  # noqa: E402

from src.connections.database.models import Wine, WineImage  # noqa: E402
from src.connections.database.postgres import DatabaseClient  # noqa: E402
from src.connections.minio import MinioClient  # noqa: E402
from src.settings.settings import all_settings  # noqa: E402

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "images_extended"
DEFAULT_PATTERNS = ["main", "yandex_*", "vivino_*", "flux_*"]
WINE_IMAGES_JSON = PROJECT_ROOT / "data" / "db" / "wine_images.json"
IMAGE_ID_NAMESPACE = uuid.UUID("5b0e8f3c-6f1d-4d1a-9f35-1f6a4b1c2d3e")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Photo:
    wine_id: str
    photo_dir: str
    path: Path

    @property
    def object_name(self) -> str:
        return f"{self.wine_id}/{self.photo_dir}.jpg"

    @property
    def is_generated(self) -> bool:
        return self.photo_dir.startswith("flux")


def collect_photos(images_dir: Path, patterns: list[str]) -> list[Photo]:
    photos: list[Photo] = []
    for wine_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for photo_dir in sorted(p for p in wine_dir.iterdir() if p.is_dir()):
            if not any(fnmatch.fnmatchcase(photo_dir.name, pat) for pat in patterns):
                continue
            source = photo_dir / "original.jpg"
            if source.is_file():
                photos.append(Photo(wine_dir.name, photo_dir.name, source))
            elif others := sorted(p.name for p in photo_dir.glob("original.*")):
                # весь пайплайн (кропы, Qdrant, MinIO) читает только original.jpg
                logger.warning("%s/%s: no original.jpg (found %s) — convert to JPEG", wine_dir.name, photo_dir.name, others)
    return photos


async def upload(minio: MinioClient, photos: list[Photo], concurrency: int, overwrite: bool) -> None:
    existing = set() if overwrite else {item.object_name for item in await minio.list_files("")}
    todo = [photo for photo in photos if photo.object_name not in existing]
    logger.info("photos: %s, already in bucket: %s, to upload: %s", len(photos), len(photos) - len(todo), len(todo))
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def put(photo: Photo) -> None:
        nonlocal done
        data = photo.path.read_bytes()
        async with semaphore:
            await minio.storage.put_object(
                bucket_name=minio.bucket,
                object_name=photo.object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type="image/jpeg",
            )
        done += 1
        if done % 500 == 0:
            logger.info("uploaded %s/%s", done, len(todo))

    await asyncio.gather(*(put(photo) for photo in todo))
    logger.info("upload finished: %s files", len(todo))


async def prune_minio(minio: MinioClient, photos: list[Photo], dry_run: bool) -> None:
    """Удалить из бакета объекты, которых нет среди фото."""
    wanted = {photo.object_name for photo in photos}
    extra = [item.object_name for item in await minio.list_files("") if item.object_name not in wanted]
    logger.info("MinIO: %s objects not backed by photos%s", len(extra), " (dry run)" if dry_run else "")
    if not dry_run:
        for object_name in extra:
            await minio.storage.remove_object(minio.bucket, object_name)


def image_id(photo: Photo) -> uuid.UUID:
    return uuid.uuid5(IMAGE_ID_NAMESPACE, photo.object_name)


async def sync_db(database: DatabaseClient, photos: list[Photo], dry_run: bool) -> list[dict]:
    """wine_images := фото из images_extended (для вин, которые есть в каталоге).

    Существующие строки с тем же minio_path сохраняют id и source_url; строки, для которых
    фото больше нет, удаляются; новые — добавляются. Возвращает итоговые строки (для JSON).
    """
    await database.create_tables()  # таблицы/колонки (is_generated), как при старте API
    async with database.session() as session:
        wine_ids = {str(w) for w in (await session.scalars(select(Wine.id))).all()}
        current = {row.minio_path: row for row in (await session.scalars(select(WineImage))).all()}
        wanted = {p.object_name: p for p in photos if p.wine_id in wine_ids}
        skipped = len(photos) - len(wanted)
        to_delete = [path for path in current if path not in wanted]
        to_add = [p for path, p in wanted.items() if path not in current]
        logger.info(
            "wine_images: now %s rows; keep %s, delete %s (no photo), add %s; skipped %s photos of wines not in DB",
            len(current), len(current) - len(to_delete), len(to_delete), len(to_add), skipped,
        )
        if not dry_run:
            if to_delete:
                await session.execute(delete(WineImage).where(WineImage.minio_path.in_(to_delete)))
            session.add_all(
                WineImage(
                    id=image_id(p),
                    wine_id=uuid.UUID(p.wine_id),
                    source_url=None,
                    is_main=p.photo_dir == "main",
                    is_generated=p.is_generated,
                    minio_path=p.object_name,
                )
                for p in to_add
            )
            await session.commit()
    rows = []
    for path, photo in sorted(wanted.items()):
        row = current.get(path)
        rows.append(
            {
                "id": str(row.id) if row else str(image_id(photo)),
                "wine_id": photo.wine_id,
                "source_url": row.source_url if row else None,
                "source_path": str(photo.path.relative_to(PROJECT_ROOT)),
                "is_main": photo.photo_dir == "main",
                "is_generated": photo.is_generated,
                "minio_path": path,
            }
        )
    return rows


async def main(args: argparse.Namespace) -> None:
    photos = collect_photos(args.images_dir, args.photo_pattern)
    by_kind: dict[str, int] = {}
    for photo in photos:
        kind = photo.photo_dir.split("_")[0]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    logger.info("found %s photos in %s: %s", len(photos), args.images_dir, by_kind)

    if not args.skip_upload or args.prune_minio:
        minio = MinioClient(all_settings.minio)
        await minio.start()
        try:
            if not args.skip_upload:
                await upload(minio, photos, args.concurrency, args.overwrite)
            if args.prune_minio:
                await prune_minio(minio, photos, args.dry_run)
        finally:
            await minio.stop()

    if args.sync_db:
        database = DatabaseClient(all_settings.database)
        await database.connect()
        try:
            rows = await sync_db(database, photos, args.dry_run)
        finally:
            await database.close()
        if args.write_json and not args.dry_run:
            WINE_IMAGES_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("wrote %s rows to %s", len(rows), WINE_IMAGES_JSON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--photo-pattern", nargs="+", default=DEFAULT_PATTERNS)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="перезалить даже уже загруженные")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--sync-db", action="store_true", help="привести таблицу wine_images к фото")
    parser.add_argument("--dry-run", action="store_true", help="с --sync-db/--prune-minio: только показать изменения")
    parser.add_argument("--write-json", action="store_true", help="с --sync-db: обновить data/db/wine_images.json")
    parser.add_argument("--prune-minio", action="store_true", help="удалить из бакета объекты без фото")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(parse_args()))
