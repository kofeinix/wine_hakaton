#!/usr/bin/env python3
"""Собрать архив фото для MinIO: data/photos.tar.gz.

Содержимое — ровно фото из data/db/wine_images.json: файл `source_path` кладётся в архив
под именем `minio_path` (<wine_id>/<photo_dir>.jpg). Сервис minio-init в docker compose
распаковывает архив в bucket, поэтому имена объектов совпадают с wine_images в БД.

  uv run python scripts/build_photos_archive.py
  uv run python scripts/build_photos_archive.py --webp   # только webp главных фото → data/photos_webp.tar.gz

webp главных фото (scripts/download_main_webp.py) лежат рядом с main/original.jpg как
main/original.webp и идут отдельным маленьким архивом под именем `webp_minio_path`: основной
архив на ~5 ГБ из-за них пересобирать не нужно.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINE_IMAGES = PROJECT_ROOT / "data" / "db" / "wine_images.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "photos.tar.gz"
DEFAULT_WEBP_OUTPUT = PROJECT_ROOT / "data" / "photos_webp.tar.gz"

logger = logging.getLogger(__name__)


def build_archive(
    wine_images: Path,
    output: Path,
    include_generated: bool = True,
    limit_wines: int | None = None,
    webp: bool = False,
) -> tuple[int, list[str]]:
    rows = json.loads(wine_images.read_text(encoding="utf-8"))
    if webp:
        # webp лежит рядом с original.jpg; в bucket — под webp_minio_path
        rows = [
            {
                **row,
                "source_path": str(Path(row["source_path"]).with_name("original.webp")),
                "minio_path": row["webp_minio_path"],
            }
            for row in rows
            if row.get("webp_minio_path")
        ]
    if not include_generated:
        rows = [row for row in rows if not row.get("is_generated")]
    if limit_wines is not None:
        wines = sorted({row["wine_id"] for row in rows})[:limit_wines]
        rows = [row for row in rows if row["wine_id"] in set(wines)]
    missing: list[str] = []
    added = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f"{output.name}.tmp")
    try:
        # JPEG почти не сжимается — минимальный уровень gzip, чтобы сборка шла быстро
        with tarfile.open(tmp_output, "w:gz", compresslevel=1) as archive:
            for row in rows:
                source = PROJECT_ROOT / row["source_path"]
                if not source.is_file():
                    missing.append(row["source_path"])
                    continue
                info = archive.gettarinfo(str(source), arcname=row["minio_path"])
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source.open("rb") as file_obj:
                    archive.addfile(info, file_obj)
                added += 1
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise
    return added, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wine-images", type=Path, default=DEFAULT_WINE_IMAGES)
    parser.add_argument("--output", type=Path, default=None, help="по умолчанию data/photos.tar.gz или data/photos_webp.tar.gz")
    parser.add_argument("--webp", action="store_true", help="только webp главных фото — отдельный архив")
    parser.add_argument("--no-generated", action="store_true", help="без сгенерированных flux-фото (легче в ~4 раза)")
    parser.add_argument("--limit-wines", type=int, default=None, help="только первые N вин (для проверки)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.output = args.output or (DEFAULT_WEBP_OUTPUT if args.webp else DEFAULT_OUTPUT)
    added, missing = build_archive(
        args.wine_images,
        args.output,
        include_generated=not args.no_generated,
        limit_wines=args.limit_wines,
        webp=args.webp,
    )
    size_mb = args.output.stat().st_size / 1e6
    logger.info("Wrote %s photos to %s (%.0f MB)", added, args.output, size_mb)
    if missing:
        logger.warning("%s photos from %s not found on disk, e.g. %s", len(missing), args.wine_images, missing[:3])
