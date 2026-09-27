#!/usr/bin/env python3
"""Главные фото вин в webp с vino-svoe.ru → data/images_extended/<wine_id>/main/original.webp.

Сайт отдаёт главную картинку вина в webp с прозрачным фоном; наш main/original.jpg когда-то
был сделан из неё. Ссылка уже есть в data/db/wine_images.json (source_url главного фото); для вин
без ссылки её берём из API страницы вина (ответы кешируются в data/site_cache, как в
enrich_wines_from_site.py).

Аккуратно с сайтом: не больше одного запроса в --delay секунд (по умолчанию 2), уже скачанное
пропускается (скрипт можно перезапускать), на 403/429/503 — остановка.

В конце в wine_images.json главным фото с webp проставляется webp_minio_path (<wine_id>/main.webp):
API отдаёт webp вместо jpg, а build_photos_archive.py --webp собирает их в архив для MinIO.

    uv run python scripts/download_main_webp.py
    uv run python scripts/download_main_webp.py --limit 20     # проба
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from enrich_wines_from_site import DEFAULT_CACHE_DIR, STOP_STATUSES, StopScraping, fetch_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data" / "db"
IMAGES_DIR = PROJECT_ROOT / "data" / "images_extended"
RESIZE_URL = "https://api.vino-svoe.ru/v1/img/str-api/1200/1200/resize{path}"  # так фото отдаёт сам сайт
USER_AGENT = "WineHakaton main photo import (one request per 2s)"

logger = logging.getLogger("download_main_webp")


class Throttle:
    """Не чаще одного запроса к сайту в delay секунд — на весь скрипт, какой бы ни был запрос."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.last = 0.0

    def wait(self) -> None:
        pause = self.last + self.delay - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self.last = time.monotonic()


def webp_target(wine_id: str) -> Path:
    return IMAGES_DIR / wine_id / "main" / "original.webp"


def valid_webp(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            return image.format == "WEBP"
    except Exception:
        return False


def download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in STOP_STATUSES:
            raise StopScraping(f"HTTP {exc.code} for {url}") from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать главные фото вин в webp")
    parser.add_argument("--delay", type=float, default=2.0, help="Секунд между запросами к сайту")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None, help="Обработать не больше N вин (проба)")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    images_path = DB_DIR / "wine_images.json"
    rows = json.loads(images_path.read_text(encoding="utf-8"))
    wines = {wine["id"]: wine for wine in json.loads((DB_DIR / "wines.json").read_text(encoding="utf-8"))}
    mains = [row for row in rows if row["is_main"]]
    todo = [row for row in mains if not webp_target(row["wine_id"]).is_file()]
    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("главных фото: %s, уже есть webp: %s, к загрузке: %s", len(mains), len(mains) - len(todo) if args.limit is None else "—", len(todo))

    throttle = Throttle(args.delay)
    stats = {"downloaded": 0, "no_image": 0, "failed": 0}
    started = time.monotonic()
    try:
        for number, row in enumerate(todo, start=1):
            wine_id = row["wine_id"]
            url = row.get("source_url") or ""
            if not url.endswith(".webp"):
                # ссылки нет в каталоге — берём из API страницы вина (ответ кешируется)
                slug = (wines.get(wine_id, {}).get("source_url") or "").rstrip("/").rsplit("/", 1)[-1]
                cached = (args.cache_dir / f"{slug}.json").exists()
                if not cached:
                    throttle.wait()
                payload, _ = fetch_payload(slug, args.cache_dir, args.timeout) if slug else (None, False)
                image_path = ((payload or {}).get("image") or {}).get("url") or ""
                if not image_path.endswith(".webp"):
                    stats["no_image"] += 1
                    logger.info("[%s/%s] %s: на сайте нет webp", number, len(todo), slug or wine_id)
                    continue
                url = RESIZE_URL.format(path=image_path)
            throttle.wait()
            try:
                data = download(url, args.timeout)
            except urllib.error.URLError as exc:
                stats["failed"] += 1
                logger.warning("[%s/%s] %s: %s", number, len(todo), wine_id, exc)
                continue
            if not valid_webp(data):
                stats["failed"] += 1
                logger.warning("[%s/%s] %s: ответ не webp (%s байт)", number, len(todo), wine_id, len(data))
                continue
            target = webp_target(wine_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".webp.part")
            partial.write_bytes(data)
            partial.replace(target)
            stats["downloaded"] += 1
            if number % 50 == 0 or number == len(todo):
                elapsed = time.monotonic() - started
                logger.info(
                    "[%s/%s] скачано %s, осталось ~%s мин",
                    number,
                    len(todo),
                    stats["downloaded"],
                    round(elapsed / number * (len(todo) - number) / 60),
                )
    except StopScraping as exc:
        logger.error("сайт просит притормозить (%s) — остановились; перезапустите позже", exc)
    except KeyboardInterrupt:
        logger.info("прервано — уже скачанное сохранено")

    # webp_minio_path — только тем главным фото, у которых файл реально есть
    marked = 0
    for row in rows:
        if row["is_main"] and webp_target(row["wine_id"]).is_file():
            row["webp_minio_path"] = f"{row['wine_id']}/main.webp"
            marked += 1
        else:
            row.pop("webp_minio_path", None)
    images_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("итог: %s; webp есть у %s из %s вин (wine_images.json обновлён)", stats, marked, len(mains))


if __name__ == "__main__":
    main()
