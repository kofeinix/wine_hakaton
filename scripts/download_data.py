"""Скачать большие данные, которых нет в репозитории, с Яндекс Диска.

- data/embeddings/*.npz  — эмбеддинги фото каталога для Qdrant (~190 МБ);
- data/photos.tar.gz     — фото вин для MinIO (~5 ГБ).

Уже скачанное пропускается. Только стандартная библиотека Python: запускается и в контейнере
(сервис data-init в docker-compose.yml), и на хосте: python3 scripts/download_data.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# публичные ссылки Яндекс Диска (папка или zip с тремя .npz; файл photos.tar.gz)
EMBEDDINGS_URL = ""
PHOTOS_URL = ""

NPZ_FILES = ("original.npz", "bottle_crop.npz", "label_crop.npz")
YANDEX_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="


def direct_url(public_url: str) -> str:
    """Публичная ссылка Яндекс Диска -> прямая ссылка на скачивание (папку Диск отдаёт zip-архивом)."""
    with urllib.request.urlopen(YANDEX_API + urllib.parse.quote(public_url, safe=""), timeout=60) as response:
        return json.load(response)["href"]


def download(public_url: str, target: Path) -> None:
    url = direct_url(public_url) if "disk.yandex" in public_url or "yadi.sk" in public_url else public_url
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if time.monotonic() - last > 2:
                last = time.monotonic()
                share = f" ({done * 100 // total}%)" if total else ""
                print(f"  {target.name}: {done >> 20} МБ{share}", flush=True)
    partial.replace(target)
    print(f"  {target.name}: готово, {target.stat().st_size >> 20} МБ", flush=True)


def ensure_embeddings(data_dir: Path) -> None:
    target_dir = data_dir / "embeddings"
    if all((target_dir / name).is_file() for name in NPZ_FILES):
        print("Эмбеддинги уже на месте:", target_dir)
        return
    if not EMBEDDINGS_URL:
        sys.exit("Не задана ссылка EMBEDDINGS_URL в scripts/download_data.py")
    print("Скачиваю эмбеддинги для Qdrant...")
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        archive = Path(tmp) / "embeddings.download"
        download(EMBEDDINGS_URL, archive)
        target_dir.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    name = Path(member).name
                    if name in NPZ_FILES:  # берём .npz из любой вложенной папки архива
                        with zf.open(member) as src, (target_dir / name).open("wb") as dst:
                            shutil.copyfileobj(src, dst)
        missing = [name for name in NPZ_FILES if not (target_dir / name).is_file()]
        if missing:
            sys.exit(f"В скачанном архиве нет файлов: {', '.join(missing)}")
    print("Эмбеддинги готовы:", target_dir)


def ensure_photos(data_dir: Path) -> None:
    target = data_dir / "photos.tar.gz"
    if target.is_file():
        print("Архив фото уже на месте:", target)
        return
    if not PHOTOS_URL:
        sys.exit("Не задана ссылка PHOTOS_URL в scripts/download_data.py")
    print("Скачиваю архив фото для MinIO (~5 ГБ, это может занять время)...")
    download(PHOTOS_URL, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать эмбеддинги и фото вин с Яндекс Диска.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--skip-photos", action="store_true", help="Не качать архив фото (без него не будет картинок вин)")
    args = parser.parse_args()
    ensure_embeddings(args.data_dir)
    if not args.skip_photos:
        ensure_photos(args.data_dir)


if __name__ == "__main__":
    main()
