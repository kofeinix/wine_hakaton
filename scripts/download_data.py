"""Скачать большие данные, которых нет в репозитории, из публичной папки Яндекс Диска.

- data/embeddings/siglip2_384/*.npz — эмбеддинги фото каталога (SigLIP2-384) для Qdrant (~200 МБ);
- data/photos.tar.gz     — фото вин для MinIO (~5,4 ГБ);
- data/photos_webp.tar.gz — главные фото в webp для показа (~75 МБ, необязательный: без него — jpg).

Уже скачанное пропускается, целостность проверяется по md5. Только стандартная библиотека Python:
запускается и в контейнере (сервис data-init в docker-compose.yml), и на хосте:
python3 scripts/download_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# публичная папка Яндекс Диска: siglip2_384_<view>.npz, photos.tar.gz, photos_webp.tar.gz
DATA_URL = "https://disk.yandex.ru/d/TW3su5DtKTNtfQ"

# файл на Диске -> путь в data/embeddings (scripts/index_siglip2_views_qdrant.py --output-dir data/embeddings/siglip2_384)
NPZ_FILES = {
    f"siglip2_384_{view}.npz": Path("siglip2_384") / f"{view}.npz" for view in ("original", "bottle_crop", "label_crop")
}
PHOTOS_FILE = "photos.tar.gz"
WEBP_FILE = "photos_webp.tar.gz"  # необязательный
API = "https://cloud-api.yandex.net/v1/disk/public/resources"


def api_get(endpoint: str, **params: str) -> dict:
    query = urllib.parse.urlencode({"public_key": DATA_URL, **params})
    with urllib.request.urlopen(f"{API}{endpoint}?{query}", timeout=60) as response:
        return json.load(response)


def remote_files() -> dict[str, dict]:
    """Файлы в папке: имя -> {size, md5}."""
    listing = api_get("", limit="100")
    return {item["name"]: item for item in listing["_embedded"]["items"] if item["type"] == "file"}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, meta: dict, target: Path) -> None:
    if target.is_file() and target.stat().st_size == meta["size"] and md5sum(target) == meta["md5"]:
        print(f"{target.name}: уже скачан", flush=True)
        return
    href = api_get("/download", path=f"/{name}")["href"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"{target.name}: скачиваю {meta['size'] >> 20} МБ...", flush=True)
    with urllib.request.urlopen(href, timeout=60) as response, partial.open("wb") as out:
        done, last = 0, 0.0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if time.monotonic() - last > 5:
                last = time.monotonic()
                print(f"  {target.name}: {done >> 20}/{meta['size'] >> 20} МБ ({done * 100 // meta['size']}%)", flush=True)
    if md5sum(partial) != meta["md5"]:
        partial.unlink()
        sys.exit(f"{target.name}: контрольная сумма не совпала — запустите скачивание ещё раз")
    partial.replace(target)
    print(f"{target.name}: готово", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать эмбеддинги и фото вин с Яндекс Диска.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--skip-photos", action="store_true", help="Не качать архив фото (без него не будет картинок вин)")
    args = parser.parse_args()

    embeddings_dir = args.data_dir / "embeddings"
    photos = args.data_dir / PHOTOS_FILE
    webp = args.data_dir / WEBP_FILE
    need_embeddings = not all((embeddings_dir / local).is_file() for local in NPZ_FILES.values())
    need_photos = not args.skip_photos and not photos.is_file()
    need_webp = not args.skip_photos and not webp.is_file()
    if not need_embeddings and not need_photos and not need_webp:
        print("Данные уже на месте:", args.data_dir)
        return

    try:
        files = remote_files()
    except OSError as exc:
        if not need_embeddings and not need_photos:
            # не хватает только необязательного webp — запуск не блокируем
            print(f"{WEBP_FILE}: не удалось открыть {DATA_URL} ({exc}) — главные фото будут в jpg", flush=True)
            return
        sys.exit(f"Не удалось открыть {DATA_URL}: {exc}. Скачайте файлы вручную (см. README).")

    wanted = [(name, embeddings_dir / local) for name, local in NPZ_FILES.items()] if need_embeddings else []
    if need_photos:
        wanted.append((PHOTOS_FILE, photos))
    missing = [name for name, _ in wanted if name not in files]
    if missing:
        sys.exit(f"В папке {DATA_URL} нет файлов: {', '.join(missing)}")
    for name, target in wanted:
        download(name, files[name], target)
    if need_webp:
        if WEBP_FILE in files:
            download(WEBP_FILE, files[WEBP_FILE], webp)
        else:
            print(f"{WEBP_FILE}: нет в папке — главные фото будут в jpg", flush=True)


if __name__ == "__main__":
    main()
