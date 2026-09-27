"""Загрузить эмбеддинги фото вин в Qdrant: по коллекции на ракурс (original, bottle_crop, label_crop).

Файлы <view>.npz в --embeddings-dir: vectors (N x D, float), point_ids, payloads_json
(или wine_ids/photo_ids/views/slugs), collection — имя коллекции.
Если коллекция уже содержит столько же точек, загрузка пропускается (--recreate — перезалить).
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client.http import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connections.qdrant import QdrantClient
from src.settings.settings import all_settings

DEFAULT_EMBEDDINGS_DIR = Path("/data/embeddings")
NPZ_FILES = ("original.npz", "bottle_crop.npz", "label_crop.npz")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NpzEmbeddings:
    collection_name: str
    vectors: np.ndarray
    point_ids: list
    payloads: list[dict]


def scalar_string(value: np.ndarray, fallback: str) -> str:
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return fallback


def read_npz(path: Path) -> NpzEmbeddings:
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"]
    if vectors.ndim != 2:
        raise ValueError(f"{path}: ожидается матрица векторов N x D, получено {vectors.shape}")
    vectors = vectors.astype("float32", copy=False)
    point_ids = data["point_ids"].tolist()
    if "payloads_json" in data:
        payloads = [json.loads(value) for value in data["payloads_json"].tolist()]
    else:
        payloads = [
            {"wine_id": wine_id, "photo_id": photo_id, "view": view, "slug": slug}
            for wine_id, photo_id, view, slug in zip(
                data["wine_ids"].tolist(),
                data["photo_ids"].tolist(),
                data["views"].tolist(),
                data["slugs"].tolist(),
                strict=True,
            )
        ]
    if not len(point_ids) == len(payloads) == len(vectors):
        raise ValueError(
            f"{path}: несогласованные строки: vectors={len(vectors)}, "
            f"point_ids={len(point_ids)}, payloads={len(payloads)}"
        )
    collection = scalar_string(data["collection"], path.stem) if "collection" in data else path.stem
    return NpzEmbeddings(collection, vectors, point_ids, payloads)


async def load_collection(qdrant: QdrantClient, data: NpzEmbeddings, batch_size: int, recreate: bool) -> None:
    client = qdrant.client
    name = data.collection_name
    expected = len(data.vectors)

    if not recreate and await client.collection_exists(name):
        existing = (await client.count(collection_name=name, exact=True)).count
        if existing == expected:
            logger.info("Коллекция %s уже загружена (%s точек) — пропускаю", name, existing)
            return
        logger.info("Коллекция %s: %s точек вместо %s — перезаливаю", name, existing, expected)

    await client.recreate_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(
            size=int(data.vectors.shape[1]),
            distance=qdrant_models.Distance.COSINE,
        ),
        optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1),
    )
    for offset in range(0, expected, batch_size):
        end = min(offset + batch_size, expected)
        await client.upsert(
            collection_name=name,
            points=[
                qdrant_models.PointStruct(id=str(point_id), vector=vector.tolist(), payload=payload)
                for point_id, vector, payload in zip(
                    data.point_ids[offset:end], data.vectors[offset:end], data.payloads[offset:end], strict=True
                )
            ],
            wait=True,
        )
        logger.info("%s: загружено %s/%s", name, end, expected)


async def init_qdrant(embeddings_dir: Path, batch_size: int, recreate: bool) -> None:
    paths = [embeddings_dir / name for name in NPZ_FILES if (embeddings_dir / name).is_file()]
    if not paths:
        raise FileNotFoundError(
            f"Нет эмбеддингов в {embeddings_dir}: скачайте их (см. README, шаг «Данные») в data/embeddings/"
        )
    qdrant = QdrantClient(all_settings.qdrant)
    await qdrant.connect()
    try:
        for path in paths:
            data = read_npz(path)
            logger.info("%s -> коллекция %s, %s векторов", path.name, data.collection_name, len(data.vectors))
            await load_collection(qdrant, data, batch_size, recreate)
    finally:
        await qdrant.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузить эмбеддинги фото вин в Qdrant.")
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true", help="Перезалить коллекции, даже если они уже загружены.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(init_qdrant(args.embeddings_dir, args.batch_size, args.recreate))
