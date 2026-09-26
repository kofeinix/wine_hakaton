import argparse
import asyncio
import gzip
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from qdrant_client.http import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connections.qdrant import QdrantClient
from src.settings.settings import all_settings


DEFAULT_EMBEDDINGS_PATH = Path("/data/embeddings/wine_siglip2.jsonl.gz")
DEFAULT_EMBEDDINGS_DIR = Path("/data/embeddings")
DEFAULT_NPZ_FILES = (
    "original.npz",
    "bottle_crop.npz",
    "label_crop.npz",
)

logger = logging.getLogger(__name__)


def open_text_reader(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_embedding_rows(path: Path) -> Iterator[dict]:
    with open_text_reader(path) as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "photo_id" not in row or "vector" not in row:
                raise ValueError(f"Invalid embedding row at line {line_number}")
            yield row


def batched(rows: Iterator[dict], batch_size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def vector_size(path: Path) -> int:
    for row in read_embedding_rows(path):
        return len(row["vector"])
    raise ValueError(f"No embeddings found in {path}")


def point_id(value: object) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        pass
    return str(value)


def embedding_count(path: Path) -> int:
    return sum(1 for _ in read_embedding_rows(path))


def scalar_string(value: np.ndarray, fallback: str) -> str:
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return fallback


@dataclass(frozen=True)
class NpzEmbeddingData:
    collection_name: str
    vector_size: int
    is_multivector: bool
    datatype: qdrant_models.Datatype | None
    vectors: np.ndarray
    point_ids: list
    payloads: list[dict]


def read_npz_data(path: Path) -> NpzEmbeddingData:
    logger.info("Reading NPZ embeddings from %s", path)
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"]
    if vectors.ndim not in (2, 3):
        raise ValueError(f"{path}: vectors must be a 2D or 3D array")
    is_multivector = vectors.ndim == 3
    if not is_multivector and vectors.dtype != np.float32:
        vectors = vectors.astype("float32", copy=False)
    datatype = qdrant_models.Datatype.FLOAT16 if is_multivector and vectors.dtype == np.float16 else None
    if len(vectors) == 0:
        return NpzEmbeddingData(
            collection_name=scalar_string(data.get("collection", np.array(path.stem)), path.stem),
            vector_size=0,
            is_multivector=is_multivector,
            datatype=datatype,
            vectors=vectors,
            point_ids=[],
            payloads=[],
        )

    collection_name = scalar_string(data.get("collection", np.array(path.stem)), path.stem)
    logger.info(
        "Loaded NPZ %s: collection=%s shape=%s dtype=%s multivector=%s qdrant_datatype=%s",
        path,
        collection_name,
        vectors.shape,
        vectors.dtype,
        is_multivector,
        datatype,
    )
    point_ids = data["point_ids"].tolist()
    if "payloads_json" in data:
        payloads = [json.loads(value) for value in data["payloads_json"].tolist()]
    else:
        payloads = [
            {
                "wine_id": wine_id,
                "photo_id": photo_id,
                "view": view,
                "slug": slug,
            }
            for wine_id, photo_id, view, slug in zip(
                data["wine_ids"].tolist(),
                data["photo_ids"].tolist(),
                data["views"].tolist(),
                data["slugs"].tolist(),
                strict=True,
            )
        ]

    if len(point_ids) != len(vectors) or len(payloads) != len(vectors):
        raise ValueError(
            f"{path}: inconsistent rows: vectors={len(vectors)}, "
            f"point_ids={len(point_ids)}, payloads={len(payloads)}"
        )

    return NpzEmbeddingData(
        collection_name=collection_name,
        vector_size=int(vectors.shape[-1]),
        is_multivector=is_multivector,
        datatype=datatype,
        vectors=vectors,
        point_ids=point_ids,
        payloads=payloads,
    )


def read_npz_points(
    path: Path,
) -> tuple[str, int, bool, qdrant_models.Datatype | None, list[qdrant_models.PointStruct]]:
    data = read_npz_data(path)
    points = list(iter_npz_point_batches(data, batch_size=len(data.vectors)))[0] if len(data.vectors) else []
    return data.collection_name, data.vector_size, data.is_multivector, data.datatype, points


def iter_npz_point_batches(
    data: NpzEmbeddingData,
    batch_size: int,
) -> Iterator[list[qdrant_models.PointStruct]]:
    total = len(data.vectors)
    for offset in range(0, total, batch_size):
        end = min(offset + batch_size, total)
        logger.info(
            "Converting NPZ rows %s-%s/%s to Qdrant PointStruct for %s",
            offset + 1,
            end,
            total,
            data.collection_name,
        )
        yield [
            qdrant_models.PointStruct(
                id=str(point_id),
                vector=vector.tolist(),
                payload=payload,
            )
            for point_id, vector, payload in zip(
                data.point_ids[offset:end],
                data.vectors[offset:end],
                data.payloads[offset:end],
                strict=True,
            )
        ]


def batched_points(
    points: list[qdrant_models.PointStruct],
    batch_size: int,
) -> Iterator[list[qdrant_models.PointStruct]]:
    for offset in range(0, len(points), batch_size):
        yield points[offset : offset + batch_size]


async def init_collection(
    qdrant: QdrantClient,
    collection_name: str,
    vector_size: int,
    points: list[qdrant_models.PointStruct],
    batch_size: int,
    recreate: bool,
    indexing_threshold: int,
    is_multivector: bool = False,
    datatype: qdrant_models.Datatype | None = None,
) -> None:
    vector_params = qdrant_models.VectorParams(
        size=vector_size,
        distance=qdrant_models.Distance.COSINE,
        datatype=datatype,
        multivector_config=(
            qdrant_models.MultiVectorConfig(
                comparator=qdrant_models.MultiVectorComparator.MAX_SIM,
            )
            if is_multivector
            else None
        ),
    )
    if recreate:
        await qdrant.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    elif not await qdrant.client.collection_exists(collection_name):
        await qdrant.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    else:
        await qdrant.client.update_collection(
            collection_name=collection_name,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )

    total = 0
    for batch in batched_points(points, batch_size):
        await qdrant.client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        total += len(batch)
        logger.info("Uploaded %s/%s vectors to %s", total, len(points), collection_name)


async def init_collection_from_npz_data(
    qdrant: QdrantClient,
    data: NpzEmbeddingData,
    batch_size: int,
    multivector_batch_size: int,
    recreate: bool,
    indexing_threshold: int,
) -> None:
    await ensure_collection(
        qdrant=qdrant,
        collection_name=data.collection_name,
        vector_size=data.vector_size,
        recreate=recreate,
        indexing_threshold=indexing_threshold,
        is_multivector=data.is_multivector,
        datatype=data.datatype,
    )

    total = 0
    expected = len(data.vectors)
    effective_batch_size = min(batch_size, multivector_batch_size) if data.is_multivector else batch_size
    logger.info(
        "Using upload batch size %s for %s",
        effective_batch_size,
        data.collection_name,
    )
    for batch in iter_npz_point_batches(data, batch_size=effective_batch_size):
        await qdrant.client.upsert(
            collection_name=data.collection_name,
            points=batch,
            wait=True,
        )
        total += len(batch)
        logger.info("Uploaded %s/%s vectors to %s", total, expected, data.collection_name)


async def ensure_collection(
    qdrant: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
    indexing_threshold: int,
    is_multivector: bool = False,
    datatype: qdrant_models.Datatype | None = None,
) -> None:
    vector_params = qdrant_models.VectorParams(
        size=vector_size,
        distance=qdrant_models.Distance.COSINE,
        datatype=datatype,
        multivector_config=(
            qdrant_models.MultiVectorConfig(
                comparator=qdrant_models.MultiVectorComparator.MAX_SIM,
            )
            if is_multivector
            else None
        ),
    )
    if recreate:
        await qdrant.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    elif not await qdrant.client.collection_exists(collection_name):
        await qdrant.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )
    else:
        await qdrant.client.update_collection(
            collection_name=collection_name,
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )


async def init_qdrant(
    embeddings_path: Path,
    collection_name: str,
    batch_size: int,
    recreate: bool,
    indexing_threshold: int,
) -> None:
    qdrant = QdrantClient(all_settings.qdrant)
    await qdrant.connect()

    try:
        size = vector_size(embeddings_path)
        expected_total = embedding_count(embeddings_path)
        logger.info(
            "Initializing Qdrant collection %s from %s embeddings",
            collection_name,
            expected_total,
        )
        points = [
            qdrant_models.PointStruct(
                id=point_id(row["photo_id"]),
                vector=row["vector"],
                payload={},
            )
            for row in read_embedding_rows(embeddings_path)
        ]
        await init_collection(
            qdrant=qdrant,
            collection_name=collection_name,
            vector_size=size,
            points=points,
            batch_size=batch_size,
            recreate=recreate,
            indexing_threshold=indexing_threshold,
        )

        logger.info("Qdrant collection %s initialized with %s vectors", collection_name, expected_total)
    finally:
        await qdrant.close()


async def init_qdrant_from_npz_dir(
    embeddings_dir: Path,
    batch_size: int,
    multivector_batch_size: int,
    recreate: bool,
    indexing_threshold: int,
) -> None:
    qdrant = QdrantClient(all_settings.qdrant)
    await qdrant.connect()

    try:
        paths = [embeddings_dir / name for name in DEFAULT_NPZ_FILES if (embeddings_dir / name).is_file()]
        if not paths:
            raise FileNotFoundError(f"No NPZ embeddings found in {embeddings_dir}")

        for path in paths:
            logger.info("Preparing NPZ file for Qdrant init: %s", path)
            data = read_npz_data(path)
            if len(data.vectors) == 0:
                logger.info("Skipping empty NPZ embeddings file: %s", path)
                continue
            logger.info(
                "Initializing Qdrant collection %s from %s vectors in %s",
                data.collection_name,
                len(data.vectors),
                path,
            )
            await init_collection_from_npz_data(
                qdrant=qdrant,
                batch_size=batch_size,
                multivector_batch_size=multivector_batch_size,
                recreate=recreate,
                indexing_threshold=indexing_threshold,
                data=data,
            )
            logger.info("Qdrant collection %s initialized with %s vectors", data.collection_name, len(data.vectors))
    finally:
        await qdrant.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Qdrant with wine image embeddings.")
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--collection-name", default=all_settings.qdrant.collection_name)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--multivector-batch-size", type=int, default=8)
    parser.add_argument("--indexing-threshold", type=int, default=1)
    parser.add_argument("--no-recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.embeddings_dir is not None:
        asyncio.run(
            init_qdrant_from_npz_dir(
                embeddings_dir=args.embeddings_dir,
                batch_size=args.batch_size,
                multivector_batch_size=args.multivector_batch_size,
                recreate=not args.no_recreate,
                indexing_threshold=args.indexing_threshold,
            )
        )
    else:
        asyncio.run(
            init_qdrant(
                embeddings_path=args.embeddings_path,
                collection_name=args.collection_name,
                batch_size=args.batch_size,
                recreate=not args.no_recreate,
                indexing_threshold=args.indexing_threshold,
            )
        )
