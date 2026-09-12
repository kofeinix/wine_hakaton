import argparse
import asyncio
import gzip
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from qdrant_client.http import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connections.qdrant import QdrantClient
from src.settings.settings import all_settings


DEFAULT_EMBEDDINGS_PATH = Path("/data/embeddings/wine_main_siglip2.jsonl.gz")

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
            if "wine_id" not in row or "vector" not in row:
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
        if recreate:
            await qdrant.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=size,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold,
                ),
            )
        elif not await qdrant.client.collection_exists(collection_name):
            await qdrant.client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=size,
                    distance=qdrant_models.Distance.COSINE,
                ),
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
        for batch in batched(read_embedding_rows(embeddings_path), batch_size):
            points = [
                qdrant_models.PointStruct(
                    id=point_id(row["wine_id"]),
                    vector=row["vector"],
                    payload={
                        "wine_id": str(row["wine_id"]),
                        "main_photo_path": row.get("main_photo_path"),
                    },
                )
                for row in batch
            ]
            await qdrant.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            total += len(points)
            logger.info(
                "Uploaded %s/%s vectors to %s",
                total,
                expected_total,
                collection_name,
            )

        logger.info("Qdrant collection %s initialized with %s vectors", collection_name, total)
    finally:
        await qdrant.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Qdrant with wine image embeddings.")
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--collection-name", default=all_settings.qdrant.collection_name)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--indexing-threshold", type=int, default=1)
    parser.add_argument("--no-recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(
        init_qdrant(
            embeddings_path=args.embeddings_path,
            collection_name=args.collection_name,
            batch_size=args.batch_size,
            recreate=not args.no_recreate,
            indexing_threshold=args.indexing_threshold,
        )
    )
