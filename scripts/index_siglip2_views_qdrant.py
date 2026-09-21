from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.vision_features import extract_global_vector
from src.settings.settings import all_settings


DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_WINES_JSON = Path("data/db/wines.json")
DEFAULT_OUTPUT_DIR = Path("data/embeddings")
VIEWS = ("original", "bottle_crop", "label_crop")
POINT_NAMESPACE = uuid.UUID("92670dbe-559b-55c3-ae90-ac7c7b9e50bd")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewImage:
    path: Path
    wine_id: str
    photo_id: str
    view: str
    slug: str


@dataclass(frozen=True)
class EncodedPoint:
    id: str
    vector: np.ndarray
    payload: dict[str, str]
    path: str


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_slug_by_wine_id(wines_json: Path) -> dict[str, str]:
    rows = json.loads(wines_json.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for row in rows:
        wine_id = str(row["id"])
        slug = row.get("slug") or row.get("sku")
        if not slug:
            raise ValueError(f"Wine {wine_id} has no slug/sku")
        result[wine_id] = str(slug)
    return result


def iter_view_images(images_dir: Path, slug_by_wine_id: dict[str, str]) -> Iterable[ViewImage]:
    for wine_dir in sorted((p for p in images_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        wine_id = wine_dir.name
        slug = slug_by_wine_id.get(wine_id)
        if slug is None:
            logger.warning("Skipping image directory without wine row: %s", wine_dir)
            continue

        for photo_dir in sorted((p for p in wine_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            photo_id = photo_dir.name
            for view in VIEWS:
                path = photo_dir / f"{view}.jpg"
                if path.is_file():
                    yield ViewImage(
                        path=path,
                        wine_id=wine_id,
                        photo_id=photo_id,
                        view=view,
                        slug=slug,
                    )


def batched(items: Iterable[ViewImage], batch_size: int) -> Iterable[list[ViewImage]]:
    batch: list[ViewImage] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_images(batch: list[ViewImage]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in batch:
        with Image.open(item.path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    return images


def point_id(item: ViewImage) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{item.wine_id}/{item.photo_id}/{item.view}"))


def encode_batch(
    model,
    processor,
    device: torch.device,
    batch: list[ViewImage],
) -> list[EncodedPoint]:
    images = load_images(batch)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    synchronize_device(device)

    with torch.inference_mode():
        vectors = extract_global_vector(model, inputs)
        synchronize_device(device)

    array = vectors.detach().cpu().numpy().astype("float32")
    points: list[EncodedPoint] = []
    for item, vector in zip(batch, array, strict=True):
        points.append(
            EncodedPoint(
                id=point_id(item),
                vector=vector.astype("float32"),
                payload={
                    "wine_id": item.wine_id,
                    "photo_id": item.photo_id,
                    "view": item.view,
                    "slug": item.slug,
                },
                path=item.path.as_posix(),
            )
        )

    for image in images:
        image.close()
    return points


def count_by_view(items: Iterable[ViewImage]) -> dict[str, int]:
    counts = {view: 0 for view in VIEWS}
    for item in items:
        counts[item.view] += 1
    return counts


def collection_name(view: str, collection_encoder: str) -> str:
    return f"wine_{view}_{collection_encoder}"


def save_view_npz(
    output_dir: Path,
    view: str,
    points: list[EncodedPoint],
    vector_size: int,
    collection_encoder: str,
) -> Path:
    if not points:
        raise ValueError(f"Cannot save empty view: {view}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{view}.npz"
    tmp_path = output_dir / f"{view}.tmp.npz"

    vectors = np.stack([point.vector for point in points]).astype("float32")
    if vectors.shape[1] != vector_size:
        raise ValueError(f"Unexpected vector size for {view}: {vectors.shape[1]} != {vector_size}")

    point_ids = np.array([point.id for point in points], dtype=str)
    wine_ids = np.array([point.payload["wine_id"] for point in points], dtype=str)
    photo_ids = np.array([point.payload["photo_id"] for point in points], dtype=str)
    views = np.array([point.payload["view"] for point in points], dtype=str)
    slugs = np.array([point.payload["slug"] for point in points], dtype=str)
    paths = np.array([point.path for point in points], dtype=str)
    payloads_json = np.array(
        [json.dumps(point.payload, ensure_ascii=False, sort_keys=True) for point in points],
        dtype=str,
    )

    np.savez_compressed(
        tmp_path,
        vectors=vectors,
        point_ids=point_ids,
        wine_ids=wine_ids,
        photo_ids=photo_ids,
        views=views,
        slugs=slugs,
        paths=paths,
        payloads_json=payloads_json,
        collection=np.array(collection_name(view, collection_encoder)),
        distance=np.array("COSINE"),
    )
    tmp_path.replace(output_path)
    return output_path


def load_processor(encoder: str, model_dir: Path):
    if encoder == "dinov3":
        return AutoImageProcessor.from_pretrained(model_dir)
    return AutoProcessor.from_pretrained(model_dir)


def index_image_views(
    images_dir: Path,
    wines_json: Path,
    encoder: str,
    model_dir: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
    limit: int | None,
    collection_encoder: str,
) -> None:
    slug_by_wine_id = load_slug_by_wine_id(wines_json)
    all_items = list(iter_view_images(images_dir, slug_by_wine_id))
    if limit is not None:
        all_items = all_items[:limit]
    if not all_items:
        raise ValueError(f"No view images found under {images_dir}")

    expected = count_by_view(all_items)
    logger.info("Found view images: %s", expected)

    device = resolve_device(device_name)
    logger.info("Loading %s from %s on %s", encoder, model_dir, device)
    processor = load_processor(encoder, model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).to(device)
    model.eval()
    synchronize_device(device)

    first_points = encode_batch(model, processor, device, [all_items[0]])
    vector_size = int(first_points[0].vector.shape[0])
    logger.info("%s vector size: %s", encoder, vector_size)

    points_by_view: dict[str, list[EncodedPoint]] = {view: [] for view in VIEWS}
    totals = {view: 0 for view in VIEWS}
    start = time.perf_counter()
    for batch in batched(all_items, batch_size):
        encoded = encode_batch(model, processor, device, batch)
        for point in encoded:
            view = point.payload["view"]
            points_by_view[view].append(point)
            totals[view] += 1
        logger.info("Encoded batch. totals=%s elapsed=%.1fs", totals, time.perf_counter() - start)

    for view in VIEWS:
        if not points_by_view[view]:
            logger.info("Skipping %s: no vectors", collection_name(view, collection_encoder))
            (output_dir / f"{view}.npz").unlink(missing_ok=True)
            (output_dir / f"{view}.tmp.npz").unlink(missing_ok=True)
            continue
        output_path = save_view_npz(
            output_dir,
            view,
            points_by_view[view],
            vector_size=vector_size,
            collection_encoder=collection_encoder,
        )
        logger.info(
            "Saved %s vectors for %s to %s",
            len(points_by_view[view]),
            collection_name(view, collection_encoder),
            output_path,
        )

    logger.info("Completed %s embedding export. totals=%s", encoder, totals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export global image embeddings for original/bottle/label views to NPZ files.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--wines-json", type=Path, default=DEFAULT_WINES_JSON)
    parser.add_argument("--encoder", choices=["siglip2", "dinov3"], default="siglip2")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--collection-encoder",
        default=None,
        help=(
            "Collection name suffix for NPZ metadata. "
            "Default: selected encoder. Use 'siglip2' to overwrite current runtime collections."
        ),
    )
    return parser.parse_args()


def default_model_dir(encoder: str) -> Path:
    if encoder == "dinov3":
        return resolve_model_dir(Path(all_settings.dinov3.model_dir))
    return resolve_model_dir(Path(all_settings.embeddings.model_dir))


def resolve_model_dir(path: Path) -> Path:
    if path.exists():
        return path
    if path.is_absolute() and len(path.parts) >= 3 and path.parts[1] == "models":
        local_path = PROJECT_ROOT / "models" / Path(*path.parts[2:])
        if local_path.exists():
            return local_path
    return path


def default_device(encoder: str) -> str:
    if encoder == "dinov3":
        return all_settings.dinov3.device
    return all_settings.embeddings.device


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    model_dir = args.model_dir or default_model_dir(args.encoder)
    device = args.device or default_device(args.encoder)
    collection_encoder = args.collection_encoder or args.encoder
    index_image_views(
        images_dir=args.images_dir,
        wines_json=args.wines_json,
        encoder=args.encoder,
        model_dir=model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=device,
        limit=args.limit,
        collection_encoder=collection_encoder,
    )
