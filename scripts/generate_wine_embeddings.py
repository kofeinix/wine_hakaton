import argparse
import gzip
import json
import logging
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TextIO

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.settings.settings import all_settings


DEFAULT_PHOTOS_ARCHIVE = Path("data/photos.tar.gz")
DEFAULT_IMAGES_JSON = Path("data/db/wine_images.json")
DEFAULT_OUTPUT = Path("data/embeddings/wine_siglip2.jsonl.gz")

logger = logging.getLogger(__name__)


@dataclass
class StageTimer:
    values: dict[str, float]

    @classmethod
    def empty(cls) -> "StageTimer":
        return cls(values=defaultdict(float))

    def add(self, name: str, seconds: float) -> None:
        self.values[name] += seconds

    def format(self) -> str:
        total = sum(self.values.values())
        parts = [f"{name}={seconds:.2f}s" for name, seconds in sorted(self.values.items())]
        parts.append(f"total={total:.2f}s")
        return ", ".join(parts)


@dataclass(frozen=True)
class WinePhoto:
    photo_id: str
    wine_id: str
    archive_path: str


@dataclass(frozen=True)
class PhotoImage:
    photo: WinePhoto
    image: Image.Image


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


def open_text_writer(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz") or path.name.endswith(".gz.tmp"):
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def load_photo_index(images_json: Path) -> dict[str, tuple[str, str]]:
    with images_json.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return {
        row["minio_path"].strip("/"): (row["id"], row["wine_id"])
        for row in rows
        if row.get("minio_path") and row.get("id") and row.get("wine_id")
    }


def photo_from_member(
    member: tarfile.TarInfo,
    photo_index: dict[str, tuple[str, str]],
) -> WinePhoto | None:
    if not member.isfile():
        return None
    path = Path(member.name)
    if len(path.parts) < 2:
        return None
    if path.suffix.lower() not in {".jpg", ".jpeg", ".webp"}:
        return None

    photo_key = path.as_posix().strip("/")
    photo_row = photo_index.get(photo_key)
    if photo_row is None:
        return None
    photo_id, wine_id = photo_row
    return WinePhoto(
        photo_id=photo_id,
        wine_id=wine_id,
        archive_path=member.name,
    )


def load_image(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Image.Image:
    file_obj = archive.extractfile(member)
    if file_obj is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with Image.open(BytesIO(file_obj.read())) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def extract_image_features(model, inputs) -> torch.Tensor:
    outputs = model.get_image_features(**inputs)
    if isinstance(outputs, torch.Tensor):
        embeddings = outputs
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embeddings = outputs.pooler_output
    elif hasattr(outputs, "image_embeds"):
        embeddings = outputs.image_embeds
    else:
        embeddings = outputs.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(embeddings, dim=-1)


def write_batch(
    writer: TextIO,
    model,
    processor,
    device: torch.device,
    batch: list[PhotoImage],
    profile: bool,
) -> StageTimer:
    batch_timer = StageTimer.empty()

    stage_start = time.perf_counter()
    inputs = processor(images=[item.image for item in batch], return_tensors="pt")
    batch_timer.add("processor", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    synchronize_device(device)
    batch_timer.add("to_device", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    with torch.inference_mode():
        embeddings_tensor = extract_image_features(model, inputs)
        synchronize_device(device)
    batch_timer.add("inference", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    embeddings = embeddings_tensor.cpu().numpy().astype("float32")
    batch_timer.add("to_cpu_numpy", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    for item, vector in zip(batch, embeddings, strict=True):
        writer.write(
            json.dumps(
                {
                    "photo_id": item.photo.photo_id,
                    "vector": vector.tolist(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    batch_timer.add("json_gzip_write", time.perf_counter() - stage_start)

    for item in batch:
        item.image.close()

    if profile:
        logger.info("Batch profile: %s", batch_timer.format())
    return batch_timer


def generate_embeddings(
    photos_archive: Path,
    images_json: Path,
    output: Path,
    model_dir: Path,
    batch_size: int,
    device_name: str,
    limit: int | None,
    profile: bool,
) -> None:
    device = resolve_device(device_name)
    start = time.perf_counter()
    logger.info("Loading SigLIP2 model from %s on %s", model_dir, device)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).to(device)
    model.eval()
    synchronize_device(device)
    logger.info("Loaded SigLIP2 model in %.2fs", time.perf_counter() - start)
    photo_index = load_photo_index(images_json)
    logger.info("Loaded %s photo ids from %s", len(photo_index), images_json)

    tmp_output = output.with_name(f"{output.name}.tmp")
    written = 0
    totals = StageTimer.empty()
    try:
        with tarfile.open(photos_archive, "r:gz") as archive, open_text_writer(tmp_output) as writer:
            logger.info("Streaming wine photos from %s", photos_archive)
            batch: list[PhotoImage] = []
            tar_stage_start = time.perf_counter()

            for member in archive:
                photo = photo_from_member(member, photo_index)
                if photo is None:
                    continue

                image = load_image(archive, member)
                batch.append(PhotoImage(photo=photo, image=image))

                if len(batch) < batch_size:
                    if limit is None or written + len(batch) < limit:
                        continue

                totals.add("tar_stream_read_decode", time.perf_counter() - tar_stage_start)
                batch_start = time.perf_counter()
                batch_timer = write_batch(
                    writer=writer,
                    model=model,
                    processor=processor,
                    device=device,
                    batch=batch,
                    profile=profile,
                )
                for name, seconds in batch_timer.values.items():
                    totals.add(name, seconds)
                logger.info(
                    "Generated %s embeddings in %.2fs",
                    written + len(batch),
                    time.perf_counter() - batch_start,
                )
                written += len(batch)
                batch = []
                tar_stage_start = time.perf_counter()

                if limit is not None and written >= limit:
                    break

            if batch:
                totals.add("tar_stream_read_decode", time.perf_counter() - tar_stage_start)
                batch_start = time.perf_counter()
                batch_timer = write_batch(
                    writer=writer,
                    model=model,
                    processor=processor,
                    device=device,
                    batch=batch,
                    profile=profile,
                )
                for name, seconds in batch_timer.values.items():
                    totals.add(name, seconds)
                written += len(batch)
                logger.info(
                    "Generated %s embeddings in %.2fs",
                    written,
                    time.perf_counter() - batch_start,
                )
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    if profile:
        logger.info("Total profile: %s", totals.format())
    logger.info("Wrote %s embeddings to %s", written, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SigLIP2 embeddings for wine photos.")
    parser.add_argument("--photos-archive", type=Path, default=DEFAULT_PHOTOS_ARCHIVE)
    parser.add_argument("--images-json", type=Path, default=DEFAULT_IMAGES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=Path(all_settings.embeddings.model_dir))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    generate_embeddings(
        photos_archive=args.photos_archive,
        images_json=args.images_json,
        output=args.output,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
        profile=args.profile,
    )
