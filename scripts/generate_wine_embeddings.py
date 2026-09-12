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
DEFAULT_OUTPUT = Path("data/embeddings/wine_main_siglip2.jsonl.gz")
DEFAULT_MINIO_PREFIX = ""

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
class MainPhoto:
    wine_id: str
    archive_path: str
    minio_path: str


@dataclass(frozen=True)
class PhotoImage:
    photo: MainPhoto
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


def main_photo_from_member(member: tarfile.TarInfo, minio_prefix: str) -> MainPhoto | None:
    prefix = minio_prefix.strip("/")
    if not member.isfile():
        return None
    path = Path(member.name)
    if len(path.parts) < 2:
        return None
    if not path.name.startswith("main") or path.suffix.lower() not in {".jpg", ".jpeg", ".webp"}:
        return None

    wine_id = path.parts[0]
    minio_path = "/".join((prefix, *path.parts)) if prefix else path.as_posix()
    return MainPhoto(
        wine_id=wine_id,
        archive_path=member.name,
        minio_path=minio_path,
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
                    "wine_id": item.photo.wine_id,
                    "main_photo_path": item.photo.minio_path,
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
    output: Path,
    model_dir: Path,
    minio_prefix: str,
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

    tmp_output = output.with_name(f"{output.name}.tmp")
    written = 0
    totals = StageTimer.empty()
    try:
        with tarfile.open(photos_archive, "r:gz") as archive, open_text_writer(tmp_output) as writer:
            logger.info("Streaming main wine photos from %s", photos_archive)
            batch: list[PhotoImage] = []
            seen_wine_ids: set[str] = set()
            tar_stage_start = time.perf_counter()

            for member in archive:
                photo = main_photo_from_member(member, minio_prefix)
                if photo is None or photo.wine_id in seen_wine_ids:
                    continue

                seen_wine_ids.add(photo.wine_id)
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
    parser = argparse.ArgumentParser(description="Generate SigLIP2 embeddings for main wine photos.")
    parser.add_argument("--photos-archive", type=Path, default=DEFAULT_PHOTOS_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=Path(all_settings.embeddings.model_dir))
    parser.add_argument("--minio-prefix", default=DEFAULT_MINIO_PREFIX)
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
        output=args.output,
        model_dir=args.model_dir,
        minio_prefix=args.minio_prefix,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
        profile=args.profile,
    )
