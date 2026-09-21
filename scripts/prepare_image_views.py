from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import YoloSettings


DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_BOTTLE_MODEL = "models/yolo/yolo26x-seg.pt"
DEFAULT_LABEL_MODEL = "models/yolo/label.pt"

logger = logging.getLogger(__name__)


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def normalize_label_crop(image: Image.Image, max_side: int) -> Image.Image:
    normalized = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    normalized = normalized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))

    scale = min(1.0, max_side / max(normalized.size))
    if scale < 1.0:
        new_size = (
            max(1, int(round(normalized.width * scale))),
            max(1, int(round(normalized.height * scale))),
        )
        normalized = normalized.resize(new_size, Image.Resampling.LANCZOS)
    return normalized


def iter_source_images(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    existing_originals = sorted(images_dir.rglob("original.jpg"))
    if existing_originals:
        return existing_originals

    for wine_dir in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not wine_dir.is_dir():
            continue
        for path in sorted(wine_dir.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                paths.append(path)
    return paths


def materialize_original(source_path: Path, overwrite: bool, quality: int) -> tuple[Path, Image.Image] | None:
    if source_path.name == "original.jpg":
        return source_path, load_image(source_path)

    view_dir = source_path.with_suffix("")
    original_path = view_dir / "original.jpg"

    if view_dir.exists() and not view_dir.is_dir():
        raise RuntimeError(f"Expected directory path, got file: {view_dir}")
    view_dir.mkdir(parents=True, exist_ok=True)

    if original_path.exists() and not overwrite:
        source_path.unlink(missing_ok=True)
        return original_path, load_image(original_path)

    image = load_image(source_path)
    save_jpeg(image, original_path, quality)
    source_path.unlink(missing_ok=True)
    return original_path, image


def process_image(
    source_path: Path,
    bottle_cropper: YoloBottleCropper,
    label_cropper: YoloLabelCropper,
    overwrite: bool,
    quality: int,
    normalized_max_side: int,
) -> dict[str, object]:
    result: dict[str, object] = {"source": str(source_path)}
    materialized = materialize_original(source_path, overwrite=overwrite, quality=quality)
    if materialized is None:
        result["status"] = "skipped"
        return result

    original_path, image = materialized
    view_dir = original_path.parent
    result["original"] = str(original_path)

    bottle_crop = bottle_cropper.crop(image)
    if bottle_crop is not None:
        save_jpeg(bottle_crop.image, view_dir / "bottle_crop.jpg", quality)
        result["bottle_confidence"] = round(bottle_crop.confidence, 4)
    else:
        (view_dir / "bottle_crop.jpg").unlink(missing_ok=True)
        result["bottle_confidence"] = None

    label_crop = label_cropper.crop(image)
    if label_crop is not None:
        save_jpeg(label_crop.image, view_dir / "label_crop.jpg", quality)
        normalized = normalize_label_crop(label_crop.image, max_side=normalized_max_side)
        save_jpeg(normalized, view_dir / "normalized_label_crop.jpg", quality)
        result["label_confidence"] = round(label_crop.confidence, 4)
    else:
        (view_dir / "label_crop.jpg").unlink(missing_ok=True)
        (view_dir / "normalized_label_crop.jpg").unlink(missing_ok=True)
        result["label_confidence"] = None

    image.close()
    result["status"] = "processed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical image views for every data/images JPG.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--bottle-model", default=DEFAULT_BOTTLE_MODEL)
    parser.add_argument("--label-model", default=DEFAULT_LABEL_MODEL)
    parser.add_argument("--normalized-max-side", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)

    source_images = iter_source_images(args.images_dir)
    if args.limit is not None:
        source_images = source_images[: args.limit]

    bottle_cropper = YoloBottleCropper(YoloSettings(model_path=args.bottle_model))
    label_cropper = YoloLabelCropper(YoloSettings(model_path=args.label_model))
    await bottle_cropper.start()
    await label_cropper.start()

    try:
        processed = 0
        bottle_found = 0
        label_found = 0
        for source_path in source_images:
            stats = process_image(
                source_path=source_path,
                bottle_cropper=bottle_cropper,
                label_cropper=label_cropper,
                overwrite=args.overwrite,
                quality=args.quality,
                normalized_max_side=args.normalized_max_side,
            )
            processed += 1
            bottle_found += stats.get("bottle_confidence") is not None
            label_found += stats.get("label_confidence") is not None
            logger.info(
                "%s bottle=%s label=%s",
                stats["original"],
                stats["bottle_confidence"],
                stats["label_confidence"],
            )

        logger.info(
            "Processed %s images. bottle crops: %s, label crops: %s.",
            processed,
            bottle_found,
            label_found,
        )
    finally:
        await bottle_cropper.stop()
        await label_cropper.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
