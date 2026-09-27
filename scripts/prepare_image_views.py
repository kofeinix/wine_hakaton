from __future__ import annotations

import argparse
import asyncio
import fnmatch
import logging
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import YoloSettings


DEFAULT_IMAGES_DIR = Path("data/images_extended")
DEFAULT_BOTTLE_MODEL = "models/yolo/yolo26x.pt"
DEFAULT_LABEL_MODEL = "models/yolo/label.pt"

logger = logging.getLogger(__name__)


def normalize_relative_photo_dir(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None

    # Accept raw log lines like: "2026-... INFO __main__:   wine_id/photo_id".
    if "__main__:" in value:
        value = value.rsplit("__main__:", maxsplit=1)[-1].strip()
    if value.startswith("Missing ") or value.startswith("Processed "):
        return None

    path = Path(value)
    if path.name == "original.jpg":
        path = path.parent
    return path.as_posix().strip("/")


def matches_any_pattern(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


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


def pad_narrow_image_for_crop(image: Image.Image) -> Image.Image:
    if image.height <= image.width * 3:
        return image

    target_width = (image.height + 2) // 3
    horizontal_padding = target_width - image.width
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding
    return ImageOps.expand(
        image,
        border=(left_padding, 0, right_padding, 0),
        fill=(255, 255, 255),
    )


def iter_source_images(images_dir: Path, photo_dir_patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    existing_originals = sorted(
        path for path in images_dir.rglob("original.jpg") if matches_any_pattern(path.parent.name, photo_dir_patterns)
    )
    if existing_originals:
        return existing_originals

    for wine_dir in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not wine_dir.is_dir():
            continue
        for path in sorted(wine_dir.iterdir(), key=lambda item: item.name):
            if (
                path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg"}
                and matches_any_pattern(path.stem, photo_dir_patterns)
            ):
                paths.append(path)
    return paths


def load_photo_dirs_from_file(path: Path) -> list[str]:
    result: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        normalized = normalize_relative_photo_dir(line)
        if normalized is not None:
            result.append(normalized)
    return result


def source_images_from_photo_dirs(images_dir: Path, photo_dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in photo_dirs:
        normalized = normalize_relative_photo_dir(value)
        if normalized is None:
            continue
        original_path = images_dir / normalized / "original.jpg"
        if original_path in seen:
            continue
        seen.add(original_path)
        if original_path.is_file():
            paths.append(original_path)
        else:
            logger.warning("Skipping missing original image: %s", original_path)
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
    bottle_cropper: YoloBottleCropper | None,
    label_cropper: YoloLabelCropper | None,
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
    result["view_dir"] = str(view_dir)

    crop_source_image = pad_narrow_image_for_crop(image)
    bottle_crop = None
    if bottle_cropper is not None:
        bottle_crop = bottle_cropper.crop(crop_source_image)
        if bottle_crop is not None:
            save_jpeg(bottle_crop.image, view_dir / "bottle_crop.jpg", quality)
            result["bottle_confidence"] = round(bottle_crop.confidence, 4)
        else:
            # при повторной подготовке не оставляем устаревший кроп от прошлого прогона
            (view_dir / "bottle_crop.jpg").unlink(missing_ok=True)
            result["bottle_confidence"] = None

    if label_cropper is not None:
        preferred_region = bottle_crop.box if bottle_crop is not None else None
        label_crop = label_cropper.crop(crop_source_image, preferred_region=preferred_region)
        if label_crop is not None:
            save_jpeg(label_crop.image, view_dir / "label_crop.jpg", quality)
            normalized = normalize_label_crop(label_crop.image, max_side=normalized_max_side)
            save_jpeg(normalized, view_dir / "normalized_label_crop.jpg", quality)
            result["label_confidence"] = round(label_crop.confidence, 4)
        else:
            (view_dir / "label_crop.jpg").unlink(missing_ok=True)
            (view_dir / "normalized_label_crop.jpg").unlink(missing_ok=True)
            result["label_confidence"] = None

    if crop_source_image is not image:
        crop_source_image.close()
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
    parser.add_argument(
        "--crop-kind",
        nargs="+",
        choices=["bottle", "label"],
        default=["bottle", "label"],
        help="Which crop types to generate. Default: bottle label.",
    )
    parser.add_argument(
        "--photo-dir",
        nargs="+",
        default=[],
        help="Explicit photo folders relative to --images-dir, for example 0063fa53-.../main.",
    )
    parser.add_argument(
        "--photo-dir-file",
        type=Path,
        default=None,
        help="Text file with one photo folder per line, relative to --images-dir. Log lines are also accepted.",
    )
    parser.add_argument(
        "--photo-dir-pattern",
        nargs="+",
        default=["*"],
        help=(
            "Photo folder masks to process under each wine directory. "
            "Example: --photo-dir-pattern main 'yandex_*' 'flux_*' 'vivino_*'."
        ),
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)

    explicit_photo_dirs = list(args.photo_dir)
    if args.photo_dir_file is not None:
        explicit_photo_dirs.extend(load_photo_dirs_from_file(args.photo_dir_file))

    if explicit_photo_dirs:
        source_images = source_images_from_photo_dirs(args.images_dir, explicit_photo_dirs)
        logger.info("Found %s source images from explicit photo folders.", len(source_images))
    else:
        source_images = iter_source_images(args.images_dir, args.photo_dir_pattern)
        logger.info("Found %s source images matching photo folders: %s", len(source_images), args.photo_dir_pattern)
    if args.limit is not None:
        source_images = source_images[: args.limit]

    crop_kinds = set(args.crop_kind)
    bottle_cropper = YoloBottleCropper(YoloSettings(model_path=args.bottle_model)) if "bottle" in crop_kinds else None
    label_cropper = YoloLabelCropper(YoloSettings(model_path=args.label_model)) if "label" in crop_kinds else None
    if bottle_cropper is not None:
        await bottle_cropper.start()
    if label_cropper is not None:
        await label_cropper.start()

    try:
        processed = 0
        bottle_found = 0
        label_found = 0
        missing_bottle_dirs: list[str] = []
        missing_label_dirs: list[str] = []
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
            if bottle_cropper is not None:
                bottle_found += stats.get("bottle_confidence") is not None
            if label_cropper is not None:
                label_found += stats.get("label_confidence") is not None
            view_dir = Path(str(stats["view_dir"]))
            relative_view_dir = view_dir.relative_to(args.images_dir).as_posix()
            if bottle_cropper is not None and stats.get("bottle_confidence") is None:
                missing_bottle_dirs.append(relative_view_dir)
            if label_cropper is not None and stats.get("label_confidence") is None:
                missing_label_dirs.append(relative_view_dir)
            logger.info(
                "%s bottle=%s label=%s",
                stats["original"],
                stats.get("bottle_confidence"),
                stats.get("label_confidence"),
            )

        logger.info(
            "Processed %s images. bottle crops: %s, label crops: %s.",
            processed,
            bottle_found,
            label_found,
        )
        if bottle_cropper is not None:
            logger.info("Missing bottle_crop folders (%s):", len(missing_bottle_dirs))
            for folder in missing_bottle_dirs:
                logger.info("  %s", folder)
        if label_cropper is not None:
            logger.info("Missing label_crop folders (%s):", len(missing_label_dirs))
            for folder in missing_label_dirs:
                logger.info("  %s", folder)
    finally:
        if bottle_cropper is not None:
            await bottle_cropper.stop()
        if label_cropper is not None:
            await label_cropper.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
