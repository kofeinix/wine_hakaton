from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_MODEL = "models/yolo/yolo26x-seg.pt"
BOTTLE_CLASS_ID = 39

logger = logging.getLogger(__name__)


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    if path.name == "original.jpg":
        raise RuntimeError(f"Refusing to overwrite original: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def iter_original_images(images_dir: Path) -> list[Path]:
    return sorted(images_dir.rglob("original.jpg"))


def extract_bottle_crop(
    model: YOLO,
    original_path: Path,
    conf: float,
    margin_ratio: float,
) -> tuple[Image.Image, float] | None:
    """
    Возвращает (crop на белом фоне, confidence) или None, если бутылка не найдена.
    """
    results = model.predict(
        source=str(original_path),
        classes=[BOTTLE_CLASS_ID],
        conf=conf,
        retina_masks=True,
        verbose=False,
    )
    if not results:
        return None

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    # берём самое уверенное детектирование
    confs = result.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    confidence = float(confs[best])

    # маска в исходном разрешении изображения (retina_masks=True)
    mask = result.masks.data[best].cpu().numpy().astype(bool)  # (H, W)
    img_h, img_w = result.orig_shape
    if mask.shape != (img_h, img_w):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize((img_w, img_h), Image.Resampling.BILINEAR)
        mask = np.asarray(mask_img) > 127

    # bbox в пикселях исходного изображения + margin
    x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy().astype(float)
    w = x2 - x1
    h = y2 - y1
    margin = max(w, h) * margin_ratio
    bx1 = max(0, int(x1 - margin))
    by1 = max(0, int(y1 - margin))
    bx2 = min(img_w, int(x2 + margin))
    by2 = min(img_h, int(y2 + margin))
    if bx2 <= bx1 or by2 <= by1:
        return None

    image = Image.open(original_path).convert("RGB")
    crop = image.crop((bx1, by1, bx2, by2))
    crop_mask = mask[by1:by2, bx1:bx2]

    mask_image = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=mask_image)
    return white, confidence


def process_original(
    original_path: Path,
    model: YOLO,
    overwrite: bool,
    conf: float,
    margin_ratio: float,
    quality: int,
) -> dict[str, object]:
    result: dict[str, object] = {"original": str(original_path)}
    target = original_path.parent / "bottle_crop.jpg"

    if target.exists() and not overwrite:
        result["status"] = "skipped"
        result["bottle_confidence"] = None
        return result

    try:
        extracted = extract_bottle_crop(
            model=model,
            original_path=original_path,
            conf=conf,
            margin_ratio=margin_ratio,
        )
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        result["status"] = "error"
        result["bottle_confidence"] = None
        result["error"] = str(exc)
        return result

    if extracted is None:
        target.unlink(missing_ok=True)
        result["status"] = "no_bottle"
        result["bottle_confidence"] = None
        return result

    crop, confidence = extracted
    save_jpeg(crop, target, quality)
    result["status"] = "processed"
    result["bottle_confidence"] = round(confidence, 4)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bottle_crop.jpg next to every original.jpg using a YOLO PT segmentation model."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model path or Ultralytics model reference, for example yolo26x-seg.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)

    originals = iter_original_images(args.images_dir)
    if args.limit is not None:
        originals = originals[: args.limit]

    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    processed = 0
    skipped = 0
    no_bottle = 0
    errors = 0

    for original_path in originals:
        stats = process_original(
            original_path=original_path,
            model=model,
            overwrite=args.overwrite,
            conf=args.conf,
            margin_ratio=args.margin,
            quality=args.quality,
        )
        status = stats["status"]
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        elif status == "error":
            errors += 1
            logger.error("%s error=%s", stats["original"], stats.get("error"))
        else:
            no_bottle += 1

        logger.info("%s status=%s conf=%s", stats["original"], status, stats["bottle_confidence"])

    attempted = processed + no_bottle + errors
    success_rate = (processed / attempted * 100.0) if attempted else 0.0

    logger.info("=" * 60)
    logger.info("Total originals found : %s", len(originals))
    logger.info("Successfully extracted: %s", processed)
    logger.info("No bottle detected    : %s", no_bottle)
    logger.info("Errors                : %s", errors)
    logger.info("Skipped (already done): %s", skipped)
    logger.info("Success rate          : %.1f%% (%s/%s)", success_rate, processed, attempted)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
