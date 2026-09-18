from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps


DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_BOTTLE_MODEL = Path("models/yolo/yolo26n-seg.onnx")
DEFAULT_LABEL_MODEL = Path("models/yolo/label.onnx")
BOTTLE_CLASS_ID = 39

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    input_width: int
    input_height: int


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None = None


def sigmoid(array: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-array))


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def preprocess(image: Image.Image, input_size: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    target_w, target_h = input_size
    scale = min(target_w / image.width, target_h / image.height)
    resized_w = int(round(image.width * scale))
    resized_h = int(round(image.height * scale))
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2

    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    info = LetterboxInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        input_width=target_w,
        input_height=target_h,
    )
    return tensor, info


def model_input_size(session: ort.InferenceSession) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    height = int(shape[2]) if isinstance(shape[2], int) else 640
    width = int(shape[3]) if isinstance(shape[3], int) else 640
    return width, height


def to_original_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    info: LetterboxInfo,
    margin_ratio: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = (x1 - info.pad_x) / info.scale
    y1 = (y1 - info.pad_y) / info.scale
    x2 = (x2 - info.pad_x) / info.scale
    y2 = (y2 - info.pad_y) / info.scale

    width = x2 - x1
    height = y2 - y1
    if width <= 1 or height <= 1:
        return None

    margin = max(width, height) * margin_ratio
    image_w, image_h = image_size
    out = (
        max(0, int(x1 - margin)),
        max(0, int(y1 - margin)),
        min(image_w, int(x2 + margin)),
        min(image_h, int(y2 + margin)),
    )
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


class BottleSegmenter:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        margin_ratio: float,
    ) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = model_input_size(self.session)
        self.confidence_threshold = confidence_threshold
        self.margin_ratio = margin_ratio

    def detect(self, image: Image.Image) -> Detection | None:
        input_tensor, info = preprocess(image, self.input_size)
        prediction, prototypes = self.session.run(None, {self.input_name: input_tensor})
        rows = np.asarray(prediction)[0]
        prototypes = np.asarray(prototypes)[0]

        bottle_rows = rows[
            (rows[:, 5].astype(np.int32) == BOTTLE_CLASS_ID)
            & (rows[:, 4] >= self.confidence_threshold)
        ]
        if bottle_rows.size == 0:
            return None

        row = bottle_rows[int(np.argmax(bottle_rows[:, 4]))]
        model_box = tuple(float(v) for v in row[:4])
        box = to_original_box(model_box, image.size, info, self.margin_ratio)
        if box is None:
            return None

        mask = self._build_original_mask(row[6:], prototypes, image.size, info)
        return Detection(box=box, confidence=float(row[4]), mask=mask)

    def _build_original_mask(
        self,
        coefficients: np.ndarray,
        prototypes: np.ndarray,
        image_size: tuple[int, int],
        info: LetterboxInfo,
    ) -> np.ndarray:
        proto_h, proto_w = prototypes.shape[1:]
        mask_logits = coefficients.astype(np.float32) @ prototypes.reshape(prototypes.shape[0], -1)
        mask = sigmoid(mask_logits).reshape(proto_h, proto_w)

        model_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        model_mask = model_mask.resize((info.input_width, info.input_height), Image.Resampling.BILINEAR)

        crop = model_mask.crop(
            (
                info.pad_x,
                info.pad_y,
                int(round(info.pad_x + image_size[0] * info.scale)),
                int(round(info.pad_y + image_size[1] * info.scale)),
            )
        )
        original_mask = crop.resize(image_size, Image.Resampling.BILINEAR)
        return np.asarray(original_mask) > 127

    @staticmethod
    def crop_without_background(image: Image.Image, detection: Detection) -> Image.Image:
        x1, y1, x2, y2 = detection.box
        crop = image.crop(detection.box)
        if detection.mask is None:
            return crop

        mask = detection.mask[y1:y2, x1:x2]
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        white = Image.new("RGB", crop.size, (255, 255, 255))
        white.paste(crop, mask=mask_image)
        return white


class LabelDetector:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        margin_ratio: float,
    ) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = model_input_size(self.session)
        self.confidence_threshold = confidence_threshold
        self.margin_ratio = margin_ratio

    def detect(self, image: Image.Image) -> Detection | None:
        input_tensor, info = preprocess(image, self.input_size)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        prediction = np.asarray(outputs[0]).squeeze()
        if prediction.ndim != 2:
            return None
        if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 16:
            prediction = prediction.T

        best: tuple[float, float, float, float, float] | None = None
        for row in prediction:
            if row.shape[0] < 5:
                continue
            x, y, w, h = row[:4].astype(float)
            objectness = float(row[4])
            class_score = float(np.max(row[5:])) if row.shape[0] > 5 else 1.0
            confidence = objectness * class_score if row.shape[0] > 5 else objectness
            if confidence < self.confidence_threshold:
                continue
            candidate = (x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence)
            if best is None or candidate[4] > best[4]:
                best = candidate

        if best is None:
            return None

        box = to_original_box(best[:4], image.size, info, self.margin_ratio)
        if box is None:
            return None
        return Detection(box=box, confidence=best[4])


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
    for wine_dir in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not wine_dir.is_dir():
            continue
        for path in sorted(wine_dir.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                paths.append(path)
    return paths


def materialize_original(source_path: Path, overwrite: bool, quality: int) -> tuple[Path, Image.Image] | None:
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
    bottle_segmenter: BottleSegmenter,
    label_detector: LabelDetector,
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

    bottle_detection = bottle_segmenter.detect(image)
    if bottle_detection is not None:
        bottle_crop = bottle_segmenter.crop_without_background(image, bottle_detection)
        save_jpeg(bottle_crop, view_dir / "bottle_crop.jpg", quality)
        result["bottle_confidence"] = round(bottle_detection.confidence, 4)
    else:
        (view_dir / "bottle_crop.jpg").unlink(missing_ok=True)
        result["bottle_confidence"] = None

    label_detection = label_detector.detect(image)
    if label_detection is not None:
        label_crop = image.crop(label_detection.box)
        save_jpeg(label_crop, view_dir / "label_crop.jpg", quality)
        normalized = normalize_label_crop(label_crop, max_side=normalized_max_side)
        save_jpeg(normalized, view_dir / "normalized_label_crop.jpg", quality)
        result["label_confidence"] = round(label_detection.confidence, 4)
    else:
        (view_dir / "label_crop.jpg").unlink(missing_ok=True)
        (view_dir / "normalized_label_crop.jpg").unlink(missing_ok=True)
        result["label_confidence"] = None

    result["status"] = "processed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical image views for every data/images JPG.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--bottle-model", type=Path, default=DEFAULT_BOTTLE_MODEL)
    parser.add_argument("--label-model", type=Path, default=DEFAULT_LABEL_MODEL)
    parser.add_argument("--bottle-conf", type=float, default=0.15)
    parser.add_argument("--label-conf", type=float, default=0.15)
    parser.add_argument("--bottle-margin", type=float, default=0.04)
    parser.add_argument("--label-margin", type=float, default=0.04)
    parser.add_argument("--normalized-max-side", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if not args.bottle_model.is_file():
        raise FileNotFoundError(args.bottle_model)
    if not args.label_model.is_file():
        raise FileNotFoundError(args.label_model)

    source_images = iter_source_images(args.images_dir)
    if args.limit is not None:
        source_images = source_images[: args.limit]

    bottle_segmenter = BottleSegmenter(
        model_path=args.bottle_model,
        confidence_threshold=args.bottle_conf,
        margin_ratio=args.bottle_margin,
    )
    label_detector = LabelDetector(
        model_path=args.label_model,
        confidence_threshold=args.label_conf,
        margin_ratio=args.label_margin,
    )

    processed = 0
    bottle_found = 0
    label_found = 0
    for source_path in source_images:
        stats = process_image(
            source_path=source_path,
            bottle_segmenter=bottle_segmenter,
            label_detector=label_detector,
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


if __name__ == "__main__":
    main()
