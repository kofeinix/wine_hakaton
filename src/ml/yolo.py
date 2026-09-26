import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.settings.settings import YoloSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class LabelCrop:
    image: Image.Image
    box: tuple[int, int, int, int]
    confidence: float
    # все уверенные детекции на изображении (выбранная — одна из них), по убыванию уверенности
    candidates: tuple[Detection, ...] = ()


class UltralyticsCropper:
    class_id: int | None = None
    confidence_threshold = 0.15
    selection_min_confidence = 0.25
    confidence_weight = 0.35
    center_weight = 0.45
    size_weight = 0.20
    preferred_region_weight = 0.0
    margin_ratio = 0.04
    remove_background = False

    def __init__(self, settings: YoloSettings) -> None:
        self.model_ref = settings.model_path
        self._model: YOLO | None = None
        logger.info("YOLO cropper initialized for %s", self.model_ref)

    async def start(self) -> None:
        if self._model is not None:
            return
        self._model = YOLO(self.model_ref)
        logger.info("YOLO model loaded: %s", self.model_ref)

    async def stop(self) -> None:
        self._model = None
        logger.info("YOLO model unloaded: %s", self.model_ref)

    def crop(
        self,
        image: Image.Image,
        preferred_region: tuple[int, int, int, int] | None = None,
    ) -> LabelCrop | None:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet")

        results = self._model.predict(
            # ultralytics считает numpy-массив BGR (как из cv2); RGB от PIL переворачиваем,
            # иначе красное вино «синеет» и детекции теряют уверенность
            source=np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1]),
            classes=[self.class_id] if self.class_id is not None else None,
            conf=self.confidence_threshold,
            retina_masks=True,
            verbose=False,
            device="mps",
        )
        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        img_h, img_w = result.orig_shape
        best = self._select_best_box(result, img_w, img_h, preferred_region=preferred_region)
        if best is None:
            return None

        confs = result.boxes.conf.cpu().numpy()
        confidence = float(confs[best])

        x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy().astype(float)
        width = x2 - x1
        height = y2 - y1
        margin = max(width, height) * self.margin_ratio
        bx1 = max(0, int(x1 - margin))
        by1 = max(0, int(y1 - margin))
        bx2 = min(img_w, int(x2 + margin))
        by2 = min(img_h, int(y2 + margin))
        if bx2 <= bx1 or by2 <= by1:
            return None

        box = (bx1, by1, bx2, by2)
        crop = image.crop(box)
        if self.remove_background and result.masks is not None:
            crop = self._crop_without_background(image, crop, box, result, best)
        candidates = tuple(
            Detection(box=tuple(int(v) for v in xyxy), confidence=float(conf))
            for xyxy, conf in sorted(
                zip(result.boxes.xyxy.cpu().numpy(), confs, strict=True), key=lambda item: -item[1]
            )
            if conf >= self.selection_min_confidence
        )
        return LabelCrop(image=crop, box=box, confidence=confidence, candidates=candidates)

    @staticmethod
    def _crop_without_background(
        image: Image.Image,
        crop: Image.Image,
        box: tuple[int, int, int, int],
        result,
        best: int,
    ) -> Image.Image:
        bx1, by1, bx2, by2 = box
        img_h, img_w = result.orig_shape
        mask = result.masks.data[best].cpu().numpy().astype(bool)
        if mask.shape != (img_h, img_w):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_img = mask_img.resize((img_w, img_h), Image.Resampling.BILINEAR)
            mask = np.asarray(mask_img) > 127

        crop_mask = mask[by1:by2, bx1:bx2]
        mask_image = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L")
        white = Image.new("RGB", crop.size, (255, 255, 255))
        white.paste(crop, mask=mask_image)
        crop.close()
        return white

    @classmethod
    def _select_best_box(
        cls,
        result,
        img_w: int,
        img_h: int,
        preferred_region: tuple[int, int, int, int] | None = None,
    ) -> int | None:
        boxes = result.boxes.xyxy.cpu().numpy().astype(float)
        confs = result.boxes.conf.cpu().numpy().astype(float)
        valid = confs >= cls.selection_min_confidence
        if not np.any(valid):
            return None

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        widths = np.maximum(0.0, x2 - x1)
        heights = np.maximum(0.0, y2 - y1)
        areas = widths * heights
        max_area = float(np.max(areas[valid]))
        if max_area <= 0:
            return None

        center_x = x1 + widths / 2
        dx = (center_x - img_w / 2) / max(img_w / 2, 1)
        center_scores = np.clip(1.0 - np.abs(dx), 0.0, 1.0)
        size_scores = areas / max_area

        scores = (
            cls.confidence_weight * confs
            + cls.center_weight * center_scores
            + cls.size_weight * size_scores
        )

        if preferred_region is not None and cls.preferred_region_weight > 0:
            region_scores = cls._candidate_coverage_by_region(boxes, preferred_region)
            scores = scores + cls.preferred_region_weight * region_scores

        scores = np.where(valid, scores, -np.inf)
        return int(np.argmax(scores))

    @staticmethod
    def _candidate_coverage_by_region(
        boxes: np.ndarray,
        preferred_region: tuple[int, int, int, int],
    ) -> np.ndarray:
        rx1, ry1, rx2, ry2 = preferred_region
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        inter_w = np.maximum(0.0, np.minimum(x2, rx2) - np.maximum(x1, rx1))
        inter_h = np.maximum(0.0, np.minimum(y2, ry2) - np.maximum(y1, ry1))
        inter_area = inter_w * inter_h
        candidate_area = np.maximum(1.0, (x2 - x1) * (y2 - y1))
        return inter_area / candidate_area


class YoloLabelCropper(UltralyticsCropper):
    """Label cropper backed by an Ultralytics PT model."""

    preferred_region_weight = 0.25


class YoloBottleCropper(UltralyticsCropper):
    """Bottle cropper backed by an Ultralytics detection PT model."""

    class_id = 39
    confidence_weight = 0.30
    center_weight = 0.45
    size_weight = 0.25
