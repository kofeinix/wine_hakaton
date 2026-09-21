import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.settings.settings import YoloSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelCrop:
    image: Image.Image
    box: tuple[int, int, int, int]
    confidence: float


class UltralyticsCropper:
    class_id: int | None = None
    confidence_threshold = 0.15
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

    def crop(self, image: Image.Image) -> LabelCrop | None:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet")

        results = self._model.predict(
            source=np.asarray(image.convert("RGB")),
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

        confs = result.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        confidence = float(confs[best])
        img_h, img_w = result.orig_shape

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
        return LabelCrop(image=crop, box=box, confidence=confidence)

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


class YoloLabelCropper(UltralyticsCropper):
    """Label cropper backed by an Ultralytics PT model."""


class YoloBottleCropper(UltralyticsCropper):
    """Bottle segmentation cropper backed by an Ultralytics PT model."""

    class_id = 39
    remove_background = True
