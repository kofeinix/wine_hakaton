
from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

from src.api.schemas import CropResponse
from src.ml.utils import open_rgb_image
from src.ml.yolo import LabelCrop, YoloLabelCropper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryImageViews:
    images: dict[str, Image.Image]
    crops: dict[str, CropResponse]


class WinePhotoService:
    def __init__(
        self,
        label_cropper: YoloLabelCropper | None = None,
        bottle_cropper: YoloLabelCropper | None = None,
    ) -> None:
        self.label_cropper = label_cropper
        self.bottle_cropper = bottle_cropper

    def build_query_views(self, image_bytes: bytes) -> QueryImageViews:
        original = open_rgb_image(image_bytes)
        images: dict[str, Image.Image] = {"original": original}
        crops: dict[str, CropResponse] = {
            "original": CropResponse(
                available=True,
                box=(0, 0, original.width, original.height),
                confidence=1.0,
                width=original.width,
                height=original.height,
            )
        }

        # bottle_crop is intentionally disabled for runtime search. Keep the
        # implementation nearby while we validate original+label cosine fusion.
        # bottle_crop = self._try_crop("bottle_crop", self.bottle_cropper, original)
        # if bottle_crop is not None:
        #     images["bottle_crop"] = bottle_crop.image
        #     crops["bottle_crop"] = self._crop_response(bottle_crop)
        # else:
        #     crops["bottle_crop"] = CropResponse(available=False)
        crops["bottle_crop"] = CropResponse(available=False)

        label_crop = self._try_crop("label_crop", self.label_cropper, original)
        if label_crop is not None:
            images["label_crop"] = label_crop.image
            crops["label_crop"] = self._crop_response(label_crop)
        else:
            crops["label_crop"] = CropResponse(available=False)

        if label_crop is None:
            images["label_crop"] = original

        return QueryImageViews(images=images, crops=crops)

    @staticmethod
    def _try_crop(
        view: str,
        cropper: YoloLabelCropper | None,
        image: Image.Image,
    ) -> LabelCrop | None:
        if cropper is None:
            return None
        try:
            return cropper.crop(image)
        except Exception:
            logger.exception("Failed to build %s with YOLO", view)
            return None

    @staticmethod
    def _crop_response(crop: LabelCrop) -> CropResponse:
        return CropResponse(
            available=True,
            box=crop.box,
            confidence=crop.confidence,
            width=crop.image.width,
            height=crop.image.height,
        )
