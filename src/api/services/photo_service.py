
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PIL import Image

from src.api.schemas import CropResponse, DetectionResponse, DetectionsResponse, ImageSize
from src.ml.utils import open_rgb_image
from src.ml.yolo import Detection, LabelCrop, YoloBottleCropper, YoloLabelCropper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryImageViews:
    images: dict[str, Image.Image]
    crops: dict[str, CropResponse]
    image: ImageSize | None = None
    detections: DetectionsResponse = field(default_factory=DetectionsResponse)


class WinePhotoService:
    def __init__(
        self,
        label_cropper: YoloLabelCropper | None = None,
        bottle_cropper: YoloBottleCropper | None = None,
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
                source_view="original",
            )
        }

        bottle_crop = self._try_crop("bottle_crop", self.bottle_cropper, original)
        if bottle_crop is not None:
            images["bottle_crop"] = bottle_crop.image
            crops["bottle_crop"] = self._crop_response(bottle_crop, source_view="original")
        else:
            crops["bottle_crop"] = CropResponse(available=False)

        preferred_region = bottle_crop.box if bottle_crop is not None else None
        label_crop = self._try_crop(
            "label_crop",
            self.label_cropper,
            original,
            preferred_region=preferred_region,
        )
        if label_crop is not None:
            images["label_crop"] = label_crop.image
            crops["label_crop"] = self._crop_response(label_crop, source_view="original")
        elif bottle_crop is not None:
            label_crop = self._try_crop("label_crop", self.label_cropper, bottle_crop.image)
            if label_crop is not None:
                images["label_crop"] = label_crop.image
                crops["label_crop"] = self._crop_response(label_crop, source_view="bottle_crop")
            else:
                crops["label_crop"] = CropResponse(
                    available=True,
                    box=(0, 0, original.width, original.height),
                    confidence=1.0,
                    width=original.width,
                    height=original.height,
                    source_view="original_fallback",
                )
        else:
            crops["label_crop"] = CropResponse(
                available=True,
                box=(0, 0, original.width, original.height),
                confidence=1.0,
                width=original.width,
                height=original.height,
                source_view="original_fallback",
            )

        if label_crop is None:
            images["label_crop"] = original

        # все детекции на фото (этикетки — только если искали по всему фото, а не внутри бутылки)
        label_candidates = crops["label_crop"].source_view == "original" and label_crop is not None
        detections = DetectionsResponse(
            bottles=_detections(bottle_crop.candidates if bottle_crop else ()),
            labels=_detections(label_crop.candidates if label_candidates else ()),
        )
        return QueryImageViews(
            images=images,
            crops=crops,
            image=ImageSize(width=original.width, height=original.height),
            detections=detections,
        )

    @staticmethod
    def _try_crop(
        view: str,
        cropper: YoloLabelCropper | YoloBottleCropper | None,
        image: Image.Image,
        preferred_region: tuple[int, int, int, int] | None = None,
    ) -> LabelCrop | None:
        if cropper is None:
            return None
        try:
            return cropper.crop(image, preferred_region=preferred_region)
        except Exception:
            logger.exception("Failed to build %s with YOLO", view)
            return None

    @staticmethod
    def _crop_response(crop: LabelCrop, source_view: str) -> CropResponse:
        return CropResponse(
            available=True,
            box=crop.box,
            confidence=crop.confidence,
            width=crop.image.width,
            height=crop.image.height,
            source_view=source_view,
        )



def _detections(items: tuple[Detection, ...]) -> list[DetectionResponse]:
    return [DetectionResponse(box=item.box, confidence=round(item.confidence, 4)) for item in items]
