import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.settings.settings import YoloSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelCrop:
    image: Image.Image
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    input_width: int
    input_height: int


def _to_original_box(
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


def _sigmoid(array: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-array))


class YoloLabelCropper:
    def __init__(self, settings: YoloSettings) -> None:
        self.model_path = Path(settings.model_path)
        self._session = None
        self._input_name: str | None = None
        self._input_size: tuple[int, int] | None = None
        logger.info('YOLO class initialized')

    async def start(self):
        if self._session is not None:
            return
        if not self.model_path.is_file():
            raise RuntimeError(f"YOLO model not found: {self.model_path}")

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        height = int(shape[2]) if isinstance(shape[2], int) else 640
        width = int(shape[3]) if isinstance(shape[3], int) else 640
        self._input_size = (width, height)
        logger.info('YOLO model loaded!')


    def crop(self, image: Image.Image) -> LabelCrop | None:
        if not self._session:
            raise RuntimeError('Model is not loaded yet')

        assert self._input_name is not None
        assert self._input_size is not None

        input_tensor, info = self._preprocess(image, self._input_size)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        prediction = np.asarray(outputs[0]).squeeze()
        detection = self._best_detection(prediction)
        if detection is None:
            return None

        x1, y1, x2, y2, confidence = detection
        box = _to_original_box((x1, y1, x2, y2), image.size, info, margin_ratio=0.04)
        if box is None:
            return None

        return LabelCrop(image=image.crop(box), box=box, confidence=float(confidence))

    @staticmethod
    def _preprocess(
        image: Image.Image,
        input_size: tuple[int, int],
    ) -> tuple[np.ndarray, LetterboxInfo]:
        target_w, target_h = input_size
        scale = min(target_w / image.width, target_h / image.height)
        resized_w = int(round(image.width * scale))
        resized_h = int(round(image.height * scale))
        pad_x = (target_w - resized_w) // 2
        pad_y = (target_h - resized_h) // 2

        canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
        canvas.paste(image.resize((resized_w, resized_h)), (pad_x, pad_y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        info = LetterboxInfo(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            input_width=target_w,
            input_height=target_h,
        )
        return np.transpose(array, (2, 0, 1))[None, ...], info

    @staticmethod
    def _best_detection(prediction: np.ndarray) -> tuple[float, float, float, float, float] | None:
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
            if confidence < 0.15:
                continue
            candidate = (x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence)
            if best is None or candidate[4] > best[4]:
                best = candidate
        return best

    async def stop(self) -> None:
        if self._session is None:
            return
        self._session = None
        self._input_name = None
        self._input_size = None
        logger.info('YOLO model unloaded')


class YoloBottleCropper(YoloLabelCropper):
    bottle_class_id = 39

    def crop(self, image: Image.Image) -> LabelCrop | None:
        if not self._session:
            raise RuntimeError('Model is not loaded yet')

        assert self._input_name is not None
        assert self._input_size is not None

        input_tensor, info = self._preprocess(image, self._input_size)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        if len(outputs) < 2:
            return None

        prediction = np.asarray(outputs[0])
        prototypes = np.asarray(outputs[1])
        if prediction.ndim == 3:
            rows = prediction[0]
        elif prediction.ndim == 2:
            rows = prediction
        else:
            return None
        if prototypes.ndim == 4:
            prototypes = prototypes[0]

        if rows.ndim != 2 or rows.shape[1] < 6:
            return None

        bottle_rows = rows[
            (rows[:, 5].astype(np.int32) == self.bottle_class_id)
            & (rows[:, 4] >= 0.15)
        ]
        if bottle_rows.size == 0:
            return None

        row = bottle_rows[int(np.argmax(bottle_rows[:, 4]))]
        box = _to_original_box(
            tuple(float(value) for value in row[:4]),
            image.size,
            info,
            margin_ratio=0.04,
        )
        if box is None:
            return None

        crop = image.crop(box)
        if row.shape[0] > 6 and prototypes.ndim == 3:
            crop = self._crop_without_background(image, box, row[6:], prototypes, info)
        return LabelCrop(image=crop, box=box, confidence=float(row[4]))

    @staticmethod
    def _crop_without_background(
        image: Image.Image,
        box: tuple[int, int, int, int],
        coefficients: np.ndarray,
        prototypes: np.ndarray,
        info: LetterboxInfo,
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        crop = image.crop(box)
        proto_h, proto_w = prototypes.shape[1:]
        mask_logits = coefficients.astype(np.float32) @ prototypes.reshape(prototypes.shape[0], -1)
        mask = _sigmoid(mask_logits).reshape(proto_h, proto_w)

        model_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        model_mask = model_mask.resize((info.input_width, info.input_height), Image.Resampling.BILINEAR)
        image_w, image_h = image.size
        original_crop = model_mask.crop(
            (
                info.pad_x,
                info.pad_y,
                int(round(info.pad_x + image_w * info.scale)),
                int(round(info.pad_y + image_h * info.scale)),
            )
        )
        original_mask = original_crop.resize(image.size, Image.Resampling.BILINEAR)
        mask_array = np.asarray(original_mask) > 127
        crop_mask = mask_array[y1:y2, x1:x2]
        mask_image = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L")
        white = Image.new("RGB", crop.size, (255, 255, 255))
        white.paste(crop, mask=mask_image)
        return white
