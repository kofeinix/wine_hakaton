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

        input_tensor, scale, pad_x, pad_y = self._preprocess(image, self._input_size)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        prediction = np.asarray(outputs[0]).squeeze()
        detection = self._best_detection(prediction)
        if detection is None:
            return None

        x1, y1, x2, y2, confidence = detection
        x1 = int(max(0, min(image.width, (x1 - pad_x) / scale)))
        y1 = int(max(0, min(image.height, (y1 - pad_y) / scale)))
        x2 = int(max(0, min(image.width, (x2 - pad_x) / scale)))
        y2 = int(max(0, min(image.height, (y2 - pad_y) / scale)))
        if x2 <= x1 or y2 <= y1:
            return None

        margin_x = int((x2 - x1) * 0.04)
        margin_y = int((y2 - y1) * 0.04)
        box = (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(image.width, x2 + margin_x),
            min(image.height, y2 + margin_y),
        )
        return LabelCrop(image=image.crop(box), box=box, confidence=float(confidence))

    @staticmethod
    def _preprocess(
        image: Image.Image,
        input_size: tuple[int, int],
    ) -> tuple[np.ndarray, float, int, int]:
        target_w, target_h = input_size
        scale = min(target_w / image.width, target_h / image.height)
        resized_w = int(round(image.width * scale))
        resized_h = int(round(image.height * scale))
        pad_x = (target_w - resized_w) // 2
        pad_y = (target_h - resized_h) // 2

        canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
        canvas.paste(image.resize((resized_w, resized_h)), (pad_x, pad_y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))[None, ...], scale, pad_x, pad_y

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
