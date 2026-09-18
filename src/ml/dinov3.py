import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.ml.vision_features import (
    VisionFeatures,
    extract_global_vector,
    extract_patch_tokens,
    features_to_numpy,
)
from src.settings.settings import DinoV3Settings

logger = logging.getLogger(__name__)


class DinoV3ImageEmbedder:
    def __init__(self, settings: DinoV3Settings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._device: torch.device | None = None
        logger.info("DINOv3 class initialized")

    async def start(self) -> None:
        if self._model is not None:
            return

        model_path = Path(self.settings.model_dir)
        model_id = str(model_path if model_path.exists() else self.settings.model_id)
        device = self._resolve_device()

        logger.info("Loading DINOv3 model %s on %s", model_id, device)
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device)
        self._model.eval()
        self._device = device
        logger.info("DINOv3 model loaded")

    def embed(self, image: Image.Image) -> list[float]:
        return self.embed_features(image).global_vector

    def embed_patch_tokens(
        self,
        image: Image.Image,
    ) -> tuple[list[list[float]], tuple[int, int] | None]:
        features = self.embed_features(image)
        return features.patch_tokens, features.patch_grid

    def embed_features(self, image: Image.Image) -> VisionFeatures:
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            embedding = extract_global_vector(self._model, inputs)
            patch_tokens, patch_grid = extract_patch_tokens(self._model, inputs)
        global_vectors, patch_arrays = features_to_numpy(embedding, patch_tokens)
        return VisionFeatures(
            global_vector=global_vectors[0].tolist(),
            patch_tokens=patch_arrays[0].tolist() if patch_arrays is not None else [],
            patch_grid=patch_grid,
        )

    async def stop(self) -> None:
        if self._model is None and self._processor is None:
            return

        self._model = None
        self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._device = None
        logger.info("DINOv3 model unloaded")

    def _resolve_device(self) -> torch.device:
        if self.settings.device != "auto":
            return torch.device(self.settings.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
