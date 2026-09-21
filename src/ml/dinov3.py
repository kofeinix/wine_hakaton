import logging
from pathlib import Path

import numpy as np
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
        return self.embed_many([image])[0]

    def embed_many(self, images: list[Image.Image]) -> list[list[float]]:
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        if not images:
            return []

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            embedding = extract_global_vector(self._model, inputs)
        global_vectors, _ = features_to_numpy(embedding, None)
        return global_vectors.tolist()

    def embed_patch_tokens(
        self,
        image: Image.Image,
    ) -> tuple[list[list[float]], tuple[int, int] | None]:
        return self.embed_patch_tokens_many([image])[0]

    def embed_patch_tokens_many(
        self,
        images: list[Image.Image],
    ) -> list[tuple[list[list[float]], tuple[int, int] | None]]:
        return [
            (tokens.tolist() if tokens is not None else [], grid)
            for tokens, grid in self.embed_patch_token_arrays_many(images)
        ]

    def embed_patch_token_arrays_many(
        self,
        images: list[Image.Image],
    ) -> list[tuple[np.ndarray | None, tuple[int, int] | None]]:
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        if not images:
            return []

        results: list[tuple[np.ndarray | None, tuple[int, int] | None]] = []
        batch_size = max(1, self.settings.patch_batch_size)
        for offset in range(0, len(images), batch_size):
            batch = images[offset : offset + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                patch_tokens, patch_grid = extract_patch_tokens(self._model, inputs)
            if patch_tokens is None:
                results.extend((None, None) for _image in batch)
                continue
            patch_arrays = patch_tokens.detach().cpu().numpy().astype("float32")
            results.extend((array, patch_grid) for array in patch_arrays)
        return results

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
