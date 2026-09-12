import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.settings.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class SiglipImageEmbedder:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self._processor = None
        self._model = None
        self._device: torch.device | None = None
        logger.info(f"SigLIP2 class initialized")

    async def start(self) -> None:
        if self._model is not None:
            return

        model_path = Path(self.settings.model_dir)
        model_id = str(model_path if model_path.exists() else self.settings.model_id)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info("Loading SigLIP2 model %s on %s", model_id, device)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device)
        self._model.eval()
        self._device = device
        logger.info(f"SigLIP2 model loaded")


    def embed(self, image: Image.Image) -> list[float]:
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model.get_image_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                embedding = outputs
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                embedding = outputs.image_embeds
            else:
                embedding = outputs.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].detach().cpu().numpy().astype("float32").tolist()

    async def stop(self) -> None:
        if self._model is None and self._processor is None:
            return

        self._model = None
        self._processor = None

        # Free cached GPU memory held by PyTorch's allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._device = None
        logger.info("SigLIP2 model unloaded")
