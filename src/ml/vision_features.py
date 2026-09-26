from __future__ import annotations

from typing import Any

import torch


def extract_global_vector(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Нормализованный глобальный эмбеддинг изображения (SigLIP2 и совместимые модели)."""
    if hasattr(model, "get_image_features"):
        outputs = model.get_image_features(**inputs)
    else:
        outputs = model(**inputs)

    if isinstance(outputs, torch.Tensor):
        embeddings = outputs
    elif getattr(outputs, "pooler_output", None) is not None:
        embeddings = outputs.pooler_output
    elif getattr(outputs, "image_embeds", None) is not None:
        embeddings = outputs.image_embeds
    elif getattr(outputs, "last_hidden_state", None) is not None:
        embeddings = outputs.last_hidden_state[:, 0]
    else:
        raise ValueError("Model output does not contain an image embedding")

    return torch.nn.functional.normalize(embeddings, dim=-1)
