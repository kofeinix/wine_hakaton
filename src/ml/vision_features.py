from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VisionFeatures:
    global_vector: list[float]
    patch_tokens: list[list[float]]
    patch_grid: tuple[int, int] | None


def extract_global_vector(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
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


def extract_patch_tokens(
    model: Any,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, tuple[int, int] | None]:
    outputs = model(**inputs)
    hidden_state = getattr(outputs, "last_hidden_state", None)
    if hidden_state is None:
        return None, None

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None or pixel_values.ndim != 4:
        return None, None

    patch_size = _resolve_patch_size(model)
    height, width = pixel_values.shape[-2:]
    patch_grid = (height // patch_size, width // patch_size)
    patch_count = patch_grid[0] * patch_grid[1]

    if hidden_state.shape[1] < patch_count:
        return None, None

    patch_tokens = hidden_state[:, -patch_count:, :]
    patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    return patch_tokens, patch_grid


def features_to_numpy(
    global_vector: torch.Tensor,
    patch_tokens: torch.Tensor | None,
) -> tuple[Any, Any | None]:
    global_array = global_vector.detach().cpu().numpy().astype("float32")
    if patch_tokens is None:
        return global_array, None
    patch_array = patch_tokens.detach().cpu().numpy().astype("float32")
    return global_array, patch_array


def _resolve_patch_size(model: Any) -> int:
    config = getattr(model, "config", None)
    patch_size = getattr(config, "patch_size", None)
    if patch_size is not None:
        return int(patch_size)

    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if patch_size is not None:
        return int(patch_size)

    return 16
