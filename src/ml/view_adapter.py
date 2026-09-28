"""Дообученный адаптер векторов SigLIP2 — свой для каждого ракурса (original, bottle_crop, label_crop).

f(x) = normalize(x + B·(A·x)): небольшая поправка ранга r поверх исходного вектора. Применяется одинаково
к фото каталога (scripts/init_qdrant.py) и к запросу (VisualSearcher.embed), модель SigLIP2 не меняется.
Веса обучает scripts/train_view_adapter.py и хранит в .npz: "<view>/a" (r×D), "<view>/b" (D×r), "meta" (JSON).

Версия адаптера (md5 файла) входит в имя коллекций Qdrant: переобученный адаптер загружается в новые
коллекции, и векторы каталога и запроса не разъезжаются.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ViewAdapter:
    def __init__(self, weights: dict[str, tuple[np.ndarray, np.ndarray]], version: str, meta: dict) -> None:
        self.weights = weights
        self.version = version
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path | None) -> ViewAdapter | None:
        """None — адаптер не задан или файла нет: поиск по исходным векторам."""
        if not path:
            return None
        path = Path(path)
        if not path.is_file():
            logger.warning("View adapter %s not found — searching with raw SigLIP2 vectors", path)
            return None
        data = np.load(path, allow_pickle=False)
        views = sorted({key.split("/", 1)[0] for key in data.files if "/" in key})
        weights = {view: (data[f"{view}/a"].astype("float32"), data[f"{view}/b"].astype("float32")) for view in views}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        version = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        logger.info("View adapter %s loaded: version %s, views %s", path, version, views)
        return cls(weights, version, meta)

    def apply(self, view: str, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if view in self.weights:
            a, b = self.weights[view]
            vectors = vectors + (vectors @ a.T) @ b.T
        return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def collection_encoder(base: str, adapter: ViewAdapter | None) -> str:
    """Суффикс коллекций Qdrant wine_<view>_<суффикс>: с адаптером — вместе с его версией."""
    return f"{base}_{adapter.version}" if adapter is not None else base


def resolve_model_path(path: str | Path, project_root: Path) -> Path:
    """/models/... в контейнере; при запуске скриптов на хосте — ./models/... в репозитории."""
    path = Path(path)
    if path.exists() or not path.is_absolute() or len(path.parts) < 3 or path.parts[1] != "models":
        return path
    return project_root / "models" / Path(*path.parts[2:])
