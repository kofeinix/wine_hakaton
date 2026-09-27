#!/usr/bin/env python3
"""Подготовить модели в ./models (монтируется в контейнер как /models).

- models/yolo/label.pt   — своя модель детекции этикеток, лежит в репозитории;
- models/yolo/yolo26x.pt — COCO-детектор бутылок, скачивается с релизов Ultralytics;
- models/siglip2/        — SigLIP2 для эмбеддингов, скачивается с Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
LABEL_MODEL = Path("yolo") / "label.pt"
BOTTLE_MODEL = Path("yolo") / "yolo26x.pt"
SIGLIP2_DIR_NAME = "siglip2"
SIGLIP2_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачать и проверить модели для docker compose (YOLO, SigLIP2)."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"Local directory mounted as /models. Default: {DEFAULT_MODEL_ROOT}",
    )
    parser.add_argument(
        "--siglip2-model-id",
        default=DEFAULT_SIGLIP2_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_SIGLIP2_MODEL_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit SHA for SigLIP2.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face access token for gated models. "
            "Prefer HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars."
        ),
    )
    return parser.parse_args()


def prepare_yolo(model_root: Path) -> None:
    label = model_root / LABEL_MODEL
    if not label.is_file():
        raise FileNotFoundError(f"Нет модели этикеток {label}: она хранится в репозитории, проверьте клон.")

    bottle = model_root / BOTTLE_MODEL
    if not bottle.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        bottle.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(bottle))  # официальный релиз Ultralytics (github.com/ultralytics/assets)
    if not bottle.is_file():
        raise FileNotFoundError(f"Не удалось скачать {bottle}")
    print(f"YOLO ready: {label}, {bottle}")


def download_siglip2(
    model_root: Path,
    model_id: str,
    revision: str | None,
    token: str | None,
) -> None:
    siglip2_dir = model_root / SIGLIP2_DIR_NAME
    siglip2_dir.mkdir(parents=True, exist_ok=True)

    if all((siglip2_dir / file_name).is_file() for file_name in SIGLIP2_REQUIRED_FILES):
        cleanup_huggingface_local_cache(siglip2_dir)
        print(f"SigLIP2 ready: {siglip2_dir}")
        return

    local_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=siglip2_dir,
        token=token,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
    )
    cleanup_huggingface_local_cache(siglip2_dir)
    print(f"SigLIP2 ready: {local_path}")


def cleanup_huggingface_local_cache(model_dir: Path) -> None:
    cache_dir = model_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def main() -> int:
    args = parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    prepare_yolo(args.model_root)
    download_siglip2(args.model_root, args.siglip2_model_id, args.revision, hf_token)
    print("Model directory ready for docker compose: ./models -> /models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
