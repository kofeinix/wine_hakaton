#!/usr/bin/env python3
"""Prepare local model files mounted by Docker Compose."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
YOLO_ONNX_PATH = Path("yolo") / "best.onnx"
SIGLIP2_DIR_NAME = "siglip2"
SIGLIP2_REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO ONNX and SigLIP2 weights for docker compose."
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
        help="Optional Hugging Face revision, tag, or commit SHA.",
    )
    return parser.parse_args()


def assert_yolo_model(model_root: Path) -> None:
    yolo_path = model_root / YOLO_ONNX_PATH
    if not yolo_path.is_file():
        raise FileNotFoundError(
            f"Missing {yolo_path}. This repo should include the YOLO ONNX model."
        )
    print(f"YOLO ONNX ready: {yolo_path}")


def download_siglip2(model_root: Path, model_id: str, revision: str | None) -> None:
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

    assert_yolo_model(args.model_root)
    download_siglip2(args.model_root, args.siglip2_model_id, args.revision)
    print("Model directory ready for docker compose: ./models -> /models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
