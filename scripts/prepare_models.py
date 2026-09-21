#!/usr/bin/env python3
"""Prepare local model files mounted by Docker Compose."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError


DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
YOLO_REQUIRED_FILES = (
    Path("yolo") / "label.pt",
    Path("yolo") / "yolo26x-seg.pt",
)
SIGLIP2_DIR_NAME = "siglip2"
DINO_V3_DIR_NAME = "dinov3"
SIGLIP2_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
DINO_V3_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO PT, SigLIP2, and DINOv3 weights for docker compose."
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
        "--dinov3-model-id",
        default=DEFAULT_DINOV3_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_DINOV3_MODEL_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit SHA for both models.",
    )
    parser.add_argument(
        "--dinov3-revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit SHA for DINOv3.",
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


def assert_yolo_model(model_root: Path) -> None:
    missing = [model_root / relative for relative in YOLO_REQUIRED_FILES if not (model_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing YOLO PT model(s): "
            + ", ".join(str(path) for path in missing)
            + ". Put label.pt and yolo26x-seg.pt under models/yolo."
        )
    print("YOLO PT ready: " + ", ".join(str(model_root / relative) for relative in YOLO_REQUIRED_FILES))


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


def download_dinov3(
    model_root: Path,
    model_id: str,
    revision: str | None,
    token: str | None,
) -> None:
    dinov3_dir = model_root / DINO_V3_DIR_NAME
    dinov3_dir.mkdir(parents=True, exist_ok=True)

    if all((dinov3_dir / file_name).is_file() for file_name in DINO_V3_REQUIRED_FILES):
        cleanup_huggingface_local_cache(dinov3_dir)
        print(f"DINOv3 ready: {dinov3_dir}")
        return

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=dinov3_dir,
            token=token,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "*.safetensors",
                "preprocessor_config.json",
            ],
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"DINOv3 model '{model_id}' is gated on Hugging Face. "
            "Accept the model terms on Hugging Face, then run with "
            "HF_TOKEN=<your_token> ./prepare_models or pass --hf-token."
        ) from exc
    cleanup_huggingface_local_cache(dinov3_dir)
    print(f"DINOv3 ready: {local_path}")


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

    assert_yolo_model(args.model_root)
    download_siglip2(args.model_root, args.siglip2_model_id, args.revision, hf_token)
    try:
        download_dinov3(
            args.model_root,
            args.dinov3_model_id,
            args.dinov3_revision or args.revision,
            hf_token,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Model directory ready for docker compose: ./models -> /models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
