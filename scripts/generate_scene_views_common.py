from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INPUT_DIR = Path("data/images_extended")
DEFAULT_OUTPUT_DIR = Path("data/images")


@dataclass(frozen=True)
class SceneView:
    index: int
    slug: str
    scene: str
    lighting: str
    angle: int
    prompt_scene: str


SCENE_VIEWS: tuple[SceneView, ...] = (
    SceneView(
        1,
        "shelf_bright_minus10",
        "wine_shop_shelf",
        "bright",
        -10,
        "standing fully on a metal shelf in the alcoholic drinks aisle of a real grocery supermarket, set slightly back from the shelf edge with its entire base supported by the shelf, photographed like a close-up shopper phone photo while facing the product section, with the bottle filling more than half of the frame height, next to other wine and spirits bottles, barcode shelf labels, price rails, promo tags, aisle signs, and a shopping cart in the background, bright fluorescent store lighting",
    ),
    SceneView(
        2,
        "shelf_bright_plus10",
        "wine_shop_shelf",
        "bright",
        10,
        "standing fully on a metal shelf in the alcoholic drinks aisle of a real grocery supermarket, set slightly back from the shelf edge with its entire base supported by the shelf, photographed like a close-up shopper phone photo while facing the product section, with the bottle filling more than half of the frame height, next to other wine and spirits bottles, barcode shelf labels, price rails, promo tags, aisle signs, and a shopping cart in the background, bright fluorescent store lighting",
    ),
    SceneView(
        3,
        "shelf_dark_zero",
        "wine_shop_shelf",
        "dark",
        0,
        "standing fully on a metal shelf in the alcoholic drinks aisle of a real grocery supermarket, set slightly back from the shelf edge with its entire base supported by the shelf, photographed like a close-up shopper phone photo while facing the product section, with the bottle filling more than half of the frame height, next to other wine and spirits bottles, barcode shelf labels, price rails, promo tags, aisle signs, and a shopping cart in the background, dimmer evening supermarket lighting",
    ),
    SceneView(
        4,
        "table_bright_minus10",
        "home_table",
        "bright",
        -10,
        "standing on a table at home, realistic kitchen or dining room background, bright natural daylight",
    ),
    SceneView(
        5,
        "table_bright_plus10",
        "home_table",
        "bright",
        10,
        "standing on a table at home, realistic kitchen or dining room background, bright natural daylight",
    ),
    SceneView(
        6,
        "table_dark_zero",
        "home_table",
        "dark",
        0,
        "standing on a table at home, realistic kitchen or dining room background, dim evening light",
    ),
    SceneView(
        7,
        "hands_bright_minus10",
        "human_hand",
        "bright",
        -10,
        "held naturally in one person's hand only, with a single visible hand gripping below the label and the other hand out of frame, realistic hand-to-bottle proportions like a normal 750 ml wine bottle, realistic casual indoor background, bright soft lighting",
    ),
    SceneView(
        8,
        "hands_bright_plus10",
        "human_hand",
        "bright",
        10,
        "held naturally in one person's hand only, with a single visible hand gripping below the label and the other hand out of frame, realistic hand-to-bottle proportions like a normal 750 ml wine bottle, realistic casual indoor background, bright soft lighting",
    ),
    SceneView(
        9,
        "hands_dark_zero",
        "human_hand",
        "dark",
        0,
        "held naturally in one person's hand only, with a single visible hand gripping below the label and the other hand out of frame, realistic hand-to-bottle proportions like a normal 750 ml wine bottle, realistic casual indoor background, dim warm lighting",
    ),
)


NEGATIVE_PROMPT = (
    "changed wine label, different logo, different bottle, altered brand, fake text, unreadable label, "
    "warped label, duplicate bottle, extra label, cropped bottle, missing bottle, broken glass, watermark"
)


def parse_common_args(
    description: str,
    model_id: str,
    output_prefix: str,
    pipeline_class: str,
    default_num_inference_steps: int,
    default_guidance_scale: float,
    default_true_cfg_scale: float | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=model_id)
    parser.add_argument("--pipeline-class", default=pipeline_class)
    parser.add_argument("--output-prefix", default=output_prefix)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wine-id",
        action="append",
        default=[],
        help="Wine UUID to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--wine-ids-file",
        type=Path,
        default=None,
        help="Text file with one wine UUID per line. Empty lines and lines starting with # are ignored.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")
    parser.add_argument(
        "--quantization",
        choices=("none", "fp8wo", "fp8dq"),
        default="none",
        help="Optional TorchAO quantization. Use fp8wo for Flux on RTX 4090/L40/H100-class GPUs.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=default_num_inference_steps)
    parser.add_argument("--guidance-scale", type=float, default=default_guidance_scale)
    parser.add_argument("--true-cfg-scale", type=float, default=default_true_cfg_scale)
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Override negative prompt. For Qwen Edit 2511 the official default is a single space.",
    )
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def iter_main_originals(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(input_dir.glob("*/main/original.*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def wine_id_from_source(path: Path) -> str:
    return path.parent.parent.name


def load_requested_wine_ids(args: argparse.Namespace) -> set[str] | None:
    requested = {wine_id.strip() for wine_id in args.wine_id if wine_id.strip()}
    if args.wine_ids_file is not None:
        if not args.wine_ids_file.is_file():
            raise FileNotFoundError(args.wine_ids_file)
        for line in args.wine_ids_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                requested.add(cleaned)
    return requested or None


def load_image(path: Path, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def fit_image_to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    if width <= 0 or height <= 0:
        return image
    scale = min(width / image.width, height / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    offset = ((width - resized.width) // 2, (height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def build_qwen_prompt(view: SceneView) -> str:
    if view.scene == "qwen_store_bad_phone":
        return (
            "Create a realistic image edit using the input wine bottle as the authoritative visual reference. "
            "Preserve the bottle and label identity: same glass shape, neck, cap, capsule, label size, label position, "
            "label colors, logo placement, typography layout, and visible text. Place the same bottle standing fully on a metal shelf "
            "in the alcoholic drinks aisle of a real grocery supermarket, beside other wine and spirits bottles, price rails, barcode "
            "shelf labels, promo tags, aisle signage, and part of the store background. The bottle is centered, fully in frame, fully "
            "supported by the shelf, not hanging over the edge, and fills roughly 55 to 70 percent of the image height. Make it look "
            "like a real imperfect phone photo: slight camera tilt, mild motion blur, mild defocus, imperfect exposure, natural JPEG "
            "compression/noise, and realistic glare/reflections on the glass. The label may be slightly softened by the phone-photo "
            "domain but remains sufficiently readable and visually searchable. Change only the scene, camera imperfections, lighting, "
            "shadows, reflections, and background."
        )

    if view.scene == "qwen_table_extreme_geometry":
        return (
            "Create a realistic image edit using the input wine bottle as the authoritative visual reference. Preserve the bottle and "
            "label identity: same glass shape, neck, cap, capsule, label size, label position, label colors, logo placement, typography "
            "layout, and visible text. Place the same bottle on a table at home in a realistic kitchen or dining room. Use an extreme-ish "
            "but believable geometry: the camera looks from the side and slightly above at about 35 to 40 degrees, so the front label is "
            "still visible but clearly in perspective. Keep the entire bottle fully in frame. Make the side of the bottle noticeably "
            "visible, including glass thickness, curvature, shoulder shape, and side contour. The capsule and neck must show clear "
            "perspective. The label should not become unreadable. Change only the viewpoint, scene, lighting, shadows, reflections, "
            "and background while preserving the bottle and label."
        )

    is_hand_scene = view.scene == "human_hand"
    if view.angle:
        posture = (
            "Keep the bottle structurally straight, not bent or melted. "
            if is_hand_scene
            else "Keep the bottle standing naturally upright on its base, with a straight vertical axis. "
        )
        side = "left" if view.angle < 0 else "right"
        angle_instruction = (
            posture
            + f"Show a noticeable 20 to 25 degree three-quarter product view by changing only the bottle's orientation around its vertical axis. "
            + "Keep the shelf and surrounding scene in a normal shopper-facing view. This must not be a perfectly straight-on label view. "
            + "The front label must no longer look flat to the camera; its plane is clearly angled in perspective and not parallel to the camera sensor. "
            + f"The front label remains readable, but the {side} glass sidewall, shoulder curve, bottle thickness, "
            + "and side contour are clearly visible."
        )
    else:
        angle_instruction = "Keep the bottle standing naturally upright on its base, with the front label facing the camera."
    if is_hand_scene:
        angle_instruction += " Because it is hand-held, give the whole bottle a noticeable but natural diagonal lean of about 8 to 12 degrees while keeping the label unobstructed."
    return (
        "Create a realistic photo edit using the input wine bottle as the authoritative visual reference. "
        "Preserve the exact bottle identity from the reference image: glass shape, neck, cap, capsule, label size, "
        "label position, label colors, logo placement, typography layout, and visible text. "
        f"Place this same bottle {view.prompt_scene}. "
        f"{angle_instruction} "
        "Show exactly one complete bottle, centered, fully visible from top to bottom, with the main front label unobstructed. "
        "Change only the surrounding scene, camera framing, natural shadows, reflections, and lighting. "
        "Use photorealistic product-photo quality with normal perspective and enough label sharpness for visual search."
    )


def build_flux_prompt(view: SceneView) -> str:
    is_hand_scene = view.scene == "human_hand"
    if view.angle < 0:
        angle = "turned 25 degrees to the left around its vertical axis"
        side_detail = "left glass sidewall, shoulder curve, bottle thickness, and side contour"
    elif view.angle > 0:
        angle = "turned 25 degrees to the right around its vertical axis"
        side_detail = "right glass sidewall, shoulder curve, bottle thickness, and side contour"
    else:
        angle = "seen straight-on from the front"
        side_detail = "front label, bottle shoulders, and glass contours"

    scene_text = {
        "wine_shop_shelf": (
            "standing fully on a metal supermarket shelf in the alcoholic drinks aisle, set slightly back from the shelf edge with its entire base supported by the shelf, photographed like a close-up shopper phone photo while facing the product section, the bottle filling more than half of the frame height, beside other wine and spirits bottles, barcode shelf labels, price rails, promo tags, aisle signage, and a shopping cart behind the shelf area"
        ),
        "home_table": (
            "on a simple home dining table, with a quiet kitchen and everyday glassware softly blurred behind it"
        ),
        "human_hand": (
            "held in one visible hand only, fingers gripping below the label, realistic 750 ml scale against the hand, casual indoor room behind it"
        ),
    }[view.scene]
    framing_text = {
        "wine_shop_shelf": "The bottle is fully resting on the shelf, not hanging over the edge; it is centered and large in the frame, occupying roughly 55 to 70 percent of the image height, with the shelf seen from a natural shopper-facing angle.",
        "home_table": "The bottle is framed naturally at tabletop distance, like a casual phone photo at home.",
        "human_hand": "The bottle is framed naturally at hand-held distance, with realistic scale against the hand.",
    }[view.scene]
    lighting_text = {
        ("wine_shop_shelf", "bright"): "bright fluorescent supermarket lighting and practical overhead store lights",
        ("wine_shop_shelf", "dark"): "dimmer evening supermarket lighting, overhead fluorescent spill, and deeper shelf shadows",
        ("home_table", "bright"): "soft daylight from a nearby window and gentle natural shadows",
        ("home_table", "dark"): "low warm evening light from a nearby lamp and subdued room shadows",
        ("human_hand", "bright"): "soft bright indoor light and gentle natural shadows",
        ("human_hand", "dark"): "dim warm indoor light with controlled glass highlights",
    }[(view.scene, view.lighting)]
    hand_tilt = (
        " The bottle has a natural 10-degree diagonal hand-held lean."
        if is_hand_scene
        else ""
    )
    subject_pose = (
        "A three-quarter product photograph of the same reference wine bottle"
        if is_hand_scene
        else "A three-quarter product photograph of the same reference wine bottle"
    )
    return (
        f"{subject_pose}, {angle}, {scene_text}. "
        f"{framing_text} The label remains identifiable in perspective, with the {side_detail} visible and the bottle structurally straight. "
        f"{hand_tilt} The scene is lit by {lighting_text}, with realistic glass reflections, natural perspective, "
        "calm atmosphere, and product-photo clarity."
    )


def build_prompt(view: SceneView, pipeline_name: str) -> str:
    if pipeline_name == "Flux2KleinPipeline":
        return build_flux_prompt(view)
    return build_qwen_prompt(view)


def torch_dtype(dtype_name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_pipeline(
    model_id: str,
    pipeline_class: str,
    dtype_name: str,
    device_map: str,
    token: str | None,
    quantization: str,
    allow_tf32: bool,
    torch_compile: bool,
    compile_mode: str,
    disable_progress: bool,
) -> Any:
    import torch
    import diffusers

    if allow_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = torch_dtype(dtype_name)
    pipeline_type = getattr(diffusers, pipeline_class)
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    if quantization != "none":
        if pipeline_class != "Flux2KleinPipeline":
            raise ValueError("--quantization is currently supported only for Flux2KleinPipeline.")
        from diffusers import PipelineQuantizationConfig, TorchAoConfig
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig

        if quantization == "fp8wo":
            ao_config = Float8WeightOnlyConfig()
        elif quantization == "fp8dq":
            ao_config = Float8DynamicActivationFloat8WeightConfig()
        else:
            raise ValueError(f"Unsupported quantization: {quantization}")
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={"transformer": TorchAoConfig(ao_config)}
        )

    logger.info(
        "Loading %s with %s (%s, quantization=%s). This can take several minutes on the first run.",
        model_id,
        pipeline_class,
        dtype_name,
        quantization,
    )
    try:
        pipe = pipeline_type.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        pipe = pipeline_type.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    logger.info("Model weights loaded. Moving pipeline to %s.", device_map)

    if device_map == "cuda":
        pipe.to("cuda")
    elif device_map:
        maybe_offload = getattr(pipe, "enable_model_cpu_offload", None)
        if device_map == "cpu_offload" and callable(maybe_offload):
            maybe_offload()
        else:
            pipe.to(device_map)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=disable_progress)
    if torch_compile:
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise ValueError("--torch-compile requested, but pipeline has no transformer attribute.")
        logger.info("Compiling transformer with torch.compile(mode=%s). First generation will be slow.", compile_mode)
        pipe.transformer = torch.compile(transformer, mode=compile_mode, fullgraph=False)
    logger.info("Pipeline is ready.")
    return pipe


def call_pipeline(
    pipe: Any,
    image: Image.Image,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
) -> Image.Image:
    import torch

    generator = torch.Generator(device="cuda" if args.device_map == "cuda" else "cpu").manual_seed(seed)
    signature = inspect.signature(pipe.__call__)
    supported = set(signature.parameters)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    pipeline_name = type(pipe).__name__
    pipeline_image: Image.Image | list[Image.Image]
    if pipeline_name == "QwenImageEditPlusPipeline":
        pipeline_image = [image]
    else:
        pipeline_image = image
    negative_prompt = args.negative_prompt
    if negative_prompt is None and pipeline_name == "QwenImageEditPlusPipeline":
        negative_prompt = " "

    candidates: dict[str, Any] = {
        "image": pipeline_image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "strength": args.strength,
        "width": args.width,
        "height": args.height,
        "generator": generator,
        "num_images_per_prompt": 1,
    }
    kwargs = {
        key: value
        for key, value in candidates.items()
        if value is not None and (key in supported or accepts_kwargs)
    }

    if args.timings and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(**kwargs)
    image = result.images[0].convert("RGB")
    if args.timings and torch.cuda.is_available():
        torch.cuda.synchronize()
    if args.timings:
        logger.info("Timing pipeline_call=%.3fs", time.perf_counter() - started)
    return image


def call_pipeline_batch(
    pipe: Any,
    images: list[Image.Image],
    prompts: list[str],
    seeds: list[int],
    args: argparse.Namespace,
) -> list[Image.Image]:
    import torch

    if type(pipe).__name__ != "Flux2KleinPipeline":
        raise ValueError("--batch-size > 1 is supported only for Flux2KleinPipeline.")

    generator_device = "cuda" if args.device_map == "cuda" else "cpu"
    generators = [torch.Generator(device=generator_device).manual_seed(seed) for seed in seeds]
    signature = inspect.signature(pipe.__call__)
    supported = set(signature.parameters)
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())

    candidates: dict[str, Any] = {
        "image": images,
        "prompt": prompts,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "generator": generators,
        "num_images_per_prompt": 1,
    }
    kwargs = {
        key: value
        for key, value in candidates.items()
        if value is not None and (key in supported or accepts_kwargs)
    }

    with torch.inference_mode():
        result = pipe(**kwargs)
    return [image.convert("RGB") for image in result.images]


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)
    elapsed = time.perf_counter() - started
    if elapsed > 0.1:
        logger.debug("JPEG save took %.3fs: %s", elapsed, path)


def negative_prompt_for_metadata(args: argparse.Namespace, pipe: Any) -> str | None:
    pipeline_name = type(pipe).__name__
    if args.negative_prompt is not None:
        return args.negative_prompt
    if pipeline_name == "QwenImageEditPlusPipeline":
        return " "
    return None


def metadata_for(
    source: Path,
    model_id: str,
    view: SceneView,
    seed: int,
    prompt: str,
    negative_prompt: str | None,
) -> dict[str, Any]:
    return {
        "source_image": str(source),
        "model_id": model_id,
        "view_index": view.index,
        "view_slug": view.slug,
        "scene": view.scene,
        "lighting": view.lighting,
        "angle_degrees": view.angle,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }


def output_path_for(output_dir: Path, wine_id: str, output_prefix: str, view: SceneView) -> Path:
    dirname = f"{output_prefix}_{view.index:02d}_{view.slug}"
    return output_dir / wine_id / dirname / "original.jpg"


def run_generation(args: argparse.Namespace, scene_views: tuple[SceneView, ...] = SCENE_VIEWS) -> None:
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)

    source_paths = iter_main_originals(args.input_dir)
    requested_wine_ids = load_requested_wine_ids(args)
    if requested_wine_ids is not None:
        source_paths = [
            source_path
            for source_path in source_paths
            if wine_id_from_source(source_path) in requested_wine_ids
        ]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        source_paths = [
            source_path
            for index, source_path in enumerate(source_paths)
            if index % args.num_shards == args.shard_index
        ]
    if args.start_index:
        source_paths = source_paths[args.start_index :]
    if args.limit is not None:
        source_paths = source_paths[: args.limit]

    if not source_paths:
        logger.warning("No source images found under %s/*/main/original.*", args.input_dir)
        return

    random.seed(args.seed)
    pipe = load_pipeline(
        model_id=args.model_id,
        pipeline_class=args.pipeline_class,
        dtype_name=args.dtype,
        device_map=args.device_map,
        token=args.hf_token,
        quantization=args.quantization,
        allow_tf32=args.allow_tf32,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        disable_progress=args.disable_progress,
    )
    pipeline_name = type(pipe).__name__
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1 and pipeline_name != "Flux2KleinPipeline":
        raise ValueError("--batch-size > 1 is supported only for Flux2KleinPipeline.")

    total = len(source_paths) * len(scene_views)
    done = 0
    for image_index, source_path in enumerate(source_paths, start=args.start_index):
        wine_id = wine_id_from_source(source_path)
        source_image = fit_image_to_canvas(
            load_image(source_path, max_side=args.max_side),
            width=args.width,
            height=args.height,
        )
        try:
            batch: list[dict[str, Any]] = []

            def flush_batch() -> None:
                nonlocal done
                if not batch:
                    return
                logger.info("Generate Flux batch size=%s", len(batch))
                generated_images = call_pipeline_batch(
                    pipe,
                    images=[item["image"] for item in batch],
                    prompts=[item["prompt"] for item in batch],
                    seeds=[item["seed"] for item in batch],
                    args=args,
                )
                for item, generated in zip(batch, generated_images, strict=True):
                    save_jpeg(generated, item["target_path"], quality=args.quality)
                    item["image"].close()
                    item["metadata_path"].write_text(
                        json.dumps(
                            metadata_for(
                                item["source_path"],
                                args.model_id,
                                item["view"],
                                item["seed"],
                                item["prompt"],
                                negative_prompt_for_metadata(args, pipe),
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    done += 1
                    logger.info("Saved %s (%s/%s)", item["target_path"], done, total)
                batch.clear()

            for view in scene_views:
                target_path = output_path_for(args.output_dir, wine_id, args.output_prefix, view)
                metadata_path = target_path.parent / "metadata.json"
                if target_path.exists() and not args.overwrite:
                    done += 1
                    logger.info("Skip existing %s (%s/%s)", target_path, done, total)
                    continue

                view_seed = args.seed + image_index * 1000 + view.index
                prompt = build_prompt(view, pipeline_name=type(pipe).__name__)
                logger.info("Generate %s view=%s seed=%s", wine_id, view.slug, view_seed)
                if args.batch_size > 1:
                    batch.append(
                        {
                            "source_path": source_path,
                            "image": source_image.copy(),
                            "view": view,
                            "target_path": target_path,
                            "metadata_path": metadata_path,
                            "seed": view_seed,
                            "prompt": prompt,
                        }
                    )
                    if len(batch) >= args.batch_size:
                        flush_batch()
                    continue

                total_started = time.perf_counter()
                generated = call_pipeline(pipe, source_image, prompt, view_seed, args)
                save_started = time.perf_counter()
                save_jpeg(generated, target_path, quality=args.quality)
                if args.timings:
                    logger.info(
                        "Timing save=%.3fs total_item=%.3fs",
                        time.perf_counter() - save_started,
                        time.perf_counter() - total_started,
                    )
                metadata_path.write_text(
                    json.dumps(
                        metadata_for(
                            source_path,
                            args.model_id,
                            view,
                            view_seed,
                            prompt,
                            negative_prompt_for_metadata(args, pipe),
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                done += 1
                logger.info("Saved %s (%s/%s)", target_path, done, total)
            flush_batch()
        finally:
            source_image.close()


def main_for_model(
    description: str,
    model_id: str,
    output_prefix: str,
    pipeline_class: str,
    default_num_inference_steps: int,
    default_guidance_scale: float,
    default_true_cfg_scale: float | None = None,
    scene_views: tuple[SceneView, ...] = SCENE_VIEWS,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_generation(
        args=parse_common_args(
            description,
            model_id=model_id,
            output_prefix=output_prefix,
            pipeline_class=pipeline_class,
            default_num_inference_steps=default_num_inference_steps,
            default_guidance_scale=default_guidance_scale,
            default_true_cfg_scale=default_true_cfg_scale,
        ),
        scene_views=scene_views,
    )
