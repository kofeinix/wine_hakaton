from __future__ import annotations

from generate_scene_views_common import SceneView, main_for_model


MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
OUTPUT_PREFIX = "qwen_generated"
PIPELINE_CLASS = "QwenImageEditPlusPipeline"
QWEN_SCENE_VIEWS = (
    SceneView(
        1,
        "store_bad_phone",
        "qwen_store_bad_phone",
        "mixed",
        0,
        "",
    ),
    SceneView(
        2,
        "table_extreme_geometry",
        "qwen_table_extreme_geometry",
        "natural",
        40,
        "",
    ),
)


def main() -> None:
    main_for_model(
        description="Generate wine bottle scene views with Qwen/Qwen-Image-Edit-2511.",
        model_id=MODEL_ID,
        output_prefix=OUTPUT_PREFIX,
        pipeline_class=PIPELINE_CLASS,
        default_num_inference_steps=40,
        default_guidance_scale=1.0,
        default_true_cfg_scale=4.0,
        scene_views=QWEN_SCENE_VIEWS,
    )


if __name__ == "__main__":
    main()
