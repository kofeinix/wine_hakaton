from __future__ import annotations

from generate_scene_views_common import main_for_model


MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
OUTPUT_PREFIX = "flux_generated"
PIPELINE_CLASS = "Flux2KleinPipeline"


def main() -> None:
    main_for_model(
        description="Generate wine bottle scene views with black-forest-labs/FLUX.2-klein-9B.",
        model_id=MODEL_ID,
        output_prefix=OUTPUT_PREFIX,
        pipeline_class=PIPELINE_CLASS,
        default_num_inference_steps=4,
        default_guidance_scale=1.0,
    )


if __name__ == "__main__":
    main()
