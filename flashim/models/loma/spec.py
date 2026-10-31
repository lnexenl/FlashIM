"""LoMa's pipelines ``loma-<variant>`` (b128 | b | l | g | r; ``loma`` = ``loma-b``, upstream's default):
``loma-extractor`` + ``loma-matcher`` of the same variant in a :class:`~flashim.models.lightglue.pipeline.SparsePipeline`
(the extractor's pixel-center keypoints and original image size are what the matcher normalizes itself, so no
coordinate hand-off is needed). A pipeline's preset is both components'; ``components.extractor.preset`` /
``components.matcher.preset`` set one of them."""

from __future__ import annotations

import dataclasses

from flashim.build import PRECISIONS, ModelSpec
from flashim.models.lightglue.spec import _compose
from flashim.models.loma.matcher import VARIANTS

# bf16, upstream's own default precision (matcher and extractor autocast): measured on 20 MegaDepth-1500 pairs (1024 px,
# 4096 keypoints, benchmarks/accuracy/fidelity.py) it keeps 91.2-91.7% of the fp32 matches, int8 85.7-87.1% and fp8
# 80.0-84.3%, for a 1.13-1.2x (int8) / 1.5-1.6x (fp8) faster eager pair (loma-b / loma-g); the quantized presets wait
# for the MegaDepth-1500 AUC gate (results/loma/README.md).
DEFAULT_PRESET = "bf16"
NAMES = {"b128": "B128", "b": "B", "l": "L", "g": "G", "r": "R (rotation-augmented B)"}


def _pipeline(variant: str) -> ModelSpec:
    return ModelSpec(
        name=f"loma-{variant}",
        role="pipeline",
        description=f"LoMa-{NAMES[variant]}: DaD keypoints + LoMa DeDoDe descriptors + LoMa matcher",
        load=None,
        presets={
            preset: {
                "components": {
                    "extractor": {"model": "loma-extractor", "preset": preset, "variant": variant},
                    "matcher": {"model": "loma-matcher", "preset": preset, "variant": variant},
                }
            }
            for preset in PRECISIONS
        },
        default_preset=DEFAULT_PRESET,
        compose=_compose,
    )


PIPELINES = {variant: _pipeline(variant) for variant in VARIANTS}
LOMA_B128, LOMA_B, LOMA_L, LOMA_G, LOMA_R = (PIPELINES[v] for v in ("b128", "b", "l", "g", "r"))
LOMA = dataclasses.replace(LOMA_B, name="loma", description=f"{LOMA_B.description} (= loma-b)")
