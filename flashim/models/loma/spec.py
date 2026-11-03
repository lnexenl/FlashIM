"""LoMa's pipelines ``loma-<variant>`` (b128 | b | l | g | r; ``loma`` = ``loma-b``, upstream's default):
``loma-extractor`` + ``loma-matcher`` of the same variant in a :class:`~flashim.models.lightglue.pipeline.SparsePipeline`
(the extractor's pixel-center keypoints and original image size are what the matcher normalizes itself, so no
coordinate hand-off is needed). A pipeline's preset is both components'; ``components.extractor.preset`` /
``components.matcher.preset`` set one of them."""

from __future__ import annotations

import dataclasses

from flashim.build import PRECISIONS, ModelSpec
from flashim.models.lightglue.spec import _compose
from flashim.models.loma.extractor_spec import PASSES as EXTRACTOR_PASSES
from flashim.models.loma.matcher import VARIANTS

# bf16, upstream's own default precision (matcher and extractor autocast): measured on 20 MegaDepth-1500 pairs (1024 px,
# 4096 keypoints, benchmarks/accuracy/fidelity.py) it keeps 91.2-91.7% of the fp32 matches, int8 85.7-87.1% and fp8
# 80.0-84.3%. int8 (every pass of both components) is MegaDepth-1500-gated against bf16 on sm_89 / sm_90 (1024 px,
# 4096 keypoints, OpenCV RANSAC 0.5 px, 10 seeds, paired bootstrap): AUC@5 L20 -0.10 .. -0.30, H20 -0.09 .. +0.23 over the
# five variants, for a 1.68-1.77x (L20) / 1.53-1.65x (H20) faster eager pair (2 extracts from the JPEG + match).
DEFAULT_PRESET = "bf16"
NAMES = {"b128": "B128", "b": "B", "l": "L", "g": "G", "r": "R (rotation-augmented B)"}
# Per-variant extractor overlays of the int8 pipeline. loma-l keeps the bit-exact stem (no tensor-core first conv, no
# patch-embedding GEMM) and SageAttention on ViT blocks 4-23 only: with every int8 pass, MegaDepth-1500 AUC@5 on the
# L20 was 0.51 below bf16 in the prototypes (the gate is 0.5); with this overlay the library preset is -0.30 [-0.80, +0.22]
# on the L20 and +0.23 [-0.26, +0.72] on the H20, at 82.5 / 77.0 ms per pair.
INT8_EXTRACTOR = {
    "l": {"optimize": {
        "passes": [p for p in EXTRACTOR_PASSES if p not in ("tensor_core_first_conv", "patch_embed_gemm")],
        "options": {"vit_attention": {"layers": "4-23"}},
    }},
}


def _pipeline(variant: str) -> ModelSpec:
    return ModelSpec(
        name=f"loma-{variant}",
        role="pipeline",
        description=f"LoMa-{NAMES[variant]}: DaD keypoints + LoMa DeDoDe descriptors + LoMa matcher",
        load=None,
        presets={
            preset: {
                "components": {
                    "extractor": {"model": "loma-extractor", "preset": preset, "variant": variant,
                                  **(INT8_EXTRACTOR.get(variant, {}) if preset == "int8" else {})},
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
