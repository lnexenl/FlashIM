"""Build the RaCo + ALIKED extractor."""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.engines.compile import compile_regions
from flashim.models.aliked.build import load_aliked
from flashim.models.raco.config import RaCoConfig
from flashim.models.raco.extractor import RaCoALIKED
from flashim.models.raco.fuse import fold_batchnorm
from flashim.models.raco.model import RaCo

PRESETS = ("fp32", "bf16")
COMPILE_REGIONS = ("block1", "block2", "block3", "block4", "ranker_head")


def load_raco_aliked(
    preset: str = "fp32",
    *,
    engine: str = "eager",
    device: torch.device | str | None = None,
    raco_checkpoint: str | Path | None = None,
    raco_config: RaCoConfig | None = None,
    aliked_kwargs: dict | None = None,
    fused_ranker: bool = True,
) -> RaCoALIKED:
    """RaCo keypoints + ALIKED (``aliked-n16``) descriptors. ``preset`` sets the ALIKED precision; RaCo's detector
    trunk and ranker follow it (the ranker as fused Triton blocks when ``fused_ranker``, CUDA only) while its score
    head and NMS/soft-argmax stay FP32."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {PRESETS}")
    raco = RaCo.from_pretrained(raco_config, checkpoint=raco_checkpoint, device=device)
    raco.fused_batchnorm = fold_batchnorm(raco)
    if preset == "bf16":
        raco.set_compute_dtype(torch.bfloat16)
        if fused_ranker and raco.device.type == "cuda":
            raco.use_fused_ranker()
    if engine == "compile":
        compile_regions(raco, COMPILE_REGIONS)
    aliked = load_aliked(preset, engine=engine, device=device, **(aliked_kwargs or {}))
    extractor = RaCoALIKED(raco, aliked)
    extractor.preset, extractor.engine = preset, engine
    return extractor
