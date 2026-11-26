"""RIPE++ checkpoints (``ripe++.ckpt``, ``ripe++_tokyo_megadepth.ckpt``, ``ripe++_scared.ckpt``) and the weakly
supervised RIPE++ LightGlue matcher checkpoint (``ripe++lightglue.tar``).

Host note:
- ``ripe++.ckpt`` (default MegaDepth) and ``ripe++lightglue.tar`` are hosted on the authors' Hugging Face space
  ``JohannesK14/RIPE`` (identical SHA-256 to ``https://cvg.hhi.fraunhofer.de/RIPE++/``) and download over standard TLS.
- ``ripe++_tokyo_megadepth.ckpt`` (``aachen``) and ``ripe++_scared.ckpt`` (``scared``) are only hosted on
  ``https://cvg.hhi.fraunhofer.de/RIPE%2B%2B/``. That server omits the ``GEANT TLS ECC 1`` intermediate certificate
  (``http://crt.harica.gr/HARICA-GEANT-TLS-E1.cer``, chaining to ``HARICA TLS ECC Root CA 2021``), so TLS verification
  fails on machines without that intermediate in their trust store. FlashIM never disables TLS verification; supply a
  verified local copy via ``checkpoint=``, ``$FLASHIM_RIPEPP_CHECKPOINT``, or ``$TORCH_HOME/hub/checkpoints/``.

Loading note:
- All four upstream archives store plain Python dicts (``OmegaConf.to_container(..., resolve=True)``) and standard
  PyTorch tensor storages (``collections.OrderedDict``, ``torch.FloatStorage``, ``torch.LongStorage``,
  ``torch._utils._rebuild_tensor_v2``), with no OmegaConf / Hydra pickle classes. They are therefore loaded strictly
  with ``torch.load(..., weights_only=True)`` (never ``weights_only=False``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from flashim.models.weights_util import Checkpoint, find_weights

_HF_BASE = "https://huggingface.co/spaces/JohannesK14/RIPE/resolve/main/weights"
_HHI_BASE = "https://cvg.hhi.fraunhofer.de/RIPE%2B%2B"

CHECKPOINTS = {
    "default": Checkpoint(
        "ripe++.ckpt",
        f"{_HF_BASE}/ripe++.ckpt",
        "709ebb1f000914d45c08e3edf64727993ee339519e2174231de86954efb95577",
    ),
    "aachen": Checkpoint(
        "ripe++_tokyo_megadepth.ckpt",
        f"{_HHI_BASE}/ripe++_tokyo_megadepth.ckpt",
        "99ea66bab7db563dc9c177e3b9a612196fccdf7166d3bdb853a7f1e70a399def",
    ),
    "scared": Checkpoint(
        "ripe++_scared.ckpt",
        f"{_HHI_BASE}/ripe++_scared.ckpt",
        "d5df1afe185a7abf7b33be436633010b01ed12ce221031e7e5368c2453146eb8",
    ),
}
VARIANT_ALIASES = {"aachen_day_night": "aachen"}

LIGHTGLUE_CHECKPOINT = Checkpoint(
    "ripe++lightglue.tar",
    f"{_HF_BASE}/ripe++lightglue.tar",
    "61c48c4649339dfa550805d8ca1d6b269c67a0ee525d8daf4bc0c18d2d2db0cf",
)

LICENSES = {
    "code": (
        "RIPE++ (fraunhoferhhi/RIPEpp) Software Copyright License for Academic Use v2.0 (non-commercial; "
        "upstream partly based on DALF Apache-2.0, DeDoDe MIT, DISK Apache-2.0; matcher in JohannesK14/glue-factory "
        "Apache-2.0); FlashIM re-implementation"
    ),
    "weights": (
        "ripe++.ckpt, ripe++_tokyo_megadepth.ckpt, ripe++_scared.ckpt, ripe++lightglue.tar "
        "(RIPE++, LIMIT@ECCV 2026; Fraunhofer Academic Use License v2.0 — non-commercial use only)"
    ),
}

_HHI_TLS_HINT = (
    " (Note: cvg.hhi.fraunhofer.de omits the 'GEANT TLS ECC 1' intermediate certificate "
    "[http://crt.harica.gr/HARICA-GEANT-TLS-E1.cer], causing standard TLS verification to fail; "
    "download the file with that intermediate certificate and pass checkpoint= or set $FLASHIM_RIPEPP_CHECKPOINT)"
)


def resolve_variant(variant: str) -> str:
    canonical = VARIANT_ALIASES.get(variant, variant)
    if canonical not in CHECKPOINTS:
        choices = sorted((*CHECKPOINTS, *VARIANT_ALIASES))
        raise ValueError(f"Unknown RIPE++ variant {variant!r}; choose from {choices}")
    return canonical


def load_state_dict(
    variant: str = "default",
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load the extractor ``state_dict`` using ``torch.load(..., weights_only=True)``."""
    canonical = resolve_variant(variant)
    checkpoint = CHECKPOINTS[canonical]
    try:
        file = find_weights(checkpoint, path=path, env="FLASHIM_RIPEPP_CHECKPOINT")
    except FileNotFoundError as exc:
        if checkpoint.url.startswith(_HHI_BASE) and path is None:
            raise FileNotFoundError(f"{exc}{_HHI_TLS_HINT}") from exc
        raise
    raw = torch.load(file, map_location=map_location, weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def load_lightglue_state_dict(
    num_heads: int,
    num_layers: int,
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load and convert the RIPE++ weakly-supervised LightGlue matcher weights from ``ripe++lightglue.tar``."""
    from flashim.models.lightglue.weights import convert_state_dict

    file = find_weights(LIGHTGLUE_CHECKPOINT, path=path, env="FLASHIM_RIPEPP_LIGHTGLUE_CHECKPOINT")
    raw = torch.load(file, map_location=map_location, weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    matcher_state = {
        k.removeprefix("matcher."): v
        for k, v in state.items()
        if not k.startswith("extractor.")
    }
    return convert_state_dict(matcher_state, num_heads, num_layers)
