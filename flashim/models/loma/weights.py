"""LoMa checkpoints (davnords/storage release ``loma``): one file per variant holding the matcher and its
frozen DaD detector (``_detector.*``) and DeDoDe descriptor (``_descriptor.*``, with DINOv2 ViT-L/14 for every
variant but B128).

Measured on the releases: ``_detector.*`` is bit-equal to ``dad.pth``; the descriptors are LoMa's own (none is
DeDoDe's release), B / L / G share one DeDoDe-G descriptor, B128 has a 128-dim DeDoDe-B one and R a rotation
trained DeDoDe-G one; the DINOv2 copy is ``dinov2_vitl14_pretrain.pth`` rounded to BF16 (stored as BF16).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import torch

from flashim.models.weights_util import Checkpoint, find_weights

RELEASE = "https://github.com/davnords/storage/releases/download/loma"
CHECKPOINTS = {
    "b128": Checkpoint(
        "loma_B128.pth", f"{RELEASE}/loma_B128.pth",
        "b509cb295b543e6eae37dea76185e805c4e87ff7e35e0a84cad301be1b5984fc",
    ),
    "b": Checkpoint(
        "loma_B.pt", f"{RELEASE}/loma_B.pt",
        "3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758",
    ),
    "l": Checkpoint(
        "loma_L.pth", f"{RELEASE}/loma_L.pth",
        "d0ae981176d935c5fec0dabe48334fb841c7d6348ae609560d1aae79628e12b9",
    ),
    "g": Checkpoint(
        "loma_G.pth", f"{RELEASE}/loma_G.pth",
        "9d925dc9861fb8e52489a1d2d2354780d5d38e34cbe0c6b10830d0ebe6055941",
    ),
    "r": Checkpoint(
        "loma_R.pth", f"{RELEASE}/loma_R.pth",
        "ade1379fb52143da65c36730f24b69eaa49618bb277890c6d2a0fb1630d33d67",
    ),
}
LICENSES = {
    "code": "MIT (davnords/LoMa) except the matcher, derived from cvg/LightGlue: Apache-2.0; DaD / DeDoDe "
    "MIT; DINOv2 Apache-2.0; FlashIM re-implementation",
    "weights": "LoMa release checkpoints loma_{B128,B,L,G,R} (matcher + frozen DaD detector + LoMa-trained "
    "DeDoDe descriptor, DINOv2 ViT-L/14 inside for B / L / G / R)",
}

_DINOV2 = "encoder.frozen_dinov2.dinov2_vitl14."
_LAYER = re.compile(r"^(transformers|log_assignment)\.(\d+)\.")


class LoMaState(NamedTuple):
    """A checkpoint split by module, each in the layout of the matching upstream release file.

    ``matcher``: LoMa's keys (``input_proj`` only when the input and embedding widths differ, ``posenc.Wr``,
    ``transformers.N.*`` with upstream's interleaved ``Wqkv`` rows ``[head, dim, (q, k, v)]``,
    ``log_assignment.N.{final_proj,matchability}``). ``detector``: ``dad.pth``'s. ``descriptor``:
    ``dedode_descriptor_{B,G}.pth``'s (DINOv2 excluded). ``dinov2``: ``dinov2_vitl14_pretrain.pth``'s, upcast
    to FP32 (exact: the stored copy is BF16), ``None`` for B128."""

    matcher: dict[str, torch.Tensor]
    detector: dict[str, torch.Tensor]
    descriptor: dict[str, torch.Tensor]
    dinov2: dict[str, torch.Tensor] | None


def find_checkpoint(variant: str, path: str | Path | None = None) -> Path:
    return find_weights(CHECKPOINTS[variant], path=path, env=f"FLASHIM_LOMA_{variant.upper()}_CHECKPOINT")


def split_state_dict(state: dict[str, torch.Tensor], n_layers: int = 9) -> LoMaState:
    """Upstream ``LoMa.state_dict()`` -> :class:`LoMaState`; matcher layers ``>= n_layers`` are dropped
    (upstream loads with ``strict=False`` and ignores them; the releases hold exactly 9)."""
    matcher, detector, descriptor, dinov2 = {}, {}, {}, {}
    for key, value in state.items():
        if key.startswith("_detector."):
            detector[key.removeprefix("_detector.")] = value
        elif key.startswith(f"_descriptor.{_DINOV2}"):
            dinov2[key.removeprefix(f"_descriptor.{_DINOV2}")] = value.float()
        elif key.startswith("_descriptor."):
            descriptor[key.removeprefix("_descriptor.")] = value
        elif (layer := _LAYER.match(key)) is None or int(layer.group(2)) < n_layers:
            matcher[key] = value
    return LoMaState(matcher, detector, descriptor, dinov2 or None)


def load_state_dicts(
    variant: str, n_layers: int = 9, path: str | Path | None = None, map_location="cpu"
) -> LoMaState:
    state = torch.load(find_checkpoint(variant, path), map_location=map_location, weights_only=True)
    return split_state_dict(state, n_layers)
