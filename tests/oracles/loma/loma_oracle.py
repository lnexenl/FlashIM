"""Builds the vendored upstream LoMa (``loma/``) from a local release checkpoint, without any download.

Not upstream code (FlashIM's, MIT like the rest of the repository). Put this directory on ``sys.path`` and::

    from loma_oracle import load_oracle
    oracle = load_oracle("b", "~/.cache/torch/hub/checkpoints/loma_B.pt")
    kpts_a, kpts_b = oracle.match("assets/0015_A.jpg", "assets/0015_B.jpg")  # pixel-edge coordinates

Upstream's constructors fetch ``dad.pth`` (``DaD.__init__``) and ``dinov2_vitl14_pretrain.pth``
(``FrozenDINOv2.__init__``) with ``torch.hub.load_state_dict_from_url`` and the variant checkpoint when
``weights_url`` is set. Here ``weights_url`` is cleared, ``torch.hub.load_state_dict_from_url`` answers the
two sub-model URLs from the variant checkpoint itself (its ``_detector.*`` is bit-equal to ``dad.pth``, its
DINOv2 is ``dinov2_vitl14_pretrain.pth`` rounded to BF16) and refuses any other URL, and the checkpoint is
then loaded with upstream's own checks (no missing keys, unexpected keys only for matcher layers
``>= n_layers``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest import mock

import torch

VARIANTS = {"b128": "LoMaB128", "b": "LoMaB", "l": "LoMaL", "g": "LoMaG", "r": "LoMaR"}
_DINOV2 = "_descriptor.encoder.frozen_dinov2.dinov2_vitl14."


def _strip(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_oracle(variant: str, checkpoint: str | Path, *, mp: bool = True, extractor_amp: bool = True, **cfg):
    """Upstream ``LoMa(LoMa<variant>())`` on ``loma.device.device`` (CUDA when available), in eval mode.

    ``mp``: the matcher's autocast (``LoMa.Cfg.mp``, upstream default on; dtype ``loma.device.amp_dtype``:
    BF16 on sm_80+). ``extractor_amp``: upstream hard-codes autocast in the DaD / DeDoDe nets and keeps
    DINOv2 in that dtype (independent of ``mp``); ``False`` turns every ``amp`` flag off and runs DINOv2 in
    FP32 on the checkpoint's values, the FP32 reference. ``cfg``: further ``LoMa.Cfg`` fields
    (``num_keypoints``, ...)."""
    import loma.loma as upstream

    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
    sub_models = {
        "/dad.pth": _strip(state, "_detector."),
        "/dinov2_vitl14_pretrain.pth": _strip(state, _DINOV2),
    }

    def from_checkpoint(url, *args, **kwargs):
        for suffix, weights in sub_models.items():
            if url.endswith(suffix) and weights:
                return {k: v.float() for k, v in weights.items()}
        raise RuntimeError(f"the LoMa oracle does not download ({url})")

    config = dataclasses.replace(getattr(upstream, VARIANTS[variant])(), weights_url=None, mp=mp, **cfg)
    with mock.patch.object(torch.hub, "load_state_dict_from_url", from_checkpoint):
        model = upstream.LoMa(config)
    if not extractor_amp:
        for module in model.modules():
            if hasattr(module, "amp"):
                module.amp = False
            if hasattr(module, "dinov2_vitl14"):
                module.dinov2_vitl14.float()
                module.amp_dtype = torch.float32
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, missing
    extra = [k for k in unexpected if not model._is_unexpected_extra_layer_key(k, config.n_layers)]
    assert not extra, extra
    return model.eval()
