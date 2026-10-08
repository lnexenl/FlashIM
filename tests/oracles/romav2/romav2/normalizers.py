# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/normalizers.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

import torch

_CONSTANTS: dict[tuple[torch.device, tuple[float, ...]], torch.Tensor] = {}


def _constants(img: torch.Tensor, values: tuple[float, ...]) -> torch.Tensor:
    """Return cached device constants so graph replay has no host-to-device copy."""
    key = (img.device, values)
    constants = _CONSTANTS.get(key)
    if constants is None:
        constants = torch.tensor(values, device=img.device)
        _CONSTANTS[key] = constants
    return constants


def imagenet(img: torch.Tensor) -> torch.Tensor:
    imagenet_mean = _constants(img, (0.485, 0.456, 0.406))
    imagenet_std = _constants(img, (0.229, 0.224, 0.225))
    return (img - imagenet_mean[None, :, None, None]) / imagenet_std[
        None, :, None, None
    ]


def inception(img: torch.Tensor) -> torch.Tensor:
    inception_mean = _constants(img, (0.5, 0.5, 0.5))
    inception_std = _constants(img, (0.5, 0.5, 0.5))
    return (img - inception_mean[None, :, None, None]) / inception_std[
        None, :, None, None
    ]
