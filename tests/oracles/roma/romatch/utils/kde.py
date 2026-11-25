# Vendored from https://github.com/Parskatt/RoMa (romatch/utils/kde.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2023 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.
# Note: romatch/models/transformer/ is DINOv2 code (Meta, Apache-2.0: LICENSE_DINOv2); files keep their own Meta headers.

import torch


def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density