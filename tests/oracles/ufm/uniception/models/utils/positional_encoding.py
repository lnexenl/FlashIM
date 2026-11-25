# Vendored from https://github.com/castacks/UniCeption @ ee7fa0b (uniception/models/utils/positional_encoding.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2024, AirLab Stacks. SPDX-License-Identifier: BSD-3-Clause (see LICENSE_UniCeption in this directory).
# Changes: none besides this header.

"""
Helper function for positional encoding in UniCeption
"""

import torch


class PositionGetter(object):
    "Helper class to return positions of patches."

    def __init__(self):
        "Initialize the position getter."
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        "Get the positions for a given batch size, height, and width. Uses caching."
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()

        return pos
