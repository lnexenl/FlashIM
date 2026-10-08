# Vendored from https://github.com/Parskatt/DeDoDe (DeDoDe/transformer/__init__.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2023 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.
# Note: DeDoDe/transformer/ is DINOv2 code (Meta, Apache-2.0: LICENSE_DINOv2); files keep their own Meta headers.

import torch
import torch.nn as nn
import torch.nn.functional as F

from DeDoDe.utils import get_grid
from .layers.block import Block
from .layers.attention import MemEffAttention
from .dinov2 import vit_large