# Vendored from https://github.com/Parskatt/DeDoDe (DeDoDe/transformer/layers/__init__.py) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2023 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.
# Note: DeDoDe/transformer/ is DINOv2 code (Meta, Apache-2.0: LICENSE_DINOv2); files keep their own Meta headers.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .dino_head import DINOHead
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .block import NestedTensorBlock
from .attention import MemEffAttention
