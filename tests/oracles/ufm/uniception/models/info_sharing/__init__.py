# Trimmed from https://github.com/castacks/UniCeption @ ee7fa0b (uniception/models/info_sharing/__init__.py) for the UFM parity test.
# Copyright (c) 2024, AirLab Stacks. SPDX-License-Identifier: BSD-3-Clause (see LICENSE_UniCeption in this directory).
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.global_attention_transformer import (
    MultiViewGlobalAttentionTransformer,
    MultiViewGlobalAttentionTransformerIFR,
)

INFO_SHARING_CLASSES = {
    "global_attention": (
        MultiViewGlobalAttentionTransformer,
        MultiViewGlobalAttentionTransformerIFR,
    ),
}
