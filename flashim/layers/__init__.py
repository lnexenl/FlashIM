"""Model building blocks that FlashIM's passes recognize by type.

A model assembled from these layers gets, without model-specific code, what the quantization and fusion passes
know how to do with them: quantized linears with fused producers (:class:`TransformerBlock`), fused separable conv
blocks (:class:`SeparableConvBlock`), the fused quantized deformable encoder (:class:`DeformableEncoder`). Parameter
names follow the common upstream conventions (timm / Deformable DETR) so checkpoints load strictly; the passes rely
on the layer types, never on attribute-name guessing.
"""

from flashim.layers.conv import IndexedSeparableConvBlock, SeparableConvBlock
from flashim.layers.deformable import DeformableEncoder, DeformableEncoderLayer, MSDeformAttn
from flashim.layers.transformer import Attention, LayerScale, LinearKMaskedBias, Mlp, TransformerBlock

__all__ = [
    "Attention",
    "DeformableEncoder",
    "DeformableEncoderLayer",
    "IndexedSeparableConvBlock",
    "LayerScale",
    "LinearKMaskedBias",
    "MSDeformAttn",
    "Mlp",
    "SeparableConvBlock",
    "TransformerBlock",
]
