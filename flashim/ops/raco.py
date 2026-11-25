"""RaCo-specific ops: the ranker's residual block and 5x5 head on padded NHWC BF16 activations.

The ranker keeps its activation channels-last with the channel dim zero-padded to 16 (``C_IN`` = 4 for the RGB
input), which is the layout the fused kernels read and write; padding channels are always zero. Weights are the
BatchNorm-folded module parameters (``weights=`` operands: impls repack them once via :meth:`Op.bind`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.ops._registry import Op

WIDTH = 16  # padded channels of the ranker's activations


def _to_nchw(x: Tensor, channels: int) -> Tensor:
    return x[..., :channels].permute(2, 0, 1)[None].float()


def _resblock_reference(
    x: Tensor, cin: int, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, wm: Tensor, bm: Tensor
) -> Tensor:
    """``selu(conv3x3(selu(conv3x3(x) + b1)) + b2 + conv1x1(x) + bm)`` in FP32.

    ``x``: NHWC BF16 ``[H, W, C_IN]`` whose first ``cin`` channels are valid; ``w1`` ``[C, cin, 3, 3]``, ``w2``
    ``[C, C, 3, 3]``, ``wm`` ``[C, cin, 1, 1]`` (``C <= 16``), biases ``[C]``. Returns NHWC BF16 ``[H, W, 16]``,
    channels ``>= C`` zero.
    """
    h, w, _ = x.shape
    inp = _to_nchw(x, cin)
    hidden = F.selu(F.conv2d(inp, w1.float(), b1.float(), padding=1))
    out = F.selu(F.conv2d(hidden, w2.float(), b2.float(), padding=1) + F.conv2d(inp, wm.float(), bm.float()))
    padded = torch.zeros(h, w, WIDTH, device=x.device, dtype=torch.bfloat16)
    padded[..., : out.shape[1]] = out[0].permute(1, 2, 0)
    return padded


raco_resblock = Op("raco_resblock", _resblock_reference, weights=6)


def _head_reference(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """5x5 conv ``C -> 1`` with reflect padding in FP32: ``x`` NHWC BF16 ``[H, W, 16]``, ``weight`` ``[1, C, 5, 5]``,
    ``bias`` ``[1]``. Returns FP32 ``[1, 1, H, W]``."""
    inp = F.pad(_to_nchw(x, weight.shape[1]), (2, 2, 2, 2), mode="reflect")
    return F.conv2d(inp, weight.float(), bias.float())


raco_head5x5 = Op("raco_head5x5", _head_reference, weights=2)


def _score_head_reference(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    """RaCo detector ``score_head[2:]`` in FP32: ``conv3x3(8->4) -> SELU -> conv3x3(4->4) -> SELU -> conv3x3(4->1)``.

    ``x``: ``[B, 8, H, W]`` FP32; ``w1`` ``[4, 8, 3, 3]``, ``w2`` ``[4, 4, 3, 3]``, ``w3`` ``[1, 4, 3, 3]`` ->
    ``[B, 1, H, W]`` FP32.
    """
    h = F.selu(F.conv2d(x.float(), w1.float(), padding=1))
    h = F.selu(F.conv2d(h, w2.float(), padding=1))
    return F.conv2d(h, w3.float(), padding=1)


raco_score_head = Op("raco_score_head", _score_head_reference, weights=3)
