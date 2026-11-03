"""ViT patch embedding as one cuBLAS GEMM (``ops.patch_embed_tokens``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim import ops
from flashim.ops._registry import DeviceSpec

_ALIGN = 8  # K padded to a multiple of 8 BF16 elements (16-byte rows for the GEMM)


def _prepare(weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor, int, int]:
    dim, channels, patch, _ = weight.shape
    k = channels * patch * patch
    kpad = -(-k // _ALIGN) * _ALIGN
    packed = F.pad(weight.detach().reshape(dim, k), (0, kpad - k)).contiguous()
    return packed.t(), bias.detach(), k, patch  # [Kpad, D] view: cuBLAS reads the transpose


def _patch_embed_cublas(
    image: Tensor, cls_token: Tensor, pos_embed: Tensor, weight_t: Tensor, bias: Tensor, k: int, patch: int
):
    """Patches ``[h w, Kpad]`` (the image's 14x14 blocks, zero-padded K) x the weight into the token buffer after the
    class token, then the bias and positional embedding added in BF16 like the reference's two adds. Only the GEMM's FP32
    summation order differs from cuDNN's convolution."""
    dtype = bias.dtype
    kpad, dim = weight_t.shape
    _, channels, height, width = image.shape
    h, w = height // patch, width // patch
    image = image[..., : h * patch, : w * patch].to(dtype)
    patches = torch.empty(h * w, kpad, device=image.device, dtype=dtype)
    patches[:, k:].zero_()
    patches[:, :k].view(h, w, channels, patch, patch).copy_(
        image[0].reshape(channels, h, patch, w, patch).permute(1, 3, 0, 2, 4)
    )
    tokens = torch.empty(1, 1 + h * w, dim, device=image.device, dtype=dtype)
    tokens[0, 0] = cls_token.to(dtype)[0, 0]
    body = tokens[0, 1:]
    torch.mm(patches, weight_t, out=body)
    body.add_(bias)
    return tokens.add_(pos_embed.to(dtype))


def _eligible(image: Tensor, cls_token: Tensor, pos_embed: Tensor, weight: Tensor, bias: Tensor) -> bool:
    patch = weight.shape[-1]
    return (
        image.ndim == 4 and image.shape[0] == 1 and weight.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
        and weight.shape[-2] == patch and image.shape[1] == weight.shape[1]
        and image.shape[2] >= patch and image.shape[3] >= patch
        and pos_embed.shape[1] == 1 + (image.shape[2] // patch) * (image.shape[3] // patch)
    )


# sm_89 and sm_90 only: the two architectures it was measured on.
ops.patch_embed_tokens.register(
    "cublas", device=DeviceSpec("cuda", min_capability=(8, 9), max_capability=(9, 0)),
    when=_eligible, prepare=_prepare,
)(_patch_embed_cublas)
