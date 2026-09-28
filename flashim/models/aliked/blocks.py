"""ALIKED encoder building blocks (module names follow the released checkpoints)."""

from __future__ import annotations

from torch import Tensor, nn

try:
    from torchvision.ops import deform_conv2d
except ImportError as exc:  # pragma: no cover - torchvision is a core dependency, this is a clearer error
    raise ImportError("flashim.models.aliked needs torchvision (for torchvision.ops.deform_conv2d)") from exc


class DeformableConv2d(nn.Module):
    """3x3 deformable convolution: a plain conv predicts per-pixel sampling offsets, clamped to
    ``max(H, W) / 4``, for a second conv's ``deform_conv2d`` (ALIKED's ``block3``/``block4``; no mask, matching
    every released variant)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size**2, kernel_size, padding=padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        max_offset = max(x.shape[-2:]) / 4.0
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        dtype = x.dtype
        # torchvision's deform_conv2d kernel has no BF16/FP16 implementation (CPU or CUDA); run it in FP32.
        out = deform_conv2d(
            x.float(),
            offset.float(),
            self.regular_conv.weight.float(),
            bias=self.regular_conv.bias.float() if self.regular_conv.bias is not None else None,
            padding=(self.padding, self.padding),
        )
        return out.to(dtype)


def get_conv(in_channels: int, out_channels: int, conv_type: str) -> nn.Module:
    if conv_type == "conv":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    if conv_type == "dcn":
        return DeformableConv2d(in_channels, out_channels)
    raise ValueError(f"Unknown conv_type {conv_type!r}")


class ConvBlock(nn.Module):
    """conv -> BN -> gate, twice (ALIKED's ``block1``, always plain convolutions)."""

    def __init__(self, in_channels: int, out_channels: int, gate: nn.Module, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.gate = gate
        self.conv1 = get_conv(in_channels, out_channels, "conv")
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(out_channels, out_channels, "conv")
        self.bn2 = norm_layer(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate(self.bn1(self.conv1(x)))
        return self.gate(self.bn2(self.conv2(x)))


class ResBlock(nn.Module):
    """Pre-activation-free residual block (torchvision's ``BasicBlock``, ``conv_type`` swaps in deformable convs
    for ``block3``/``block4``)."""

    def __init__(
        self, in_channels: int, out_channels: int, downsample: nn.Module, gate: nn.Module, norm_layer: type[nn.Module], conv_type: str
    ) -> None:
        super().__init__()
        self.gate = gate
        self.conv1 = get_conv(in_channels, out_channels, conv_type)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(out_channels, out_channels, conv_type)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x))
        out = self.gate(out)
        out = self.bn2(self.conv2(out))
        return self.gate(out + self.downsample(x))
