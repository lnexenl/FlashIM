"""RaCo conv blocks (module names follow the released checkpoint)."""

from __future__ import annotations

from torch import Tensor, nn


class ConvBlock(nn.Module):
    """conv3x3 -> BN -> SELU, twice."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = nn.SELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate(self.bn1(self.conv1(x)))
        return self.gate(self.bn2(self.conv2(x)))


class ResBlock(nn.Module):
    """Residual block with a 1x1 projection shortcut (``match_dims``)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = nn.SELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.match_dims = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.match_dims(x)
        out = self.gate(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.gate(out + identity)
