"""RDD keypoint detector: a small multi-scale conv net on the image (the release's ``legacy`` detector)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.utils import device_constant

MEAN = (0.430, 0.411, 0.296)
STD = (0.213, 0.156, 0.143)


def _conv3x3(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(cin, cout), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = _conv3x3(cout, cout), nn.BatchNorm2d(cout)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv1, self.bn1 = _conv3x3(cin, cout), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = _conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.downsample = nn.Conv2d(cin, cout, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + self.downsample(x))


class Detector(nn.Module):
    """Image ``[B, 3, H, W]`` in ``[0, 1]`` -> keypoint score map ``[B, 1, H, W]`` in ``(0, 1)``."""

    def __init__(self, dims: tuple[int, ...] = (8, 16, 32, 64)) -> None:
        super().__init__()
        self.block1 = ConvBlock(3, dims[0])
        self.block2 = ResBlock(dims[0], dims[1])
        self.block3 = ResBlock(dims[1], dims[2])
        self.block4 = ResBlock(dims[2], dims[3])
        hidden = dims[-1]
        self.conv1, self.conv2, self.conv3, self.conv4 = (
            nn.Conv2d(dim, hidden // 4, 1, bias=False) for dim in dims
        )
        self.convhead2 = nn.Sequential(nn.Conv2d(hidden, 1, 1, bias=False), nn.Sigmoid())

    def set_dtype(self, dtype: torch.dtype) -> None:
        """Run the conv trunk in ``dtype``; the score head (1x1 conv + sigmoid) stays FP32."""
        self.to(dtype)
        self.convhead2.float()

    def forward(self, image: Tensor) -> Tensor:
        mean, std = (device_constant(v, image).view(1, 3, 1, 1) for v in (MEAN, STD))
        x = ((image - mean) / std).to(self.block1.conv1.weight.dtype)
        x1 = self.block1(x)
        x2 = self.block2(F.max_pool2d(x1, 2))
        x3 = self.block3(F.max_pool2d(x2, 4))
        x4 = self.block4(F.max_pool2d(x3, 4))
        x1, x2, x3, x4 = (F.relu(conv(feat)) for conv, feat in zip((self.conv1, self.conv2, self.conv3, self.conv4), (x1, x2, x3, x4)))

        def up(feat: Tensor, scale: int) -> Tensor:
            return F.interpolate(feat, scale_factor=scale, mode="bilinear", align_corners=True)

        return self.convhead2(torch.cat([x1, up(x2, 2), up(x3, 8), up(x4, 32)], dim=1).float())
