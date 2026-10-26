"""VGG19-BN fine features for the RoMaV2 refiners (first 27 torchvision layers)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flashim.layers.conv import chains, fused_relu_end
from flashim.models.romav2.geometry import imagenet_normalize

# torchvision vgg19_bn.features[:27]: conv/bn/relu x2, pool, x2, pool, x4 (256), pool.
_CONFIG = (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M")


class VGG19BN(nn.Module):
    """Returns ``{1: [B,H,W,64], 2: [B,H/2,W/2,128], 4: [B,H/4,W/4,256]}`` (BF16, NHWC views)."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 3
        for item in _CONFIG:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(channels, item, 3, padding=1), nn.BatchNorm2d(item), nn.ReLU(inplace=True)]
                channels = item
        self.layers = nn.ModuleList(layers)
        self.channels_last = False

    def fold_batchnorm(self, *, channels_last: bool = True) -> int:
        """Fold eval BatchNorm into the preceding convs (inference only; changes rounding)."""
        folded = 0
        for i in range(len(self.layers) - 1):
            conv, norm = self.layers[i], self.layers[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                self.layers[i] = nn.utils.fusion.fuse_conv_bn_eval(conv, norm)
                self.layers[i + 1] = nn.Identity()
                folded += 1
        if channels_last:
            self.channels_last = True
            for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = layer.weight.data.to(memory_format=torch.channels_last)
        return folded

    def forward(self, x: Tensor) -> dict[int, Tensor]:
        x = imagenet_normalize(x)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(x.device.type, torch.bfloat16):
            feats: dict[int, Tensor] = {}
            scale, i = 1, 0
            while i < len(self.layers):
                layer = self.layers[i]
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale *= 2
                    if scale > 4:
                        break  # the final pool's output is never used
                relu = fused_relu_end(self.layers, i)
                if relu is None:
                    x = layer(x)
                    i += 1
                    continue
                # A quantized conv (BatchNorm folded): ReLU in its epilogue, and the next conv's quantized input
                # straight from it where the platform chains. It gets BF16, as autocast gives the plain conv.
                consumer = self.layers[relu + 1]  # a conv or the pool: the config ends with "M"
                if isinstance(x, Tensor) and x.dtype == torch.float32:
                    x = x.to(torch.bfloat16)
                x = layer(x, relu=True, quantize_for=consumer) if chains(layer, consumer) else layer(x, relu=True)
                i = relu + 1
            return feats
