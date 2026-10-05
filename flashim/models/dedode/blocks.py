"""Shared DeDoDe / DaD networks: VGG(-bn) and DINOv2 encoders, ConvRefiner decoder, and the multi-scale forward.

Parameter names follow the released checkpoints (``encoder.layers.N`` (torchvision ``vgg*_bn().features`` indices),
``decoder.layers.<scale>.block1 / hidden_blocks.i / out_conv``); DINOv2's weights (not part of DeDoDe-G's checkpoint)
live under ``encoder.dinov2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch import Tensor, nn

from flashim import ops
from flashim.layers import IndexedSeparableConvBlock, fused_relu_end, sequential_forward
from flashim.layers.transformer import TransformerBlock

VGG_CUT = {"11": (tvm.vgg11_bn, 22), "13": (tvm.vgg13_bn, 28), "19": (tvm.vgg19_bn, 40)}


class VGG(nn.Module):
    """``vgg*_bn().features[:cut]``; returns the maps feeding each max-pool (strides 1, 2, 4, 8) and their sizes."""

    def __init__(self, size: str = "19") -> None:
        super().__init__()
        factory, cut = VGG_CUT[size]
        self.layers = nn.ModuleList(factory(weights=None).features[:cut])

    def forward(self, x: Tensor) -> tuple[list[Tensor], list[tuple[int, int]]]:
        feats, sizes = [], []
        layers, i = self.layers, 0
        while i < len(layers):
            layer = layers[i]
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                sizes.append(x.shape[-2:])
            end = fused_relu_end(layers, i)  # conv -> folded BN -> ReLU as one quantized conv with a ReLU epilogue
            x = layer(x) if end is None else layer(x, relu=True)
            i = i + 1 if end is None else end + 1
        return feats, sizes

    def batchnorm_pairs(self, prefix: str) -> list[tuple[str, str]]:
        layers = list(self.layers)
        return [
            (f"{prefix}layers.{i}", f"{prefix}layers.{i + 1}")
            for i, layer in enumerate(layers[:-1])
            if isinstance(layer, nn.Conv2d) and isinstance(layers[i + 1], nn.BatchNorm2d)
        ]


class DinoV2(nn.Module):
    """DINOv2 ViT-L/14 patch features (``x_norm_patchtokens``) as ``[B, C, H/14, W/14]``; upstream's positional
    embedding interpolation (bicubic, ``scale_factor`` with the ``+0.1`` guard) is reproduced exactly."""

    def __init__(self, dim: int = 1024, depth: int = 24, heads: int = 16, patch: int = 14, img_size: int = 518) -> None:
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid**2, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # unused (checkpoint parity)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, eps=1e-6, layer_scale=True) for _ in range(depth)
        )
        for block in self.blocks:
            block.attn.sdpa_impl = None  # dispatch (the reference impl is torch's SDPA)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self._pos_cache: dict[tuple[int, int, torch.dtype], Tensor] = {}

    def positions(self, h: int, w: int, dtype: torch.dtype) -> Tensor:
        key = (h, w, dtype)
        if key not in self._pos_cache:
            pe = self.pos_embed.float()
            dim = pe.shape[-1]
            patch_pe = F.interpolate(
                pe[:, 1:].reshape(1, self.grid, self.grid, dim).permute(0, 3, 1, 2),
                scale_factor=((h + 0.1) / self.grid, (w + 0.1) / self.grid), mode="bicubic",
            )
            assert patch_pe.shape[-2:] == (h, w)
            self._pos_cache[key] = torch.cat([pe[:, :1], patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)], 1).to(dtype)
        return self._pos_cache[key]

    def forward(self, x: Tensor) -> Tensor:
        b, _, height, width = x.shape
        h, w = height // self.patch, width // self.patch
        tokens = self.patch_embed.proj(x.to(self.patch_embed.proj.weight.dtype)).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.to(tokens.dtype).expand(b, -1, -1), tokens], 1)
        tokens = tokens + self.positions(h, w, tokens.dtype).to(tokens.device)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:].transpose(1, 2).reshape(b, -1, h, w)


class VGGDINOv2(nn.Module):
    """DeDoDe-G's encoder: VGG19 maps (strides 1..8) + DINOv2 patch features (stride 14, the coarsest scale)."""

    def __init__(self) -> None:
        super().__init__()
        self.vgg = VGG("19")
        self.dinov2 = DinoV2()

    def forward(self, x: Tensor):
        feats, sizes = self.vgg(x)
        h, w = x.shape[-2:]
        return feats + [self.dinov2(x)], sizes + [(h // 14, w // 14)]

    def batchnorm_pairs(self, prefix: str) -> list[tuple[str, str]]:
        return self.vgg.batchnorm_pairs(f"{prefix}vgg.")


class ConvRefiner(nn.Module):
    """``block1`` (1x1) -> ``hidden_blocks`` (depthwise ``k x k`` -> BN -> ReLU -> 1x1, each) -> optional residual
    ``(x + x0) / 1.4`` -> ``out_conv`` (1x1)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden_blocks: int = 5, kernel_size: int = 5,
                 residual: bool = True) -> None:
        super().__init__()
        self.block1 = self.create_block(in_dim, hidden_dim, dw=False, kernel_size=1)
        self.hidden_blocks = nn.Sequential(*[self.create_block(hidden_dim, hidden_dim, dw=True, kernel_size=kernel_size) for _ in range(hidden_blocks)])
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.residual = residual

    @staticmethod
    def create_block(cin: int, cout: int, dw: bool, kernel_size: int) -> nn.Module:
        if dw:  # the separable-conv layer type (fused-kernel passes), with the checkpoints' Sequential parameter names
            return IndexedSeparableConvBlock(cout, kernel_size)
        conv1 = nn.Conv2d(cin, cout, kernel_size, 1, kernel_size // 2, bias=True)
        return nn.Sequential(conv1, nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 1, 1, 0))

    def hidden(self, x: Tensor) -> Tensor:
        """Everything before ``out_conv``."""
        x0 = sequential_forward(self.block1, x)
        x = self.hidden_blocks(x0)
        return (x + x0) / 1.4 if self.residual else x

    def forward(self, x: Tensor) -> Tensor:
        return self.out_conv(self.hidden(x))


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleDict, num_prototypes: int = 1) -> None:
        super().__init__()
        self.layers = layers
        self.num_prototypes = num_prototypes

    @staticmethod
    def refiner_input(features: Tensor, context: Tensor | None) -> Tensor:
        """``cat(features, context)``; a coarser ``context`` is upsampled (bilinear, FP32) in the same pass."""
        if context is None:
            return features
        if context.shape[-2:] != features.shape[-2:]:
            return ops.upsample_concat(features, context)
        return torch.cat((features, context.to(features.dtype)), dim=1)

    def forward(self, features: Tensor, context: Tensor | None, scale: str) -> tuple[Tensor, Tensor]:
        out = self.layers[scale](self.refiner_input(features, context))
        return out[:, : self.num_prototypes], out[:, self.num_prototypes :]


@dataclass(frozen=True)
class NetConfig:
    """One ``encoder + decoder`` network: ``refiners`` maps scale -> ``(in_dim, hidden_dim, out_dim)``."""

    encoder: str  # "vgg11" | "vgg19" | "vgg19+dinov2"
    refiners: dict[str, tuple[int, int, int]]
    hidden_blocks: int
    num_prototypes: int
    interpolation: str  # "bicubic" (detectors) | "bilinear" (descriptors)

    @property
    def scales(self) -> list[str]:
        return list(self.refiners)


class DeDoDeNet(nn.Module):
    """Encoder + multi-scale decoder: per scale (coarse to fine) a refiner predicts an increment of the output map
    (``num_prototypes`` channels) and the context passed to the next scale; the map / context are upsampled to the
    next encoder feature size in between. The sum is kept in FP32."""

    fused_upsample = False  # see forward

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder: nn.Module = {"vgg11": lambda: VGG("11"), "vgg19": lambda: VGG("19"), "vgg19+dinov2": VGGDINOv2}[config.encoder]()
        self.decoder = Decoder(
            nn.ModuleDict({s: ConvRefiner(*dims, hidden_blocks=config.hidden_blocks) for s, dims in config.refiners.items()}),
            config.num_prototypes,
        )

    def forward(self, x: Tensor, return_features: bool = False, points: Tensor | None = None):
        """The summed output map; with ``return_features`` also the encoder's feature maps (RIPE's hypercolumns). With
        ``points`` ``[K, 2]`` (pixel-index ``(x, y)`` at the input resolution) the output is only read there: ``[K,
        num_prototypes]`` FP32, the bilinear samples (``align_corners=False``) of the map, which is then never formed
        at the last scale (:meth:`_read_last_scale`)."""
        cfg = self.config
        features, sizes = self.encoder(x)
        # With ``fused_upsample`` (spec pass ``fused_decoder_upsample``) the context and (bilinear) output maps stay
        # coarse between scales: their upsampling runs inside the next scale's consumer (ops.upsample_concat building
        # the refiner input, ops.upsample_add adding the next delta) instead of writing upsampled FP32 maps and reading
        # them back for a cast / concat / add. Measured extract at 1200 px on sm_120, off -> on: DeDoDe int8 47.3 ->
        # 39.7 ms, bf16 64.5 -> 56.9, DaD fp8 82.2 -> 74.8, RIPE int8 27.6 -> 24.4 (keypoints unchanged).
        fused = self.fused_upsample
        out, context, coarse = 0, None, False
        last = len(cfg.scales) - 1
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), cfg.scales)):
            if points is not None and idx == last:
                hidden = self.decoder.layers[scale].hidden(self.decoder.refiner_input(feature_map, context))
                return self._read_last_scale(self.decoder.layers[scale].out_conv, hidden, out, points)
            delta, context = self.decoder(feature_map, context, scale)
            out = ops.upsample_add(out, delta) if coarse else out + delta.float()
            coarse = False
            if idx < last:
                size = sizes[-(idx + 2)]
                if fused and cfg.interpolation == "bilinear":
                    coarse = True  # upsampled with the next delta, or read at the points
                elif points is None or idx < last - 1:  # with points, the last upsample happens at the read pixels
                    out = F.interpolate(out, size=size, mode=cfg.interpolation, align_corners=False)
                if not fused:
                    context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        return (out, features) if return_features else out

    def _read_last_scale(self, out_conv: nn.Module, hidden: Tensor, previous: Tensor, points: Tensor) -> Tensor:
        """``bilinear_at(upsample(previous) + out_conv(hidden)[:, :P], points)`` evaluated at the 4 neighbouring pixels
        of each point only: ``out_conv`` (a 1x1 convolution, called as the module so quantized / protected variants
        keep their own math) on the gathered hidden pixels, the previous scale's output upsampled at those pixel
        centres (``grid_sample`` with border clamping = ``F.interpolate``'s source-index clamp), then the bilinear
        weights. Same math as the dense map, FP32 rounding in another order. Measured on DeDoDe int8 at 1200 px: the
        dense last scale (out_conv, FP32 upsample of 256 channels to 896x1200, cast, add, bias) was 13.6 of 66 ms."""
        _, _, height, width = hidden.shape
        x0, y0 = points[:, 0].floor(), points[:, 1].floor()
        fx, fy = points[:, 0] - x0, points[:, 1] - y0
        xs = torch.stack((x0, x0 + 1, x0, x0 + 1), 1)  # [K, 4] neighbours: (x0, y0), (x1, y0), (x0, y1), (x1, y1)
        ys = torch.stack((y0, y0, y0 + 1, y0 + 1), 1)
        weights = torch.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), 1)
        weights = weights * ((xs >= 0) & (xs < width) & (ys >= 0) & (ys < height))  # zero padding, as grid_sample
        xi, yi = xs.clamp(0, width - 1).long(), ys.clamp(0, height - 1).long()
        pixels = hidden.flatten(2)[:, :, (yi * width + xi).reshape(-1)]  # [1, C, 4K]
        prototypes = self.decoder.num_prototypes
        delta = out_conv(pixels[..., None])[0, :prototypes, :, 0].float()  # [P, 4K]
        grid = torch.stack(((xi + 0.5) * 2 / width - 1, (yi + 0.5) * 2 / height - 1), -1).reshape(1, 1, -1, 2)
        mode = self.config.interpolation
        previous = F.grid_sample(
            previous, grid.to(previous.dtype), mode=mode, padding_mode="border", align_corners=False
        )[0, :, 0]  # [P, 4K]
        values = (previous + delta).reshape(prototypes, -1, 4)
        return (values * weights[None]).sum(-1).t()

    def batchnorm_pairs(self, prefix: str = "") -> list[tuple[str, str]]:
        pairs = self.encoder.batchnorm_pairs(f"{prefix}encoder.")
        for scale, refiner in self.decoder.layers.items():
            base = f"{prefix}decoder.layers.{scale}."
            pairs += [(f"{base}block1.0", f"{base}block1.1")]
            pairs += [(f"{base}hidden_blocks.{i}.0", f"{base}hidden_blocks.{i}.1") for i in range(len(refiner.hidden_blocks))]
        return pairs


# -- network configurations ------------------------------------------------------------------------------

def _detector(hidden: int, dims: dict[str, tuple[int, int, int]], encoder: str) -> NetConfig:
    return NetConfig(encoder, dims, hidden, 1, "bicubic")


DETECTOR_L = _detector(8, {"8": (512, 512, 257), "4": (512, 256, 129), "2": (256, 128, 65), "1": (128, 64, 2)}, "vgg19")
DETECTOR_S = _detector(3, {"8": (512, 512, 257), "4": (512, 256, 129), "2": (256, 64, 33), "1": (96, 32, 2)}, "vgg11")  # DaD
RIPE_DETECTOR = NetConfig("vgg19", DETECTOR_L.refiners, 8, 1, "bilinear")  # RIPE: DeDoDe-L's detector with bilinear upsampling
DESCRIPTOR_B = NetConfig(
    "vgg19", {"8": (512, 512, 512), "4": (512, 256, 384), "2": (256, 64, 288), "1": (96, 32, 257)}, 5, 256, "bilinear"
)
DESCRIPTOR_G = NetConfig(
    "vgg19+dinov2",
    {"14": (1024, 768, 768), "8": (1024, 512, 512), "4": (512, 256, 384), "2": (256, 64, 288), "1": (96, 32, 257)},
    5, 256, "bilinear",
)
DESCRIPTORS = {"B": DESCRIPTOR_B, "G": DESCRIPTOR_G}
