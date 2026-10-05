"""LoFTR (Sun et al., CVPR 2021): ResNetFPN 1/8 + 1/2 features, an 8-layer linear-attention transformer on the coarse
tokens, dual-softmax coarse matching, and a 5x5-window fine refinement."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.conv import conv_relu
from flashim.models import sparse
from flashim.models.loftr import weights
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import SparseMatchResult
from flashim.utils import device_constant


def conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, stride, 0, bias=False)


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1, self.conv2 = conv3x3(cin, cout, stride), conv3x3(cout, cout)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None if stride == 1 else nn.Sequential(conv1x1(cin, cout, stride), nn.BatchNorm2d(cout))

    def forward(self, x: Tensor) -> Tensor:
        y = conv_relu(self.conv1, x, self.bn1)
        identity = x if self.downsample is None else self.downsample(x)
        if getattr(self.conv2, "fuses_relu", False) and isinstance(self.bn2, nn.Identity):
            # the residual add and ReLU in the quantized conv's epilogue (one BF16 rounding instead of two)
            return self.conv2(y, relu=True, residual=identity)
        return self.relu(identity + self.bn2(self.conv2(y)))


class ResNetFPN_8_2(nn.Module):
    """Returns ``[coarse (1/8, 256 ch), fine (1/2, 128 ch)]``."""

    def __init__(self, initial_dim: int = 128, block_dims: tuple[int, int, int] = (128, 196, 256)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, initial_dim, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        d1, d2, d3 = block_dims
        self.layer1 = nn.Sequential(BasicBlock(initial_dim, d1, 1), BasicBlock(d1, d1))
        self.layer2 = nn.Sequential(BasicBlock(d1, d2, 2), BasicBlock(d2, d2))
        self.layer3 = nn.Sequential(BasicBlock(d2, d3, 2), BasicBlock(d3, d3))
        self.layer3_outconv = conv1x1(d3, d3)
        self.layer2_outconv = conv1x1(d2, d3)
        self.layer2_outconv2 = nn.Sequential(conv3x3(d3, d3), nn.BatchNorm2d(d3), nn.LeakyReLU(), conv3x3(d3, d2))
        self.layer1_outconv = conv1x1(d1, d2)
        self.layer1_outconv2 = nn.Sequential(conv3x3(d2, d2), nn.BatchNorm2d(d2), nn.LeakyReLU(), conv3x3(d2, d1))

    def forward(self, x: Tensor) -> list[Tensor]:
        x1 = self.layer1(self.relu(self.bn1(self.conv1(x))))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3_out = self.layer3_outconv(x3)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + F.interpolate(x3_out, size=x2_out.shape[2:], mode="bilinear", align_corners=True))
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + F.interpolate(x2_out, size=x1_out.shape[2:], mode="bilinear", align_corners=True))
        return [x3_out, x1_out]


def linear_attention(q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6, gemm_dtype: torch.dtype | None = None) -> Tensor:
    """ELU+1 kernel attention on ``[N, L, H, D]`` / ``[N, S, H, D]`` (FP32 core; values pre-divided by S like upstream).

    ``gemm_dtype`` (e.g. BF16) runs the two GEMMs (``K^T V`` and ``Q (K^T V)``) on operands of that dtype, i.e. on
    tensor cores instead of FP32 SIMT kernels; the ELU, the normalizer and the final scaling stay FP32. Measured
    75.4 -> 71.2 ms per 1200 px pair on sm_120; not exact (the pass ``bf16_linear_attention`` enables it)."""
    q, k, v = F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float()
    length = v.shape[1]
    z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + eps)
    if gemm_dtype is None:
        kv = torch.einsum("nshd,nshv->nhdv", k, v / length)
        return torch.einsum("nlhd,nhdv,nlh->nlhv", q, kv, z) * length
    kv = torch.einsum("nshd,nshv->nhdv", k.to(gemm_dtype), (v / length).to(gemm_dtype))
    return torch.einsum("nlhd,nhdv->nlhv", q.to(gemm_dtype), kv).float() * z[..., None] * length


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.dim, self.nhead = d_model // nhead, nhead
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(3))
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model, bias=False), nn.ReLU(True), nn.Linear(2 * d_model, d_model, bias=False))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attention_gemm_dtype: torch.dtype | None = None

    def forward(self, x: Tensor, source: Tensor) -> Tensor:
        bs = x.shape[0]
        q = self.q_proj(x).reshape(bs, -1, self.nhead, self.dim)
        k = self.k_proj(source).reshape(bs, -1, self.nhead, self.dim)
        v = self.v_proj(source).reshape(bs, -1, self.nhead, self.dim)
        message = self.merge(linear_attention(q, k, v, gemm_dtype=self.attention_gemm_dtype).to(x.dtype).reshape(bs, -1, self.nhead * self.dim))
        message = self.mlp(torch.cat([x, self.norm1(message)], dim=2))
        return x + self.norm2(message)


class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, layer_names: list[str]) -> None:
        super().__init__()
        self.layer_names = layer_names
        self.layers = nn.ModuleList(LoFTREncoderLayer(d_model, nhead) for _ in layer_names)

    def forward(self, feat0: Tensor, feat1: Tensor) -> tuple[Tensor, Tensor]:
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0, feat1 = layer(feat0, feat0), layer(feat1, feat1)
            else:  # sequential, like upstream: image 1 attends to the *updated* image-0 features
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
        return feat0, feat1


def position_encoding(d_model: int, height: int, width: int, temp_bug_fix: bool, device) -> Tensor:
    """Sine encoding ``[1, C, H, W]`` (upstream keeps the ``temp_bug_fix=False`` variant for the outdoor weights)."""
    pe = torch.zeros((d_model, height, width))
    y = torch.ones((height, width)).cumsum(0).float().unsqueeze(0)
    x = torch.ones((height, width)).cumsum(1).float().unsqueeze(0)
    scale = -math.log(10000.0) / (d_model // 2) if temp_bug_fix else -math.log(10000.0) / d_model // 2
    div = torch.exp(torch.arange(0, d_model // 2, 2).float() * scale)[:, None, None]
    pe[0::4], pe[1::4], pe[2::4], pe[3::4] = torch.sin(x * div), torch.cos(x * div), torch.sin(y * div), torch.cos(y * div)
    return pe[None].to(device)


def _inner_cells(height: int, width: int, border: int, device: torch.device) -> Tensor:
    """``[height * width]`` bool: cells at least ``border`` cells away from every edge of the grid."""
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[border : height - border, border : width - border] = True
    return mask.reshape(-1)


class LoFTR(nn.Module):
    """Module paths follow the official checkpoints (``backbone``, ``loftr_coarse``, ``fine_preprocess``, ``loftr_fine``);
    loads strictly. ``match(img_a, img_b)`` -> :class:`SparseMatchResult` in each original image's pixel coordinates
    (LoFTR's convention: coarse cell index x 8 plus the fine offset).

    Precision presets: the backbone and both transformers run in the compute dtype; the attention core, the coarse
    similarity / dual softmax (temperature 0.1) and the fine expectation stay FP32."""

    div_by = 8
    fold_temperature = False  # see spec.py's fold_similarity_temperature pass
    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None
    fp32_modules: tuple[str, ...] = ()
    compute_dtype = torch.float32

    def __init__(self, temp_bug_fix: bool = False, threshold: float = 0.2, border_rm: int = 2, fine_window: int = 5) -> None:
        super().__init__()
        self.temp_bug_fix, self.threshold, self.border_rm, self.W = temp_bug_fix, threshold, border_rm, fine_window
        self.backbone = ResNetFPN_8_2()
        self.loftr_coarse = LocalFeatureTransformer(256, 8, ["self", "cross"] * 4)
        self.fine_preprocess = nn.Module()
        self.fine_preprocess.down_proj = nn.Linear(256, 128, bias=True)
        self.fine_preprocess.merge_feat = nn.Linear(256, 128, bias=True)
        self.loftr_fine = LocalFeatureTransformer(128, 8, ["self", "cross"])
        self._pe: dict[tuple[int, int], Tensor] = {}

    @classmethod
    def from_pretrained(cls, *, variant: str = "outdoor", checkpoint: str | Path | None = None,
                        device: torch.device | str | None = None, **kwargs) -> LoFTR:
        with torch.device("meta"):
            model = cls(temp_bug_fix=variant == "indoor", **kwargs)
        model.load_state_dict(weights.load_state_dict(variant, checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    @property
    def device(self) -> torch.device:
        return self.backbone.conv1.weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.to(dtype)
        self.compute_dtype = dtype

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        pairs = [("backbone.conv1", "backbone.bn1")]
        for layer in ("layer1", "layer2", "layer3"):
            for block in (0, 1):
                base = f"backbone.{layer}.{block}"
                pairs += [(f"{base}.conv1", f"{base}.bn1"), (f"{base}.conv2", f"{base}.bn2")]
                if self.get_submodule(base).downsample is not None:
                    pairs.append((f"{base}.downsample.0", f"{base}.downsample.1"))
        pairs += [(f"backbone.{n}.0", f"backbone.{n}.1") for n in ("layer2_outconv2", "layer1_outconv2")]
        return pairs

    fold_batchnorm = sparse.SparseExtractor.fold_batchnorm

    def positions(self, h: int, w: int) -> Tensor:
        if (h, w) not in self._pe:
            self._pe[(h, w)] = position_encoding(256, h, w, self.temp_bug_fix, self.device)
        return self._pe[(h, w)]

    def coarse_features(self, gray: Tensor) -> tuple[Tensor, Tensor]:
        feat_c, feat_f = self.backbone(gray.to(self.compute_dtype))
        return feat_c, feat_f

    @torch.inference_mode()
    def match_prepared(self, gray0: Tensor, gray1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[1, 1, H, W]`` grayscale pair (sides multiples of 8) -> matches ``(kpts0 [M, 2], kpts1 [M, 2], conf [M])`` in
        the inputs' pixel coordinates."""
        h0, h1 = gray0.shape[-2], gray1.shape[-2]
        (c0, f0), (c1, f1) = self.coarse_features(gray0), self.coarse_features(gray1)
        (hc0, wc0), (hc1, wc1) = c0.shape[-2:], c1.shape[-2:]
        tokens0 = (c0.float() + self.positions(hc0, wc0)).to(c0.dtype).permute(0, 2, 3, 1).reshape(1, -1, c0.shape[1])
        tokens1 = (c1.float() + self.positions(hc1, wc1)).to(c1.dtype).permute(0, 2, 3, 1).reshape(1, -1, c1.shape[1])
        tokens0, tokens1 = self.loftr_coarse(tokens0, tokens1)

        # coarse matching: dual softmax over the normalized similarity, mutual maxima above the threshold. The
        # ``[L, S]`` probabilities are never formed (``ops.dual_softmax_mutual``): measured 35 -> 4 ms and 2.1 GB less
        # peak memory at 1200 px (16.8k x 16.5k tokens) with identical matches.
        a, b = (t.float() / t.shape[-1] ** 0.5 for t in (tokens0, tokens1))
        if self.fold_temperature:  # 1 / temperature on the [L, C] operand, not a pass over the [L, S] FP32 matrix
            sim = torch.einsum("nlc,nsc->nls", a / 0.1, b)
        else:
            sim = torch.einsum("nlc,nsc->nls", a, b) / 0.1
        log_conf, all_j, mutual = ops.dual_softmax_mutual(sim[0])
        del sim
        conf = log_conf.exp()
        valid = mutual & (conf > self.threshold)
        r = self.border_rm
        if r > 0:  # drop matches whose cell in either image lies within ``r`` cells of the border
            inner0, inner1 = (_inner_cells(h, w, r, conf.device) for h, w in ((hc0, wc0), (hc1, wc1)))
            valid &= inner0 & inner1[all_j]
        i_ids = torch.nonzero(valid)[:, 0]
        j_ids = all_j[i_ids]
        mconf = conf[i_ids]
        stride_c0, stride_c1 = h0 / hc0, h1 / hc1
        kpts0 = torch.stack([i_ids % wc0, i_ids // wc0], dim=1).float() * stride_c0
        kpts1 = torch.stack([j_ids % wc1, j_ids // wc1], dim=1).float() * stride_c1
        if len(i_ids) == 0:
            return kpts0, kpts1, mconf

        # fine refinement: 5x5 windows of the 1/2 features around each coarse match
        W = self.W
        stride = f0.shape[2] // hc0
        def windows(feat: Tensor) -> Tensor:
            unfold = F.unfold(feat, kernel_size=(W, W), stride=stride, padding=W // 2)
            return unfold.reshape(1, feat.shape[1], W * W, -1).permute(0, 3, 2, 1)[0]  # [L, WW, C]
        fw0, fw1 = windows(f0)[i_ids], windows(f1)[j_ids]
        pre = self.fine_preprocess
        coarse = pre.down_proj(torch.cat([tokens0[0, i_ids], tokens1[0, j_ids]], 0).to(pre.down_proj.weight.dtype))
        merged = pre.merge_feat(torch.cat([torch.cat([fw0, fw1], 0), coarse.unsqueeze(1).repeat(1, W * W, 1)], -1))
        fw0, fw1 = torch.chunk(merged, 2, dim=0)
        fw0, fw1 = self.loftr_fine(fw0, fw1)
        picked = fw0[:, W * W // 2, :].float()
        sim_f = torch.einsum("mc,mrc->mr", picked, fw1.float())
        heat = F.softmax(sim_f / fw0.shape[-1] ** 0.5, dim=1).view(-1, W, W)
        grid = torch.linspace(-1, 1, W, device=heat.device)
        coords = torch.stack([(heat.sum(1) * grid).sum(1), (heat.sum(2) * grid).sum(1)], dim=-1)  # (x, y) expectation
        scale = h0 / f0.shape[2]
        return kpts0, kpts1 + coords * (W // 2) * scale, mconf

    @torch.inference_mode()
    def match(self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None, top_k: int | None = None,
              threshold: float | None = None) -> SparseMatchResult:
        """``resize``: long side of the working resolution (sides floored to multiples of 8); ``top_k``: keep the best
        ``top_k`` matches by confidence (``None`` / ``<= 0``: all); ``threshold``: coarse confidence threshold."""
        images = [load_image(im, self.device) for im in (img_a, img_b)]
        prepared, sizes = [], []
        for image in images:
            height, width = image.shape[-2:]
            new_h, new_w = sparse.target_size(height, width, resize, self.div_by)
            if (new_h, new_w) != (height, width):
                image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
            prepared.append(sparse.rgb_to_gray(image))
            sizes.append((height / new_h, width / new_w))
        previous = self.threshold
        self.threshold = previous if threshold is None else threshold
        try:
            k0, k1, conf = self.match_prepared(*prepared)
        finally:
            self.threshold = previous
        if top_k is not None and 0 < top_k < len(conf):
            conf, index = conf.topk(top_k)
            k0, k1 = k0[index], k1[index]
        k0 = k0 * device_constant([sizes[0][1], sizes[0][0]], k0)
        k1 = k1 * device_constant([sizes[1][1], sizes[1][0]], k1)
        return SparseMatchResult(k0, k1, conf)
