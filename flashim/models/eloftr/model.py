"""Efficient LoFTR (Wang et al., CVPR 2024): RepVGG backbone (1/2, 1/4, 1/8), aggregated RoPE attention transformer on
1/8 coarse features, dual-softmax (``full``) or direct-MNN (``opt``) coarse matching, and two-stage (pixel + 3x3
sub-pixel) fine refinement on 1/1 FPN features."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.conv import conv_relu
from flashim.models import sparse
from flashim.models.base import ImageMatcher
from flashim.models.eloftr import weights
from flashim.models.eloftr.config import ELoFTRConfig
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches
from flashim.utils import device_constant


def _conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, stride, 0, bias=False)


def _conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


def _conv_bn(cin: int, cout: int, kernel_size: int, stride: int, padding: int, groups: int = 1) -> nn.Sequential:
    seq = nn.Sequential()
    seq.add_module("conv", nn.Conv2d(cin, cout, kernel_size, stride, padding, groups=groups, bias=False))
    seq.add_module("bn", nn.BatchNorm2d(cout))
    return seq


def _fuse_bn_branch(
    branch: nn.Module | None, in_channels: int, groups: int, device: torch.device
) -> tuple[Tensor | int, Tensor | int]:
    if branch is None:
        return 0, 0
    if isinstance(branch, nn.Sequential):
        kernel = branch.conv.weight.float()
        bn = branch.bn
    else:
        assert isinstance(branch, nn.BatchNorm2d)
        input_dim = in_channels // groups
        kernel = torch.zeros((in_channels, input_dim, 3, 3), dtype=torch.float32, device=device)
        idx = torch.arange(in_channels, device=device)
        kernel[idx, idx % input_dim, 1, 1] = 1.0
        bn = branch
    std = (bn.running_var.float() + bn.eps).sqrt()
    scale = (bn.weight.float() / std).reshape(-1, 1, 1, 1)
    return kernel * scale, bn.bias.float() - bn.running_mean.float() * bn.weight.float() / std


class RepVGGBlock(nn.Module):
    """Training-time 3-branch RepVGG block (``rbr_dense``, ``rbr_1x1``, optional ``rbr_identity``) that folds into one
    ``rbr_reparam`` 3x3 conv via :meth:`switch_to_deploy`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.rbr_identity: nn.BatchNorm2d | None = (
            nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        )
        self.rbr_dense = _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.rbr_1x1 = _conv_bn(in_channels, out_channels, 1, stride, padding - kernel_size // 2, groups)

    @property
    def deployed(self) -> bool:
        return hasattr(self, "rbr_reparam")

    def switch_to_deploy(self) -> bool:
        if self.deployed:
            return False
        conv3 = self.rbr_dense.conv
        device, dtype = conv3.weight.device, conv3.weight.dtype
        k3, b3 = _fuse_bn_branch(self.rbr_dense, self.in_channels, self.groups, device)
        k1, b1 = _fuse_bn_branch(self.rbr_1x1, self.in_channels, self.groups, device)
        kid, bid = _fuse_bn_branch(self.rbr_identity, self.in_channels, self.groups, device)
        kernel = (k3 + F.pad(k1, (1, 1, 1, 1)) + kid).to(dtype)
        bias = (b3 + b1 + bid).to(dtype)
        reparam = nn.Conv2d(
            conv3.in_channels,
            conv3.out_channels,
            conv3.kernel_size,
            stride=conv3.stride,
            padding=conv3.padding,
            dilation=conv3.dilation,
            groups=conv3.groups,
            bias=True,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.rbr_reparam = reparam
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        return True

    def forward(self, x: Tensor, *, feeds: nn.Module | None = None) -> Tensor:
        if self.deployed:
            return conv_relu(self.rbr_reparam, x, feeds=feeds)
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return F.relu(out)


class RepVGG_8_1_align(nn.Module):
    """RepVGG backbone returning ``(x3 [1/8, 256 ch], x2 [1/4, 128 ch], x1 [1/2, 64 ch])``."""

    def __init__(self, block_dims: tuple[int, int, int] = (64, 128, 256)) -> None:
        super().__init__()
        d1, d2, d3 = block_dims
        self.layer0 = RepVGGBlock(1, d1, 3, stride=2, padding=1)
        self.layer1 = self._make_stage(d1, d1, num_blocks=2, stride=1)
        self.layer2 = self._make_stage(d1, d2, num_blocks=4, stride=2)
        self.layer3 = self._make_stage(d2, d3, num_blocks=14, stride=2)

    @staticmethod
    def _make_stage(in_planes: int, planes: int, num_blocks: int, stride: int) -> nn.ModuleList:
        blocks = [RepVGGBlock(in_planes, planes, 3, stride=stride, padding=1)]
        blocks.extend(RepVGGBlock(planes, planes, 3, stride=1, padding=1) for _ in range(num_blocks - 1))
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: Tensor) -> Tensor:
        last = len(stage) - 1
        for i, block in enumerate(stage):
            nxt = stage[i + 1] if i < last else None
            feeds = getattr(nxt, "rbr_reparam", None) if nxt is not None else None
            x = block(x, feeds=feeds)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x1 = self._run_stage(self.layer1, self.layer0(x))
        x2 = self._run_stage(self.layer2, x1)
        x3 = self._run_stage(self.layer3, x2)
        return x3, x2, x1


def rope_2d_tables(
    d_model: int, height: int, width: int, npe: tuple[int, int, int, int], device: torch.device
) -> tuple[Tensor, Tensor]:
    """2D interleaved RoPE ``(cos, sin)`` ``[1, 1, H * W, d_model]`` in FP16 (upstream ``RoPEPositionEncodingSine``)."""
    train_h, train_w, test_h, test_w = npe
    i_pos = torch.arange(1, height + 1, dtype=torch.float32, device=device)[:, None, None] * (train_h / test_h)
    j_pos = torch.arange(1, width + 1, dtype=torch.float32, device=device)[None, :, None] * (train_w / test_w)
    div = torch.exp(
        torch.arange(0, d_model // 4, dtype=torch.float32, device=device) * (-math.log(10000.0) / (d_model // 4))
    )[None, None, :]
    sin = torch.empty((height, width, d_model // 2), dtype=torch.float16, device=device)
    cos = torch.empty((height, width, d_model // 2), dtype=torch.float16, device=device)
    sin[:, :, 0::2] = torch.sin(i_pos * div).half()
    sin[:, :, 1::2] = torch.sin(j_pos * div).half()
    cos[:, :, 0::2] = torch.cos(i_pos * div).half()
    cos[:, :, 1::2] = torch.cos(j_pos * div).half()
    sin = sin.repeat_interleave(2, dim=-1).reshape(1, 1, height * width, d_model)
    cos = cos.repeat_interleave(2, dim=-1).reshape(1, 1, height * width, d_model)
    return cos, sin


class AggregatedEncoderLayer(nn.Module):
    """Aggregated-attention transformer block (upstream ``AG_RoPE_EncoderLayer``): depthwise strided conv on queries,
    max-pool on keys/values, interleaved 2D RoPE on self-attention, SDPA, bilinear upsample, post-norm FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        agg_size0: int = 4,
        agg_size1: int = 4,
        rope: bool = False,
    ) -> None:
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        self.aggregate: nn.Module = (
            nn.Conv2d(d_model, d_model, agg_size0, stride=agg_size0, padding=0, bias=False, groups=d_model)
            if agg_size0 != 1
            else nn.Identity()
        )
        self.max_pool: nn.Module = (
            nn.MaxPool2d(kernel_size=agg_size1, stride=agg_size1) if agg_size1 != 1 else nn.Identity()
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.sdpa_impl: str | None = "reference"

    def forward(self, x: Tensor, source: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        bs, channels, _, _ = x.shape
        q_feat = self.norm1(self.aggregate(x).permute(0, 2, 3, 1))
        kv_feat = self.norm1(self.max_pool(source).permute(0, 2, 3, 1))
        hq, wq = q_feat.shape[1], q_feat.shape[2]
        q = self.q_proj(q_feat).reshape(bs, hq * wq, channels)
        k = self.k_proj(kv_feat).reshape(bs, -1, channels)
        v = self.v_proj(kv_feat).reshape(bs, -1, channels)

        if self.rope and rope is not None:
            cos, sin = rope
            if bs > 1:
                cos, sin = cos.expand(bs, -1, -1, -1), sin.expand(bs, -1, -1, -1)
            q_rot, k_rot = ops.rope_qk_interleaved(q[:, None], k[:, None], cos, sin)
            q, k = q_rot[:, 0], k_rot[:, 0]

        q = q.reshape(bs, hq * wq, self.nhead, self.dim).transpose(1, 2).contiguous()
        k = k.reshape(bs, -1, self.nhead, self.dim).transpose(1, 2).contiguous()
        v = v.reshape(bs, -1, self.nhead, self.dim).transpose(1, 2).contiguous()
        m = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        m = self.merge(m.transpose(1, 2).reshape(bs, hq * wq, channels))

        m = m.reshape(bs, hq, wq, channels).permute(0, 3, 1, 2)
        if self.agg_size0 != 1:
            m = F.interpolate(m, scale_factor=self.agg_size0, mode="bilinear", align_corners=False)

        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1))
        m = self.norm2(m).permute(0, 3, 1, 2)
        return x + m


class LocalFeatureTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        layer_names: tuple[str, ...],
        agg_size0: int = 4,
        agg_size1: int = 4,
        rope: bool = True,
        npe: tuple[int, int, int, int] = (832, 832, 832, 832),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_names = list(layer_names)
        self.agg_size0 = agg_size0
        self.rope = rope
        self.npe = npe
        self.layers = nn.ModuleList(
            AggregatedEncoderLayer(d_model, nhead, agg_size0, agg_size1, rope=(rope and name == "self"))
            for name in layer_names
        )
        self._rope_cache: dict[tuple[int, int, torch.device], tuple[Tensor, Tensor]] = {}

    def _rope(self, h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
        key = (h, w, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_2d_tables(self.d_model, h, w, self.npe, device)
        return self._rope_cache[key]

    def forward(self, feat0: Tensor, feat1: Tensor) -> tuple[Tensor, Tensor]:
        rope0 = rope1 = None
        if self.rope:
            h0, w0 = feat0.shape[-2] // self.agg_size0, feat0.shape[-1] // self.agg_size0
            h1, w1 = feat1.shape[-2] // self.agg_size0, feat1.shape[-1] // self.agg_size0
            rope0 = self._rope(h0, w0, feat0.device)
            rope1 = rope0 if (h1, w1) == (h0, w0) else self._rope(h1, w1, feat1.device)
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, rope0)
                feat1 = layer(feat1, feat1, rope1)
            else:  # sequential like upstream: image 1 attends to the updated image-0 features
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
        return feat0, feat1


class FinePreprocess(nn.Module):
    """FPN upsampler turning coarse 1/8 + backbone 1/4 & 1/2 features into full-resolution (1/1) 64-ch maps."""

    def __init__(self, block_dims: tuple[int, int, int] = (64, 128, 256)) -> None:
        super().__init__()
        d1, d2, d3 = block_dims
        self.layer3_outconv = _conv1x1(d3, d3)
        self.layer2_outconv = _conv1x1(d2, d3)
        self.layer2_outconv2 = nn.Sequential(
            _conv3x3(d3, d3), nn.BatchNorm2d(d3), nn.LeakyReLU(), _conv3x3(d3, d2)
        )
        self.layer1_outconv = _conv1x1(d1, d2)
        self.layer1_outconv2 = nn.Sequential(
            _conv3x3(d2, d2), nn.BatchNorm2d(d2), nn.LeakyReLU(), _conv3x3(d2, d1)
        )

    def inter_fpn(self, feat_c: Tensor, x2: Tensor, x1: Tensor) -> Tensor:
        feat_c = F.interpolate(self.layer3_outconv(feat_c), scale_factor=2.0, mode="bilinear", align_corners=False)
        x2 = self.layer2_outconv2(self.layer2_outconv(x2) + feat_c)
        x2 = F.interpolate(x2, scale_factor=2.0, mode="bilinear", align_corners=False)
        x1 = self.layer1_outconv2(self.layer1_outconv(x1) + x2)
        return F.interpolate(x1, scale_factor=2.0, mode="bilinear", align_corners=False)


def _inner_cells(height: int, width: int, border: int, device: torch.device) -> Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[border : height - border, border : width - border] = True
    return mask.reshape(-1)


class ELoFTR(ImageMatcher, nn.Module):
    """Efficient LoFTR semi-dense matcher.

    Module paths match upstream checkpoints (``backbone``, ``loftr_coarse``, ``fine_preprocess``) and load strictly.
    ``match(img_a, img_b)`` -> :class:`~flashim.types.Matches` in each original image's pixel coordinates.
    """

    div_by = 32
    fold_temperature = False
    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None
    fp32_modules: tuple[str, ...] = ()
    compute_dtype = torch.float32

    def __init__(self, config: ELoFTRConfig | None = None) -> None:
        super().__init__()
        self.config = config or ELoFTRConfig()
        c = self.config
        self.threshold = c.resolved_threshold
        self.border_rm = c.border_rm
        self.skip_softmax = c.resolved_skip_softmax
        self.fp16matmul = c.resolved_fp16matmul
        self.W = c.fine_window
        self.local_regress_temperature = c.local_regress_temperature
        self.local_regress_slicedim = c.local_regress_slicedim

        self.backbone = RepVGG_8_1_align(c.block_dims)
        self.loftr_coarse = LocalFeatureTransformer(
            c.d_model, c.nhead, c.layer_names, c.agg_size0, c.agg_size1, c.rope, c.npe
        )
        self.fine_preprocess = FinePreprocess(c.block_dims)

    @classmethod
    def from_pretrained(
        cls,
        *,
        variant: str = "full",
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        **arch,
    ) -> ELoFTR:
        if "npe" in arch and isinstance(arch["npe"], list):
            arch["npe"] = tuple(arch["npe"])
        if "block_dims" in arch and isinstance(arch["block_dims"], list):
            arch["block_dims"] = tuple(arch["block_dims"])
        if "layer_names" in arch and isinstance(arch["layer_names"], list):
            arch["layer_names"] = tuple(arch["layer_names"])
        cfg = ELoFTRConfig(variant=variant, **arch)
        with torch.device("meta"):
            model = cls(cfg)
        model.load_state_dict(weights.load_state_dict(checkpoint), strict=True, assign=True)
        return model.to(torch.device(device or default_device())).eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.to(dtype)
        self.compute_dtype = dtype

    def reparameterize(self) -> list[str]:
        """Fuse every :class:`RepVGGBlock`'s 3x3 + 1x1 + identity branches and BatchNorms into a single 3x3 conv."""
        if self.training:
            raise RuntimeError("reparameterize requires model.eval()")
        fused = []
        for name, module in self.named_modules():
            if isinstance(module, RepVGGBlock) and module.switch_to_deploy():
                fused.append(name)
        return fused

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return [
            ("fine_preprocess.layer2_outconv2.0", "fine_preprocess.layer2_outconv2.1"),
            ("fine_preprocess.layer1_outconv2.0", "fine_preprocess.layer1_outconv2.1"),
        ]

    fold_batchnorm = sparse.SparseExtractor.fold_batchnorm

    def _coarse_match(
        self, a: Tensor, b: Tensor, hc0: int, wc0: int, hc1: int, wc1: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.skip_softmax:
            a_m, b_m = (a, b) if self.fp16matmul else (a.float(), b.float())
            if self.fold_temperature:
                sim = torch.einsum("nlc,nsc->nls", a_m / 0.1, b_m)
            else:
                sim = torch.einsum("nlc,nsc->nls", a_m, b_m) / 0.1
            conf, all_j = sim[0].max(dim=1)
            best_row = sim[0].max(dim=0).indices
            del sim
            mutual = best_row[all_j] == torch.arange(conf.shape[0], device=conf.device)
        else:
            a_f, b_f = a.float(), b.float()
            if self.fold_temperature:
                sim = torch.einsum("nlc,nsc->nls", a_f / 0.1, b_f)
            else:
                sim = torch.einsum("nlc,nsc->nls", a_f, b_f) / 0.1
            log_conf, all_j, mutual = ops.dual_softmax_mutual(sim[0])
            del sim
            conf = log_conf.exp()

        valid = mutual & (conf > self.threshold)
        r = self.border_rm
        if r > 0:
            inner0 = _inner_cells(hc0, wc0, r, conf.device)
            inner1 = _inner_cells(hc1, wc1, r, conf.device)
            valid &= inner0 & inner1[all_j]
        i_ids = torch.nonzero(valid)[:, 0]
        j_ids = all_j[i_ids]
        return i_ids, j_ids, conf[i_ids].float()

    def _gather_windows(
        self, feat0: Tensor, feat1: Tensor, i_ids: Tensor, j_ids: Tensor, hc0: int, wc0: int, hc1: int, wc1: int
    ) -> tuple[Tensor, Tensor]:
        """Gather only the matched fine windows (bit-identical to ``F.unfold`` + indexing, without materializing all
        ``H/8 * W/8`` windows)."""
        W = self.W
        stride = feat0.shape[-2] // hc0
        c = feat0.shape[1]
        # feat0: kernel=(W, W), stride=stride, padding=0 (stride == W == 8 -> non-overlapping blocks)
        if stride == W:
            iy, ix = i_ids // wc0, i_ids % wc0
            fw0 = feat0[0].reshape(c, hc0, W, wc0, W).permute(1, 3, 2, 4, 0)[iy, ix].reshape(-1, W * W, c)
        else:
            u0 = F.unfold(feat0, kernel_size=(W, W), stride=stride, padding=0)
            fw0 = u0.reshape(1, c, W * W, -1).permute(0, 3, 2, 1)[0, i_ids]

        # feat1: kernel=(W+2, W+2), stride=stride, padding=1
        w1 = W + 2
        pad1 = F.pad(feat1, (1, 1, 1, 1))
        jy, jx = j_ids // wc1, j_ids % wc1
        view1 = pad1[0].unfold(1, w1, stride).unfold(2, w1, stride)  # [C, hc1, wc1, w1, w1]
        fw1 = view1.permute(1, 2, 3, 4, 0)[jy, jx].reshape(-1, w1 * w1, c)
        return fw0, fw1

    def _fine_match(
        self, fw0: Tensor, fw1: Tensor, kpts0: Tensor, kpts1: Tensor, scale0: float, scale1: float
    ) -> tuple[Tensor, Tensor]:
        """Two-stage fine matching: 1) dual-softmax pixel match inside WxW; 2) 3x3 sub-pixel expectation on the last
        ``local_regress_slicedim`` channels."""
        M, WW, C = fw0.shape
        W = self.W
        sdim = self.local_regress_slicedim
        fw0_f, fw1_f = fw0.float(), fw1.float()

        feat_f0, feat_f1 = fw0_f[..., :-sdim] / (C**0.5), fw1_f[..., :-sdim] / (C**0.5)
        feat_ff0, feat_ff1 = fw0_f[..., -sdim:], fw1_f[..., -sdim:] / (sdim**0.5)

        conf_f = torch.einsum("mlc,mrc->mlr", feat_f0, feat_f1)
        conf_ff = torch.einsum("mlc,mrc->mlr", feat_ff0, feat_ff1)

        # Stage 1: pixel-level dual softmax, crop 1-px border from the (W+2)x(W+2) right window
        sm_f = F.softmax(conf_f, dim=1) * F.softmax(conf_f, dim=2)
        sm_f = sm_f.reshape(M, WW, W + 2, W + 2)[..., 1:-1, 1:-1].reshape(M, WW * WW)
        idx = sm_f.argmax(dim=-1)
        idx_l, idx_r = idx // WW, idx % WW

        offset = W // 2 - 0.5
        delta_l = torch.stack([(idx_l % W).float() - offset, (idx_l // W).float() - offset], dim=-1)
        delta_r = torch.stack([(idx_r % W).float() - offset, (idx_r // W).float() - offset], dim=-1)
        kpts0 = kpts0 + delta_l * scale0
        kpts1 = kpts1 + delta_r * scale1

        # Stage 2: 3x3 local regression around idx_r in the (W+2)x(W+2) similarity map.
        # Matches upstream's trained indexing directly (idx_r in [0, W-1] + {-1, 0, +1} into W+2).
        r_row, r_col = idx_r // W, idx_r % W
        d3 = torch.tensor((-1, 0, 1), dtype=torch.long, device=fw0.device)
        m_ids = torch.arange(M, dtype=torch.long, device=fw0.device)[:, None, None]
        patch = conf_ff.reshape(M, WW, W + 2, W + 2)[
            m_ids, idx_l[:, None, None], r_row[:, None, None] + d3[:, None], r_col[:, None, None] + d3[None, :]
        ]
        heat = F.softmax(patch.reshape(M, 9) / self.local_regress_temperature, dim=-1).reshape(M, 3, 3)
        grid = torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float32, device=fw0.device)
        coords = torch.stack([(heat.sum(dim=1) * grid).sum(dim=1), (heat.sum(dim=2) * grid).sum(dim=1)], dim=-1)
        return kpts0, kpts1 + coords * scale1

    @torch.inference_mode()
    def match_prepared(self, gray0: Tensor, gray1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[1, 1, H, W]`` grayscale pair (sides multiples of 32) -> ``(kpts0 [M, 2], kpts1 [M, 2], conf [M])`` in
        the inputs' pixel coordinates."""
        h0, h1 = gray0.shape[-2], gray1.shape[-2]
        same_shape = gray0.shape == gray1.shape
        g0, g1 = gray0.to(self.compute_dtype), gray1.to(self.compute_dtype)

        if same_shape:
            c, x2, x1 = self.backbone(torch.cat([g0, g1], dim=0))
            c0, c1 = c.chunk(2, dim=0)
        else:
            c0, x2_0, x1_0 = self.backbone(g0)
            c1, x2_1, x1_1 = self.backbone(g1)

        (hc0, wc0), (hc1, wc1) = c0.shape[-2:], c1.shape[-2:]
        c0, c1 = self.loftr_coarse(c0, c1)

        # Normalize once: used by both coarse matching and fine_preprocess (matches upstream loftr.py).
        inv_sqrt_c = c0.shape[1] ** -0.5
        t0 = c0.permute(0, 2, 3, 1).reshape(1, -1, c0.shape[1]) * inv_sqrt_c
        t1 = c1.permute(0, 2, 3, 1).reshape(1, -1, c1.shape[1]) * inv_sqrt_c

        i_ids, j_ids, mconf = self._coarse_match(t0, t1, hc0, wc0, hc1, wc1)
        stride_c0, stride_c1 = h0 / hc0, h1 / hc1
        kpts0 = torch.stack([i_ids % wc0, i_ids // wc0], dim=1).float() * stride_c0
        kpts1 = torch.stack([j_ids % wc1, j_ids // wc1], dim=1).float() * stride_c1
        if len(i_ids) == 0:
            return kpts0, kpts1, mconf

        c0_norm = t0.permute(0, 2, 1).reshape(1, -1, hc0, wc0)
        c1_norm = t1.permute(0, 2, 1).reshape(1, -1, hc1, wc1)
        if same_shape:
            f0, f1 = self.fine_preprocess.inter_fpn(torch.cat([c0_norm, c1_norm], dim=0), x2, x1).chunk(2, dim=0)
        else:
            f0 = self.fine_preprocess.inter_fpn(c0_norm, x2_0, x1_0)
            f1 = self.fine_preprocess.inter_fpn(c1_norm, x2_1, x1_1)

        fw0, fw1 = self._gather_windows(f0, f1, i_ids, j_ids, hc0, wc0, hc1, wc1)
        kpts0, kpts1 = self._fine_match(fw0, fw1, kpts0, kpts1, h0 / f0.shape[-2], h1 / f1.shape[-2])
        return kpts0, kpts1, mconf

    def _match(
        self,
        img_a: ImageLike,
        img_b: ImageLike,
        *,
        resize: int | None,
        max_matches: int | None,
        threshold: float | None,
    ) -> Matches:
        """``resize``: long side of working resolution (sides floored to multiples of 32); ``max_matches``: keep the
        most confident matches (``None``: all); ``threshold``: coarse confidence threshold."""
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
        if max_matches is not None and max_matches < len(conf):
            conf, index = conf.topk(max_matches)
            k0, k1 = k0[index], k1[index]
        k0 = k0 * device_constant([sizes[0][1], sizes[0][0]], k0)
        k1 = k1 * device_constant([sizes[1][1], sizes[1][0]], k1)
        return Matches(k0, k1, conf)
