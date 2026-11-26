"""LoFTR (Sun et al., CVPR 2021): ResNetFPN 1/8 + 1/2 features, an 8-layer linear-attention transformer on the coarse
tokens, dual-softmax coarse matching, and a 5x5-window fine refinement."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.conv import conv_relu
from flashim.models import sparse
from flashim.models.base import ImageMatcher
from flashim.models.loftr import weights
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches
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
        y = conv_relu(self.conv1, x, self.bn1, feeds=self.conv2)
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
        self.fused_upsample: bool = False

    def forward(self, x: Tensor) -> list[Tensor]:
        x1 = self.layer1(self.relu(self.bn1(self.conv1(x))))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3_out = self.layer3_outconv(x3)
        x2_out = self.layer2_outconv(x2)
        if self.fused_upsample:
            x2_out = self.layer2_outconv2(ops.upsample_add(x3_out, x2_out, align_corners=True, out_dtype=x2_out.dtype))
            x1_out = self.layer1_outconv(x1)
            x1_out = self.layer1_outconv2(ops.upsample_add(x2_out, x1_out, align_corners=True, out_dtype=x1_out.dtype))
        else:
            x2_out = self.layer2_outconv2(x2_out + F.interpolate(x3_out, size=x2_out.shape[2:], mode="bilinear", align_corners=True))
            x1_out = self.layer1_outconv(x1)
            x1_out = self.layer1_outconv2(x1_out + F.interpolate(x2_out, size=x1_out.shape[2:], mode="bilinear", align_corners=True))
        return [x3_out, x1_out]


def linear_attention(q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-6, gemm_dtype: torch.dtype | None = None) -> Tensor:
    """ELU+1 kernel attention on ``[N, L, H, D]`` / ``[N, S, H, D]`` (pure-torch reference, used in FP32)."""
    qf, kf, vf = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0, v.float()
    length = vf.shape[1]
    z = 1.0 / (torch.einsum("nlhd,nhd->nlh", qf, kf.sum(dim=1)) + eps)
    if gemm_dtype is None:
        kv = torch.einsum("nshd,nshv->nhdv", kf, vf / length)
        return (torch.einsum("nlhd,nhdv,nlh->nlhv", qf, kv, z) * length).to(q.dtype)
    kv = torch.einsum("nshd,nshv->nhdv", kf.to(gemm_dtype), (vf / length).to(gemm_dtype))
    return (torch.einsum("nlhd,nhdv->nlhv", qf.to(gemm_dtype), kv).float() * z[..., None] * length).to(q.dtype)


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.dim, self.nhead = d_model // nhead, nhead
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(3))
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model, bias=False), nn.ReLU(True), nn.Linear(2 * d_model, d_model, bias=False))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attention_gemm_dtype: torch.dtype | None = None
        self.use_fused_linear_attention: bool = False
        self.fused_epilogues: bool = False
        self.qkv_proj: nn.Module | None = None
        self.kv_proj: nn.Module | None = None

    def _mlp_and_residual(self, x: Tensor, message: Tensor) -> Tensor:
        if not self.fused_epilogues:
            message = self._mlp_forward(torch.cat([x, self.norm1(message)], dim=2))
            return x + self.norm2(message)
        from flashim.quant.methods.int8_linear import DynamicInt8Linear, StaticInt8Linear
        from flashim.quant.methods.linear import StaticMXFP8Linear

        fc1, fc2 = self.mlp[0], self.mlp[2]
        if isinstance(fc1, DynamicInt8Linear):
            q, s = ops.layernorm_cat_quantize_int8_rows(
                x, message, self.norm1.weight, self.norm1.bias, self.norm1.eps, channel_scale=fc1.channel_scale
            )
            h = ops.int8_linear(
                q.reshape(-1, fc1.in_features), fc1.weight_int8, s.reshape(-1, 1), fc1.weight_scale, fc1.bias, relu=True
            ).reshape(*x.shape[:-1], fc1.out_features).to(x.dtype)
            out = fc2(h)
        elif isinstance(fc1, StaticInt8Linear) and fc1.channel_scale is None:
            q = ops.layernorm_cat_quantize_int8(
                x, message, self.norm1.weight, self.norm1.bias, self.norm1.eps, fc1.input_scale
            )
            out = fc2(fc1.forward_prequantized(q, relu=True).to(x.dtype))
        elif isinstance(fc1, StaticMXFP8Linear):
            q = ops.layernorm_cat_quantize_mxfp8(
                x, message, self.norm1.weight, self.norm1.bias, self.norm1.eps, fc1.input_exponent
            )
            out = fc2(F.relu(fc1.forward_prequantized(q).to(x.dtype)))
        else:
            out = self.mlp(ops.layernorm_cat(x, message, self.norm1.weight, self.norm1.bias, self.norm1.eps))
        return ops.layernorm_add(x, out, self.norm2.weight, self.norm2.bias, self.norm2.eps)

    def forward_self_pair(self, feat0: Tensor, feat1: Tensor) -> tuple[Tensor, Tensor]:
        """Batch token-wise projections/MLP across two different-length images while keeping self-attention separate."""
        l0, l1 = feat0.shape[1], feat1.shape[1]
        both = torch.cat((feat0, feat1), dim=1)
        qkv = self.qkv_proj(both)
        qkv0 = qkv[:, :l0].view(1, l0, 3, self.nhead, self.dim)
        qkv1 = qkv[:, l0:].view(1, l1, 3, self.nhead, self.dim)
        gemm_dt = torch.bfloat16 if qkv0.dtype in (torch.bfloat16, torch.float16) else None
        a0 = ops.elu_linear_attention(qkv0[:, :, 0], qkv0[:, :, 1], qkv0[:, :, 2], 1e-6, gemm_dt).reshape(1, l0, -1)
        a1 = ops.elu_linear_attention(qkv1[:, :, 0], qkv1[:, :, 1], qkv1[:, :, 2], 1e-6, gemm_dt).reshape(1, l1, -1)
        message = self.merge(torch.cat((a0, a1), dim=1))
        out = self._mlp_and_residual(both, message)
        return out[:, :l0].contiguous(), out[:, l0:].contiguous()

    def _project_qkv(self, x: Tensor, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        from flashim.quant.methods.int8_linear import StaticInt8Linear
        from flashim.quant.tensor import Int8Tensor

        qp, kp, vp = self.q_proj, self.k_proj, self.v_proj
        if (
            x is source
            and isinstance(qp, StaticInt8Linear)
            and isinstance(kp, StaticInt8Linear)
            and isinstance(vp, StaticInt8Linear)
            and qp.channel_scale is None and kp.channel_scale is None and vp.channel_scale is None
            and x.is_contiguous()
        ):
            flat = x.reshape(-1, x.shape[-1])
            qq, qk, qv = (torch.empty(flat.shape, device=x.device, dtype=torch.int8) for _ in range(3))
            ops.quantize_int8_multi(flat, [qp.input_scale, kp.input_scale, vp.input_scale], [qq, qk, qv])
            q = qp.forward_prequantized(Int8Tensor(qq.reshape(x.shape), qp.input_scale)).to(x.dtype)
            k = kp.forward_prequantized(Int8Tensor(qk.reshape(x.shape), kp.input_scale)).to(x.dtype)
            v = vp.forward_prequantized(Int8Tensor(qv.reshape(x.shape), vp.input_scale)).to(x.dtype)
            return q, k, v
        if (
            x is not source
            and isinstance(kp, StaticInt8Linear)
            and isinstance(vp, StaticInt8Linear)
            and kp.channel_scale is None and vp.channel_scale is None
            and source.is_contiguous()
        ):
            q = qp(x)
            flat = source.reshape(-1, source.shape[-1])
            qk, qv = (torch.empty(flat.shape, device=source.device, dtype=torch.int8) for _ in range(2))
            ops.quantize_int8_multi(flat, [kp.input_scale, vp.input_scale], [qk, qv])
            k = kp.forward_prequantized(Int8Tensor(qk.reshape(source.shape), kp.input_scale)).to(source.dtype)
            v = vp.forward_prequantized(Int8Tensor(qv.reshape(source.shape), vp.input_scale)).to(source.dtype)
            return q, k, v
        return qp(x), kp(source), vp(source)

    def _mlp_forward(self, x: Tensor) -> Tensor:
        from flashim.quant.methods.int8_linear import StaticInt8Linear

        first, _, last = self.mlp
        if isinstance(first, StaticInt8Linear):
            return last(first.forward_prequantized(first.quantize(x), relu=True).to(x.dtype))
        return self.mlp(x)

    def forward(self, x: Tensor, source: Tensor) -> Tensor:
        bs = x.shape[0]
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x).view(bs, -1, 3, self.nhead, self.dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        elif self.kv_proj is not None:
            q = self.q_proj(x).reshape(bs, -1, self.nhead, self.dim)
            kv = self.kv_proj(source).view(bs, -1, 2, self.nhead, self.dim)
            k, v = kv[:, :, 0], kv[:, :, 1]
        else:
            q, k, v = self._project_qkv(x, source)
            q = q.reshape(bs, -1, self.nhead, self.dim)
            k = k.reshape(bs, -1, self.nhead, self.dim)
            v = v.reshape(bs, -1, self.nhead, self.dim)
        if self.use_fused_linear_attention:
            gemm_dt = (
                self.attention_gemm_dtype
                if self.attention_gemm_dtype is not None
                else (torch.bfloat16 if q.dtype in (torch.bfloat16, torch.float16) else None)
            )
            attn_out = ops.elu_linear_attention(q, k, v, 1e-6, gemm_dt)
        else:
            attn_out = linear_attention(q, k, v, gemm_dtype=self.attention_gemm_dtype).to(x.dtype)
        message = self.merge(attn_out.reshape(bs, -1, self.nhead * self.dim))
        return self._mlp_and_residual(x, message)


class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, layer_names: list[str]) -> None:
        super().__init__()
        self.layer_names = layer_names
        self.layers = nn.ModuleList(LoFTREncoderLayer(d_model, nhead) for _ in layer_names)
        self.batch_self: bool = False

    def forward(self, feat0: Tensor, feat1: Tensor) -> tuple[Tensor, Tensor]:
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                if self.batch_self and feat0.shape == feat1.shape:
                    both = torch.cat((feat0, feat1), dim=0)
                    both = layer(both, both)
                    feat0, feat1 = both[: feat0.shape[0]], both[feat0.shape[0] :]
                elif self.batch_self and layer.qkv_proj is not None and layer.use_fused_linear_attention and feat0.shape[0] == 1:
                    feat0, feat1 = layer.forward_self_pair(feat0, feat1)
                else:
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


class LoFTR(ImageMatcher, nn.Module):
    """Module paths follow the official checkpoints (``backbone``, ``loftr_coarse``, ``fine_preprocess``, ``loftr_fine``);
    loads strictly. ``match(img_a, img_b)`` -> :class:`Matches` in each original image's pixel coordinates
    (LoFTR's convention: coarse cell index x 8 plus the fine offset).

    Precision presets: the backbone and both transformers run in the compute dtype; the attention core, the coarse
    similarity / dual softmax (temperature 0.1) and the fine expectation stay FP32."""

    div_by = 8
    fold_temperature = False  # see spec.py's fold_similarity_temperature pass
    bf16_similarity = False   # see spec.py's bf16_coarse_similarity pass (ops.matmul_nt_fp32)
    batch_backbone = False    # see spec.py's batch_backbone pass (minima-loftr only; regresses loftr int8/fp8)
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
        self._pe: OrderedDict[tuple[int, int], Tensor] = OrderedDict()
        self._inner_cache: OrderedDict[tuple[int, int, int, torch.device], Tensor] = OrderedDict()

    @classmethod
    def from_pretrained(cls, *, variant: str = "outdoor", checkpoint: str | Path | None = None,
                        device: torch.device | str | None = None, **kwargs) -> LoFTR:
        temp_bug_fix = kwargs.pop("temp_bug_fix", variant != "outdoor")
        with torch.device("meta"):
            model = cls(temp_bug_fix=temp_bug_fix, **kwargs)
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
        key = (h, w)
        pe = self._pe.get(key)
        if pe is None:
            pe = position_encoding(256, h, w, self.temp_bug_fix, self.device)
            self._pe[key] = pe
            while len(self._pe) > 16:
                self._pe.popitem(last=False)
        else:
            self._pe.move_to_end(key)
        return pe

    def coarse_features(self, gray: Tensor) -> tuple[Tensor, Tensor]:
        feat_c, feat_f = self.backbone(gray.to(self.compute_dtype))
        return feat_c, feat_f

    def _cached_inner_cells(self, h: int, w: int, border: int, device: torch.device) -> Tensor:
        key = (h, w, border, device)
        mask = self._inner_cache.get(key)
        if mask is None:
            mask = _inner_cells(h, w, border, device)
            self._inner_cache[key] = mask
            while len(self._inner_cache) > 32:
                self._inner_cache.popitem(last=False)
        else:
            self._inner_cache.move_to_end(key)
        return mask

    @torch.inference_mode()
    def match_prepared(self, gray0: Tensor, gray1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[1, 1, H, W]`` grayscale pair (sides multiples of 8) -> matches ``(kpts0 [M, 2], kpts1 [M, 2], conf [M])`` in
        the inputs' pixel coordinates."""
        h0, h1 = gray0.shape[-2], gray1.shape[-2]
        if self.batch_backbone and gray0.shape == gray1.shape:
            c, f = self.coarse_features(torch.cat((gray0, gray1), dim=0))
            c0, c1, f0, f1 = c[:1], c[1:], f[:1], f[1:]
        else:
            # Keep B=1 x2 for loftr: B=2 regressed quantized presets heavily on sm_120 (int8 44.5->76.9 ms, fp8 48.4->181.4 ms).
            (c0, f0), (c1, f1) = self.coarse_features(gray0), self.coarse_features(gray1)
        (hc0, wc0), (hc1, wc1) = c0.shape[-2:], c1.shape[-2:]
        tokens0 = (c0.float() + self.positions(hc0, wc0)).to(c0.dtype).permute(0, 2, 3, 1).reshape(1, -1, c0.shape[1])
        tokens1 = (c1.float() + self.positions(hc1, wc1)).to(c1.dtype).permute(0, 2, 3, 1).reshape(1, -1, c1.shape[1])
        tokens0, tokens1 = self.loftr_coarse(tokens0, tokens1)

        # coarse matching: dual softmax over the normalized similarity, mutual maxima above the threshold. The
        # ``[L, S]`` probabilities are never formed (``ops.dual_softmax_mutual``): measured 35 -> 4 ms and 2.1 GB less
        # peak memory at 1200 px (16.8k x 16.5k tokens) with identical matches.
        if self.bf16_similarity and tokens0.dtype in (torch.bfloat16, torch.float16):
            sim = ops.matmul_nt_fp32(tokens0[0], tokens1[0], 1.0 / (tokens0.shape[-1] * 0.1))
            log_conf, all_j, mutual = ops.dual_softmax_mutual(sim)
        else:
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
            inner0 = self._cached_inner_cells(hc0, wc0, r, conf.device)
            inner1 = self._cached_inner_cells(hc1, wc1, r, conf.device)
            valid &= inner0 & inner1[all_j]
        i_ids = torch.nonzero(valid)[:, 0]
        j_ids = all_j[i_ids]
        mconf = conf[i_ids]
        stride_c0, stride_c1 = h0 / hc0, h1 / hc1
        kpts0 = torch.stack([i_ids % wc0, i_ids // wc0], dim=1).float() * stride_c0
        kpts1 = torch.stack([j_ids % wc1, j_ids // wc1], dim=1).float() * stride_c1
        if len(i_ids) == 0:
            return kpts0, kpts1, mconf

        # fine refinement: 5x5 windows of the 1/2 features gathered directly at coarse matches (bit-identical to
        # F.unfold(feat, (W, W), stride, pad)[ids] without materializing the [C*W*W, hc*wc] unfolded map).
        W = self.W
        stride = f0.shape[2] // hc0
        pad = W // 2
        offs = torch.arange(-pad, pad + 1, device=f0.device)

        def gather_windows(feat: Tensor, ids: Tensor, wc: int) -> Tensor:
            _, c, hf, wf = feat.shape
            cy, cx = ids // wc, ids % wc
            iy = (cy * stride)[:, None, None] + offs[None, :, None]
            ix = (cx * stride)[:, None, None] + offs[None, None, :]
            inside = (iy >= 0) & (iy < hf) & (ix >= 0) & (ix < wf)
            flat_idx = (iy.clamp(0, hf - 1) * wf + ix.clamp(0, wf - 1)).reshape(-1)
            if feat.is_contiguous(memory_format=torch.channels_last):
                out = feat[0].permute(1, 2, 0).reshape(hf * wf, c)[flat_idx].reshape(len(ids), W * W, c)
            else:
                out = feat[0].reshape(c, hf * wf)[:, flat_idx].transpose(0, 1).reshape(len(ids), W * W, c)
            return out.masked_fill(~inside.reshape(len(ids), W * W, 1), 0)

        fw0, fw1 = gather_windows(f0, i_ids, wc0), gather_windows(f1, j_ids, wc1)
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

    def _match(self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None, max_matches: int | None,
               threshold: float | None) -> Matches:
        """``resize``: long side of the working resolution (sides floored to multiples of 8); ``max_matches``: keep the
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
