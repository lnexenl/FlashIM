"""OmniGlue matcher (Jiang et al., CVPR 2024): Keypoint Position-Guided Attention with DINOv2 masking and
100-iteration Log-Sinkhorn optimal transport.

Architecture and parameter mapping match Google Research's released ``og_export`` TF2 SavedModel:

- ``kenc``: 5-layer MLP ``3 -> 32 -> 64 -> 128 -> 256 -> 256`` with ``BatchNorm1d(eps=1e-5) + ReLU`` after
  the first four layers (folded into their preceding ``Linear`` layers in ``pre_quant``).
- ``gnn``: 18 ``AttentionalPropagation`` layers alternating self-attention (even) and DINOv2-masked
  cross-attention (odd). Unlike SuperGlue, spatial encoding ``kenc`` is added freshly to Q and K at every layer
  while V and the residual stream remain pure appearance descriptors (Keypoint Position-Guided Attention).
- ``final_proj`` (256 -> 256) + scaled dot product (``/ sqrt(256)``) + learned dustbin ``bin_score`` +
  100-iteration Log-Sinkhorn optimal transport (``ops.sinkhorn_mutual``).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.log import warn_once
from flashim.models.omniglue import weights
from flashim.models.omniglue.config import OmniGlueConfig
from flashim.platforms import default_device
from flashim.quant.methods.int8_linear import DynamicInt8Linear
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant

logger = logging.getLogger(__name__)


@dataclass
class OmniGlueFeatures(SparseFeatures):
    """Per-image SuperPoint features + bilinearly sampled DINOv2 ViT-B/14 descriptors ``[B, K, 768]``."""

    dino_descriptors: Tensor = None  # type: ignore[assignment]


class ChannelBatchNorm1d(nn.BatchNorm1d):
    """``nn.BatchNorm1d`` operating on the last dimension of ``[B, N, C]`` tensors (keeps standard BN state keys)."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def _fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """Exact eval-mode fold of ``bn(linear(x))`` into a single :class:`nn.Linear`."""
    with torch.no_grad():
        scale = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
        w = linear.weight.float() * scale[:, None]
        b = linear.bias.float() if linear.bias is not None else torch.zeros_like(bn.running_mean, dtype=torch.float32)
        b = (b - bn.running_mean.float()) * scale + bn.bias.float()
        fused = nn.Linear(
            linear.in_features, linear.out_features, bias=True, device=linear.weight.device, dtype=linear.weight.dtype
        )
        fused.weight.copy_(w.to(linear.weight.dtype))
        fused.bias.copy_(b.to(linear.weight.dtype))
        return fused


def normalize_keypoints(keypoints: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Upstream OmniGlue keypoint normalization: ``(kpt - [W/2, H/2]) / (0.7 * max(W, H))``."""
    height, width = image_size
    center = device_constant([width / 2.0, height / 2.0], keypoints, torch.float32)
    scale = float(max(width, height)) * float(torch.tensor(0.7, dtype=torch.float32))
    return (keypoints.float() - center) / scale


def dino_attention_mask(d0: Tensor, d1: Tensor) -> tuple[Tensor, Tensor]:
    """DINOv2-guided inter-image attention masks (``__inference_get_dino_attn_mask_2321``).

    Returns boolean masks ``(mask0 [B, 1, N, M], mask1 [B, 1, M, N])`` where ``True`` marks pairs strictly
    above the ``ceil(0.5 * L)``-th largest cosine similarity in each row (matching TF ``SelectV2(sim > thresh, 0, -inf)``).
    """
    d0 = d0.float()
    d1 = d1.float()
    n0 = d0 * torch.rsqrt((d0 * d0).sum(dim=-1, keepdim=True).clamp(min=1e-12))
    n1 = d1 * torch.rsqrt((d1 * d1).sum(dim=-1, keepdim=True).clamp(min=1e-12))
    sim = torch.bmm(n0, n1.transpose(1, 2))  # [B, N, M]
    n, m = sim.shape[1], sim.shape[2]
    k0 = math.ceil(0.5 * m)
    k1 = math.ceil(0.5 * n)
    thresh0 = torch.topk(sim, k0, dim=-1).values[..., -1:]
    mask0 = (sim > thresh0).unsqueeze(1)
    sim_t = sim.transpose(1, 2)
    thresh1 = torch.topk(sim_t, k1, dim=-1).values[..., -1:]
    mask1 = (sim_t > thresh1).unsqueeze(1)
    return mask0, mask1


class _KeypointLayer(nn.Module):
    def __init__(self, cin: int, cout: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.linear = nn.Linear(cin, cout)
        self.bn: nn.Module = ChannelBatchNorm1d(cout, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.linear(x)))


class KeypointEncoder(nn.Module):
    """5-layer MLP mapping ``[x_norm, y_norm, score]`` -> ``[B, N, 256]``."""

    def __init__(self, dims: tuple[int, ...] = (32, 64, 128, 256), out_dim: int = 256, eps: float = 1e-5) -> None:
        super().__init__()
        channels = (3, *dims)
        self.layers = nn.ModuleList(_KeypointLayer(channels[i], channels[i + 1], eps=eps) for i in range(len(dims)))
        self.out = nn.Linear(dims[-1], out_dim)

    def forward(self, kpts_norm: Tensor, scores: Tensor) -> Tensor:
        x = torch.cat((kpts_norm.float(), scores.float().unsqueeze(-1)), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class MultiHeadAttention(nn.Module):
    """4-head attention with separate Q/K/V/Out projections (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``).

    :meth:`pack_qk` merges ``q_proj`` and ``k_proj`` (which always consume the same ``desc + kenc`` input)
    into one ``qk_proj`` linear (``dim -> 2 * dim``), halving Q/K GEMM and dynamic-INT8 quantization launches.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj: nn.Module | None = nn.Linear(dim, dim)
        self.k_proj: nn.Module | None = nn.Linear(dim, dim)
        self.qk_proj: nn.Module | None = None
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.sdpa_impl: str | None = "reference"

    def pack_qk(self) -> bool:
        """Exact merge of ``q_proj`` + ``k_proj`` into ``qk_proj`` (called in ``pre_quant``)."""
        if self.qk_proj is not None or not (isinstance(self.q_proj, nn.Linear) and isinstance(self.k_proj, nn.Linear)):
            return False
        q, k = self.q_proj, self.k_proj
        fused = nn.Linear(
            q.in_features,
            q.out_features + k.out_features,
            bias=True,
            device=q.weight.device,
            dtype=q.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(torch.cat((q.weight, k.weight), dim=0))
            fused.bias.copy_(torch.cat((q.bias, k.bias), dim=0))
        self.qk_proj = fused
        self.q_proj = None
        self.k_proj = None
        return True

    def _heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def project_qk(self, qk_in: Tensor) -> tuple[Tensor, Tensor]:
        if self.qk_proj is not None:
            q, k = self.qk_proj(qk_in).chunk(2, dim=-1)
            return self._heads(q), self._heads(k)
        assert self.q_proj is not None and self.k_proj is not None
        return self._heads(self.q_proj(qk_in)), self._heads(self.k_proj(qk_in))

    def forward(
        self, q_in: Tensor, k_in: Tensor, v_in: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        if q_in is k_in:
            q, k = self.project_qk(q_in)
        else:
            assert self.q_proj is not None and self.k_proj is not None
            q, k = self._heads(self.q_proj(q_in)), self._heads(self.k_proj(k_in))
        v = self._heads(self.v_proj(v_in))
        if attn_mask is None:
            out = ops.sdpa.get(self.sdpa_impl)(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        b, _, n, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim))


class AttentionalPropagation(nn.Module):
    """One OmniGlue attentional aggregation layer: multi-head attention + ``[desc, msg] -> 512 -> BN -> ReLU -> 256``."""

    def __init__(self, dim: int, num_heads: int, *, cross: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.cross = cross
        self.attn = MultiHeadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            ChannelBatchNorm1d(2 * dim, eps=eps),
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim, dim),
        )

    def _mlp_residual(self, desc: Tensor, msg: Tensor) -> Tensor:
        x = torch.cat((desc, msg), dim=-1)
        m0, m1, _, m3 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        if isinstance(m0, DynamicInt8Linear) and isinstance(m1, nn.Identity) and isinstance(m3, DynamicInt8Linear):
            rows = x.reshape(-1, x.shape[-1])
            q0, s0 = ops.quantize_int8_rows(rows if rows.is_contiguous() else rows.contiguous())
            h = ops.int8_linear(q0, m0.weight_int8, s0, m0.weight_scale, m0.bias, relu=True)
            q1, s1 = ops.quantize_int8_rows(h)
            res = desc.reshape(-1, desc.shape[-1])
            out = ops.int8_linear(
                q1, m3.weight_int8, s1, m3.weight_scale, m3.bias,
                residual=res if res.is_contiguous() else res.contiguous(),
            )
            return out.reshape_as(desc).to(desc.dtype)
        return desc + self.mlp(x)

    def forward_packed(self, desc: Tensor, kenc: Tensor, mask: Tensor | None = None) -> Tensor:
        """Packed ``[2, N, C]`` forward (equal keypoint counts, avoids per-layer ``cat`` / ``chunk``)."""
        qk_in = desc + kenc
        q, k = self.attn.project_qk(qk_in)
        v = self.attn._heads(self.attn.v_proj(desc))
        if not self.cross:
            att = ops.sdpa.get(self.attn.sdpa_impl)(q, k, v)
        else:
            k_swap = torch.cat((k[1:2], k[0:1]), dim=0)
            v_swap = torch.cat((v[1:2], v[0:1]), dim=0)
            att = F.scaled_dot_product_attention(q, k_swap, v_swap, attn_mask=mask, dropout_p=0.0)
        b2, _, n, _ = att.shape
        msg = self.attn.out_proj(att.transpose(1, 2).reshape(b2, n, -1))
        return self._mlp_residual(desc, msg)

    def forward(
        self,
        desc0: Tensor,
        desc1: Tensor,
        kenc0: Tensor,
        kenc1: Tensor,
        mask0: Tensor | None = None,
        mask1: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if desc0.shape == desc1.shape:
            desc = torch.cat((desc0, desc1), dim=0)
            kenc = torch.cat((kenc0, kenc1), dim=0)
            mask = torch.cat((mask0, mask1), dim=0) if (self.cross and mask0 is not None and mask1 is not None) else None
            return self.forward_packed(desc, kenc, mask).chunk(2, dim=0)

        qk0, qk1 = desc0 + kenc0, desc1 + kenc1
        q0, k0 = self.attn.project_qk(qk0)
        q1, k1 = self.attn.project_qk(qk1)
        v0, v1 = self.attn._heads(self.attn.v_proj(desc0)), self.attn._heads(self.attn.v_proj(desc1))
        if not self.cross:
            a0 = ops.sdpa.get(self.attn.sdpa_impl)(q0, k0, v0)
            a1 = ops.sdpa.get(self.attn.sdpa_impl)(q1, k1, v1)
        else:
            a0 = F.scaled_dot_product_attention(q0, k1, v1, attn_mask=mask0, dropout_p=0.0)
            a1 = F.scaled_dot_product_attention(q1, k0, v0, attn_mask=mask1, dropout_p=0.0)
        m0 = self.attn.out_proj(a0.transpose(1, 2).reshape(desc0.shape))
        m1 = self.attn.out_proj(a1.transpose(1, 2).reshape(desc1.shape))
        return self._mlp_residual(desc0, m0), self._mlp_residual(desc1, m1)


class AttentionalGNN(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            AttentionalPropagation(dim, num_heads, cross=(i % 2 == 1), eps=eps) for i in range(2 * num_layers)
        )

    def forward(
        self,
        desc0: Tensor,
        desc1: Tensor,
        kenc0: Tensor,
        kenc1: Tensor,
        mask0: Tensor,
        mask1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if desc0.shape == desc1.shape:
            desc = torch.cat((desc0, desc1), dim=0)
            kenc = torch.cat((kenc0, kenc1), dim=0)
            mask = torch.cat((mask0, mask1), dim=0)
            for layer in self.layers:
                desc = layer.forward_packed(desc, kenc, mask if layer.cross else None)
            return desc.chunk(2, dim=0)
        for layer in self.layers:
            desc0, desc1 = layer(desc0, desc1, kenc0, kenc1, mask0, mask1)
        return desc0, desc1


def log_optimal_transport(scores: Tensor, bin_score: Tensor, iters: int = 100) -> Tensor:
    """Log-space Sinkhorn optimal transport with a learned dustbin score (SuperGlue / OmniGlue, FP32)."""
    scores = scores.float()
    b, m, n = scores.shape
    alpha = bin_score.float()
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha_corner = alpha.expand(b, 1, 1)
    couplings = torch.cat(
        (torch.cat((scores, bins0), dim=-1), torch.cat((bins1, alpha_corner), dim=-1)),
        dim=1,
    )
    norm = -math.log(m + n)
    log_mu = torch.empty((b, m + 1), device=scores.device, dtype=torch.float32)
    log_mu[:, :m] = norm
    log_mu[:, m] = math.log(n) + norm
    log_nu = torch.empty((b, n + 1), device=scores.device, dtype=torch.float32)
    log_nu[:, :n] = norm
    log_nu[:, n] = math.log(m) + norm

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
    return torch.exp(couplings + u.unsqueeze(2) + v.unsqueeze(1) - norm)


class OmniGlueMatcher(nn.Module):
    """OmniGlue sparse matcher: :class:`OmniGlueFeatures` x :class:`OmniGlueFeatures` -> :class:`Matches`."""

    low_precision_similarity: bool = False
    graphs: int = 0

    def __init__(self, config: OmniGlueConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or OmniGlueConfig()
        self.kenc = KeypointEncoder(cfg.keypoint_encoder, cfg.descriptor_dim, eps=cfg.bn_eps)
        self.gnn = AttentionalGNN(cfg.descriptor_dim, cfg.num_heads, cfg.num_layers, eps=cfg.bn_eps)
        self.final_proj = nn.Linear(cfg.descriptor_dim, cfg.descriptor_dim)
        self.bin_score = nn.Parameter(torch.tensor(1.0))

        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None
        self.compute_dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        config: OmniGlueConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> OmniGlueMatcher:
        device = torch.device(device or default_device())
        config = config or OmniGlueConfig()
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_matcher_state_dict(checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.bin_score.device

    def fold_batchnorm(self) -> list[str]:
        """Fold every ``Linear + BatchNorm1d`` pair in ``kenc`` (4) and ``gnn`` (18); exact in eval mode."""
        if self.training:
            raise RuntimeError("fold_batchnorm requires model.eval()")
        folded: list[str] = []
        for i, layer in enumerate(self.kenc.layers):
            if isinstance(layer.bn, nn.BatchNorm1d):
                layer.linear = _fold_linear_bn(layer.linear, layer.bn)
                layer.bn = nn.Identity()
                folded.append(f"kenc.layers.{i}.linear")
        for i, layer in enumerate(self.gnn.layers):
            if isinstance(layer.mlp[1], nn.BatchNorm1d):
                layer.mlp[0] = _fold_linear_bn(layer.mlp[0], layer.mlp[1])
                layer.mlp[1] = nn.Identity()
                folded.append(f"gnn.layers.{i}.mlp.0")
        return folded

    def pack_qk(self) -> list[str]:
        """Merge ``q_proj`` and ``k_proj`` into ``qk_proj`` in every GNN layer."""
        return [
            f"gnn.layers.{i}.attn"
            for i, layer in enumerate(self.gnn.layers)
            if layer.attn.pack_qk()
        ]

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the GNN and final projection in ``dtype``; keep keypoint encoder and Sinkhorn in FP32."""
        self.gnn.to(dtype)
        self.final_proj.to(dtype)
        self.compute_dtype = dtype

    def _similarity(self, d0: Tensor, d1: Tensor) -> Tensor:
        """``[B, M, N]`` FP32 similarity ``final_proj(d0) @ final_proj(d1)^T / sqrt(dim)``."""
        scale = self.config.descriptor_dim**-0.5
        if self.low_precision_similarity and self.compute_dtype != torch.float32:
            m0, m1 = self.final_proj(d0), self.final_proj(d1)
            sims = [ops.matmul_nt_fp32(a, b, scale) for a, b in zip(m0, m1)]
            return sims[0][None] if len(sims) == 1 else torch.stack(sims)
        m0 = self.final_proj(d0).float()
        m1 = self.final_proj(d1).float()
        return torch.bmm(m0, m1.transpose(1, 2)) * scale

    def soft_assignment(
        self,
        keypoints0: Tensor,
        keypoints1: Tensor,
        descriptors0: Tensor,
        descriptors1: Tensor,
        scores0: Tensor,
        scores1: Tensor,
        dino0: Tensor,
        dino1: Tensor,
        image_size0: tuple[int, int],
        image_size1: tuple[int, int],
    ) -> Tensor:
        """Compute full soft assignment matrix ``[B, N+1, M+1]`` (matching TF ``serving_default`` output)."""
        mask0, mask1 = dino_attention_mask(dino0, dino1)
        kpts0_norm = normalize_keypoints(keypoints0, image_size0)
        kpts1_norm = normalize_keypoints(keypoints1, image_size1)
        kenc0 = self.kenc(kpts0_norm, scores0).to(self.compute_dtype)
        kenc1 = self.kenc(kpts1_norm, scores1).to(self.compute_dtype)

        d0 = descriptors0.to(self.compute_dtype, memory_format=torch.contiguous_format)
        d1 = descriptors1.to(self.compute_dtype, memory_format=torch.contiguous_format)
        d0, d1 = self.gnn(d0, d1, kenc0, kenc1, mask0, mask1)
        sim = self._similarity(d0, d1)
        return log_optimal_transport(sim, self.bin_score, iters=self.config.sinkhorn_iterations)

    def _match_core(
        self,
        kpts0_norm: Tensor,
        kpts1_norm: Tensor,
        descriptors0: Tensor,
        descriptors1: Tensor,
        scores0: Tensor,
        scores1: Tensor,
        dino0: Tensor,
        dino1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Capturable core: DINO mask + kenc + 18-layer GNN + similarity + ``ops.sinkhorn_mutual``."""
        mask0, mask1 = dino_attention_mask(dino0, dino1)
        kenc0 = self.kenc(kpts0_norm, scores0).to(self.compute_dtype)
        kenc1 = self.kenc(kpts1_norm, scores1).to(self.compute_dtype)
        d0 = descriptors0.to(self.compute_dtype, memory_format=torch.contiguous_format)
        d1 = descriptors1.to(self.compute_dtype, memory_format=torch.contiguous_format)
        d0, d1 = self.gnn(d0, d1, kenc0, kenc1, mask0, mask1)
        sim = self._similarity(d0, d1)[0]
        return ops.sinkhorn_mutual(sim, self.bin_score, self.config.sinkhorn_iterations)

    def _run_match_core(
        self,
        kpts0_norm: Tensor,
        kpts1_norm: Tensor,
        descriptors0: Tensor,
        descriptors1: Tensor,
        scores0: Tensor,
        scores1: Tensor,
        dino0: Tensor,
        dino1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        args = (kpts0_norm, kpts1_norm, descriptors0, descriptors1, scores0, scores1, dino0, dino1)
        if not self.graphs or not descriptors0.is_cuda or torch.is_grad_enabled():
            return self._match_core(*args)
        from flashim.engines.cuda_graph import CudaGraphRunner

        cache = self.__dict__.setdefault("_graph_runners", OrderedDict())
        key = tuple((t.shape, t.dtype, t.device) for t in args)
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            return runner(*args)
        runner = CudaGraphRunner(self._match_core)
        try:
            outputs = runner(*args)
        except RuntimeError as exc:
            warn_once(
                logger, f"omniglue-graph-{id(self)}",
                "OmniGlue matcher: CUDA graph capture failed, running eagerly from now on (%s)", exc,
            )
            self.graphs = 0
            torch.cuda.synchronize()
            return self._match_core(*args)
        cache[key] = runner
        while len(cache) > self.graphs:
            cache.popitem(last=False)
        return outputs

    @torch.inference_mode()
    def match_features(
        self, a: OmniGlueFeatures, b: OmniGlueFeatures, threshold: float | None = None
    ) -> Matches:
        if a.keypoints.shape[0] != 1 or b.keypoints.shape[0] != 1:
            raise ValueError("OmniGlue matches one image pair at a time (batch size 1)")
        if a.dino_descriptors is None or b.dino_descriptors is None:
            raise ValueError("OmniGlueMatcher requires OmniGlueFeatures with dino_descriptors")
        if a.keypoints.shape[1] == 0 or b.keypoints.shape[1] == 0:
            empty = a.keypoints.new_zeros(0, 2)
            return Matches(empty, empty, a.keypoints.new_zeros(0))

        kpts0_norm = normalize_keypoints(a.keypoints, a.image_size)
        kpts1_norm = normalize_keypoints(b.keypoints, b.image_size)
        prob, index, mutual = self._run_match_core(
            kpts0_norm, kpts1_norm,
            a.descriptors, b.descriptors,
            a.scores, b.scores,
            a.dino_descriptors, b.dino_descriptors,
        )

        limit = self.config.match_threshold if threshold is None else threshold
        keep_mask = mutual & (prob > limit) & (a.scores[0] > 0.0) & (b.scores[0][index] > 0.0)
        ia = torch.nonzero(keep_mask)[:, 0]
        ib = index[ia]
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], prob[ia], ia, ib)
