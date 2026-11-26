"""UFM (UniFlowMatch) dense correspondence matcher, native FlashIM implementation.

Module paths match the ``infinity1096/UFM-Base`` checkpoint (``encoder.model.*``, ``info_sharing.*``,
``head1.*``, ``uncertainty_head.*``), so ``state_dict`` loads with ``strict=True`` and all 36 transformer blocks
(24 DINOv2 encoder + 12 global-attention info-sharing) are :class:`flashim.layers.TransformerBlock` instances
that receive FlashIM's quantization and block-fusion passes by layer type.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.layers.transformer import TransformerBlock
from flashim.log import get_logger, warn_once
from flashim.models.base import ImageMatcher
from flashim.models.romav2.geometry import imagenet_normalize, normalized_grid, to_pixel
from flashim.models.romav2.model import ImageLike, load_image
from flashim.models.ufm import weights
from flashim.models.ufm.config import UFMConfig
from flashim.platforms import default_device
from flashim.quant.methods.conv2d import StaticInt8Conv2d
from flashim.quant.methods.int8_linear import StaticInt8Conv1x1
from flashim.types import DenseMatchResult, Matches

logger = get_logger(__name__)


def _sinusoid_table(n_position: int, dim: int, base: float = 10000.0) -> Tensor:
    """Sinusoid view-index positional encoding table ``[n_position, dim]`` (FP32)."""
    pos = torch.arange(n_position, dtype=torch.float64)[:, None]
    idx = torch.arange(dim, dtype=torch.float64)[None, :]
    angles = pos / (base ** (2 * (idx // 2) / dim))
    table = torch.empty_like(angles)
    table[:, 0::2] = torch.sin(angles[:, 0::2])
    table[:, 1::2] = torch.cos(angles[:, 1::2])
    return table.float()


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT-L/14 feature extractor matching ``encoder.model.*`` in the UFM checkpoint (no ``mask_token``)."""

    patch_embed_op: ops.BoundOp | None = None

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 14,
        img_size: int = 518,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid * self.grid, dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, num_heads, eps=1e-6, layer_scale=True) for _ in range(depth)
        )
        for block in self.blocks:
            block.attn.sdpa_impl = None  # dispatch (reference is torch SDPA)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self._pos_cache: dict[tuple[int, int, torch.dtype, torch.device], Tensor] = {}

    def positions(self, h: int, w: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        key = (h, w, dtype, device)
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        if h == self.grid and w == self.grid:
            pe = self.pos_embed.to(device=device, dtype=dtype)
        else:
            pe_f = self.pos_embed.float()
            dim = pe_f.shape[-1]
            patch_pe = F.interpolate(
                pe_f[:, 1:].reshape(1, self.grid, self.grid, dim).permute(0, 3, 1, 2),
                scale_factor=((h + 0.1) / self.grid, (w + 0.1) / self.grid),
                mode="bicubic",
            )
            assert patch_pe.shape[-2:] == (h, w)
            pe = torch.cat(
                (pe_f[:, :1], patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)), dim=1
            ).to(device=device, dtype=dtype)
        self._pos_cache[key] = pe
        return pe

    def forward(self, x: Tensor) -> Tensor:
        b, _, height, width = x.shape
        h, w = height // self.patch_size, width // self.patch_size
        weight_dtype = self.patch_embed.proj.weight.dtype
        if self.patch_embed_op is not None:
            tokens = self.patch_embed_op(x, self.cls_token, self.positions(h, w, weight_dtype, x.device))
        else:
            tokens = self.patch_embed.proj(x.to(weight_dtype)).flatten(2).transpose(1, 2)
            tokens = torch.cat((self.cls_token.to(tokens.dtype).expand(b, -1, -1), tokens), dim=1)
            tokens = tokens + self.positions(h, w, tokens.dtype, tokens.device)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1:].reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class DINOv2Encoder(nn.Module):
    """Wrapper holding ``self.model`` so checkpoint paths ``encoder.model.*`` match strictly."""

    def __init__(self, cfg: UFMConfig) -> None:
        super().__init__()
        self.model = DinoV2Backbone(
            dim=cfg.encoder_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.encoder_heads,
            patch_size=cfg.patch_size,
            img_size=cfg.pos_embed_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class GlobalAttentionTransformer(nn.Module):
    """12-layer multi-view global self-attention transformer (``info_sharing.*``).

    Concatenates patch tokens of both views, projects ``1024 -> 768``, adds view-index sinusoid PE
    (view 0 = reference, view 1 = target), runs 12 :class:`TransformerBlock` layers and returns normalized
    view-0 feature maps at ``indices`` (5, 8) and the final layer (11).
    """

    def __init__(self, cfg: UFMConfig) -> None:
        super().__init__()
        self.dim = cfg.info_dim
        self.indices = tuple(cfg.info_indices)
        self.proj_embed = nn.Linear(cfg.encoder_dim, cfg.info_dim, bias=True)
        self.self_attention_blocks = nn.ModuleList(
            TransformerBlock(cfg.info_dim, cfg.info_heads, eps=1e-6, layer_scale=False)
            for _ in range(cfg.info_depth)
        )
        for block in self.self_attention_blocks:
            block.attn.sdpa_impl = None
        self.norm = nn.LayerNorm(cfg.info_dim, eps=1e-6)
        self.register_buffer("view_pos_table", _sinusoid_table(2, cfg.info_dim))

    def forward(self, feat1: Tensor, feat2: Tensor) -> list[Tensor]:
        """``feat1, feat2``: ``[B, 1024, h, w]`` -> ``[inter_5, inter_8, final]`` for view 0, each ``[B, 768, h, w]``."""
        b, c, h, w = feat1.shape
        n = h * w
        tokens = torch.stack((feat1, feat2), dim=1).permute(0, 1, 3, 4, 2).reshape(b, 2 * n, c).contiguous()
        tokens = self.proj_embed(tokens)
        pe = self.view_pos_table.to(dtype=tokens.dtype)
        # Apply view-0 and view-1 positional encodings
        ref_tokens = tokens[:, :n] + pe[0]
        tgt_tokens = tokens[:, n:] + pe[1]
        tokens = torch.cat((ref_tokens, tgt_tokens), dim=1)

        out_tokens: list[Tensor] = []
        for idx, block in enumerate(self.self_attention_blocks):
            tokens = block(tokens)
            if idx in self.indices:
                out_tokens.append(self.norm(tokens[:, :n]))
        out_tokens.append(self.norm(tokens[:, :n]))
        return [t.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous() for t in out_tokens]


class ResidualConvUnit(nn.Module):
    """DPT pre-activation residual unit (non-inplace ReLU, matching UniCeption/CroCo).

    When ``conv1`` / ``conv2`` are :class:`StaticInt8Conv2d`, fuses the middle ReLU, conv1 -> conv2 INT8
    quantization chaining (when ``conv1.chain_output`` is enabled), and trailing residual add into the
    INT8 conv epilogues.
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.conv1, StaticInt8Conv2d) and isinstance(self.conv2, StaticInt8Conv2d):
            out = self.conv1(self.activation(x), relu=True, quantize_for=self.conv2)
            return self.conv2(out, residual=x)
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT fusion stage (2x bilinear upsampling with ``align_corners=True`` + 1x1 ``out_conv``)."""

    fused_upsample: bool = False

    def __init__(self, features: int, *, has_residual: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs: Tensor) -> Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if self.fused_upsample:
            size = (output.shape[-2] * 2, output.shape[-1] * 2)
            if isinstance(self.out_conv, StaticInt8Conv1x1):
                q = ops.upsample_bilinear(output, size, align_corners=True, out_scale=self.out_conv.input_scale)
                return self.out_conv.forward_prequantized(q).permute(0, 3, 1, 2)
            scale = self.out_conv.input_scale if isinstance(self.out_conv, StaticInt8Conv2d) else None
            output = ops.upsample_bilinear(output, size, align_corners=True, out_scale=scale)
            return self.out_conv(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class DPTFeature(nn.Module):
    """Multi-scale DPT feature pyramid (``*.0.0.*`` in checkpoint)."""

    def __init__(
        self,
        input_dims: tuple[int, ...] = (1024, 768, 768, 768),
        layer_dims: tuple[int, ...] = (96, 192, 384, 768),
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        acts = [
            nn.Sequential(
                nn.Conv2d(input_dims[0], layer_dims[0], 1),
                nn.ConvTranspose2d(layer_dims[0], layer_dims[0], kernel_size=4, stride=4),
            ),
            nn.Sequential(
                nn.Conv2d(input_dims[1], layer_dims[1], 1),
                nn.ConvTranspose2d(layer_dims[1], layer_dims[1], kernel_size=2, stride=2),
            ),
            nn.Sequential(nn.Conv2d(input_dims[2], layer_dims[2], 1)),
            nn.Sequential(
                nn.Conv2d(input_dims[3], layer_dims[3], 1),
                nn.Conv2d(layer_dims[3], layer_dims[3], kernel_size=3, stride=2, padding=1),
            ),
        ]
        self.input_process = nn.ModuleList(
            nn.Sequential(act, nn.Conv2d(ld, feature_dim, kernel_size=3, padding=1, bias=False))
            for act, ld in zip(acts, layer_dims)
        )
        scratch = nn.Module()
        scratch.refinenet1 = FeatureFusionBlock(feature_dim, has_residual=True)
        scratch.refinenet2 = FeatureFusionBlock(feature_dim, has_residual=True)
        scratch.refinenet3 = FeatureFusionBlock(feature_dim, has_residual=True)
        scratch.refinenet4 = FeatureFusionBlock(feature_dim, has_residual=False)
        self.scratch = scratch

    def forward(self, features: list[Tensor]) -> Tensor:
        layers = [proc(feat) for proc, feat in zip(self.input_process, features)]
        s = self.scratch
        path4 = s.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path3 = s.refinenet3(path4, layers[2])
        path2 = s.refinenet2(path3, layers[1])
        return s.refinenet1(path2, layers[0])


class DPTRegressionProcessor(nn.Module):
    """DPT upsampling regression head (``*.0.1.*`` in checkpoint)."""

    fused_upsample: bool = False

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: tuple[int, int] = (128, 128),
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dims[1], output_dim, kernel_size=1),
        )

    def forward(self, x: Tensor, target_shape: tuple[int, int]) -> Tensor:
        x = self.conv1(x)
        conv2_0 = self.conv2[0]
        if self.fused_upsample:
            if isinstance(conv2_0, StaticInt8Conv2d):
                q = ops.upsample_bilinear(x, target_shape, align_corners=True, out_scale=conv2_0.input_scale)
                return self.conv2[2](conv2_0(q, relu=True))
            x = ops.upsample_bilinear(x, target_shape, align_corners=True)
            return self.conv2(x)
        x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=True)
        if isinstance(conv2_0, StaticInt8Conv2d):
            return self.conv2[2](conv2_0(x, relu=True))
        return self.conv2(x)


class FlowAdaptor(nn.Module):
    """Unnormalizes raw 2-channel flow using ``scale_both`` against ``base_shape=(224, 224)``."""

    def __init__(self, base_shape: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.base_h, self.base_w = base_shape
        self.register_buffer("flow_mean", torch.zeros(1, 2, 1, 1))
        self.register_buffer("flow_std", torch.full((1, 2, 1, 1), 25.0))

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        x = x.float()
        x_scale = w / self.base_w
        y_scale = h / self.base_h
        mean = self.flow_mean.float().clone()
        mean[:, 0] *= x_scale
        mean[:, 1] *= y_scale
        std = self.flow_std.float().clone()
        std[:, 0] *= x_scale
        std[:, 1] *= y_scale
        return x * std + mean


class MaskAdaptor(nn.Module):
    """Converts 1-channel covisibility logits into ``(mask, logits)``."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = x.float()
        return torch.sigmoid(logits), logits


class AdaptorMap(nn.Module):
    """Matches checkpoint's ``*.1.adaptors.<name>`` container."""

    def __init__(self, name: str, adaptor: nn.Module) -> None:
        super().__init__()
        self.name = name
        self.adaptors = nn.ModuleDict({name: adaptor})

    def forward(self, x: Tensor):
        return self.adaptors[self.name](x)


class DPTHead(nn.Sequential):
    """``nn.Sequential(nn.Sequential(DPTFeature, DPTRegressionProcessor), AdaptorMap)`` matching checkpoint keys."""

    def __init__(self, feature: DPTFeature, processor: DPTRegressionProcessor, adaptor_map: AdaptorMap) -> None:
        super().__init__(nn.Sequential(feature, processor), adaptor_map)

    def forward(self, features: list[Tensor], target_shape: tuple[int, int]):
        feat_8x = self[0][0](features)
        raw = self[0][1](feat_8x, target_shape)
        return self[1](raw)


def _working_size(h: int, w: int, resize: int | None, cfg: UFMConfig) -> tuple[int, int]:
    """Working resolution ``(H_w, W_w)`` divisible by ``patch_size`` (default: ``(cfg.height, cfg.width)``)."""
    if resize is None:
        return cfg.height, cfg.width
    p = cfg.patch_size
    scale = resize / max(h, w)
    nh = max(p, round(h * scale / p) * p)
    nw = max(p, round(w * scale / p) * p)
    return nh, nw


def unmap_flow(flow: Tensor, src_hw: tuple[int, int], tgt_hw: tuple[int, int]) -> Tensor:
    """Upstream ``unmap_predicted_flow`` specialized for full-frame resize (no crop/padding)."""
    _, _, h_w, w_w = flow.shape
    h0, w0 = src_hw
    h1, w1 = tgt_hw
    src_grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, w_w, device=flow.device, dtype=torch.float32) + 0.5,
            torch.arange(0, h_w, device=flow.device, dtype=torch.float32) + 0.5,
            indexing="xy",
        ),
        dim=0,
    ).unsqueeze(0)
    src_valid = F.interpolate(src_grid, size=(h0, w0), mode="bilinear", align_corners=False)
    tgt_valid = F.interpolate(flow.float(), size=(h0, w0), mode="nearest") + src_valid
    src_scale = torch.tensor([w0 / w_w, h0 / h_w], device=flow.device, dtype=torch.float32).view(1, 2, 1, 1)
    tgt_scale = torch.tensor([w1 / w_w, h1 / h_w], device=flow.device, dtype=torch.float32).view(1, 2, 1, 1)
    return tgt_valid * tgt_scale - src_valid * src_scale


def unmap_covisibility(mask: Tensor, src_hw: tuple[int, int]) -> Tensor:
    """Upstream ``unmap_predicted_channels`` specialized for full-frame resize -> ``[B, H0, W0]``."""
    return F.interpolate(mask.float(), size=src_hw, mode="nearest").squeeze(1)


class UFM(ImageMatcher, nn.Module):
    """UFM (UniFlowMatch) dense correspondence matcher: DINOv2 ViT-L/14 -> 12-layer global-attention
    info-sharing -> two DPT heads (flow + covisibility).

    ``match`` draws ``max_matches`` KDE-balanced confident matches (5000 by default) from the dense flow
    and attaches the full dense prediction as ``dense`` (:class:`DenseMatchResult`).
    """

    match_defaults: Mapping[str, int] = {"max_matches": 5000}

    def __init__(self, config: UFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or UFMConfig()
        cfg = self.config
        self.encoder = DINOv2Encoder(cfg)
        self.info_sharing = GlobalAttentionTransformer(cfg)
        self.head1 = DPTHead(
            DPTFeature(cfg.dpt_input_dims, cfg.dpt_layer_dims, cfg.dpt_feature_dim),
            DPTRegressionProcessor(cfg.dpt_feature_dim, cfg.dpt_hidden_dims, output_dim=2),
            AdaptorMap("flow", FlowAdaptor(cfg.flow_base_shape)),
        )
        self.uncertainty_head = DPTHead(
            DPTFeature(cfg.dpt_input_dims, cfg.dpt_layer_dims, cfg.dpt_feature_dim),
            DPTRegressionProcessor(cfg.dpt_feature_dim, cfg.dpt_hidden_dims, output_dim=1),
            AdaptorMap("non_occluded_mask", MaskAdaptor()),
        )
        self.compute_dtype: torch.dtype = torch.float32
        self.heads_dtype: torch.dtype = torch.float32
        self.channels_last_heads: bool = False
        # Filled in by the builder.
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None

    @classmethod
    def from_pretrained(
        cls,
        config: UFMConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> UFM:
        device = torch.device(device or default_device())
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.encoder.model.cls_token.device

    def set_compute_dtype(self, dtype: torch.dtype, *, heads_dtype: torch.dtype | None = None) -> None:
        """Cast encoder and info-sharing to ``dtype``; DPT heads stay in ``heads_dtype`` (default FP32, matching
        upstream's ``autocast(float32)`` around the prediction heads)."""
        self.compute_dtype = dtype
        self.heads_dtype = torch.float32 if heads_dtype is None else heads_dtype
        self.encoder.to(dtype)
        self.info_sharing.to(dtype)
        self.head1[0].to(self.heads_dtype)
        self.head1[1].to(torch.float32)
        self.uncertainty_head[0].to(self.heads_dtype)

    def load_image(self, img: ImageLike) -> Tensor:
        return load_image(img, self.device)

    def prepare(
        self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None
    ) -> tuple[Tensor, Tensor, tuple[int, int], tuple[int, int]]:
        """Load, ImageNet-normalize and resize ``(img_a, img_b)`` to the working resolution (upstream order)."""
        a, b = self.load_image(img_a), self.load_image(img_b)
        src_hw, tgt_hw = a.shape[-2:], b.shape[-2:]
        h_w, w_w = _working_size(*src_hw, resize, self.config)
        a_norm = imagenet_normalize(a.float())
        b_norm = imagenet_normalize(b.float())
        a_res = F.interpolate(a_norm, size=(h_w, w_w), mode="bilinear", align_corners=False, antialias=True)
        b_res = F.interpolate(b_norm, size=(h_w, w_w), mode="bilinear", align_corners=False, antialias=True)
        return a_res, b_res, src_hw, tgt_hw

    def _run_heads(self, features: list[Tensor], target_shape: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        mem = torch.channels_last if self.channels_last_heads else torch.contiguous_format
        head_feats = [f.to(dtype=self.heads_dtype, memory_format=mem) for f in features]
        flow = self.head1(head_feats, target_shape)
        covis_mask, covis_logits = self.uncertainty_head(head_feats, target_shape)
        return flow, covis_mask, covis_logits

    # ``cuda_graph`` pass: :meth:`forward` replayed from one CUDA graph per working resolution,
    # keeping the ``graphs`` most recent shapes in an LRU cache (0 = eager).
    graphs: int = 0

    def _forward_eager(self, img_a: Tensor, img_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _, _, h, w = img_a.shape
        stacked = torch.cat((img_a, img_b), dim=0).to(self.compute_dtype)
        feat1, feat2 = self.encoder(stacked).chunk(2, dim=0)
        shared = self.info_sharing(feat1, feat2)
        features = [feat1, shared[0], shared[1], shared[2]]
        return self._run_heads(features, (h, w))

    @torch.inference_mode()
    def forward(self, img_a: Tensor, img_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalized ``[B, 3, H, W]`` pair (equal shape, multiples of 14) -> ``(flow [B,2,H,W],
        covisibility [B,1,H,W], covisibility_logits [B,1,H,W])`` in FP32 at working resolution."""
        _, _, h, w = img_a.shape
        if img_b.shape[-2:] != (h, w):
            raise ValueError(f"UFM requires equal input shapes, got {img_a.shape[-2:]} and {img_b.shape[-2:]}")
        if not self.graphs or not img_a.is_cuda or torch.is_grad_enabled():
            return self._forward_eager(img_a, img_b)
        from flashim.engines.cuda_graph import CudaGraphRunner

        cache = self.__dict__.setdefault("_graph_runners", OrderedDict())
        key = (img_a.shape, img_a.dtype, img_a.device, img_b.shape, img_b.dtype)
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            flow, covis_mask, covis_logits = runner(img_a, img_b)
            return flow.clone(), covis_mask.clone(), covis_logits.clone()
        runner = CudaGraphRunner(self._forward_eager)
        try:
            flow, covis_mask, covis_logits = runner(img_a, img_b)
        except RuntimeError as exc:
            warn_once(
                logger, f"ufm-graph-{id(self)}",
                "UFM: CUDA graph capture failed, running eagerly from now on (%s)", exc,
            )
            self.graphs = 0
            torch.cuda.synchronize()
            return self._forward_eager(img_a, img_b)
        cache[key] = runner
        while len(cache) > self.graphs:
            cache.popitem(last=False)
        return flow.clone(), covis_mask.clone(), covis_logits.clone()

    @torch.inference_mode()
    def predict_flow(
        self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Upstream-equivalent dense prediction: ``(flow [B, 2, H_a, W_a], covisibility [B, H_a, W_a])``
        unmapped to original image resolution."""
        a, b, src_hw, tgt_hw = self.prepare(img_a, img_b, resize=resize)
        flow, covis_mask, _ = self(a, b)
        return unmap_flow(flow, src_hw, tgt_hw), unmap_covisibility(covis_mask, src_hw)

    @staticmethod
    def _to_dense_result(flow: Tensor, covis_mask: Tensor, covis_logits: Tensor) -> DenseMatchResult:
        """Convert working-resolution ``(flow, covis_mask, covis_logits)`` to :class:`DenseMatchResult`."""
        b, _, h, w = flow.shape
        grid = normalized_grid(b, h, w, flow.device)  # [B, H, W, 2] in [-1, 1]
        scale = torch.tensor([2.0 / w, 2.0 / h], device=flow.device, dtype=torch.float32).view(1, 1, 1, 2)
        warp_ab = grid + flow.permute(0, 2, 3, 1) * scale
        return DenseMatchResult(
            warp_AB=warp_ab,
            confidence_AB=covis_logits.permute(0, 2, 3, 1),
            overlap_AB=covis_mask.permute(0, 2, 3, 1),
            precision_AB=flow.new_empty(b, h, w, 0, 0),
        )

    @torch.inference_mode()
    def match_dense(
        self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None
    ) -> DenseMatchResult:
        """Dense prediction as :class:`DenseMatchResult` at working resolution."""
        a, b, _, _ = self.prepare(img_a, img_b, resize=resize)
        return self._to_dense_result(*self(a, b))

    def sample(
        self,
        result: DenseMatchResult,
        num: int,
        *,
        threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, None, None]:
        """Sample ``num`` KDE-balanced confident matches ``[N, 4]`` ``(xA, yA, xB, yB)`` in ``[-1, 1]``."""
        warp = result.warp_AB[0]
        h, w, _ = warp.shape
        grid = normalized_grid(1, h, w, warp.device)[0].reshape(-1, 2)
        warp_flat = warp.reshape(-1, 2)
        in_frame = (warp_flat[:, 0].abs() <= 1.0 - 1.0 / w) & (warp_flat[:, 1].abs() <= 1.0 - 1.0 / h)
        overlap = result.overlap_AB[0].reshape(-1).clamp_min(0.0)
        if threshold is not None:
            in_frame = in_frame & (overlap >= threshold)
        confidence = overlap * in_frame.float()
        avail = int((confidence > 0).sum().item())
        if avail == 0 or num <= 0:
            empty = warp.new_empty(0, 4)
            return empty, confidence.new_empty(0), None, None
        n_cand = min(4 * num, avail)
        inds = torch.multinomial(confidence, n_cand, replacement=False, generator=generator)
        matches = torch.cat((grid[inds], warp_flat[inds]), dim=-1)
        conf_cand = confidence[inds]
        if len(conf_cand) <= num:
            return matches, conf_cand, None, None
        density = ops.kde_density(matches)
        p = 1.0 / (density + 1.0)
        p[density < 10] = 1e-7
        balanced = torch.multinomial(p, num_samples=min(num, len(conf_cand)), replacement=False, generator=generator)
        return matches[balanced], conf_cand[balanced], None, None

    @staticmethod
    def to_pixel_coordinates(
        matches: Tensor, h_a: int, w_a: int, h_b: int, w_b: int
    ) -> tuple[Tensor, Tensor]:
        return to_pixel(matches[..., :2], h=h_a, w=w_a), to_pixel(matches[..., 2:], h=h_b, w=w_b)

    def _match(
        self,
        img_a: ImageLike,
        img_b: ImageLike,
        *,
        resize: int | None,
        max_matches: int | None,
        threshold: float | None,
        generator: torch.Generator | None = None,
    ) -> Matches:
        """``max_matches`` KDE-balanced confident matches as pixel coordinates of the *original* images
        (RoMaV2 ``to_pixel`` convention), their covisibility as ``confidence``; ``dense`` carries the full
        working-resolution prediction."""
        if max_matches is None:
            raise ValueError("UFM.match needs max_matches (samples of the dense flow); match_dense returns the warp")
        a, b, (h_a, w_a), (h_b, w_b) = self.prepare(img_a, img_b, resize=resize)
        dense = self._to_dense_result(*self(a, b))
        matches, confidence, _, _ = self.sample(dense, max_matches, threshold=threshold, generator=generator)
        keypoints_a, keypoints_b = self.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
        return Matches(keypoints_a, keypoints_b, confidence, dense=dense)

    def extra_repr(self) -> str:
        return f"resolution=({self.config.height}, {self.config.width}), compute_dtype={self.compute_dtype}"
