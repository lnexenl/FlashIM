"""MASt3R model: CroCo ViT-L encoder, asymmetric ViT-B cross-attention decoder, CatMLP+DPT heads and reciprocal
nearest-neighbor matching."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.log import get_logger, warn_once
from flashim.models.base import ImageMatcher
from flashim.models.mast3r import weights
from flashim.models.mast3r.config import MASt3RConfig
from flashim.models.mast3r.croco import (
    CroCoStereoBackbone,
    DPTOutputAdapter,
    reg_dense_conf,
    reg_dense_depth,
)
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches
from flashim.utils import device_constant

logger = get_logger("models.mast3r")


class LocalFeatureMlp(nn.Module):
    """Two-layer GELU MLP projecting concatenated encoder+decoder tokens to per-pixel descriptors (+ confidence)."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CatMLPDPTHead(nn.Module):
    """MASt3R ``Cat_MLP_LocalFeatures_DPT_Pts3d``: DPT head for 3D points + confidence, concatenated encoder/decoder
    MLP + pixel shuffle for local descriptors + descriptor confidence."""

    def __init__(self, cfg: MASt3RConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.local_feat_dim = cfg.local_feat_dim
        self.depth_mode = cfg.depth_mode
        self.conf_mode = cfg.conf_mode
        self.desc_mode = cfg.desc_mode
        self.two_confs = cfg.two_confs
        self.desc_conf_mode = cfg.desc_conf_mode

        has_conf = cfg.conf_mode is not None
        l2 = cfg.dec_depth
        hooks = (0, l2 * 2 // 4, l2 * 3 // 4, l2)
        dim_tokens = (cfg.enc_embed_dim, cfg.dec_embed_dim, cfg.dec_embed_dim, cfg.dec_embed_dim)
        self.dpt = DPTOutputAdapter(
            num_channels=3 + int(has_conf),
            patch_size=cfg.patch_size,
            hooks=hooks,
            dim_tokens=dim_tokens,
            feature_dim=cfg.dpt_feature_dim,
            last_dim=cfg.dpt_feature_dim // 2,
        )

        idim = cfg.enc_embed_dim + cfg.dec_embed_dim
        out_dim = (cfg.local_feat_dim + int(cfg.two_confs)) * (cfg.patch_size ** 2)
        self.head_local_features = LocalFeatureMlp(idim, int(cfg.hidden_dim_factor * idim), out_dim)

    def _local_features_map(self, decout: list[Tensor], image_size: tuple[int, int]) -> Tensor:
        h, w = image_size
        cat_in = torch.cat((decout[0], decout[-1]), dim=-1)
        mlp_weight = getattr(self.head_local_features.fc1, "weight", None)
        if mlp_weight is not None and cat_in.dtype != mlp_weight.dtype:
            cat_in = cat_in.to(mlp_weight.dtype)
        local_features = self.head_local_features(cat_in)
        batch = cat_in.shape[0]
        local_features = local_features.transpose(-1, -2).reshape(
            batch, -1, h // self.patch_size, w // self.patch_size
        )
        return F.pixel_shuffle(local_features, self.patch_size)

    def forward_local_features(self, decout: list[Tensor], image_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Compute only ``(desc, desc_conf)`` from the parallel local-feature MLP branch (skips the 3D DPT head;
        bit-identical when ``two_confs`` is True)."""
        lf = self._local_features_map(decout, image_size).permute(0, 2, 3, 1).float()
        desc = lf[..., : self.local_feat_dim]
        if "norm" in self.desc_mode:
            desc = desc / desc.norm(dim=-1, keepdim=True)
        desc_conf = reg_dense_conf(lf[..., self.local_feat_dim], self.desc_conf_mode)
        return desc, desc_conf

    def forward(self, decout: list[Tensor], image_size: tuple[int, int]) -> dict[str, Tensor]:
        h, w = image_size
        dpt_dtype = self.dpt.act_postprocess[0][0].weight.dtype
        dpt_in = [t if t.dtype == dpt_dtype else t.to(dpt_dtype) for t in decout]
        pts3d_raw = self.dpt(dpt_in, image_size=(h, w)).permute(0, 2, 3, 1).float()
        lf = self._local_features_map(decout, image_size).permute(0, 2, 3, 1).float()

        # Post-processing runs in FP32 for numerical stability of exp / norm.
        res: dict[str, Tensor] = {"pts3d": reg_dense_depth(pts3d_raw[..., 0:3], self.depth_mode)}
        if self.conf_mode is not None:
            res["conf"] = reg_dense_conf(pts3d_raw[..., 3], self.conf_mode)
        desc = lf[..., : self.local_feat_dim]
        if "norm" in self.desc_mode:
            desc = desc / desc.norm(dim=-1, keepdim=True)
        res["desc"] = desc
        if self.two_confs:
            res["desc_conf"] = reg_dense_conf(lf[..., self.local_feat_dim], self.desc_conf_mode)
        else:
            res["desc_conf"] = res["conf"].clone()
        return res


def prepare_image(
    image: Tensor,
    *,
    resize: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
) -> tuple[Tensor, tuple[int, int, int, int, int, int]]:
    """Resize long side to ``resize``, center-crop to multiples of ``patch_size``, and normalize to ``[-1, 1]``.

    Returns ``(tensor [1, 3, crop_h, crop_w], (orig_h, orig_w, resized_h, resized_w, crop_y0, crop_x0))``.
    Matches upstream ``dust3r.utils.image.load_images`` geometry.
    """
    _, _, orig_h, orig_w = image.shape
    long_side = max(orig_h, orig_w)
    if resize == 224:
        scale_target = round(resize * max(orig_w / orig_h, orig_h / orig_w))
        new_h = round(orig_h * scale_target / long_side)
        new_w = round(orig_w * scale_target / long_side)
    else:
        new_h = round(orig_h * resize / long_side)
        new_w = round(orig_w * resize / long_side)
    if (new_h, new_w) != (orig_h, orig_w):
        image = F.interpolate(image, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True).clamp_(
            0.0, 1.0
        )
    cx, cy = new_w // 2, new_h // 2
    if resize == 224:
        half = min(cx, cy)
        halfw, halfh = half, half
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and new_w == new_h:
            halfh = (3 * halfw) // 4
    crop_x0 = cx - halfw
    crop_y0 = cy - halfh
    cropped = image[:, :, crop_y0 : crop_y0 + 2 * halfh, crop_x0 : crop_x0 + 2 * halfw]
    return cropped * 2.0 - 1.0, (orig_h, orig_w, new_h, new_w, crop_y0, crop_x0)


class MASt3R(CroCoStereoBackbone, ImageMatcher):
    """MASt3R 3D-grounded dense matcher (``MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric``).

    ``forward(img1, img2)`` returns ``(res1, res2)`` dicts with ``pts3d`` (``pts3d_in_other_view`` for view 2),
    ``conf``, ``desc``, and ``desc_conf`` on normalized ``[-1, 1]`` ``[B, 3, H, W]`` inputs.
    ``match(img_a, img_b, *, resize, max_matches, threshold)`` returns :class:`~flashim.types.Matches` in original
    image pixel coordinates.
    """

    fp32_modules: tuple[str, ...] = ()
    compute_dtype = torch.float32
    fp16_nn = False
    graphs = 0

    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None

    def __init__(self, config: MASt3RConfig | None = None) -> None:
        cfg = config or MASt3RConfig()
        super().__init__(
            patch_size=cfg.patch_size,
            enc_embed_dim=cfg.enc_embed_dim,
            enc_depth=cfg.enc_depth,
            enc_num_heads=cfg.enc_num_heads,
            dec_embed_dim=cfg.dec_embed_dim,
            dec_depth=cfg.dec_depth,
            dec_num_heads=cfg.dec_num_heads,
            mlp_ratio=cfg.mlp_ratio,
            norm_eps=cfg.norm_eps,
            norm_im2_in_dec=cfg.norm_im2_in_dec,
            rope_freq=cfg.rope_freq,
        )
        self.config = cfg
        self.downstream_head1 = CatMLPDPTHead(cfg)
        self.downstream_head2 = CatMLPDPTHead(cfg)
        self.resize = cfg.resize
        self.subsample = cfg.subsample
        self.border = cfg.border
        self.square_ok = cfg.square_ok

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        from flashim.quant.methods.int8_linear import DynamicInt8Linear, StaticInt8Linear

        for mod in self.modules():
            if isinstance(mod, (DynamicInt8Linear, StaticInt8Linear)):
                continue
            for p in mod.parameters(recurse=False):
                if p.is_floating_point():
                    p.data = p.data.to(dtype)
            for name, buf in mod.named_buffers(recurse=False):
                if buf is not None and buf.is_floating_point():
                    mod._buffers[name] = buf.to(dtype)
        for path in self.fp32_modules:
            self.get_submodule(path).float()
        self.compute_dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        config: MASt3RConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> MASt3R:
        model = cls(config)
        state = weights.load_state_dict(path=checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        return model.to(default_device() if device is None else device)

    def _run_graphed(self, cache_attr: str, fn, img1: Tensor, img2: Tensor):
        from flashim.engines.cuda_graph import CudaGraphRunner

        cache = self.__dict__.setdefault(cache_attr, OrderedDict())
        key = (img1.shape, img2.shape, img1.dtype, img2.dtype, img1.device)
        runner = cache.get(key)
        if runner is not None:
            cache.move_to_end(key)
            return runner(img1, img2)
        runner = CudaGraphRunner(fn)
        try:
            outputs = runner(img1, img2)
        except RuntimeError as exc:
            warn_once(
                logger,
                f"mast3r-graph-{id(self)}",
                "MASt3R: CUDA graph capture failed, running eagerly from now on (%s)",
                exc,
            )
            self.graphs = 0
            torch.cuda.synchronize()
            return fn(img1, img2)
        cache[key] = runner
        while len(cache) > self.graphs:
            cache.popitem(last=False)
        return outputs

    def _forward_eager(self, img1: Tensor, img2: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        h1, w1 = img1.shape[-2], img1.shape[-1]
        h2, w2 = img2.shape[-2], img2.shape[-1]
        x1 = img1 if img1.dtype == self.compute_dtype else img1.to(self.compute_dtype)
        x2 = img2 if img2.dtype == self.compute_dtype else img2.to(self.compute_dtype)
        feat1, feat2, grid1, grid2 = self.encode_pair(x1, x2)
        dec1, dec2 = self.decode(feat1, grid1, feat2, grid2)
        res1 = self.downstream_head1(dec1, (h1, w1))
        res2 = self.downstream_head2(dec2, (h2, w2))
        res2["pts3d_in_other_view"] = res2.pop("pts3d")
        return res1, res2

    def forward(self, img1: Tensor, img2: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run encoder, two-branch decoder and CatMLP+DPT heads on ``[-1, 1]`` ``[B, 3, H, W]`` tensors."""
        if self.graphs and img1.is_cuda and not torch.is_grad_enabled():
            r1, r2 = self._run_graphed("_fwd_graphs", self._forward_eager, img1, img2)
            return ({k: v.clone() for k, v in r1.items()}, {k: v.clone() for k, v in r2.items()})
        return self._forward_eager(img1, img2)

    def _forward_descriptors_eager(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h1, w1 = img1.shape[-2], img1.shape[-1]
        h2, w2 = img2.shape[-2], img2.shape[-1]
        x1 = img1 if img1.dtype == self.compute_dtype else img1.to(self.compute_dtype)
        x2 = img2 if img2.dtype == self.compute_dtype else img2.to(self.compute_dtype)
        feat1, feat2, grid1, grid2 = self.encode_pair(x1, x2)
        dec1, dec2 = self.decode(feat1, grid1, feat2, grid2)
        desc1, dconf1 = self.downstream_head1.forward_local_features(dec1, (h1, w1))
        desc2, dconf2 = self.downstream_head2.forward_local_features(dec2, (h2, w2))
        return desc1, dconf1, desc2, dconf2

    @torch.inference_mode()
    def match_prepared(
        self,
        img1: Tensor,
        img2: Tensor,
        *,
        subsample: int | None = None,
        border: int | None = None,
        threshold: float | None = None,
        max_matches: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Match two prepared ``[-1, 1]`` ``[1, 3, H, W]`` tensors -> ``(xy1 [M, 2], xy2 [M, 2], conf [M])``
        in prepared-image pixel coordinates."""
        if self.downstream_head1.two_confs and self.downstream_head2.two_confs:
            if self.graphs and img1.is_cuda:
                d1, c1, d2, c2 = self._run_graphed("_desc_graphs", self._forward_descriptors_eager, img1, img2)
            else:
                d1, c1, d2, c2 = self._forward_descriptors_eager(img1, img2)
            desc1, desc2 = d1[0], d2[0]
            dconf1, dconf2 = c1[0], c2[0]
        else:
            res1, res2 = self.forward(img1, img2)
            desc1, desc2 = res1["desc"][0], res2["desc"][0]
            dconf1, dconf2 = res1["desc_conf"][0], res2["desc_conf"][0]
        h1, w1 = desc1.shape[:2]
        h2, w2 = desc2.shape[:2]

        sub = self.subsample if subsample is None else subsample
        nn_d1 = desc1.half() if self.fp16_nn and desc1.is_cuda else desc1
        nn_d2 = desc2.half() if self.fp16_nn and desc2.is_cuda else desc2
        xy1, xy2 = ops.fast_reciprocal_nn(nn_d1, nn_d2, subsample=sub, max_iter=10, block_size=8192, dist="dot")

        b = self.border if border is None else border
        if b > 0 and xy1.shape[0] > 0:
            valid = (
                (xy1[:, 0] >= b) & (xy1[:, 0] < w1 - b) & (xy1[:, 1] >= b) & (xy1[:, 1] < h1 - b)
                & (xy2[:, 0] >= b) & (xy2[:, 0] < w2 - b) & (xy2[:, 1] >= b) & (xy2[:, 1] < h2 - b)
            )
            xy1, xy2 = xy1[valid], xy2[valid]

        if xy1.shape[0] == 0:
            empty = torch.empty((0, 2), dtype=torch.float32, device=desc1.device)
            return empty, empty, torch.empty(0, dtype=torch.float32, device=desc1.device)

        conf = torch.minimum(dconf1[xy1[:, 1], xy1[:, 0]], dconf2[xy2[:, 1], xy2[:, 0]])
        if threshold is not None:
            keep = conf >= threshold
            xy1, xy2, conf = xy1[keep], xy2[keep], conf[keep]
        if max_matches is not None and max_matches < conf.shape[0]:
            conf, idx = conf.topk(max_matches)
            xy1, xy2 = xy1[idx], xy2[idx]
        return xy1.float(), xy2.float(), conf

    def _match(
        self,
        img_a: ImageLike,
        img_b: ImageLike,
        *,
        resize: int | None,
        max_matches: int | None,
        threshold: float | None,
    ) -> Matches:
        target = self.resize if resize is None else resize
        im1, meta1 = prepare_image(
            load_image(img_a, self.device), resize=target, patch_size=self.patch_size, square_ok=self.square_ok
        )
        im2, meta2 = prepare_image(
            load_image(img_b, self.device), resize=target, patch_size=self.patch_size, square_ok=self.square_ok
        )
        xy1, xy2, conf = self.match_prepared(im1, im2, threshold=threshold, max_matches=max_matches)
        orig_h1, orig_w1, new_h1, new_w1, cy1, cx1 = meta1
        orig_h2, orig_w2, new_h2, new_w2, cy2, cx2 = meta2
        kpts_a = (xy1 + device_constant([cx1 + 0.5, cy1 + 0.5], xy1)) * device_constant(
            [orig_w1 / new_w1, orig_h1 / new_h1], xy1
        ) - 0.5
        kpts_b = (xy2 + device_constant([cx2 + 0.5, cy2 + 0.5], xy2)) * device_constant(
            [orig_w2 / new_w2, orig_h2 / new_h2], xy2
        ) - 0.5
        return Matches(kpts_a, kpts_b, conf)
