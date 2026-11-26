"""DUSt3R model: CroCo ViT-L encoder, asymmetric ViT-B cross-attention decoder, DPT 3D-pointmap heads and
3D reciprocal nearest-neighbor matching (reusing MASt3R's CroCo backbone and DPT head)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor, nn

from flashim import ops
from flashim.log import get_logger, warn_once
from flashim.models.base import ImageMatcher
from flashim.models.dust3r import weights
from flashim.models.dust3r.config import DUSt3RConfig
from flashim.models.mast3r.croco import (
    CroCoStereoBackbone,
    DPTOutputAdapter,
    reg_dense_conf,
    reg_dense_depth,
)
from flashim.models.mast3r.model import prepare_image
from flashim.models.romav2.model import ImageLike, load_image
from flashim.platforms import default_device
from flashim.types import Matches
from flashim.utils import device_constant

logger = get_logger("models.dust3r")


class DPTHead(nn.Module):
    """DUSt3R ``PixelwiseTaskWithDPT``: DPT head producing dense 3D points and per-pixel confidence."""

    def __init__(self, cfg: DUSt3RConfig) -> None:
        super().__init__()
        self.depth_mode = cfg.depth_mode
        self.conf_mode = cfg.conf_mode
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

    def forward(self, decout: list[Tensor], image_size: tuple[int, int]) -> dict[str, Tensor]:
        dpt_dtype = self.dpt.act_postprocess[0][0].weight.dtype
        dpt_in = [t if t.dtype == dpt_dtype else t.to(dpt_dtype) for t in decout]
        out = self.dpt(dpt_in, image_size=image_size)
        if (
            self.depth_mode == ("exp", float("-inf"), float("inf"))
            and self.conf_mode is not None
            and self.conf_mode[0] == "exp"
            and self.conf_mode[2] == float("inf")
        ):
            pts3d, conf = ops.dust3r_pts3d_conf(out, self.conf_mode[1])
            return {"pts3d": pts3d, "conf": conf}
        raw = out.permute(0, 2, 3, 1).float()
        # Post-processing runs in FP32 for numerical stability of exp / norm.
        res: dict[str, Tensor] = {"pts3d": reg_dense_depth(raw[..., 0:3], self.depth_mode)}
        if self.conf_mode is not None:
            res["conf"] = reg_dense_conf(raw[..., 3], self.conf_mode)
        return res


class DUSt3R(CroCoStereoBackbone, ImageMatcher):
    """DUSt3R stereo 3D pointmap matcher (``DUSt3R_ViTLarge_BaseDecoder_512_dpt``).

    ``forward(img1, img2)`` returns ``(res1, res2)`` dicts with ``pts3d`` (``pts3d_in_other_view`` for view 2)
    and ``conf`` on normalized ``[-1, 1]`` ``[B, 3, H, W]`` inputs.
    ``match(img_a, img_b, *, resize, max_matches, threshold)`` finds reciprocal nearest neighbors in the predicted
    3D pointmaps (upstream ``find_reciprocal_matches``) and returns :class:`~flashim.types.Matches` in original
    image pixel coordinates.
    """

    fp32_modules: tuple[str, ...] = ()
    compute_dtype = torch.float32
    graphs = 0

    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None

    def __init__(self, config: DUSt3RConfig | None = None) -> None:
        cfg = config or DUSt3RConfig()
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
        self.downstream_head1 = DPTHead(cfg)
        self.downstream_head2 = DPTHead(cfg)
        self.resize = cfg.resize
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
        config: DUSt3RConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> DUSt3R:
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
                f"dust3r-graph-{id(self)}",
                "DUSt3R: CUDA graph capture failed, running eagerly from now on (%s)",
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
        """Run encoder, two-branch decoder and DPT heads on ``[-1, 1]`` ``[B, 3, H, W]`` tensors."""
        if self.graphs and img1.is_cuda and not torch.is_grad_enabled():
            r1, r2 = self._run_graphed("_fwd_graphs", self._forward_eager, img1, img2)
            return ({k: v.clone() for k, v in r1.items()}, {k: v.clone() for k, v in r2.items()})
        return self._forward_eager(img1, img2)

    @torch.inference_mode()
    def match_prepared(
        self,
        img1: Tensor,
        img2: Tensor,
        *,
        threshold: float | None = None,
        max_matches: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Match two prepared ``[-1, 1]`` ``[1, 3, H, W]`` tensors -> ``(xy1 [M, 2], xy2 [M, 2], conf [M])``
        in prepared-image pixel coordinates via 3D reciprocal nearest neighbors."""
        if self.graphs and img1.is_cuda:
            res1, res2 = self._run_graphed("_fwd_graphs", self._forward_eager, img1, img2)
        else:
            res1, res2 = self._forward_eager(img1, img2)

        pts1 = res1["pts3d"][0]
        pts2 = res2["pts3d_in_other_view"][0]
        conf1 = res1["conf"][0]
        conf2 = res2["conf"][0]
        _, w1 = pts1.shape[:2]
        _, w2 = pts2.shape[:2]
        device = pts1.device

        if threshold is not None:
            flat_mask1 = (conf1 >= threshold).reshape(-1)
            flat_mask2 = (conf2 >= threshold).reshape(-1)
            valid1 = flat_mask1.nonzero(as_tuple=True)[0]
            valid2 = flat_mask2.nonzero(as_tuple=True)[0]
            if valid1.numel() == 0 or valid2.numel() == 0:
                empty = torch.empty((0, 2), dtype=torch.float32, device=device)
                return empty, empty, torch.empty(0, dtype=torch.float32, device=device)
            p1 = pts1.reshape(-1, 3)[valid1]
            p2 = pts2.reshape(-1, 3)[valid2]
            idx1, idx2 = ops.reciprocal_nn(p1, p2)
            flat1 = valid1[idx1]
            flat2 = valid2[idx2]
        else:
            flat1, flat2 = ops.reciprocal_nn(pts1.reshape(-1, 3), pts2.reshape(-1, 3))

        if flat1.numel() == 0:
            empty = torch.empty((0, 2), dtype=torch.float32, device=device)
            return empty, empty, torch.empty(0, dtype=torch.float32, device=device)

        c1 = conf1.reshape(-1)[flat1]
        c2 = conf2.reshape(-1)[flat2]
        conf = torch.minimum(c1, c2)
        if max_matches is not None and max_matches < conf.shape[0]:
            conf, top = conf.topk(max_matches)
            flat1, flat2 = flat1[top], flat2[top]

        xy1 = torch.stack((flat1 % w1, flat1 // w1), dim=-1).float()
        xy2 = torch.stack((flat2 % w2, flat2 // w2), dim=-1).float()
        return xy1, xy2, conf

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
