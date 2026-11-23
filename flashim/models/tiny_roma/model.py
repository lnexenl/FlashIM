"""Tiny-RoMa dense matcher (``tiny_roma_v1_outdoor`` from Parskatt/RoMa), re-implemented natively for FlashIM.

Architecture: fine-tuned XFeat convolutional backbone (stride-4 fine features, 24 ch; stride-8 coarse features,
64 ch) -> global stride-8 correlation + positional embedding -> 4-layer 256-ch coarse conv refiner -> stride-4
upsample + warp -> 4-layer 64-ch fine conv refiner.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models import sparse
from flashim.models.base import ImageMatcher
from flashim.models.romav2.geometry import to_pixel
from flashim.models.romav2.model import ImageLike, load_image
from flashim.models.tiny_roma import weights
from flashim.models.tiny_roma.config import TinyRoMaConfig
from flashim.models.xfeat.model import BasicLayer
from flashim.platforms import default_device
from flashim.types import DenseMatchResult, Matches
from flashim.utils import device_constant, grid_sample, set_submodule


class XFeatBackbone(nn.Module):
    """XFeat convolutional trunk without its sparse heads (``heatmap_head``, ``keypoint_head``, ``fine_matcher``).

    Module paths match ``xfeat.0.*`` inside ``tiny_roma_v1_outdoor.pth``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)
        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4), nn.Conv2d(1, 24, 1, stride=1, padding=0))
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )
        self.block2 = nn.Sequential(BasicLayer(24, 24, stride=1), BasicLayer(24, 24, stride=1))
        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, 1, padding=0)
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2), BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1)
        )
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1), BasicLayer(64, 64, stride=1), nn.Conv2d(64, 64, 1, padding=0)
        )
        self.compute_dtype = torch.float32
        self.channels_last = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``[B, 3, H, W]`` RGB in ``[0, 1]`` -> ``(fine [B, 24, H/4, W/4], coarse [B, 64, H/8, W/8])``."""
        gray = ops.gray_instance_norm(x, self.norm.eps, self.compute_dtype)
        if self.channels_last:
            gray = gray.to(memory_format=torch.channels_last)
        x1 = self.block1(gray)
        x2 = self.block2(x1 + self.skip1(gray))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        size = x3.shape[-2:]
        x4 = F.interpolate(x4, size, mode="bilinear")
        x5 = F.interpolate(x5, size, mode="bilinear")
        feats = self.block_fusion(x3 + x4 + x5)
        return x2, feats


class _MatcherHead(nn.Sequential):
    """Four :class:`BasicLayer`s followed by a 1x1 output conv kept in FP32 under reduced compute precision."""

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self) - 1):
            x = self[i](x)
        head = self[-1]
        return head(x.float() if head.weight.dtype == torch.float32 else x)


class TinyRoMa(ImageMatcher, nn.Module):
    """``xfeat.0`` (XFeat backbone) -> global stride-8 correlation -> ``coarse_matcher`` -> ``fine_matcher``.

    Module paths match ``tiny_roma_v1_outdoor.pth`` so checkpoints load with ``strict=True``. ``match`` draws
    ``max_matches`` KDE-balanced samples (5000 by default) from the dense warp, attached as ``dense``.
    """

    div_by: int = 32
    match_defaults: Mapping[str, Any] = {"max_matches": 5000}
    fp32_modules: tuple[str, ...] = ("coarse_matcher.4", "fine_matcher.4")

    def __init__(self, config: TinyRoMaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyRoMaConfig()
        self.xfeat = nn.ModuleList([XFeatBackbone()])
        match_dim = 256
        self.coarse_matcher = _MatcherHead(
            BasicLayer(64 + 64 + 2, match_dim),
            BasicLayer(match_dim, match_dim),
            BasicLayer(match_dim, match_dim),
            BasicLayer(match_dim, match_dim),
            nn.Conv2d(match_dim, 3, kernel_size=1, bias=True, padding=0),
        )
        fine_match_dim = 64
        self.fine_matcher = _MatcherHead(
            BasicLayer(24 + 24 + 2, fine_match_dim),
            BasicLayer(fine_match_dim, fine_match_dim),
            BasicLayer(fine_match_dim, fine_match_dim),
            BasicLayer(fine_match_dim, fine_match_dim),
            nn.Conv2d(fine_match_dim, 3, kernel_size=1, bias=True, padding=0),
        )
        self.compute_dtype = torch.float32
        self.fast_sampling = True
        self.coarse_coord_scale: float = 1.0
        self.fine_coord_scale: float = 1.0
        self.fused_batchnorm: list[str] = []
        # Filled in by the builder.
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None

    @classmethod
    def from_pretrained(
        cls,
        config: TinyRoMaConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> TinyRoMa:
        device = torch.device(device or default_device())
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.fine_matcher[-1].weight.device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.to(dtype)
        for path in self.fp32_modules:
            self.get_submodule(path).float()
        self.compute_dtype = dtype
        self.xfeat[0].compute_dtype = dtype

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return [(f"{n}.layer.0", f"{n}.layer.1") for n, m in self.named_modules() if isinstance(m, BasicLayer)]

    def fold_batchnorm(self) -> list[str]:
        """Exact eval-mode Conv2d + BatchNorm2d folding for all 26 :class:`BasicLayer`s."""
        if self.training:
            raise RuntimeError("fold_batchnorm requires model.eval()")
        folded = []
        for conv_path, bn_path in self.batchnorm_pairs():
            conv, bn = self.get_submodule(conv_path), self.get_submodule(bn_path)
            if not isinstance(bn, nn.BatchNorm2d):
                continue
            set_submodule(self, conv_path, nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            set_submodule(self, bn_path, nn.Identity())
            folded.append(conv_path)
        return folded

    @staticmethod
    def preprocess_tensor(x: Tensor) -> tuple[Tensor, float, float]:
        """Floor both spatial dimensions to multiples of 32 (bilinear, ``align_corners=False``)."""
        h, w = x.shape[-2:]
        new_h, new_w = max(32, (h // 32) * 32), max(32, (w // 32) * 32)
        rh, rw = h / new_h, w / new_w
        if (new_h, new_w) != (h, w):
            x = F.interpolate(x, (new_h, new_w), mode="bilinear", align_corners=False)
        return x, rh, rw

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.xfeat[0](x)

    @staticmethod
    def corr_volume(feat0: Tensor, feat1: Tensor) -> Tensor:
        """``[B, C, H0, W0]``, ``[B, C, H1, W1]`` -> scaled dot-product volume ``[B, H1, W1, H0, W0]``."""
        b, c, h0, w0 = feat0.shape
        _, _, h1, w1 = feat1.shape
        f0 = feat0.reshape(b, c, h0 * w0)
        f1 = feat1.reshape(b, c, h1 * w1)
        return torch.einsum("bci,bcj->bji", f0, f1).reshape(b, h1, w1, h0, w0) / math.sqrt(c)

    def pos_embed(self, corr_volume: Tensor) -> Tensor:
        """Correlation volume ``[B, H1, W1, H0, W0]`` -> coarse normalized target coordinates ``[B, 2, H0, W0]``."""
        b, h1, w1, h0, w0 = corr_volume.shape
        device = corr_volume.device
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1 + 1 / w1, 1 - 1 / w1, w1, device=device),
                torch.linspace(-1 + 1 / h1, 1 - 1 / h1, h1, device=device),
                indexing="xy",
            ),
            dim=-1,
        ).float().reshape(h1 * w1, 2)
        down = 4
        if not self.training and not self.config.exact_softmax:
            grid_lr = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1 + down / w1, 1 - down / w1, w1 // down, device=device),
                    torch.linspace(-1 + down / h1, 1 - down / h1, h1 // down, device=device),
                    indexing="xy",
                ),
                dim=-1,
            ).float().reshape((h1 // down) * (w1 // down), 2)
            best_match = corr_volume.reshape(b, h1 * w1, h0, w0).argmax(dim=1)
            cv_lr = corr_volume[:, ::down, ::down].reshape(b, (h1 // down) * (w1 // down), h0, w0).float()
            p_lowres = torch.cat((cv_lr, best_match[:, None].float()), dim=1).softmax(dim=1)
            pos_embeddings = torch.einsum("bchw,cd->bdhw", p_lowres[:, :-1], grid_lr)
            pos_embeddings = pos_embeddings + p_lowres[:, -1:] * grid[best_match].permute(0, 3, 1, 2)
        else:
            p = corr_volume.reshape(b, h1 * w1, h0, w0).float().softmax(dim=1)
            pos_embeddings = torch.einsum("bchw,cd->bdhw", p, grid)
        return pos_embeddings

    @torch.inference_mode()
    def forward(self, im_a: Tensor, im_b: Tensor) -> dict[int, dict[str, Tensor]]:
        """``im_a``, ``im_b`` ``[B, 3, H, W]`` floats in ``[0, 1]`` -> stride-8 and stride-4 flow + certainty logits."""
        im0, _, _ = self.preprocess_tensor(im_a)
        im1, _, _ = self.preprocess_tensor(im_b)
        _, _, h1, w1 = im1.shape
        to_normalized = device_constant((2.0 / w1, 2.0 / h1, 1.0), im0, torch.float32)[None, :, None, None]

        if im0.shape[-2:] == im1.shape[-2:]:
            feats_f, feats_c = self.forward_single(torch.cat((im0, im1), dim=0))
            feats_x0_c, feats_x1_c = feats_c.chunk(2)
            feats_x0_f, feats_x1_f = feats_f.chunk(2)
        else:
            feats_x0_f, feats_x0_c = self.forward_single(im0)
            feats_x1_f, feats_x1_c = self.forward_single(im1)

        coarse_warp = ops.tiny_roma_corr_pos_embed(feats_x0_c, feats_x1_c, self.config.exact_softmax)
        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:, -1:])), dim=1)

        # Keep warp coordinates in FP32 for grid_sample so BF16 mantissa truncation never quantizes pixel coordinates.
        coarse_grid = coarse_matches[:, :2].permute(0, 2, 3, 1)
        feats_x1_c_warped = grid_sample(
            feats_x1_c.float(), coarse_grid, mode="bilinear", align_corners=False
        ).to(self.compute_dtype)
        cw_in = coarse_warp.to(self.compute_dtype)
        if self.coarse_coord_scale != 1.0:
            cw_in = cw_in * self.coarse_coord_scale
        coarse_in = torch.cat((feats_x0_c, feats_x1_c_warped, cw_in), dim=1)
        coarse_delta = self.coarse_matcher(coarse_in)
        coarse_matches = coarse_matches + coarse_delta * to_normalized

        coarse_matches_up = F.interpolate(
            coarse_matches, size=feats_x0_f.shape[-2:], mode="bilinear", align_corners=False
        )
        fine_grid = coarse_matches_up[:, :2].permute(0, 2, 3, 1)
        feats_x1_f_warped = grid_sample(
            feats_x1_f.float(), fine_grid, mode="bilinear", align_corners=False
        ).to(self.compute_dtype)
        fw_in = coarse_matches_up[:, :2].to(self.compute_dtype)
        if self.fine_coord_scale != 1.0:
            fw_in = fw_in * self.fine_coord_scale
        fine_in = torch.cat((feats_x0_f, feats_x1_f_warped, fw_in), dim=1)
        fine_delta = self.fine_matcher(fine_in)
        fine_matches = coarse_matches_up + fine_delta * to_normalized

        return {
            8: {"flow": coarse_matches[:, :2], "certainty": coarse_matches[:, 2:]},
            4: {"flow": fine_matches[:, :2], "certainty": fine_matches[:, 2:]},
        }

    # -- image-level API -------------------------------------------------------------------

    def load_image(self, img: ImageLike) -> Tensor:
        return load_image(img, self.device)

    @staticmethod
    def _resize_image(image: Tensor, resize: int | None) -> Tensor:
        if resize is None:
            return image
        h, w = image.shape[-2:]
        new_h, new_w = sparse.target_size(h, w, resize, TinyRoMa.div_by)
        if (new_h, new_w) == (h, w):
            return image
        return F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def postprocess(self, corresps: dict[int, dict[str, Tensor]], size: tuple[int, int]) -> DenseMatchResult:
        """Upsample stride-4 predictions to ``size`` ``(H0, W0)`` and wrap in :class:`DenseMatchResult`."""
        flow = F.interpolate(
            corresps[4]["flow"].float(), size=size, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).contiguous()
        certainty = F.interpolate(
            corresps[4]["certainty"].float(), size=size, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).contiguous()
        return DenseMatchResult(
            warp_AB=flow,
            confidence_AB=certainty,
            overlap_AB=certainty.sigmoid(),
            precision_AB=None,
        )

    @torch.inference_mode()
    def match_dense(self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None) -> DenseMatchResult:
        """Dense warp and certainty at the working resolution of ``img_a`` (original size when ``resize=None``)."""
        a = self._resize_image(self.load_image(img_a), resize)
        b = self._resize_image(self.load_image(img_b), resize)
        return self.postprocess(self.forward(a, b), a.shape[-2:])

    def sample(
        self,
        result: DenseMatchResult,
        num: int,
        *,
        threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample ``num`` KDE-balanced matches ``[N, 4]`` (xA, yA, xB, yB normalized) and certainty ``[N]`` from
        batch item 0.

        Matches upstream ``TinyRoMa.sample`` bit-identically when ``fast_sampling=False`` (using the reference FP16
        KDE on CUDA, or strided FP32 KDE on CPU); ``fast_sampling=True`` uses FlashIM's tiled Triton ``ops.kde_density``
        on CUDA.
        """
        warp = result.warp_AB[0]
        h, w, _ = warp.shape
        device = warp.device
        certainty = result.overlap_AB[0, ..., 0]
        if "threshold" in self.config.sample_mode:
            upper_thresh = self.config.sample_thresh if threshold is None else threshold
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1.0
        certainty_flat = certainty.reshape(-1)
        warp_flat = warp.reshape(-1, 2)

        expansion_factor = 4 if "balanced" in self.config.sample_mode else 1
        good_samples = torch.multinomial(
            certainty_flat,
            num_samples=min(expansion_factor * num, certainty_flat.shape[0]),
            replacement=False,
            generator=generator,
        )
        # Reconstruct (xA, yA) only for the drawn candidates (bit-identical to full meshgrid materialization).
        xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device)
        ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device)
        y_idx = good_samples // w
        x_idx = good_samples % w
        good_grid = torch.stack((xs[x_idx], ys[y_idx]), dim=-1)
        good_matches = torch.cat((good_grid, warp_flat[good_samples]), dim=-1)
        good_certainty = certainty_flat[good_samples]

        if "balanced" not in self.config.sample_mode:
            return good_matches, good_certainty
        if device.type == "cuda":
            if self.fast_sampling:
                density = ops.kde_density(good_matches)
            else:
                density = ops.kde_density.get("reference")(good_matches)
        else:
            pts = good_matches.float()
            density = (-(torch.cdist(pts, pts[::8]) ** 2) / (2 * 0.1**2)).exp().sum(dim=-1)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7
        balanced = torch.multinomial(
            p, num_samples=min(num, good_certainty.shape[0]), replacement=False, generator=generator
        )
        return good_matches[balanced], good_certainty[balanced]

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
        """``max_matches`` KDE-balanced samples of the dense warp as pixel coordinates of each *original* image."""
        if max_matches is None:
            raise ValueError("TinyRoMa.match needs max_matches (samples of the dense warp); match_dense returns the warp")
        a, b = self.load_image(img_a), self.load_image(img_b)
        (h_a, w_a), (h_b, w_b) = a.shape[-2:], b.shape[-2:]
        a_work, b_work = self._resize_image(a, resize), self._resize_image(b, resize)
        dense = self.postprocess(self.forward(a_work, b_work), a_work.shape[-2:])
        matches, confidence = self.sample(dense, max_matches, threshold=threshold, generator=generator)
        keypoints_a, keypoints_b = self.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
        return Matches(keypoints_a, keypoints_b, confidence, dense=dense)

    @staticmethod
    def to_pixel_coordinates(matches: Tensor, h_a: int, w_a: int, h_b: int, w_b: int) -> tuple[Tensor, Tensor]:
        return to_pixel(matches[..., :2], h=h_a, w=w_a), to_pixel(matches[..., 2:], h=h_b, w=w_b)


def apply_inference(model: TinyRoMa, inference: Mapping[str, Any]) -> None:
    model.fast_sampling = inference.get("fast_sampling", os.environ.get("FLASHIM_FAST_SAMPLING", "1") != "0")
