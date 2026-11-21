# Portions adapted from RoMa (MIT, Copyright (c) 2023 Johan Edstedt); see THIRD_PARTY_NOTICES.md.
"""RoMa (v1) dense matcher, re-implemented natively for FlashIM."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models.base import ImageMatcher
from flashim.models.roma import weights
from flashim.models.roma.config import RoMaConfig
from flashim.models.roma.decoder import Decoder
from flashim.models.roma.encoder import CNNandDinov2
from flashim.models.romav2.geometry import imagenet_normalize, normalized_grid, to_pixel
from flashim.models.romav2.model import ImageLike, _highest_matmul_precision, load_image_raw
from flashim.platforms import default_device
from flashim.types import DenseMatchResult, Matches


def prepare_pair(
    img_a: ImageLike, img_b: ImageLike, config: RoMaConfig, device: torch.device
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, tuple[int, int], tuple[int, int]]:
    """Load an image pair, resize with PIL-matched bicubic (``ops.pil_bicubic_resize``) to coarse (and upsample)
    resolutions, and return ``(a_lr, b_lr, a_hr, b_hr, (h_a, w_a), (h_b, w_b))`` in ``[0, 1]`` FP32."""
    raw_a, raw_b = load_image_raw(img_a, device), load_image_raw(img_b, device)
    size_a = (raw_a.shape[-2], raw_a.shape[-1])
    size_b = (raw_b.shape[-2], raw_b.shape[-1])
    sizes = [config.coarse_res, config.upsample_res] if config.upsample_preds else [config.coarse_res]
    a_resized = ops.pil_bicubic_resize(raw_a, sizes)
    b_resized = ops.pil_bicubic_resize(raw_b, sizes)
    if config.upsample_preds:
        return a_resized[0], b_resized[0], a_resized[1], b_resized[1], size_a, size_b
    return a_resized[0], b_resized[0], None, None, size_a, size_b


class RoMa(ImageMatcher, nn.Module):
    """RoMa (v1): ``encoder`` (VGG19-BN + frozen DINOv2 ViT-L/14) -> ``decoder`` (scale-16 GP + 5-layer Transformer
    decoder + 5-scale ConvRefiners).

    Parameter names match the upstream ``roma_{outdoor,indoor}.pth`` checkpoints plus ``encoder.dinov2.*`` for the
    frozen DINOv2 weights, so checkpoints load with ``strict=True``. ``match`` draws ``max_matches`` KDE-balanced
    correspondences (5000 by default, matching :class:`~flashim.models.romav2.RoMaV2`) and attaches the full
    :class:`~flashim.types.DenseMatchResult` as ``dense``.
    """

    match_defaults: Mapping[str, object] = {"max_matches": 5000}

    def __init__(self, config: RoMaConfig | None = None) -> None:
        super().__init__()
        self.config = config or RoMaConfig()
        self.encoder = CNNandDinov2()
        self.decoder = Decoder()
        # sample(): ops.kde_density (Triton FP32 KDE, no O(N^2) matrix) instead of upstream's FP16 cdist matrix.
        self.fast_sampling = True
        # Filled in by the builder.
        self.preset: str | None = None
        self.engine: str | None = None
        self.quant_report = None
        self.calibration_metadata: dict[str, str] | None = None

    @classmethod
    def from_pretrained(
        cls,
        config: RoMaConfig | None = None,
        *,
        variant: str = "outdoor",
        checkpoint: str | Path | None = None,
        dinov2_checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
        dinov2_state: dict[str, Tensor] | None = None,
    ) -> RoMa:
        device = torch.device(device or default_device())
        with torch.device("meta"):
            model = cls(config)
        roma_state = state_dict if state_dict is not None else weights.load_state_dict(variant, path=checkpoint)
        dino_state = dinov2_state if dinov2_state is not None else weights.load_dinov2(path=dinov2_checkpoint)
        merged = dict(roma_state)
        merged.update({f"encoder.dinov2.{k}": v for k, v in dino_state.items()})
        model.load_state_dict(merged, strict=True, assign=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return self.encoder.batchnorm_pairs("encoder.") + self.decoder.batchnorm_pairs("decoder.")

    def _run_stage(
        self,
        im_a: Tensor,
        im_b: Tensor,
        *,
        upsample: bool,
        flow: Tensor | None = None,
        certainty: Tensor | None = None,
    ) -> dict[int, dict[str, Tensor]]:
        hs, ws = im_a.shape[-2:]
        scale_factor = math.sqrt(hs * ws / (560.0**2))
        stacked = torch.cat((im_a, im_b), dim=0)
        pyramid = self.encoder(stacked, upsample=upsample)
        if self.config.symmetric:
            return self.decoder(
                pyramid,
                pyramid,
                upsample=upsample,
                flow=flow,
                certainty=certainty,
                scale_factor=scale_factor,
                symmetric=True,
            )
        f_q = {s: v.chunk(2, dim=0)[0] for s, v in pyramid.items()}
        f_s = {s: v.chunk(2, dim=0)[1] for s, v in pyramid.items()}
        return self.decoder(
            f_q,
            f_s,
            upsample=upsample,
            flow=flow,
            certainty=certainty,
            scale_factor=scale_factor,
            symmetric=False,
        )

    @torch.inference_mode()
    def forward_normalized(
        self,
        img_a_lr: Tensor,
        img_b_lr: Tensor,
        img_a_hr: Tensor | None = None,
        img_b_hr: Tensor | None = None,
    ) -> DenseMatchResult:
        """Dense match on already ImageNet-normalized ``[B, 3, H, W]`` tensors (matches upstream tensor API)."""
        cfg = self.config
        with _highest_matmul_precision():
            corresps = self._run_stage(img_a_lr, img_b_lr, upsample=False)
            out_h, out_w = (img_a_hr.shape[-2:] if (cfg.upsample_preds and img_a_hr is not None) else img_a_lr.shape[-2:])

            low_res_cert: Tensor | None = None
            if cfg.attenuate_cert:
                low_res_cert = F.interpolate(
                    corresps[16]["certainty"], size=(out_h, out_w), align_corners=False, mode="bilinear"
                )
                low_res_cert = 0.5 * low_res_cert * (low_res_cert < 0)

            finest = corresps[1]
            if cfg.upsample_preds and img_a_hr is not None and img_b_hr is not None:
                corresps = self._run_stage(
                    img_a_hr,
                    img_b_hr,
                    upsample=True,
                    flow=finest["flow"],
                    certainty=finest["certainty"],
                )
                finest = corresps[1]

        flow = finest["flow"].permute(0, 2, 3, 1)  # [2B or B, H, W, 2]
        raw_cert = finest["certainty"]
        if low_res_cert is not None:
            raw_cert = raw_cert - low_res_cert
        overlap = raw_cert.sigmoid()
        wrong = (flow.abs() > 1).any(dim=-1)
        overlap = overlap.masked_fill(wrong[:, None], 0.0)
        flow = flow.clamp(-1.0, 1.0)
        raw_cert_bhwc = raw_cert.permute(0, 2, 3, 1)
        overlap_bhwc = overlap.permute(0, 2, 3, 1)

        if cfg.symmetric:
            warp_ab, warp_ba = flow.chunk(2, dim=0)
            conf_ab, conf_ba = raw_cert_bhwc.chunk(2, dim=0)
            ov_ab, ov_ba = overlap_bhwc.chunk(2, dim=0)
            return DenseMatchResult(
                warp_AB=warp_ab,
                confidence_AB=conf_ab,
                overlap_AB=ov_ab,
                precision_AB=None,
                warp_BA=warp_ba,
                confidence_BA=conf_ba,
                overlap_BA=ov_ba,
                precision_BA=None,
            )
        return DenseMatchResult(
            warp_AB=flow,
            confidence_AB=raw_cert_bhwc,
            overlap_AB=overlap_bhwc,
            precision_AB=None,
        )

    @torch.inference_mode()
    def forward(
        self,
        img_a_lr: Tensor,
        img_b_lr: Tensor,
        img_a_hr: Tensor | None = None,
        img_b_hr: Tensor | None = None,
    ) -> DenseMatchResult:
        """Images are ``[B, 3, H, W]`` floats in ``[0, 1]`` at the coarse (and optional upsample) resolutions."""
        a_lr, b_lr = imagenet_normalize(img_a_lr), imagenet_normalize(img_b_lr)
        a_hr = imagenet_normalize(img_a_hr) if img_a_hr is not None else None
        b_hr = imagenet_normalize(img_b_hr) if img_b_hr is not None else None
        return self.forward_normalized(a_lr, b_lr, a_hr, b_hr)

    @torch.inference_mode()
    def match_dense(self, img_a: ImageLike, img_b: ImageLike) -> DenseMatchResult:
        """Dense warp + certainty prediction for a pair of images."""
        a_lr, b_lr, a_hr, b_hr, _, _ = prepare_pair(img_a, img_b, self.config, self.device)
        return self.forward(a_lr, b_lr, a_hr, b_hr)

    def sample(
        self,
        result: DenseMatchResult,
        num: int = 5000,
        *,
        threshold: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample ``num`` KDE-balanced normalized matches ``([N, 4], [N])`` from batch item 0.

        Matches upstream ``RegressionMatcher.sample`` (``threshold_balanced`` mode): certainties above
        ``threshold`` (default :attr:`RoMaConfig.sample_thresh` = 0.05) are clamped to 1, ``4 * num`` candidates are
        drawn without replacement, and ``num`` are subsampled inversely to Gaussian KDE density (``std = 0.1``).
        Uses the exact upstream ``[H, 2W]`` row-major ordering so fixed-seed draws match upstream bit-for-bit when
        ``kde_density:reference`` is selected.
        """
        thresh = self.config.sample_thresh if threshold is None else threshold
        warp_ab = result.warp_AB[0]
        h, w, _ = warp_ab.shape
        grid = normalized_grid(1, h, w, warp_ab.device)[0]
        cert_ab = result.overlap_AB[0, ..., 0]
        if result.bidirectional:
            assert result.warp_BA is not None and result.overlap_BA is not None
            cert = torch.cat((cert_ab, result.overlap_BA[0, ..., 0]), dim=1).reshape(-1)
        else:
            cert = cert_ab.reshape(-1)
        if thresh is not None:
            cert = torch.where(cert > thresh, torch.ones_like(cert), cert)

        expansion = 4
        n_cand = min(expansion * num, cert.numel())
        good_idx = torch.multinomial(cert, num_samples=n_cand, replacement=False, generator=generator)
        good_cert = cert[good_idx]

        if result.bidirectional:
            row = good_idx // (2 * w)
            col = good_idx % (2 * w)
            is_ab = col < w
            pix = row * w + torch.where(is_ab, col, col - w)
            g_flat = grid.reshape(-1, 2)[pix]
            w_ab = warp_ab.reshape(-1, 2)[pix]
            w_ba = result.warp_BA[0].reshape(-1, 2)[pix]
            good_matches = torch.where(is_ab[:, None], torch.cat((g_flat, w_ab), -1), torch.cat((w_ba, g_flat), -1))
        else:
            good_matches = torch.cat((grid.reshape(-1, 2)[good_idx], warp_ab.reshape(-1, 2)[good_idx]), -1)

        density = ops.kde_density(good_matches, std=0.1) if self.fast_sampling else ops.kde_density.get("reference")(good_matches, std=0.1)
        p = 1.0 / (density + 1.0)
        p[density < 10] = 1e-7
        balanced = torch.multinomial(p, num_samples=min(num, len(good_cert)), replacement=False, generator=generator)
        return good_matches[balanced], good_cert[balanced]

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
        """``max_matches`` balanced matches as pixel keypoints of the original images; attaches ``dense``."""
        if resize is not None:
            raise ValueError("RoMa.match takes no resize: set arch.coarse_res / arch.upsample_res instead")
        if max_matches is None:
            raise ValueError("RoMa.match needs max_matches (samples of the dense warp); match_dense returns the warp")
        a_lr, b_lr, a_hr, b_hr, (h_a, w_a), (h_b, w_b) = prepare_pair(img_a, img_b, self.config, self.device)
        dense = self.forward(a_lr, b_lr, a_hr, b_hr)
        matches, confidence = self.sample(dense, max_matches, threshold=threshold, generator=generator)
        keypoints_a, keypoints_b = self.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
        return Matches(keypoints_a, keypoints_b, confidence, dense=dense)

    @staticmethod
    def to_pixel_coordinates(matches: Tensor, h_a: int, w_a: int, h_b: int, w_b: int) -> tuple[Tensor, Tensor]:
        return to_pixel(matches[..., :2], h=h_a, w=w_a), to_pixel(matches[..., 2:], h=h_b, w=w_b)
