"""RaCo keypoint detector + ranker (architecture of cvg/RaCo).

A small ALIKED-style CNN (no deformable convs) predicts a keypoint logit map; local maxima are taken as
``candidate_multiplier x K`` candidates (soft-argmax refined); a second, full-resolution residual CNN (the
*ranker*) scores each candidate and the top ``K`` are kept. Module paths follow ``raco.pth``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim import ops
from flashim.models.raco import keypoints as kp
from flashim.models.raco import weights
from flashim.models.raco.blocks import ConvBlock, ResBlock
from flashim.models.raco.config import RaCoConfig
from flashim.platforms import default_device
from flashim.utils import device_constant


class RaCo(nn.Module):
    def __init__(self, config: RaCoConfig | None = None) -> None:
        super().__init__()
        self.config = cfg = config or RaCoConfig()
        if cfg.nms_radius % 2 == 0:
            raise ValueError("nms_radius must be odd")
        self.gate = nn.SELU()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ResBlock(16, 32)
        self.block3 = ResBlock(32, 64)
        self.block4 = ResBlock(64, 128)
        self.conv1 = nn.Conv2d(16, 32, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 32, 3, padding=1, bias=False)
        self.score_head = nn.Sequential(
            nn.Conv2d(128, 8, 1, bias=False),
            nn.SELU(),
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            nn.SELU(),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.SELU(),
            nn.Conv2d(4, 1, 3, padding=1, bias=False),
        )
        width = cfg.ranker_width
        self.ranker_head = nn.Sequential(
            ResBlock(3, width),
            *(ResBlock(width, width) for _ in range(cfg.ranker_blocks - 1)),
            nn.Conv2d(width, 1, 5, padding=2, padding_mode="reflect"),
        )
        self.preset: str | None = None
        self.engine: str | None = None
        self.compute_dtype = torch.float32
        self._fused_ranker: tuple | None = None
        self.ranker_backend: str | None = None
        self._ranker_buffers: dict[tuple[int, int], Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        config: RaCoConfig | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> RaCo:
        device = torch.device(device or default_device())
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict(checkpoint)
        model.load_state_dict(state, strict=True, assign=True)
        model.to(device)
        return model.eval()

    @property
    def device(self) -> torch.device:
        return self.block1.conv1.weight.device

    def normalize(self, image: Tensor) -> Tensor:
        mean, std = (device_constant(v, image).view(1, 3, 1, 1) for v in (self.config.mean, self.config.std))
        return (image - mean) / std

    def detector_logits(self, image: Tensor) -> Tensor:
        """Keypoint logit map ``[B, 1, H, W]`` for a normalized image (H, W multiples of 32)."""
        x = image.to(self.compute_dtype)
        x1 = self.block1(x)
        x2 = self.block2(F.avg_pool2d(x1, 2))
        x3 = self.block3(F.avg_pool2d(x2, 4))
        x4 = self.block4(F.avg_pool2d(x3, 4))

        head_weight = self.score_head[0].weight.reshape(self.score_head[0].out_channels, -1)
        head, _ = ops.multiscale_fuse(
            self.gate(self.conv1(x1)),
            self.gate(self.conv2(x2)),
            self.gate(self.conv3(x3)),
            self.gate(self.conv4(x4)),
            head_weight,
            normalize=False,
        )
        return self.score_head[2:](head)

    def ranker_map(self, image: Tensor) -> Tensor:
        """Ranker score map ``[B, 1, H, W]`` (FP32) for a normalized image."""
        if self._fused_ranker is not None and image.shape[0] == 1:
            return self._fused_ranker_map(image)
        return self.ranker_head(image.to(self.compute_dtype)).float()

    def use_fused_ranker(self, backend: str = "auto") -> None:
        """Run the ranker's residual blocks and 5x5 head through the fused ops (``ops.raco_resblock`` /
        ``ops.raco_head5x5``: one read + one write of the activation per block instead of ~5 passes) at batch
        size 1. ``backend``: ``"auto"`` (normal dispatch: the CUDA C++ kernel when nvcc builds it, else Triton, else
        the torch reference) or an impl name to pin (``"cuda"``, ``"triton"``). Needs BatchNorm folded
        (:func:`~flashim.models.raco.fuse.fold_batchnorm`); ``ranker_head`` stays for batched inputs. Faster than
        the compiled BF16 ranker and closer to FP32 (99.8% of the selected keypoints identical vs 99.6%)."""
        blocks = [m for m in self.ranker_head if isinstance(m, ResBlock)]
        if any(isinstance(b.bn1, nn.BatchNorm2d) for b in blocks):
            raise RuntimeError("use_fused_ranker needs BatchNorm folded first")
        pin = None if backend == "auto" else backend
        resblocks = [
            ops.raco_resblock.bind(
                b.conv1.weight, b.conv1.bias, b.conv2.weight, b.conv2.bias, b.match_dims.weight, b.match_dims.bias,
                impl=pin,
            )
            for b in blocks
        ]
        head = self.ranker_head[-1]
        if pin is not None and pin not in resblocks[0].available:
            raise RuntimeError(f"raco_resblock impl {pin!r} cannot run on {resblocks[0].platform.describe()}")
        self.ranker_backend = resblocks[0].primary
        self._fused_ranker = (resblocks, ops.raco_head5x5.bind(head.weight, head.bias))

    def _fused_ranker_map(self, image: Tensor) -> Tensor:
        resblocks, head = self._fused_ranker
        h, w = image.shape[-2:]
        buffer = self._ranker_buffers.get((h, w))
        if buffer is None:  # 4 channels per pixel (3 valid): the input's zero padding is written once
            buffer = self._ranker_buffers[(h, w)] = torch.zeros(h, w, 4, device=image.device, dtype=torch.bfloat16)
        buffer[..., :3] = image[0].permute(1, 2, 0)
        current, cin = buffer, 3
        for block in resblocks:
            current = block(current, cin)
            cin = self.config.ranker_width
        return head(current)

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Run the detector trunk and the ranker in ``dtype``; the detector's score head, NMS / soft-argmax and the
        ranker's sampling stay FP32. (BF16 ranker: 99.6% of the selected keypoints identical, score corr 0.9999.)"""
        for module in (self.block1, self.block2, self.block3, self.block4, self.conv1, self.conv2, self.conv3, self.conv4, self.ranker_head):
            module.to(dtype)
        self.compute_dtype = dtype

    def config_for(self, num_keypoints: int | None) -> RaCoConfig:
        return self.config if num_keypoints is None else dataclasses.replace(self.config, num_keypoints=num_keypoints)

    def detect(self, image: Tensor, num_keypoints: int | None = None) -> Tensor:
        """Detector candidates ``[B, N_candidates, 2]`` (``(x, y)`` pixel-index, soft-argmax refined) for a
        normalized image."""
        cfg = self.config_for(num_keypoints)
        logits = self.detector_logits(image)
        return kp.detect_candidates(logits, cfg.num_candidates, cfg.nms_radius, cfg.subpixel_temperature)

    def rank(self, ranker_map: Tensor, candidates: Tensor, num_keypoints: int | None = None) -> tuple[Tensor, Tensor]:
        """Top ``num_keypoints`` candidates by ranker score: ``(keypoints [B, K, 2] (+0.5 pixel-edge convention),
        scores [B, K])``, best first."""
        cfg = self.config_for(num_keypoints)
        top_scores, order = kp.sample(ranker_map, candidates).topk(cfg.num_keypoints, dim=1)
        return candidates.gather(1, order[..., None].expand(-1, -1, 2)) + 0.5, top_scores

    @torch.inference_mode()
    def extract_keypoints(self, image: Tensor, num_keypoints: int | None = None) -> tuple[Tensor, Tensor]:
        """``image`` ``[B, 3, H, W]`` RGB in ``[0, 1]`` (H, W multiples of 32) -> ``(keypoints [B, K, 2],
        ranker_scores [B, K])``. Keypoints are ``(x, y)`` in RaCo's pixel-edge convention (pixel index + 0.5),
        highest ranker score first. ``num_keypoints`` overrides the configured count (the candidate pool follows)."""
        image = self.normalize(image)
        return self.rank(self.ranker_map(image), self.detect(image, num_keypoints), num_keypoints)
