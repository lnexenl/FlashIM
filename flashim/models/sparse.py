"""Shared plumbing of the sparse detector/descriptor models (SuperPoint, ALIKE, DISK, XFeat, DeDoDe, RIPE, ...).

A subclass implements :meth:`SparseExtractor.detect` (``[1, 3, H, W]`` RGB in ``[0, 1]`` at the working resolution ->
keypoints ``[K, 2]`` ``(x, y)`` in working-resolution pixel indices, scores ``[K]``, descriptors ``[K, C]``);
the base class handles loading, resizing, mapping keypoints back to the original image and mutual-nearest-neighbour
matching. Batch size is 1 (as for ALIKED / RDD).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashim.models.romav2.model import ImageLike, load_image
from flashim.types import SparseFeatures, SparseMatchResult
from flashim.utils import device_constant, set_submodule


def target_size(height: int, width: int, long_side: int | None, div_by: int = 8) -> tuple[int, int]:
    """Working resolution: the long side scaled to ``long_side`` (if given), both sides floored to ``div_by``."""
    if long_side is not None:
        aspect = width / height
        height, width = (long_side, int(long_side * aspect)) if height > width else (int(long_side / aspect), long_side)
    return max(div_by, (height // div_by) * div_by), max(div_by, (width // div_by) * div_by)


def rgb_to_gray(image: Tensor) -> Tensor:
    """ITU-R 601 luma (``kornia.color.rgb_to_grayscale``): ``[B, 3, H, W] -> [B, 1, H, W]``."""
    r, g, b = image.unbind(1)
    return (0.299 * r + 0.587 * g + 0.114 * b)[:, None]


def simple_nms(scores: Tensor, radius: int) -> Tensor:
    """SuperPoint's fast non-maximum suppression over ``[B, H, W]`` (or ``[B, 1, H, W]``) score maps."""
    def max_pool(x):
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k(keypoints: Tensor, scores: Tensor, k: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the ``k`` best-scoring keypoints (all if ``k`` is None / >= count); also returns the kept indices."""
    if k is None or k <= 0 or k >= len(keypoints):
        return keypoints, scores, torch.arange(len(keypoints), device=keypoints.device)
    scores, index = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[index], scores, index


def sample_descriptors(keypoints: Tensor, descriptors: Tensor, stride: int) -> Tensor:
    """Bilinear descriptors ``[K, C]`` (L2-normalized) from a ``[1, C, h, w]`` map at ``keypoints`` ``[K, 2]`` given in
    pixel indices of the image the map was computed from (``stride`` = the map's downsampling factor)."""
    _, _, h, w = descriptors.shape
    kpts = (keypoints.float() - stride / 2 + 0.5) / device_constant(
        [w * stride - stride / 2 - 0.5, h * stride - stride / 2 - 0.5], keypoints
    )
    grid = (kpts * 2 - 1).view(1, 1, -1, 2)
    out = F.grid_sample(descriptors.float(), grid, mode="bilinear", align_corners=True)[0, :, 0].t()
    return F.normalize(out, dim=-1)


def mutual_nn_match(desc0: Tensor, desc1: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
    """Mutual-nearest-neighbour cosine matches with similarity >= ``threshold`` (L2-normalized ``[N, C]``)."""
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:  # no keypoints in one image: argmax has no dim to reduce
        empty = torch.zeros(0, dtype=torch.long, device=desc0.device)
        return empty, empty, desc0.new_zeros(0)
    sim = desc0.float() @ desc1.float().t()
    nn01, nn10 = sim.argmax(dim=1), sim.argmax(dim=0)
    best = sim.gather(1, nn01[:, None])[:, 0]
    ids0 = torch.arange(sim.shape[0], device=sim.device)
    idx0 = ((ids0 == nn10[nn01]) & (best >= threshold)).nonzero()[:, 0]  # the only host sync (match count)
    return idx0, nn01[idx0], best[idx0]


class SparseExtractor(nn.Module):
    """Base class: ``extract`` / ``match_features`` / ``match`` over a subclass' :meth:`detect`."""

    div_by: int = 8
    match_threshold: float = 0.0  # mutual-NN cosine threshold of the built-in matcher
    default_top_k: int | None = None

    fp32_modules: tuple[str, ...] = ()  # submodule paths kept in FP32 under a reduced compute dtype
    compute_dtype = torch.float32

    # Filled by the builder.
    preset: str | None = None
    engine: str | None = None
    quant_report = None
    calibration_metadata = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.to(dtype)
        for path in self.fp32_modules:
            self.get_submodule(path).float()
        self.compute_dtype = dtype

    def fold_batchnorm(self) -> list[str]:
        """Fold every ``(conv, bn)`` of :meth:`batchnorm_pairs` (exact in eval mode); returns the folded conv paths."""
        if self.training:
            raise RuntimeError("fold_batchnorm requires model.eval()")
        folded = []
        for conv_path, bn_path in self.batchnorm_pairs():
            conv, bn = self.get_submodule(conv_path), self.get_submodule(bn_path)
            if not isinstance(bn, nn.BatchNorm2d):
                continue  # already folded
            set_submodule(self, conv_path, nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
            set_submodule(self, bn_path, nn.Identity())
            folded.append(conv_path)
        return folded

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return []

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        """Keypoints / descriptors of one image. ``resize``: long side of the working resolution; keypoints come
        back in the original image's pixel coordinates (integer = pixel center)."""
        image = load_image(image, self.device)
        if image.shape[0] != 1:
            raise ValueError(f"{type(self).__name__}.extract processes one image at a time (batch size 1)")
        height, width = image.shape[-2:]
        new_h, new_w = target_size(height, width, resize, self.div_by)
        if (new_h, new_w) != (height, width):
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        keypoints, scores, descriptors = self.detect(image, self.default_top_k if top_k is None else top_k)
        scale = device_constant([width / new_w, height / new_h], keypoints)
        keypoints = (keypoints.float() + 0.5) * scale - 0.5
        return SparseFeatures(keypoints[None], scores.float()[None], descriptors[None], (height, width))

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> SparseMatchResult:
        ia, ib, conf = mutual_nn_match(
            a.descriptors[0], b.descriptors[0], self.match_threshold if threshold is None else threshold
        )
        return SparseMatchResult(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)

    @torch.inference_mode()
    def match(
        self, img_a: ImageLike, img_b: ImageLike, *, resize: int | None = None, top_k: int | None = None,
        threshold: float | None = None,
    ) -> SparseMatchResult:
        return self.match_features(
            self.extract(img_a, resize=resize, top_k=top_k), self.extract(img_b, resize=resize, top_k=top_k), threshold
        )
