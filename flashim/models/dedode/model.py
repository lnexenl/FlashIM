"""DeDoDe (Edstedt et al., 2024) and, with DaD's detector, DaD + DeDoDe descriptors: detector network, descriptor
network, dual-softmax matching. Detector and descriptor are separate networks (``detector`` / ``descriptor``)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim import ops
from flashim.models import sparse
from flashim.models.dedode import blocks
from flashim.platforms import default_device
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SamplerConfig:
    """Keypoint sampling from the detector's dense probability map (upstream ``sample_keypoints``)."""

    nms_size: int = 0  # 0: no NMS
    remove_borders: bool = False
    increase_coverage: bool = True
    coverage_pow: float = 0.5
    coverage_size: int = 51
    subpixel: bool = False
    subpixel_temp: float = 0.5


DEDODE_SAMPLER = SamplerConfig()
DAD_SAMPLER = SamplerConfig(nms_size=3, remove_borders=True, increase_coverage=False, subpixel=True, subpixel_temp=0.5)


def sample_keypoints(logits: Tensor, top_k: int, cfg: SamplerConfig) -> tuple[Tensor, Tensor]:
    """``logits`` ``[1, H, W]`` -> keypoints ``[K, 2]`` ``(x, y)`` in pixel indices (float) and their probabilities."""
    _, h, w = logits.shape
    probs = logits.reshape(1, h * w).softmax(-1).reshape(1, h, w)
    if cfg.increase_coverage:
        weights_1d = (-(torch.linspace(-2, 2, steps=cfg.coverage_size, device=logits.device) ** 2)).exp()[None, None]
        density = F.conv2d((probs[:, None] + 1e-6) * 10000, weights_1d[..., None, :], padding=(0, cfg.coverage_size // 2))
        density = F.conv2d(density, weights_1d[..., None], padding=(cfg.coverage_size // 2, 0))[:, 0]
        probs = probs * (density + 1e-8) ** (-cfg.coverage_pow)
    if cfg.nms_size:
        probs = probs * (probs == F.max_pool2d(probs, cfg.nms_size, stride=1, padding=cfg.nms_size // 2))
    if cfg.remove_borders:
        frame = torch.zeros_like(probs)
        frame[..., 4:-4, 4:-4] = 1
        probs = probs * frame
    index = torch.topk(probs.reshape(-1), k=min(top_k, h * w)).indices
    keypoints = torch.stack((index % w, index // w), dim=-1).float()
    if cfg.subpixel:
        k = cfg.nms_size
        patches = F.unfold(logits[:, None], kernel_size=k, padding=k // 2)[0][:, index]  # [k*k, N]
        offsets = torch.stack(torch.meshgrid(*[torch.arange(k, device=logits.device) - k // 2] * 2, indexing="ij"), -1)
        offsets = offsets.reshape(k * k, 2).flip(-1).float()  # (dx, dy), row-major like the patch unfolding
        keypoints = keypoints + torch.einsum("kn,kd->nd", (patches / cfg.subpixel_temp).softmax(0), offsets)
    return keypoints, probs.reshape(-1)[index]


def bilinear_at(feature_map: Tensor, keypoints: Tensor) -> Tensor:
    """``[1, C, H, W]`` map sampled at pixel-index ``keypoints`` ``[K, 2]`` (``align_corners=False``) -> ``[K, C]``."""
    h, w = feature_map.shape[-2:]
    grid = ((keypoints + 0.5) * 2 / device_constant([w, h], keypoints) - 1).view(1, 1, -1, 2)
    return F.grid_sample(feature_map.float(), grid, mode="bilinear", align_corners=False)[0, :, 0].t()


class DeDoDe(sparse.SparseExtractor):
    """``detect``: ImageNet-normalized image -> detector logits (one map) -> softmax over all pixels -> sampler (top-k) ->
    descriptor map sampled bilinearly at the keypoints, L2-normalized. Matching is DeDoDe's dual softmax (inverse
    temperature 20, mutual maxima above 0.01) on the normalized descriptors."""

    default_top_k = 10_000
    inv_temperature = 20.0
    match_threshold = 0.01

    def __init__(self, detector: blocks.NetConfig, descriptor: blocks.NetConfig, sampler: SamplerConfig) -> None:
        super().__init__()
        self.detector, self.descriptor = blocks.DeDoDeNet(detector), blocks.DeDoDeNet(descriptor)
        self.sampler = sampler
        self.div_by = 56 if descriptor.encoder == "vgg19+dinov2" else 8  # lcm(VGG stride 8, DINOv2 patch 14)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, detector: blocks.NetConfig, descriptor: blocks.NetConfig, sampler: SamplerConfig, *,
                        detector_state: dict, descriptor_state: dict, dinov2_state: dict | None = None,
                        device: torch.device | str | None = None) -> DeDoDe:
        with torch.device("meta"):
            model = cls(detector, descriptor, sampler)
        state = {f"detector.{k}": v for k, v in detector_state.items()}
        state |= {f"descriptor.{k}": v for k, v in descriptor_state.items()}
        if dinov2_state is not None:
            state |= {f"descriptor.encoder.dinov2.{k}": v for k, v in dinov2_state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        model.mean, model.std = (torch.tensor(v).view(1, 3, 1, 1) for v in (MEAN, STD))  # meta-built buffers
        return model.to(torch.device(device or default_device())).eval()

    def batchnorm_pairs(self) -> list[tuple[str, str]]:
        return self.detector.batchnorm_pairs("detector.") + self.descriptor.batchnorm_pairs("descriptor.")

    def normalize(self, image: Tensor) -> Tensor:
        return ((image - self.mean) / self.std).to(self.compute_dtype)

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        x = self.normalize(image)
        logits = self.detector(x)[0]  # [1, H, W]
        keypoints, probs = sample_keypoints(logits, top_k or self.default_top_k, self.sampler)
        # = bilinear_at(self.descriptor(x), keypoints), without forming the last scale's dense map
        descriptors = F.normalize(self.descriptor(x, points=keypoints), dim=-1)
        return keypoints, probs, descriptors

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        return F.normalize(self.descriptor(self.normalize(image), points=keypoints.float()), dim=-1)

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        corr = (a.descriptors[0].float() @ b.descriptors[0].float().t()) * self.inv_temperature
        # mutual maxima of softmax(corr, -2) * softmax(corr, -1) above the threshold, without the [M, N] probabilities:
        # measured 5.7 -> 0.74 ms for 4096 x 4096 keypoints (sm_120), identical match lists on 100 MegaDepth pairs
        log_conf, index, mutual = ops.dual_softmax_mutual(corr)
        conf = log_conf.exp()
        limit = self.match_threshold if threshold is None else threshold
        ia = torch.nonzero(mutual & (conf > limit))[:, 0]
        ib = index[ia]
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf[ia], ia, ib)
