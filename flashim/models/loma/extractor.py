"""LoMa's keypoint extractor: DaD's detector (VGG11, ``dad.pth``) and the variant's LoMa-trained DeDoDe descriptor
(B128: DeDoDe-B with 128-dim output; B / L / G / R: DeDoDe-G with DINOv2 ViT-L/14), as upstream's
``LoMa.detect_and_describe`` on an image path."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from flashim.models import sparse
from flashim.models.dedode import blocks
from flashim.models.dedode.model import DeDoDe, SamplerConfig, sample_keypoints
from flashim.models.loma import weights
from flashim.models.romav2.model import ImageLike, load_image
from flashim.types import Matches, SparseFeatures
from flashim.utils import device_constant

# Upstream ``DaD.Cfg`` as ``LoMa`` builds it: NMS 3, sub-pixel refinement, top-k, borders kept (FlashIM's DAD_SAMPLER
# removes a 4 px frame, DaD's own default).
LOMA_SAMPLER = SamplerConfig(nms_size=3, remove_borders=False, increase_coverage=False, subpixel=True, subpixel_temp=0.5)
# DeDoDe-B with ``descriptor_dim=128`` (LoMa-B128): each refiner emits 128 descriptor + the context channels.
DESCRIPTOR_B128 = blocks.NetConfig(
    "vgg19", {"8": (512, 512, 384), "4": (512, 256, 256), "2": (256, 64, 160), "1": (96, 32, 129)}, 5, 128, "bilinear"
)
# variant -> descriptor network; B / L / G store bit-equal descriptors (one LoMa DeDoDe-G), R its own, B128 its own.
DESCRIPTORS = {"b128": DESCRIPTOR_B128, "b": blocks.DESCRIPTOR_G, "l": blocks.DESCRIPTOR_G, "g": blocks.DESCRIPTOR_G,
               "r": blocks.DESCRIPTOR_G}


def _quantize8(image: Tensor) -> Tensor:
    return (image * 255).round().clamp(0, 255) / 255


def pil_resize(image: Tensor, size: tuple[int, int]) -> Tensor:
    """``PIL.Image.resize(size)`` (BICUBIC) of an 8-bit image, in torch: PIL resamples width first, then height, with
    an antialiased bicubic (a = -0.5, torch's ``antialias=True`` kernel) and rounds to 8 bits after each pass. Measured
    against Pillow 12.3 on two MegaDepth / toronto images, 1024 px and 784 x 784, CPU and CUDA: at most 1/255 apart,
    >= 99.95% of the values equal (one bicubic pass without the rounding: up to 6/255, 86% equal; bilinear: 31/255)."""
    height, width = size
    if image.shape[-1] != width:
        image = _quantize8(F.interpolate(image, size=(image.shape[-2], width), mode="bicubic", antialias=True))
    if image.shape[-2] != height:
        image = _quantize8(F.interpolate(image, size=(height, width), mode="bicubic", antialias=True))
    return image


def detector_size(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Upstream ``DaD.load_image``: ``int(scale * side // 8 * 8)`` with ``scale = long_side / max(H, W)``."""
    scale = long_side / max(height, width)
    return int(scale * height // 8 * 8), int(scale * width // 8 * 8)


class LoMaExtractor(DeDoDe):
    """``extract``: the detector on the image resized to ``resize`` (long side, default upstream's 1024, aspect kept,
    sides floored to 8) and ImageNet-normalized; the descriptor on a separate ``descriptor_size`` resize (784 x 784,
    aspect not kept) of the original, *not* normalized (upstream feeds the [0, 1] image to VGG19 / DINOv2), sampled
    bilinearly at the detector's keypoints (the same [-1, 1] frame of the full image). Both resizes reproduce PIL's
    bicubic (:func:`pil_resize`), as upstream's path API that its benchmarks use. Descriptors are not L2-normalized
    (the matcher's input; mean norms measured on upstream's 0015_A.jpg: 19 for B128, 40 for B / L / G, 32 for R).
    Keypoints are FlashIM's (original pixels, integer = pixel center); upstream's normalized ones are
    ``(k + 0.5) * 2 / (W, H) - 1`` of these.

    ``detect(image)`` is upstream's tensor API: detector and descriptor on the same image, no resizing. The built-in
    ``match_features`` (mutual nearest neighbours of the L2-normalized descriptors) is for inspection only: LoMa's
    descriptors are trained for its matcher (``loma-*`` pipelines)."""

    default_top_k = 2048  # LoMa.Cfg.num_keypoints
    resize = 1024  # DaD.Cfg.resize
    descriptor_size = (784, 784)  # DeDoDeDescriptor.describe_keypoints_from_path
    match_threshold = 0.0

    @classmethod
    def from_variant(cls, variant: str, *, checkpoint: str | None = None,
                     device: torch.device | str | None = None) -> LoMaExtractor:
        if variant not in DESCRIPTORS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(DESCRIPTORS)}")
        state = weights.load_state_dicts(variant, path=checkpoint)
        return cls.from_pretrained(
            blocks.DETECTOR_S, DESCRIPTORS[variant], LOMA_SAMPLER, detector_state=state.detector,
            descriptor_state=state.descriptor, dinov2_state=state.dinov2, device=device,
        )

    def _detect(self, image: Tensor, descriptor_image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.detector(self.normalize(image))[0]  # [1, H, W]
        keypoints, probs = sample_keypoints(logits, top_k or self.default_top_k, self.sampler)
        (height, width), (desc_h, desc_w) = image.shape[-2:], descriptor_image.shape[-2:]
        points = (keypoints + 0.5) * device_constant([desc_w / width, desc_h / height], keypoints) - 0.5
        descriptors = self.descriptor(descriptor_image.to(self.compute_dtype), points=points)
        return keypoints, probs, descriptors

    def detect(self, image: Tensor, top_k: int | None) -> tuple[Tensor, Tensor, Tensor]:
        return self._detect(image, image, top_k)

    def describe_keypoints(self, image: Tensor, keypoints: Tensor) -> Tensor:
        return F.normalize(self.descriptor(image.to(self.compute_dtype), points=keypoints.float()), dim=-1)

    @torch.inference_mode()
    def extract(self, image: ImageLike, *, resize: int | None = None, top_k: int | None = None) -> SparseFeatures:
        image = load_image(image, self.device)
        if image.shape[0] != 1:
            raise ValueError(f"{type(self).__name__}.extract processes one image at a time (batch size 1)")
        height, width = image.shape[-2:]
        new_h, new_w = detector_size(height, width, self.resize if resize is None else resize)
        keypoints, scores, descriptors = self._detect(
            pil_resize(image, (new_h, new_w)), pil_resize(image, self.descriptor_size),
            self.default_top_k if top_k is None else top_k,
        )
        keypoints = (keypoints + 0.5) * device_constant([width / new_w, height / new_h], keypoints) - 0.5
        return SparseFeatures(keypoints[None], scores.float()[None], descriptors[None], (height, width))

    @torch.inference_mode()
    def match_features(self, a: SparseFeatures, b: SparseFeatures, threshold: float | None = None) -> Matches:
        ia, ib, conf = sparse.mutual_nn_match(
            F.normalize(a.descriptors[0].float(), dim=-1), F.normalize(b.descriptors[0].float(), dim=-1),
            self.match_threshold if threshold is None else threshold,
        )
        return Matches(a.keypoints[0][ia], b.keypoints[0][ib], conf, ia, ib)
