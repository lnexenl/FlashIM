"""RoMaV2 dense matcher, re-implemented natively for FlashIM."""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from flashim.models.romav2 import weights
from flashim.models.romav2.config import SETTINGS, RoMaV2Config, Setting
from flashim.models.romav2.dinov3 import DinoV3
from flashim.models.romav2.fine_features import VGG19BN
from flashim.models.romav2.geometry import (
    bhwc_grid_sample,
    bhwc_interpolate,
    normalized_grid,
    precision_matrix,
    to_pixel,
)
from flashim.models.romav2.matcher import Matcher
from flashim.models.romav2.refiner import Refiners
from flashim.types import DenseMatchResult

ImageLike = str | Path | Image.Image | np.ndarray | Tensor


@contextlib.contextmanager
def _highest_matmul_precision():
    # RoMaV2's FP32 similarity/softmax path is validated with full-precision matmuls.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _resize_state(warp: Tensor, confidence: Tensor, size: tuple[int, int], zero_precision: bool):
    warp = bhwc_interpolate(warp, size=size, mode="bilinear", align_corners=False)
    if zero_precision:
        # Precision deltas at stride 4 are absolute: the HR pass must not reuse the LR ones.
        confidence[..., 1:] = 0.0
    confidence = bhwc_interpolate(confidence, size=size, mode="bilinear", align_corners=False)
    return warp, confidence


def load_image(img: ImageLike, device: torch.device) -> Tensor:
    """Path / PIL / HWC numpy / ``[B,3,H,W]`` tensor -> float ``[B,3,H,W]`` in [0, 1] on ``device``."""
    if isinstance(img, (str, Path)):
        pil = Image.open(img)
        if pil.mode == "I;16":
            raise NotImplementedError("16-bit images are not supported")
        img = pil.convert("RGB")
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, np.ndarray):
        if img.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB array, got {img.shape}")
        img = torch.from_numpy(img).permute(2, 0, 1)
    if img.ndim == 3:
        img = img[None]
    if img.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {tuple(img.shape)}")
    img = img.to(device)
    return img.float() / 255.0 if img.dtype == torch.uint8 else img


def prepare_pair(
    img_a: ImageLike, img_b: ImageLike, setting: Setting, device: torch.device
) -> tuple[Tensor | None, ...]:
    """Load a pair and resize (bicubic, antialiased) to the setting's LR and HR resolutions."""
    a, b = load_image(img_a, device), load_image(img_b, device)

    def resize(x: Tensor, h: int, w: int) -> Tensor:
        return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)

    lr = (resize(a, setting.h_lr, setting.w_lr), resize(b, setting.h_lr, setting.w_lr))
    if setting.h_hr is None or setting.w_hr is None:
        return (*lr, None, None)
    return (*lr, resize(a, setting.h_hr, setting.w_hr), resize(b, setting.h_hr, setting.w_hr))


def _kde(x: Tensor, std: float = 0.1) -> Tensor:
    x = x.half()
    return (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp().sum(dim=-1)


class RoMaV2(nn.Module):
    """``f`` (DINOv3) -> ``matcher`` (MV-ViT + DPT) -> ``refiners`` at strides 4/2/1.

    Module paths match the RoMaV2 checkpoint, so ``state_dict`` loads strictly and
    quantization recipes/calibrations address the same names.
    """

    def __init__(self, config: RoMaV2Config | None = None) -> None:
        super().__init__()
        self.config = config or RoMaV2Config()
        self.f = DinoV3()
        self.matcher = Matcher()
        self.refiners = Refiners()
        self.refiner_features = VGG19BN()
        self.apply_setting(self.config.setting)

    @classmethod
    def from_pretrained(
        cls,
        config: RoMaV2Config | None = None,
        *,
        device: torch.device | str | None = None,
        state_dict: dict[str, Tensor] | None = None,
    ) -> RoMaV2:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.device("meta"):
            model = cls(config)
        state = state_dict if state_dict is not None else weights.load_state_dict("cpu")
        # assign=True adopts checkpoint dtypes: DINOv3 in BF16, everything else FP32.
        model.load_state_dict(state, strict=True, assign=True)
        if model.config.truncate_descriptor:
            model.f.truncate()
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.matcher.omega.device

    def apply_setting(self, setting: str | Setting) -> None:
        if isinstance(setting, str):
            if setting not in SETTINGS:
                raise ValueError(f"Unknown setting {setting!r}; choose from {sorted(SETTINGS)}")
            setting = SETTINGS[setting]
        self.setting = setting

    def _fine_features(self, img_a: Tensor, img_b: Tensor) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        if not self.config.batch_refiner_features or img_a.shape != img_b.shape:
            return self.refiner_features(img_a), self.refiner_features(img_b)
        features = self.refiner_features(torch.cat((img_a, img_b)))
        batch = img_a.shape[0]
        return {k: v[:batch] for k, v in features.items()}, {k: v[batch:] for k, v in features.items()}

    @torch.inference_mode()
    def forward(
        self,
        img_a_lr: Tensor,
        img_b_lr: Tensor,
        img_a_hr: Tensor | None = None,
        img_b_hr: Tensor | None = None,
    ) -> dict[str, object]:
        """Images are ``[B, 3, H, W]`` floats in ``[0, 1]`` at the setting's resolutions."""
        bidirectional = self.setting.bidirectional
        preds: dict[str, object] = OrderedDict()
        with _highest_matmul_precision():
            f_a, f_b = self.f(img_a_lr), self.f(img_b_lr)
            matcher_out = self.matcher(f_a, f_b, bidirectional=bidirectional)
            preds["matcher"] = matcher_out
            warp_ab, conf_ab = matcher_out["warp_AB"], matcher_out["confidence_AB"]
            warp_ba, conf_ba = matcher_out["warp_BA"], matcher_out["confidence_BA"]
            stages = [(img_a_lr, img_b_lr)]
            if img_a_hr is not None and img_b_hr is not None:
                stages.append((img_a_hr, img_b_hr))
            for stage, (img_a, img_b) in enumerate(stages):
                _, _, height, width = img_a.shape
                scale_factor = torch.empty(2, device=img_a.device, dtype=img_a.dtype)
                scale_factor[0].fill_(width / self.config.anchor_width)
                scale_factor[1].fill_(height / self.config.anchor_height)
                feats_a, feats_b = self._fine_features(img_a, img_b)
                for stride_key, refiner in self.refiners.items():
                    stride = int(stride_key)
                    size = (height // stride, width // stride)
                    zero_precision = len(stages) > 1 and stride == 4 and stage == 1
                    warp_ab, conf_ab = _resize_state(warp_ab, conf_ab, size, zero_precision)
                    warp_ab, conf_ab = refiner(feats_a[stride], feats_b[stride], warp_ab, conf_ab, scale_factor)
                    preds[f"refiner_{stride}_AB"] = {"warp": warp_ab, "confidence": conf_ab}
                    if bidirectional:
                        warp_ba, conf_ba = _resize_state(warp_ba, conf_ba, size, zero_precision)
                        warp_ba, conf_ba = refiner(feats_b[stride], feats_a[stride], warp_ba, conf_ba, scale_factor)
                        preds[f"refiner_{stride}_BA"] = {"warp": warp_ba, "confidence": conf_ba}
                    else:
                        preds[f"refiner_{stride}_BA"] = None
        preds.update(warp_AB=warp_ab, confidence_AB=conf_ab, warp_BA=warp_ba, confidence_BA=conf_ba)
        return preds

    # -- image-level API -------------------------------------------------------------------

    def load_image(self, img: ImageLike) -> Tensor:
        return load_image(img, self.device)

    def prepare(self, img_a: ImageLike, img_b: ImageLike) -> tuple[Tensor | None, ...]:
        return prepare_pair(img_a, img_b, self.setting, self.device)

    def postprocess(self, preds: dict[str, object]) -> DenseMatchResult:
        def split(conf: Tensor) -> tuple[Tensor, Tensor]:
            overlap = conf[..., :1].sigmoid()
            if self.setting.threshold is not None:
                overlap[overlap > self.setting.threshold] = 1.0
            return overlap, precision_matrix(conf[..., 1:4])

        overlap_ab, precision_ab = split(preds["confidence_AB"])
        result = DenseMatchResult(
            preds["warp_AB"].clone(), preds["confidence_AB"].clone(), overlap_ab, precision_ab
        )
        if preds["warp_BA"] is not None:
            overlap_ba, precision_ba = split(preds["confidence_BA"])
            result.warp_BA = preds["warp_BA"].clone()
            result.confidence_BA = preds["confidence_BA"].clone()
            result.overlap_BA, result.precision_BA = overlap_ba, precision_ba
        return result

    @torch.inference_mode()
    def match(self, img_a: ImageLike, img_b: ImageLike) -> DenseMatchResult:
        return self.postprocess(self(*self.prepare(img_a, img_b)))

    def sample(
        self, result: DenseMatchResult, num: int
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Sample ``num`` KDE-balanced matches ``[N, 4]`` (xA, yA, xB, yB normalized) from batch item 0.

        Returns ``(matches, confidence, precision_A, precision_B)``; precisions are
        ``[N, 2, 2]`` (``None`` when unidirectional).
        """
        warp = result.warp_AB[0]
        h, w, _ = warp.shape
        grid = normalized_grid(1, h, w, warp.device)[0]
        matches = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
        confidence = result.overlap_AB[0].reshape(-1)
        precision = None
        if result.bidirectional:
            warp_ba = result.warp_BA[0]
            prec_ab, prec_ba = result.precision_AB[0], result.precision_BA[0]
            prec_a = bhwc_grid_sample(prec_ba[None].reshape(1, h, w, -1), warp[None], align_corners=False)
            prec_b = bhwc_grid_sample(prec_ab[None].reshape(1, h, w, -1), warp_ba[None], align_corners=False)
            precision = torch.cat(
                (
                    torch.stack((prec_a.reshape(h, w, 2, 2), prec_ab), dim=-3).reshape(-1, 2, 2, 2),
                    torch.stack((prec_ba, prec_b.reshape(h, w, 2, 2)), dim=-3).reshape(-1, 2, 2, 2),
                )
            )
            matches = torch.cat((matches, torch.cat((warp_ba, grid), dim=-1).reshape(-1, 4)))
            confidence = torch.cat((confidence, result.overlap_BA[0].reshape(-1)))

        confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / h).float()
        inds = torch.multinomial(confidence, 4 * num, replacement=False)
        matches, confidence = matches[inds], confidence[inds]
        if precision is not None:
            precision = precision[inds]
        density = _kde(matches)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7  # need ~10 perfect neighbours to be trusted
        balanced = torch.multinomial(p, num_samples=min(num, len(confidence)), replacement=False)
        if precision is None:
            return matches[balanced], confidence[balanced], None, None
        return matches[balanced], confidence[balanced], precision[balanced][:, 0], precision[balanced][:, 1]

    @staticmethod
    def to_pixel_coordinates(matches: Tensor, h_a: int, w_a: int, h_b: int, w_b: int) -> tuple[Tensor, Tensor]:
        return to_pixel(matches[..., :2], h=h_a, w=w_a), to_pixel(matches[..., 2:], h=h_b, w=w_b)
