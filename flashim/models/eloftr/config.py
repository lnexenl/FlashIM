"""EfficientLoFTR configuration: ``full`` (dual-softmax coarse matching) and ``opt`` (skip-softmax, faster)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    """Coarse-matching mode (upstream ``full_default_cfg`` vs ``opt_default_cfg``)."""

    skip_softmax: bool
    fp16matmul: bool
    threshold: float


SETTINGS: dict[str, Setting] = {
    "full": Setting(skip_softmax=False, fp16matmul=False, threshold=0.2),
    "opt": Setting(skip_softmax=True, fp16matmul=True, threshold=25.0),
}


@dataclass(frozen=True)
class ELoFTRConfig:
    """Architecture and matching parameters for :class:`~flashim.models.eloftr.model.ELoFTR`."""

    variant: str = "full"
    npe: tuple[int, int, int, int] = (832, 832, 832, 832)
    threshold: float | None = None
    border_rm: int = 2
    skip_softmax: bool | None = None
    fp16matmul: bool | None = None
    fine_window: int = 8
    local_regress_temperature: float = 10.0
    local_regress_slicedim: int = 8
    block_dims: tuple[int, int, int] = (64, 128, 256)
    d_model: int = 256
    nhead: int = 8
    agg_size0: int = 4
    agg_size1: int = 4
    layer_names: tuple[str, ...] = ("self", "cross") * 4
    rope: bool = True

    def __post_init__(self) -> None:
        if self.variant not in SETTINGS:
            raise ValueError(f"Unknown ELoFTR variant {self.variant!r}; choose from {sorted(SETTINGS)}")
        if len(self.npe) != 4:
            raise ValueError(f"npe must have 4 elements (train_h, train_w, test_h, test_w), got {self.npe!r}")

    @property
    def setting(self) -> Setting:
        return SETTINGS[self.variant]

    @property
    def resolved_threshold(self) -> float:
        return self.setting.threshold if self.threshold is None else self.threshold

    @property
    def resolved_skip_softmax(self) -> bool:
        return self.setting.skip_softmax if self.skip_softmax is None else self.skip_softmax

    @property
    def resolved_fp16matmul(self) -> bool:
        return self.setting.fp16matmul if self.fp16matmul is None else self.fp16matmul
