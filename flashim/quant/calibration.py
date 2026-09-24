"""Static activation calibration: collection, storage and import.

A :class:`CalibrationTable` holds, per linear module, the E8M0 block-32 exponents of
its input (and output) activations. Tables are stored as safetensors with JSON
metadata; RoMaV2's ``romav2.full-fp8-calibration`` ``.pt`` files can be imported.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from flashim.quant.formats.mx import BLOCK, padded_channels, scale_exponent_from_amax

FORMAT = "flashim.calibration.mx-block32"
VERSION = 1


@dataclass(frozen=True)
class ModuleScales:
    input_exponent: Tensor  # int8 [padded_in / 32]
    output_exponent: Tensor | None = None  # int8 [padded_out / 32]


@dataclass
class CalibrationTable:
    linear: dict[str, ModuleScales]
    metadata: dict[str, str] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.linear

    def __getitem__(self, name: str) -> ModuleScales:
        return self.linear[name]

    def save(self, path: str | Path) -> None:
        tensors: dict[str, Tensor] = {}
        for name, scales in self.linear.items():
            tensors[f"{name}::input"] = scales.input_exponent.cpu().contiguous()
            if scales.output_exponent is not None:
                tensors[f"{name}::output"] = scales.output_exponent.cpu().contiguous()
        meta = {"format": FORMAT, "version": str(VERSION), **self.metadata}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata=meta)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationTable:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            meta = dict(handle.metadata() or {})
        if meta.get("format") != FORMAT or meta.get("version") != str(VERSION):
            raise ValueError(f"Unsupported calibration file {path}: {meta.get('format')} v{meta.get('version')}")
        tensors = load_file(str(path))
        names = sorted({key.rsplit("::", 1)[0] for key in tensors})
        linear = {
            name: ModuleScales(tensors[f"{name}::input"].to(torch.int8), tensors.get(f"{name}::output"))
            for name in names
        }
        meta.pop("format"), meta.pop("version")
        return cls(linear, meta)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_romav2_calibration(path: str | Path) -> CalibrationTable:
    """Convert a RoMaV2 ``romav2.full-fp8-calibration`` v1 ``.pt`` (linear entries only)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "romav2.full-fp8-calibration" or payload.get("version") != 1:
        raise ValueError(f"Not a RoMaV2 full-FP8 calibration file: {path}")
    linear = {
        name: ModuleScales(
            value["input_scale_exponent"].to(torch.int8), value["output_scale_exponent"].to(torch.int8)
        )
        for name, value in payload["linear"].items()
    }
    meta = {
        "source": "romav2.full-fp8-calibration",
        "source_file": Path(path).name,
        "source_sha256": file_sha256(path),
        "observer": "amax",
    }
    return CalibrationTable(linear, meta)


class Calibrator:
    """Record block-32 amax of every targeted linear's input and output.

    Use as a context manager around representative forward passes, then
    :meth:`table` returns exponents ``ceil(log2(amax / 448))`` per block.
    """

    def __init__(self, model: nn.Module, names: Iterable[str]) -> None:
        modules = dict(model.named_modules())
        self._amax_in: dict[str, Tensor] = {}
        self._amax_out: dict[str, Tensor] = {}
        self._targets = {}
        for name in names:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{name!r} is not an nn.Linear")
            self._targets[name] = module
        self._handles: list = []

    @staticmethod
    def _block_amax(value: Tensor) -> Tensor:
        channels = value.shape[-1]
        padded = padded_channels(channels)
        flat = value.detach().float().reshape(-1, channels).abs()
        amax = torch.zeros(padded, device=flat.device)
        amax[:channels] = flat.amax(dim=0)
        return amax.reshape(padded // BLOCK, BLOCK).amax(dim=1)

    def _merge(self, store: dict[str, Tensor], name: str, value: Tensor) -> None:
        amax = self._block_amax(value)
        store[name] = amax if name not in store else torch.maximum(store[name], amax)

    def __enter__(self) -> Calibrator:
        for name, module in self._targets.items():

            def hook(mod, args, output, name=name):
                self._merge(self._amax_in, name, args[0])
                self._merge(self._amax_out, name, output)

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def table(self, metadata: dict[str, str] | None = None) -> CalibrationTable:
        missing = sorted(set(self._targets) - set(self._amax_in))
        if missing:
            raise RuntimeError(f"No activations recorded for: {missing}")
        linear = {
            name: ModuleScales(
                scale_exponent_from_amax(self._amax_in[name]).cpu(),
                scale_exponent_from_amax(self._amax_out[name]).cpu(),
            )
            for name in self._targets
        }
        return CalibrationTable(linear, {"observer": "amax", **(metadata or {})})
