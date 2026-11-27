"""Capture successful JIT launches, including the configuration forwarded by an autotuner."""

import dataclasses
import json
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch
import triton.language as tl
from triton.runtime import JITFunction

from flashim.platforms import Feature, Platform, Vendor
from flashim.platforms.nvidia import _has_mxfp8_api, isa_features

ARCHS = (80, 86, 87, 89, 90, 100, 110, 120)


def simulated_platform(arch):
    capability = divmod(arch, 10)
    features = {Feature.TRITON, *isa_features(capability)}
    if arch >= 89 and hasattr(torch, "float8_e4m3fn"):
        features.add(Feature.FP8_E4M3)
        if hasattr(torch, "_scaled_mm"):
            features.add(Feature.FP8_ROWWISE_SCALED_MM)
    if arch >= 100 and _has_mxfp8_api():
        features.add(Feature.MXFP8_SCALED_MM)
    return Platform(
        "cuda", capability=capability, features=frozenset(features), vendor=Vendor.NVIDIA, arch=f"sm_{arch}"
    )


def call_site():
    from flashim.ops import Op

    frame = sys._getframe(1)
    while frame:
        if frame.f_code is Op._run.__code__:
            call = frame.f_locals
            if call["fn"].__module__.startswith("flashim.backends.triton."):
                return call
        frame = frame.f_back
    return None


def applicable_archs(call):
    from flashim.platforms.detect import _PLATFORMS

    op = call["self"]
    impl = op._find(call["name"])
    device = torch.cuda.current_device()
    allowed = []
    for arch in ARCHS:
        platform = dataclasses.replace(simulated_platform(arch), device_index=device)
        devices = {("cuda", device): platform, ("cuda", None): platform}
        with (
            patch.dict(_PLATFORMS, devices),
            patch.object(torch.cuda, "get_device_capability", return_value=platform.capability),
        ):
            if call["pinned"]:
                selected = impl.rejection(platform, call["args"], call["kwargs"]) is None
            else:
                selected = op._choose(call["args"], call["kwargs"], strict=False) is impl
            if selected:
                allowed.append(arch)
    return allowed


def install(path: str) -> None:
    run = JITFunction.run
    seen = defaultdict(set)

    @wraps(run)
    def record(fn, *args, **kwargs):
        compiled = run(fn, *args, **kwargs)
        if not fn.module.startswith("flashim.") or kwargs.get("warmup") or compiled is None:
            return compiled
        call = call_site()
        route = [call["self"].name, call["name"]] if call else []
        key = (compiled.hash, *route, bool(call and call["pinned"]))
        previous = seen.get(key)
        if previous is not None and len(previous) == len(ARCHS):
            return compiled
        architectures = applicable_archs(call) if call else ARCHS
        if previous is not None and previous.issuperset(architectures):
            return compiled
        source = compiled.src
        entry = {
            "kernel": f"{fn.module}:{fn.fn.__qualname__}",
            "signature": source.signature,
            "constants": {
                fn.arg_names[idx]: str(val) if isinstance(val, tl.dtype) else val
                for (idx,), val in source.constants.items()
            },
            "attrs": {fn.arg_names[idx]: val for (idx,), val in source.attrs.items()},
            "options": {
                key: getattr(compiled.metadata, key)
                for key in ("num_warps", "num_stages", "enable_fp_fusion", "maxnreg")
            },
            "route": route,
            "architectures": architectures,
        }
        if compiled.metadata.num_ctas != 1:
            entry["options"]["num_ctas"] = compiled.metadata.num_ctas
        with Path(path).open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        seen[key].update(architectures)
        return compiled

    JITFunction.run = record
