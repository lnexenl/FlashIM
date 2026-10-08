# Vendored from https://github.com/Parskatt/RoMaV2 (romav2/graph_fusion.py @ 95c9968) as the reference implementation of FlashIM's parity tests.
# Copyright (c) 2025 Johan Edstedt. SPDX-License-Identifier: MIT (see LICENSE in this directory).
# Changes: none besides this header.

"""Inference-only RoPE adaptation for externally loaded DINOv3 attention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import Final
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

from romav2.rope_fusion import _eager_rope_apply_qk, _triton_ready, fused_rope_apply_qk
from romav2.vit.attention import SelfAttention

__all__ = ["optimize_rope_for_inference"]


_ORIGINAL_APPLY_ROPE_ATTR: Final = "_romav2_original_apply_rope"


@dataclass
class _RopeAdapterState:
    compatible_contracts: set[tuple[object, ...]] = field(default_factory=set)
    incompatible_contracts: set[tuple[object, ...]] = field(default_factory=set)


_PATCHED: WeakKeyDictionary[nn.Module, _RopeAdapterState] = WeakKeyDictionary()
_VENDOR_MODULE_PREFIX: Final = "dinov3"
_TEST_MODULE_PREFIXES: Final = ("test_", "tests.")

def _rope_contract_signature(
    q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
) -> tuple[object, ...]:
    """Identify every tensor-layout detail the Triton path relies on."""
    return (
        q.dtype,
        k.dtype,
        sin.dtype,
        cos.dtype,
        q.device,
        k.device,
        sin.device,
        cos.device,
        tuple(q.shape),
        tuple(k.shape),
        tuple(sin.shape),
        tuple(cos.shape),
        q.stride(),
        k.stride(),
        sin.stride(),
        cos.stride(),
    )


def _vendor_outputs_match(
    fused: tuple[Tensor, Tensor], eager: tuple[Tensor, Tensor]
) -> bool:
    """Require bit-exact values before allowing transformer-wide fusion.

    A single BF16 rounding difference in every attention block amplifies into
    visible dense-match drift, even when an elementwise tolerance accepts it.
    """
    return torch.equal(fused[0], eager[0]) and torch.equal(fused[1], eager[1])


def _adapter_state(module: nn.Module) -> _RopeAdapterState | None:
    state = _PATCHED.get(module)
    if state is not None:
        return state
    if not callable(getattr(module, _ORIGINAL_APPLY_ROPE_ATTR, None)):
        return None
    state = _RopeAdapterState()
    _PATCHED[module] = state
    return state


def _original_apply_rope(module: nn.Module) -> Callable[..., tuple[Tensor, Tensor]]:
    original = getattr(module, _ORIGINAL_APPLY_ROPE_ATTR)
    if not callable(original):
        raise TypeError("patched RoPE adapter is missing its original method")
    return original


def apply_rope(
    module: nn.Module,
    q: Tensor,
    k: Tensor,
    rope: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    """Use Triton only after it agrees with this module's eager vendor method."""
    state = _adapter_state(module)
    if state is None:
        sin, cos = rope
        return _eager_rope_apply_qk(q, k, sin, cos)
    if module.training or not (
        isinstance(rope, tuple)
        and len(rope) == 2
        and isinstance(rope[0], Tensor)
        and isinstance(rope[1], Tensor)
    ):
        return _original_apply_rope(module)(q, k, rope)

    sin, cos = rope
    contract = _rope_contract_signature(q, k, sin, cos)
    if contract in state.incompatible_contracts or not _triton_ready(q, k, sin, cos):
        return _original_apply_rope(module)(q, k, rope)
    if contract in state.compatible_contracts:
        return fused_rope_apply_qk(q, k, sin, cos)

    eager = _original_apply_rope(module)(q, k, rope)
    fused = fused_rope_apply_qk(q, k, sin, cos)
    if _vendor_outputs_match(fused, eager):
        state.compatible_contracts.add(contract)
        return fused
    state.incompatible_contracts.add(contract)
    return eager


def _is_vendor_or_test_module(module: nn.Module) -> bool:
    """Limit runtime adaptation to DINOv3 and explicit test doubles."""
    module_name = type(module).__module__
    return module_name == _VENDOR_MODULE_PREFIX or module_name.startswith(
        (f"{_VENDOR_MODULE_PREFIX}.", *_TEST_MODULE_PREFIXES)
    )


def _linear_feature_dimensions(module: object) -> tuple[int, int] | None:
    """Return validated feature dimensions for a linear-like module."""
    if not isinstance(module, nn.Module):
        return None

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if (
        not isinstance(in_features, int)
        or isinstance(in_features, bool)
        or in_features <= 0
        or not isinstance(out_features, int)
        or isinstance(out_features, bool)
        or out_features <= 0
    ):
        return None
    return in_features, out_features


def _has_dinov3_attention_contract(module: nn.Module) -> bool:
    """Check the structural seam shared by DINOv3 self-attention modules."""
    if not _is_vendor_or_test_module(module) or isinstance(module, SelfAttention):
        return False

    apply_rope = getattr(module, "apply_rope", None)
    qkv_dimensions = _linear_feature_dimensions(getattr(module, "qkv", None))
    proj_dimensions = _linear_feature_dimensions(getattr(module, "proj", None))
    num_heads = getattr(module, "num_heads", None)
    if not (
        callable(apply_rope)
        and qkv_dimensions is not None
        and proj_dimensions is not None
        and isinstance(num_heads, int)
        and not isinstance(num_heads, bool)
        and num_heads > 0
    ):
        return False

    channels, qkv_out_features = qkv_dimensions
    proj_in_features, proj_out_features = proj_dimensions
    return (
        channels % num_heads == 0
        and qkv_out_features == 3 * channels
        and proj_in_features == channels
        and proj_out_features == channels
    )


def _is_patched(module: nn.Module) -> bool:
    """Recognize both the weak marker and copies made during model handling."""
    return _PATCHED.get(module) is not None or getattr(
        getattr(module, "apply_rope", None), "__func__", None
    ) is apply_rope


def optimize_rope_for_inference(model: nn.Module) -> int:
    """Patch eligible vendor attention modules with a verified RoPE adapter.

    The first observed tensor contract is compared against the original vendor
    method before the Triton path is cached. Unsupported layouts always retain
    the exact vendor eager method.
    """
    if model.training:
        raise ValueError("optimize_rope_for_inference requires model.eval()")

    patched = 0
    for module in model.modules():
        if module.training or not _has_dinov3_attention_contract(module):
            continue
        if _is_patched(module):
            continue
        original = module.apply_rope
        setattr(module, _ORIGINAL_APPLY_ROPE_ATTR, original)
        _PATCHED[module] = _RopeAdapterState()
        module.apply_rope = MethodType(apply_rope, module)
        patched += 1
    return patched
