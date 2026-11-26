"""Replay recorded FlashIM launch specializations against an explicit GPU target (no driver)."""

import argparse
import inspect
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from pkgutil import resolve_name

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton.compiler import ASTSource
from triton.runtime import JITFunction

from flashim import ops
from flashim.backends import LOADED
from flashim.backends.triton import correlation, quantize
from flashim.backends.triton._record import ARCHS, simulated_platform
from flashim.platforms.nvidia import _ptxas_assembles

MANIFEST = Path(__file__).with_name("triton.jsonl")


@cache
def kernel(name):
    fn = resolve_name(name)
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    return fn


def direct_route(name, platform):
    """Standalone tests bypass dispatch. Use the corresponding public op's architecture restrictions."""
    module, symbol = name.split(":")
    if module == correlation.__name__ and symbol in {
        "_col_lse_row_partials_kernel",
        "_dual_argmax_kernel",
        "_row_argmax_reduce_kernel",
        "_row_lse_reduce_kernel",
    }:
        return (
            ("dual_softmax_mutual", "triton")
            if platform.capability in correlation.TWO_PASS_CAPABILITIES
            else None
        )
    return {
        "flashim.backends.triton._libdevice:_gelu_probe_kernel": ("int8_linear", "triton"),
        "flashim.backends.triton.attention:_deformable_attention_kernel": ("deformable_attention", "triton"),
    }[name]


def eligible(entry, platform):
    arch = int(platform.arch.removeprefix("sm_"))
    for route, arches in entry["routes"]:
        if arch not in arches:
            continue
        route = route or direct_route(entry["kernel"], platform)
        if route:
            op, name = route
            if any(impl.name == name for impl in ops.OPS[op].candidates(platform)):
                return True
    return False


def compile_entry(entry, arch):
    fn = kernel(entry["kernel"])
    signature = {name: entry["signature"][name] for name in fn.arg_names}
    attrs = {
        (fn.arg_names.index(name),): [tuple(attr) for attr in values]
        for name, values in entry["attrs"].items()
    }
    constants = {
        name: tl.dtype(value) if tl.dtype.is_dtype(value) else value
        for name, value in entry["constants"].items()
    }
    source = ASTSource(fn, signature, constants, attrs)
    result = triton.compile(source, target=GPUTarget("cuda", arch, 32), options=entry["options"])
    assert result.asm["cubin"], (entry["kernel"], arch)


def configure_target(arch, kernel_names):
    """Resolve JIT dependency hashes in the same order regardless of worker scheduling."""
    quantize._E4M3_INLINE_ASM = tl.constexpr(arch < 90)
    for name in kernel_names:
        _ = kernel(name).cache_key


def checked_compile(entry, arch):
    try:
        compile_entry(entry, arch)
    except Exception as error:
        raise RuntimeError(f"sm_{arch}: {entry}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--arch", type=int, choices=ARCHS, nargs="+", default=ARCHS)
    parser.add_argument("--workers", type=int, default=1, help="parallel compiler processes per target")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(args.arch) > 1:
        for arch in args.arch:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--arch",
                    str(arch),
                    "--manifest",
                    args.manifest,
                    "--workers",
                    str(args.workers),
                ],
                check=True,
            )
        return
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if "triton" not in LOADED:
        raise RuntimeError("FlashIM's Triton backend failed to register")
    for arch in args.arch:
        platform = simulated_platform(arch)
        if arch == 110:
            tool = get_ptxas(arch) if inspect.signature(get_ptxas).parameters else get_ptxas()
            if not _ptxas_assembles(tool.path, "sm_110a"):
                print("sm_110: installed Triton's assembler does not support this architecture", flush=True)
                continue
        selected = [entry for entry in entries if eligible(entry, platform)]
        if not selected:
            raise RuntimeError(f"sm_{arch}: no eligible kernel specializations")
        compile_one = partial(checked_compile, arch=arch)
        kernel_names = sorted({entry["kernel"] for entry in selected})
        if args.workers == 1:
            configure_target(arch, kernel_names)
            for entry in selected:
                compile_one(entry)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=configure_target,
                initargs=(arch, kernel_names),
            ) as pool:
                list(pool.map(compile_one, selected, chunksize=4))
        print(f"sm_{arch}: compiled {len(selected)}/{len(entries)} specializations", flush=True)


if __name__ == "__main__":
    main()
