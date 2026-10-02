"""Block-scaled MXFP8 GEMM kernel builder (imported on first call by ``gemm.py``).

cuBLAS's block-scaled MXFP8 GEMM (``F.scaled_mm``) needs the padded ``N`` and ``K`` both divisible by 128 (a
cutlass tiling choice built for large training-scale GEMMs, not a hardware limit: this GPU has no
``Feature.TCGEN05``, so cuBLAS's own kernel is built from the same ``mma.sync`` tensor-core instruction this one
uses). Below that, ``ops.mxfp8_linear`` falls back to a Triton GEMM that dequantizes the whole activation and
weight once and runs one dense matmul — see ``backends/triton/gemm.py``.

This kernel loops over ``K`` in ``chunk`` (a multiple of 32, chosen per shape by :func:`pick_chunk` to divide it
evenly) rather than the native 32-wide scale-block granularity directly: each chunk is copied into shared memory
*once*, then every 32-wide sub-block within it runs one unscaled ``T.gemm`` (real E4M3 tensor cores, ``mma.sync``)
whose result is multiplied by ``(activation_scale[block] * weight_scale[n, block])`` and added into the running FP32
total. A first version that copied and rescaled at the native 32-wide granularity directly (one global-memory round
trip per scale block) measured 3-16x *slower* than the Triton fallback at every shape tried — the per-32-element
copy+barrier overhead dominated. Batching several blocks per shared-memory load removes most of that overhead
without needing the *whole* of K resident at once (this GPU's opt-in shared memory per block is only ~99KB —
`torch.cuda.get_device_properties(0).shared_memory_per_block_optin` — far less than datacenter Blackwell, so ``K``
can't just be loaded whole for every shape).

Net effect, measured against the Triton fallback (``M`` in the 20k-140k range, matching real `extract()` calls):
roughly break-even (0.94x-1.01x) for ``K`` up to ~1600, but a clear and growing win from ``K`` ~2000 up (1.1x-2.5x).
Real quantized layers with ``K >= 2000`` exist only in DISK (its wide `path_up` refiner convs, K up to 3200) and
LoFTR (most of its ResNet backbone, K up to 2304); every other model's quantized conv/linear layers have K <= 576,
squarely in the break-even-or-slightly-behind zone. ``gemm.py`` therefore only routes to this backend above that
threshold, leaving the small-K layers on the Triton path that already handles them at least as well.
"""

import functools

import tilelang
import tilelang.language as T

BLOCK_M = 64
_CHUNK_CANDIDATES = (256, 224, 192, 160, 128, 96, 64, 32)
# This GPU's opt-in shared memory per block is ~99KB (`torch.cuda.get_device_properties(0).shared_memory_per_block_optin`
# measured 101376 on the RTX 5070 Ti this was built against), far less than datacenter Blackwell's ~227KB, and the
# double-buffered ``T.Pipelined(num_stages=2)`` needs *two* resident copies of the a/b tiles. Stay well clear of that
# limit (measured failing right around it, not gracefully) rather than compute it exactly per device.
_SHARED_BUDGET = 90_000


def pick_chunk(k: int, block_m: int, block_n: int) -> int:
    """Largest candidate chunk size (a multiple of 32, dividing ``k`` evenly) whose double-buffered a/b tiles fit
    the shared-memory budget; always finds one (32 divides any ``k`` here — the MXFP8 scale-block size — and
    ``2 * (block_m + block_n) * 32`` is small)."""
    return next(c for c in _CHUNK_CANDIDATES if k % c == 0 and 2 * (block_m + block_n) * c <= _SHARED_BUDGET)


@tilelang.jit
def _build_mxfp8_gemm(m_padded: int, n: int, k: int, block_n: int, k_chunk: int):
    kb = k // 32
    chunks = k // k_chunk
    sub = k_chunk // 32

    @T.prim_func
    def main(
        a: T.Tensor((m_padded, k), T.float8_e4m3fn),
        b: T.Tensor((n, k), T.float8_e4m3fn),
        a_scale: T.Tensor((kb,), T.float32),
        b_scale: T.Tensor((n, kb), T.float32),
        out: T.Tensor((m_padded, n), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m_padded, BLOCK_M), threads=128) as (bx, by):
            a_s = T.alloc_shared((BLOCK_M, k_chunk), T.float8_e4m3fn)
            b_s = T.alloc_shared((block_n, k_chunk), T.float8_e4m3fn)
            a_scale_s = T.alloc_shared((sub,), T.float32)
            b_scale_s = T.alloc_shared((block_n, sub), T.float32)
            partial = T.alloc_fragment((BLOCK_M, block_n), T.float32)
            acc = T.alloc_fragment((BLOCK_M, block_n), T.float32)
            T.clear(acc)
            for c in T.Pipelined(chunks, num_stages=2):
                T.copy(a[by * BLOCK_M : (by + 1) * BLOCK_M, c * k_chunk : (c + 1) * k_chunk], a_s)
                T.copy(b[bx * block_n : (bx + 1) * block_n, c * k_chunk : (c + 1) * k_chunk], b_s)
                T.copy(a_scale[c * sub : (c + 1) * sub], a_scale_s)
                T.copy(b_scale[bx * block_n : (bx + 1) * block_n, c * sub : (c + 1) * sub], b_scale_s)
                for s in T.unroll(sub):
                    T.clear(partial)
                    T.gemm(a_s[:, s * 32 : (s + 1) * 32], b_s[:, s * 32 : (s + 1) * 32], partial,
                          transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(BLOCK_M, block_n):
                        acc[i, j] += partial[i, j] * a_scale_s[s] * b_scale_s[j, s]
            T.copy(acc, out[by * BLOCK_M : (by + 1) * BLOCK_M, bx * block_n : (bx + 1) * block_n])

    return main


@functools.cache
def mxfp8_gemm_kernel(m_padded: int, n: int, k: int):
    block_n = min(n, 128)
    return _build_mxfp8_gemm(m_padded, n, k, block_n, pick_chunk(k, BLOCK_M, block_n))
