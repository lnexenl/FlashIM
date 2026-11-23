// Register-resident persistent log-space Sinkhorn optimal transport (SuperGlue, Sarlin et al. 2020).
//
// Each warp owns one row (scores[gw, 0..N-1]) and one column (scores[0..M-1, gw]) in registers (32 floats/thread for
// MAX_C = 1024), so all 100 iterations run with zero DRAM/L2/L1 reads of the similarity matrix and zero intra-CTA
// barriers per row (5-step warp-shuffle reductions). The tiny dual vectors u/v (<= 4 KB) are broadcast once per CTA
// per half-iteration into shared memory (zero-padded beyond M/N for branchless inner loops) and synchronized across
// resident CTAs with a GPU-scope monotonic barrier.
//
// Inner-loop exponentials use PTX ``ex2.approx.ftz.f32`` (single SFU MUFU.EX2 instruction on x <= 0, matching Triton's
// tl.exp2) while ``log2f`` runs once per row in IEEE FP32. Measured on RTX 5070 Ti (sm_120): replaces 13-instruction
// software exp2f + 17.4M branches seen in ncu (cuda_v1: 0.759 ms at 1024x1024).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_CTA = 8;
constexpr int THREADS_PER_CTA = WARP_SIZE * WARPS_PER_CTA;  // 256

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float load_cg(const float* ptr) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ void grid_barrier(int* bar_ptr, int target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("red.global.gpu.release.add.s32 [%0], 1;" :: "l"(bar_ptr) : "memory");
        int cur;
        do {
            asm volatile("ld.global.gpu.acquire.s32 %0, [%1];" : "=r"(cur) : "l"(bar_ptr) : "memory");
        } while (cur < target);
    }
    __syncthreads();
}

template <int MAX_C>
__global__ __launch_bounds__(THREADS_PER_CTA, 2) void sinkhorn_reg_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ alpha_ptr,
    float* __restrict__ u_buf,
    float* __restrict__ v_buf,
    int* __restrict__ bar_buf,
    float* __restrict__ out,
    int M,
    int N,
    int iters,
    float norm2,
    float norm_e,
    float log_mu_dust2,
    float log_nu_dust2,
    int num_ctas
) {
    constexpr int EPT = MAX_C / WARP_SIZE;  // elements per thread (32 for MAX_C=1024)
    __shared__ float sh_vec[MAX_C + 1];

    const int bid = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gw = bid * WARPS_PER_CTA + warp;

    const float* s_base = scores + static_cast<int64_t>(b) * M * N;
    float* u = u_buf + static_cast<int64_t>(b) * (M + 1);
    float* v = v_buf + static_cast<int64_t>(b) * (N + 1);
    float* o_base = out + static_cast<int64_t>(b) * (M + 1) * (N + 1);
    int* bar = bar_buf + b;

    const float alpha_e = alpha_ptr[0];
    const float alpha2 = alpha_e * LOG2E;

    // Load this warp's row and column into registers once (scaled by LOG2E; -INF out of bounds).
    float r_row[EPT];
    float r_col[EPT];
    const bool has_row = gw < M;
    const bool has_col = gw < N;

    if (has_row) {
        const float* row_ptr = s_base + static_cast<int64_t>(gw) * N;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int c = k * WARP_SIZE + lane;
            r_row[k] = (c < N) ? (row_ptr[c] * LOG2E) : -INFINITY;
        }
    }
    if (has_col && iters > 0) {
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int r = k * WARP_SIZE + lane;
            r_col[k] = (r < M) ? (s_base[static_cast<int64_t>(r) * N + gw] * LOG2E) : -INFINITY;
        }
    }

    // Initialize v[0..N] = 0 (and u[0..M] = 0 if iters == 0).
    const int global_tid = bid * THREADS_PER_CTA + tid;
    const int global_threads = num_ctas * THREADS_PER_CTA;
    for (int j = global_tid; j <= N; j += global_threads) {
        v[j] = 0.0f;
    }
    if (iters == 0) {
        for (int i = global_tid; i <= M; i += global_threads) {
            u[i] = 0.0f;
        }
    }

    int phase = 1;
    grid_barrier(bar, phase * num_ctas);

    // Use a dedicated idle warp for dustbin row/col when available; else warp 0 of the last CTA.
    const int max_mn = (M > N) ? M : N;
    const int dust_gw = (num_ctas * WARPS_PER_CTA > max_mn) ? max_mn : (num_ctas - 1) * WARPS_PER_CTA;
    const bool is_dust_warp = (gw == dust_gw);

    for (int it = 0; it < iters; ++it) {
        // --- Row step: broadcast v[0..N] into sh_vec (zero-padded to MAX_C for branchless inner loops) ---
        for (int j = tid; j < MAX_C; j += THREADS_PER_CTA) {
            sh_vec[j] = (j < N) ? load_cg(v + j) : 0.0f;
        }
        if (tid == 0) {
            sh_vec[MAX_C] = load_cg(v + N);
        }
        __syncthreads();

        const float v_dust = sh_vec[MAX_C];
        const float dust_term = alpha2 + v_dust;
        if (has_row) {
            float peak = dust_term;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                peak = fmaxf(peak, r_row[k] + sh_vec[k * WARP_SIZE + lane]);
            }
            peak = warp_max(peak);
            float acc = (lane == 0) ? fast_exp2f(dust_term - peak) : 0.0f;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                acc += fast_exp2f((r_row[k] + sh_vec[k * WARP_SIZE + lane]) - peak);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                u[gw] = norm2 - (peak + log2f(acc));
            }
        }
        if (is_dust_warp) {
            float peak = v_dust;
            for (int c = lane; c < N; c += WARP_SIZE) {
                peak = fmaxf(peak, sh_vec[c]);
            }
            peak = warp_max(peak);
            float acc = (lane == 0) ? fast_exp2f(v_dust - peak) : 0.0f;
            for (int c = lane; c < N; c += WARP_SIZE) {
                acc += fast_exp2f(sh_vec[c] - peak);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                u[M] = log_mu_dust2 - (alpha2 + peak + log2f(acc));
            }
        }

        ++phase;
        grid_barrier(bar, phase * num_ctas);

        // --- Column step: broadcast u[0..M] into sh_vec (zero-padded to MAX_C) ---
        for (int i = tid; i < MAX_C; i += THREADS_PER_CTA) {
            sh_vec[i] = (i < M) ? load_cg(u + i) : 0.0f;
        }
        if (tid == 0) {
            sh_vec[MAX_C] = load_cg(u + M);
        }
        __syncthreads();

        const float u_dust = sh_vec[MAX_C];
        const float dust_term_u = alpha2 + u_dust;
        if (has_col) {
            float peak = dust_term_u;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                peak = fmaxf(peak, r_col[k] + sh_vec[k * WARP_SIZE + lane]);
            }
            peak = warp_max(peak);
            float acc = (lane == 0) ? fast_exp2f(dust_term_u - peak) : 0.0f;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                acc += fast_exp2f((r_col[k] + sh_vec[k * WARP_SIZE + lane]) - peak);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                v[gw] = norm2 - (peak + log2f(acc));
            }
        }
        if (is_dust_warp) {
            float peak = u_dust;
            for (int r = lane; r < M; r += WARP_SIZE) {
                peak = fmaxf(peak, sh_vec[r]);
            }
            peak = warp_max(peak);
            float acc = (lane == 0) ? fast_exp2f(u_dust - peak) : 0.0f;
            for (int r = lane; r < M; r += WARP_SIZE) {
                acc += fast_exp2f(sh_vec[r] - peak);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                v[N] = log_nu_dust2 - (alpha2 + peak + log2f(acc));
            }
        }

        ++phase;
        grid_barrier(bar, phase * num_ctas);
    }

    // --- Epilogue: write out[M + 1, N + 1] ---
    for (int j = tid; j <= N; j += THREADS_PER_CTA) {
        sh_vec[j] = load_cg(v + j) * LN2;
    }
    __syncthreads();

    const int out_stride = N + 1;
    if (has_row) {
        const float u_val = load_cg(u + gw) * LN2 - norm_e;
        float* row_out = o_base + static_cast<int64_t>(gw) * out_stride;
        const float* row_in = s_base + static_cast<int64_t>(gw) * N;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int c = k * WARP_SIZE + lane;
            if (c < N) {
                row_out[c] = row_in[c] + u_val + sh_vec[c];
            }
        }
        if (lane == 0) {
            row_out[N] = alpha_e + u_val + sh_vec[N];
        }
    }
    if (is_dust_warp) {
        const float u_dust_e = load_cg(u + M) * LN2 - norm_e;
        float* dust_out = o_base + static_cast<int64_t>(M) * out_stride;
        for (int c = lane; c <= N; c += WARP_SIZE) {
            dust_out[c] = alpha_e + u_dust_e + sh_vec[c];
        }
    }
}

}  // namespace

extern "C" int sinkhorn_max_resident_ctas(int max_c, int sm_count) {
    int blocks_per_sm = 0;
    cudaError_t err;
    if (max_c <= 256) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sinkhorn_reg_kernel<256>, THREADS_PER_CTA, 0);
    } else if (max_c <= 512) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sinkhorn_reg_kernel<512>, THREADS_PER_CTA, 0);
    } else if (max_c <= 1024) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sinkhorn_reg_kernel<1024>, THREADS_PER_CTA, 0);
    } else {
        return 0;
    }
    if (err != cudaSuccess) {
        return 0;
    }
    return blocks_per_sm * sm_count;
}

extern "C" int launch_sinkhorn(
    const void* scores,
    const void* alpha_ptr,
    void* u_buf,
    void* v_buf,
    void* bar_buf,
    void* out,
    int B,
    int M,
    int N,
    int iters,
    int num_ctas,
    int max_c,
    void* stream_ptr
) {
    const float norm_e = -logf(static_cast<float>(M + N));
    const float norm2 = -log2f(static_cast<float>(M + N));
    const float log_mu_dust2 = log2f(static_cast<float>(N)) + norm2;
    const float log_nu_dust2 = log2f(static_cast<float>(M)) + norm2;

    dim3 grid(num_ctas, B);
    dim3 block(THREADS_PER_CTA);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (max_c <= 256) {
        sinkhorn_reg_kernel<256><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(alpha_ptr),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<float*>(out), M, N, iters, norm2, norm_e, log_mu_dust2, log_nu_dust2, num_ctas);
    } else if (max_c <= 512) {
        sinkhorn_reg_kernel<512><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(alpha_ptr),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<float*>(out), M, N, iters, norm2, norm_e, log_mu_dust2, log_nu_dust2, num_ctas);
    } else if (max_c <= 1024) {
        sinkhorn_reg_kernel<1024><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(alpha_ptr),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<float*>(out), M, N, iters, norm2, norm_e, log_mu_dust2, log_nu_dust2, num_ctas);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}
