// Register-resident persistent log-space Sinkhorn optimal transport (SuperGlue / OmniGlue).
//
// Two ops share this library:
// 1. ``launch_sinkhorn`` (ops.log_optimal_transport): register-resident persistent Sinkhorn returning the full
//    ``[B, M + 1, N + 1]`` log-assignment matrix for ``M, N <= 1024``.
// 2. ``launch_sinkhorn_mutual`` (ops.sinkhorn_mutual): same fast register-resident Sinkhorn core fused with
//    row/column argmax + mutual check for ``M, N <= 1024`` (zero ``scores_T`` allocation), plus an L2-streaming
//    cooperative fallback for larger ``M, N > 1024``.
//
// Each warp owns one row (scores[gw, 0..N-1]) and one column (scores[0..M-1, gw]) in registers (32 floats/thread for
// MAX_C = 1024), so all 100 iterations run with zero DRAM/L2/L1 reads of the similarity matrix and zero intra-CTA
// barriers per row (5-step warp-shuffle reductions). The tiny dual vectors u/v (<= 4 KB) are broadcast once per CTA
// per half-iteration into shared memory (zero-padded beyond M/N for branchless inner loops) and synchronized across
// resident CTAs with a GPU-scope monotonic barrier.
//
// Inner-loop exponentials use PTX ``ex2.approx.ftz.f32`` (single SFU MUFU.EX2 instruction on x <= 0, matching Triton's
// tl.exp2) while ``log2f`` runs once per row in IEEE FP32. Measured on RTX 5070 Ti (sm_120): 0.533 ms at 1024x1024
// and 0.522 ms at 816x1024 (25-58% faster than software exp2f + cooperative_groups + transpose).

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_CTA = 8;
constexpr int THREADS_PER_CTA = WARP_SIZE * WARPS_PER_CTA;  // 256
constexpr int STREAM_SMEM_VEC_MAX = 4096;

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

__device__ __forceinline__ void warp_argmax(float& val, int& idx) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, val, offset);
        int oi = __shfl_down_sync(0xffffffffu, idx, offset);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
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

// -- 1) Register-resident log_optimal_transport ([B, M+1, N+1]) ----------------------------------------------

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

// -- 2) Register-resident sinkhorn_mutual (same fast core + fused mutual argmax) -----------------------------

template <int MAX_C>
__global__ __launch_bounds__(THREADS_PER_CTA, 2) void sinkhorn_mutual_reg_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ alpha_ptr,
    float* __restrict__ u,
    float* __restrict__ v,
    int* __restrict__ bar,
    int* __restrict__ best_row,
    float* __restrict__ prob_out,
    int64_t* __restrict__ index_out,
    bool* __restrict__ mutual_out,
    int M,
    int N,
    int iters,
    float norm2,
    float log_mu_dust2,
    float log_nu_dust2,
    int num_ctas
) {
    constexpr int EPT = MAX_C / WARP_SIZE;
    __shared__ float sh_vec[MAX_C + 1];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gw = bid * WARPS_PER_CTA + warp;

    const float alpha2 = alpha_ptr[0] * LOG2E;

    float r_row[EPT];
    float r_col[EPT];
    const bool has_row = gw < M;
    const bool has_col = gw < N;

    if (has_row) {
        const float* row_ptr = scores + static_cast<int64_t>(gw) * N;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int c = k * WARP_SIZE + lane;
            r_row[k] = (c < N) ? (row_ptr[c] * LOG2E) : -INFINITY;
        }
    }
    if (has_col) {
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int r = k * WARP_SIZE + lane;
            r_col[k] = (r < M) ? (scores[static_cast<int64_t>(r) * N + gw] * LOG2E) : -INFINITY;
        }
    }

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

    const int max_mn = (M > N) ? M : N;
    const int dust_gw = (num_ctas * WARPS_PER_CTA > max_mn) ? max_mn : (num_ctas - 1) * WARPS_PER_CTA;
    const bool is_dust_warp = (gw == dust_gw);

    for (int it = 0; it < iters; ++it) {
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

    // Row argmax from register-resident r_row + smem v.
    for (int j = tid; j < N; j += THREADS_PER_CTA) {
        sh_vec[j] = load_cg(v + j);
    }
    __syncthreads();
    if (has_row) {
        float bval = -INFINITY;
        int bidx = 0;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int c = k * WARP_SIZE + lane;
            if (c < N) {
                const float val = r_row[k] + sh_vec[c];
                if (val > bval || (val == bval && c < bidx)) {
                    bval = val;
                    bidx = c;
                }
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            prob_out[gw] = exp2f(bval + load_cg(u + gw) - norm2);
            index_out[gw] = static_cast<int64_t>(bidx);
        }
    }

    // Col argmax from register-resident r_col + smem u.
    __syncthreads();
    for (int i = tid; i < M; i += THREADS_PER_CTA) {
        sh_vec[i] = load_cg(u + i);
    }
    __syncthreads();
    if (has_col) {
        float bval = -INFINITY;
        int bidx = 0;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int r = k * WARP_SIZE + lane;
            if (r < M) {
                const float val = r_col[k] + sh_vec[r];
                if (val > bval || (val == bval && r < bidx)) {
                    bval = val;
                    bidx = r;
                }
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            best_row[gw] = bidx;
        }
    }

    ++phase;
    grid_barrier(bar, phase * num_ctas);

    for (int r = global_tid; r < M; r += global_threads) {
        mutual_out[r] = (best_row[index_out[r]] == r);
    }
}

// -- 3) L2-streaming cooperative sinkhorn_mutual fallback (arbitrary M, N > 1024) ----------------------------

__device__ __forceinline__ void coop_transpose_log2(
    const float* __restrict__ scores,
    float* __restrict__ scores_T,
    int M,
    int N,
    float (&tile)[32][33]
) {
    const int tiles_x = (N + 31) >> 5;
    const int tiles_y = (M + 31) >> 5;
    const int total_tiles = tiles_x * tiles_y;
    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;

    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int by = t / tiles_x;
        const int bx = t - by * tiles_x;
        const int col = (bx << 5) + tx;
        const int row_base = by << 5;
#pragma unroll
        for (int k = 0; k < 32; k += 8) {
            const int r = row_base + ty + k;
            if (r < M && col < N) {
                tile[ty + k][tx] = scores[static_cast<int64_t>(r) * N + col] * LOG2E;
            }
        }
        __syncthreads();
        const int out_row = (bx << 5) + ty;
        const int out_col = (by << 5) + tx;
#pragma unroll
        for (int k = 0; k < 32; k += 8) {
            const int r = out_row + k;
            if (r < N && out_col < M) {
                scores_T[static_cast<int64_t>(r) * M + out_col] = tile[tx][ty + k];
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void dustbin_lse2_vec(
    const float* __restrict__ vec,
    int len,
    int lane,
    float alpha2,
    float target_log2_marginal,
    float* __restrict__ out_ptr
) {
    float m = -INFINITY;
    for (int j = lane; j < len; j += 32) {
        m = fmaxf(m, vec[j]);
    }
    m = warp_max(m);
    float s = 0.0f;
    for (int j = lane; j < len; j += 32) {
        s += fast_exp2f(vec[j] - m);
    }
    s = warp_sum(s);
    if (lane == 0) {
        *out_ptr = target_log2_marginal - (alpha2 + m + log2f(s));
    }
}

__global__ __launch_bounds__(THREADS_PER_CTA, 2) void sinkhorn_mutual_stream_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ bin_score_ptr,
    float* __restrict__ scores_T,
    float* __restrict__ u,
    float* __restrict__ v,
    int* __restrict__ best_row,
    float* __restrict__ prob_out,
    int64_t* __restrict__ index_out,
    bool* __restrict__ mutual_out,
    int M,
    int N,
    int iters
) {
    __shared__ union {
        float tile[32][33];
        float vec[STREAM_SMEM_VEC_MAX + 1];
    } smem;
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    const int total_threads = gridDim.x * THREADS_PER_CTA;
    const int warp_id = blockIdx.x * WARPS_PER_CTA + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int num_warps = gridDim.x * WARPS_PER_CTA;

    const float alpha2 = bin_score_ptr[0] * LOG2E;
    const float norm2 = -log2f(static_cast<float>(M + N));
    const float log2_mu_dust = log2f(static_cast<float>(N)) + norm2;
    const float log2_nu_dust = log2f(static_cast<float>(M)) + norm2;

    coop_transpose_log2(scores, scores_T, M, N, smem.tile);
    for (int i = tid; i <= M; i += total_threads) {
        u[i] = 0.0f;
    }
    for (int j = tid; j <= N; j += total_threads) {
        v[j] = 0.0f;
    }
    grid.sync();

    const bool stage_v = (N <= STREAM_SMEM_VEC_MAX);
    const bool stage_u = (M <= STREAM_SMEM_VEC_MAX);

    for (int iter = 0; iter < iters; ++iter) {
        const float* v_src = v;
        if (stage_v) {
            __syncthreads();
            for (int j = threadIdx.x; j <= N; j += THREADS_PER_CTA) {
                smem.vec[j] = v[j];
            }
            __syncthreads();
            v_src = smem.vec;
        }
        const float dust_v = alpha2 + v_src[N];
        for (int r = warp_id; r < M; r += num_warps) {
            const float* row_ptr = scores + static_cast<int64_t>(r) * N;
            float m = dust_v;
            int j = lane;
            for (; j + 96 < N; j += 128) {
                m = fmaxf(m, fmaf(row_ptr[j], LOG2E, v_src[j]));
                m = fmaxf(m, fmaf(row_ptr[j + 32], LOG2E, v_src[j + 32]));
                m = fmaxf(m, fmaf(row_ptr[j + 64], LOG2E, v_src[j + 64]));
                m = fmaxf(m, fmaf(row_ptr[j + 96], LOG2E, v_src[j + 96]));
            }
            for (; j < N; j += 32) {
                m = fmaxf(m, fmaf(row_ptr[j], LOG2E, v_src[j]));
            }
            m = warp_max(m);

            float s = 0.0f;
            j = lane;
            for (; j + 96 < N; j += 128) {
                s += fast_exp2f(fmaf(row_ptr[j], LOG2E, v_src[j]) - m);
                s += fast_exp2f(fmaf(row_ptr[j + 32], LOG2E, v_src[j + 32]) - m);
                s += fast_exp2f(fmaf(row_ptr[j + 64], LOG2E, v_src[j + 64]) - m);
                s += fast_exp2f(fmaf(row_ptr[j + 96], LOG2E, v_src[j + 96]) - m);
            }
            for (; j < N; j += 32) {
                s += fast_exp2f(fmaf(row_ptr[j], LOG2E, v_src[j]) - m);
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += fast_exp2f(dust_v - m);
                u[r] = norm2 - (m + log2f(s));
            }
        }
        if (warp_id == 0) {
            dustbin_lse2_vec(v_src, N + 1, lane, alpha2, log2_mu_dust, &u[M]);
        }
        grid.sync();

        const float* u_src = u;
        if (stage_u) {
            __syncthreads();
            for (int i = threadIdx.x; i <= M; i += THREADS_PER_CTA) {
                smem.vec[i] = u[i];
            }
            __syncthreads();
            u_src = smem.vec;
        }
        const float dust_u = alpha2 + u_src[M];
        for (int c = warp_id; c < N; c += num_warps) {
            const float* col_ptr = scores_T + static_cast<int64_t>(c) * M;
            float m = dust_u;
            int i = lane;
            for (; i + 96 < M; i += 128) {
                m = fmaxf(m, col_ptr[i] + u_src[i]);
                m = fmaxf(m, col_ptr[i + 32] + u_src[i + 32]);
                m = fmaxf(m, col_ptr[i + 64] + u_src[i + 64]);
                m = fmaxf(m, col_ptr[i + 96] + u_src[i + 96]);
            }
            for (; i < M; i += 32) {
                m = fmaxf(m, col_ptr[i] + u_src[i]);
            }
            m = warp_max(m);

            float s = 0.0f;
            i = lane;
            for (; i + 96 < M; i += 128) {
                s += fast_exp2f(col_ptr[i] + u_src[i] - m);
                s += fast_exp2f(col_ptr[i + 32] + u_src[i + 32] - m);
                s += fast_exp2f(col_ptr[i + 64] + u_src[i + 64] - m);
                s += fast_exp2f(col_ptr[i + 96] + u_src[i + 96] - m);
            }
            for (; i < M; i += 32) {
                s += fast_exp2f(col_ptr[i] + u_src[i] - m);
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += fast_exp2f(dust_u - m);
                v[c] = norm2 - (m + log2f(s));
            }
        }
        if (warp_id == 0) {
            dustbin_lse2_vec(u_src, M + 1, lane, alpha2, log2_nu_dust, &v[N]);
        }
        grid.sync();
    }

    const float* v_src = v;
    if (stage_v) {
        __syncthreads();
        for (int j = threadIdx.x; j < N; j += THREADS_PER_CTA) {
            smem.vec[j] = v[j];
        }
        __syncthreads();
        v_src = smem.vec;
    }
    for (int r = warp_id; r < M; r += num_warps) {
        const float* row_ptr = scores + static_cast<int64_t>(r) * N;
        float bval = -INFINITY;
        int bidx = 0;
        for (int j = lane; j < N; j += 32) {
            const float val = fmaf(row_ptr[j], LOG2E, v_src[j]);
            if (val > bval || (val == bval && j < bidx)) {
                bval = val;
                bidx = j;
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            prob_out[r] = exp2f(bval + u[r] - norm2);
            index_out[r] = static_cast<int64_t>(bidx);
        }
    }

    const float* u_src = u;
    if (stage_u) {
        __syncthreads();
        for (int i = threadIdx.x; i < M; i += THREADS_PER_CTA) {
            smem.vec[i] = u[i];
        }
        __syncthreads();
        u_src = smem.vec;
    }
    for (int c = warp_id; c < N; c += num_warps) {
        const float* col_ptr = scores_T + static_cast<int64_t>(c) * M;
        float bval = -INFINITY;
        int bidx = 0;
        for (int i = lane; i < M; i += 32) {
            const float val = col_ptr[i] + u_src[i];
            if (val > bval || (val == bval && i < bidx)) {
                bval = val;
                bidx = i;
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            best_row[c] = bidx;
        }
    }
    grid.sync();

    for (int r = tid; r < M; r += total_threads) {
        mutual_out[r] = (best_row[index_out[r]] == r);
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

extern "C" int sinkhorn_mutual_max_resident_ctas(int max_c, int sm_count) {
    int blocks_per_sm = 0;
    cudaError_t err;
    if (max_c <= 256) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, sinkhorn_mutual_reg_kernel<256>, THREADS_PER_CTA, 0
        );
    } else if (max_c <= 512) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, sinkhorn_mutual_reg_kernel<512>, THREADS_PER_CTA, 0
        );
    } else if (max_c <= 1024) {
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, sinkhorn_mutual_reg_kernel<1024>, THREADS_PER_CTA, 0
        );
    } else {
        return 0;
    }
    if (err != cudaSuccess) {
        return 0;
    }
    return blocks_per_sm * sm_count;
}

extern "C" int sinkhorn_cooperative_supported(int device_id) {
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device_id) != cudaSuccess) {
        return 0;
    }
    return supported;
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

extern "C" int launch_sinkhorn_mutual_reg(
    const void* scores,
    const void* bin_score,
    void* u_buf,
    void* v_buf,
    void* bar_buf,
    void* best_row,
    void* prob_out,
    void* index_out,
    void* mutual_out,
    int M,
    int N,
    int iters,
    int num_ctas,
    int max_c,
    void* stream_ptr
) {
    const float norm2 = -log2f(static_cast<float>(M + N));
    const float log_mu_dust2 = log2f(static_cast<float>(N)) + norm2;
    const float log_nu_dust2 = log2f(static_cast<float>(M)) + norm2;

    dim3 grid(num_ctas);
    dim3 block(THREADS_PER_CTA);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (max_c <= 256) {
        sinkhorn_mutual_reg_kernel<256><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(bin_score),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<int*>(best_row), static_cast<float*>(prob_out), static_cast<int64_t*>(index_out),
            static_cast<bool*>(mutual_out), M, N, iters, norm2, log_mu_dust2, log_nu_dust2, num_ctas);
    } else if (max_c <= 512) {
        sinkhorn_mutual_reg_kernel<512><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(bin_score),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<int*>(best_row), static_cast<float*>(prob_out), static_cast<int64_t*>(index_out),
            static_cast<bool*>(mutual_out), M, N, iters, norm2, log_mu_dust2, log_nu_dust2, num_ctas);
    } else if (max_c <= 1024) {
        sinkhorn_mutual_reg_kernel<1024><<<grid, block, 0, stream>>>(
            static_cast<const float*>(scores), static_cast<const float*>(bin_score),
            static_cast<float*>(u_buf), static_cast<float*>(v_buf), static_cast<int*>(bar_buf),
            static_cast<int*>(best_row), static_cast<float*>(prob_out), static_cast<int64_t*>(index_out),
            static_cast<bool*>(mutual_out), M, N, iters, norm2, log_mu_dust2, log_nu_dust2, num_ctas);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_sinkhorn_mutual_stream(
    const void* scores,
    const void* bin_score,
    void* scores_T,
    void* u,
    void* v,
    void* best_row,
    void* prob_out,
    void* index_out,
    void* mutual_out,
    int M,
    int N,
    int iters,
    int device_id,
    void* stream_ptr
) {
    int num_sms = 0;
    if (cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id) != cudaSuccess || num_sms <= 0) {
        return -1;
    }
    int blocks_per_sm = 0;
    cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, sinkhorn_mutual_stream_kernel, THREADS_PER_CTA, 0
    );
    if (occ_err != cudaSuccess || blocks_per_sm <= 0) {
        return static_cast<int>(occ_err != cudaSuccess ? occ_err : cudaErrorInvalidConfiguration);
    }
    const int max_blocks = blocks_per_sm * num_sms;
    const int max_dim = (M > N) ? M : N;
    int grid_blocks = (max_dim + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    if (grid_blocks > max_blocks) {
        grid_blocks = max_blocks;
    }
    if (grid_blocks < 1) {
        grid_blocks = 1;
    }
    void* args[] = {
        const_cast<void**>(&scores),
        const_cast<void**>(&bin_score),
        &scores_T,
        &u,
        &v,
        &best_row,
        &prob_out,
        &index_out,
        &mutual_out,
        &M,
        &N,
        &iters,
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(sinkhorn_mutual_stream_kernel),
        dim3(grid_blocks),
        dim3(THREADS_PER_CTA),
        args,
        0,
        static_cast<cudaStream_t>(stream_ptr)
    );
    return static_cast<int>(err);
}
