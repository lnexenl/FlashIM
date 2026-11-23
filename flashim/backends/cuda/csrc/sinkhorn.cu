// Single-launch cooperative Log-Sinkhorn optimal transport + mutual argmax (FP32).
//
// Uses mathematically identical base-2 scaling (LOG2E = log2(e)) so every inner-loop exp/log is a
// single SFU instruction (exp2f / log2f).
//
// Two paths, selected automatically from (M, N) and device SM occupancy:
//   1. Register-resident (M <= 1024, N <= 1024, active warps >= max(M, N)):
//      Each warp owns one row (r_vals[32]) and one column (c_vals[32]) in registers for all iters.
//      Global memory for `scores` is read ONCE; `u`/`v` (<= 4 KB) are cooperatively staged in smem.
//   2. General L2-streaming (any M, N >= 1, any SM count):
//      Tiled shared-memory transpose scores -> scores_T once, then branchless 2-pass warp-strided
//      log2-sum-exp2 over scores (rows) and scores_T (columns), staging u/v in smem when <= 4096.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / 32;
constexpr int kRegMaxDim = 1024;
constexpr int kRegPerLane = kRegMaxDim / 32;  // 32 floats/lane
constexpr int kStreamSmemVecMax = 4096;       // 16 KB smem for staging u / v
constexpr float kLog2E = 1.44269504088896340736f;

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return __shfl_sync(0xffffffffu, v, 0);
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

// Cooperative tiled 32x32 transpose: scores [M, N] -> scores_T [N, M] (pre-scaled by LOG2E)
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
    const int ty = threadIdx.x >> 5;  // 0..7

    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int by = t / tiles_x;
        const int bx = t - by * tiles_x;
        const int col = (bx << 5) + tx;
        const int row_base = by << 5;
        #pragma unroll
        for (int k = 0; k < 32; k += 8) {
            const int r = row_base + ty + k;
            if (r < M && col < N) {
                tile[ty + k][tx] = scores[static_cast<int64_t>(r) * N + col] * kLog2E;
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
        s += exp2f(vec[j] - m);
    }
    s = warp_sum(s);
    if (lane == 0) {
        *out_ptr = target_log2_marginal - (alpha2 + m + log2f(s));
    }
}

// Path A: register-resident cooperative kernel (M <= 1024, N <= 1024, total_warps >= max(M, N)).
__global__ __launch_bounds__(kBlockThreads, 2)
void sinkhorn_mutual_reg_kernel(
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
        float vec[kRegMaxDim + 1];
    } smem;

    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * kBlockThreads + threadIdx.x;
    const int total_threads = gridDim.x * kBlockThreads;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_id = blockIdx.x * kWarpsPerBlock + warp_in_block;
    const int num_warps = gridDim.x * kWarpsPerBlock;

    const float alpha2 = bin_score_ptr[0] * kLog2E;
    const float norm2 = -log2f(static_cast<float>(M + N));
    const float log2_mu_dust = log2f(static_cast<float>(N)) + norm2;
    const float log2_nu_dust = log2f(static_cast<float>(M)) + norm2;

    // Step 0: cooperative transpose scores*LOG2E -> scores_T and zero u, v.
    coop_transpose_log2(scores, scores_T, M, N, smem.tile);
    for (int i = tid; i <= M; i += total_threads) {
        u[i] = 0.0f;
    }
    for (int j = tid; j <= N; j += total_threads) {
        v[j] = 0.0f;
    }
    grid.sync();

    // Load warp's own row (scaled by LOG2E) and column (already scaled in scores_T) into registers ONCE.
    float r_vals[kRegPerLane];
    float c_vals[kRegPerLane];
    if (warp_id < M) {
        const float* row_ptr = scores + static_cast<int64_t>(warp_id) * N;
        #pragma unroll
        for (int k = 0; k < kRegPerLane; ++k) {
            const int j = lane + (k << 5);
            r_vals[k] = (j < N) ? (row_ptr[j] * kLog2E) : 0.0f;
        }
    }
    if (warp_id < N) {
        const float* col_ptr = scores_T + static_cast<int64_t>(warp_id) * M;
        #pragma unroll
        for (int k = 0; k < kRegPerLane; ++k) {
            const int i = lane + (k << 5);
            c_vals[k] = (i < M) ? col_ptr[i] : 0.0f;
        }
    }

    const int dust_row_warp = (M < num_warps) ? M : 0;
    const int dust_col_warp = (N < num_warps) ? N : 0;

    // Step 1: Sinkhorn iterations in base-2 (scores never read from memory again).
    for (int iter = 0; iter < iters; ++iter) {
        __syncthreads();
        for (int j = threadIdx.x; j <= N; j += kBlockThreads) {
            smem.vec[j] = v[j];
        }
        __syncthreads();

        if (warp_id < M) {
            const float dust_v = alpha2 + smem.vec[N];
            float m = dust_v;
            #pragma unroll
            for (int k = 0; k < kRegPerLane; ++k) {
                const int j = lane + (k << 5);
                if (j < N) {
                    m = fmaxf(m, r_vals[k] + smem.vec[j]);
                }
            }
            m = warp_max(m);
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < kRegPerLane; ++k) {
                const int j = lane + (k << 5);
                if (j < N) {
                    s += exp2f(r_vals[k] + smem.vec[j] - m);
                }
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += exp2f(dust_v - m);
                u[warp_id] = norm2 - (m + log2f(s));
            }
        }
        if (warp_id == dust_row_warp) {
            dustbin_lse2_vec(smem.vec, N + 1, lane, alpha2, log2_mu_dust, &u[M]);
        }
        grid.sync();

        __syncthreads();
        for (int i = threadIdx.x; i <= M; i += kBlockThreads) {
            smem.vec[i] = u[i];
        }
        __syncthreads();

        if (warp_id < N) {
            const float dust_u = alpha2 + smem.vec[M];
            float m = dust_u;
            #pragma unroll
            for (int k = 0; k < kRegPerLane; ++k) {
                const int i = lane + (k << 5);
                if (i < M) {
                    m = fmaxf(m, c_vals[k] + smem.vec[i]);
                }
            }
            m = warp_max(m);
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < kRegPerLane; ++k) {
                const int i = lane + (k << 5);
                if (i < M) {
                    s += exp2f(c_vals[k] + smem.vec[i] - m);
                }
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += exp2f(dust_u - m);
                v[warp_id] = norm2 - (m + log2f(s));
            }
        }
        if (warp_id == dust_col_warp) {
            dustbin_lse2_vec(smem.vec, M + 1, lane, alpha2, log2_nu_dust, &v[N]);
        }
        grid.sync();
    }

    // Step 2: Row argmax from register-resident r_vals + smem v.
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += kBlockThreads) {
        smem.vec[j] = v[j];
    }
    __syncthreads();
    if (warp_id < M) {
        float bval = -INFINITY;
        int bidx = 0;
        #pragma unroll
        for (int k = 0; k < kRegPerLane; ++k) {
            const int j = lane + (k << 5);
            if (j < N) {
                const float val = r_vals[k] + smem.vec[j];
                if (val > bval || (val == bval && j < bidx)) {
                    bval = val;
                    bidx = j;
                }
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            prob_out[warp_id] = exp2f(bval + u[warp_id] - norm2);
            index_out[warp_id] = static_cast<int64_t>(bidx);
        }
    }

    // Step 3: Col argmax from register-resident c_vals + smem u.
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += kBlockThreads) {
        smem.vec[i] = u[i];
    }
    __syncthreads();
    if (warp_id < N) {
        float bval = -INFINITY;
        int bidx = 0;
        #pragma unroll
        for (int k = 0; k < kRegPerLane; ++k) {
            const int i = lane + (k << 5);
            if (i < M) {
                const float val = c_vals[k] + smem.vec[i];
                if (val > bval || (val == bval && i < bidx)) {
                    bval = val;
                    bidx = i;
                }
            }
        }
        warp_argmax(bval, bidx);
        if (lane == 0) {
            best_row[warp_id] = bidx;
        }
    }
    grid.sync();

    // Step 4: Mutual check.
    for (int r = tid; r < M; r += total_threads) {
        mutual_out[r] = (best_row[index_out[r]] == r);
    }
}

// Path B: general L2-streaming cooperative kernel for arbitrary (M, N) >= 1.
// Uses branchless 2-pass exact base-2 logsumexp and stages u/v in shared memory when <= 4096.
__global__ __launch_bounds__(kBlockThreads, 2)
void sinkhorn_mutual_stream_kernel(
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
        float vec[kStreamSmemVecMax + 1];
    } smem;
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * kBlockThreads + threadIdx.x;
    const int total_threads = gridDim.x * kBlockThreads;
    const int warp_id = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int num_warps = gridDim.x * kWarpsPerBlock;

    const float alpha2 = bin_score_ptr[0] * kLog2E;
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

    const bool stage_v = (N <= kStreamSmemVecMax);
    const bool stage_u = (M <= kStreamSmemVecMax);

    for (int iter = 0; iter < iters; ++iter) {
        const float* v_src = v;
        if (stage_v) {
            __syncthreads();
            for (int j = threadIdx.x; j <= N; j += kBlockThreads) {
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
                m = fmaxf(m, fmaf(row_ptr[j], kLog2E, v_src[j]));
                m = fmaxf(m, fmaf(row_ptr[j + 32], kLog2E, v_src[j + 32]));
                m = fmaxf(m, fmaf(row_ptr[j + 64], kLog2E, v_src[j + 64]));
                m = fmaxf(m, fmaf(row_ptr[j + 96], kLog2E, v_src[j + 96]));
            }
            for (; j < N; j += 32) {
                m = fmaxf(m, fmaf(row_ptr[j], kLog2E, v_src[j]));
            }
            m = warp_max(m);

            float s = 0.0f;
            j = lane;
            for (; j + 96 < N; j += 128) {
                s += exp2f(fmaf(row_ptr[j], kLog2E, v_src[j]) - m);
                s += exp2f(fmaf(row_ptr[j + 32], kLog2E, v_src[j + 32]) - m);
                s += exp2f(fmaf(row_ptr[j + 64], kLog2E, v_src[j + 64]) - m);
                s += exp2f(fmaf(row_ptr[j + 96], kLog2E, v_src[j + 96]) - m);
            }
            for (; j < N; j += 32) {
                s += exp2f(fmaf(row_ptr[j], kLog2E, v_src[j]) - m);
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += exp2f(dust_v - m);
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
            for (int i = threadIdx.x; i <= M; i += kBlockThreads) {
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
                s += exp2f(col_ptr[i] + u_src[i] - m);
                s += exp2f(col_ptr[i + 32] + u_src[i + 32] - m);
                s += exp2f(col_ptr[i + 64] + u_src[i + 64] - m);
                s += exp2f(col_ptr[i + 96] + u_src[i + 96] - m);
            }
            for (; i < M; i += 32) {
                s += exp2f(col_ptr[i] + u_src[i] - m);
            }
            s = warp_sum(s);
            if (lane == 0) {
                s += exp2f(dust_u - m);
                v[c] = norm2 - (m + log2f(s));
            }
        }
        if (warp_id == 0) {
            dustbin_lse2_vec(u_src, M + 1, lane, alpha2, log2_nu_dust, &v[N]);
        }
        grid.sync();
    }

    // Row argmax
    const float* v_src = v;
    if (stage_v) {
        __syncthreads();
        for (int j = threadIdx.x; j < N; j += kBlockThreads) {
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
            const float val = fmaf(row_ptr[j], kLog2E, v_src[j]);
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

    // Col argmax
    const float* u_src = u;
    if (stage_u) {
        __syncthreads();
        for (int i = threadIdx.x; i < M; i += kBlockThreads) {
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

extern "C" int sinkhorn_cooperative_supported(int device_id) {
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device_id) != cudaSuccess) {
        return 0;
    }
    return supported;
}

extern "C" int launch_sinkhorn_mutual(
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

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    const int max_dim = (M > N) ? M : N;

    if (M <= kRegMaxDim && N <= kRegMaxDim) {
        int blocks_per_sm = 0;
        cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, sinkhorn_mutual_reg_kernel, kBlockThreads, 0
        );
        if (occ_err == cudaSuccess && blocks_per_sm > 0) {
            const int max_blocks = blocks_per_sm * num_sms;
            if (max_blocks * kWarpsPerBlock >= max_dim) {
                int needed_blocks = (max_dim + 1 + kWarpsPerBlock - 1) / kWarpsPerBlock;
                if (needed_blocks > max_blocks) {
                    needed_blocks = max_blocks;
                }
                if (needed_blocks < 1) {
                    needed_blocks = 1;
                }
                cudaError_t err = cudaLaunchCooperativeKernel(
                    reinterpret_cast<const void*>(sinkhorn_mutual_reg_kernel),
                    dim3(needed_blocks),
                    dim3(kBlockThreads),
                    args,
                    0,
                    stream
                );
                return static_cast<int>(err);
            }
        }
    }

    int blocks_per_sm = 0;
    cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, sinkhorn_mutual_stream_kernel, kBlockThreads, 0
    );
    if (occ_err != cudaSuccess || blocks_per_sm <= 0) {
        return static_cast<int>(occ_err != cudaSuccess ? occ_err : cudaErrorInvalidConfiguration);
    }
    const int max_blocks = blocks_per_sm * num_sms;
    int grid_blocks = (max_dim + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (grid_blocks > max_blocks) {
        grid_blocks = max_blocks;
    }
    if (grid_blocks < 1) {
        grid_blocks = 1;
    }
    cudaError_t err = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(sinkhorn_mutual_stream_kernel),
        dim3(grid_blocks),
        dim3(kBlockThreads),
        args,
        0,
        stream
    );
    return static_cast<int>(err);
}
