// Fused ALIKED score_head[2:]: conv3x3(8->4)+SELU -> conv3x3(4->4)+SELU -> conv3x3(4->1) [+ sigmoid], FP32.
// Single-kernel tiled halo fusion (halo 3 -> 2 -> 1 -> 0): reads [B, 8, H, W] once and writes [B, 1, H, W] once,
// eliminating two full-res [B, 4, H, W] DRAM round-trips and 3 slow tiny-channel cuDNN kernels.
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

constexpr int TH = 12;
constexpr int TW = 32;
constexpr int H0 = TH + 6;  // 18 (halo 3)
constexpr int W0 = TW + 6;  // 38
constexpr int H1 = TH + 4;  // 16 (halo 2)
constexpr int W1 = TW + 4;  // 36
constexpr int H2 = TH + 2;  // 14 (halo 1)
constexpr int W2 = TW + 2;  // 34
constexpr int THREADS = 256;

__device__ __forceinline__ float selu_f32(float v) {
  return 1.0507009873554805f * (v > 0.f ? v : 1.6732632423543772f * (expf(v) - 1.f));
}

template <bool APPLY_SIGMOID>
__global__ void __launch_bounds__(THREADS)
aliked_score_head_kernel(
    const float* __restrict__ head,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    float* __restrict__ out,
    int H, int W, int tiles_x) {
  __shared__ float s_a[8 * H0 * W0];   // 21.9 KB: stage 0 [8, 18, 38], reused for stage 2 [4, 14, 34]
  __shared__ float s_b[4 * H1 * W1];   // 9.2 KB:  stage 1 [4, 16, 36]
  __shared__ float s_w1[4 * 8 * 9];    // 1.15 KB
  __shared__ float s_w2[4 * 4 * 9];    // 0.58 KB
  __shared__ float s_w3[4 * 9];        // 0.14 KB

  const int tid = threadIdx.x;
  const int b = blockIdx.z;
  const int oy0 = blockIdx.y * TH;
  const int ox0 = blockIdx.x * TW;

  // Load all 468 weights cooperatively.
  for (int i = tid; i < 4 * 8 * 9; i += THREADS) s_w1[i] = w1[i];
  for (int i = tid; i < 4 * 4 * 9; i += THREADS) s_w2[i] = w2[i];
  if (tid < 4 * 9) s_w3[tid] = w3[tid];

  // Stage 0: load input tile [8, H0, W0] with exact zero-padding outside [0, H) x [0, W).
  const int64_t plane = static_cast<int64_t>(H) * W;
  const float* base_in = head + static_cast<int64_t>(b) * 8 * plane;
  constexpr int N0 = 8 * H0 * W0;
  for (int idx = tid; idx < N0; idx += THREADS) {
    const int lx = idx % W0;
    const int tmp = idx / W0;
    const int ly = tmp % H0;
    const int c = tmp / H0;
    const int gy = oy0 - 3 + ly;
    const int gx = ox0 - 3 + lx;
    float v = 0.f;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      v = base_in[static_cast<int64_t>(c) * plane + static_cast<int64_t>(gy) * W + gx];
    }
    s_a[idx] = v;
  }
  __syncthreads();

  // Stage 1: conv3x3(8->4) + SELU over [H1, W1]; zero outside image bounds (exact padding for stage 2).
  constexpr int P1 = H1 * W1;
  for (int p = tid; p < P1; p += THREADS) {
    const int ly = p / W1;
    const int lx = p % W1;
    const int gy = oy0 - 2 + ly;
    const int gx = ox0 - 2 + lx;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* sp = s_a + c * (H0 * W0) + ly * W0 + lx;
        const float* wp0 = s_w1 + (0 * 8 + c) * 9;
        const float* wp1 = s_w1 + (1 * 8 + c) * 9;
        const float* wp2 = s_w1 + (2 * 8 + c) * 9;
        const float* wp3 = s_w1 + (3 * 8 + c) * 9;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = sp + ky * W0;
          #pragma unroll
          for (int kx = 0; kx < 3; ++kx) {
            const float x = row[kx];
            const int k = ky * 3 + kx;
            acc0 = fmaf(x, wp0[k], acc0);
            acc1 = fmaf(x, wp1[k], acc1);
            acc2 = fmaf(x, wp2[k], acc2);
            acc3 = fmaf(x, wp3[k], acc3);
          }
        }
      }
      s_b[0 * P1 + p] = selu_f32(acc0);
      s_b[1 * P1 + p] = selu_f32(acc1);
      s_b[2 * P1 + p] = selu_f32(acc2);
      s_b[3 * P1 + p] = selu_f32(acc3);
    } else {
      s_b[0 * P1 + p] = 0.f;
      s_b[1 * P1 + p] = 0.f;
      s_b[2 * P1 + p] = 0.f;
      s_b[3 * P1 + p] = 0.f;
    }
  }
  __syncthreads();

  // Stage 2: conv3x3(4->4) + SELU over [H2, W2], reusing s_a; zero outside image bounds (exact padding for stage 3).
  constexpr int P2 = H2 * W2;
  for (int p = tid; p < P2; p += THREADS) {
    const int ly = p / W2;
    const int lx = p % W2;
    const int gy = oy0 - 1 + ly;
    const int gx = ox0 - 1 + lx;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float* sp = s_b + c * P1 + ly * W1 + lx;
        const float* wp0 = s_w2 + (0 * 4 + c) * 9;
        const float* wp1 = s_w2 + (1 * 4 + c) * 9;
        const float* wp2 = s_w2 + (2 * 4 + c) * 9;
        const float* wp3 = s_w2 + (3 * 4 + c) * 9;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = sp + ky * W1;
          #pragma unroll
          for (int kx = 0; kx < 3; ++kx) {
            const float x = row[kx];
            const int k = ky * 3 + kx;
            acc0 = fmaf(x, wp0[k], acc0);
            acc1 = fmaf(x, wp1[k], acc1);
            acc2 = fmaf(x, wp2[k], acc2);
            acc3 = fmaf(x, wp3[k], acc3);
          }
        }
      }
      s_a[0 * P2 + p] = selu_f32(acc0);
      s_a[1 * P2 + p] = selu_f32(acc1);
      s_a[2 * P2 + p] = selu_f32(acc2);
      s_a[3 * P2 + p] = selu_f32(acc3);
    } else {
      s_a[0 * P2 + p] = 0.f;
      s_a[1 * P2 + p] = 0.f;
      s_a[2 * P2 + p] = 0.f;
      s_a[3 * P2 + p] = 0.f;
    }
  }
  __syncthreads();

  // Stage 3: conv3x3(4->1) [+ sigmoid] over [TH, TW] and store to global memory.
  constexpr int P3 = TH * TW;
  float* base_out = out + static_cast<int64_t>(b) * plane;
  for (int p = tid; p < P3; p += THREADS) {
    const int ly = p / TW;
    const int lx = p % TW;
    const int gy = oy0 + ly;
    const int gx = ox0 + lx;
    if (gy < H && gx < W) {
      float acc = 0.f;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float* sp = s_a + c * P2 + ly * W2 + lx;
        const float* wp = s_w3 + c * 9;
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = sp + ky * W2;
          #pragma unroll
          for (int kx = 0; kx < 3; ++kx) {
            acc = fmaf(row[kx], wp[ky * 3 + kx], acc);
          }
        }
      }
      if constexpr (APPLY_SIGMOID) {
        acc = 1.f / (1.f + expf(-acc));
      }
      base_out[static_cast<int64_t>(gy) * W + gx] = acc;
    }
  }
}

extern "C" int launch_aliked_score_head(
    const void* head, const void* w1, const void* w2, const void* w3, void* out,
    int B, int H, int W, int apply_sigmoid, void* stream) {
  dim3 grid((W + TW - 1) / TW, (H + TH - 1) / TH, B);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (apply_sigmoid) {
    aliked_score_head_kernel<true><<<grid, THREADS, 0, s>>>(
        static_cast<const float*>(head), static_cast<const float*>(w1),
        static_cast<const float*>(w2), static_cast<const float*>(w3),
        static_cast<float*>(out), H, W, grid.x);
  } else {
    aliked_score_head_kernel<false><<<grid, THREADS, 0, s>>>(
        static_cast<const float*>(head), static_cast<const float*>(w1),
        static_cast<const float*>(w2), static_cast<const float*>(w3),
        static_cast<float*>(out), H, W, grid.x);
  }
  return static_cast<int>(cudaGetLastError());
}
