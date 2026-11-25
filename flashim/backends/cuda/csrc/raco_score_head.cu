// Fused RaCo detector score_head[2:]: conv3x3(8->4) -> SELU -> conv3x3(4->4) -> SELU -> conv3x3(4->1), FP32.
// Reads [B, 8, H, W] NCHW once into shared memory with a 3-pixel halo and writes [B, 1, H, W] once, avoiding four
// full-resolution intermediate passes (2 small-channel convs + 2 SELU kernels). Exact IEEE FP32 accumulation.
#include <cuda_runtime.h>
#include <cmath>

constexpr int TH = 16, TW = 16;
constexpr int IN_H = TH + 6, IN_W = TW + 6;      // 22 x 22
constexpr int M1_H = TH + 4, M1_W = TW + 4;      // 20 x 20
constexpr int M2_H = TH + 2, M2_W = TW + 2;      // 18 x 18
constexpr int THREADS = TH * TW;                  // 256

__device__ __forceinline__ float selu_f32(float v) {
  constexpr float kAlpha = 1.6732632423543772848170429916717f;
  constexpr float kScale = 1.0507009873554804934193349852946f;
  return kScale * (v > 0.f ? v : kAlpha * (expf(v) - 1.f));
}

__global__ void __launch_bounds__(THREADS)
score_head_kernel(const float* __restrict__ x, float* __restrict__ out,
                  const float* __restrict__ w1, const float* __restrict__ w2, const float* __restrict__ w3,
                  int H, int W) {
  __shared__ float w1_s[4][8][9];
  __shared__ float w2_s[4][4][9];
  __shared__ float w3_s[4][9];
  __shared__ float in_s[8][IN_H][IN_W];
  __shared__ float m1_s[4][M1_H][M1_W];
  __shared__ float m2_s[4][M2_H][M2_W];

  const int tid = threadIdx.x;
  const int b = blockIdx.z;
  const int oy0 = blockIdx.y * TH;
  const int ox0 = blockIdx.x * TW;

  // Stage weights (288 + 144 + 36 = 468 floats)
  for (int i = tid; i < 288; i += THREADS) reinterpret_cast<float*>(w1_s)[i] = w1[i];
  for (int i = tid; i < 144; i += THREADS) reinterpret_cast<float*>(w2_s)[i] = w2[i];
  if (tid < 36) reinterpret_cast<float*>(w3_s)[tid] = w3[tid];

  // Stage input tile [8, 22, 22] from [B, 8, H, W] NCHW with zero padding outside [0, H) x [0, W)
  const size_t plane = static_cast<size_t>(H) * W;
  const float* xb = x + static_cast<size_t>(b) * 8 * plane;
  constexpr int IN_TOTAL = 8 * IN_H * IN_W;
  for (int i = tid; i < IN_TOTAL; i += THREADS) {
    const int rx = i % IN_W;
    const int tmp = i / IN_W;
    const int ry = tmp % IN_H;
    const int c = tmp / IN_H;
    const int iy = oy0 - 3 + ry;
    const int ix = ox0 - 3 + rx;
    float v = 0.f;
    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
      v = xb[static_cast<size_t>(c) * plane + static_cast<size_t>(iy) * W + ix];
    }
    in_s[c][ry][rx] = v;
  }
  __syncthreads();

  // Layer 1: conv3x3(8 -> 4) + SELU over [20, 20]
  constexpr int M1_PIX = M1_H * M1_W;
  for (int p = tid; p < M1_PIX; p += THREADS) {
    const int ry = p / M1_W;
    const int rx = p % M1_W;
    const int py = oy0 - 2 + ry;
    const int px = ox0 - 2 + rx;
    if (py >= 0 && py < H && px >= 0 && px < W) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int ci = 0; ci < 8; ++ci) {
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          #pragma unroll
          for (int kx = 0; kx < 3; ++kx) {
            const float xv = in_s[ci][ry + ky][rx + kx];
            const int tap = ky * 3 + kx;
            #pragma unroll
            for (int co = 0; co < 4; ++co) acc[co] = fmaf(xv, w1_s[co][ci][tap], acc[co]);
          }
        }
      }
      #pragma unroll
      for (int co = 0; co < 4; ++co) m1_s[co][ry][rx] = selu_f32(acc[co]);
    } else {
      #pragma unroll
      for (int co = 0; co < 4; ++co) m1_s[co][ry][rx] = 0.f;
    }
  }
  __syncthreads();

  // Layer 2: conv3x3(4 -> 4) + SELU over [18, 18]
  constexpr int M2_PIX = M2_H * M2_W;
  for (int p = tid; p < M2_PIX; p += THREADS) {
    const int ry = p / M2_W;
    const int rx = p % M2_W;
    const int py = oy0 - 1 + ry;
    const int px = ox0 - 1 + rx;
    if (py >= 0 && py < H && px >= 0 && px < W) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int ci = 0; ci < 4; ++ci) {
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          #pragma unroll
          for (int kx = 0; kx < 3; ++kx) {
            const float xv = m1_s[ci][ry + ky][rx + kx];
            const int tap = ky * 3 + kx;
            #pragma unroll
            for (int co = 0; co < 4; ++co) acc[co] = fmaf(xv, w2_s[co][ci][tap], acc[co]);
          }
        }
      }
      #pragma unroll
      for (int co = 0; co < 4; ++co) m2_s[co][ry][rx] = selu_f32(acc[co]);
    } else {
      #pragma unroll
      for (int co = 0; co < 4; ++co) m2_s[co][ry][rx] = 0.f;
    }
  }
  __syncthreads();

  // Layer 3: conv3x3(4 -> 1) over [16, 16] (1 pixel per thread)
  const int ty = tid >> 4;
  const int tx = tid & 15;
  const int oy = oy0 + ty;
  const int ox = ox0 + tx;
  if (oy < H && ox < W) {
    float acc = 0.f;
    #pragma unroll
    for (int ci = 0; ci < 4; ++ci) {
      #pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          acc = fmaf(m2_s[ci][ty + ky][tx + kx], w3_s[ci][ky * 3 + kx], acc);
        }
      }
    }
    out[static_cast<size_t>(b) * plane + static_cast<size_t>(oy) * W + ox] = acc;
  }
}

extern "C" int launch_score_head(const void* x, void* out, const void* w1, const void* w2, const void* w3,
                                 int B, int H, int W, void* stream) {
  dim3 grid((W + TW - 1) / TW, (H + TH - 1) / TH, B);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  score_head_kernel<<<grid, THREADS, 0, s>>>(
      static_cast<const float*>(x), static_cast<float*>(out),
      static_cast<const float*>(w1), static_cast<const float*>(w2), static_cast<const float*>(w3),
      H, W);
  return static_cast<int>(cudaGetLastError());
}
