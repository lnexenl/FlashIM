// Exact two-round fast NMS (SuperPoint r=4, ALIKE/ALIKED r=2) in shared memory using separable 1D max/OR.
// Bit-identical to PyTorch's 5x F.max_pool2d + elementwise simple_nms: out-of-image pixels use -INFINITY for max
// and 0 for mask OR.
// - r=2: 1 kernel launch (halo 5*2 = 10, 16x32 tile -> 36x52 smem = 28.1 KB).
// - r=4: 2 kernel launches (pass 1: rounds 0+1, halo 3*4 = 12 -> uint8 max1; pass 2: round 2, halo 2*4 = 8 -> out).
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

constexpr int TH = 16;
constexpr int TW = 32;
constexpr int THREADS = 256;

// Separable 1D horizontal + vertical max of a [SH, SW] float tile.
// Safe for out_s == in_s because horizontal reads in_s -> tmp_s, syncs, then vertical reads tmp_s -> out_s.
template <int R, int SH, int SW, int OH, int OW, int OFF_Y, int OFF_X>
__device__ __forceinline__ void sep_max_2d(
    const float* __restrict__ in_s,
    float* __restrict__ tmp_s,
    float* __restrict__ out_s,
    int tid) {
  constexpr int HY0 = OFF_Y - R;
  constexpr int H_ROWS = OH + 2 * R;
  constexpr int H_TOT = H_ROWS * OW;
  for (int i = tid; i < H_TOT; i += THREADS) {
    const int ry = HY0 + i / OW;
    const int rx = OFF_X + i % OW;
    float m = -INFINITY;
    #pragma unroll
    for (int d = -R; d <= R; ++d) {
      m = fmaxf(m, in_s[ry * SW + (rx + d)]);
    }
    tmp_s[ry * SW + rx] = m;
  }
  __syncthreads();
  constexpr int V_TOT = OH * OW;
  for (int i = tid; i < V_TOT; i += THREADS) {
    const int ry = OFF_Y + i / OW;
    const int rx = OFF_X + i % OW;
    float m = -INFINITY;
    #pragma unroll
    for (int d = -R; d <= R; ++d) {
      m = fmaxf(m, tmp_s[(ry + d) * SW + rx]);
    }
    out_s[ry * SW + rx] = m;
  }
  __syncthreads();
}

// Separable 1D horizontal + vertical OR of a [SH, SW] uint8 mask tile.
template <int R, int SH, int SW, int OH, int OW, int OFF_Y, int OFF_X>
__device__ __forceinline__ void sep_or_2d(
    const uint8_t* __restrict__ in_m,
    uint8_t* __restrict__ tmp_m,
    uint8_t* __restrict__ out_m,
    int tid) {
  constexpr int HY0 = OFF_Y - R;
  constexpr int H_ROWS = OH + 2 * R;
  constexpr int H_TOT = H_ROWS * OW;
  for (int i = tid; i < H_TOT; i += THREADS) {
    const int ry = HY0 + i / OW;
    const int rx = OFF_X + i % OW;
    uint8_t acc = 0;
    #pragma unroll
    for (int d = -R; d <= R; ++d) {
      acc |= in_m[ry * SW + (rx + d)];
    }
    tmp_m[ry * SW + rx] = acc;
  }
  __syncthreads();
  constexpr int V_TOT = OH * OW;
  for (int i = tid; i < V_TOT; i += THREADS) {
    const int ry = OFF_Y + i / OW;
    const int rx = OFF_X + i % OW;
    uint8_t acc = 0;
    #pragma unroll
    for (int d = -R; d <= R; ++d) {
      acc |= tmp_m[(ry + d) * SW + rx];
    }
    out_m[ry * SW + rx] = acc;
  }
  __syncthreads();
}

// Single-pass kernel for R=2 (halo 5*2 = 10 -> 36 x 52).
__global__ void __launch_bounds__(THREADS)
simple_nms_r2_kernel(const float* __restrict__ scores, float* __restrict__ out, int H, int W) {
  constexpr int R = 2;
  constexpr int HALO = 10;
  constexpr int SH = TH + 2 * HALO;  // 36
  constexpr int SW = TW + 2 * HALO;  // 52
  constexpr int N = SH * SW;

  __shared__ float s_sc[N];
  __shared__ float s_buf[N];
  __shared__ float s_tmp[N];
  __shared__ uint8_t s_m[N];
  __shared__ uint8_t s_mt[N];
  __shared__ uint8_t s_supp[N];

  const int tid = threadIdx.x;
  const int b = blockIdx.z;
  const int oy0 = blockIdx.y * TH;
  const int ox0 = blockIdx.x * TW;
  const int64_t plane = static_cast<int64_t>(H) * W;
  const float* src = scores + static_cast<int64_t>(b) * plane;

  for (int i = tid; i < N; i += THREADS) {
    const int ry = i / SW, rx = i % SW;
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_sc[i] = ok ? src[static_cast<int64_t>(gy) * W + gx] : -INFINITY;
    s_m[i] = 0;
  }
  __syncthreads();

  // Round 0: max0 over [R .. SH-R) x [R .. SW-R)
  sep_max_2d<R, SH, SW, SH - 2 * R, SW - 2 * R, R, R>(s_sc, s_tmp, s_buf, tid);
  for (int i = tid; i < (SH - 2 * R) * (SW - 2 * R); i += THREADS) {
    const int ry = R + i / (SW - 2 * R), rx = R + i % (SW - 2 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    s_m[idx] = (gy >= 0 && gy < H && gx >= 0 && gx < W && s_sc[idx] == s_buf[idx]) ? 1 : 0;
  }
  __syncthreads();

  // Round 1: supp1 over [2R .. SH-2R), s1 -> s_buf, max_pool(s1) -> s_buf over [3R .. SH-3R)
  sep_or_2d<R, SH, SW, SH - 4 * R, SW - 4 * R, 2 * R, 2 * R>(s_m, s_mt, s_supp, tid);
  for (int i = tid; i < (SH - 4 * R) * (SW - 4 * R); i += THREADS) {
    const int ry = 2 * R + i / (SW - 4 * R), rx = 2 * R + i % (SW - 4 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_buf[idx] = ok ? (s_supp[idx] ? 0.f : s_sc[idx]) : -INFINITY;
  }
  __syncthreads();
  sep_max_2d<R, SH, SW, SH - 6 * R, SW - 6 * R, 3 * R, 3 * R>(s_buf, s_tmp, s_buf, tid);
  for (int i = tid; i < (SH - 6 * R) * (SW - 6 * R); i += THREADS) {
    const int ry = 3 * R + i / (SW - 6 * R), rx = 3 * R + i % (SW - 6 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W && !s_supp[idx] && s_sc[idx] == s_buf[idx]) {
      s_m[idx] = 1;
    }
  }
  __syncthreads();

  // Round 2: supp2 over [4R .. SH-4R), s2 -> s_buf, max_pool(s2) -> s_buf over [5R .. SH-5R) = output [TH, TW]
  sep_or_2d<R, SH, SW, SH - 8 * R, SW - 8 * R, 4 * R, 4 * R>(s_m, s_mt, s_supp, tid);
  for (int i = tid; i < (SH - 8 * R) * (SW - 8 * R); i += THREADS) {
    const int ry = 4 * R + i / (SW - 8 * R), rx = 4 * R + i % (SW - 8 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_buf[idx] = ok ? (s_supp[idx] ? 0.f : s_sc[idx]) : -INFINITY;
  }
  __syncthreads();
  sep_max_2d<R, SH, SW, TH, TW, HALO, HALO>(s_buf, s_tmp, s_buf, tid);

  float* dst = out + static_cast<int64_t>(b) * plane;
  for (int i = tid; i < TH * TW; i += THREADS) {
    const int ly = i / TW, lx = i % TW;
    const int gy = oy0 + ly, gx = ox0 + lx;
    if (gy < H && gx < W) {
      const int idx = (HALO + ly) * SW + (HALO + lx);
      const bool keep = s_m[idx] || (!s_supp[idx] && s_sc[idx] == s_buf[idx]);
      dst[static_cast<int64_t>(gy) * W + gx] = keep ? s_sc[idx] : 0.f;
    }
  }
}

// Pass 1 for R=4: rounds 0 + 1 (halo 3*4 = 12 -> 40 x 56, 33.6 KB smem) -> uint8 max1.
__global__ void __launch_bounds__(THREADS)
simple_nms_r4_pass1_kernel(const float* __restrict__ scores, uint8_t* __restrict__ max1_out, int H, int W) {
  constexpr int R = 4;
  constexpr int HALO = 12;
  constexpr int SH = TH + 2 * HALO;  // 40
  constexpr int SW = TW + 2 * HALO;  // 56
  constexpr int N = SH * SW;

  __shared__ float s_sc[N];
  __shared__ float s_buf[N];
  __shared__ float s_tmp[N];
  __shared__ uint8_t s_m[N];
  __shared__ uint8_t s_mt[N];
  __shared__ uint8_t s_supp[N];

  const int tid = threadIdx.x;
  const int b = blockIdx.z;
  const int oy0 = blockIdx.y * TH;
  const int ox0 = blockIdx.x * TW;
  const int64_t plane = static_cast<int64_t>(H) * W;
  const float* src = scores + static_cast<int64_t>(b) * plane;

  for (int i = tid; i < N; i += THREADS) {
    const int ry = i / SW, rx = i % SW;
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_sc[i] = ok ? src[static_cast<int64_t>(gy) * W + gx] : -INFINITY;
    s_m[i] = 0;
  }
  __syncthreads();

  // Round 0: max0 over [R .. SH-R)
  sep_max_2d<R, SH, SW, SH - 2 * R, SW - 2 * R, R, R>(s_sc, s_tmp, s_buf, tid);
  for (int i = tid; i < (SH - 2 * R) * (SW - 2 * R); i += THREADS) {
    const int ry = R + i / (SW - 2 * R), rx = R + i % (SW - 2 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    s_m[idx] = (gy >= 0 && gy < H && gx >= 0 && gx < W && s_sc[idx] == s_buf[idx]) ? 1 : 0;
  }
  __syncthreads();

  // Round 1: supp1 over [2R .. SH-2R), s1 -> s_buf, max_pool(s1) -> s_buf over [3R .. SH-3R) = [TH, TW]
  sep_or_2d<R, SH, SW, SH - 4 * R, SW - 4 * R, 2 * R, 2 * R>(s_m, s_mt, s_supp, tid);
  for (int i = tid; i < (SH - 4 * R) * (SW - 4 * R); i += THREADS) {
    const int ry = 2 * R + i / (SW - 4 * R), rx = 2 * R + i % (SW - 4 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_buf[idx] = ok ? (s_supp[idx] ? 0.f : s_sc[idx]) : -INFINITY;
  }
  __syncthreads();
  sep_max_2d<R, SH, SW, TH, TW, HALO, HALO>(s_buf, s_tmp, s_buf, tid);

  uint8_t* dst = max1_out + static_cast<int64_t>(b) * plane;
  for (int i = tid; i < TH * TW; i += THREADS) {
    const int ly = i / TW, lx = i % TW;
    const int gy = oy0 + ly, gx = ox0 + lx;
    if (gy < H && gx < W) {
      const int idx = (HALO + ly) * SW + (HALO + lx);
      dst[static_cast<int64_t>(gy) * W + gx] = (s_m[idx] || (!s_supp[idx] && s_sc[idx] == s_buf[idx])) ? 1 : 0;
    }
  }
}

// Pass 2 for R=4: round 2 (halo 2*4 = 8 -> 32 x 48, 23.0 KB smem) -> out.
__global__ void __launch_bounds__(THREADS)
simple_nms_r4_pass2_kernel(
    const float* __restrict__ scores, const uint8_t* __restrict__ max1_in, float* __restrict__ out, int H, int W) {
  constexpr int R = 4;
  constexpr int HALO = 8;
  constexpr int SH = TH + 2 * HALO;  // 32
  constexpr int SW = TW + 2 * HALO;  // 48
  constexpr int N = SH * SW;

  __shared__ float s_sc[N];
  __shared__ float s_buf[N];
  __shared__ float s_tmp[N];
  __shared__ uint8_t s_m[N];
  __shared__ uint8_t s_mt[N];
  __shared__ uint8_t s_supp[N];

  const int tid = threadIdx.x;
  const int b = blockIdx.z;
  const int oy0 = blockIdx.y * TH;
  const int ox0 = blockIdx.x * TW;
  const int64_t plane = static_cast<int64_t>(H) * W;
  const float* src = scores + static_cast<int64_t>(b) * plane;
  const uint8_t* m1 = max1_in + static_cast<int64_t>(b) * plane;

  for (int i = tid; i < N; i += THREADS) {
    const int ry = i / SW, rx = i % SW;
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    const int64_t off = static_cast<int64_t>(gy) * W + gx;
    s_sc[i] = ok ? src[off] : -INFINITY;
    s_m[i] = ok ? m1[off] : 0;
  }
  __syncthreads();

  sep_or_2d<R, SH, SW, SH - 2 * R, SW - 2 * R, R, R>(s_m, s_mt, s_supp, tid);
  for (int i = tid; i < (SH - 2 * R) * (SW - 2 * R); i += THREADS) {
    const int ry = R + i / (SW - 2 * R), rx = R + i % (SW - 2 * R);
    const int gy = oy0 - HALO + ry, gx = ox0 - HALO + rx;
    const int idx = ry * SW + rx;
    const bool ok = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    s_buf[idx] = ok ? (s_supp[idx] ? 0.f : s_sc[idx]) : -INFINITY;
  }
  __syncthreads();
  sep_max_2d<R, SH, SW, TH, TW, HALO, HALO>(s_buf, s_tmp, s_buf, tid);

  float* dst = out + static_cast<int64_t>(b) * plane;
  for (int i = tid; i < TH * TW; i += THREADS) {
    const int ly = i / TW, lx = i % TW;
    const int gy = oy0 + ly, gx = ox0 + lx;
    if (gy < H && gx < W) {
      const int idx = (HALO + ly) * SW + (HALO + lx);
      const bool keep = s_m[idx] || (!s_supp[idx] && s_sc[idx] == s_buf[idx]);
      dst[static_cast<int64_t>(gy) * W + gx] = keep ? s_sc[idx] : 0.f;
    }
  }
}

extern "C" int launch_simple_nms(
    const void* scores, void* scratch_u8, void* out, int B, int H, int W, int radius, void* stream) {
  dim3 grid((W + TW - 1) / TW, (H + TH - 1) / TH, B);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (radius == 2) {
    simple_nms_r2_kernel<<<grid, THREADS, 0, s>>>(
        static_cast<const float*>(scores), static_cast<float*>(out), H, W);
  } else if (radius == 4) {
    simple_nms_r4_pass1_kernel<<<grid, THREADS, 0, s>>>(
        static_cast<const float*>(scores), static_cast<uint8_t*>(scratch_u8), H, W);
    simple_nms_r4_pass2_kernel<<<grid, THREADS, 0, s>>>(
        static_cast<const float*>(scores), static_cast<const uint8_t*>(scratch_u8), static_cast<float*>(out), H, W);
  } else {
    return -1;
  }
  return static_cast<int>(cudaGetLastError());
}
