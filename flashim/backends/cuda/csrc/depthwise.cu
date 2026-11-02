// Depthwise 5x5 (pad 2, stride 1) kernels for channels-last BF16 maps, launched through ctypes (no torch headers); see
// backends/cuda/conv.py. Both kernels do the arithmetic of the Triton kernels they replace (backends/triton/conv.py)
// in the same order, so their outputs are bit-identical to them:
//
//  * dw5x5_relu_int8: depthwise + bias + ReLU -> BF16 rounding -> static INT8, the producer of a packed INT8 1x1
//    (ops.depthwise5x5_relu_int8). x [B, H, W, C] bf16, w [25, C] bf16 (tap-major), bias [C] f32, scale [1] f32 ->
//    out [B * H * W, C] int8, C % 64 == 0.
//  * dwpw: depthwise + bias + ReLU -> BF16 -> pointwise C x C on tensor cores (mma.sync m16n8k16, FP32 accumulate) +
//    bias -> BF16 (ops.depthwise5x5_relu_pointwise). C in {32, 64, 128}.
//
// Per output the depthwise is an FP32 FMA chain from zero in ky-major, kx-minor order, then + bias. A thread owns two
// channels (one bf16x2 word) of COLS output columns and rolls down TH output rows: each input row is loaded once per
// thread and feeds the five accumulator rows that use it (the Triton kernels' row rolling), the 25 x 2 weights live in
// registers, the loop is fully unrolled so the accumulator rotation is register renaming, and a CTA whose input
// footprint lies inside the image runs a predicate-free copy of the body.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// -- INT8 producer ----------------------------------------------------------------------------------------------

// C == 0: the channel count is a run-time argument (any C % 64 == 0); otherwise it is a template constant, which turns
// the per-pixel offsets into load immediates.
template <int COLS, int TH, int C, bool INTERIOR>
__device__ __forceinline__ void producer_body(const unsigned int* __restrict__ xb, int8_t* __restrict__ ob,
                                              const float (&wt)[25][2], float b0, float b1, float inv, int H, int W,
                                              int channels, int x0, int y0) {
  const int cw = (C ? C : channels) / 2;  // words per pixel
  const int row_words = W * cw;
  bool col_ok[COLS + 4];
#pragma unroll
  for (int j = 0; j < COLS + 4; ++j) col_ok[j] = INTERIOR || (x0 - 2 + j >= 0 && x0 - 2 + j < W);
  // first input row y0 - 2, first column x0 - 2 (may point outside the image: only dereferenced when in bounds)
  const unsigned int* row = xb + (ptrdiff_t)(y0 - 2) * row_words + (ptrdiff_t)(x0 - 2) * cw;
  int8_t* orow = ob + ((size_t)y0 * W + x0) * (C ? C : channels);

  float acc[5][COLS][2];
#pragma unroll
  for (int s = 0; s < 5; ++s)
#pragma unroll
    for (int j = 0; j < COLS; ++j) acc[s][j][0] = acc[s][j][1] = 0.0f;

#pragma unroll
  for (int step = 0; step < TH + 4; ++step) {
    const int iy = y0 - 2 + step;
    const bool row_ok = INTERIOR || (iy >= 0 && iy < H);
    float2 v[COLS + 4];
#pragma unroll
    for (int j = 0; j < COLS + 4; ++j) {
      unsigned int bits = 0u;
      if (INTERIOR || (row_ok && col_ok[j])) bits = __ldg(row + j * cw);
      v[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
    }
    row += row_words;
    // acc[s] is output row iy - 2 + s, which takes this input row as tap ky = 4 - s (acc0 .. acc4 of the Triton kernel)
#pragma unroll
    for (int s = 0; s < 5; ++s) {
      const int ky = 4 - s;
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
          acc[s][j][0] = fmaf(v[j + kx].x, wt[ky * 5 + kx][0], acc[s][j][0]);
          acc[s][j][1] = fmaf(v[j + kx].y, wt[ky * 5 + kx][1], acc[s][j][1]);
        }
      }
    }
    if (step >= 4 && (INTERIOR || iy - 2 < H)) {
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        if (INTERIOR || x0 + j < W) {
          int q[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            float a = fmaxf(acc[0][j][k] + (k ? b1 : b0), 0.0f);
            a = __bfloat162float(__float2bfloat16_rn(a));
            // rint(a * (1 / scale)) clamped to +-127 as one convert: a >= 0 after the ReLU, so saturating at 127 is
            // the clamp and -127 cannot be reached
            asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(q[k]) : "f"(a * inv));
          }
          *reinterpret_cast<unsigned short*>(orow + j * (C ? C : channels)) =
              (unsigned short)((q[0] & 0xff) | ((q[1] & 0xff) << 8));
        }
      }
      orow += (size_t)W * (C ? C : channels);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        acc[s][j][0] = acc[s + 1][j][0];
        acc[s][j][1] = acc[s + 1][j][1];
      }
#pragma unroll
    for (int j = 0; j < COLS; ++j) acc[4][j][0] = acc[4][j][1] = 0.0f;
  }
}

// Grid (C / 64, x tiles, B * y tiles), block (32, CY): a warp covers 64 contiguous channels (128 B loads, 64 B stores
// per pixel), warp y of a block takes COLS adjacent columns.
template <int COLS, int TH, int C>
__global__ void producer_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                const float* __restrict__ bias, const float* __restrict__ scale,
                                int8_t* __restrict__ out, int H, int W, int channels) {
  const int ch = C ? C : channels;
  const int c = (blockIdx.x * 32 + threadIdx.x) * 2;
  const int x0 = (blockIdx.y * blockDim.y + threadIdx.y) * COLS;
  const int ytiles = (H + TH - 1) / TH;
  const int b = blockIdx.z / ytiles;
  const int y0 = (blockIdx.z % ytiles) * TH;
  if (x0 >= W) return;
  float wt[25][2];
#pragma unroll
  for (int t = 0; t < 25; ++t) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + t * ch + c));
    wt[t][0] = f.x;
    wt[t][1] = f.y;
  }
  const float b0 = bias[c], b1 = bias[c + 1];
  // 1 / scale as the Triton / TileLang kernels compute it: div.full.f32 (not IEEE div.rn: x * (1 / s) then differs at
  // exact .5 ties, e.g. 3.375 / 0.03 -> 112 vs 113)
  float inv;
  asm("div.full.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(scale[0]));
  const size_t image = (size_t)b * H * W * ch;
  const unsigned int* xb = reinterpret_cast<const unsigned int*>(x + image + c);
  int8_t* ob = out + image + c;
  // the CTA's input footprint: rows y0 - 2 .. y0 + TH + 1, columns x_cta - 2 .. x_cta + COLS * blockDim.y + 1
  const int xc = blockIdx.y * blockDim.y * COLS;
  const bool interior = y0 >= 2 && y0 + TH + 2 <= H && xc >= 2 && xc + COLS * (int)blockDim.y + 2 <= W;
  if (interior)
    producer_body<COLS, TH, C, true>(xb, ob, wt, b0, b1, inv, H, W, ch, x0, y0);
  else
    producer_body<COLS, TH, C, false>(xb, ob, wt, b0, b1, inv, H, W, ch, x0, y0);
}

template <int COLS, int TH, int C>
int launch_producer_c(const void* x, const void* w, const float* bias, const float* scale, void* out, int B, int H,
                      int W, int channels, int cy, cudaStream_t stream) {
  const dim3 block(32, cy);
  const dim3 grid((C ? C : channels) / 64, (W + COLS * cy - 1) / (COLS * cy), B * ((H + TH - 1) / TH));
  producer_kernel<COLS, TH, C><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w), bias, scale,
      static_cast<int8_t*>(out), H, W, channels);
  return (int)cudaGetLastError();
}

template <int COLS, int TH>
int launch_producer(const void* x, const void* w, const float* bias, const float* scale, void* out, int B, int H, int W,
                    int C, int cy, cudaStream_t stream) {
  switch (C) {
    case 64: return launch_producer_c<COLS, TH, 64>(x, w, bias, scale, out, B, H, W, C, cy, stream);
    case 128: return launch_producer_c<COLS, TH, 128>(x, w, bias, scale, out, B, H, W, C, cy, stream);
    case 256: return launch_producer_c<COLS, TH, 256>(x, w, bias, scale, out, B, H, W, C, cy, stream);
    case 512: return launch_producer_c<COLS, TH, 512>(x, w, bias, scale, out, B, H, W, C, cy, stream);
    case 768: return launch_producer_c<COLS, TH, 768>(x, w, bias, scale, out, B, H, W, C, cy, stream);
  }
  return launch_producer_c<COLS, TH, 0>(x, w, bias, scale, out, B, H, W, C, cy, stream);  // other C % 64 == 0
}

// -- BF16 depthwise + pointwise ---------------------------------------------------------------------------------

template <int C, int NW, int MT>
struct WarpSplit {
  static constexpr int NT = C / 8;
  // warps along N: as many as divide NT with >= 1 n-tile each, the rest along M
  static constexpr int pick() {
    for (int wn = NW; wn >= 1; --wn)
      if (NW % wn == 0 && NT % wn == 0 && MT % (NW / wn) == 0) return wn;
    return -1;
  }
  static constexpr int WN = pick();
  static constexpr int WM = NW / WN;
  static constexpr int NTW = NT / WN;
  static constexpr int MTW = MT / WM;
};

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// CTA: PX = NG * COLS consecutive output pixels of one image row strip, rolling down TH output rows. Thread t computes
// channels 2 * (t % (C / 2)) + {0, 1} of the COLS pixels (t / (C / 2)) * COLS ... For each finished output row the
// hidden [PX, C] tile goes to shared memory (double buffered, rows padded by 16 B: conflict-free ldmatrix), one
// __syncthreads, then each warp multiplies its (m, n) tiles: A by ldmatrix.x4, B (the warp's slice of W) kept in
// registers for the whole CTA. D = bias + hidden @ W^T is one mma chain over k = 0, 16, ... starting from the broadcast
// bias: Triton folds ``tl.dot(a, b) + bias`` into ``tt.dot(a, b, acc=bias)`` (checked in the TTGIR; accumulating from
// zero and adding the bias afterwards differs from it in ~1e-5 of the outputs).
template <int C, int COLS, int TH, int NG, bool INTERIOR>
__device__ __forceinline__ void dwpw_body(const unsigned* __restrict__ xb, __nv_bfloat16* __restrict__ ob,
                                          const float (&wt)[25][2], float b0, float b1,
                                          const unsigned (&bw)[WarpSplit<C, C / 2 * NG / 32, NG * COLS / 16>::NTW]
                                                               [C / 16][2],
                                          const float (&pb)[WarpSplit<C, C / 2 * NG / 32, NG * COLS / 16>::NTW][2],
                                          __nv_bfloat16* smem, int H, int W, int xc, int y0) {
  constexpr int CW = C / 2;  // words per pixel
  constexpr int PX = NG * COLS;
  constexpr int NW = CW * NG / 32;
  constexpr int MT = PX / 16;
  using S = WarpSplit<C, NW, MT>;
  constexpr int SROW = C + 8;  // bf16 per smem row
  constexpr int KT = C / 16;
  const int tid = threadIdx.x;
  const int cp = tid % CW, g = tid / CW;
  const int x0 = xc + g * COLS;
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp % S::WN, wm = warp / S::WN;

  const int row_words = W * CW;
  bool col_ok[COLS + 4];
#pragma unroll
  for (int j = 0; j < COLS + 4; ++j) col_ok[j] = INTERIOR || (x0 - 2 + j >= 0 && x0 - 2 + j < W);
  const unsigned* row = xb + (ptrdiff_t)(y0 - 2) * row_words + (ptrdiff_t)(x0 - 2) * CW + cp;

  float acc[5][COLS][2];
#pragma unroll
  for (int s = 0; s < 5; ++s)
#pragma unroll
    for (int j = 0; j < COLS; ++j) acc[s][j][0] = acc[s][j][1] = 0.0f;

#pragma unroll
  for (int step = 0; step < TH + 4; ++step) {
    const int iy = y0 - 2 + step;
    const bool row_ok = INTERIOR || (iy >= 0 && iy < H);
    float2 v[COLS + 4];
#pragma unroll
    for (int j = 0; j < COLS + 4; ++j) {
      unsigned raw = 0u;
      if (INTERIOR || (row_ok && col_ok[j])) raw = __ldg(row + j * CW);
      v[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
    }
    row += row_words;
#pragma unroll
    for (int s = 0; s < 5; ++s) {
      const int ky = 4 - s;
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
          acc[s][j][0] = fmaf(v[j + kx].x, wt[ky * 5 + kx][0], acc[s][j][0]);
          acc[s][j][1] = fmaf(v[j + kx].y, wt[ky * 5 + kx][1], acc[s][j][1]);
        }
      }
    }
    const int oy = iy - 2;
    if (step >= 4 && (INTERIOR || oy < H)) {  // CTA-uniform
      __nv_bfloat16* hid = smem + (step & 1) * PX * SROW;
#pragma unroll
      for (int j = 0; j < COLS; ++j)
        *reinterpret_cast<unsigned*>(hid + (g * COLS + j) * SROW + 2 * cp) =
            pack_bf16(fmaxf(acc[0][j][0] + b0, 0.0f), fmaxf(acc[0][j][1] + b1, 0.0f));
      __syncthreads();
      float d[S::MTW][S::NTW][4];
#pragma unroll
      for (int mi = 0; mi < S::MTW; ++mi)
#pragma unroll
        for (int ni = 0; ni < S::NTW; ++ni) {
          d[mi][ni][0] = d[mi][ni][2] = pb[ni][0];
          d[mi][ni][1] = d[mi][ni][3] = pb[ni][1];
        }
#pragma unroll
      for (int k = 0; k < KT; ++k) {
#pragma unroll
        for (int mi = 0; mi < S::MTW; ++mi) {
          const int m0 = (wm * S::MTW + mi) * 16;
          unsigned a[4];
          ldmatrix_x4(a, hid + (m0 + lane % 16) * SROW + k * 16 + (lane / 16) * 8);
#pragma unroll
          for (int ni = 0; ni < S::NTW; ++ni) mma_bf16(d[mi][ni], a, bw[ni][k][0], bw[ni][k][1]);
        }
      }
      // epilogue: BF16 straight from the accumulator fragments (staging through shared memory for 16 B stores
      // measured slower: L2 already merges the 4 B fragment stores)
      const int gq = lane / 4, tq = lane % 4;
      __nv_bfloat16* orow = ob + ((size_t)oy * W + xc) * C;
#pragma unroll
      for (int mi = 0; mi < S::MTW; ++mi) {
        const int m0 = (wm * S::MTW + mi) * 16;
#pragma unroll
        for (int ni = 0; ni < S::NTW; ++ni) {
          const int n = (wn * S::NTW + ni) * 8 + 2 * tq;
          const unsigned lo = pack_bf16(d[mi][ni][0], d[mi][ni][1]);
          const unsigned hi = pack_bf16(d[mi][ni][2], d[mi][ni][3]);
          if (INTERIOR || xc + m0 + gq < W) *reinterpret_cast<unsigned*>(orow + (m0 + gq) * C + n) = lo;
          if (INTERIOR || xc + m0 + gq + 8 < W) *reinterpret_cast<unsigned*>(orow + (m0 + gq + 8) * C + n) = hi;
        }
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int j = 0; j < COLS; ++j) {
        acc[s][j][0] = acc[s + 1][j][0];
        acc[s][j][1] = acc[s + 1][j][1];
      }
#pragma unroll
    for (int j = 0; j < COLS; ++j) acc[4][j][0] = acc[4][j][1] = 0.0f;
  }
}

// __launch_bounds__ asks for 16 resident warps per SM (<= 128 registers at 64 / 128 threads): ptxas spills a little
// (32 ch 144 B, 64 ch 208 B, 128 ch with 256 threads 1.1 KB on sm_89) and it measured faster on an H20 (LoMa 32 / 64 ch:
// 1.32 -> 1.10 ms per extract, 12 -> 16 warps)
template <int C, int COLS, int TH, int NG>
__global__ void __launch_bounds__(C / 2 * NG, 512 / (C / 2 * NG) > 0 ? 512 / (C / 2 * NG) : 1)
    dwpw_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ dw_w,
                const __nv_bfloat16* __restrict__ dw_b, const __nv_bfloat16* __restrict__ pw_w,
                const __nv_bfloat16* __restrict__ pw_b, __nv_bfloat16* __restrict__ out, int H, int W) {
  constexpr int CW = C / 2;
  constexpr int PX = NG * COLS;
  constexpr int NW = CW * NG / 32;
  constexpr int MT = PX / 16;
  using S = WarpSplit<C, NW, MT>;
  static_assert(CW * NG % 32 == 0 && PX % 16 == 0 && S::WN > 0, "tile");
  extern __shared__ __align__(16) __nv_bfloat16 smem[];

  const int tid = threadIdx.x;
  const int cp = tid % CW;
  const int c = 2 * cp;
  const int xc = blockIdx.x * PX;
  const int ytiles = (H + TH - 1) / TH;
  const int b = blockIdx.y / ytiles;
  const int y0 = (blockIdx.y % ytiles) * TH;

  float wt[25][2];
#pragma unroll
  for (int t = 0; t < 25; ++t) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(dw_w + t * C + c));
    wt[t][0] = f.x;
    wt[t][1] = f.y;
  }
  const float b0 = __bfloat162float(dw_b[c]), b1 = __bfloat162float(dw_b[c + 1]);

  // the warp's B fragments (W[n][k], row-major [C_out, C_in] = "col" B of m16n8k16) and pointwise bias
  const int warp = tid / 32, lane = tid % 32;
  const int wn = warp % S::WN;
  const int gq = lane / 4, tq = lane % 4;
  unsigned bw[S::NTW][C / 16][2];
  float pb[S::NTW][2];
#pragma unroll
  for (int ni = 0; ni < S::NTW; ++ni) {
    const int n0 = (wn * S::NTW + ni) * 8;
    const unsigned* wrow = reinterpret_cast<const unsigned*>(pw_w + (n0 + gq) * C);
#pragma unroll
    for (int k = 0; k < C / 16; ++k) {
      bw[ni][k][0] = __ldg(wrow + k * 8 + tq);
      bw[ni][k][1] = __ldg(wrow + k * 8 + 4 + tq);
    }
    pb[ni][0] = __bfloat162float(pw_b[n0 + 2 * tq]);
    pb[ni][1] = __bfloat162float(pw_b[n0 + 2 * tq + 1]);
  }

  const size_t image = (size_t)b * H * W * C;
  const unsigned* xb = reinterpret_cast<const unsigned*>(x + image);
  __nv_bfloat16* ob = out + image;
  const bool interior = y0 >= 2 && y0 + TH + 2 <= H && xc >= 2 && xc + PX + 2 <= W;
  if (interior)
    dwpw_body<C, COLS, TH, NG, true>(xb, ob, wt, b0, b1, bw, pb, smem, H, W, xc, y0);
  else
    dwpw_body<C, COLS, TH, NG, false>(xb, ob, wt, b0, b1, bw, pb, smem, H, W, xc, y0);
}

template <int C, int COLS, int TH, int NG>
constexpr bool valid_dwpw_tile() {
  return (C / 2 * NG) % 32 == 0 && (COLS * NG) % 16 == 0 && C / 2 * NG <= 512 &&
         WarpSplit<C, ((C / 2 * NG) / 32 > 0 ? (C / 2 * NG) / 32 : 1), ((COLS * NG) / 16 > 0 ? (COLS * NG) / 16 : 1)>::WN > 0;
}

template <int C, int COLS, int TH, int NG>
int launch_dwpw_t(const void* x, const void* dw_w, const void* dw_b, const void* pw_w, const void* pw_b, void* out,
                  int B, int H, int W, cudaStream_t stream) {
  if constexpr (valid_dwpw_tile<C, COLS, TH, NG>()) {
    constexpr int PX = NG * COLS;
    const int smem = 2 * PX * (C + 8) * 2;
    const dim3 grid((W + PX - 1) / PX, B * ((H + TH - 1) / TH));
    dwpw_kernel<C, COLS, TH, NG><<<grid, C / 2 * NG, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(dw_w),
        static_cast<const __nv_bfloat16*>(dw_b), static_cast<const __nv_bfloat16*>(pw_w),
        static_cast<const __nv_bfloat16*>(pw_b), static_cast<__nv_bfloat16*>(out), H, W);
    return (int)cudaGetLastError();
  } else {
    return 99;
  }
}

}  // namespace

// C % 64 == 0 (any, see launch_producer); returns a cudaError_t code (99: unsupported tile). The one (COLS, TH) pair
// is what the tile sweeps picked on L20 and H20 (see backends/cuda/conv.py); CY is a run-time argument.
extern "C" int launch_dw5x5_relu_int8(const void* x, const void* w, const float* bias, const float* scale, void* out,
                                      int B, int H, int W, int C, int cols, int th, int cy, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (cols == 4 && th == 8) return launch_producer<4, 8>(x, w, bias, scale, out, B, H, W, C, cy, s);
  return 99;
}

// BF16 depthwise + pointwise, C in {32, 64, 128}, tiles (COLS, TH, NG) = (4, 4, 4), (4, 8, 4), (4, 8, 8) (the ones that
// won the sweeps); returns a cudaError_t code (99: unsupported C / tile)
extern "C" int launch_dwpw(const void* x, const void* dw_w, const void* dw_b, const void* pw_w, const void* pw_b,
                           void* out, int B, int H, int W, int C, int cols, int th, int ng, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
#define DWPW_TILE(CC, T, G)                                                                       \
  if (C == CC && cols == 4 && th == T && ng == G)                                                 \
    return launch_dwpw_t<CC, 4, T, G>(x, dw_w, dw_b, pw_w, pw_b, out, B, H, W, s);
#define DWPW_C(CC) DWPW_TILE(CC, 4, 4) DWPW_TILE(CC, 8, 4) DWPW_TILE(CC, 8, 8)
  DWPW_C(32) DWPW_C(64) DWPW_C(128)
#undef DWPW_C
#undef DWPW_TILE
  return 99;
}
