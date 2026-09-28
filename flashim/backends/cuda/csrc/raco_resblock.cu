// Fused RaCo residual block: selu(conv3x3(selu(conv3x3(x)+b1)) + conv1x1(x) + b2), BatchNorm folded, <=16 channels.
// Implicit GEMM with mma.sync.m16n8k16 (bf16 -> fp32): the A-fragments are ldmatrix'ed straight from the NHWC input /
// intermediate tile in shared memory (no im2col), the intermediate never leaves the SM.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef TILE_H
#define TILE_H 14
#endif
constexpr int TH = TILE_H, TW = 14;  // output tile
constexpr int IH = TH + 4, IW = 18;  // input tile (2px halo); IW = 16 lanes + 2
constexpr int RH = TH + 2;           // conv1 region rows (16 lanes wide)
constexpr int MID_PIX = RH * 16 + 4; // + slack: invalid lanes of the last row read up to 2 px past the region
constexpr int NTAPS = 19;            // 9 conv1 + 9 conv2 + 1 shortcut
constexpr int THREADS = 256;
constexpr int RPW = (RH + 7) / 8;    // region / output rows per warp

constexpr int W_BYTES = NTAPS * 16 * 16 * 2;
constexpr int IN_BYTES = IH * IW * 32;
constexpr int MID_BYTES = MID_PIX * 32;
constexpr int SMEM_BYTES = W_BYTES + IN_BYTES + MID_BYTES;

// XOR swizzle of a 32 B record's two 16 B halves by bit 2 of its row/pixel index: 8 consecutive ldmatrix rows hit distinct banks
__device__ __forceinline__ int sw(int row, int half) { return half ^ ((row >> 2) & 1); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float selu(float v) {
  return 1.0507009873554805f * (v > 0.f ? v : 1.6732632423543772f * (__expf(v) - 1.f));
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int IN_CH>
__global__ void __launch_bounds__(THREADS)
resblock_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ wpack,
                const float* __restrict__ bias, int H, int W, int tiles_x, int num_tiles) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* w_s = smem;
  unsigned char* in_s = smem + W_BYTES;
  unsigned char* mid_s = in_s + IN_BYTES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  for (int i = tid; i < W_BYTES / 16; i += THREADS)  // [tap][k][n]: 32 B rows (k), halves swizzled by k
    reinterpret_cast<uint4*>(w_s)[(i & ~1) | sw((i >> 1) & 15, i & 1)] = reinterpret_cast<const uint4*>(wpack)[i];
  for (int i = tid; i < (MID_PIX * 32 - RH * 16 * 32) / 16; i += THREADS)  // zero the slack rows once
    reinterpret_cast<uint4*>(mid_s + RH * 16 * 32)[i] = make_uint4(0, 0, 0, 0);

  // per-thread accumulator columns: channels {2*(lane%4), +1} of each 8-wide n-tile
  const int cq = (lane & 3) * 2, g = lane >> 2;
  float b1v[4], b2v[4];
  for (int t = 0; t < 2; ++t) for (int j = 0; j < 2; ++j) {
    b1v[t * 2 + j] = bias[t * 8 + cq + j];
    b2v[t * 2 + j] = bias[16 + t * 8 + cq + j];
  }
  // ldmatrix lane addressing
  const int m = lane >> 3, r8 = lane & 7;
  const int a_row = r8 + (m & 1) * 8, a_half = m >> 1;                    // A: 16 pixels x 16 channels
  const int b_k = (m & 1) * 8 + r8, b_half = m >> 1;                      // B (trans): k row, n half
  __syncthreads();

  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const int oy0 = (tile / tiles_x) * TH, ox0 = (tile % tiles_x) * TW;
    __syncthreads();  // previous tile is done with in_s / mid_s
    // -- input tile -> smem (32 B / pixel, zero padded channels and out-of-image pixels) --
    for (int p = tid; p < IH * IW; p += THREADS) {
      const int iy = oy0 - 2 + p / IW, ix = ox0 - 2 + p % IW;
      uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);
      if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        const uint4* src = reinterpret_cast<const uint4*>(x + ((size_t)iy * W + ix) * IN_CH);
        if (IN_CH == 16) { v0 = src[0]; v1 = src[1]; }
        else { const uint2 s = *reinterpret_cast<const uint2*>(src); v0 = make_uint4(s.x, s.y, 0, 0); }
      }
      uint4* dst = reinterpret_cast<uint4*>(in_s + p * 32);
      dst[sw(p, 0)] = v0; dst[sw(p, 1)] = v1;
    }
    __syncthreads();

    // -- conv1 + SELU over the (TH+2) x 16 region: warp w owns region rows w and w+8 --
    float acc[RPW][2][4];
    for (int i = 0; i < RPW; ++i) for (int t = 0; t < 2; ++t) for (int j = 0; j < 4; ++j) acc[i][t][j] = 0.f;
    for (int tap = 0; tap < 9; ++tap) {
      const int ky = tap / 3, kx = tap % 3;
      uint32_t bfr[4];
      ldsm_x4_t(bfr, smem_u32(w_s + tap * 512 + b_k * 32 + sw(b_k, b_half) * 16));
      for (int i = 0; i < RPW; ++i) {
        const int ry = warp + 8 * i;
        uint32_t afr[4];
        { const int pp = (ry + ky) * IW + kx + a_row; ldsm_x4(afr, smem_u32(in_s + pp * 32 + sw(pp, a_half) * 16)); }
        mma16816(acc[i][0], afr, bfr[0], bfr[1]);
        mma16816(acc[i][1], afr, bfr[2], bfr[3]);
      }
    }
    for (int i = 0; i < RPW; ++i) {
      const int ry = warp + 8 * i, py = oy0 - 1 + ry;
      for (int h = 0; h < 2; ++h) {  // rows g and g+8 of the fragment
        const int rx = g + 8 * h, px = ox0 - 1 + rx;
        const bool inside = py >= 0 && py < H && px >= 0 && px < W;
        for (int t = 0; t < 2; ++t) {
          const float v0 = inside ? selu(acc[i][t][2 * h] + b1v[t * 2]) : 0.f;
          const float v1 = inside ? selu(acc[i][t][2 * h + 1] + b1v[t * 2 + 1]) : 0.f;
          { const int pp = ry * 16 + rx; *reinterpret_cast<uint32_t*>(mid_s + pp * 32 + sw(pp, t) * 16 + cq * 2) = pack_bf16(v0, v1); }
        }
      }
    }
    __syncthreads();

    // -- conv2 + 1x1 shortcut + SELU: warp w owns output rows w and w+8 (< TH) --
    for (int i = 0; i < RPW; ++i) for (int t = 0; t < 2; ++t) for (int j = 0; j < 4; ++j) acc[i][t][j] = 0.f;
    for (int tap = 0; tap < 10; ++tap) {
      uint32_t bfr[4];
      ldsm_x4_t(bfr, smem_u32(w_s + (9 + tap) * 512 + b_k * 32 + sw(b_k, b_half) * 16));
      for (int i = 0; i < RPW; ++i) {
        const int ty = warp + 8 * i;
        if (ty < TH) {
          uint32_t afr[4];
          if (tap < 9) {
            const int ky = tap / 3, kx = tap % 3;
            { const int pp = (ty + ky) * 16 + kx + a_row; ldsm_x4(afr, smem_u32(mid_s + pp * 32 + sw(pp, a_half) * 16)); }
          } else {
            { const int pp = (ty + 2) * IW + 2 + a_row; ldsm_x4(afr, smem_u32(in_s + pp * 32 + sw(pp, a_half) * 16)); }
          }
          mma16816(acc[i][0], afr, bfr[0], bfr[1]);
          mma16816(acc[i][1], afr, bfr[2], bfr[3]);
        }
      }
    }
    for (int i = 0; i < RPW; ++i) {
      const int ty = warp + 8 * i, oy = oy0 + ty;
      if (ty >= TH || oy >= H) continue;
      for (int h = 0; h < 2; ++h) {
        const int tx = g + 8 * h, ox = ox0 + tx;
        if (tx >= TW || ox >= W) continue;
        for (int t = 0; t < 2; ++t) {
          const float v0 = selu(acc[i][t][2 * h] + b2v[t * 2]);
          const float v1 = selu(acc[i][t][2 * h + 1] + b2v[t * 2 + 1]);
          *reinterpret_cast<uint32_t*>(out + ((size_t)oy * W + ox) * 16 + t * 8 + cq) = pack_bf16(v0, v1);
        }
      }
    }
  }
}

extern "C" int launch_resblock(const void* x, void* out, const void* wpack, const void* bias, int H, int W, int in_ch, int grid,
                               void* stream) {
  const int tiles_x = (W + TW - 1) / TW, tiles_y = (H + TH - 1) / TH, num_tiles = tiles_x * tiles_y;
  if (grid > num_tiles) grid = num_tiles;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  auto xp = static_cast<const __nv_bfloat16*>(x);
  auto op = static_cast<__nv_bfloat16*>(out);
  auto wp = static_cast<const __nv_bfloat16*>(wpack);
  auto bp = static_cast<const float*>(bias);
  if (in_ch == 16) resblock_kernel<16><<<grid, THREADS, SMEM_BYTES, s>>>(xp, op, wp, bp, H, W, tiles_x, num_tiles);
  else if (in_ch == 4) resblock_kernel<4><<<grid, THREADS, SMEM_BYTES, s>>>(xp, op, wp, bp, H, W, tiles_x, num_tiles);
  else return -1;
  return static_cast<int>(cudaGetLastError());
}
