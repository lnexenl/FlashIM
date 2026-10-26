// Single-output-channel KxK convolution (stride 1, "same" padding) with FP32 accumulation: the dense heatmap head of
// DISK (80 -> 1, 5x5). Launched through ctypes (no torch headers); see backends/cuda/conv.py.
//
// A block owns a TH x TW output tile and walks the input channels CB at a time through a STAGES-deep cp.async
// pipeline: each stage is the (TH+K-1)-row halo tile of CB channels in the input dtype, copied as 16-byte chunks over
// the chunk-aligned span covering the halo (zero-filled outside the image and past the last channel). Each thread
// computes an R x CX output patch from (R+K-1) rows of CX+K-1 values, read as 2-element words and widened to FP32 in
// registers, so one shared-memory wavefront feeds ~8 FMAs; the FP32 weights of all channels sit in shared memory.
// Requires a row-contiguous input (stride_w == 1) with 16-byte aligned planes and rows, and C <= MAX_C.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int MAX_C = 256;

template <typename T>
struct Elem;
template <>
struct Elem<float> {
  using Word = float2;  // two elements
  __device__ static void widen(Word v, float& a, float& b) { a = v.x; b = v.y; }
};
template <>
struct Elem<__nv_bfloat16> {
  using Word = uint32_t;
  __device__ static void widen(Word v, float& a, float& b) {
    a = __uint_as_float(v << 16);
    b = __uint_as_float(v & 0xffff0000u);
  }
};
template <>
struct Elem<__half> {
  using Word = __half2;
  __device__ static void widen(Word v, float& a, float& b) {
    const float2 f = __half22float2(v);
    a = f.x;
    b = f.y;
  }
};

template <typename T, int K, int R, int CX, int CB, int TX, int TY, int STAGES>
struct Geometry {
  static constexpr int THREADS = TX * TY;
  static constexpr int TW = TX * CX, TH = TY * R, P = K / 2;
  static constexpr int IH = TH + K - 1;
  static constexpr int N = 16 / sizeof(T);                       // elements per 16-byte chunk
  static constexpr int LEAD = (P + N - 1) / N * N;                // smem column 0 is x0 - LEAD
  static constexpr int CHUNKS = LEAD / N + (TW + P + N - 1) / N;  // per row
  static constexpr int PITCH = CHUNKS * N;                        // elements per smem row
  // a thread's reads start at an even column (words are 2-element aligned), SKIP values before its first input
  static constexpr int SKIP = (LEAD - P) % 2;
  static constexpr int WORDS = (SKIP + CX + K - 1 + 1) / 2;
  static constexpr int STAGE_ELEMS = CB * IH * PITCH;
  static constexpr int STAGE_CHUNKS = CB * IH * CHUNKS;  // chunk i lands at smem element i * N
  static constexpr int PER_THREAD = (STAGE_CHUNKS + THREADS - 1) / THREADS;
  static constexpr int KKP = (K * K + 3) / 4 * 4;        // weights per channel, float4-padded
  static constexpr int STAGE_BYTES = STAGES * STAGE_ELEMS * sizeof(T);  // + C * KKP floats of weights
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <typename T, int K, int R, int CX, int CB, int TX, int TY, int STAGES, int MINB>
__global__ void __launch_bounds__(TX * TY, MINB) conv_single_output_kernel(
    const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
    int C, int H, int W, long long sb, long long sc, long long sh, int has_bias) {
  using G = Geometry<T, K, R, CX, CB, TX, TY, STAGES>;
  using E = Elem<T>;
  extern __shared__ float4 smem[];
  T* stages = reinterpret_cast<T*>(smem);
  float* wsh = reinterpret_cast<float*>(stages + STAGES * G::STAGE_ELEMS);

  const int tid = threadIdx.y * TX + threadIdx.x;
  const int x0 = blockIdx.x * G::TW, y0 = blockIdx.y * G::TH;
  const T* base = x + blockIdx.z * sb;
  const int num_stages = (C + CB - 1) / CB;

  // per chunk this thread copies: its offset from the stage's first channel plane, -1 outside the image
  int goff[G::PER_THREAD];
#pragma unroll
  for (int i = 0; i < G::PER_THREAD; ++i) {
    const int idx = tid + i * G::THREADS;
    const int row_id = idx / G::CHUNKS, q = idx - row_id * G::CHUNKS;
    const int cb = row_id / G::IH, row = row_id - cb * G::IH;
    const int gy = y0 - G::P + row, gx = x0 - G::LEAD + q * G::N;
    goff[i] = idx < G::STAGE_CHUNKS && gy >= 0 && gy < H && gx >= 0 && gx < W ? cb * sc + gy * sh + gx : -1;
  }
  auto issue = [&](int stage) {
    if (stage < num_stages) {
      T* dst = stages + (stage % STAGES) * G::STAGE_ELEMS;
      const T* src = base + stage * CB * sc;
      const int channels = C - stage * CB;
#pragma unroll
      for (int i = 0; i < G::PER_THREAD; ++i) {
        const int idx = tid + i * G::THREADS;
        if (G::STAGE_CHUNKS % G::THREADS && idx >= G::STAGE_CHUNKS) continue;
        bool valid = goff[i] >= 0;
        if (channels < CB) valid = valid && idx / (G::CHUNKS * G::IH) < channels;  // last, partial stage
        cp_async16(dst + idx * G::N, valid ? src + goff[i] : base, valid);
      }
    }
    cp_async_commit();  // possibly empty: keeps the group count uniform
  };

  for (int i = tid; i < C * G::KKP; i += G::THREADS) {
    const int c = i / G::KKP, k = i - c * G::KKP;
    wsh[i] = k < K * K ? w[c * K * K + k] : 0.f;
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) issue(s);

  float acc[R][CX];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < CX; ++j) acc[r][j] = 0.f;

  const int col = threadIdx.x * CX + G::LEAD - G::P - G::SKIP;
  for (int stage = 0; stage < num_stages; ++stage) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    issue(stage + STAGES - 1);  // refills the buffer every thread finished reading before the barrier
    const T* tile = stages + (stage % STAGES) * G::STAGE_ELEMS;
    const int channels = min(CB, C - stage * CB);
#pragma unroll 1
    for (int cb = 0; cb < channels; ++cb) {
      float wr[G::KKP];
      const float4* wv = reinterpret_cast<const float4*>(wsh + (stage * CB + cb) * G::KKP);
#pragma unroll
      for (int i = 0; i < G::KKP / 4; ++i) {
        const float4 q = wv[i];
        wr[4 * i] = q.x;
        wr[4 * i + 1] = q.y;
        wr[4 * i + 2] = q.z;
        wr[4 * i + 3] = q.w;
      }
      const T* plane = tile + (cb * G::IH + threadIdx.y * R) * G::PITCH + col;
#pragma unroll
      for (int iy = 0; iy < R + K - 1; ++iy) {
        float in[2 * G::WORDS];
        const typename E::Word* src = reinterpret_cast<const typename E::Word*>(plane + iy * G::PITCH);
#pragma unroll
        for (int v = 0; v < G::WORDS; ++v) E::widen(src[v], in[2 * v], in[2 * v + 1]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int ky = iy - r;
          if (ky < 0 || ky >= K) continue;
#pragma unroll
          for (int kx = 0; kx < K; ++kx)
#pragma unroll
            for (int j = 0; j < CX; ++j) acc[r][j] = fmaf(in[G::SKIP + j + kx], wr[ky * K + kx], acc[r][j]);
        }
      }
    }
  }
  cp_async_wait<0>();
  const float b = has_bias ? bias[0] : 0.f;
  float* o = out + (long long)blockIdx.z * H * W;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int gy = y0 + threadIdx.y * R + r;
    const int gx = x0 + threadIdx.x * CX;
    if (gy >= H) continue;
    if (gx + CX <= W && W % 4 == 0) {
#pragma unroll
      for (int j = 0; j < CX; j += 4)
        *reinterpret_cast<float4*>(o + (long long)gy * W + gx + j) =
            make_float4(acc[r][j] + b, acc[r][j + 1] + b, acc[r][j + 2] + b, acc[r][j + 3] + b);
    } else {
#pragma unroll
      for (int j = 0; j < CX; ++j)
        if (gx + j < W) o[(long long)gy * W + gx + j] = acc[r][j] + b;
    }
  }
}

// 8 x 4 outputs per thread, 32 x 4 threads (128 x 32 tiles), 2 channels x 3 stages: the best of 7 tilings at DISK's
// head shape on both Thor (sm_110) and an L20 (sm_89); 71 KB of shared memory for K = 5 at 80 bf16 channels. FP32
// inputs stage one channel at a time (2 would need 117 KB: over the 99 KB per block of sm_86 / sm_89).
template <typename T, int K>
int launch(const void* x, const float* w, const float* bias, float* out, int B, int C, int H, int W, long long sb,
           long long sc, long long sh, cudaStream_t stream) {
  constexpr int R = 8, CX = 4, CB = sizeof(T) == 4 ? 1 : 2, TX = 32, TY = 4, STAGES = 3, MINB = 2;
  using G = Geometry<T, K, R, CX, CB, TX, TY, STAGES>;
  auto kernel = conv_single_output_kernel<T, K, R, CX, CB, TX, TY, STAGES, MINB>;
  static bool configured = false;  // one attribute call per instantiation
  if (!configured) {
    const cudaError_t error =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::STAGE_BYTES + MAX_C * G::KKP * 4);
    if (error != cudaSuccess) return static_cast<int>(error);
    configured = true;
  }
  dim3 grid((W + G::TW - 1) / G::TW, (H + G::TH - 1) / G::TH, B);
  kernel<<<grid, dim3(TX, TY), G::STAGE_BYTES + C * G::KKP * 4, stream>>>(static_cast<const T*>(x), w, bias, out, C,
                                                                          H, W, sb, sc, sh, bias != nullptr);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int launch_k(int k, const void* x, const float* w, const float* bias, float* out, int B, int C, int H, int W,
             long long sb, long long sc, long long sh, cudaStream_t s) {
  if (k == 5) return launch<T, 5>(x, w, bias, out, B, C, H, W, sb, sc, sh, s);
  if (k == 3) return launch<T, 3>(x, w, bias, out, B, C, H, W, sb, sc, sh, s);
  return -1;
}

}  // namespace

// dtype: 0 float32, 1 bfloat16, 2 float16; x strides in elements; w: [C, k, k] FP32; bias: 1 FP32 or null;
// out: [B, H, W] FP32. Returns a cudaError_t, or -1 for an unsupported k / layout.
extern "C" int launch_conv_single_output(const void* x, const float* w, const float* bias, float* out, int dtype,
                                         int k, int B, int C, int H, int W, long long sb, long long sc, long long sh,
                                         long long sw, void* stream) {
  const int elem = dtype == 0 ? 4 : 2;
  if (sw != 1 || C > MAX_C || (reinterpret_cast<uintptr_t>(x) % 16) || (sh * elem) % 16 || (sc * elem) % 16 ||
      (sb * elem) % 16)
    return -1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (dtype) {
    case 0: return launch_k<float>(k, x, w, bias, out, B, C, H, W, sb, sc, sh, s);
    case 1: return launch_k<__nv_bfloat16>(k, x, w, bias, out, B, C, H, W, sb, sc, sh, s);
    case 2: return launch_k<__half>(k, x, w, bias, out, B, C, H, W, sb, sc, sh, s);
  }
  return -1;
}
