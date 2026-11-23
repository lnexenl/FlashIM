// Exact 3D Euclidean nearest-neighbor search via Morton-sorted complete binary AABB tree (BVH).
// Used by DUSt3R's reciprocal_nn op: 100% mathematically exact (bit-identical to brute-force L2 with
// smallest-original-index tie breaking), near-child-first per-thread traversal with distance-tagged stack.
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

constexpr int LEAF = 16;

struct alignas(16) PtIdx {
  float x, y, z;
  int idx;
};

struct alignas(16) AABB {
  float minx, miny, minz, maxx;
  float maxy, maxz, pad0, pad1;
};

__device__ __forceinline__ uint32_t expand_bits_10(uint32_t v) {
  v = (v | (v << 16)) & 0x030000FFu;
  v = (v | (v << 8)) & 0x0300F00Fu;
  v = (v | (v << 4)) & 0x030C30C3u;
  v = (v | (v << 2)) & 0x09249249u;
  return v;
}

__device__ __forceinline__ float signed_log1p(float v) {
  return v >= 0.0f ? log1pf(v) : -log1pf(-v);
}

__global__ void morton3d_kernel(const float* __restrict__ pts, const float* __restrict__ bounds,
                                int32_t* __restrict__ codes, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float lo_x = __ldg(bounds + 0);
  const float lo_y = __ldg(bounds + 1);
  const float lo_z = __ldg(bounds + 2);
  const float inv_x = __ldg(bounds + 3);
  const float inv_y = __ldg(bounds + 4);
  const float inv_z = __ldg(bounds + 5);
  const int64_t base = static_cast<int64_t>(i) * 3;
  const float ux = signed_log1p(pts[base]);
  const float uy = signed_log1p(pts[base + 1]);
  const float uz = signed_log1p(pts[base + 2]);
  const uint32_t ix = static_cast<uint32_t>(fminf(fmaxf((ux - lo_x) * inv_x, 0.0f), 1023.0f));
  const uint32_t iy = static_cast<uint32_t>(fminf(fmaxf((uy - lo_y) * inv_y, 0.0f), 1023.0f));
  const uint32_t iz = static_cast<uint32_t>(fminf(fmaxf((uz - lo_z) * inv_z, 0.0f), 1023.0f));
  codes[i] = static_cast<int32_t>(expand_bits_10(ix) | (expand_bits_10(iy) << 1) | (expand_bits_10(iz) << 2));
}

__global__ void pack_and_leaf_aabb_kernel(const float* __restrict__ pts, const int64_t* __restrict__ order,
                                          PtIdx* __restrict__ packed, AABB* __restrict__ nodes, int n,
                                          int n_leaves, int P) {
  const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
  if (leaf >= P) return;
  AABB b;
  if (leaf >= n_leaves) {
    b.minx = b.miny = b.minz = FLT_MAX;
    b.maxx = b.maxy = b.maxz = -FLT_MAX;
    b.pad0 = b.pad1 = 0.0f;
    nodes[P + leaf] = b;
    return;
  }
  const int start = leaf * LEAF;
  const int end = min(start + LEAF, n);
  float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
  float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
  for (int j = start; j < end; ++j) {
    const int orig = static_cast<int>(order[j]);
    const int64_t base = static_cast<int64_t>(orig) * 3;
    const float x = pts[base];
    const float y = pts[base + 1];
    const float z = pts[base + 2];
    packed[j] = PtIdx{x, y, z, orig};
    minx = fminf(minx, x);
    miny = fminf(miny, y);
    minz = fminf(minz, z);
    maxx = fmaxf(maxx, x);
    maxy = fmaxf(maxy, y);
    maxz = fmaxf(maxz, z);
  }
  b.minx = minx;
  b.miny = miny;
  b.minz = minz;
  b.maxx = maxx;
  b.maxy = maxy;
  b.maxz = maxz;
  b.pad0 = b.pad1 = 0.0f;
  nodes[P + leaf] = b;
}

__device__ __forceinline__ AABB merge_aabb(const AABB& a, const AABB& b) {
  AABB out;
  out.minx = fminf(a.minx, b.minx);
  out.miny = fminf(a.miny, b.miny);
  out.minz = fminf(a.minz, b.minz);
  out.maxx = fmaxf(a.maxx, b.maxx);
  out.maxy = fmaxf(a.maxy, b.maxy);
  out.maxz = fmaxf(a.maxz, b.maxz);
  out.pad0 = out.pad1 = 0.0f;
  return out;
}

__global__ void build_internal_level_kernel(AABB* __restrict__ nodes, int level_start) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= level_start) return;
  const int node = level_start + idx;
  nodes[node] = merge_aabb(nodes[node << 1], nodes[(node << 1) | 1]);
}

__device__ __forceinline__ float aabb_dist2(const AABB* __restrict__ nodes, int idx, float qx, float qy, float qz) {
  const float4 b0 = __ldg(reinterpret_cast<const float4*>(nodes + idx));
  if (b0.x > b0.w) return INFINITY;
  const float2 b1 = __ldg(reinterpret_cast<const float2*>(&nodes[idx].maxy));
  const float dx = fmaxf(0.0f, fmaxf(b0.x - qx, qx - b0.w));
  const float dy = fmaxf(0.0f, fmaxf(b0.y - qy, qy - b1.x));
  const float dz = fmaxf(0.0f, fmaxf(b0.z - qz, qz - b1.y));
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ bool may_contain(float d2, float best) {
  if (!isfinite(d2)) return false;
  if (!isfinite(best)) return true;
  return d2 <= best * 1.00001f + 1e-12f;
}

__global__ void nn_bvh_query_kernel(const PtIdx* __restrict__ queries, const PtIdx* __restrict__ db,
                                    const AABB* __restrict__ nodes, int64_t* __restrict__ out_idx, int n_q,
                                    int n_db, int n_leaves, int P) {
  const int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= n_q) return;

  const float4 qv = __ldg(reinterpret_cast<const float4*>(queries + qi));
  const float qx = qv.x, qy = qv.y, qz = qv.z;
  const int q_orig = __float_as_int(qv.w);
  float best_d2 = INFINITY;
  int best_idx = 0x7FFFFFFF;

  int stack_node[32];
  float stack_d2[32];
  int sp = 0;
  int node = 1;

  while (true) {
    if (node >= P) {
      const int leaf = node - P;
      if (leaf < n_leaves) {
        const int start = leaf * LEAF;
        const int end = min(start + LEAF, n_db);
        for (int j = start; j < end; ++j) {
          const float4 pv = __ldg(reinterpret_cast<const float4*>(db + j));
          const float dx = qx - pv.x;
          const float dy = qy - pv.y;
          const float dz = qz - pv.z;
          const float d2 = dx * dx + dy * dy + dz * dz;
          const int p_orig = __float_as_int(pv.w);
          if (d2 < best_d2 || (d2 == best_d2 && p_orig < best_idx)) {
            best_d2 = d2;
            best_idx = p_orig;
          }
        }
      }
      bool popped = false;
      while (sp > 0) {
        --sp;
        if (may_contain(stack_d2[sp], best_d2)) {
          node = stack_node[sp];
          popped = true;
          break;
        }
      }
      if (!popped) break;
    } else {
      const int left = node << 1;
      const int right = left | 1;
      const float dl = aabb_dist2(nodes, left, qx, qy, qz);
      const float dr = aabb_dist2(nodes, right, qx, qy, qz);
      const bool visit_l = may_contain(dl, best_d2);
      const bool visit_r = may_contain(dr, best_d2);
      if (visit_l && visit_r) {
        if (dl <= dr) {
          stack_node[sp] = right;
          stack_d2[sp] = dr;
          ++sp;
          node = left;
        } else {
          stack_node[sp] = left;
          stack_d2[sp] = dl;
          ++sp;
          node = right;
        }
      } else if (visit_l) {
        node = left;
      } else if (visit_r) {
        node = right;
      } else {
        bool popped = false;
        while (sp > 0) {
          --sp;
          if (may_contain(stack_d2[sp], best_d2)) {
            node = stack_node[sp];
            popped = true;
            break;
          }
        }
        if (!popped) break;
      }
    }
  }

  out_idx[q_orig] = static_cast<int64_t>(best_idx);
}

extern "C" int launch_morton3d(const void* pts, const void* bounds, void* codes, int n, void* stream) {
  constexpr int TPB = 256;
  const int grid = (n + TPB - 1) / TPB;
  morton3d_kernel<<<grid, TPB, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(pts), static_cast<const float*>(bounds), static_cast<int32_t*>(codes), n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_build_bvh(const void* pts, const void* order, void* packed, void* nodes, int n,
                                int n_leaves, int P, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  constexpr int TPB = 256;
  pack_and_leaf_aabb_kernel<<<(P + TPB - 1) / TPB, TPB, 0, s>>>(
      static_cast<const float*>(pts), static_cast<const int64_t*>(order), static_cast<PtIdx*>(packed),
      static_cast<AABB*>(nodes), n, n_leaves, P);
  for (int level = P >> 1; level >= 1; level >>= 1) {
    build_internal_level_kernel<<<(level + TPB - 1) / TPB, TPB, 0, s>>>(static_cast<AABB*>(nodes), level);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_query_bvh(const void* queries, const void* db, const void* nodes, void* out_idx,
                                int n_q, int n_db, int n_leaves, int P, void* stream) {
  constexpr int TPB = 256;
  const int grid = (n_q + TPB - 1) / TPB;
  nn_bvh_query_kernel<<<grid, TPB, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const PtIdx*>(queries), static_cast<const PtIdx*>(db), static_cast<const AABB*>(nodes),
      static_cast<int64_t*>(out_idx), n_q, n_db, n_leaves, P);
  return static_cast<int>(cudaGetLastError());
}
