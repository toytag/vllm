/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../cuda_compat.h"
#include "attention_dtypes.h"

#include <float.h>
#include <type_traits>

namespace vllm {

// Q*K^T operation.
template<int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template<typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template<typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b) {
  float4 c;
  float zero = 0.f;
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
    "    {%0, %1, %2, %3}, \n"
    "    {%4, %5}, \n"
    "    {%6}, \n"
    "    {%7, %7, %7, %7}; \n"

    : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
    : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
  return c;
}

template<int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N]) {
  using A_vec = typename FloatVec<uint32_t>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  return hmma_fp32(make_uint2(float2_to_half2(qk_vec), 0u), 0x3c003c00u).x;
}

template<typename T, int THREAD_GROUP_SIZE>
struct Qk_hmma_dot {
  template<typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

template<>
struct Qk_hmma_dot<uint16_t, 4> {
  template<int N>
  static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N]) {
    return qk_hmma_dot_(q, k);
  }
};

} // namespace vllm
