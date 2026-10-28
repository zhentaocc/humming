#pragma once

// Minimal CUDA -> HIP aliases needed to compile Humming's portable device headers
// (datatype/*, utils/base.cuh, utils/ptx/math.cuh) with hipcc / hiprtc for gfx950.
// The platform layer is expected to absorb this file into its own portability header.

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#ifndef HUMMING_ROCM
  #define HUMMING_ROCM 1
#endif

using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

__device__ __forceinline__ __hip_bfloat162 __float2bfloat162_rn(float x) {
  return __float22bfloat162_rn(make_float2(x, x));
}

__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float x, float y) {
  return __float22bfloat162_rn(make_float2(x, y));
}

// Unsigned 4x8-bit dot product (CUDA __dp4a with unsigned operands).
__device__ __forceinline__ uint32_t __dp4a(uint32_t a, uint32_t b, uint32_t c) {
  return __builtin_amdgcn_udot4(a, b, c, false);
}

// Opaque placeholder so host-side signatures that mention CUtensorMap still parse.
typedef struct {
  alignas(64) uint64_t opaque[16];
} CUtensorMap;
