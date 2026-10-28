#pragma once

// CDNA3/CDNA4 (gfx950) MFMA wrappers for FP16/BF16 inputs with FP32 accumulation.
//
// Fragment layouts (wave64, verified on MI355X by tests/rocm/test_mfma_layout.py, see
// docs/rocm/mfma_layout.md). For lane = 0..63, slot s = 0..kElemsPerLane-1, register r:
//
//   16x16x32:  A[m][k]: m = lane % 16, k = (lane / 16) * 8 + s
//              B[k][n]: n = lane % 16, k = (lane / 16) * 8 + s
//              C[m][n]: n = lane % 16, m = (lane / 16) * 4 + r          (r = 0..3)
//   32x32x16:  A[m][k]: m = lane % 32, k = (lane / 32) * 8 + s
//              B[k][n]: n = lane % 32, k = (lane / 32) * 8 + s
//              C[m][n]: n = lane % 32, m = (r / 4) * 8 + (lane / 32) * 4 + r % 4   (r = 0..15)
//
// Every lane holds 8 consecutive K values of one A row and one B column, packed as
// four 32-bit registers of (even k, odd k) 16-bit pairs. That is exactly what a
// single Humming dequant<..., Float16/BFloat16>(qb, res, j) call produces.

#include <humming/datatype/dtypes.cuh>
#include <humming/utils/base.cuh>

#if defined(__HIP__)

using mfma_f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using mfma_bf16x8_t = __bf16 __attribute__((ext_vector_type(8)));
using mfma_f32x4_t = float __attribute__((ext_vector_type(4)));
using mfma_f32x16_t = float __attribute__((ext_vector_type(16)));

constexpr uint32_t kWaveSize = 64;

template <class ElementA, uint32_t kShapeM>
struct MfmaOp;

template <class ElementA>
struct MfmaOp<ElementA, 16> {
  static_assert(std::is_same<ElementA, Float16>::value || std::is_same<ElementA, BFloat16>::value);
  static constexpr uint32_t M = 16;
  static constexpr uint32_t N = 16;
  static constexpr uint32_t K = 32;
  // Number of 16-bit A/B elements and FP32 C elements held by one lane.
  static constexpr uint32_t kElemsPerLane = 8;
  static constexpr uint32_t kRegsAB = 4;
  static constexpr uint32_t kRegsC = 4;
  // Lanes that share the same row of A / column of B but hold different K slices.
  static constexpr uint32_t kLaneGroups = kWaveSize / M;

  using CRegisters = mfma_f32x4_t;

  CUDA_INLINE static uint32_t a_row(uint32_t lane) { return lane % 16; }
  CUDA_INLINE static uint32_t b_col(uint32_t lane) { return lane % 16; }
  CUDA_INLINE static uint32_t ab_k(uint32_t lane, uint32_t slot) { return lane / 16 * 8 + slot; }
  CUDA_INLINE static uint32_t c_row(uint32_t lane, uint32_t reg) { return lane / 16 * 4 + reg; }
  CUDA_INLINE static uint32_t c_col(uint32_t lane) { return lane % 16; }

  CUDA_INLINE static void fma(const uint32_t *a, const uint32_t *b, CRegisters &c) {
    if constexpr (std::is_same<ElementA, Float16>::value) {
      mfma_f16x8_t a_vec = *reinterpret_cast<const mfma_f16x8_t *>(a);
      mfma_f16x8_t b_vec = *reinterpret_cast<const mfma_f16x8_t *>(b);
      c = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vec, c, 0, 0, 0);
    } else {
      mfma_bf16x8_t a_vec = *reinterpret_cast<const mfma_bf16x8_t *>(a);
      mfma_bf16x8_t b_vec = *reinterpret_cast<const mfma_bf16x8_t *>(b);
      c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_vec, b_vec, c, 0, 0, 0);
    }
  }
};

template <class ElementA>
struct MfmaOp<ElementA, 32> {
  static_assert(std::is_same<ElementA, Float16>::value || std::is_same<ElementA, BFloat16>::value);
  static constexpr uint32_t M = 32;
  static constexpr uint32_t N = 32;
  static constexpr uint32_t K = 16;
  static constexpr uint32_t kElemsPerLane = 8;
  static constexpr uint32_t kRegsAB = 4;
  static constexpr uint32_t kRegsC = 16;
  static constexpr uint32_t kLaneGroups = kWaveSize / M;

  using CRegisters = mfma_f32x16_t;

  CUDA_INLINE static uint32_t a_row(uint32_t lane) { return lane % 32; }
  CUDA_INLINE static uint32_t b_col(uint32_t lane) { return lane % 32; }
  CUDA_INLINE static uint32_t ab_k(uint32_t lane, uint32_t slot) { return lane / 32 * 8 + slot; }
  CUDA_INLINE static uint32_t c_row(uint32_t lane, uint32_t reg) { return reg / 4 * 8 + lane / 32 * 4 + reg % 4; }
  CUDA_INLINE static uint32_t c_col(uint32_t lane) { return lane % 32; }

  CUDA_INLINE static void fma(const uint32_t *a, const uint32_t *b, CRegisters &c) {
    if constexpr (std::is_same<ElementA, Float16>::value) {
      mfma_f16x8_t a_vec = *reinterpret_cast<const mfma_f16x8_t *>(a);
      mfma_f16x8_t b_vec = *reinterpret_cast<const mfma_f16x8_t *>(b);
      c = __builtin_amdgcn_mfma_f32_32x32x16_f16(a_vec, b_vec, c, 0, 0, 0);
    } else {
      mfma_bf16x8_t a_vec = *reinterpret_cast<const mfma_bf16x8_t *>(a);
      mfma_bf16x8_t b_vec = *reinterpret_cast<const mfma_bf16x8_t *>(b);
      c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a_vec, b_vec, c, 0, 0, 0);
    }
  }
};

#endif
