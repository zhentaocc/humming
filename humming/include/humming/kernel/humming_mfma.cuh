#pragma once

// Weight-only quantized GEMM for AMD CDNA (gfx950), built on MFMA 16x16x32 FP16/BF16.
//
//   C[M, N] = A[M, K] @ dequant(B)[N, K]^T
//
// A / C: row-major FP16 or BF16. B: unsigned integer codes (1..8 bits) in the ROCm MFMA
// layout produced by humming/rocm/layout.py: chunks of 16 channels x 128 K, each chunk is
// 64 lanes x kBits int32 words, chunks ordered [N / 16][K / 128]. Lane l of a wave loads
// its kBits words with one vector load and Humming's dequant<ElementQB, ElementA>(qb, res, t)
// turns them directly into the B fragment of MFMA t (t = 0..3), no shuffles needed.
//
// Scales: [num_groups, N] in the activation dtype. Group scales are applied to the FP32
// accumulator (every lane's C fragment is a single column, so this is one FMA per
// accumulator register per group), per-channel scales in the epilogue.
// Zero points (optional): [num_groups, N] uint8; without them the code is centered at
// 2^(bits - 1) (symmetric quantization).
//
// Block = kWarpsM x kWarpsN waves; each wave owns kWarpTilesM x kWarpTilesN 16x16 output
// tiles and streams its own B directly from global memory into registers with a
// kStages-deep register prefetch. gridDim.z > 1 splits K; the last block of a tile
// reduces the FP32 partials in a fixed order (deterministic) and resets the tile counter.

#include <humming/datatype/dequant.cuh>
#include <humming/mma/mfma.cuh>
#include <humming/utils/all.cuh>


// kLdsA: stage the block's A tile through LDS (shared by all waves along N) instead of
// every wave loading its own A fragments from global memory; needed once M is large.
// kScaleB: apply group scales to the dequantized 16-bit B fragment (packed multiply, as
// Humming's CUDA mainloop does) instead of to a per-group FP32 accumulator. Saves the
// temporary accumulators and the MFMA -> VALU dependency at every group boundary, at the
// cost of rounding (code - zp) * scale to FP16/BF16.
template <
    uint32_t kWarpsM_, uint32_t kWarpsN_, uint32_t kWarpTilesM_, uint32_t kWarpTilesN_, uint32_t kStages_,
    bool kLdsA_ = false, bool kScaleB_ = false>
struct MfmaTileConfig {
  static constexpr uint32_t kWarpsM = kWarpsM_;
  static constexpr uint32_t kWarpsN = kWarpsN_;
  static constexpr uint32_t kWarpTilesM = kWarpTilesM_;
  static constexpr uint32_t kWarpTilesN = kWarpTilesN_;
  static constexpr uint32_t kStages = kStages_;
  static constexpr uint32_t kNumThreads = kWarpsM * kWarpsN * kWaveSize;
  static constexpr uint32_t kBlockM = kWarpsM * kWarpTilesM * 16;
  static constexpr uint32_t kBlockN = kWarpsN * kWarpTilesN * 16;
  static constexpr bool kLdsA = kLdsA_;
  static constexpr bool kScaleB = kScaleB_;
  // 16-byte A units (8 values) per thread per 128-wide K chunk when staging through LDS.
  static constexpr uint32_t kLdsAUnitsPerThread = kLdsA ? kBlockM * 16 / kNumThreads : 0;
  static constexpr uint32_t kLdsAUnits = kLdsA ? 2 * kBlockM * 16 : 1;  // double buffered
  static_assert(kStages >= 2, "register prefetch needs at least two stages");
  static_assert(!kLdsA || kBlockM * 16 % kNumThreads == 0);
};


template <class ElementA>
CUDA_INLINE float mfma_half_to_float(uint16_t bits) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    return __half2float(__ushort_as_half(bits));
  } else {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  }
}

// Multiply both 16-bit halves of `value` by the 16-bit `scale` (v_pk_mul_f16 for FP16).
template <class ElementA>
CUDA_INLINE uint32_t mfma_scale_half2(uint32_t value, uint16_t scale) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    using Half2 = _Float16 __attribute__((ext_vector_type(2)));
    const Half2 scale2 = __builtin_bit_cast(Half2, static_cast<uint32_t>(scale) * 0x00010001u);
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(Half2, value) * scale2);
  } else {
    using BFloat2 = __bf16 __attribute__((ext_vector_type(2)));
    const BFloat2 scale2 = __builtin_bit_cast(BFloat2, static_cast<uint32_t>(scale) * 0x00010001u);
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(BFloat2, value) * scale2);
  }
}

template <class ElementA>
CUDA_INLINE uint16_t mfma_float_to_half(float value) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    return __half_as_ushort(__float2half_rn(value));
  } else {
    // Round to nearest even; NaN inputs are not expected from finite GEMMs.
    uint32_t bits = __float_as_uint(value);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
  }
}


// Dequantize the lane's packed words into the 16-bit B fragment of MFMA `mfma_index`.
template <class ElementQB, class ElementA, bool kHasZeroPoint>
struct MfmaDequant {
  static constexpr uint32_t kBits = ElementQB::kBits;
  static_assert(ElementQB::kIsIntegerType && !ElementQB::kIsSigned);

  // Humming's magic-number path keeps (code - zero point) exact only while it fits the
  // 16-bit mantissa; BF16 with 7/8-bit codes takes a small FP32 detour instead.
  static constexpr uint32_t kMaxHummingBits = std::is_same<ElementA, Float16>::value ? 10 : 7;
  static constexpr bool kUseHummingDequant = kBits <= kMaxHummingBits - kHasZeroPoint;

  // `zero_point` is the raw code; the result is what run() expects as `prepared_zero_point`.
  CUDA_INLINE static uint32_t prepare_zero_point(uint32_t zero_point) {
    if constexpr (kUseHummingDequant && kHasZeroPoint) {
      return dequant_single_zero_point<ElementQB, ElementA>(zero_point);
    } else if constexpr (kHasZeroPoint) {
      return zero_point;
    } else {
      return 0;
    }
  }

  CUDA_INLINE static void run(const uint32_t *qb, uint32_t *regs_b, uint32_t mfma_index, uint32_t prepared_zero_point) {
    if constexpr (kUseHummingDequant) {
      uint32_t zp_vals[4] = {prepared_zero_point, prepared_zero_point, prepared_zero_point, prepared_zero_point};
      dequant<ElementQB, ElementA, kHasZeroPoint, false>(qb, regs_b, mfma_index, zp_vals);
    } else {
      static_assert(std::is_same<ElementA, BFloat16>::value && static_next_power_of_2(kBits) == 8);
      constexpr uint32_t kMask = (1u << kBits) - 1;
      const float zero_point = kHasZeroPoint ? static_cast<float>(prepared_zero_point) : static_cast<float>(1u << (kBits - 1));
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        // Same value placement as dequant_b1248/b3567 with 8-bit padded fields.
        const uint32_t word_index = (mfma_index * 4 + i) / 2;
        uint32_t word;
        if constexpr (kBits == 8) {
          word = qb[word_index];
        } else {
          word = get_quanted_value_group<kBits, true>(qb, word_index);
        }
        const uint32_t shift = i % 2 * 8;
        float low = static_cast<float>((word >> shift) & kMask) - zero_point;
        float high = static_cast<float>((word >> (shift + 16)) & kMask) - zero_point;
        // Small integers are exact in BF16, so truncating the FP32 bits is exact.
        regs_b[i] = __builtin_amdgcn_perm(__float_as_uint(high), __float_as_uint(low), 0x07060302u);
      }
    }
  }
};


template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint>
struct MfmaWeightOnlyGemm {
  using Op = MfmaOp<ElementA, 16>;
  using Dequant = MfmaDequant<ElementQB, ElementA, kHasZeroPoint>;
  using CRegisters = typename Op::CRegisters;

  static constexpr uint32_t kBits = ElementQB::kBits;
  static constexpr uint32_t kTilesM = TileConfig::kWarpTilesM;
  static constexpr uint32_t kTilesN = TileConfig::kWarpTilesN;
  static constexpr uint32_t kStages = TileConfig::kStages;
  static constexpr uint32_t kChunkK = 128;
  static constexpr uint32_t kMfmaPerChunk = kChunkK / Op::K;
  static constexpr bool kIsChannelScale = kGroupSize == 0;
  static_assert(kIsChannelScale || kGroupSize % Op::K == 0, "group size must be a multiple of 32");
  // Scales (and zero points) referenced by one 128-wide K chunk.
  static constexpr uint32_t kGroupsPerChunk = kIsChannelScale ? 1 : MAX(kChunkK / kGroupSize, 1);
  static constexpr uint32_t kMfmaPerGroup = kIsChannelScale ? kMfmaPerChunk : MIN(kGroupSize / Op::K, kMfmaPerChunk);
  static constexpr bool kLdsA = TileConfig::kLdsA;
  static constexpr bool kScaleOnB = !kIsChannelScale && TileConfig::kScaleB;
  // Group scales on a temporary FP32 accumulator (exact (code - zp), FP32 scaling).
  static constexpr bool kScaleOnC = !kIsChannelScale && !kScaleOnB;
  static constexpr uint32_t kLdsAUnits = TileConfig::kLdsAUnitsPerThread;
  // LDS A tile: kBlockM rows x 16 units of 16 bytes; unit u of row r is stored at slot
  // u ^ (r % 16) so both the row-wise stores and the column-wise MFMA fragment reads
  // (16 rows, same unit) hit 16 distinct 16-byte bank groups.
  static constexpr uint32_t kLdsABufferUnits = TileConfig::kBlockM * 16;

  const uint16_t *a_ptr;
  const uint32_t *b_ptr;
  const uint16_t *bs_ptr;
  const uint8_t *bzp_ptr;
  uint32_t shape_m;
  uint32_t shape_n;
  uint32_t shape_k;
  uint32_t lane;
  uint32_t m_offset;
  uint32_t n_tile_offset;
  uint32_t thread_id;
  uint32_t block_m_offset;
  uint32_t local_m_offset;
  mfma_u32x4_t *lds_a;

  uint32_t regs_qb[kStages][kTilesN][kBits];
  mfma_u32x4_t regs_a[kStages][kLdsA ? 1 : kTilesM][kLdsA ? 1 : kMfmaPerChunk];
  mfma_u32x4_t regs_a_share[kStages][kLdsA ? kLdsAUnits : 1];
  uint16_t regs_bs[kStages][kTilesN][kGroupsPerChunk];
  uint32_t regs_bzp[kStages][kTilesN][kGroupsPerChunk];
  CRegisters regs_c[kTilesM][kTilesN];
  CRegisters regs_group_c[kScaleOnC ? kTilesM : 1][kScaleOnC ? kTilesN : 1];

  CUDA_INLINE bool is_tile_n_valid(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 < shape_n;
  }

  CUDA_INLINE uint32_t column(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 + Op::c_col(lane);
  }

  // Out-of-range N tiles / M rows are clamped to valid addresses instead of predicated:
  // their results are never stored, and branch-free loads let the waitcnt pass keep the
  // register prefetch in flight (a guarded load forces s_waitcnt vmcnt(0) at the join).
  CUDA_INLINE uint32_t clamped_n_tile(uint32_t tile_n) const {
    return MIN(n_tile_offset + tile_n, shape_n / 16 - 1);
  }

  CUDA_INLINE uint32_t clamped_column(uint32_t tile_n) const {
    return clamped_n_tile(tile_n) * 16 + Op::c_col(lane);
  }

  CUDA_INLINE void load_chunk(uint32_t stage, uint32_t chunk_k) {
    const uint32_t num_chunks_k = shape_k / kChunkK;
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const size_t chunk_index = static_cast<size_t>(clamped_n_tile(tile_n)) * num_chunks_k + chunk_k;
      const uint32_t *src = b_ptr + (chunk_index * kWaveSize + lane) * kBits;
      if constexpr (kBits % 4 == 0) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kBits / 4; i++) {
          *reinterpret_cast<mfma_u32x4_t *>(&regs_qb[stage][tile_n][i * 4]) = reinterpret_cast<const mfma_u32x4_t *>(src)[i];
        }
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kBits; i++) regs_qb[stage][tile_n][i] = src[i];
      }

      if constexpr (!kIsChannelScale) {
        PRAGMA_UNROLL
        for (uint32_t group = 0; group < kGroupsPerChunk; group++) {
          const uint32_t group_index = (chunk_k * kChunkK + group * MIN(kGroupSize, kChunkK)) / kGroupSize;
          const size_t offset = static_cast<size_t>(group_index) * shape_n + clamped_column(tile_n);
          regs_bs[stage][tile_n][group] = bs_ptr[offset];
          if constexpr (kHasZeroPoint) regs_bzp[stage][tile_n][group] = bzp_ptr[offset];
        }
      }
    }

    if constexpr (kLdsA) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t unit_index = thread_id + i * TileConfig::kNumThreads;
        const uint32_t row = MIN(block_m_offset + unit_index / 16, shape_m - 1);
        const uint16_t *src = a_ptr + static_cast<size_t>(row) * shape_k + chunk_k * kChunkK + unit_index % 16 * 8;
        regs_a_share[stage][i] = *reinterpret_cast<const mfma_u32x4_t *>(src);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        // Rows past M re-read row M - 1: A row r only feeds C row r, which is not stored.
        const uint32_t row = MIN(m_offset + tile_m * 16 + Op::a_row(lane), shape_m - 1);
        const uint16_t *src = a_ptr + static_cast<size_t>(row) * shape_k + chunk_k * kChunkK + Op::ab_k(lane, 0);
        PRAGMA_UNROLL
        for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
          regs_a[stage][tile_m][mfma_index] = *reinterpret_cast<const mfma_u32x4_t *>(src + mfma_index * Op::K);
        }
      }
    }
  }

  CUDA_INLINE void apply_group_scale(uint32_t stage, uint32_t group) {
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const float scale = mfma_half_to_float<ElementA>(regs_bs[stage][tile_n][group]);
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
          regs_c[tile_m][tile_n][reg] += regs_group_c[tile_m][tile_n][reg] * scale;
          regs_group_c[tile_m][tile_n][reg] = 0.0f;
        }
      }
    }
  }

  CUDA_INLINE void compute_chunk(uint32_t stage, uint32_t chunk_k, uint32_t *channel_zero_points) {
    mfma_u32x4_t *lds_buffer = lds_a + chunk_k % 2 * kLdsABufferUnits;
    if constexpr (kLdsA) {
      // One barrier per chunk: the other buffer was last read before this barrier's
      // predecessor, so it is free to be overwritten by the next chunk.
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t unit_index = thread_id + i * TileConfig::kNumThreads;
        const uint32_t row = unit_index / 16;
        lds_buffer[row * 16 + (unit_index % 16 ^ row % 16)] = regs_a_share[stage][i];
      }
      __syncthreads();
    }
    PRAGMA_UNROLL
    for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
      mfma_u32x4_t lds_regs_a[kLdsA ? kTilesM : 1];
      if constexpr (kLdsA) {
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          const uint32_t row = local_m_offset + tile_m * 16 + Op::a_row(lane);
          const uint32_t unit = mfma_index * 4 + lane / 16;
          lds_regs_a[tile_m] = lds_buffer[row * 16 + (unit ^ row % 16)];
        }
      }
      const uint32_t group = mfma_index / kMfmaPerGroup % kGroupsPerChunk;
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        uint32_t zero_point = 0;
        if constexpr (kHasZeroPoint && kIsChannelScale) {
          zero_point = channel_zero_points[tile_n];
        } else if constexpr (kHasZeroPoint) {
          zero_point = Dequant::prepare_zero_point(regs_bzp[stage][tile_n][group]);
        }
        uint32_t regs_b[Op::kRegsAB];
        Dequant::run(regs_qb[stage][tile_n], regs_b, mfma_index, zero_point);
        if constexpr (kScaleOnB) {
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < Op::kRegsAB; i++) {
            regs_b[i] = mfma_scale_half2<ElementA>(regs_b[i], regs_bs[stage][tile_n][group]);
          }
        }
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          const uint32_t *regs_a_ptr;
          if constexpr (kLdsA) {
            regs_a_ptr = reinterpret_cast<const uint32_t *>(&lds_regs_a[tile_m]);
          } else {
            regs_a_ptr = reinterpret_cast<const uint32_t *>(&regs_a[stage][tile_m][mfma_index]);
          }
          if constexpr (!kScaleOnC) {
            Op::fma(regs_a_ptr, regs_b, regs_c[tile_m][tile_n]);
          } else {
            Op::fma(regs_a_ptr, regs_b, regs_group_c[tile_m][tile_n]);
          }
        }
      }
      if constexpr (kScaleOnC && kGroupSize < kChunkK) {
        if ((mfma_index + 1) % kMfmaPerGroup == 0) apply_group_scale(stage, group);
      }
    }
    if constexpr (kScaleOnC && kGroupSize >= kChunkK) {
      if ((chunk_k + 1) * kChunkK % kGroupSize == 0) apply_group_scale(stage, 0);
    }
  }

  CUDA_INLINE void mainloop(uint32_t chunk_begin, uint32_t chunk_end) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        regs_c[tile_m][tile_n] = CRegisters{};
        if constexpr (kScaleOnC) regs_group_c[tile_m][tile_n] = CRegisters{};
      }
    }

    uint32_t channel_zero_points[kTilesN];
    if constexpr (kHasZeroPoint && kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        channel_zero_points[tile_n] = is_tile_n_valid(tile_n) ? Dequant::prepare_zero_point(bzp_ptr[column(tile_n)]) : 0;
      }
    }

    const uint32_t num_iters = chunk_end - chunk_begin;
    if (num_iters == 0) return;
    // Prefetches past the end of this K range re-load the last chunk (an L2 hit) so that
    // every load is unconditional; see clamped_n_tile.
    const uint32_t last_chunk = chunk_end - 1;
    static_for<0, kStages - 1>([&](auto stage) {
      load_chunk(decltype(stage)::value, MIN(chunk_begin + decltype(stage)::value, last_chunk));
    });

    uint32_t base = 0;
    for (; base + kStages <= num_iters; base += kStages) {
      static_for<0, kStages>([&](auto stage_constant) {
        constexpr uint32_t stage = decltype(stage_constant)::value;
        constexpr uint32_t load_stage = (stage + kStages - 1) % kStages;
        load_chunk(load_stage, MIN(chunk_begin + base + stage + kStages - 1, last_chunk));
        compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
      });
    }
    // Tail (< kStages chunks): all remaining data is already in flight.
    static_for<0, kStages - 1>([&](auto stage_constant) {
      constexpr uint32_t stage = decltype(stage_constant)::value;
      if (base + stage < num_iters) compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
    });

    if constexpr (kScaleOnC && kGroupSize > kChunkK) {
      // A K split that ends inside a group still owes that group's scale.
      if (num_iters > 0 && chunk_end * kChunkK % kGroupSize != 0) {
        PRAGMA_UNROLL
        for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
          const uint32_t group_index = (chunk_end - 1) * kChunkK / kGroupSize;
          regs_bs[0][tile_n][0] = is_tile_n_valid(tile_n) ? bs_ptr[static_cast<size_t>(group_index) * shape_n + column(tile_n)] : 0;
        }
        apply_group_scale(0, 0);
      }
    }

    if constexpr (kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!is_tile_n_valid(tile_n)) continue;
        const float scale = mfma_half_to_float<ElementA>(bs_ptr[column(tile_n)]);
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          PRAGMA_UNROLL
          for (uint32_t reg = 0; reg < Op::kRegsC; reg++) regs_c[tile_m][tile_n][reg] *= scale;
        }
      }
    }
  }

  template <class Func>
  CUDA_INLINE void for_each_output(Func &&func) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!is_tile_n_valid(tile_n)) continue;
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
          const uint32_t row = m_offset + tile_m * 16 + Op::c_row(lane, reg);
          if (row < shape_m) func(row, column(tile_n), regs_c[tile_m][tile_n][reg]);
        }
      }
    }
  }
};


template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint>
__global__ __launch_bounds__(TileConfig::kNumThreads) void humming_mfma(
    const uint16_t *__restrict__ A,
    const uint32_t *__restrict__ B,
    const uint16_t *__restrict__ BS,
    const uint8_t *__restrict__ BZP,
    uint16_t *__restrict__ C,
    float *__restrict__ workspace,
    uint32_t *__restrict__ tile_counters,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k) {
  using Gemm = MfmaWeightOnlyGemm<ElementA, ElementQB, TileConfig, kGroupSize, kHasZeroPoint>;
  const uint32_t warp_id = threadIdx.x / kWaveSize;
  const uint32_t warp_m = warp_id / TileConfig::kWarpsN;
  const uint32_t warp_n = warp_id % TileConfig::kWarpsN;

  __shared__ mfma_u32x4_t lds_a[TileConfig::kLdsAUnits];
  Gemm gemm;
  gemm.lds_a = lds_a;
  gemm.thread_id = threadIdx.x;
  gemm.block_m_offset = blockIdx.y * TileConfig::kBlockM;
  gemm.local_m_offset = warp_m * TileConfig::kWarpTilesM * 16;
  gemm.a_ptr = A;
  gemm.b_ptr = B;
  gemm.bs_ptr = BS;
  gemm.bzp_ptr = BZP;
  gemm.shape_m = shape_m;
  gemm.shape_n = shape_n;
  gemm.shape_k = shape_k;
  gemm.lane = threadIdx.x % kWaveSize;
  gemm.m_offset = blockIdx.y * TileConfig::kBlockM + warp_m * TileConfig::kWarpTilesM * 16;
  gemm.n_tile_offset = (blockIdx.x * TileConfig::kBlockN + warp_n * TileConfig::kWarpTilesN * 16) / 16;

  const uint32_t num_chunks_k = shape_k / Gemm::kChunkK;
  const uint32_t num_splits = gridDim.z;
  const uint32_t chunk_begin = blockIdx.z * num_chunks_k / num_splits;
  const uint32_t chunk_end = (blockIdx.z + 1) * num_chunks_k / num_splits;
  gemm.mainloop(chunk_begin, chunk_end);

  if (num_splits == 1) {
    gemm.for_each_output([&](uint32_t row, uint32_t col, float value) {
      C[static_cast<size_t>(row) * shape_n + col] = mfma_float_to_half<ElementA>(value);
    });
    return;
  }

  // Split-K: publish FP32 partials, the last block of this output tile reduces them.
  //
  // An agent-scope release/acquire fence on gfx950 is a full L2 writeback + invalidate
  // (buffer_wbl2 sc1 / buffer_inv sc1, L2 is per XCD), which cost ~4 us per split. Instead
  // every partial is itself an agent-scope (relaxed atomic) store/load, which is coherent
  // across XCDs on its own; we only have to make sure the stores have completed
  // (s_waitcnt vmcnt(0), gfx9 counts stores in vmcnt) before the counter increment.
  const size_t split_stride = static_cast<size_t>(shape_m) * shape_n;
  gemm.for_each_output([&](uint32_t row, uint32_t col, float value) {
    float *ptr = &workspace[blockIdx.z * split_stride + static_cast<size_t>(row) * shape_n + col];
    __hip_atomic_store(ptr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  });
  __builtin_amdgcn_s_waitcnt(0x0F70);  // vmcnt(0) expcnt(7) lgkmcnt(15)
  __syncthreads();

  __shared__ uint32_t is_last_block;
  const uint32_t tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t arrived = __hip_atomic_fetch_add(&tile_counters[tile_id], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    is_last_block = arrived == num_splits - 1;
  }
  __syncthreads();
  if (!is_last_block) return;

  gemm.for_each_output([&](uint32_t row, uint32_t col, float) {
    const size_t offset = static_cast<size_t>(row) * shape_n + col;
    float sum = 0.0f;
    for (uint32_t split = 0; split < num_splits; split++) {
      sum += __hip_atomic_load(&workspace[split * split_stride + offset], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    C[offset] = mfma_float_to_half<ElementA>(sum);
  });
  if (threadIdx.x == 0) tile_counters[tile_id] = 0;
}
