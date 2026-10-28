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


template <uint32_t kWarpsM_, uint32_t kWarpsN_, uint32_t kWarpTilesM_, uint32_t kWarpTilesN_, uint32_t kStages_>
struct MfmaTileConfig {
  static constexpr uint32_t kWarpsM = kWarpsM_;
  static constexpr uint32_t kWarpsN = kWarpsN_;
  static constexpr uint32_t kWarpTilesM = kWarpTilesM_;
  static constexpr uint32_t kWarpTilesN = kWarpTilesN_;
  static constexpr uint32_t kStages = kStages_;
  static constexpr uint32_t kNumThreads = kWarpsM * kWarpsN * kWaveSize;
  static constexpr uint32_t kBlockM = kWarpsM * kWarpTilesM * 16;
  static constexpr uint32_t kBlockN = kWarpsN * kWarpTilesN * 16;
  static_assert(kStages >= 2, "register prefetch needs at least two stages");
};


template <class ElementA>
CUDA_INLINE float mfma_half_to_float(uint16_t bits) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    return __half2float(__ushort_as_half(bits));
  } else {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
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

  uint32_t regs_qb[kStages][kTilesN][kBits];
  uint4 regs_a[kStages][kTilesM][kMfmaPerChunk];
  uint16_t regs_bs[kStages][kTilesN][kGroupsPerChunk];
  uint32_t regs_bzp[kStages][kTilesN][kGroupsPerChunk];
  CRegisters regs_c[kTilesM][kTilesN];
  CRegisters regs_group_c[kIsChannelScale ? 1 : kTilesM][kIsChannelScale ? 1 : kTilesN];

  CUDA_INLINE bool is_tile_n_valid(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 < shape_n;
  }

  CUDA_INLINE uint32_t column(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 + Op::c_col(lane);
  }

  CUDA_INLINE void load_chunk(uint32_t stage, uint32_t chunk_k) {
    const uint32_t num_chunks_k = shape_k / kChunkK;
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      if (!is_tile_n_valid(tile_n)) continue;
      const size_t chunk_index = static_cast<size_t>(n_tile_offset + tile_n) * num_chunks_k + chunk_k;
      const uint32_t *src = b_ptr + (chunk_index * kWaveSize + lane) * kBits;
      if constexpr (kBits % 4 == 0) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kBits / 4; i++) {
          *reinterpret_cast<uint4 *>(&regs_qb[stage][tile_n][i * 4]) = reinterpret_cast<const uint4 *>(src)[i];
        }
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kBits; i++) regs_qb[stage][tile_n][i] = src[i];
      }

      if constexpr (!kIsChannelScale) {
        PRAGMA_UNROLL
        for (uint32_t group = 0; group < kGroupsPerChunk; group++) {
          const uint32_t group_index = (chunk_k * kChunkK + group * MIN(kGroupSize, kChunkK)) / kGroupSize;
          const size_t offset = static_cast<size_t>(group_index) * shape_n + column(tile_n);
          regs_bs[stage][tile_n][group] = bs_ptr[offset];
          if constexpr (kHasZeroPoint) regs_bzp[stage][tile_n][group] = bzp_ptr[offset];
        }
      }
    }

    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      const uint32_t row = m_offset + tile_m * 16 + Op::a_row(lane);
      const uint16_t *src = a_ptr + static_cast<size_t>(row) * shape_k + chunk_k * kChunkK + Op::ab_k(lane, 0);
      PRAGMA_UNROLL
      for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
        uint4 value = make_uint4(0, 0, 0, 0);
        if (row < shape_m) value = *reinterpret_cast<const uint4 *>(src + mfma_index * Op::K);
        regs_a[stage][tile_m][mfma_index] = value;
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
    PRAGMA_UNROLL
    for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
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
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          const uint32_t *regs_a_ptr = reinterpret_cast<const uint32_t *>(&regs_a[stage][tile_m][mfma_index]);
          if constexpr (kIsChannelScale) {
            Op::fma(regs_a_ptr, regs_b, regs_c[tile_m][tile_n]);
          } else {
            Op::fma(regs_a_ptr, regs_b, regs_group_c[tile_m][tile_n]);
          }
        }
      }
      if constexpr (!kIsChannelScale && kGroupSize < kChunkK) {
        if ((mfma_index + 1) % kMfmaPerGroup == 0) apply_group_scale(stage, group);
      }
    }
    if constexpr (!kIsChannelScale && kGroupSize >= kChunkK) {
      if ((chunk_k + 1) * kChunkK % kGroupSize == 0) apply_group_scale(stage, 0);
    }
  }

  CUDA_INLINE void mainloop(uint32_t chunk_begin, uint32_t chunk_end) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        regs_c[tile_m][tile_n] = CRegisters{};
        if constexpr (!kIsChannelScale) regs_group_c[tile_m][tile_n] = CRegisters{};
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
    static_for<0, kStages - 1>([&](auto stage) {
      if (decltype(stage)::value < num_iters) load_chunk(decltype(stage)::value, chunk_begin + decltype(stage)::value);
    });

    for (uint32_t base = 0; base < num_iters; base += kStages) {
      static_for<0, kStages>([&](auto stage_constant) {
        constexpr uint32_t stage = decltype(stage_constant)::value;
        constexpr uint32_t load_stage = (stage + kStages - 1) % kStages;
        const uint32_t iter = base + stage;
        if (iter < num_iters) {
          if (iter + kStages - 1 < num_iters) load_chunk(load_stage, chunk_begin + iter + kStages - 1);
          compute_chunk(stage, chunk_begin + iter, channel_zero_points);
        }
      });
    }

    if constexpr (!kIsChannelScale && kGroupSize > kChunkK) {
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

  Gemm gemm;
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
  const size_t split_stride = static_cast<size_t>(shape_m) * shape_n;
  gemm.for_each_output([&](uint32_t row, uint32_t col, float value) {
    workspace[blockIdx.z * split_stride + static_cast<size_t>(row) * shape_n + col] = value;
  });
  __threadfence();
  __syncthreads();

  __shared__ uint32_t is_last_block;
  const uint32_t tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) {
    uint32_t arrived = __hip_atomic_fetch_add(&tile_counters[tile_id], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    is_last_block = arrived == num_splits - 1;
  }
  __syncthreads();
  if (!is_last_block) return;
  __threadfence();

  gemm.for_each_output([&](uint32_t row, uint32_t col, float) {
    const size_t offset = static_cast<size_t>(row) * shape_n + col;
    float sum = 0.0f;
    for (uint32_t split = 0; split < num_splits; split++) {
      sum += __builtin_nontemporal_load(&workspace[split * split_stride + offset]);
    }
    C[offset] = mfma_float_to_half<ElementA>(sum);
  });
  if (threadIdx.x == 0) tile_counters[tile_id] = 0;
}
