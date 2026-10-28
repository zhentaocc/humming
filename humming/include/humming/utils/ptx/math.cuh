#pragma once

#include <humming/utils/base.cuh>

#if defined(__HIP__)
// PTX prmt (default mode) on v_perm_b32: selector nibble i picks byte i from {a, b};
// v_perm wants one selector byte per output byte with src0 providing bytes 4..7.
// The prmt sign-replicate mode (selector bit 3) is not supported.
CUDA_INLINE uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t nibbles = s & 0x7777u;
  uint32_t selector = (nibbles & 0xFu) | ((nibbles & 0xF0u) << 4) | ((nibbles & 0xF00u) << 8) | ((nibbles & 0xF000u) << 12);
  return __builtin_amdgcn_perm(b, a, selector);
};

template <uint32_t lut>
CUDA_INLINE uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t res = 0;
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 8; i++) {
    if (lut & (1u << i)) {
      uint32_t term_a = (i & 4) ? a : ~a;
      uint32_t term_b = (i & 2) ? b : ~b;
      uint32_t term_c = (i & 1) ? c : ~c;
      res |= term_a & term_b & term_c;
    }
  }
  return res;
};

CUDA_INLINE uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) | c;
};

CUDA_INLINE uint32_t lop3_and_xor(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) ^ c;
};
#else
CUDA_INLINE uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(s));
  return res;
};

template <uint32_t lut>
CUDA_INLINE uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
};

CUDA_INLINE uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  return lop3<(0xF0 & 0xCC) | 0xAA>(a, b, c);
};

CUDA_INLINE uint32_t lop3_and_xor(uint32_t a, uint32_t b, uint32_t c) {
  return lop3<(0xF0 & 0xCC) ^ 0xAA>(a, b, c);
};
#endif
