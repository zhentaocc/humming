#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
typedef struct { alignas(64) uint64_t opaque[16]; } CUtensorMap;
