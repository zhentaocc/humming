"""Standalone launcher for the gfx950 MFMA weight-only GEMM (hipcc + ctypes).

The Humming JIT integration will replace this module; it only renders the kernel
template, builds it with ``tests.rocm.hip_build`` and launches it on torch tensors.
"""

import ctypes
import dataclasses
import functools
import math

import torch

from tests.rocm.hip_build import build

SOURCE_TEMPLATE = r"""
#include <humming/kernel/humming_mfma.cuh>

using ElementA = {element_a};
using ElementQB = UInt{bits};
using Tile = MfmaTileConfig<{warps_m}, {warps_n}, {tiles_m}, {tiles_n}, {stages}, {lds_a}, {scale_b}>;
constexpr uint32_t kGroupSize = {group_size};
constexpr bool kHasZeroPoint = {has_zero_point};

extern "C" int launch(
    const void *A, const void *B, const void *BS, const void *BZP, void *C,
    void *workspace, void *tile_counters,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t num_splits, void *stream) {{
  dim3 grid(CEIL_DIV(shape_n, Tile::kBlockN), CEIL_DIV(shape_m, Tile::kBlockM), num_splits);
  hipLaunchKernelGGL(
      (humming_mfma<ElementA, ElementQB, Tile, kGroupSize, kHasZeroPoint>),
      grid, dim3(Tile::kNumThreads), 0, (hipStream_t)stream,
      (const uint16_t *)A, (const uint32_t *)B, (const uint16_t *)BS, (const uint8_t *)BZP,
      (uint16_t *)C, (float *)workspace, (uint32_t *)tile_counters, shape_m, shape_n, shape_k);
  return (int)hipGetLastError();
}}
"""


@dataclasses.dataclass(frozen=True)
class MfmaTile:
    warps_m: int = 1
    warps_n: int = 4
    tiles_m: int = 1
    tiles_n: int = 1
    stages: int = 4
    lds_a: bool = False
    scale_b: bool = False

    @property
    def block_m(self) -> int:
        return self.warps_m * self.tiles_m * 16

    @property
    def block_n(self) -> int:
        return self.warps_n * self.tiles_n * 16


@dataclasses.dataclass(frozen=True)
class MfmaKernelKey:
    a_dtype: torch.dtype
    num_bits: int
    group_size: int
    has_zero_point: bool
    tile: MfmaTile


@functools.cache
def _load_kernel(key: MfmaKernelKey):
    element_a = "Float16" if key.a_dtype == torch.float16 else "BFloat16"
    source = SOURCE_TEMPLATE.format(
        element_a=element_a,
        bits=key.num_bits,
        warps_m=key.tile.warps_m,
        warps_n=key.tile.warps_n,
        tiles_m=key.tile.tiles_m,
        tiles_n=key.tile.tiles_n,
        stages=key.tile.stages,
        lds_a="true" if key.tile.lds_a else "false",
        scale_b="true" if key.tile.scale_b else "false",
        group_size=key.group_size,
        has_zero_point="true" if key.has_zero_point else "false",
    )
    library = build(source, "humming_mfma")
    launch = library.launch
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_uint32] * 4 + [ctypes.c_void_p]
    return launch


_WORKSPACES: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_workspace(device: torch.device, num_floats: int, num_counters: int) -> tuple[torch.Tensor, torch.Tensor]:
    workspace, counters = _WORKSPACES.get(device, (None, None))
    if workspace is None or workspace.numel() < num_floats:
        workspace = torch.empty(max(num_floats, 1 << 20), dtype=torch.float32, device=device)
    if counters is None or counters.numel() < num_counters:
        # Counters must start at zero; the kernel resets each one after use.
        counters = torch.zeros(max(num_counters, 1 << 16), dtype=torch.int32, device=device)
    _WORKSPACES[device] = (workspace, counters)
    return workspace, counters


def select_tile(shape_m: int, shape_n: int, shape_k: int, num_cus: int = 256) -> tuple[MfmaTile, int]:
    """Heuristic tile + split-K selection per M bucket (see docs/rocm/log_mfma.md)."""
    if shape_m <= 16:
        tile = MfmaTile(warps_m=1, warps_n=4, tiles_m=1, tiles_n=1, stages=4)
    elif shape_m <= 32:
        tile = MfmaTile(warps_m=1, warps_n=4, tiles_m=2, tiles_n=1, stages=4)
    elif shape_m <= 64:
        tile = MfmaTile(warps_m=1, warps_n=4, tiles_m=4, tiles_n=1, stages=3)
    else:
        tile = MfmaTile(warps_m=2, warps_n=2, tiles_m=4, tiles_n=2, stages=2)
    num_blocks = math.ceil(shape_n / tile.block_n) * math.ceil(shape_m / tile.block_m)
    num_chunks_k = shape_k // 128
    num_splits = 1
    while num_blocks * num_splits < 2 * num_cus and num_chunks_k // (num_splits * 2) >= 4:
        num_splits *= 2
    return tile, num_splits


def mfma_gemm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    num_bits: int,
    group_size: int,
    tile: MfmaTile | None = None,
    num_splits: int | None = None,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """``inputs [M, K] @ dequant(weight)^T``; ``weight`` / scales / zp in the MFMA layout.

    ``weight_scale`` and ``zero_point`` are ``[num_groups, N]``; ``group_size`` 0 means per-channel.
    """
    assert inputs.dtype in (torch.float16, torch.bfloat16) and inputs.is_contiguous()
    shape_m, shape_k = inputs.shape
    shape_n = weight.size(0)
    assert shape_k % 128 == 0 and shape_n % 64 == 0
    assert weight.shape[-1] * 32 // num_bits == shape_k
    if group_size >= shape_k:
        group_size = 0
    assert weight_scale.dtype == inputs.dtype

    selected_tile, selected_splits = select_tile(shape_m, shape_n, shape_k)
    tile = tile or selected_tile
    num_splits = num_splits or selected_splits
    if outputs is None:
        outputs = torch.empty(shape_m, shape_n, dtype=inputs.dtype, device=inputs.device)
    if shape_m == 0:
        return outputs

    num_tiles = math.ceil(shape_n / tile.block_n) * math.ceil(shape_m / tile.block_m)
    workspace, counters = _get_workspace(inputs.device, num_splits * shape_m * shape_n, num_tiles)
    key = MfmaKernelKey(inputs.dtype, num_bits, group_size, zero_point is not None, tile)
    launch = _load_kernel(key)
    status = launch(
        inputs.data_ptr(),
        weight.data_ptr(),
        weight_scale.data_ptr(),
        zero_point.data_ptr() if zero_point is not None else None,
        outputs.data_ptr(),
        workspace.data_ptr(),
        counters.data_ptr(),
        shape_m,
        shape_n,
        shape_k,
        num_splits,
        torch.cuda.current_stream(inputs.device).cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"humming_mfma launch failed with hipError {status}")
    return outputs
