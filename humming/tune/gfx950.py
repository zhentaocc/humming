"""Tile / split-K heuristics for the ROCm gfx950 MFMA weight-only GEMM.

Tuned standalone on MI355X (256 CUs); see docs/rocm/log_mfma.md. The tile only depends on the
M bucket (and, for decode, on N), so one layer compiles at most four kernels. The split-K factor
depends on the runtime M and is computed by the launcher (csrc/launcher/mfma.h::mfma_num_splits);
``get_num_splits`` is the Python mirror used by tests.
"""

import dataclasses
import math

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics

MAX_SPLITS = 8
CHUNK_K = 128


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

    def to_tuning_config(self) -> dict:
        return {"mma_type": "mfma", "max_splits": MAX_SPLITS, **dataclasses.asdict(self)}

    @classmethod
    def from_tuning_config(cls, config: dict) -> "MfmaTile":
        return cls(**{field.name: config[field.name] for field in dataclasses.fields(cls) if field.name in config})


# The wide (8 x 4 MFMA tiles per wave, 128 x 256 block) tile halves the LDS A reads per MFMA
# and wins once its grid has >= WIDE_TILE_MIN_BLOCKS_PER_CU blocks per CU; below that the
# 128 x 128 tile (more blocks, split-K) is faster. BF16 never uses it: its larger dequant
# (17 vs 13 VALU per fragment) makes the 128-accumulator tile spill.
# See docs/rocm/log_perf.md.
WIDE_TILE_MIN_BLOCKS_PER_CU = 1.5


def wide_tile_min_m(shape_n: int, num_cus: int) -> int:
    """M above which the wide tile is used (exclusive bound, a multiple of 128)."""
    n_blocks = math.ceil(shape_n / 256)
    m_blocks = math.ceil(WIDE_TILE_MIN_BLOCKS_PER_CU * num_cus / n_blocks)
    return max(64, 128 * (m_blocks - 1))


def select_tile(shape_m: int, shape_n: int, num_cus: int, is_bfloat16: bool = False) -> MfmaTile:
    if shape_m <= 8:
        # Decode: no LDS, one 16-row MFMA tile; wider N per wave only when N gives enough blocks.
        tiles_n = 2 if shape_n // 128 >= num_cus // 2 else 1
        return MfmaTile(tiles_m=1, tiles_n=tiles_n, stages=4)
    if shape_m <= 32:
        return MfmaTile(tiles_m=2, tiles_n=2, stages=3, lds_a=True, scale_b=True)
    if shape_m <= 64:
        return MfmaTile(tiles_m=4, tiles_n=2, stages=3, lds_a=True, scale_b=True)
    if not is_bfloat16 and shape_m > wide_tile_min_m(shape_n, num_cus):
        return MfmaTile(tiles_m=8, tiles_n=4, stages=2, lds_a=True, scale_b=True)
    return MfmaTile(tiles_m=8, tiles_n=2, stages=2, lds_a=True, scale_b=True)


# M intervals (min exclusive, max inclusive) matching select_tile's buckets; the last bucket is
# split at wide_tile_min_m for FP16 (see get_m_intervals).
M_INTERVALS = ((0, 8), (8, 32), (32, 64), (64, 1 << 30))


def get_m_intervals(shape_n: int, num_cus: int, is_bfloat16: bool) -> tuple:
    if is_bfloat16:
        return M_INTERVALS
    wide_min_m = wide_tile_min_m(shape_n, num_cus)
    if wide_min_m <= 64:
        return M_INTERVALS
    return M_INTERVALS[:-1] + ((64, wide_min_m), (wide_min_m, 1 << 30))


def get_num_splits(tile: MfmaTile, shape_m: int, shape_n: int, shape_k: int, num_cus: int, max_splits: int = MAX_SPLITS) -> int:
    """Split K until about one block per CU (each split keeps >= 4 chunks of 128)."""
    num_blocks = math.ceil(shape_n / tile.block_n) * math.ceil(shape_m / tile.block_m)
    num_chunks_k = shape_k // CHUNK_K
    num_splits = 1
    while num_blocks * num_splits < num_cus and num_splits < max_splits and num_chunks_k // (num_splits * 2) >= 4:
        num_splits *= 2
    return num_splits


class Gfx950Heuristics(DeviceHeuristics):
    @classmethod
    def num_cus(cls) -> int:
        from humming.device import current_device

        return current_device.sm_count

    @classmethod
    def should_use_pdl_for_input(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        return False

    @classmethod
    def _check(cls, layer_config: LayerConfig, gemm_type) -> None:
        from humming.kernel.humming_mfma import check_mfma_layer_config

        check_mfma_layer_config(layer_config)
        if gemm_type is not None and GemmType(gemm_type) != GemmType.DENSE:
            raise NotImplementedError(f"ROCm MFMA GEMM supports dense GEMM only, got gemm_type={gemm_type}")

    @classmethod
    def get_config(cls, layer_config: LayerConfig, shape_m: int, gemm_type=GemmType.DENSE, **kwargs) -> dict:
        cls._check(layer_config, gemm_type)
        is_bfloat16 = layer_config.a_dtype == dtypes.bfloat16
        return select_tile(shape_m, layer_config.shape_n, cls.num_cus(), is_bfloat16).to_tuning_config()

    @classmethod
    def get_configs(cls, layer_config: LayerConfig, gemm_type=GemmType.DENSE, **kwargs) -> list:
        cls._check(layer_config, gemm_type)
        intervals = get_m_intervals(layer_config.shape_n, cls.num_cus(), layer_config.a_dtype == dtypes.bfloat16)
        return [[low, high, cls.get_config(layer_config, high, gemm_type)] for low, high in intervals]
