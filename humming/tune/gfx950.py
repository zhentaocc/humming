"""Tile / split-K heuristics for the ROCm gfx950 MFMA weight-only GEMM.

Tuned standalone on MI355X (256 CUs); see docs/rocm/log_mfma.md. The tile only depends on the
M bucket (and, for decode, on N), so one layer compiles at most four kernels. The split-K factor
depends on the runtime M and is computed by the launcher (csrc/launcher/mfma.h::mfma_num_splits);
``get_num_splits`` is the Python mirror used by tests.
"""

import dataclasses
import math

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

    def to_moe_tuning_config(self) -> dict:
        # block_shape[0] is the M block the routing must be aligned to (vLLM reads it as
        # moe_block_size); the kernel itself only reads the MfmaTile fields.
        return self.to_tuning_config() | {"block_shape": [self.block_m, self.block_n, CHUNK_K]}

    @classmethod
    def from_tuning_config(cls, config: dict) -> "MfmaTile":
        return cls(**{field.name: config[field.name] for field in dataclasses.fields(cls) if field.name in config})


def select_tile(shape_m: int, shape_n: int, num_cus: int) -> MfmaTile:
    if shape_m <= 8:
        # Decode: no LDS, one 16-row MFMA tile; wider N per wave only when N gives enough blocks.
        tiles_n = 2 if shape_n // 128 >= num_cus // 2 else 1
        return MfmaTile(tiles_m=1, tiles_n=tiles_n, stages=4)
    if shape_m <= 32:
        return MfmaTile(tiles_m=2, tiles_n=2, stages=2, lds_a=True, scale_b=True)
    if shape_m <= 64:
        return MfmaTile(tiles_m=4, tiles_n=2, stages=2, lds_a=True, scale_b=True)
    return MfmaTile(tiles_m=8, tiles_n=2, stages=2, lds_a=True, scale_b=True)


# M intervals (min exclusive, max inclusive) matching select_tile's buckets.
M_INTERVALS = ((0, 8), (8, 32), (32, 64), (64, 1 << 30))


# MoE: the M block is chosen from the routed rows per expert (valid_shape_m = tokens * top_k
# spread over the layer's experts), never from N, so that the w13 and w2 GEMMs of one layer pick
# the same block_m for one valid_shape_m: vLLM aligns the routing once (moe_align_block_size with
# the w13 config's block_shape[0]) and reuses it for w2.
MOE_ROWS_PER_EXPERT_BUCKETS = (16, 32, 64)


def select_moe_tile(valid_shape_m: int, num_experts: int) -> MfmaTile:
    rows_per_expert = math.ceil(valid_shape_m / max(num_experts, 1))
    if rows_per_expert <= 16:
        # Decode: one 16-row MFMA tile, A straight from global memory.
        return MfmaTile(tiles_m=1, tiles_n=1, stages=4)
    if rows_per_expert <= 32:
        return MfmaTile(tiles_m=2, tiles_n=2, stages=2, lds_a=True, scale_b=True)
    if rows_per_expert <= 64:
        return MfmaTile(tiles_m=4, tiles_n=2, stages=2, lds_a=True, scale_b=True)
    return MfmaTile(tiles_m=8, tiles_n=2, stages=2, lds_a=True, scale_b=True)


def moe_m_intervals(num_experts: int) -> list[tuple[int, int]]:
    """valid_shape_m intervals (min exclusive, max inclusive) matching select_moe_tile."""
    num_experts = max(num_experts, 1)
    bounds = [0] + [rows * num_experts for rows in MOE_ROWS_PER_EXPERT_BUCKETS] + [1 << 30]
    return list(zip(bounds[:-1], bounds[1:]))


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
    def _check(cls, layer_config: LayerConfig, gemm_type) -> GemmType:
        from humming.kernel.humming_mfma import check_mfma_gemm_type, check_mfma_layer_config

        check_mfma_layer_config(layer_config)
        return check_mfma_gemm_type(layer_config, gemm_type)

    @classmethod
    def get_config(cls, layer_config: LayerConfig, shape_m: int, gemm_type=GemmType.DENSE, **kwargs) -> dict:
        if cls._check(layer_config, gemm_type) != GemmType.DENSE:
            return select_moe_tile(shape_m, layer_config.num_experts).to_moe_tuning_config()
        return select_tile(shape_m, layer_config.shape_n, cls.num_cus()).to_tuning_config()

    @classmethod
    def get_configs(cls, layer_config: LayerConfig, gemm_type=GemmType.DENSE, **kwargs) -> list:
        if cls._check(layer_config, gemm_type) != GemmType.DENSE:
            return [
                [low, high, select_moe_tile(high, layer_config.num_experts).to_moe_tuning_config()]
                for low, high in moe_m_intervals(layer_config.num_experts)
            ]
        return [[low, high, cls.get_config(layer_config, high, gemm_type)] for low, high in M_INTERVALS]
