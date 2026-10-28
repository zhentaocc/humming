"""Correctness of the gfx950 MFMA weight-only GEMM against an FP32 torch reference."""

import pytest
import torch

from tests.rocm.mfma_gemm import MfmaTile, mfma_gemm
from tests.rocm.quant_utils import TOLERANCE, error_stats, make_quantized_linear


def run_case(shape_m, shape_n, shape_k, num_bits, group_size, has_zero_point, dtype, tile=None, num_splits=None):
    linear = make_quantized_linear(shape_n, shape_k, num_bits, group_size, has_zero_point, dtype)
    inputs = torch.randn(shape_m, shape_k, dtype=dtype, device="cuda")
    outputs = mfma_gemm(
        inputs,
        linear.mfma_weight,
        linear.mfma_scale,
        linear.mfma_zero_point,
        num_bits,
        group_size,
        tile=tile,
        num_splits=num_splits,
    )
    reference = inputs.float() @ linear.dequantized().T
    return error_stats(outputs, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [0, 32, 64, 128])
@pytest.mark.parametrize("has_zero_point", [False, True])
@pytest.mark.parametrize("shape_m", [1, 7, 16, 64, 255])
def test_small_shapes(dtype, num_bits, group_size, has_zero_point, shape_m):
    if has_zero_point and group_size == 0:
        pytest.skip("zero point is tested group-wise")
    _, rel = run_case(shape_m, 256, 1024, num_bits, group_size, has_zero_point, dtype)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tile",
    [
        MfmaTile(1, 4, 1, 1, 4),
        MfmaTile(1, 4, 2, 2, 2),
        MfmaTile(2, 2, 4, 2, 2),
        MfmaTile(4, 1, 1, 4, 3),
        MfmaTile(2, 2, 4, 4, 2, lds_a=True),
        MfmaTile(1, 4, 2, 2, 3, lds_a=True),
        MfmaTile(1, 4, 8, 2, 2, lds_a=True, scale_b=True),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 2, 3, 8])
def test_tiles_and_splits(dtype, tile, num_splits):
    _, rel = run_case(100, 512, 2048, 4, 128, True, dtype, tile=tile, num_splits=num_splits)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [0, 32, 128])
@pytest.mark.parametrize("shape_m", [1, 77, 300])
@pytest.mark.parametrize("scale_b", [False, True])
def test_lds_a_tile(dtype, num_bits, group_size, shape_m, scale_b):
    tile = MfmaTile(1, 4, 8, 2, 2, lds_a=True, scale_b=scale_b) if scale_b else MfmaTile(2, 2, 4, 4, 2, lds_a=True)
    _, rel = run_case(shape_m, 384, 1024, num_bits, group_size, group_size > 0, dtype, tile=tile)
    assert rel < TOLERANCE[dtype]


def test_split_k_repeated_launches():
    """Split-K counters must reset so back-to-back launches stay correct."""
    dtype = torch.float16
    linear = make_quantized_linear(1024, 4096, 4, 128, False, dtype)
    inputs = torch.randn(1, 4096, dtype=dtype, device="cuda")
    reference = inputs.float() @ linear.dequantized().T
    for _ in range(20):
        outputs = mfma_gemm(inputs, linear.mfma_weight, linear.mfma_scale, None, 4, 128, num_splits=8)
        assert error_stats(outputs, reference)[1] < TOLERANCE[dtype]
