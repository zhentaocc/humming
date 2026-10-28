"""HummingLayer on ROCm: weight transforms run; the GEMM stops with NotImplementedError."""

import pytest
import torch

from humming.layer import HummingLayer

SHAPE_N = 1024
SHAPE_K = 2048

LAYER_CASES = [
    ({"dtype": "int4"}, None),
    ({"dtype": "int8"}, None),
    ({"dtype": "float8e4m3"}, None),
    ({"dtype": "float4e2m1", "scale_dtype": "float8e8m0", "group_size": 32}, {"dtype": "float8e4m3"}),
]


@pytest.mark.parametrize(("weight_config", "input_config"), LAYER_CASES, ids=lambda case: str(case))
def test_layer_transforms_then_gemm_is_not_implemented(weight_config, input_config):
    layer = HummingLayer(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        weight_config=weight_config,
        torch_dtype=torch.bfloat16,
        input_config=input_config,
    ).cuda()
    weight = torch.randn(SHAPE_N, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    layer.load_from_unquantized(weight)
    layer.transform()
    torch.cuda.synchronize()
    for parameter_name, parameter in layer.named_parameters():
        assert parameter.is_cuda, parameter_name

    inputs = torch.randn(16, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="MFMA GEMM backend not yet integrated"):
        layer(inputs)
