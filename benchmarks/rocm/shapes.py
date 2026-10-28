"""Single source of truth for the ROCm baseline benchmark shapes.

GEMM convention everywhere: ``Y[M, N] = X[M, K] @ W[N, K]^T`` (nn.Linear layout).
"""

M_VALUES = [1, 4, 16, 32, 64, 128, 256, 1024, 4096]

# name -> (N, K)
MODEL_SHAPES = {
    "llama3-8b/o_proj": (4096, 4096),
    "llama3-8b/qkv": (6144, 4096),
    "llama3-8b/gate_up": (28672, 4096),
    "llama3-8b/down": (4096, 14336),
    "llama3-70b/o_proj": (8192, 8192),
    "llama3-70b/gate_up": (57344, 8192),
    "llama3-70b/down": (8192, 28672),
    "qwen3-32b/o_proj": (5120, 5120),
    "qwen3-32b/gate_up": (51200, 5120),
    "qwen3-32b/down": (5120, 25600),
}

# Shapes shown in the condensed headline table of the markdown report.
HEADLINE_SHAPES = ["llama3-8b/o_proj", "llama3-70b/gate_up", "llama3-70b/down"]
HEADLINE_M = [1, 16, 1024, 4096]

# Small shape used for the pre-timing correctness check of every baseline.
# Several M values are used so that M-dependent dispatch paths are exercised.
CHECK_SHAPE = (512, 1024)  # (N, K)
CHECK_M_VALUES = [1, 33, 300]
