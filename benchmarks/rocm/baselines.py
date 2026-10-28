"""Registered ROCm baseline implementations.

Every entry follows ``factory(M, N, K, dtype, bits, group_size, zp) -> (setup, run)``
(see ``common.py``). Humming itself is registered last as ``humming`` and runs through
its public ``HummingLayer`` API.
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import (
    AWQ_REVERSE_ORDER,
    Problem,
    Unsupported,
    gptq_pack,
    n_packed_pack,
    nominal_weight_bytes,
    pack_last_dim,
    register,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short=10", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


VLLM_REV = "5efbf0886a"  # /home/zhenchen/projects/vllm HEAD the kernels were copied from
AITER_REV = _git_rev("/home/zhenchen/projects/aiter")


def _check_group(group_size, K, allowed=(32, 64, 128)):
    if group_size not in allowed and group_size != K:
        raise Unsupported(f"group_size {group_size} not supported")


# ==========================================================================
# 1. Dense reference
# ==========================================================================
@register(
    "torch_dense",
    kind="dense",
    bits=(16,),
    source=f"F.linear (hipBLASLt), torch {torch.__version__}",
)
def torch_dense(M, N, K, dtype, bits, group_size, zp):
    def setup(p: Problem):
        return {"x": p.fresh_x(), "w": p.w_dequant(dtype), "weight_bytes": N * K * 2}

    def run(s):
        return F.linear(s["x"], s["w"])

    return setup, run


# ==========================================================================
# 2. vLLM pure-Triton W4A16 kernels (copied, see third_party/)
# ==========================================================================
@register(
    "vllm_triton_w4a16",
    kind="wNa16",
    bits=(4,),
    source=f"vllm@{VLLM_REV} kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel)",
)
def vllm_triton_w4a16(M, N, K, dtype, bits, group_size, zp):
    from third_party.vllm_triton_w4a16 import _triton_w4a16_gemm_impl

    _check_group(group_size, K, (32, 64, 128, 256))

    def setup(p: Problem):
        qweight, qzeros, scales = n_packed_pack(p)
        return {"x": p.fresh_x(), "qw": qweight, "qz": qzeros if zp else None, "s": scales}

    def run(s):
        return _triton_w4a16_gemm_impl(s["x"], s["qw"], s["s"], s["qz"], group_size, 8)

    return setup, run


def _awq_setup(p: Problem):
    if not p.zp:
        raise Unsupported("AWQ format always carries zero points (run with zp)")
    qweight, qzeros, scales = n_packed_pack(p, order=AWQ_REVERSE_ORDER)
    return {"x": p.fresh_x(), "qw": qweight, "qz": qzeros, "s": scales}


@register(
    "vllm_awq_triton_gemm",
    kind="wNa16",
    bits=(4,),
    source=f"vllm@{VLLM_REV} layers/quantization/awq_triton.py awq_gemm_triton (split_k=8, as auto_awq.py)",
)
def vllm_awq_triton_gemm(M, N, K, dtype, bits, group_size, zp):
    from third_party.vllm_awq_triton import awq_gemm_triton

    _check_group(group_size, K)
    if not zp:
        raise Unsupported("AWQ format always carries zero points")

    def run(s):
        return awq_gemm_triton(s["x"], s["qw"], s["s"], s["qz"], 8)

    return _awq_setup, run


@register(
    "vllm_awq_dequant_mm",
    kind="wNa16",
    bits=(4,),
    source=f"vllm@{VLLM_REV} awq_dequantize_triton + torch.matmul (vLLM AWQ path for M>=256)",
)
def vllm_awq_dequant_mm(M, N, K, dtype, bits, group_size, zp):
    from third_party.vllm_awq_triton import awq_dequantize_triton

    _check_group(group_size, K)
    if not zp:
        raise Unsupported("AWQ format always carries zero points")

    def run(s):
        w = awq_dequantize_triton(s["qw"], s["s"], s["qz"])  # [K, N]
        return torch.matmul(s["x"], w)

    return _awq_setup, run


# ==========================================================================
# 3. Conch mixed-precision Triton GEMM (vLLM ConchLinearKernel backend)
# ==========================================================================
def _conch_version():
    try:
        from importlib.metadata import version

        return version("conch-triton-kernels")
    except Exception:
        return "?"


@register(
    "conch_mixed_precision",
    kind="wNa16",
    bits=(4, 8),
    source=f"conch-triton-kernels=={_conch_version()} mixed_precision_gemm (vLLM ConchLinearKernel layout)",
)
def conch_mixed_precision(M, N, K, dtype, bits, group_size, zp):
    from conch.ops.quantization.gemm import mixed_precision_gemm

    def setup(p: Problem):
        qweight, _, scales = gptq_pack(p)
        zeros = p.zeros.t().contiguous().to(torch.uint8) if zp else None
        return {"x": p.fresh_x(), "qw": qweight, "s": scales, "z": zeros}

    bias = 0 if zp else 2 ** (bits - 1)

    def run(s):
        return mixed_precision_gemm(s["x"], s["qw"], s["s"], s["z"], bits, bias, group_size)

    return setup, run


# ==========================================================================
# 4. vLLM exllama gptq_gemm (HIP), built standalone from vLLM csrc
# ==========================================================================
_GPTQ_LOADED = False


def load_vllm_gptq_ext():
    global _GPTQ_LOADED
    if _GPTQ_LOADED:
        return
    from torch.utils.cpp_extension import load

    d = os.path.join(_HERE, "ext", "vllm_gptq")
    build = os.path.join(d, ".build")
    os.makedirs(build, exist_ok=True)
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")
    load(
        name="vllm_gptq_baseline",
        sources=[os.path.join(d, "src", "bindings.cu"), os.path.join(d, "src", "quantization", "gptq", "q_gemm.cu")],
        build_directory=build,
        extra_include_paths=[os.path.join(d, "src")],
        extra_cuda_cflags=["-O3", "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__"],
        is_python_module=False,
        verbose=False,
    )
    _GPTQ_LOADED = True


@register(
    "vllm_exllama_gptq",
    kind="wNa16",
    bits=(4, 8),
    source=f"vllm@{VLLM_REV} csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros)",
)
def vllm_exllama_gptq(M, N, K, dtype, bits, group_size, zp):
    if dtype != torch.float16:
        raise Unsupported("exllama gptq_gemm is fp16-only")
    load_vllm_gptq_ext()
    ops = torch.ops.vllm_gptq_baseline

    def setup(p: Problem):
        qweight, qzeros, scales = gptq_pack(p, zero_offset=-1)  # GPTQ v1 stores z-1
        ops.gptq_shuffle(qweight, bits)
        return {"x": p.fresh_x(), "qw": qweight, "qz": qzeros, "s": scales}

    def run(s):
        return ops.gptq_gemm(s["x"], s["qw"], s["qz"], s["s"], True, False, bits)

    return setup, run


# ==========================================================================
# 5. Generic "dequant to fp16/bf16, then hipBLASLt" fallback
# ==========================================================================
@triton.jit
def _dequant_kernel(
    q_ptr,  # [N, K // PF] int32, K packed sequentially
    s_ptr,  # [N, K // G] dtype
    z_ptr,  # [N, K // G] uint8 (unused if not HAS_ZP)
    w_ptr,  # [N, K] dtype
    N,
    K,
    G,
    BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program = BLOCK_N rows x one quant group (BLOCK_K == G) along K.
    PF: tl.constexpr = 32 // BITS
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_pk = pid_k * (BLOCK_K // PF) + tl.arange(0, BLOCK_K // PF)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    nmask = offs_n < N
    packed = tl.load(q_ptr + offs_n[:, None] * (K // PF) + offs_pk[None, :], mask=nmask[:, None], other=0)
    # replicate each int32 PF times along K (as in vLLM triton_w4a16)
    v = tl.interleave(packed, packed)
    v = tl.interleave(v, v)
    if PF == 8:
        v = tl.interleave(v, v)
    shifts = (tl.arange(0, BLOCK_K) % PF) * BITS
    v = (v >> shifts[None, :]) & ((1 << BITS) - 1)
    meta_off = offs_n * (K // G) + pid_k
    s = tl.load(s_ptr + meta_off, mask=nmask, other=0.0).to(tl.float32)
    if HAS_ZP:
        z = tl.load(z_ptr + meta_off, mask=nmask, other=0).to(tl.int32)
    else:
        z = tl.full((BLOCK_N,), ZP_BIAS, tl.int32)
    w = ((v - z[:, None]).to(tl.float32) * s[:, None]).to(w_ptr.dtype.element_ty)
    tl.store(w_ptr + offs_n[:, None] * K + offs_k[None, :], w, mask=nmask[:, None])


@register(
    "dequant_triton+hipblaslt",
    kind="wNa16",
    bits=(4, 8),
    source="this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together",
)
def dequant_hipblaslt(M, N, K, dtype, bits, group_size, zp):
    if group_size not in (32, 64, 128, 256) or K % group_size:
        raise Unsupported("dequant kernel handles group_size in {32,64,128,256}")

    def setup(p: Problem):
        return {
            "x": p.fresh_x(),
            "q": pack_last_dim(p.q, bits),
            "s": p.scales.clone(),
            "z": p.zeros.to(torch.uint8) if zp else None,
        }

    # One shared dequant workspace (as a framework would keep), not per rotating copy.
    w = torch.empty((N, K), dtype=dtype, device="cuda")
    BN, BK = 32, group_size
    grid = (triton.cdiv(N, BN), K // BK)

    def run(s):
        z = s["z"] if zp else s["s"]
        _dequant_kernel[grid](s["q"], s["s"], z, w, N, K, group_size, bits, zp, 2 ** (bits - 1), BN, BK)
        return F.linear(s["x"], w)

    return setup, run


# ==========================================================================
# 6. AITER weight-only kernels (Triton)
# ==========================================================================
@register(
    "aiter_a16w8_int8",
    kind="wNa16",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G)",
)
def aiter_a16w8_int8(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale

    if zp:
        raise Unsupported("symmetric only (no zero-point input)")

    def setup(p: Problem):
        w = (p.q - p.zeros.repeat_interleave(group_size, dim=1)).to(torch.int8)
        return {
            "x": p.fresh_x(),
            "w": w,
            "s": p.scales.float().contiguous(),  # [N, K // G] -> GROUP_N = 1, GROUP_K = G
            "weight_bytes": nominal_weight_bytes(N, K, 8, group_size, False, scale_bytes=4),
        }

    def run(s):
        return gemm_a16w8_blockscale(s["x"], s["w"], s["s"], dtype=dtype)

    return setup, run


def _fp8_dtype():
    return torch.float8_e4m3fn  # OCP e4m3 on gfx950


@register(
    "aiter_a16w8_fp8_block128",
    kind="wNa16",
    bits=(8,),
    source=f"aiter@{AITER_REV} gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant)",
)
def aiter_a16w8_fp8(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale

    if zp:
        raise Unsupported("fp8 weights: no zero point")
    if N % 128 or K % 128:
        raise Unsupported("needs N, K multiple of 128")

    def setup(p: Problem):
        w = p.w_dequant(torch.float32).view(N // 128, 128, K // 128, 128)
        amax = w.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
        scale = amax / torch.finfo(_fp8_dtype()).max
        wq = (w / scale).to(_fp8_dtype())
        w_ref = (wq.float() * scale).view(N, K)
        return {
            "x": p.fresh_x(),
            "w": wq.view(N, K).contiguous(),
            "s": scale.view(N // 128, K // 128).contiguous(),
            "w_ref": w_ref,
            "weight_bytes": N * K + (N // 128) * (K // 128) * 4,
        }

    def run(s):
        return gemm_a16w8_blockscale(s["x"], s["w"], s["s"], dtype=dtype)

    return setup, run


# --- MXFP4 helpers (from aiter op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py)
_MXFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def mxfp4_dequant(x_fp4: torch.Tensor, scales_e8m0: torch.Tensor) -> torch.Tensor:
    x = x_fp4.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=x.device)
    vals = lut[x.long()]
    sc = 2 ** (scales_e8m0.to(torch.float32) - 127)
    return vals * sc.repeat_interleave(32, dim=-1)


def _mxfp4_quant(t: torch.Tensor):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    q, s = dynamic_mxfp4_quant(t)
    return q.contiguous(), s.contiguous()


@register(
    "aiter_a16wfp4",
    kind="wNa16",
    bits=(4,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a16wfp4.py (MXFP4 W; NOTE: X is MXFP4-quantized in-kernel)",
    tol=3e-2,
)
def aiter_a16wfp4(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if zp:
        raise Unsupported("MXFP4: no zero point")

    def setup(p: Problem):
        wq, ws = _mxfp4_quant(p.w_dequant(torch.bfloat16))
        x = p.fresh_x()
        xq, xs = _mxfp4_quant(x)
        return {
            "x": x,
            "w": wq,
            "ws": ws,
            "w_ref": mxfp4_dequant(wq, ws),
            "x_ref": mxfp4_dequant(xq, xs),
            "weight_bytes": N * K // 2 + N * K // 32,
        }

    def run(s):
        return gemm_a16wfp4(s["x"], s["w"], s["ws"], dtype=dtype)

    return setup, run


# ==========================================================================
# 7. Optional activation-quantized references (GEMM only; activation
#    quantization is done in setup and NOT timed)
# ==========================================================================
@register(
    "aiter_afp4wfp4",
    kind="w4a4",
    bits=(4,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_afp4wfp4.py (MXFP4 x MXFP4, X pre-quantized, quant not timed)",
    tol=3e-2,
)
def aiter_afp4wfp4(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    if zp:
        raise Unsupported("MXFP4: no zero point")

    def setup(p: Problem):
        wq, ws = _mxfp4_quant(p.w_dequant(torch.bfloat16))
        xq, xs = _mxfp4_quant(p.x)
        return {
            "x": xq,
            "xs": xs,
            "w": wq,
            "ws": ws,
            "w_ref": mxfp4_dequant(wq, ws),
            "x_ref": mxfp4_dequant(xq, xs),
            "weight_bytes": N * K // 2 + N * K // 32,
            "act_bytes": M * K // 2 + M * K // 32,
        }

    def run(s):
        return gemm_afp4wfp4(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


def _rowwise_fp8(t: torch.Tensor):
    amax = t.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / torch.finfo(_fp8_dtype()).max
    q = (t.float() / scale).to(_fp8_dtype())
    return q, scale


@register(
    "torch_scaled_mm_fp8",
    kind="w8a8",
    bits=(8,),
    source=f"torch._scaled_mm (hipBLASLt) FP8-e4m3 rowwise X / per-channel W, torch {torch.__version__}, X quant not timed",
)
def torch_scaled_mm_fp8(M, N, K, dtype, bits, group_size, zp):
    if zp:
        raise Unsupported("fp8: no zero point")

    def setup(p: Problem):
        wq, ws = _rowwise_fp8(p.w_dequant(torch.float32))
        xq, xs = _rowwise_fp8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": wq,
            "ws": ws.t().contiguous(),  # [1, N]
            "w_ref": wq.float() * ws,
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K + N * 4,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        return torch._scaled_mm(s["x"], s["w"].t(), scale_a=s["xs"], scale_b=s["ws"], out_dtype=dtype)

    return setup, run


@register(
    "aiter_a8w8_int8",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a8w8.py INT8 per-token X / per-channel W, X quant not timed",
)
def aiter_a8w8_int8(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    if zp:
        raise Unsupported("int8 W8A8 symmetric only")

    def q8(t):
        amax = t.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        return torch.round(t.float() / scale).clamp(-127, 127).to(torch.int8), scale

    def setup(p: Problem):
        wq, ws = q8(p.w_dequant(torch.float32))
        xq, xs = q8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": wq,
            "ws": ws.t().contiguous(),
            "w_ref": wq.float() * ws,
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K + N * 4,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        return gemm_a8w8(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


@register(
    "aiter_gemm_a4w4_asm",
    kind="w4a4",
    bits=(4,),
    source=f"aiter@{AITER_REV} ops/gemm_op_a4w4.py gemm_a4w4 (gfx950 asm MXFP4 x MXFP4, preshuffled W; X quant not timed)",
    tol=3e-2,
)
def aiter_gemm_a4w4_asm(M, N, K, dtype, bits, group_size, zp):
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    if zp:
        raise Unsupported("MXFP4: no zero point")
    if dtype != torch.bfloat16:
        raise Unsupported("asm a4w4 kernels emit bf16 only")
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)

    def setup(p: Problem):
        w = p.w_dequant(torch.bfloat16)
        wq, ws = quant(w, shuffle=False)
        _, ws_sh = quant(w, shuffle=True)
        xq, xs = quant(p.x, shuffle=False)
        _, xs_sh = quant(p.x, shuffle=True)
        return {
            "x": xq,
            "xs": xs_sh,
            "w": shuffle_weight(wq, layout=(16, 16)),
            "ws": ws_sh,
            "w_ref": mxfp4_dequant(wq.view(torch.uint8), ws.view(torch.uint8)[:N, : K // 32]),
            "x_ref": mxfp4_dequant(xq.view(torch.uint8), xs.view(torch.uint8)[:M, : K // 32]),
            "weight_bytes": N * K // 2 + N * K // 32,
            "act_bytes": M * K // 2 + M * K // 32,
        }

    def run(s):
        return aiter.gemm_a4w4(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype, bpreshuffle=True)

    return setup, run


@register(
    "aiter_a8w8_fp8",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a8w8.py FP8-e4m3 per-token X / per-channel W, X quant not timed",
)
def aiter_a8w8_fp8(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    if zp:
        raise Unsupported("fp8: no zero point")

    def setup(p: Problem):
        wq, ws = _rowwise_fp8(p.w_dequant(torch.float32))
        xq, xs = _rowwise_fp8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": wq,
            "ws": ws.t().contiguous(),
            "w_ref": wq.float() * ws,
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K + N * 4,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        return gemm_a8w8(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


@register(
    "aiter_a8w8_fp8_ck",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/gemm_op_a8w8.py gemm_a8w8_CK (CK, FP8-e4m3 per-token X / per-channel W, tuned-config lookup), X quant not timed",
)
def aiter_a8w8_fp8_ck(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

    if zp:
        raise Unsupported("fp8: no zero point")

    def setup(p: Problem):
        wq, ws = _rowwise_fp8(p.w_dequant(torch.float32))
        xq, xs = _rowwise_fp8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": wq,
            "ws": ws.t().contiguous(),
            "w_ref": wq.float() * ws,
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K + N * 4,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        return gemm_a8w8_CK(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


def _block128_fp8(w: torch.Tensor):
    """FP8 weight with one FP32 scale per 128 x 128 block -> (wq [N, K], scale [N/128, K/128], w_ref)."""
    N, K = w.shape
    wb = w.float().view(N // 128, 128, K // 128, 128)
    scale = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / torch.finfo(_fp8_dtype()).max
    wq = (wb / scale).to(_fp8_dtype())
    return wq.view(N, K).contiguous(), scale.view(N // 128, K // 128).contiguous(), (wq.float() * scale).view(N, K)


def _group128_fp8(x: torch.Tensor):
    """FP8 activation with one FP32 scale per row and 128-wide K group -> (xq, scale [M, K/128], x_ref)."""
    M, K = x.shape
    xg = x.float().view(M, K // 128, 128)
    scale = xg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / torch.finfo(_fp8_dtype()).max
    xq = (xg / scale).to(_fp8_dtype())
    return xq.view(M, K).contiguous(), scale.view(M, K // 128).contiguous(), (xq.float() * scale).view(M, K)


def _blockscale_factory(run_fn):
    def setup(p: Problem):
        wq, ws, w_ref = _block128_fp8(p.w_dequant(torch.float32))
        xq, xs, x_ref = _group128_fp8(p.x)
        N, K = wq.shape
        return {
            "x": xq,
            "xs": xs,
            "w": wq,
            "ws": ws,
            "w_ref": w_ref,
            "x_ref": x_ref,
            "weight_bytes": N * K + (N // 128) * (K // 128) * 4,
            "act_bytes": xq.numel() + xs.numel() * 4,
        }

    return setup, run_fn


@register(
    "aiter_a8w8_blockscale_fp8",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a8w8_blockscale.py FP8 X 1x128 / W 128x128 scales, X quant "
    "not timed; M = 1 runs as M = 2 (zero row) with the M=16 config: Triton's M==1 specialization of this kernel "
    "crashes TritonAMDGPUCanonicalizePointers",
)
def aiter_a8w8_blockscale_fp8(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic import gemm_a8w8_blockscale as mod

    if zp:
        raise Unsupported("fp8: no zero point")
    if N % 128 or K % 128:
        raise Unsupported("needs N, K multiple of 128")
    backend = "gluon" if mod._is_gluon_available() else "triton"
    config, _ = mod._get_config(max(M, 16), N, K, backend=backend)
    setup0, _ = _blockscale_factory(None)

    def setup(p: Problem):
        s = setup0(p)
        if M == 1:  # pad to two rows; the timed kernel reads one extra (zero) row
            s["x_pad"] = torch.cat([s["x"], torch.zeros_like(s["x"])])
            s["xs_pad"] = torch.cat([s["xs"], torch.ones_like(s["xs"])])
        return s

    def run(s):
        if M == 1:
            y = mod.gemm_a8w8_blockscale(s["x_pad"], s["w"], s["xs_pad"], s["ws"], dtype=dtype, config=dict(config),
                                         backend=backend)
            return y[:1]
        return mod.gemm_a8w8_blockscale(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype, config=dict(config),
                                        backend=backend)

    return setup, run


@register(
    "aiter_a8w8_blockscale_fp8_ck",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/gemm_op_a8w8.py gemm_a8w8_blockscale (CK / CK-tile per tuned config) FP8 X 1x128 / W 128x128, X quant not timed",
)
def aiter_a8w8_blockscale_fp8_ck(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.gemm_op_a8w8 import _hip_blockscale_supported, gemm_a8w8_blockscale

    if zp:
        raise Unsupported("fp8: no zero point")
    if N % 128 or K % 128:
        raise Unsupported("needs N, K multiple of 128")
    if not _hip_blockscale_supported():
        raise Unsupported("no CK blockscale code object for this arch (would fall back to triton)")
    return _blockscale_factory(lambda s: gemm_a8w8_blockscale(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype))


@register(
    "aiter_a8w8_fp8_bpreshuffle",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/gemm_op_a8w8.py gemm_a8w8_bpreshuffle (CK / CK-tile / asm per tuned config, (16,16)-shuffled "
    "FP8 W) per-token X / per-channel W, X quant not timed",
)
def aiter_a8w8_fp8_bpreshuffle(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    if zp:
        raise Unsupported("fp8: no zero point")

    def setup(p: Problem):
        wq, ws = _rowwise_fp8(p.w_dequant(torch.float32))
        xq, xs = _rowwise_fp8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": shuffle_weight(wq, layout=(16, 16)),
            "ws": ws.t().contiguous(),
            "w_ref": wq.float() * ws,
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K + N * 4,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        return gemm_a8w8_bpreshuffle(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


@register(
    "aiter_a8w8_blockscale_fp8_bpreshuffle",
    kind="w8a8",
    bits=(8,),
    source=f"aiter@{AITER_REV} ops/gemm_op_a8w8.py gemm_a8w8_blockscale_bpreshuffle (CK / asm per tuned config, "
    "(16,16)-shuffled W, column-major X scales) FP8 X 1x128 / W 128x128, X quant not timed",
)
def aiter_a8w8_blockscale_fp8_bpreshuffle(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.gemm_op_a8w8 import _hip_blockscale_supported, gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    if zp:
        raise Unsupported("fp8: no zero point")
    if N % 128 or K % 128:
        raise Unsupported("needs N, K multiple of 128")
    if not _hip_blockscale_supported():
        raise Unsupported("no CK blockscale code object for this arch")
    setup0, _ = _blockscale_factory(None)

    def setup(p: Problem):
        s = setup0(p)
        s["w"] = shuffle_weight(s["w"], layout=(16, 16))
        s["xs"] = s["xs"].t().contiguous().view(*s["xs"].shape)  # column-major [M, K/128]
        return s

    def run(s):
        return gemm_a8w8_blockscale_bpreshuffle(s["x"], s["w"], s["xs"], s["ws"], dtype=dtype)

    return setup, run


@register(
    "aiter_a8wfp4",
    kind="w4a8",
    bits=(4,),
    source=f"aiter@{AITER_REV} ops/triton/gemm/basic/gemm_a8wfp4.py FP8 per-token X x MXFP4 W, X quant not timed",
    tol=3e-2,
)
def aiter_a8wfp4(M, N, K, dtype, bits, group_size, zp):
    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4

    if zp:
        raise Unsupported("MXFP4: no zero point")

    def setup(p: Problem):
        wq, ws = _mxfp4_quant(p.w_dequant(torch.bfloat16))
        xq, xs = _rowwise_fp8(p.x)
        return {
            "x": xq,
            "xs": xs.contiguous(),
            "w": wq,
            "ws": ws,
            "y": torch.empty(M, N, dtype=dtype, device="cuda"),
            "w_ref": mxfp4_dequant(wq, ws),
            "x_ref": xq.float() * xs,
            "weight_bytes": N * K // 2 + N * K // 32,
            "act_bytes": M * K + M * 4,
        }

    def run(s):
        gemm_a8wfp4(s["x"], s["w"], s["y"], s["xs"], s["ws"], dtype=dtype)
        return s["y"]

    return setup, run


# ==========================================================================
# 8. Humming (this repo) through its public API: HummingLayer.load_from_tensors
#    -> transform() -> layer(x), i.e. the gfx950 MFMA GEMM behind humming_forward.
# ==========================================================================
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_HUMMING_LAYER_CACHE: dict = {}


@register(
    "humming",
    kind="wNa16",
    bits=(1, 2, 3, 4, 5, 6, 7, 8),
    source=f"humming@{_git_rev(_REPO_ROOT)} HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh)",
)
def humming_layer(M, N, K, dtype, bits, group_size, zp):
    import copy
    import sys

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from humming import ops as humming_ops
    from humming.layer import HummingLayer

    if K % group_size or group_size % 32:
        raise Unsupported(f"group_size {group_size}")

    def build(p: Problem):
        weight_config = {"dtype": f"uint{bits}" if zp else f"int{bits}", "has_zero_point": zp}
        if group_size != K:
            weight_config["group_size"] = group_size
        layer = HummingLayer(shape_n=N, shape_k=K, weight_config=weight_config, torch_dtype=dtype).cuda()
        tensors = {
            "weight": humming_ops.pack_weight(p.q.contiguous(), bits),
            "weight_scale": p.scales.contiguous() if group_size != K else p.scales.reshape(N, 1).contiguous(),
        }
        if zp:  # Humming packs zero points along N: [N * bits / 32, groups]
            zeros = p.zeros.t().contiguous()
            tensors["zero_point"] = humming_ops.pack_weight(zeros, bits).t().contiguous()
        layer.load_from_tensors(tensors)
        layer.transform()
        return layer

    def setup(p: Problem):
        # The harness slices one Problem per shape for every M: transform once per weight, then the
        # rotating copies (and the other M values) deep-copy the transformed layer.
        key = (p.q.data_ptr(), p.scales.data_ptr(), p.zeros.data_ptr(), N, K, bits, group_size, zp, dtype)
        if _HUMMING_LAYER_CACHE.get("key") != key:
            _HUMMING_LAYER_CACHE.clear()
            _HUMMING_LAYER_CACHE.update(key=key, layer=build(p))
            torch.cuda.empty_cache()
        layer = copy.deepcopy(_HUMMING_LAYER_CACHE["layer"])
        params = dict(layer.named_parameters())
        state = {"x": p.fresh_x(), "layer": layer, **params}
        state["weight_bytes"] = sum(t.numel() * t.element_size() for t in params.values())
        return state

    def run(s):
        return s["layer"](s["x"])

    return setup, run


# ==========================================================================
# 9. Humming low-bit activation GEMM (gfx950, humming/kernel/humming_mfma_f8.cuh) through
#    HummingLayer. The weight is re-quantized from the canonical problem by
#    load_from_unquantized; the activation is quantized in setup with Humming's own
#    may_quant_input and the timed call is the GEMM only: layer(q, input_scale=s).
#    ``humming_w8a8_fp8_e2e`` times quantization + GEMM (layer(x)).
# ==========================================================================
_FP8_TOKEN = {"dtype": "float8e4m3"}
_FP8_G128 = {"dtype": "float8e4m3", "group_size": 128}
_INT8_TOKEN = {"dtype": "int8"}
_MXFP4 = {"dtype": "float4e2m1", "group_size": 32, "scale_dtype": "float8e8m0"}
_HUMMING_F8_CACHE: dict = {}


def _humming_dequant_act(q: torch.Tensor, s: torch.Tensor, K: int) -> torch.Tensor:
    """FP32 view of Humming's quantized activation (q, input_scale)."""
    if q.dtype == torch.uint8:  # packed E2M1, low nibble first
        lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=q.device)
        qa = lut[torch.stack([q & 0xF, q >> 4], dim=-1).reshape(q.size(0), K).long()]
    else:
        qa = q.float()
    if s.numel() == 1:
        return qa * s.float()
    if s.dtype in (torch.int32, torch.uint8, torch.float8_e8m0fnu):  # E8M0 bytes per row
        e = s.contiguous().view(torch.uint8)[:, : K // 32].float()
        return qa * torch.exp2(e - 127).repeat_interleave(K // e.size(1), dim=1)
    if s.size(1) == 1:
        return qa * s.float()
    return qa * s.float()[:, : K // 128].repeat_interleave(128, dim=1)  # FP32 1x128 group scales


def _humming_f8_factory(weight_config_fn, input_config, e2e=False):
    def factory(M, N, K, dtype, bits, group_size, zp):
        import copy
        import sys

        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from humming.forward import may_quant_input
        from humming.layer import HummingLayer

        weight_config = weight_config_fn(bits, group_size, zp, K)

        def build(p: Problem):
            layer = HummingLayer(shape_n=N, shape_k=K, weight_config=dict(weight_config), torch_dtype=dtype,
                                 input_config=dict(input_config)).cuda()
            layer.load_from_unquantized(p.w_dequant(dtype))
            w_ref = layer.weight_schema.dequant_tensors(dict(layer.state_dict())).float()
            layer.transform()
            return layer, w_ref

        def setup(p: Problem):
            key = (p.q.data_ptr(), p.scales.data_ptr(), N, K, dtype, repr(weight_config), repr(input_config))
            if _HUMMING_F8_CACHE.get("key") != key:
                _HUMMING_F8_CACHE.clear()
                layer, w_ref = build(p)
                _HUMMING_F8_CACHE.update(key=key, layer=layer, w_ref=w_ref)
                torch.cuda.empty_cache()
            layer = copy.deepcopy(_HUMMING_F8_CACHE["layer"])
            params = dict(layer.named_parameters())
            x = p.fresh_x()
            q, s = may_quant_input(layer.humming_config, x)
            state = {"layer": layer, **params, "x": x, "q": q, "s": s,
                     "w_ref": _HUMMING_F8_CACHE["w_ref"], "x_ref": _humming_dequant_act(q, s, K)}
            state["weight_bytes"] = sum(t.numel() * t.element_size() for t in params.values())
            state["act_bytes"] = q.numel() * q.element_size() + s.numel() * s.element_size()
            return state

        if e2e:
            def run(s):
                return s["layer"](s["x"])
        else:
            def run(s):
                return s["layer"](s["q"], input_scale=s["s"])

        return setup, run

    return factory


def _int_weight(bits, group_size, zp, K):
    config = {"dtype": f"uint{bits}" if zp else f"int{bits}", "has_zero_point": zp}
    if group_size != K:
        config["group_size"] = group_size
    return config


def _no_zp(config):
    def fn(bits, group_size, zp, K):
        if zp:
            raise Unsupported("float / int8 weights: symmetric only")
        return dict(config)

    return fn


_HUMMING_SRC = f"humming@{_git_rev(_REPO_ROOT)} HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh)"
for _name, _kind, _bits, _wfn, _ic, _desc, _tol, _e2e in [
    ("humming_w8a8_fp8", "w8a8", (8,), _no_zp({"dtype": "float8e4m3"}), _FP8_TOKEN,
     "FP8 per-token X x FP8 per-channel W, X quant not timed", 2e-2, False),
    ("humming_w8a8_fp8_e2e", "w8a8", (8,), _no_zp({"dtype": "float8e4m3"}), _FP8_TOKEN,
     "FP8 per-token X x FP8 per-channel W, dynamic X quant timed", 2e-2, True),
    ("humming_w8a8_fp8_block128", "w8a8", (8,), _no_zp({"dtype": "float8e4m3", "group_size": 128, "group_size_n": 128}),
     _FP8_G128, "FP8 X 1x128 x FP8 W 128x128 block scales, X quant not timed", 2e-2, False),
    ("humming_w4a8_int", "w4a8", (4,), _int_weight, _FP8_TOKEN,
     "FP8 per-token X x INT4 (group G, optional zp) W dequantized to FP8, X quant not timed", 3e-2, False),
    ("humming_w4a8_mxfp4", "w4a8", (4,), _no_zp(_MXFP4), _FP8_TOKEN,
     "FP8 per-token X x MXFP4 W (scaled MFMA f8f6f4), X quant not timed", 3e-2, False),
    ("humming_w4a4_mxfp4", "w4a4", (4,), _no_zp(_MXFP4), _MXFP4,
     "MXFP4 X x MXFP4 W (scaled MFMA f8f6f4), X quant not timed", 3e-2, False),
    ("humming_w8a8_int8", "w8a8", (8,), _no_zp({"dtype": "int8"}), _INT8_TOKEN,
     "INT8 per-token X x INT8 per-channel W, X quant not timed", 2e-2, False),
    ("humming_w4a8_int_i8", "w4a8", (4,), _int_weight, _INT8_TOKEN,
     "INT8 per-token X x INT4 (group G, optional zp) W, X quant not timed", 3e-2, False),
]:
    register(_name, kind=_kind, bits=_bits, source=f"{_HUMMING_SRC}: {_desc}", tol=_tol)(
        _humming_f8_factory(_wfn, _ic, e2e=_e2e))
