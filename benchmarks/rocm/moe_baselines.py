"""MoE layer implementations for ``bench_moe.py``.

Every implementation computes the routed-expert part of an MoE block

    out[t] = sum_j  w[t, j] * W2[e_tj] @ (silu(W1g[e_tj] x_t) * W1u[e_tj] x_t)

from ``x [T, H]``, ``topk_ids [T, top_k]`` (int32) and ``topk_weights [T, top_k]`` (fp32), i.e.
routing preparation (align / sort) + w13 GEMM + SiLU-and-mul + w2 GEMM + weighted top-k reduction.
The router itself (gate GEMM + top-k) is excluded. ``w13`` is ``[E, 2I, H]`` (gate rows first),
``w2`` is ``[E, H, I]``.

Each implementation quantizes the shared bf16/fp16 master weights in its own format and exposes
``dequant(experts)`` so the harness can check it against an FP32 reference on *its own* weights.

Implementations:

* ``humming``          Humming MFMA MoE kernel, indexed layout (the vLLM path):
                       ``humming.ops.moe_align_block_size`` + w13 (top_k) + w2 (top_k = 1).
* ``triton_wna16``     vLLM ``fused_moe_kernel_gptq_awq`` (int4_w4a16 / int8_w8a16), the kernel vLLM
                       runs for GPTQ/AWQ MoE on ROCm, with vLLM's default config.
* ``triton_wna16_tuned`` same kernel, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) searched per point.
* ``triton_fp16``      vLLM ``fused_moe_kernel`` unquantized, vLLM default config (incl. its
                       naive-block-assignment fast path for tiny batches).
* ``aiter_bf16``       AITER ``fused_moe`` unquantized (CK 2-stage, preshuffled), bf16 only.
* ``aiter_a16wi4``     AITER ``fused_moe`` a16wi4 (per_1x32, FlyDSL 2-stage): sym int4, group 32,
                       bf16 only.

Routing for humming / triton uses ``humming.ops.moe_align_block_size`` (one-workgroup HIP kernel,
same contract as vLLM's ``_C`` op, which does not load here, at a similar cost: ~4 us at decode);
its cost is included and reported separately as ``align_us``. AITER uses its own ``moe_sorting``
kernel (included).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Callable

import torch
import triton
import triton.language as tl


class Unsupported(Exception):
    pass


@dataclasses.dataclass
class Routing:
    x: torch.Tensor  # [T, H]
    topk_ids: torch.Tensor  # [T, top_k] int32
    topk_weights: torch.Tensor  # [T, top_k] fp32

    @property
    def tokens(self) -> int:
        return self.x.size(0)

    @property
    def top_k(self) -> int:
        return self.topk_ids.size(1)


# Glue kernels. vLLM runs one fused kernel for each (``_C.silu_and_mul``, ``moe_sum`` / the
# routed-weight multiply in the w2 GEMM); single-kernel Triton equivalents keep the harness from
# charging 2-4 extra torch launches per step to the implementations that need glue.
@triton.jit
def _silu_and_mul_kernel(h_ptr, out_ptr, inter, stride_h, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    gate = tl.load(h_ptr + row * stride_h + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(h_ptr + row * stride_h + inter + cols, mask=mask, other=0.0).to(tl.float32)
    act = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + row * stride_out + cols, act.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _weighted_topk_sum_kernel(y_ptr, w_ptr, out_ptr, hidden, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        weight = tl.load(w_ptr + token * TOP_K + j)
        row = tl.load(y_ptr + (token * TOP_K + j) * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        acc += weight * row
    tl.store(out_ptr + token * hidden + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(h: torch.Tensor) -> torch.Tensor:
    """[M, 2I] (gate | up) -> silu(gate) * up, [M, I]."""
    rows, inter = h.size(0), h.size(-1) // 2
    out = torch.empty(rows, inter, dtype=h.dtype, device=h.device)
    if rows:
        block = 1024
        _silu_and_mul_kernel[(rows, triton.cdiv(inter, block))](h, out, inter, h.stride(0), out.stride(0), BLOCK=block)
    return out


def weighted_topk_sum(out: torch.Tensor, routing: Routing) -> torch.Tensor:
    """out: [T * top_k, H] rows ``t * top_k + j`` -> sum_j w[t, j] * out[t * top_k + j], [T, H]."""
    tokens, top_k = routing.topk_ids.shape
    out = out.contiguous()
    hidden = out.size(-1)
    result = torch.empty(tokens, hidden, dtype=routing.x.dtype, device=out.device)
    if tokens:
        block = 1024
        _weighted_topk_sum_kernel[(tokens, triton.cdiv(hidden, block))](
            out, routing.topk_weights.contiguous(), result, hidden, TOP_K=top_k, BLOCK=block)
    return result


def sym_group_quant(w: torch.Tensor, bits: int, group_size: int, chunk: int = 8):
    """Symmetric group quantization of [E, N, K]: q in [0, 2^bits) with implicit zero 2^(bits-1).

    Returns (q uint8 [E, N, K], scales [E, N, K // G] in w.dtype). Done per expert chunk to
    bound the fp32 temporaries.
    """
    num_experts, shape_n, shape_k = w.shape
    qmax = 2 ** (bits - 1) - 1
    q = torch.empty(w.shape, dtype=torch.uint8, device=w.device)
    s = torch.empty((num_experts, shape_n, shape_k // group_size), dtype=w.dtype, device=w.device)
    for e0 in range(0, num_experts, chunk):
        wg = w[e0 : e0 + chunk].float().view(-1, shape_n, shape_k // group_size, group_size)
        scale = (wg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8).to(w.dtype)
        qq = torch.round(wg / scale.float()).clamp(-qmax - 1, qmax) + 2 ** (bits - 1)
        q[e0 : e0 + chunk] = qq.to(torch.uint8).view(-1, shape_n, shape_k)
        s[e0 : e0 + chunk] = scale.squeeze(-1)
    return q, s


def sym_group_dequant(q: torch.Tensor, s: torch.Tensor, bits: int) -> torch.Tensor:
    num_experts, shape_n, shape_k = q.shape
    groups = s.size(-1)
    w = (q.float() - 2 ** (bits - 1)).view(num_experts, shape_n, groups, -1) * s.float().unsqueeze(-1)
    return w.view(num_experts, shape_n, shape_k)


class MoEImpl:
    """``build`` quantizes once; ``prepare`` returns ``run(copy_index) -> [T, H]`` (graph-safe)."""

    name: str = ""
    supports_rotation = True

    def __init__(self, w13: torch.Tensor, w2: torch.Tensor, dtype: torch.dtype, bits: int, group_size: int):
        self.num_experts, self.inter2, self.hidden = w13.shape
        self.inter = self.inter2 // 2
        self.dtype, self.bits, self.group_size = dtype, bits, group_size
        self.copies: list = []
        self.build(w13, w2)

    def build(self, w13, w2):
        raise NotImplementedError

    def add_copies(self, n: int) -> None:
        while len(self.copies) < n:
            self.copies.append(self.clone_weights(self.copies[0]))

    def clone_weights(self, weights):
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in weights)

    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.copies[0] if isinstance(t, torch.Tensor))

    def dequant(self, experts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def prepare(self, routing: Routing, config: dict | None = None) -> Callable[[int], torch.Tensor]:
        raise NotImplementedError

    def describe(self) -> str:
        return ""


# --------------------------------------------------------------------------------------------
# Humming
# --------------------------------------------------------------------------------------------
class HummingMoE(MoEImpl):
    name = "humming"

    def build(self, w13, w2):
        from humming.layer import HummingLayer

        config = {"dtype": f"int{self.bits}", "has_zero_point": False, "group_size": self.group_size}
        layers, self.reference_states = [], []
        for weight in (w13, w2):
            layer = HummingLayer(
                shape_n=weight.size(1),
                shape_k=weight.size(2),
                weight_config=dict(config),
                torch_dtype=self.dtype,
                num_experts=self.num_experts,
            ).cuda()
            layer.load_from_unquantized(weight)
            # Pre-transform quantized tensors, kept for the per-expert dequant reference.
            self.reference_states.append({k: v.clone() for k, v in layer.state_dict().items() if k != "locks"})
            layer.transform()
            layers.append(layer)
        self.copies = [tuple(layers)]

    def clone_weights(self, weights):
        return tuple(copy.deepcopy(layer) for layer in weights)

    def weight_bytes(self) -> int:
        total = 0
        for layer in self.copies[0]:
            total += sum(p.numel() * p.element_size() for n, p in layer.state_dict().items() if n != "locks")
        return total

    def dequant(self, experts):
        out = []
        for layer, state in zip(self.copies[0], self.reference_states):
            sliced = {k: v[experts] for k, v in state.items()}
            out.append(layer.weight_schema.dequant_tensors(sliced).float())
        return tuple(out)

    def block_m(self, valid_shape_m: int) -> int:
        from humming.tune import get_heuristics_config

        layer = self.copies[0][0]
        for low, high, cfg in get_heuristics_config(layer.humming_config, gemm_type="indexed"):
            if low < valid_shape_m <= high:
                return cfg["block_shape"][0]
        raise AssertionError(valid_shape_m)

    def describe(self) -> str:
        return f"int{self.bits} sym g{self.group_size}"

    def prepare(self, routing, config=None):
        from humming import ops

        top_k = routing.top_k
        valid_shape_m = routing.tokens * top_k
        block_m = self.block_m(valid_shape_m)
        self.last_config = f"block_m={block_m}"
        num_experts = self.num_experts
        compute_config = {"gemm_type": "indexed"}

        def align():
            return ops.moe_align_block_size(routing.topk_ids, block_m, num_experts)

        def gemms(index, routed):
            w13, w2 = self.copies[index]
            sorted_ids, expert_ids, num_tokens_padded = routed
            kwargs = dict(
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_padded=num_tokens_padded,
                valid_shape_m=valid_shape_m,
                compute_config=compute_config,
            )
            h = w13(routing.x, top_k=top_k, **kwargs)
            return w2(silu_and_mul(h), top_k=1, **kwargs)

        def run(index):
            return weighted_topk_sum(gemms(index, align()), routing)

        # Stage callables for the GEMM-only breakdown.
        routed = align()
        inter = torch.randn(valid_shape_m, self.inter, dtype=self.dtype, device="cuda")

        def run_w13(index):
            w13 = self.copies[index][0]
            return w13(routing.x, top_k=top_k, sorted_ids=routed[0], expert_ids=routed[1],
                       num_tokens_padded=routed[2], valid_shape_m=valid_shape_m, compute_config=compute_config)

        def run_w2(index):
            w2 = self.copies[index][1]
            return w2(inter, top_k=1, sorted_ids=routed[0], expert_ids=routed[1], num_tokens_padded=routed[2],
                      valid_shape_m=valid_shape_m, compute_config=compute_config)

        run.stages = {"align": lambda index: align(), "w13": run_w13, "w2": run_w2}
        return run


# --------------------------------------------------------------------------------------------
# vLLM Triton fused MoE (copied kernels)
# --------------------------------------------------------------------------------------------
def _tl_compute_type(dtype):
    import triton.language as tl

    return tl.bfloat16 if dtype == torch.bfloat16 else tl.float16


def vllm_wna16_default_config(tokens: int) -> dict:
    """vLLM get_default_config (int4_w4a16 / int8_w8a16 branch) + get_moe_wna16_block_config (triton)."""
    if tokens <= 20:
        config = {"BLOCK_SIZE_M": 16}
    elif tokens <= 40:
        config = {"BLOCK_SIZE_M": 32}
    else:
        config = {"BLOCK_SIZE_M": 64}
    config.update({"GROUP_SIZE_M": 1, "SPLIT_K": 1})
    if tokens == 1:
        config.update({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64})
    else:
        config.update({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32})
    return config


def vllm_unquantized_default_config(tokens: int, num_experts: int) -> dict:
    """vLLM get_default_config, general bf16/fp16 branch, ROCm."""
    if tokens <= 32:
        block_m = 16
    elif tokens <= 96:
        block_m = 32
    elif tokens <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if tokens <= 64 else 128,
        "BLOCK_SIZE_K": 128 if tokens <= 64 else 64,
        "GROUP_SIZE_M": 16 if tokens // max(num_experts, 1) > 128 else 1,
        "SPLIT_K": 1,
        "num_warps": 4 if tokens <= 128 else 8,
        "num_stages": 2,
    }


class TritonWNA16MoE(MoEImpl):
    """vLLM fused_moe_kernel_gptq_awq: B uint8 [E, N, K/2] (int4, low nibble = even k) or
    [E, N, K] (int8), unsigned with implicit zero 8 / 128; scales [E, N, K / G]."""

    name = "triton_wna16"

    def build(self, w13, w2):
        self.master = []
        weights = []
        for w in (w13, w2):
            q, s = sym_group_quant(w, self.bits, self.group_size)
            self.master.append((q, s))
            if self.bits == 4:
                packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
            else:
                packed = q.clone()
            weights += [packed, s.clone()]
        self.copies = [tuple(weights)]

    def dequant(self, experts):
        return tuple(sym_group_dequant(q[experts], s[experts], self.bits) for q, s in self.master)

    def describe(self) -> str:
        return f"int{self.bits} sym g{self.group_size}"

    def gemm(self, a, b, c, scale, routing, routed, config, top_k, mul_routed_weight):
        from third_party.vllm_fused_moe_triton import fused_moe_kernel_gptq_awq
        import triton

        sorted_ids, expert_ids, num_tokens_padded = routed
        em = sorted_ids.size(0)
        if a.size(0) < config["BLOCK_SIZE_M"]:
            em = min(em, a.size(0) * top_k * config["BLOCK_SIZE_M"])
        shape_n, shape_k = b.size(1), a.size(1)
        grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(shape_n, config["BLOCK_SIZE_N"]),)
        fused_moe_kernel_gptq_awq[grid](
            a, b, c, scale, None, routing.topk_weights, sorted_ids, expert_ids, num_tokens_padded,
            shape_n, shape_k, em, a.size(0) * top_k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1), c.stride(-2), c.stride(-1),
            scale.stride(0), scale.stride(2), scale.stride(1), 0, 0, 0,
            block_k_diviable=shape_k % config["BLOCK_SIZE_K"] == 0,
            group_size=self.group_size,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=_tl_compute_type(self.dtype),
            has_zp=False,
            use_int4_w4a16=self.bits == 4,
            use_int8_w8a16=self.bits == 8,
            **config,
        )
        return c

    def configs(self, tokens):
        default = vllm_wna16_default_config(tokens)
        return default, default

    def prepare(self, routing, config=None):
        from humming import ops

        cfg13, cfg2 = config if config is not None else self.configs(routing.tokens)
        assert cfg13["BLOCK_SIZE_M"] == cfg2["BLOCK_SIZE_M"]
        block_m = cfg13["BLOCK_SIZE_M"]
        self.last_config = f"w13={_cfg_str(cfg13)} w2={_cfg_str(cfg2)}"
        tokens, top_k = routing.tokens, routing.top_k
        num_experts, hidden, inter = self.num_experts, self.hidden, self.inter

        def align():
            return ops.moe_align_block_size(routing.topk_ids, block_m, num_experts)

        def w13_gemm(index, routed, a):
            b13, s13 = self.copies[index][:2]
            c = torch.empty(tokens, top_k, 2 * inter, dtype=self.dtype, device="cuda")
            return self.gemm(a, b13, c, s13, routing, routed, cfg13, top_k, False)

        def w2_gemm(index, routed, a):
            b2, s2 = self.copies[index][2:]
            c = torch.empty(tokens, top_k, hidden, dtype=self.dtype, device="cuda")
            return self.gemm(a, b2, c, s2, routing, routed, cfg2, 1, True)

        def run(index):
            routed = align()
            h = w13_gemm(index, routed, routing.x)
            out = w2_gemm(index, routed, silu_and_mul(h.view(tokens * top_k, -1)))
            return out.sum(1)

        routed = align()
        inter_act = torch.randn(tokens * top_k, inter, dtype=self.dtype, device="cuda")
        run.stages = {
            "align": lambda index: align(),
            "w13": lambda index: w13_gemm(index, routed, routing.x),
            "w2": lambda index: w2_gemm(index, routed, inter_act),
        }
        return run


class TritonWNA16TunedMoE(TritonWNA16MoE):
    name = "triton_wna16_tuned"
    # (BLOCK_N, BLOCK_K, num_warps); BLOCK_M is searched jointly (shared by w13 / w2).
    NK_CANDIDATES = [(32, 64, 4), (32, 128, 4), (64, 32, 4), (64, 64, 4), (64, 128, 4), (128, 64, 4),
                     (128, 64, 8), (128, 128, 8)]

    def block_m_candidates(self, tokens, top_k):
        rows_per_expert = tokens * top_k / self.num_experts
        if rows_per_expert <= 16:
            return [16, 32]
        if rows_per_expert <= 64:
            return [16, 32, 64]
        return [32, 64, 128]

    def candidate_configs(self, tokens, top_k):
        for block_m in self.block_m_candidates(tokens, top_k):
            for block_n, block_k, warps in self.NK_CANDIDATES:
                if block_k > self.group_size and block_k % self.group_size:
                    continue
                yield {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
                       "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": warps, "num_stages": 2}


class TritonUnquantizedMoE(MoEImpl):
    """vLLM fused_moe_kernel, unquantized; B is [E, N, K] in the activation dtype."""

    name = "triton_fp16"

    def build(self, w13, w2):
        self.copies = [(w13.to(self.dtype).contiguous(), w2.to(self.dtype).contiguous())]

    def dequant(self, experts):
        return tuple(w[experts].float() for w in self.copies[0])

    def describe(self) -> str:
        return "unquantized"

    def gemm(self, a, b, c, routing, routed, config, top_k, mul_routed_weight):
        from third_party.vllm_fused_moe_triton import fused_moe_kernel
        import triton

        sorted_ids, expert_ids, num_tokens_padded = routed
        config = dict(config)
        block_k = config.pop("BLOCK_SIZE_K")
        if sorted_ids is not None:
            em = sorted_ids.size(0)
            if a.size(0) < config["BLOCK_SIZE_M"]:
                em = min(em, a.size(0) * top_k * config["BLOCK_SIZE_M"])
        else:
            em = a.size(0) * top_k * config["BLOCK_SIZE_M"]
        grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(b.size(1), config["BLOCK_SIZE_N"]),)
        fused_moe_kernel[grid](
            a, b, c, None, None, None, routing.topk_weights, sorted_ids, expert_ids, num_tokens_padded,
            b.size(1), b.size(2), em, a.size(0) * top_k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1), c.stride(-2), c.stride(-1),
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            naive_block_assignment=sorted_ids is None,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=_tl_compute_type(self.dtype),
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            HAS_BIAS=False,
            BLOCK_SIZE_K=block_k,
            SWAP_AB=False,
            USE_TD=False,
            **config,
        )
        return c

    def prepare(self, routing, config=None):
        from humming import ops

        tokens, top_k = routing.tokens, routing.top_k
        cfg = vllm_unquantized_default_config(tokens, self.num_experts)
        naive = tokens * top_k * 4 <= self.num_experts
        self.last_config = _cfg_str(cfg) + (" naive" if naive else "")
        num_experts, hidden = self.num_experts, self.hidden

        def align():
            if naive:
                padded = torch.full((1,), routing.topk_ids.numel() * cfg["BLOCK_SIZE_M"], dtype=torch.int32,
                                    device="cuda")
                return None, routing.topk_ids.view(-1), padded
            return ops.moe_align_block_size(routing.topk_ids, cfg["BLOCK_SIZE_M"], num_experts)

        def run(index):
            w13, w2 = self.copies[index]
            routed = align()
            h = torch.empty(tokens, top_k, w13.size(1), dtype=self.dtype, device="cuda")
            self.gemm(routing.x, w13, h, routing, routed, cfg, top_k, False)
            out = torch.empty(tokens, top_k, hidden, dtype=self.dtype, device="cuda")
            self.gemm(silu_and_mul(h.view(tokens * top_k, -1)), w2, out, routing, routed, cfg, 1, True)
            return out.sum(1)

        return run


# --------------------------------------------------------------------------------------------
# AITER
# --------------------------------------------------------------------------------------------
class AiterBf16MoE(MoEImpl):
    name = "aiter_bf16"
    supports_rotation = True

    def build(self, w13, w2):
        if self.dtype != torch.bfloat16:
            raise Unsupported("AITER fused_moe CK path benchmarked for bf16 only")
        from aiter.ops.shuffle import shuffle_weight

        self.copies = [(shuffle_weight(w13.contiguous(), layout=(16, 16)), shuffle_weight(w2.contiguous(), layout=(16, 16)))]
        self.raw = (w13, w2)

    def dequant(self, experts):
        return tuple(w[experts].float() for w in self.raw)

    def describe(self) -> str:
        return "unquantized (CK 2-stage, preshuffled)"

    def prepare(self, routing, config=None):
        from aiter.fused_moe import fused_moe

        self.last_config = "aiter heuristics"

        def run(index):
            w13, w2 = self.copies[index]
            return fused_moe(routing.x, w13, w2, routing.topk_weights, routing.topk_ids)

        return run


class AiterA16Wi4MoE(MoEImpl):
    name = "aiter_a16wi4"

    def build(self, w13, w2):
        if self.dtype != torch.bfloat16:
            raise Unsupported("AITER a16wi4 is bf16-only")
        if self.bits != 4:
            raise Unsupported("AITER a16wi4 is int4-only")
        import aiter
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_i4_quant
        from aiter.ops.shuffle import pack_int8_to_packed_int4, shuffle_scale_for_int4, shuffle_weight

        self.quant_type = aiter.QuantType.per_1x32
        self.master, weights = [], []
        for w in (w13, w2):
            num_experts, shape_n, shape_k = w.shape
            q, s = [], []
            for e0 in range(0, num_experts, 16):  # bound temporaries
                qq, ss = per_1x32_i4_quant(w[e0 : e0 + 16].contiguous())
                q.append(qq.view(torch.int8))
                s.append(ss)
            q, s = torch.cat(q), torch.cat(s)  # [E, N, K] int8 in [-8, 7], [E, K/32, N] bf16
            self.master.append((q, s))
            packed = pack_int8_to_packed_int4(shuffle_weight(q, (16, 16)))
            packed = packed.view(num_experts, shape_n, shape_k // 2).view(dtypes.i4x2)
            weights += [packed, shuffle_scale_for_int4(s, group_size=32).view(-1).contiguous()]
        self.copies = [tuple(weights)]

    def dequant(self, experts):
        out = []
        for q, s in self.master:
            q, s = q[experts].float(), s[experts].float()
            n, k = q.shape[1:]
            out.append((q.view(len(experts), n, k // 32, 32) * s.transpose(1, 2).unsqueeze(-1)).view(-1, n, k))
        return tuple(out)

    def describe(self) -> str:
        return "int4 sym g32 (per_1x32), FlyDSL 2-stage"

    def prepare(self, routing, config=None):
        from aiter.fused_moe import fused_moe

        self.last_config = "aiter heuristics"

        def run(index):
            w13, s13, w2, s2 = self.copies[index]
            return fused_moe(routing.x, w13, w2, routing.topk_weights, routing.topk_ids, w1_scale=s13,
                             w2_scale=s2, quant_type=self.quant_type)

        return run


def _cfg_str(cfg: dict) -> str:
    keys = [("BLOCK_SIZE_M", "m"), ("BLOCK_SIZE_N", "n"), ("BLOCK_SIZE_K", "k"), ("num_warps", "w")]
    return "/".join(f"{short}{cfg[key]}" for key, short in keys if key in cfg)


IMPLS = {
    cls.name: cls
    for cls in [HummingMoE, TritonWNA16MoE, TritonWNA16TunedMoE, TritonUnquantizedMoE, AiterBf16MoE, AiterA16Wi4MoE]
}
QUANTIZED = {"humming", "triton_wna16", "triton_wna16_tuned", "aiter_a16wi4"}
