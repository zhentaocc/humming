"""Problem definition, packing helpers and the baseline registry.

A baseline is registered as ``{name: factory}`` where

    factory(M, N, K, dtype, bits, group_size, zp) -> (setup, run)

* ``setup(problem) -> state``: converts the canonical quantized problem into the
  implementation's own layout. It must allocate *new* device tensors (no
  aliasing of ``problem`` tensors) because the harness calls it several times
  to build rotating buffer copies. ``state`` is any object; if it is a dict it
  may carry optional keys:
    - ``"w_ref"`` / ``"x_ref"``: fp32 tensors overriding the reference weight /
      activation used by the correctness check (for impls with their own quant
      format, e.g. MXFP4/FP8, or activation quantization).
    - ``"weight_bytes"``: bytes of weight storage read per call (overrides the
      nominal estimate used for effective TB/s).
* ``run(state) -> Y[M, N]``: the timed call. Must be CUDA-graph capturable
  after a warm-up call (no host sync), otherwise the harness falls back to
  eager timing.

Raise ``Unsupported("reason")`` from the factory to skip a configuration.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch


class Unsupported(Exception):
    pass


@dataclasses.dataclass
class BaselineSpec:
    name: str
    factory: Callable
    kind: str  # "dense" | "wNa16" | "w8a8" | "w4a4"
    bits: tuple[int, ...]  # weight bits it can run (dense: (16,))
    source: str  # where the kernel comes from (repo/file@commit)
    tol: float = 2e-2  # relative Frobenius error tolerance vs fp32 reference


REGISTRY: dict[str, BaselineSpec] = {}


def register(name: str, *, kind: str, bits, source: str, tol: float = 2e-2):
    def deco(factory):
        REGISTRY[name] = BaselineSpec(name, factory, kind, tuple(bits), source, tol)
        return factory

    return deco


# --------------------------------------------------------------------------
# Canonical quantized problem
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Problem:
    """Canonical integer weight quantization (GPTQ/AWQ semantics).

    w[n, k] = (q[n, k] - z[n, k // G]) * s[n, k // G]
    q in [0, 2^bits), z in [1, 2^bits) when zp else z = 2^(bits-1) (symmetric,
    uint4b8 / uint8b128). Scales are stored rounded to ``dtype``.
    """

    M: int
    N: int
    K: int
    dtype: torch.dtype
    bits: int
    group_size: int
    zp: bool
    x: torch.Tensor  # [M, K] dtype
    q: torch.Tensor  # [N, K] int32
    scales: torch.Tensor  # [N, K // G] dtype
    zeros: torch.Tensor  # [N, K // G] int32 (bias value when not zp)

    @property
    def num_groups(self) -> int:
        return self.K // self.group_size

    def w_dequant(self, dtype=torch.float32) -> torch.Tensor:
        """Dequantized weight [N, K] (computed in fp32 then cast)."""
        G = self.group_size
        s = self.scales.float().repeat_interleave(G, dim=1)
        z = self.zeros.float().repeat_interleave(G, dim=1)
        return ((self.q.float() - z) * s).to(dtype)

    def fresh_x(self) -> torch.Tensor:
        return self.x.clone()


def make_problem(M, N, K, dtype, bits, group_size, zp, seed=0, device="cuda") -> Problem:
    if group_size == -1:
        group_size = K
    if K % group_size:
        raise Unsupported(f"K={K} not divisible by group_size={group_size}")
    gen = torch.Generator(device=device).manual_seed(seed)
    qmax = 2**bits
    ng = K // group_size
    q = torch.randint(0, qmax, (N, K), generator=gen, device=device, dtype=torch.int32)
    if zp:
        zeros = torch.randint(1, qmax, (N, ng), generator=gen, device=device, dtype=torch.int32)
    else:
        zeros = torch.full((N, ng), qmax // 2, device=device, dtype=torch.int32)
    # Scale so that |w| ~ 1/sqrt(K) * O(1): keeps outputs O(1) in fp16.
    base = 2.0 / (qmax * K**0.5)
    scales = (torch.rand((N, ng), generator=gen, device=device) * 0.5 + 0.75) * base
    scales = scales.to(dtype)
    x = torch.randn((M, K), generator=gen, device=device).to(dtype)
    return Problem(M, N, K, dtype, bits, group_size, zp, x, q, scales, zeros)


def nominal_weight_bytes(N, K, bits, group_size, zp, scale_bytes=2) -> int:
    if bits >= 16:
        return N * K * bits // 8
    ng = K // group_size
    b = N * K * bits // 8 + N * ng * scale_bytes
    if zp:
        b += N * ng * bits // 8
    return b


# --------------------------------------------------------------------------
# Packing helpers (all on GPU, int32 two's-complement bit patterns)
# --------------------------------------------------------------------------
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]  # vLLM awq_triton: column j <- nibble rev[j]


def pack_last_dim(v: torch.Tensor, bits: int, order: Optional[list[int]] = None) -> torch.Tensor:
    """Pack int values along the last dim, 32//bits per int32.

    Element j of each pack goes to bit offset ``order[j] * bits`` (sequential
    when ``order`` is None).
    """
    pf = 32 // bits
    *lead, L = v.shape
    assert L % pf == 0, (v.shape, pf)
    v = v.to(torch.int32).reshape(*lead, L // pf, pf) & ((1 << bits) - 1)
    out = torch.zeros((*lead, L // pf), dtype=torch.int32, device=v.device)
    for j in range(pf):
        pos = order[j] if order is not None else j
        out |= v[..., j] << (pos * bits)
    return out


def gptq_pack(problem: Problem, zero_offset: int = 0):
    """GPTQ/exllama layout: qweight [K//pf, N] (K packed), qzeros [G, N//pf]
    (N packed, stored as ``z + zero_offset``), scales [G, N]."""
    qweight = pack_last_dim(problem.q, problem.bits).t().contiguous()
    qzeros = pack_last_dim((problem.zeros + zero_offset).t().contiguous(), problem.bits)
    scales = problem.scales.t().contiguous()
    return qweight, qzeros, scales


def n_packed_pack(problem: Problem, order=None):
    """[K, N//pf] layout with N packed (vLLM triton_w4a16 / AWQ)."""
    qweight = pack_last_dim(problem.q.t().contiguous(), problem.bits, order)
    qzeros = pack_last_dim(problem.zeros.t().contiguous(), problem.bits, order)
    scales = problem.scales.t().contiguous()
    return qweight, qzeros, scales
