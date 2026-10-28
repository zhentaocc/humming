# Dense weight-only MFMA kernel: large-M performance log (gfx950 / MI355X)

Kernel: `humming/include/humming/kernel/humming_mfma.cuh`, tile heuristic
`humming/tune/gfx950.py`. Goal: close the large-M gap to hipBLASLt fp16 without
regressing M <= 256.

Method (unless noted): `python -m tests.rocm.bench_mfma --ms ... --shapes NxK --dtype ...`
on one GPU of a shared node, best of 2 runs, HIP-graph timing, weights rotated over
>= 1 GB. A fp16/bf16, uint4 symmetric, group 128. "hipBLASLt" = `torch.matmul` on the
dequantized weight (same dtype). Tile notation `wm,wn,tm,tn,stages` = waves (m, n),
16x16 MFMA tiles per wave (m, n), pipeline stages; block tile = (wm*tm*16) x (wn*tn*16).
Starting point: `4288c98` (default tile for large M: 1,4,8,2,2 lds = 128x128, 2 blocks/CU).

## Iteration 0: where the time goes (profile)

8192x28672 (N x K), M=4096, fp16: 1849 us, hipBLASLt 1283 us (0.69x).
MFMA peak for this GEMM (2*M*N*K = 1.92 TFLOP at 2516 TF) is 765 us, so hipBLASLt runs at
60% of MFMA peak and Humming at 41%.

Main-loop profile (rocprofv3 counters + ISA instruction mix, `/tmp/perfwork/loops.py`),
fp16, measured right after iteration 1:

| kernel / tile | LDS reads / MFMA | VALU / MFMA |
|---|---|---|
| Humming 8x2 (128x128) | 0.5 | 1.96 |
| Humming 8x4 (128x256) | 0.25 | 1.95 |
| hipBLASLt MT256x256 | 0.25 | 0.04 |

The dequant is 15 VALU per 8-value int4 fragment (3 lshr, 4 bitop3, 4 pk_add, 4 pk_mul).

Co-issue microbenchmark (2 waves/SIMD, a dependent-free MFMA stream with N independent
VALU interleaved): VALU per MFMA 2 -> +14% time, 3 -> +50%, 4 -> +53%. MFMA 16x16x32 is
16 cycles and the VALU in the same SIMD only hides behind it up to about 2 ops per
MFMA. So the dequant VALU is on the critical path at ~1.95/MFMA and the first lever
is VALU per MFMA, the second is reuse of each dequantized B fragment over more M rows
(bigger tm or wm).

hipBLASLt for the same GEMM: MT256x256 macro tile, 4 waves with 128x128 wave tiles,
135 KB LDS, 256 persistent WGs, 1 wave/SIMD, 59.5% MFMA utilization.

## Iteration 1: direct-to-LDS A loads (`68df7fd`, kept)

Hypothesis: A staging via registers (`global_load` -> `ds_write`) costs VGPRs and VALU
issue slots; `buffer_load_dwordx4 ... lds` writes LDS directly and frees both, which in
turn makes the 8x4 (128x256) tile fit without spills.

| fp16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 (128x128) | 1849 | 1698 |
| 8x4 (128x256) | spills | 1476 |

Removing the `vmcnt` wait on the A chunk (relying only on the barrier) changed nothing.

Side effect found later: with 2 stages, the DMA path must drain `vmcnt(0)` before issuing
the next chunk (the LDS slot being overwritten is the one just read), which exposes full
load latency at small M. Fixed in iteration 5.

## Iteration 2: shift-light integer dequant (`57ac1de`, kept)

Hypothesis: the dequant does `(w >> shift) & mask | magic` per field, i.e. one shift per
nibble. A field at bit offset r inside a 16-bit half can be decoded in place:
`(w & (mask << r)) | magic(2^(10-r))` is the fp16 number `2^(10-r) + code*2^0` scaled
so that subtracting `2^(10-r) + zp` (one `v_pk_add_f16` / `v_pk_fma_f16`) gives the exact
code. One shift per 32-bit word instead of per field. Bit-identical output (all values
are small integers, exact in fp16).

| fp16 int4 | before | after |
|---|---|---|
| VALU per fragment (8 values) | 15 | 13 |
| main-loop VALU / MFMA (lshr count 96 -> 32) | 1.95 | 1.66 |

| fp16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 | 1698 | 1626 |
| 8x4 | 1476 | 1472 |

Main loop fp16 is now 1.66 VALU/MFMA. 8x4 barely moves (it already amortizes the
dequant over 8 M-tiles), 8x2 gains 4%.

## Iteration 3: bf16 dequant via `v_fma_mix_f32` (`cb9476d`, kept)

bf16 has no 2^(10-r) trick on the packed bf16 side (7-bit mantissa), so the old path was
int -> f32 -> scale -> bf16, 25 VALU/fragment. New path: decode the field as fp16 with
the shift-light trick, then `v_fma_mix_f32(x_f16, scale_f32, -(2^(10-r)+zp)*scale)`
(fp16 operand converted in the FMA), then `v_cvt_pk_bf16_f32`. The FMA is exact (all
products are <= 19 significant bits), so the result is rounded once -> bit-identical to
the reference `((code - zp) * scale).to(bf16)`. LLVM never formed `fma_mix` from the C++,
so it is inline asm with a host fallback. Group scale is folded into the dequant
(`run_scaled`) for bf16.

| VALU per int4 fragment | before | after |
|---|---|---|
| bf16 | 25 | 17 |

| bf16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 | 1909 | 1677 |
| 8x4 | 2198 (spills) | 1779 (spills) |

Main-loop VALU/MFMA for bf16 8x2: 2.2.

Rejected variant: subtract in fp16 then mix (`HUMMING_BF16_SUBF16`) -> 64 VGPR spills in
the 8x4 tile, 3113 us.

bf16 still cannot use 8x4: the per-residual f32 offsets push it to 56 spilled VGPRs
(1779 us vs 1677 us for 8x2). `amdgpu-schedule-*` / `-mllvm` scheduler flags did not remove
the spills. Other bf16 tiles at M=4096, 8192x28672 (us): 8x3 1743, 2x2 waves 8x4 2004,
2x4 waves 8x2 1810, 16x2 1836. So bf16 stays on 8x2.

## Iteration 4: 128x256 tile for large-M fp16 (`1112a1d`, kept)

Hypothesis: 8x4 halves the A LDS reads per MFMA and dequant per MFMA stays the same, but
it has half as many blocks, so it only wins when the grid fills the GPU. Swept tile x split
over N in {4096, 8192, 28672, 57344} and M in 64..4096 (`/tmp/perfwork/sweep_1422.txt`).
8x4 wins once `ceil(N/256) * ceil(M/128) >= 1.5 * CUs`. Boundary samples (8x2 vs 8x4, us):

| N | M | 8x2 | 8x4 |
|---|---|---|---|
| 57344 | 128 | 128.8 | 126.3 |
| 57344 | 192 | 231 | 207 |
| 28672 | 448 | 119 | 106 |
| 8192 | 1536 | 637 | 647 |
| 4096 | 3072 | 98.2 | 101.4 |

Change: `wide_tile_min_m(N, cus) = max(64, 128 * (ceil(1.5*cus / ceil(N/256)) - 1))`;
`select_tile` uses 8x4 for fp16 above it, and `get_m_intervals` splits the last M bucket
at that N-dependent bound so the compiled config table matches. bf16 is excluded (spills).

| fp16 8192x28672 M=4096 | before (8x2) | after (8x4) |
|---|---|---|
| us | 1626 | 1478 |

The end-to-end effect on the other shapes is in the final table (M=256..4096 for
57344x8192 and M=4096 for 4096x4096 switch to 8x4).

Tests: `test_wide_tile` (bits 1-5, 8; g0/g128; zp; M 77/300; fp16+bf16) and
`test_tile_buckets` (interval consistency, tuner == test mirror).

## Iteration 5: 3-stage pipeline for the M <= 64 tiles (`3682ec7`, kept)

Bisect showed iteration 1 regressed small M (fp16 8192x28672 M=16: 39.2 -> 44.9 us,
M=64: 51.2 -> 54). Cause: the DMA path with 2 stages waits `vmcnt(0)` before the next
chunk is issued. With 3 stages there is always one chunk in flight. The M <= 32 (2x2) and
M <= 64 (4x2) LDS tiles now use 3 stages:

| fp16 8192x28672 | 4288c98 | 68df7fd | 3682ec7 |
|---|---|---|---|
| M=16 | 38.6 | 44.9 | 35.6 |
| M=64 | 51.2 | 54.0 | 47.3 |

3 stages for 8x2 at M=128/256 was mixed or worse (kept at 2).

## Rejected experiments

| experiment | fp16 8192x28672 M=4096 | verdict |
|---|---|---|
| STREAM_A reorder (A fragment reads interleaved with B dequant) | 1509-1520 vs 1476 | worse |
| `sched_group_barrier` MFMA/VALU interleave, 8x2 | 1637-1855 vs 1626 | worse, main-loop spills |
| same, 8x4 | 2668-3743 | much worse, spills |
| same, 16x4 | 1927-2330 | worse |
| 16x4 tile (occupancy 1) | 1698 | worse: accumulator in AGPRs is rotated, ~416 `v_accvgpr` moves per 512 MFMA |
| 16x2 / 12x4 / 16x3 | 1815 / 1710 / 1762 | worse |
| rowsum zero-point correction (emulated: drop the per-fragment subtract, VALU 1.66 -> 1.22 / MFMA) | 1434 vs 1477 | only 3%, needs an extra rowsum(A) pass and an epilogue; not implemented |

### 32x32x16 MFMA layout: rejected

Considered switching to `v_mfma_f32_32x32x16_f16`. It does not reduce LDS reads or VALU
per FLOP for this kernel: the dequant cost is per B element, and a 32x32 tile consumes
the same B elements per FLOP as a 16x16 wave tile with the same M extent (a 128-row wave
tile already reuses each dequantized B fragment 8 times). A-fragment LDS traffic per FLOP
is set by the wave tile shape, not by the MFMA shape. It would also need a new B
pre-shuffle layout (32-lane K-groups), new dequant indexing and new tests for every bit
width, so it is not worth doing.

## Bound: why M=1024 >= 1.1x and M=4096 >= 0.9x are not reached

Both M=1024 and M=4096 are compute-bound for both kernels (weight bytes are < 10% of the
time at the measured TB/s).

* Fake-dequant run (B registers fed without any dequant, same tile 8x4): 1243 us at
  8192x28672 M=4096, vs hipBLASLt 1283 us (1.03x). With no dequant at all the kernel is
  only 3% faster than hipBLASLt; 12x4 / 16x4 without dequant give 1516 / 1419 us, so a
  bigger tile does not help either (registers / AGPR rotation). The real kernel is at
  1478 us; the remaining 19% over fake-dequant is the dequant VALU (1.66/MFMA, above the
  ~1-2 VALU free co-issue window from the microbenchmark). The rowsum emulation shows the
  dequant could only shrink by about 3% more without changing the format.
* M=4096 target 0.9x = 1426 us at 8192x28672: reached for 57344x8192 at 0.89x and
  8192x28672 at 0.87x (2-3% short); 4096x4096 is 0.82x because it has only 256 128x256
  blocks (1 per CU), below the 1.5 blocks/CU where 8x4 wins, so it runs 8x2.
* M=1024 at 1.1x would need 331 us at 8192x28672, i.e. 57.8% of MFMA peak, while
  hipBLASLt itself runs at 52.5% (364 us) with no dequant. At M=1024, N=8192 there are
  only 32x8 = 256 128x256 blocks (1/CU), so the 8x2 tile (512 blocks) is used and is
  at 45%. Getting there would need a kernel faster than hipBLASLt on a plain fp16 GEMM.
  4096x4096 M=1024 reaches 1.06x; 57344x8192 0.97x.
* bf16 vs fp16 (target <= 5%): 14% slower at M=4096 (1681 vs 1478 us) because bf16 is on
  8x2 (17 vs 13 VALU/fragment pushes 8x4 into spills). At M <= 256 bf16 is within 3-9%.

## Results: before (4288c98) -> after (3682ec7)

us, best of 2; ratio = hipBLASLt us / Humming us (>1 = Humming faster).

| dtype | N x K | M | before | after | speedup | hipBLASLt | before ratio | after ratio |
|---|---|---|---|---|---|---|---|---|
| fp16 | 4096x4096 | 1 | 7.2 | 7.2 | 1.00x | 12.1 | 1.68x | 1.68x |
| fp16 | 4096x4096 | 16 | 10.2 | 10.3 | 0.99x | 10.1 | 0.99x | 0.98x |
| fp16 | 4096x4096 | 64 | 12.9 | 12.9 | 1.00x | 13.0 | 1.01x | 1.01x |
| fp16 | 4096x4096 | 256 | 22.1 | 20.7 | 1.07x | 22.4 | 1.01x | 1.08x |
| fp16 | 4096x4096 | 1024 | 45.6 | 39.7 | 1.15x | 42.2 | 0.93x | 1.06x |
| fp16 | 4096x4096 | 4096 | 143.8 | 117.2 | 1.23x | 96.1 | 0.67x | 0.82x |
| fp16 | 8192x28672 | 1 | 34.5 | 30.8 | 1.12x | 75.6 | 2.19x | 2.45x |
| fp16 | 8192x28672 | 16 | 38.6 | 35.6 | 1.08x | 76.8 | 1.99x | 2.16x |
| fp16 | 8192x28672 | 64 | 51.2 | 47.3 | 1.08x | 80.8 | 1.58x | 1.71x |
| fp16 | 8192x28672 | 256 | 150.4 | 127.7 | 1.18x | 123.3 | 0.82x | 0.97x |
| fp16 | 8192x28672 | 1024 | 482.6 | 427.1 | 1.13x | 364.2 | 0.75x | 0.85x |
| fp16 | 8192x28672 | 4096 | 1841.7 | 1477.6 | 1.25x | 1282.6 | 0.70x | 0.87x |
| fp16 | 57344x8192 | 1 | 49.9 | 49.3 | 1.01x | 155.1 | 3.11x | 3.15x |
| fp16 | 57344x8192 | 16 | 60.4 | 56.4 | 1.07x | 161.3 | 2.67x | 2.86x |
| fp16 | 57344x8192 | 64 | 88.3 | 82.1 | 1.08x | 157.6 | 1.78x | 1.92x |
| fp16 | 57344x8192 | 256 | 268.4 | 220.6 | 1.22x | 212.7 | 0.79x | 0.96x |
| fp16 | 57344x8192 | 1024 | 960.0 | 771.1 | 1.24x | 747.5 | 0.78x | 0.97x |
| fp16 | 57344x8192 | 4096 | 3765.3 | 2997.5 | 1.26x | 2671.8 | 0.71x | 0.89x |
| bf16 | 4096x4096 | 1 | 7.7 | 7.7 | 1.00x | 12.3 | 1.60x | 1.60x |
| bf16 | 4096x4096 | 16 | 11.0 | 10.8 | 1.02x | 10.1 | 0.92x | 0.94x |
| bf16 | 4096x4096 | 64 | 13.8 | 13.4 | 1.03x | 17.4 | 1.26x | 1.30x |
| bf16 | 4096x4096 | 256 | 23.9 | 21.5 | 1.11x | 19.7 | 0.82x | 0.92x |
| bf16 | 4096x4096 | 1024 | 50.6 | 40.8 | 1.24x | 45.1 | 0.89x | 1.11x |
| bf16 | 4096x4096 | 4096 | 155.2 | 131.6 | 1.18x | 94.9 | 0.61x | 0.72x |
| bf16 | 8192x28672 | 1 | 36.7 | 32.9 | 1.12x | 75.8 | 2.07x | 2.30x |
| bf16 | 8192x28672 | 16 | 45.2 | 39.0 | 1.16x | 76.9 | 1.70x | 1.97x |
| bf16 | 8192x28672 | 64 | 58.4 | 49.4 | 1.18x | 81.1 | 1.39x | 1.64x |
| bf16 | 8192x28672 | 256 | 165.0 | 131.5 | 1.25x | 120.6 | 0.73x | 0.92x |
| bf16 | 8192x28672 | 1024 | 532.8 | 437.6 | 1.22x | 353.0 | 0.66x | 0.81x |
| bf16 | 8192x28672 | 4096 | 2070.9 | 1681.1 | 1.23x | 1230.5 | 0.59x | 0.73x |
| bf16 | 57344x8192 | 1 | 57.9 | 53.3 | 1.09x | 155.7 | 2.69x | 2.92x |
| bf16 | 57344x8192 | 16 | 75.8 | 61.6 | 1.23x | 158.7 | 2.09x | 2.58x |
| bf16 | 57344x8192 | 64 | 102.1 | 87.5 | 1.17x | 157.2 | 1.54x | 1.80x |
| bf16 | 57344x8192 | 256 | 297.3 | 250.5 | 1.19x | 204.9 | 0.69x | 0.82x |
| bf16 | 57344x8192 | 1024 | 1080.5 | 878.4 | 1.23x | 724.8 | 0.67x | 0.83x |
| bf16 | 57344x8192 | 4096 | 4303.8 | 3450.2 | 1.25x | 2569.7 | 0.60x | 0.74x |

No M <= 256 point regresses by more than 1% (4096x4096 M=16 fp16 10.2 -> 10.3 is noise).
Full baseline grid (W4 sym/zp g128, W8; fp16/bf16; all baseline impls):
`benchmarks/rocm/results/humming_perf_2026-10-28.{csv,md}`.

Tests: `pytest tests/rocm` 709 passed / 40 skipped at start -> 919 passed / 74 skipped.
