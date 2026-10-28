# MFMA weight-only GEMM optimization log (gfx950 / MI355X)

Kernel: `humming/include/humming/kernel/humming_mfma.cuh`. Benchmark:
`HIP_VISIBLE_DEVICES=4 python -m tests.rocm.bench_mfma [--bits 4 --group 128 ...]`.
Unless noted: A fp16, uint4 symmetric, group 128, weights rotated over >= 1 GB of copies
(defeats the 256 MB MALL), kernel launches captured in a HIP graph (host overhead
excluded), torch baseline = `torch.matmul` fp16 (hipBLASLt) on the dequantized weight,
also rotated. TB/s counts quantized weight + scales + A + C bytes.

## Iteration 0: functional v1

Direct global -> register loads of A and B (no LDS), kStages register prefetch,
16x16x32 MFMA, group scale on the fp32 accumulator, split-K with workspace + counter.
Correct on the full shape matrix (max rel err ~3-4e-4 fp16), but slow:

| shape (M, N, K) | us | TB/s | torch us |
|---|---|---|---|
| 1, 4096, 4096 | 49.3 | 0.18 | 15.2 |
| 1, 28672, 8192 | 238.6 | 0.51 | 91.6 |
| 1024, 28672, 8192 | 1622.8 | 0.12 | 375.6 |

First timing used plain HIP events around Python launches; switching to graph capture
changed nothing, so the kernel itself was slow.

## Iteration 1: branch-free loads (no effect)

ISA showed `s_waitcnt vmcnt(0)` right after issuing the prefetch: loads were guarded by
`row < M`, `tile_n valid`, `iter < num_iters`, and the waitcnt pass is conservative at
control-flow joins, so the register pipeline was fully serialized. Clamped rows / N tiles
/ chunk indices (out-of-range data is loaded but never stored) and split the main loop
into an unconditional steady state + guarded tail. ISA now keeps 18-23 loads in flight.
**No measurable change** -> the bottleneck was elsewhere (kept: it is required anyway).

## Iteration 2: split-K without L2 writeback fences

Timing `num_splits` 1/2/4/8 at M=1, 4096x4096: 24 / 31 / 39 / 57 us. `__threadfence()`
(agent scope) lowers to `buffer_wbl2 sc1` + `buffer_inv sc1` on gfx950 (per-XCD L2 is
written back / invalidated), three times per block. Replaced by agent-scope relaxed
atomic stores/loads for the fp32 partials (`global_store/load ... sc1`, coherent across
XCDs on their own), an explicit `s_waitcnt vmcnt(0)` and a relaxed atomic counter.

| shape (M, N, K) | before us | after us | TB/s | torch us | speedup vs torch |
|---|---|---|---|---|---|
| 1, 4096, 4096 | 56.3 | 9.5 | 0.91 | 12.3 | 1.30x |
| 1, 11008, 4096 | 112.6 | 11.7 | 1.99 | 19.1 | 1.63x |
| 1, 8192, 28672 | 125.6 | 35.0 | 3.47 | 76.8 | 2.20x |
| 1, 28672, 8192 | 242.4 | 36.5 | 3.32 | 93.1 | 2.55x |
| 16, 8192, 28672 | 146.3 | 62.7 | 1.95 | 76.7 | 1.22x |
| 64, 8192, 28672 | 290.2 | 219.4 | 0.57 | 83.9 | 0.38x |
| 1024, 28672, 8192 | 1660.1 | 1664.7 | 0.12 | 370.6 | 0.22x |

Large M is now the problem: each wave owns only 16 output columns and re-reads A from
L2 for every 16 columns (A traffic = M*K*2 * N/16).

## Iteration 3: tile sweep + LDS-staged A

Larger per-wave tiles (`tiles_m x tiles_n` MFMAs per chunk) cut A re-reads, but with
direct global loads every wave of the block loads the same A rows. `kLdsA` loads the
block's A chunk once (spread over all threads), stores it to a double-buffered LDS
ring (one `__syncthreads` per chunk) and each wave reads its fragments from LDS.
16-byte unit `u` of row `r` lives at slot `u ^ (r % 16)`, which makes both the store
and the fragment reads bank-conflict free.
Dead end: HIP's `uint4` is a class; arrays of it in the LDS path got scratch allocas
(256 B/lane). Switching to a native `ext_vector_type(4)` (`mfma_u32x4_t`) fixed it.

## Iteration 4: B-side group scaling (`kScaleB`)

With group scales, the fp32 accumulator is rescaled per group (C-side): tiles_m*tiles_n*4
v_fma per group, and partial sums are flushed per group. `kScaleB` multiplies the
dequantized B fragment by the group scale (`v_pk_mul_f16`, one per 2 values) before the
MFMA, so the accumulator never has to be touched. That is cheaper when tiles_m >= 4.
bf16 has no packed multiply on gfx950 (it goes through fp32), so the win there is small.
At the 2,4,4,2 tile: group C-side 591 us vs per-channel 490 us (M=1024, 28672x8192).
B-side scaling recovers most of that gap.

## Iteration 5: gfx950 `v_bitop3_b32` for `lop3`

The kernel is VALU-bound: 16x16x32 MFMA ~16 cycles vs ~4 cycles per wave64 VALU op, so
more than about 4 VALU ops per MFMA stalls the matrix core. gfx9 VOP3 cannot encode
literals, so the generic `(a & mask) | magic` costs 2 ops plus a constant in a register.
`__builtin_amdgcn_bitop3_b32(a, b, c, lut)` uses the same LUT convention as PTX `lop3`
(checked on hardware) and does it in one op. That covers all of Humming's dequant.

28672x8192, 4-bit g128 fp16 (torch fp16: ~90 us M<=64, 134 us M=255, 370 us M=1024):

| tile | M=1 | M=16 | M=64 | M=255 | M=1024 |
|---|---|---|---|---|---|
| 1,4,1,1,4 (no LDS) | 38.5 | 71.3 | 234.8 | - | - |
| 1,4,4,2,2,lds | 50.9 | 53.5 | 65.4 | 170.1 | 593 |
| 1,4,8,2,2,lds,sb (before bitop3) | - | - | - | - | 498 |
| 1,4,8,2,2,lds,sb | 72.1 | 74.6 | 83.2 | 140.0 | 482.1 (998 TFLOPs) |

Dead ends: 2,2,8,2 (128 KB LDS, occupancy 1) and 1,4,4,4,2,lds,sb (174 VGPR + 88 AGPR,
occupancy 1) both ran about 2x slower at M=1024.
Remaining VALU per 128 MFMAs in the 1,4,8,2 loop: 64 bitop3, 64 pk_add, 64 pk_mul,
48 lshr, about 34 64-bit address ops (`v_lshl_add_u64`).
