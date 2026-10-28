# ROCm baseline GEMM benchmark (2026-10-28T17:08:13)

Y[M,N] = X[M,K] @ W[N,K]^T; weight-only quant unless noted; group size 128; zp = asymmetric zero point, sym = uint{4,8}b{8,128}.

## Environment

- device: `AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, HIP_VISIBLE_DEVICES=4`
- torch: `2.11.0+rocm7.2`
- triton: `3.6.0`
- hip: `7.2.26015`
- date: `2026-10-28T17:08:13`
- args: `--impls humming_w8a8_fp8 humming_w8a8_fp8_block128 humming_w4a8_int humming_w4a8_mxfp4 humming_w4a4_mxfp4 humming_w8a8_int8 humming_w4a8_int_i8 humming --dtypes bf16 --bits 4 8 --zp sym --shapes llama3-8b/o_proj llama3-70b/gate_up llama3-70b/down --m 1 16 1024 4096 --windows 3 --prefix humming_f8_merged --tag 2026-10-28`

## MI355X peaks (vendor) and measured probes

| item | value |
|---|---|
| peak: HBM3E bandwidth (TB/s) | 8.0 |
| peak: FP16/BF16 dense MFMA (TFLOPs) | 2516.6 |
| peak: FP8 dense (TFLOPs) | 5033.2 |
| peak: MXFP4 dense (TFLOPs) | 10066.3 |
| peak: Infinity Cache / MALL (MB) | 256 |
| peak: L2 per XCD (MB) | 4 |
| peak: CUs | 256 |
| measured: copy 4 GiB (read+write) TB/s | 4.92 |
| measured: sum 4 GiB (read-only) TB/s | 3.48 |
| measured: fp16 8192^3 F.linear TFLOPs | 1463.64 |
| measured: bf16 8192^3 F.linear TFLOPs | 1624.38 |

## Methodology

- HIP-event timing; warmup >= 20; >= 100 timed calls; calls captured into a HIP graph (no host launch overhead) unless the `timing` column says eager. Best of 3 timing window(s) per point.
- L2/MALL defeat: rotating buffers. Each config builds n_copies independent (X, packed W, scales, zeros, workspace) sets so the cyclic footprint is >= 1024 MiB (4x the 256 MB MALL, capped at 256 copies); calls cycle through the copies, so every call reads cold weights from HBM.
- TFLOPs = 2MNK / t. Effective TB/s = (quantized weight bytes incl. scales/zeros + activation bytes + fp16 output bytes) / t. For dequant+GEMM fallbacks the intermediate fp16 weight traffic is NOT counted (so TB/s is comparable across implementations as 'useful bytes').
- w8a8 / w4a4 rows time the GEMM only (activation quantization done in setup, not timed).
- Correctness: each config is checked against an fp32 torch reference on N,K=(512, 1024), M in [1, 33, 300] before timing (relative Frobenius error).

## Implementation status

| config | status | source |
|---|---|---|
| humming_w8a8_fp8 [bf16 w8 sym] | OK: ok (max rel err 1.70e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): FP8 per-token X x FP8 per-channel W, X quant not timed |
| humming_w8a8_fp8_block128 [bf16 w8 sym] | OK: ok (max rel err 1.66e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): FP8 X 1x128 x FP8 W 128x128 block scales, X quant not timed |
| humming_w4a8_int [bf16 w4 sym] | OK: ok (max rel err 1.66e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): FP8 per-token X x INT4 (group G, optional zp) W dequantized to FP8, X quant not timed |
| humming_w4a8_mxfp4 [bf16 w4 sym] | OK: ok (max rel err 1.71e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): FP8 per-token X x MXFP4 W (scaled MFMA f8f6f4), X quant not timed |
| humming_w4a4_mxfp4 [bf16 w4 sym] | OK: ok (max rel err 1.71e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): MXFP4 X x MXFP4 W (scaled MFMA f8f6f4), X quant not timed |
| humming_w8a8_int8 [bf16 w8 sym] | OK: ok (max rel err 1.66e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): INT8 per-token X x INT8 per-channel W, X quant not timed |
| humming_w4a8_int_i8 [bf16 w4 sym] | OK: ok (max rel err 1.67e-03) | humming@bf4b4e8fcd HummingLayer, gfx950 low-bit MFMA GEMM (humming/kernel/humming_mfma_f8.cuh): INT8 per-token X x INT4 (group G, optional zp) W, X quant not timed |
| humming [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | humming@bf4b4e8fcd HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w8 sym] | OK: ok (max rel err 2.35e-03) | humming@bf4b4e8fcd HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |

## Headline (bf16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| humming_w8a8_fp8 w8 sym | 8.3 (2.03 TB/s) | 9.3 (1.82 TB/s) | 32.7 (1050 TF) | 103.5 (1327 TF) | 78.8 (5.97 TB/s) | 86.8 (5.44 TB/s) | 607.4 (1584 TF) | 2364.8 (1627 TF) | 42.0 (5.59 TB/s) | 44.9 (5.25 TB/s) | 270.2 (1780 TF) | 1119.9 (1718 TF) |
| humming_w8a8_fp8_block128 w8 sym | 9.3 (1.87 TB/s) | 9.8 (1.79 TB/s) | 39.9 (861 TF) | 145.9 (942 TF) | 81.6 (5.94 TB/s) | 91.9 (5.29 TB/s) | 905.2 (1063 TF) | 3504.3 (1098 TF) | 45.3 (5.35 TB/s) | 49.9 (4.87 TB/s) | 451.5 (1066 TF) | 1671.0 (1151 TF) |
| humming_w4a8_int w4 sym | 8.1 (1.11 TB/s) | 9.6 (0.95 TB/s) | 35.3 (974 TF) | 107.0 (1284 TF) | 45.8 (5.46 TB/s) | 53.8 (4.68 TB/s) | 622.5 (1545 TF) | 2391.5 (1609 TF) | 38.2 (3.27 TB/s) | 46.1 (2.72 TB/s) | 298.0 (1614 TF) | 1151.1 (1672 TF) |
| humming_w4a8_mxfp4 w4 sym | 8.0 (1.12 TB/s) | 8.9 (1.03 TB/s) | 30.7 (1119 TF) | 88.0 (1562 TF) | 42.2 (5.92 TB/s) | 48.8 (5.16 TB/s) | 520.2 (1849 TF) | 2001.3 (1923 TF) | 34.6 (3.61 TB/s) | 41.6 (3.02 TB/s) | 229.1 (2100 TF) | 911.5 (2111 TF) |
| humming_w4a4_mxfp4 w4 sym | 7.4 (1.20 TB/s) | 11.2 (0.81 TB/s) | 43.4 (791 TF) | 115.3 (1192 TF) | 42.6 (5.86 TB/s) | 53.2 (4.73 TB/s) | 655.1 (1469 TF) | 2476.8 (1554 TF) | 26.7 (4.68 TB/s) | 30.3 (4.14 TB/s) | 351.2 (1370 TF) | 1285.4 (1497 TF) |
| humming_w8a8_int8 w8 sym | 8.5 (1.97 TB/s) | 9.4 (1.81 TB/s) | 34.0 (1009 TF) | 94.0 (1463 TF) | 78.6 (5.98 TB/s) | 86.5 (5.46 TB/s) | 549.2 (1752 TF) | 2161.9 (1780 TF) | 41.7 (5.63 TB/s) | 43.2 (5.45 TB/s) | 277.9 (1731 TF) | 1267.5 (1518 TF) |
| humming_w4a8_int_i8 w4 sym | 8.0 (1.12 TB/s) | 9.2 (0.99 TB/s) | 36.7 (937 TF) | 113.1 (1215 TF) | 42.5 (5.89 TB/s) | 47.7 (5.28 TB/s) | 673.7 (1428 TF) | 2599.2 (1481 TF) | 37.3 (3.35 TB/s) | 44.4 (2.83 TB/s) | 327.0 (1471 TF) | 1244.2 (1546 TF) |
| humming w4 sym | 7.7 (1.12 TB/s) | 9.7 (0.92 TB/s) | 42.0 (818 TF) | 132.9 (1034 TF) | 53.9 (4.50 TB/s) | 54.4 (4.49 TB/s) | 865.5 (1112 TF) | 3411.2 (1128 TF) | 33.4 (3.62 TB/s) | 33.4 (3.67 TB/s) | 437.5 (1100 TF) | 1660.2 (1159 TF) |
| humming w8 sym | 8.5 (2.02 TB/s) | 10.3 (1.68 TB/s) | 47.9 (717 TF) | 141.0 (975 TF) | 84.5 (5.65 TB/s) | 90.7 (5.28 TB/s) | 1068.3 (901 TF) | 4357.1 (883 TF) | 44.6 (5.35 TB/s) | 44.5 (5.38 TB/s) | 460.6 (1044 TF) | 1788.1 (1076 TF) |

## Full results (bf16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|
| humming_w8a8_fp8 w8 sym | 8.3 / 4.1 / 2.03 | 9.3 / 57.6 / 1.82 | 32.7 / 1049.7 / 0.90 | 103.5 / 1327.3 / 0.65 |
| humming_w8a8_fp8_block128 w8 sym | 9.3 / 3.6 / 1.87 | 9.8 / 54.8 / 1.79 | 39.9 / 861.1 / 0.75 | 145.9 / 942.0 / 0.47 |
| humming_w4a8_int w4 sym | 8.1 / 4.2 / 1.11 | 9.6 / 56.0 / 0.95 | 35.3 / 973.8 / 0.61 | 107.0 / 1284.5 / 0.55 |
| humming_w4a8_mxfp4 w4 sym | 8.0 / 4.2 / 1.12 | 8.9 / 60.4 / 1.03 | 30.7 / 1119.4 / 0.70 | 88.0 / 1561.5 / 0.67 |
| humming_w4a4_mxfp4 w4 sym | 7.4 / 4.5 / 1.20 | 11.2 / 48.1 / 0.81 | 43.4 / 791.3 / 0.45 | 115.3 / 1191.8 / 0.45 |
| humming_w8a8_int8 w8 sym | 8.5 / 3.9 / 1.97 | 9.4 / 57.1 / 1.81 | 34.0 / 1009.2 / 0.86 | 94.0 / 1462.9 / 0.71 |
| humming_w4a8_int_i8 w4 sym | 8.0 / 4.2 / 1.12 | 9.2 / 58.4 / 0.99 | 36.7 / 937.5 / 0.59 | 113.1 / 1215.3 / 0.52 |
| humming w4 sym | 7.7 / 4.3 / 1.12 | 9.7 / 55.2 / 0.92 | 42.0 / 818.4 / 0.61 | 132.9 / 1034.4 / 0.57 |
| humming w8 sym | 8.5 / 4.0 / 2.02 | 10.3 / 52.1 / 1.68 | 47.9 / 717.4 / 0.71 | 141.0 / 974.5 / 0.60 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|
| humming_w8a8_fp8 w8 sym | 78.8 / 11.9 / 5.97 | 86.8 / 173.2 / 5.44 | 607.4 / 1584.0 / 0.98 | 2364.8 / 1627.3 / 0.41 |
| humming_w8a8_fp8_block128 w8 sym | 81.6 / 11.5 / 5.94 | 91.9 / 163.6 / 5.29 | 905.2 / 1062.9 / 0.68 | 3504.3 / 1098.2 / 0.28 |
| humming_w4a8_int w4 sym | 45.8 / 20.5 / 5.46 | 53.8 / 279.3 / 4.68 | 622.5 / 1545.5 / 0.60 | 2391.5 / 1609.1 / 0.32 |
| humming_w4a8_mxfp4 w4 sym | 42.2 / 22.3 / 5.92 | 48.8 / 308.2 / 5.16 | 520.2 / 1849.4 / 0.72 | 2001.3 / 1922.9 / 0.38 |
| humming_w4a4_mxfp4 w4 sym | 42.6 / 22.0 / 5.86 | 53.2 / 282.3 / 4.73 | 655.1 / 1468.5 / 0.57 | 2476.8 / 1553.7 / 0.30 |
| humming_w8a8_int8 w8 sym | 78.6 / 12.0 / 5.98 | 86.5 / 173.7 / 5.46 | 549.2 / 1751.9 / 1.08 | 2161.9 / 1780.0 / 0.45 |
| humming_w4a8_int_i8 w4 sym | 42.5 / 22.1 / 5.89 | 47.7 / 315.0 / 5.28 | 673.7 / 1428.1 / 0.56 | 2599.2 / 1480.6 / 0.29 |
| humming w4 sym | 53.9 / 17.4 / 4.50 | 54.4 / 276.1 / 4.49 | 865.5 / 1111.6 / 0.43 | 3411.2 / 1128.1 / 0.23 |
| humming w8 sym | 84.5 / 11.1 / 5.65 | 90.7 / 165.7 / 5.28 | 1068.3 / 900.6 / 0.57 | 4357.1 / 883.2 / 0.23 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|
| humming_w8a8_fp8 w8 sym | 42.0 / 11.2 / 5.59 | 44.9 / 167.6 / 5.25 | 270.2 / 1780.3 / 1.04 | 1119.9 / 1718.2 / 0.38 |
| humming_w8a8_fp8_block128 w8 sym | 45.3 / 10.4 / 5.35 | 49.9 / 150.5 / 4.87 | 451.5 / 1065.5 / 0.64 | 1671.0 / 1151.5 / 0.26 |
| humming_w4a8_int w4 sym | 38.2 / 12.3 / 3.27 | 46.1 / 163.0 / 2.72 | 298.0 / 1614.3 / 0.57 | 1151.1 / 1671.6 / 0.27 |
| humming_w4a8_mxfp4 w4 sym | 34.6 / 13.6 / 3.61 | 41.6 / 180.8 / 3.02 | 229.1 / 2099.7 / 0.75 | 911.5 / 2110.9 / 0.34 |
| humming_w4a4_mxfp4 w4 sym | 26.7 / 17.6 / 4.68 | 30.3 / 248.4 / 4.14 | 351.2 / 1369.6 / 0.45 | 1285.4 / 1496.9 / 0.20 |
| humming_w8a8_int8 w8 sym | 41.7 / 11.3 / 5.63 | 43.2 / 174.0 / 5.45 | 277.9 / 1730.8 / 1.01 | 1267.5 / 1518.1 / 0.33 |
| humming_w4a8_int_i8 w4 sym | 37.3 / 12.6 / 3.35 | 44.4 / 169.3 / 2.83 | 327.0 / 1471.2 / 0.52 | 1244.2 / 1546.5 / 0.25 |
| humming w4 sym | 33.4 / 14.0 / 3.62 | 33.4 / 225.3 / 3.67 | 437.5 / 1099.6 / 0.45 | 1660.2 / 1159.0 / 0.26 |
| humming w8 sym | 44.6 / 10.5 / 5.35 | 44.5 / 168.7 / 5.38 | 460.6 / 1044.4 / 0.68 | 1788.1 / 1076.1 / 0.30 |
