# ROCm baseline GEMM benchmark (2026-10-28T07:20:39)

Y[M,N] = X[M,K] @ W[N,K]^T; weight-only quant unless noted; group size 128; zp = asymmetric zero point, sym = uint{4,8}b{8,128}.

## Environment

- device: `AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, HIP_VISIBLE_DEVICES=3`
- torch: `2.11.0+rocm7.2`
- triton: `3.6.0`
- hip: `7.2.26015`
- date: `2026-10-28T07:20:39`
- args: `--impls torch_dense vllm_triton_w4a16 vllm_exllama_gptq dequant_triton+hipblaslt aiter_a16w8_int8 aiter_a16w8_fp8_block128 humming --bits 2 3 4 8 --windows 3 --out-dir benchmarks/rocm/results --prefix humming_vs_baselines --tag 2026-10-28`

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
| measured: copy 4 GiB (read+write) TB/s | 4.65 |
| measured: sum 4 GiB (read-only) TB/s | 3.18 |
| measured: fp16 8192^3 F.linear TFLOPs | 1443.46 |
| measured: bf16 8192^3 F.linear TFLOPs | 1591.27 |

## Methodology

- HIP-event timing; warmup >= 20; >= 100 timed calls; calls captured into a HIP graph (no host launch overhead) unless the `timing` column says eager. Best of 3 timing window(s) per point.
- L2/MALL defeat: rotating buffers. Each config builds n_copies independent (X, packed W, scales, zeros, workspace) sets so the cyclic footprint is >= 1024 MiB (4x the 256 MB MALL, capped at 256 copies); calls cycle through the copies, so every call reads cold weights from HBM.
- TFLOPs = 2MNK / t. Effective TB/s = (quantized weight bytes incl. scales/zeros + activation bytes + fp16 output bytes) / t. For dequant+GEMM fallbacks the intermediate fp16 weight traffic is NOT counted (so TB/s is comparable across implementations as 'useful bytes').
- w8a8 / w4a4 rows time the GEMM only (activation quantization done in setup, not timed).
- Correctness: each config is checked against an fp32 torch reference on N,K=(512, 1024), M in [1, 33, 300] before timing (relative Frobenius error).

## Implementation status

| config | status | source |
|---|---|---|
| torch_dense [fp16 dense] | OK: ok (max rel err 2.74e-04) | F.linear (hipBLASLt), torch 2.11.0+rocm7.2 |
| vllm_triton_w4a16 [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_triton_w4a16 [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_exllama_gptq [fp16 w4 sym] | OK: ok (max rel err 6.09e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w4 zp] | OK: ok (max rel err 6.05e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 sym] | OK: ok (max rel err 7.66e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 zp] | OK: ok (max rel err 8.24e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| dequant_triton+hipblaslt [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 sym] | OK: ok (max rel err 3.17e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 zp] | OK: ok (max rel err 3.12e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| aiter_a16w8_int8 [fp16 w8 sym] | OK: ok (max rel err 2.12e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_int8 [fp16 w8 zp] | SKIP: unsupported: symmetric only (no zero-point input) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_fp8_block128 [fp16 w8 sym] | OK: ok (max rel err 2.13e-04) | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| aiter_a16w8_fp8_block128 [fp16 w8 zp] | SKIP: unsupported: fp8 weights: no zero point | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| humming [fp16 w2 sym] | OK: ok (max rel err 2.19e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w2 zp] | OK: ok (max rel err 2.43e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w3 sym] | OK: ok (max rel err 2.48e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w3 zp] | OK: ok (max rel err 2.76e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w4 zp] | OK: ok (max rel err 2.85e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w8 sym] | OK: ok (max rel err 2.93e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w8 zp] | OK: ok (max rel err 2.90e-04) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| torch_dense [bf16 dense] | OK: ok (max rel err 2.21e-03) | F.linear (hipBLASLt), torch 2.11.0+rocm7.2 |
| vllm_triton_w4a16 [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_triton_w4a16 [bf16 w4 zp] | OK: ok (max rel err 2.47e-03) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_exllama_gptq [bf16 w4 sym] | SKIP: unsupported: exllama gptq_gemm is fp16-only | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [bf16 w4 zp] | SKIP: unsupported: exllama gptq_gemm is fp16-only | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [bf16 w8 sym] | SKIP: unsupported: exllama gptq_gemm is fp16-only | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [bf16 w8 zp] | SKIP: unsupported: exllama gptq_gemm is fp16-only | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| dequant_triton+hipblaslt [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [bf16 w4 zp] | OK: ok (max rel err 2.47e-03) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [bf16 w8 sym] | OK: ok (max rel err 2.40e-03) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [bf16 w8 zp] | OK: ok (max rel err 2.53e-03) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| aiter_a16w8_int8 [bf16 w8 sym] | OK: ok (max rel err 1.69e-03) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_int8 [bf16 w8 zp] | SKIP: unsupported: symmetric only (no zero-point input) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_fp8_block128 [bf16 w8 sym] | OK: ok (max rel err 1.68e-03) | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| aiter_a16w8_fp8_block128 [bf16 w8 zp] | SKIP: unsupported: fp8 weights: no zero point | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| humming [bf16 w2 sym] | OK: ok (max rel err 1.75e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w2 zp] | OK: ok (max rel err 1.93e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w3 sym] | OK: ok (max rel err 2.01e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w3 zp] | OK: ok (max rel err 2.18e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w4 zp] | OK: ok (max rel err 2.27e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w8 sym] | OK: ok (max rel err 2.35e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w8 zp] | OK: ok (max rel err 2.31e-03) | humming@355a62d101 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |

## Headline (fp16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 (2.73 TB/s) | 10.3 (3.29 TB/s) | 42.5 (809 TF) | 96.9 (1418 TF) | 156.5 (6.00 TB/s) | 156.1 (6.03 TB/s) | 744.8 (1292 TF) | 2661.3 (1446 TF) | 75.7 (6.21 TB/s) | 77.4 (6.08 TB/s) | 364.8 (1319 TF) | 1275.6 (1508 TF) |
| vllm_triton_w4a16 w4 sym | 74.7 (0.12 TB/s) | 75.6 (0.12 TB/s) | 105.1 (327 TF) | 247.5 (555 TF) | 196.9 (1.23 TB/s) | 223.7 (1.09 TB/s) | 1678.7 (573 TF) | 6486.1 (593 TF) | 504.0 (0.24 TB/s) | 507.1 (0.24 TB/s) | 854.8 (563 TF) | 3299.1 (583 TF) |
| vllm_triton_w4a16 w4 zp | 80.5 (0.11 TB/s) | 81.2 (0.11 TB/s) | 112.1 (307 TF) | 262.5 (524 TF) | 214.9 (1.14 TB/s) | 236.9 (1.04 TB/s) | 1788.9 (538 TF) | 6872.4 (560 TF) | 544.9 (0.22 TB/s) | 545.9 (0.23 TB/s) | 913.7 (526 TF) | 3505.3 (549 TF) |
| vllm_exllama_gptq w4 sym | 56.9 (0.15 TB/s) | 119.9 (0.07 TB/s) | 67.2 (511 TF) | 138.2 (994 TF) | 254.3 (0.95 TB/s) | 347.9 (0.70 TB/s) | 1052.7 (914 TF) | 3284.3 (1172 TF) | 1315.3 (0.09 TB/s) | 1128.5 (0.11 TB/s) | 710.8 (677 TF) | 1600.6 (1202 TF) |
| vllm_exllama_gptq w4 zp | 56.9 (0.15 TB/s) | 120.0 (0.07 TB/s) | 67.3 (511 TF) | 138.8 (990 TF) | 252.8 (0.97 TB/s) | 348.0 (0.71 TB/s) | 1051.7 (915 TF) | 3291.9 (1169 TF) | 1317.3 (0.09 TB/s) | 1112.7 (0.11 TB/s) | 711.2 (676 TF) | 1601.7 (1201 TF) |
| vllm_exllama_gptq w8 sym | 56.2 (0.30 TB/s) | 124.3 (0.14 TB/s) | 69.9 (491 TF) | 145.8 (942 TF) | 250.4 (1.91 TB/s) | 432.0 (1.11 TB/s) | 1090.3 (882 TF) | 3343.0 (1151 TF) | 1271.3 (0.19 TB/s) | 825.1 (0.29 TB/s) | 717.4 (671 TF) | 1619.9 (1188 TF) |
| vllm_exllama_gptq w8 zp | 55.7 (0.31 TB/s) | 124.4 (0.14 TB/s) | 69.7 (493 TF) | 145.5 (945 TF) | 250.3 (1.92 TB/s) | 432.3 (1.12 TB/s) | 1090.5 (882 TF) | 3348.7 (1149 TF) | 1279.7 (0.19 TB/s) | 826.0 (0.29 TB/s) | 715.9 (672 TF) | 1622.5 (1186 TF) |
| dequant_triton+hipblaslt w4 sym | 25.2 (0.34 TB/s) | 22.9 (0.39 TB/s) | 51.2 (671 TF) | 108.3 (1269 TF) | 419.8 (0.58 TB/s) | 420.1 (0.58 TB/s) | 999.7 (962 TF) | 2881.8 (1335 TF) | 187.8 (0.65 TB/s) | 213.3 (0.57 TB/s) | 493.0 (976 TF) | 1385.6 (1389 TF) |
| dequant_triton+hipblaslt w4 zp | 25.3 (0.34 TB/s) | 23.0 (0.39 TB/s) | 52.6 (654 TF) | 108.2 (1270 TF) | 443.4 (0.55 TB/s) | 430.9 (0.57 TB/s) | 1015.6 (947 TF) | 2902.4 (1326 TF) | 201.8 (0.60 TB/s) | 229.8 (0.54 TB/s) | 501.0 (960 TF) | 1402.8 (1372 TF) |
| dequant_triton+hipblaslt w8 sym | 24.8 (0.69 TB/s) | 23.6 (0.73 TB/s) | 51.4 (669 TF) | 110.4 (1245 TF) | 449.5 (1.06 TB/s) | 455.5 (1.05 TB/s) | 1033.2 (931 TF) | 2936.5 (1311 TF) | 213.7 (1.12 TB/s) | 229.8 (1.04 TB/s) | 492.4 (977 TF) | 1405.5 (1369 TF) |
| dequant_triton+hipblaslt w8 zp | 25.0 (0.69 TB/s) | 23.4 (0.75 TB/s) | 51.4 (669 TF) | 110.4 (1245 TF) | 454.1 (1.06 TB/s) | 459.6 (1.05 TB/s) | 1031.8 (932 TF) | 2948.1 (1305 TF) | 233.6 (1.03 TB/s) | 215.8 (1.12 TB/s) | 514.2 (935 TF) | 1415.2 (1360 TF) |
| aiter_a16w8_int8 w8 sym | 9.2 (1.89 TB/s) | 10.6 (1.66 TB/s) | 68.3 (503 TF) | 245.0 (561 TF) | 94.9 (5.11 TB/s) | 123.7 (3.93 TB/s) | 1729.8 (556 TF) | 6643.7 (579 TF) | 48.7 (4.97 TB/s) | 57.8 (4.21 TB/s) | 868.1 (554 TF) | 3313.9 (581 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.15 TB/s) | 9.5 (1.79 TB/s) | 55.3 (621 TF) | 199.4 (689 TF) | 76.6 (6.13 TB/s) | 98.5 (4.79 TB/s) | 1371.5 (701 TF) | 5261.4 (731 TF) | 41.9 (5.61 TB/s) | 46.2 (5.11 TB/s) | 677.9 (710 TF) | 2596.4 (741 TF) |
| humming w2 sym | 7.2 (0.62 TB/s) | 10.4 (0.46 TB/s) | 45.4 (757 TF) | 143.2 (960 TF) | 40.7 (3.07 TB/s) | 56.8 (2.23 TB/s) | 909.7 (1058 TF) | 3577.4 (1076 TF) | 28.7 (2.18 TB/s) | 37.8 (1.68 TB/s) | 467.8 (1028 TF) | 1795.8 (1071 TF) |
| humming w2 zp | 7.3 (0.63 TB/s) | 10.6 (0.46 TB/s) | 47.8 (719 TF) | 148.6 (925 TF) | 43.5 (2.95 TB/s) | 59.6 (2.19 TB/s) | 938.3 (1025 TF) | 3697.0 (1041 TF) | 31.4 (2.04 TB/s) | 40.6 (1.61 TB/s) | 476.2 (1010 TF) | 1835.8 (1048 TF) |
| humming w3 sym | 7.3 (0.90 TB/s) | 10.7 (0.64 TB/s) | 46.7 (735 TF) | 146.5 (938 TF) | 47.1 (3.90 TB/s) | 62.7 (2.96 TB/s) | 948.9 (1014 TF) | 3741.5 (1029 TF) | 31.6 (2.90 TB/s) | 40.2 (2.31 TB/s) | 484.5 (993 TF) | 1854.8 (1037 TF) |
| humming w3 zp | 7.5 (0.89 TB/s) | 10.8 (0.65 TB/s) | 47.5 (723 TF) | 149.1 (922 TF) | 49.6 (3.78 TB/s) | 63.6 (2.98 TB/s) | 953.3 (1009 TF) | 3725.2 (1033 TF) | 34.1 (2.74 TB/s) | 40.9 (2.32 TB/s) | 479.8 (1003 TF) | 1852.0 (1039 TF) |
| humming w4 sym | 7.3 (1.18 TB/s) | 10.6 (0.84 TB/s) | 46.3 (742 TF) | 150.3 (915 TF) | 50.4 (4.80 TB/s) | 63.1 (3.87 TB/s) | 962.5 (1000 TF) | 3771.3 (1020 TF) | 34.5 (3.51 TB/s) | 39.8 (3.07 TB/s) | 489.6 (982 TF) | 1861.4 (1034 TF) |
| humming w4 zp | 7.6 (1.15 TB/s) | 10.7 (0.85 TB/s) | 50.0 (687 TF) | 151.4 (908 TF) | 52.6 (4.68 TB/s) | 65.2 (3.81 TB/s) | 988.9 (973 TF) | 3915.8 (983 TF) | 34.4 (3.58 TB/s) | 40.5 (3.06 TB/s) | 487.8 (986 TF) | 1862.2 (1033 TF) |
| humming w8 sym | 8.5 (2.00 TB/s) | 11.1 (1.56 TB/s) | 51.8 (664 TF) | 153.9 (893 TF) | 85.2 (5.60 TB/s) | 89.5 (5.36 TB/s) | 1139.2 (845 TF) | 4578.6 (840 TF) | 45.6 (5.24 TB/s) | 47.0 (5.10 TB/s) | 500.2 (962 TF) | 1952.8 (985 TF) |
| humming w8 zp | 8.6 (2.00 TB/s) | 11.2 (1.56 TB/s) | 54.2 (634 TF) | 157.8 (871 TF) | 87.4 (5.50 TB/s) | 90.3 (5.35 TB/s) | 1173.9 (820 TF) | 4681.7 (822 TF) | 46.1 (5.21 TB/s) | 48.2 (5.01 TB/s) | 505.6 (951 TF) | 1976.5 (974 TF) |

## Headline (bf16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 (2.72 TB/s) | 10.3 (3.27 TB/s) | 47.0 (732 TF) | 94.3 (1458 TF) | 151.5 (6.20 TB/s) | 156.4 (6.02 TB/s) | 723.3 (1330 TF) | 2544.0 (1513 TF) | 75.1 (6.25 TB/s) | 74.7 (6.30 TB/s) | 355.1 (1355 TF) | 1218.1 (1580 TF) |
| vllm_triton_w4a16 w4 sym | 84.8 (0.10 TB/s) | 85.1 (0.10 TB/s) | 135.7 (253 TF) | 307.1 (447 TF) | 283.6 (0.85 TB/s) | 297.3 (0.82 TB/s) | 2092.8 (460 TF) | 8104.0 (475 TF) | 577.6 (0.21 TB/s) | 576.5 (0.21 TB/s) | 1061.4 (453 TF) | 4185.9 (460 TF) |
| vllm_triton_w4a16 w4 zp | 88.8 (0.10 TB/s) | 89.4 (0.10 TB/s) | 126.4 (272 TF) | 339.6 (405 TF) | 307.8 (0.79 TB/s) | 317.0 (0.78 TB/s) | 2267.8 (424 TF) | 8887.8 (433 TF) | 605.1 (0.20 TB/s) | 601.1 (0.20 TB/s) | 1187.4 (405 TF) | 4578.3 (420 TF) |
| dequant_triton+hipblaslt w4 sym | 24.9 (0.35 TB/s) | 22.9 (0.39 TB/s) | 58.0 (592 TF) | 106.7 (1288 TF) | 418.0 (0.58 TB/s) | 417.8 (0.58 TB/s) | 968.6 (993 TF) | 2767.7 (1390 TF) | 186.8 (0.65 TB/s) | 201.7 (0.61 TB/s) | 476.2 (1010 TF) | 1333.2 (1443 TF) |
| dequant_triton+hipblaslt w4 zp | 25.0 (0.35 TB/s) | 23.4 (0.38 TB/s) | 57.7 (596 TF) | 106.8 (1286 TF) | 432.1 (0.56 TB/s) | 431.0 (0.57 TB/s) | 980.3 (981 TF) | 2795.6 (1377 TF) | 201.5 (0.61 TB/s) | 219.5 (0.56 TB/s) | 494.9 (972 TF) | 1346.5 (1429 TF) |
| dequant_triton+hipblaslt w8 sym | 24.9 (0.69 TB/s) | 23.3 (0.74 TB/s) | 58.0 (593 TF) | 106.9 (1286 TF) | 467.0 (1.02 TB/s) | 471.5 (1.02 TB/s) | 1020.2 (943 TF) | 2809.4 (1370 TF) | 211.4 (1.13 TB/s) | 212.4 (1.13 TB/s) | 491.1 (980 TF) | 1343.5 (1432 TF) |
| dequant_triton+hipblaslt w8 zp | 25.6 (0.67 TB/s) | 23.5 (0.74 TB/s) | 58.6 (587 TF) | 108.6 (1266 TF) | 466.6 (1.03 TB/s) | 467.6 (1.03 TB/s) | 1009.6 (953 TF) | 2810.1 (1369 TF) | 225.9 (1.06 TB/s) | 230.3 (1.05 TB/s) | 492.8 (976 TF) | 1352.9 (1422 TF) |
| aiter_a16w8_int8 w8 sym | 9.6 (1.80 TB/s) | 12.7 (1.39 TB/s) | 64.1 (536 TF) | 231.4 (594 TF) | 99.8 (4.86 TB/s) | 133.4 (3.65 TB/s) | 1597.1 (602 TF) | 6166.1 (624 TF) | 49.3 (4.92 TB/s) | 70.4 (3.46 TB/s) | 795.0 (605 TF) | 3052.8 (630 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.15 TB/s) | 9.5 (1.80 TB/s) | 54.7 (628 TF) | 197.3 (697 TF) | 76.5 (6.14 TB/s) | 101.0 (4.67 TB/s) | 1358.5 (708 TF) | 5227.7 (736 TF) | 42.2 (5.58 TB/s) | 47.3 (4.99 TB/s) | 671.6 (716 TF) | 2568.3 (749 TF) |
| humming w2 sym | 7.8 (0.58 TB/s) | 11.1 (0.43 TB/s) | 50.0 (688 TF) | 159.5 (862 TF) | 55.8 (2.24 TB/s) | 75.9 (1.67 TB/s) | 1017.9 (945 TF) | 4098.7 (939 TF) | 33.6 (1.86 TB/s) | 45.5 (1.40 TB/s) | 527.1 (913 TF) | 2020.1 (952 TF) |
| humming w2 zp | 7.9 (0.58 TB/s) | 11.3 (0.43 TB/s) | 50.9 (676 TF) | 159.9 (859 TF) | 57.7 (2.23 TB/s) | 77.1 (1.69 TB/s) | 1032.6 (932 TF) | 4128.0 (932 TF) | 36.9 (1.74 TB/s) | 48.4 (1.35 TB/s) | 536.1 (897 TF) | 2045.7 (941 TF) |
| humming w3 sym | 7.9 (0.83 TB/s) | 11.3 (0.60 TB/s) | 51.1 (673 TF) | 166.3 (826 TF) | 60.2 (3.05 TB/s) | 81.2 (2.29 TB/s) | 1064.2 (904 TF) | 4288.1 (897 TF) | 35.4 (2.59 TB/s) | 47.0 (1.98 TB/s) | 536.1 (897 TF) | 2077.2 (926 TF) |
| humming w3 zp | 8.1 (0.82 TB/s) | 11.4 (0.61 TB/s) | 51.4 (668 TF) | 161.5 (851 TF) | 62.2 (3.01 TB/s) | 80.9 (2.34 TB/s) | 1067.0 (902 TF) | 4280.9 (899 TF) | 40.0 (2.34 TB/s) | 48.1 (1.97 TB/s) | 538.0 (894 TF) | 2087.6 (922 TF) |
| humming w4 sym | 7.8 (1.12 TB/s) | 11.2 (0.80 TB/s) | 51.4 (669 TF) | 166.1 (827 TF) | 60.6 (4.00 TB/s) | 79.3 (3.08 TB/s) | 1083.2 (888 TF) | 4361.4 (882 TF) | 35.1 (3.45 TB/s) | 46.5 (2.63 TB/s) | 537.7 (895 TF) | 2096.5 (918 TF) |
| humming w4 zp | 8.1 (1.08 TB/s) | 11.4 (0.79 TB/s) | 52.7 (652 TF) | 167.4 (821 TF) | 61.8 (3.98 TB/s) | 80.4 (3.08 TB/s) | 1103.3 (872 TF) | 4433.6 (868 TF) | 40.6 (3.03 TB/s) | 47.9 (2.59 TB/s) | 549.5 (875 TF) | 2117.2 (909 TF) |
| humming w8 sym | 8.5 (2.00 TB/s) | 11.8 (1.47 TB/s) | 55.9 (615 TF) | 174.2 (789 TF) | 88.0 (5.42 TB/s) | 97.8 (4.90 TB/s) | 1271.1 (757 TF) | 5054.3 (761 TF) | 45.5 (5.25 TB/s) | 54.8 (4.37 TB/s) | 563.3 (854 TF) | 2232.5 (862 TF) |
| humming w8 zp | 8.7 (1.98 TB/s) | 11.9 (1.46 TB/s) | 56.8 (605 TF) | 174.1 (789 TF) | 90.0 (5.34 TB/s) | 99.0 (4.88 TB/s) | 1292.8 (744 TF) | 5123.5 (751 TF) | 46.4 (5.19 TB/s) | 54.9 (4.40 TB/s) | 576.1 (835 TF) | 2282.8 (843 TF) |

## Full results (fp16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 / 2.7 / 2.73 | 9.4 / 14.3 / 3.57 | 10.3 / 52.3 / 3.29 | 12.6 / 85.1 / 2.70 | 13.1 / 163.8 / 2.64 | 17.0 / 253.0 / 2.10 | 22.6 / 380.8 / 1.67 | 42.5 / 808.9 / 1.19 | 96.9 / 1418.2 / 1.04 |
| vllm_triton_w4a16 w4 sym | 74.7 / 0.4 / 0.12 | 75.0 / 1.8 / 0.12 | 75.6 / 7.1 / 0.12 | 76.3 / 14.1 / 0.12 | 77.1 / 27.9 / 0.13 | 105.3 / 40.8 / 0.10 | 104.2 / 82.4 / 0.12 | 105.1 / 327.0 / 0.24 | 247.5 / 555.4 / 0.31 |
| vllm_triton_w4a16 w4 zp | 80.5 / 0.4 / 0.11 | 80.8 / 1.7 / 0.11 | 81.2 / 6.6 / 0.11 | 81.7 / 13.1 / 0.11 | 84.6 / 25.4 / 0.12 | 112.7 / 38.1 / 0.10 | 111.1 / 77.3 / 0.12 | 112.1 / 306.6 / 0.23 | 262.5 / 523.6 / 0.29 |
| vllm_exllama_gptq w4 sym | 56.9 / 0.6 / 0.15 | 71.2 / 1.9 / 0.12 | 119.9 / 4.5 / 0.07 | 167.4 / 6.4 / 0.06 | 41.4 / 51.9 / 0.23 | 43.7 / 98.3 / 0.25 | 49.3 / 174.1 / 0.26 | 67.2 / 511.0 / 0.38 | 138.2 / 994.4 / 0.55 |
| vllm_exllama_gptq w4 zp | 56.9 / 0.6 / 0.15 | 71.1 / 1.9 / 0.12 | 120.0 / 4.5 / 0.07 | 167.0 / 6.4 / 0.06 | 41.4 / 51.8 / 0.24 | 43.7 / 98.3 / 0.25 | 49.3 / 174.1 / 0.26 | 67.3 / 510.6 / 0.38 | 138.8 / 990.2 / 0.55 |
| vllm_exllama_gptq w8 sym | 56.2 / 0.6 / 0.30 | 74.5 / 1.8 / 0.23 | 124.3 / 4.3 / 0.14 | 44.7 / 24.0 / 0.39 | 45.0 / 47.8 / 0.40 | 47.0 / 91.3 / 0.41 | 53.1 / 161.9 / 0.40 | 69.9 / 491.2 / 0.48 | 145.8 / 942.4 / 0.58 |
| vllm_exllama_gptq w8 zp | 55.7 / 0.6 / 0.31 | 72.4 / 1.9 / 0.24 | 124.4 / 4.3 / 0.14 | 44.7 / 24.0 / 0.40 | 45.0 / 47.7 / 0.41 | 46.8 / 91.8 / 0.41 | 53.2 / 161.6 / 0.40 | 69.7 / 492.6 / 0.49 | 145.5 / 944.7 / 0.58 |
| dequant_triton+hipblaslt w4 sym | 25.2 / 1.3 / 0.34 | 22.5 / 6.0 / 0.39 | 22.9 / 23.5 / 0.39 | 24.7 / 43.5 / 0.37 | 25.0 / 86.0 / 0.39 | 28.7 / 149.9 / 0.38 | 32.2 / 266.7 / 0.40 | 51.2 / 670.9 / 0.50 | 108.3 / 1268.8 / 0.70 |
| dequant_triton+hipblaslt w4 zp | 25.3 / 1.3 / 0.34 | 22.0 / 6.1 / 0.40 | 23.0 / 23.3 / 0.39 | 24.3 / 44.2 / 0.38 | 24.7 / 86.8 / 0.40 | 28.4 / 151.3 / 0.38 | 32.1 / 267.9 / 0.40 | 52.6 / 653.8 / 0.48 | 108.2 / 1270.2 / 0.70 |
| dequant_triton+hipblaslt w8 sym | 24.8 / 1.4 / 0.69 | 22.4 / 6.0 / 0.76 | 23.6 / 22.8 / 0.73 | 25.0 / 42.9 / 0.70 | 25.3 / 85.0 / 0.72 | 28.2 / 152.1 / 0.68 | 32.7 / 262.6 / 0.65 | 51.4 / 668.8 / 0.66 | 110.4 / 1244.7 / 0.76 |
| dequant_triton+hipblaslt w8 zp | 25.0 / 1.3 / 0.69 | 21.7 / 6.2 / 0.79 | 23.4 / 23.0 / 0.75 | 24.4 / 44.0 / 0.72 | 25.2 / 85.2 / 0.72 | 28.8 / 149.4 / 0.67 | 32.2 / 266.5 / 0.66 | 51.4 / 669.0 / 0.66 | 110.4 / 1245.4 / 0.76 |
| aiter_a16w8_int8 w8 sym | 9.2 / 3.7 / 1.89 | 10.0 / 13.4 / 1.74 | 10.6 / 50.6 / 1.66 | 12.4 / 86.8 / 1.44 | 15.5 / 138.5 / 1.18 | 26.1 / 164.3 / 0.74 | 34.6 / 248.2 / 0.62 | 68.3 / 502.9 / 0.50 | 245.0 / 560.9 / 0.34 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.15 | 8.7 / 15.5 / 1.94 | 9.5 / 56.5 / 1.79 | 11.9 / 90.6 / 1.46 | 13.9 / 154.3 / 1.28 | 24.0 / 178.7 / 0.79 | 37.0 / 232.4 / 0.57 | 55.3 / 621.2 / 0.61 | 199.4 / 689.3 / 0.42 |
| humming w2 sym | 7.2 / 4.7 / 0.62 | 7.9 / 16.9 / 0.57 | 10.4 / 51.7 / 0.46 | 10.5 / 102.1 / 0.47 | 13.2 / 162.6 / 0.42 | 20.7 / 208.0 / 0.32 | 22.7 / 378.7 / 0.38 | 45.4 / 756.7 / 0.47 | 143.2 / 959.8 / 0.50 |
| humming w2 zp | 7.3 / 4.6 / 0.63 | 8.0 / 16.8 / 0.58 | 10.6 / 50.9 / 0.46 | 10.8 / 99.8 / 0.47 | 13.3 / 161.6 / 0.42 | 20.6 / 208.0 / 0.32 | 22.7 / 379.0 / 0.39 | 47.8 / 719.4 / 0.45 | 148.6 / 925.2 / 0.48 |
| humming w3 sym | 7.3 / 4.6 / 0.90 | 8.0 / 16.8 / 0.83 | 10.7 / 50.3 / 0.64 | 10.9 / 98.4 / 0.65 | 13.5 / 158.6 / 0.56 | 21.3 / 201.8 / 0.41 | 22.8 / 376.7 / 0.47 | 46.7 / 735.3 / 0.50 | 146.5 / 937.8 / 0.50 |
| humming w3 zp | 7.5 / 4.5 / 0.89 | 8.1 / 16.6 / 0.83 | 10.8 / 49.8 / 0.65 | 11.0 / 97.5 / 0.66 | 13.7 / 156.5 / 0.56 | 21.2 / 202.8 / 0.41 | 23.1 / 371.3 / 0.47 | 47.5 / 722.7 / 0.49 | 149.1 / 921.6 / 0.49 |
| humming w4 sym | 7.3 / 4.6 / 1.18 | 8.0 / 16.7 / 1.08 | 10.6 / 50.8 / 0.84 | 10.7 / 99.9 / 0.85 | 13.5 / 159.2 / 0.72 | 21.0 / 204.7 / 0.51 | 22.7 / 378.5 / 0.57 | 46.3 / 742.3 / 0.55 | 150.3 / 914.7 / 0.50 |
| humming w4 zp | 7.6 / 4.4 / 1.15 | 8.2 / 16.3 / 1.08 | 10.7 / 50.4 / 0.85 | 10.8 / 99.3 / 0.86 | 13.9 / 154.5 / 0.71 | 21.3 / 201.7 / 0.51 | 23.0 / 374.2 / 0.56 | 50.0 / 687.1 / 0.51 | 151.4 / 907.6 / 0.50 |
| humming w8 sym | 8.5 / 3.9 / 2.00 | 9.0 / 14.9 / 1.90 | 11.1 / 48.3 / 1.56 | 11.4 / 94.5 / 1.55 | 14.3 / 149.8 / 1.26 | 22.4 / 191.6 / 0.85 | 24.5 / 351.2 / 0.87 | 51.8 / 663.7 / 0.65 | 153.9 / 893.2 / 0.55 |
| humming w8 zp | 8.6 / 3.9 / 2.00 | 9.1 / 14.8 / 1.90 | 11.2 / 48.1 / 1.56 | 11.4 / 94.2 / 1.55 | 14.1 / 152.7 / 1.30 | 22.2 / 193.6 / 0.87 | 24.5 / 350.7 / 0.87 | 54.2 / 634.4 / 0.63 | 157.8 / 870.8 / 0.53 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 13.9 / 3.6 / 3.61 | 14.0 / 14.3 / 3.59 | 13.8 / 58.5 / 3.68 | 14.4 / 111.7 / 3.54 | 17.9 / 179.5 / 2.88 | 22.3 / 289.0 / 2.38 | 27.6 / 467.3 / 2.02 | 65.6 / 785.6 / 1.09 | 165.1 / 1248.3 / 0.81 |
| vllm_triton_w4a16 w4 sym | 75.0 / 0.7 / 0.17 | 75.5 / 2.7 / 0.17 | 75.6 / 10.7 / 0.18 | 76.2 / 21.1 / 0.18 | 77.3 / 41.7 / 0.18 | 105.3 / 61.2 / 0.15 | 104.3 / 123.5 / 0.17 | 122.7 / 420.1 / 0.28 | 365.5 / 564.1 / 0.27 |
| vllm_triton_w4a16 w4 zp | 80.6 / 0.6 / 0.16 | 81.0 / 2.5 / 0.16 | 81.2 / 9.9 / 0.17 | 81.6 / 19.7 / 0.17 | 84.8 / 38.0 / 0.17 | 112.4 / 57.3 / 0.14 | 111.4 / 115.7 / 0.16 | 131.8 / 391.1 / 0.26 | 389.4 / 529.4 / 0.25 |
| vllm_exllama_gptq w4 sym | 47.0 / 1.1 / 0.28 | 71.3 / 2.8 / 0.18 | 140.5 / 5.7 / 0.10 | 161.6 / 10.0 / 0.08 | 44.3 / 72.7 / 0.32 | 49.6 / 129.8 / 0.31 | 61.1 / 211.0 / 0.30 | 96.6 / 533.3 / 0.35 | 211.1 / 976.4 / 0.46 |
| vllm_exllama_gptq w4 zp | 48.7 / 1.0 / 0.27 | 71.2 / 2.8 / 0.18 | 141.2 / 5.7 / 0.10 | 161.9 / 9.9 / 0.09 | 44.2 / 72.9 / 0.33 | 49.6 / 130.0 / 0.32 | 61.4 / 209.9 / 0.30 | 96.4 / 534.7 / 0.35 | 211.7 / 973.8 / 0.46 |
| vllm_exllama_gptq w8 sym | 52.2 / 1.0 / 0.49 | 72.6 / 2.8 / 0.35 | 146.9 / 5.5 / 0.18 | 44.9 / 35.9 / 0.58 | 46.2 / 69.7 / 0.58 | 52.0 / 123.9 / 0.54 | 63.9 / 201.7 / 0.48 | 99.4 / 518.3 / 0.47 | 216.1 / 954.0 / 0.51 |
| vllm_exllama_gptq w8 zp | 52.0 / 1.0 / 0.49 | 72.6 / 2.8 / 0.36 | 147.8 / 5.4 / 0.18 | 45.0 / 35.8 / 0.59 | 46.3 / 69.6 / 0.58 | 52.2 / 123.5 / 0.54 | 64.0 / 201.5 / 0.48 | 99.6 / 517.3 / 0.47 | 216.1 / 954.1 / 0.51 |
| dequant_triton+hipblaslt w4 sym | 28.4 / 1.8 / 0.46 | 28.9 / 7.0 / 0.45 | 28.3 / 28.4 / 0.47 | 28.9 / 55.7 / 0.47 | 34.3 / 93.8 / 0.42 | 35.6 / 180.8 / 0.44 | 41.3 / 311.8 / 0.44 | 78.4 / 657.4 / 0.43 | 176.8 / 1165.7 / 0.55 |
| dequant_triton+hipblaslt w4 zp | 29.8 / 1.7 / 0.44 | 29.3 / 6.9 / 0.45 | 28.5 / 28.3 / 0.47 | 29.6 / 54.4 / 0.46 | 34.1 / 94.5 / 0.42 | 35.8 / 179.9 / 0.44 | 41.7 / 309.1 / 0.44 | 79.2 / 650.5 / 0.43 | 178.2 / 1156.8 / 0.54 |
| dequant_triton+hipblaslt w8 sym | 30.9 / 1.6 / 0.83 | 30.7 / 6.6 / 0.84 | 29.6 / 27.2 / 0.88 | 30.7 / 52.5 / 0.85 | 35.2 / 91.6 / 0.76 | 37.8 / 170.3 / 0.74 | 43.3 / 297.5 / 0.71 | 81.4 / 632.8 / 0.57 | 179.0 / 1151.6 / 0.61 |
| dequant_triton+hipblaslt w8 zp | 30.8 / 1.6 / 0.84 | 31.2 / 6.5 / 0.83 | 29.6 / 27.2 / 0.88 | 30.8 / 52.2 / 0.86 | 35.1 / 91.7 / 0.77 | 38.0 / 169.7 / 0.75 | 42.2 / 305.2 / 0.73 | 81.6 / 631.4 / 0.57 | 179.7 / 1147.5 / 0.61 |
| aiter_a16w8_int8 w8 sym | 11.0 / 4.6 / 2.37 | 11.7 / 17.3 / 2.23 | 12.7 / 63.3 / 2.07 | 14.7 / 109.3 / 1.81 | 18.9 / 170.6 / 1.44 | 29.0 / 222.5 / 0.99 | 38.9 / 330.8 / 0.80 | 127.6 / 403.9 / 0.37 | 366.5 / 562.5 / 0.30 |
| aiter_a16w8_fp8_block128 w8 sym | 8.9 / 5.6 / 2.82 | 9.7 / 20.7 / 2.59 | 11.0 / 73.0 / 2.31 | 13.9 / 115.7 / 1.85 | 17.4 / 185.3 / 1.52 | 26.7 / 241.5 / 1.04 | 38.8 / 331.7 / 0.78 | 100.5 / 512.8 / 0.46 | 297.1 / 693.8 / 0.37 |
| humming w2 sym | 8.8 / 5.7 / 0.76 | 9.9 / 20.4 / 0.69 | 11.0 / 73.1 / 0.64 | 11.2 / 143.2 / 0.65 | 14.9 / 215.9 / 0.54 | 24.2 / 265.9 / 0.38 | 28.0 / 460.5 / 0.43 | 63.8 / 808.4 / 0.43 | 209.0 / 986.5 / 0.43 |
| humming w2 zp | 9.0 / 5.6 / 0.76 | 10.0 / 20.0 / 0.69 | 11.1 / 72.6 / 0.65 | 11.3 / 142.8 / 0.67 | 15.0 / 214.3 / 0.55 | 24.1 / 267.2 / 0.39 | 28.8 / 447.9 / 0.42 | 66.7 / 773.0 / 0.42 | 218.0 / 945.9 / 0.42 |
| humming w3 sym | 9.0 / 5.6 / 1.09 | 9.9 / 20.3 / 1.00 | 11.3 / 71.2 / 0.90 | 11.5 / 139.6 / 0.91 | 15.4 / 209.7 / 0.72 | 24.6 / 262.3 / 0.51 | 29.0 / 444.0 / 0.52 | 66.5 / 774.6 / 0.46 | 213.9 / 963.6 / 0.44 |
| humming w3 zp | 9.2 / 5.5 / 1.09 | 10.1 / 19.9 / 1.00 | 11.4 / 70.7 / 0.91 | 11.7 / 137.5 / 0.91 | 15.4 / 208.7 / 0.73 | 24.4 / 263.8 / 0.52 | 29.1 / 443.1 / 0.53 | 66.5 / 775.4 / 0.47 | 216.7 / 951.3 / 0.43 |
| humming w4 sym | 9.1 / 5.5 / 1.43 | 10.0 / 20.2 / 1.31 | 11.2 / 72.1 / 1.19 | 11.4 / 140.9 / 1.19 | 14.8 / 217.6 / 0.96 | 24.1 / 266.9 / 0.65 | 28.7 / 448.8 / 0.64 | 67.2 / 766.5 / 0.51 | 214.2 / 962.6 / 0.45 |
| humming w4 zp | 9.4 / 5.3 / 1.40 | 10.2 / 19.7 / 1.30 | 11.2 / 71.8 / 1.20 | 11.5 / 139.6 / 1.20 | 15.5 / 207.5 / 0.93 | 24.4 / 264.2 / 0.65 | 29.5 / 437.0 / 0.62 | 69.9 / 737.3 / 0.49 | 223.1 / 924.0 / 0.43 |
| humming w8 sym | 11.1 / 4.5 / 2.30 | 11.8 / 17.0 / 2.17 | 12.6 / 63.8 / 2.05 | 13.0 / 124.0 / 2.02 | 16.8 / 191.4 / 1.60 | 27.8 / 231.8 / 1.01 | 31.5 / 409.3 / 0.98 | 69.1 / 746.4 / 0.67 | 223.0 / 924.5 / 0.49 |
| humming w8 zp | 11.2 / 4.5 / 2.30 | 12.0 / 16.8 / 2.15 | 12.8 / 62.9 / 2.04 | 13.1 / 122.5 / 2.01 | 16.6 / 193.5 / 1.63 | 27.4 / 235.0 / 1.03 | 31.5 / 408.9 / 0.98 | 71.5 / 721.2 / 0.65 | 230.0 / 896.3 / 0.48 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 48.0 / 4.9 / 4.89 | 50.3 / 18.7 / 4.68 | 45.3 / 82.9 / 5.21 | 48.4 / 155.3 / 4.90 | 47.1 / 319.4 / 5.08 | 50.7 / 592.6 / 4.79 | 67.8 / 886.9 / 3.71 | 195.3 / 1231.8 / 1.55 | 663.4 / 1450.2 / 0.76 |
| vllm_triton_w4a16 w4 sym | 78.7 / 3.0 / 0.77 | 78.9 / 11.9 / 0.77 | 79.3 / 47.4 / 0.78 | 79.5 / 94.5 / 0.79 | 80.4 / 187.0 / 0.81 | 105.9 / 284.0 / 0.65 | 130.1 / 462.2 / 0.59 | 449.7 / 534.9 / 0.28 | 1662.2 / 578.8 / 0.20 |
| vllm_triton_w4a16 w4 zp | 86.6 / 2.7 / 0.70 | 87.8 / 10.7 / 0.70 | 88.1 / 42.6 / 0.70 | 88.5 / 85.0 / 0.71 | 90.1 / 166.9 / 0.72 | 113.6 / 264.6 / 0.61 | 139.1 / 432.1 / 0.56 | 477.0 / 504.2 / 0.27 | 1775.0 / 542.0 / 0.19 |
| vllm_exllama_gptq w4 sym | 79.4 / 3.0 / 0.76 | 138.5 / 6.8 / 0.44 | 194.5 / 19.3 / 0.32 | 195.1 / 38.5 / 0.32 | 101.1 / 148.7 / 0.64 | 108.2 / 278.0 / 0.64 | 127.7 / 470.7 / 0.60 | 273.2 / 880.2 / 0.47 | 858.4 / 1120.8 / 0.38 |
| vllm_exllama_gptq w4 zp | 79.2 / 3.0 / 0.77 | 137.5 / 6.8 / 0.45 | 194.0 / 19.4 / 0.32 | 194.8 / 38.6 / 0.32 | 102.2 / 147.1 / 0.64 | 109.0 / 275.9 / 0.64 | 128.8 / 466.8 / 0.60 | 274.5 / 876.3 / 0.47 | 860.4 / 1118.2 / 0.38 |
| vllm_exllama_gptq w8 sym | 89.4 / 2.6 / 1.33 | 138.1 / 6.8 / 0.87 | 182.0 / 20.6 / 0.66 | 118.4 / 63.5 / 1.02 | 109.2 / 137.7 / 1.13 | 117.2 / 256.6 / 1.09 | 142.6 / 421.7 / 0.95 | 286.6 / 839.3 / 0.65 | 873.1 / 1101.9 / 0.44 |
| vllm_exllama_gptq w8 zp | 87.2 / 2.7 / 1.38 | 136.4 / 6.9 / 0.88 | 181.9 / 20.7 / 0.67 | 118.6 / 63.4 / 1.03 | 109.0 / 137.9 / 1.14 | 117.1 / 256.8 / 1.10 | 142.7 / 421.5 / 0.96 | 283.4 / 848.5 / 0.66 | 874.7 / 1099.9 / 0.44 |
| dequant_triton+hipblaslt w4 sym | 99.8 / 2.4 / 0.61 | 88.7 / 10.6 / 0.69 | 99.4 / 37.8 / 0.62 | 100.3 / 75.0 / 0.62 | 110.3 / 136.3 / 0.59 | 103.4 / 290.7 / 0.67 | 147.2 / 408.4 / 0.53 | 243.2 / 988.8 / 0.53 | 711.0 / 1353.1 / 0.46 |
| dequant_triton+hipblaslt w4 zp | 89.8 / 2.6 / 0.68 | 91.7 / 10.2 / 0.67 | 94.4 / 39.8 / 0.66 | 113.5 / 66.2 / 0.56 | 99.2 / 151.5 / 0.66 | 115.6 / 260.1 / 0.60 | 128.0 / 469.9 / 0.61 | 248.3 / 968.5 / 0.52 | 705.0 / 1364.7 / 0.47 |
| dequant_triton+hipblaslt w8 sym | 119.5 / 2.0 / 1.00 | 120.4 / 7.8 / 0.99 | 107.8 / 34.8 / 1.12 | 120.6 / 62.3 / 1.01 | 118.2 / 127.1 / 1.04 | 121.7 / 247.1 / 1.05 | 158.8 / 378.5 / 0.86 | 258.6 / 930.2 / 0.72 | 715.3 / 1345.0 / 0.54 |
| dequant_triton+hipblaslt w8 zp | 118.6 / 2.0 / 1.01 | 105.0 / 9.0 / 1.15 | 106.5 / 35.3 / 1.14 | 120.4 / 62.4 / 1.02 | 117.7 / 127.7 / 1.06 | 122.7 / 245.0 / 1.05 | 158.4 / 379.6 / 0.86 | 259.8 / 925.9 / 0.72 | 720.9 / 1334.6 / 0.54 |
| aiter_a16w8_int8 w8 sym | 29.2 / 8.0 / 4.15 | 32.7 / 28.7 / 3.71 | 39.2 / 95.8 / 3.11 | 45.3 / 165.8 / 2.72 | 58.9 / 255.3 / 2.13 | 91.1 / 330.2 / 1.42 | 151.4 / 397.1 / 0.91 | 444.4 / 541.2 / 0.42 | 1701.7 / 565.4 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 23.2 / 10.1 / 5.07 | 26.5 / 35.5 / 4.44 | 32.7 / 115.1 / 3.63 | 44.3 / 169.6 / 2.70 | 54.7 / 274.9 / 2.23 | 81.1 / 370.8 / 1.55 | 148.4 / 405.2 / 0.91 | 361.3 / 665.7 / 0.51 | 1378.3 / 698.0 / 0.28 |
| humming w2 sym | 15.2 / 15.5 / 2.06 | 16.3 / 57.5 / 1.93 | 19.4 / 194.1 / 1.67 | 19.7 / 381.6 / 1.69 | 27.6 / 545.5 / 1.28 | 46.1 / 652.7 / 0.86 | 69.3 / 867.4 / 0.69 | 242.7 / 991.2 / 0.41 | 930.1 / 1034.4 / 0.32 |
| humming w2 zp | 16.0 / 14.7 / 2.01 | 16.9 / 55.5 / 1.91 | 19.8 / 190.1 / 1.68 | 20.2 / 371.7 / 1.69 | 28.0 / 537.1 / 1.30 | 46.2 / 650.3 / 0.88 | 71.2 / 844.8 / 0.69 | 252.3 / 953.4 / 0.39 | 969.5 / 992.4 / 0.31 |
| humming w3 sym | 16.2 / 14.5 / 2.84 | 17.1 / 55.0 / 2.70 | 20.7 / 181.3 / 2.26 | 20.9 / 359.0 / 2.29 | 28.5 / 526.8 / 1.75 | 47.7 / 630.8 / 1.14 | 74.2 / 810.1 / 0.84 | 251.1 / 957.8 / 0.45 | 966.5 / 995.4 / 0.33 |
| humming w3 zp | 17.1 / 13.7 / 2.73 | 18.0 / 52.1 / 2.61 | 21.0 / 178.9 / 2.28 | 21.8 / 345.0 / 2.24 | 30.2 / 498.5 / 1.69 | 48.1 / 624.8 / 1.15 | 75.1 / 800.5 / 0.85 | 253.5 / 948.9 / 0.45 | 969.5 / 992.4 / 0.33 |
| humming w4 sym | 17.1 / 13.7 / 3.54 | 18.1 / 51.9 / 3.36 | 19.8 / 189.6 / 3.11 | 20.3 / 369.7 / 3.08 | 29.8 / 504.5 / 2.17 | 49.1 / 612.2 / 1.40 | 75.3 / 798.8 / 1.03 | 250.7 / 959.5 / 0.51 | 965.0 / 996.9 / 0.34 |
| humming w4 zp | 17.8 / 13.2 / 3.46 | 18.5 / 50.7 / 3.33 | 20.3 / 184.9 / 3.08 | 21.1 / 356.5 / 3.02 | 30.5 / 492.5 / 2.15 | 49.1 / 612.4 / 1.42 | 75.9 / 792.6 / 1.03 | 259.0 / 928.7 / 0.50 | 997.2 / 964.7 / 0.33 |
| humming w8 sym | 27.1 / 8.7 / 4.40 | 27.9 / 33.7 / 4.28 | 28.6 / 131.6 / 4.21 | 29.5 / 255.0 / 4.12 | 35.2 / 427.3 / 3.51 | 54.0 / 556.4 / 2.36 | 80.4 / 748.3 / 1.69 | 267.0 / 900.8 / 0.70 | 1025.5 / 938.2 / 0.38 |
| humming w8 zp | 27.5 / 8.5 / 4.38 | 28.2 / 33.4 / 4.28 | 28.7 / 130.9 / 4.22 | 30.2 / 248.8 / 4.05 | 35.7 / 421.2 / 3.48 | 55.3 / 543.3 / 2.32 | 80.7 / 744.8 / 1.70 | 278.1 / 864.9 / 0.67 | 1059.9 / 907.7 / 0.37 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.3 / 4.8 / 4.84 | 24.5 / 19.2 / 4.80 | 26.3 / 71.5 / 4.49 | 29.6 / 127.0 / 4.01 | 29.4 / 256.1 / 4.08 | 34.1 / 440.8 / 3.58 | 52.4 / 573.3 / 2.42 | 118.4 / 1015.4 / 1.31 | 316.1 / 1521.7 / 0.85 |
| vllm_triton_w4a16 w4 sym | 254.0 / 0.5 / 0.12 | 255.7 / 1.8 / 0.12 | 255.3 / 7.4 / 0.12 | 258.4 / 14.5 / 0.12 | 262.2 / 28.7 / 0.12 | 358.1 / 42.0 / 0.10 | 353.1 / 85.2 / 0.11 | 356.1 / 337.7 / 0.19 | 840.7 / 572.2 / 0.22 |
| vllm_triton_w4a16 w4 zp | 274.2 / 0.4 / 0.11 | 274.0 / 1.7 / 0.11 | 276.0 / 6.8 / 0.11 | 276.8 / 13.6 / 0.11 | 288.5 / 26.1 / 0.11 | 382.3 / 39.3 / 0.09 | 380.1 / 79.1 / 0.10 | 382.3 / 314.6 / 0.18 | 893.0 / 538.7 / 0.20 |
| vllm_exllama_gptq w4 sym | 335.2 / 0.3 / 0.09 | 418.5 / 1.1 / 0.07 | 556.2 / 3.4 / 0.06 | 758.7 / 5.0 / 0.04 | 158.7 / 47.3 / 0.21 | 175.8 / 85.5 / 0.20 | 233.1 / 129.0 / 0.17 | 329.1 / 365.4 / 0.21 | 424.4 / 1133.4 / 0.43 |
| vllm_exllama_gptq w4 zp | 334.7 / 0.4 / 0.09 | 415.3 / 1.1 / 0.07 | 542.9 / 3.5 / 0.06 | 735.5 / 5.1 / 0.04 | 159.0 / 47.3 / 0.21 | 176.2 / 85.3 / 0.20 | 236.0 / 127.4 / 0.17 | 328.5 / 366.1 / 0.21 | 425.1 / 1131.6 / 0.43 |
| vllm_exllama_gptq w8 sym | 340.9 / 0.3 / 0.17 | 391.4 / 1.2 / 0.15 | 556.4 / 3.4 / 0.11 | 161.8 / 23.2 / 0.38 | 172.7 / 43.5 / 0.36 | 190.1 / 79.1 / 0.34 | 249.9 / 120.3 / 0.28 | 341.8 / 351.9 / 0.28 | 445.1 / 1080.8 / 0.47 |
| vllm_exllama_gptq w8 zp | 365.4 / 0.3 / 0.17 | 392.1 / 1.2 / 0.15 | 556.3 / 3.4 / 0.11 | 162.1 / 23.2 / 0.38 | 172.9 / 43.5 / 0.36 | 190.0 / 79.1 / 0.34 | 249.9 / 120.3 / 0.28 | 342.2 / 351.5 / 0.29 | 446.4 / 1077.7 / 0.47 |
| dequant_triton+hipblaslt w4 sym | 54.1 / 2.2 / 0.56 | 54.4 / 8.6 / 0.56 | 55.6 / 33.8 / 0.56 | 57.9 / 64.9 / 0.54 | 58.8 / 127.9 / 0.56 | 65.4 / 229.8 / 0.54 | 85.1 / 353.3 / 0.47 | 142.6 / 843.5 / 0.48 | 338.5 / 1421.3 / 0.54 |
| dequant_triton+hipblaslt w4 zp | 57.6 / 2.0 / 0.53 | 57.2 / 8.2 / 0.54 | 58.7 / 32.0 / 0.53 | 60.3 / 62.3 / 0.53 | 61.8 / 121.7 / 0.53 | 68.2 / 220.4 / 0.52 | 87.2 / 344.8 / 0.46 | 144.7 / 830.9 / 0.47 | 343.2 / 1401.8 / 0.53 |
| dequant_triton+hipblaslt w8 sym | 57.6 / 2.0 / 1.04 | 58.0 / 8.1 / 1.03 | 58.9 / 31.9 / 1.02 | 60.9 / 61.7 / 1.00 | 60.7 / 123.9 / 1.02 | 67.8 / 221.8 / 0.95 | 88.3 / 340.6 / 0.78 | 144.8 / 830.4 / 0.67 | 341.3 / 1409.3 / 0.62 |
| dequant_triton+hipblaslt w8 zp | 57.9 / 2.0 / 1.04 | 59.4 / 7.9 / 1.01 | 59.7 / 31.5 / 1.02 | 62.0 / 60.6 / 0.99 | 61.7 / 121.8 / 1.01 | 68.4 / 219.6 / 0.95 | 88.0 / 341.6 / 0.79 | 146.1 / 822.9 / 0.67 | 344.6 / 1395.8 / 0.61 |
| aiter_a16w8_int8 w8 sym | 19.8 / 5.9 / 3.06 | 20.6 / 22.8 / 2.94 | 20.2 / 92.9 / 3.02 | 27.6 / 136.4 / 2.24 | 35.6 / 211.0 / 1.77 | 66.4 / 226.5 / 0.98 | 112.8 / 266.5 / 0.62 | 224.5 / 535.8 / 0.44 | 840.6 / 572.3 / 0.25 |
| aiter_a16w8_fp8_block128 w8 sym | 16.6 / 7.1 / 3.53 | 17.7 / 26.6 / 3.33 | 18.6 / 101.2 / 3.19 | 27.2 / 138.3 / 2.21 | 30.4 / 247.0 / 2.01 | 56.6 / 265.5 / 1.12 | 121.2 / 248.1 / 0.56 | 175.1 / 686.8 / 0.55 | 655.0 / 734.4 / 0.32 |
| humming w2 sym | 11.2 / 10.5 / 1.39 | 12.9 / 36.5 / 1.22 | 16.3 / 115.2 / 0.99 | 16.5 / 228.0 / 1.02 | 21.2 / 354.3 / 0.85 | 33.0 / 455.6 / 0.62 | 46.6 / 645.8 / 0.54 | 141.4 / 850.7 / 0.38 | 468.7 / 1026.4 / 0.35 |
| humming w2 zp | 12.0 / 9.8 / 1.34 | 13.1 / 35.7 / 1.23 | 16.8 / 111.6 / 0.99 | 17.0 / 220.8 / 1.01 | 21.6 / 348.3 / 0.85 | 33.2 / 452.7 / 0.63 | 47.2 / 636.7 / 0.54 | 147.3 / 816.6 / 0.36 | 478.3 / 1005.7 / 0.35 |
| humming w3 sym | 11.6 / 10.1 / 1.97 | 12.9 / 36.4 / 1.79 | 16.8 / 111.7 / 1.40 | 17.3 / 217.4 / 1.40 | 22.2 / 339.0 / 1.14 | 34.0 / 442.5 / 0.81 | 48.5 / 620.1 / 0.67 | 147.1 / 817.3 / 0.41 | 481.7 / 998.6 / 0.36 |
| humming w3 zp | 12.6 / 9.3 / 1.86 | 13.6 / 34.4 / 1.73 | 17.2 / 109.4 / 1.40 | 17.3 / 217.3 / 1.42 | 22.5 / 334.4 / 1.15 | 34.0 / 442.4 / 0.83 | 48.7 / 617.5 / 0.67 | 148.2 / 811.6 / 0.41 | 483.1 / 995.7 / 0.36 |
| humming w4 sym | 12.0 / 9.8 / 2.53 | 13.1 / 35.7 / 2.31 | 16.5 / 113.8 / 1.87 | 16.8 / 223.9 / 1.87 | 21.3 / 353.2 / 1.53 | 32.7 / 459.2 / 1.07 | 47.7 / 630.7 / 0.83 | 146.0 / 823.6 / 0.47 | 485.8 / 990.1 / 0.37 |
| humming w4 zp | 12.9 / 9.1 / 2.39 | 13.7 / 34.2 / 2.25 | 16.8 / 112.0 / 1.87 | 17.1 / 219.5 / 1.86 | 21.6 / 347.3 / 1.53 | 33.9 / 442.9 / 1.04 | 49.0 / 614.2 / 0.82 | 152.4 / 789.1 / 0.45 | 491.1 / 979.5 / 0.37 |
| humming w8 sym | 14.8 / 8.0 / 4.04 | 15.4 / 30.5 / 3.88 | 18.2 / 103.1 / 3.31 | 18.6 / 202.0 / 3.27 | 23.4 / 321.6 / 2.65 | 37.2 / 403.6 / 1.73 | 53.4 / 563.3 / 1.29 | 161.1 / 746.6 / 0.60 | 500.9 / 960.4 / 0.42 |
| humming w8 zp | 15.1 / 7.8 / 3.98 | 15.7 / 29.8 / 3.83 | 18.4 / 101.9 / 3.29 | 18.8 / 199.7 / 3.26 | 23.5 / 319.8 / 2.66 | 37.7 / 398.6 / 1.72 | 54.7 / 549.8 / 1.27 | 167.3 / 718.8 / 0.58 | 506.9 / 949.1 / 0.42 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 25.9 / 5.2 / 5.19 | 26.2 / 20.5 / 5.14 | 27.2 / 78.9 / 4.95 | 29.0 / 147.9 / 4.66 | 30.5 / 281.5 / 4.47 | 40.8 / 420.7 / 3.39 | 64.2 / 535.3 / 2.22 | 148.1 / 928.1 / 1.13 | 366.8 / 1498.6 / 0.73 |
| vllm_triton_w4a16 w4 sym | 146.6 / 0.9 / 0.24 | 146.6 / 3.7 / 0.24 | 147.7 / 14.5 / 0.24 | 147.8 / 29.1 / 0.24 | 149.5 / 57.5 / 0.25 | 206.6 / 83.2 / 0.19 | 203.8 / 168.6 / 0.21 | 250.6 / 548.4 / 0.27 | 949.8 / 578.8 / 0.18 |
| vllm_triton_w4a16 w4 zp | 157.2 / 0.9 / 0.22 | 157.6 / 3.4 / 0.22 | 158.4 / 13.6 / 0.22 | 159.1 / 27.0 / 0.23 | 164.5 / 52.2 / 0.23 | 219.9 / 78.1 / 0.18 | 218.1 / 157.6 / 0.20 | 268.5 / 511.8 / 0.26 | 1008.2 / 545.3 / 0.17 |
| vllm_exllama_gptq w4 sym | 164.4 / 0.8 / 0.21 | 186.8 / 2.9 / 0.19 | 300.8 / 7.1 / 0.12 | 284.8 / 15.1 / 0.12 | 153.9 / 55.8 / 0.24 | 166.6 / 103.1 / 0.23 | 223.4 / 153.8 / 0.19 | 228.9 / 600.5 / 0.30 | 491.3 / 1119.1 / 0.34 |
| vllm_exllama_gptq w4 zp | 166.1 / 0.8 / 0.21 | 186.8 / 2.9 / 0.19 | 296.2 / 7.2 / 0.12 | 283.7 / 15.1 / 0.13 | 154.0 / 55.8 / 0.24 | 166.8 / 103.0 / 0.23 | 223.2 / 153.9 / 0.19 | 228.6 / 601.1 / 0.30 | 491.8 / 1117.8 / 0.34 |
| vllm_exllama_gptq w8 sym | 156.9 / 0.9 / 0.43 | 183.6 / 2.9 / 0.37 | 301.8 / 7.1 / 0.23 | 115.9 / 37.1 / 0.60 | 167.1 / 51.4 / 0.42 | 179.7 / 95.6 / 0.40 | 236.4 / 145.3 / 0.32 | 242.2 / 567.5 / 0.42 | 511.1 / 1075.7 / 0.40 |
| vllm_exllama_gptq w8 zp | 158.6 / 0.8 / 0.43 | 184.0 / 2.9 / 0.37 | 302.8 / 7.1 / 0.23 | 114.9 / 37.4 / 0.61 | 166.3 / 51.6 / 0.43 | 179.2 / 95.8 / 0.41 | 235.4 / 146.0 / 0.33 | 241.4 / 569.3 / 0.42 | 511.8 / 1074.1 / 0.40 |
| dequant_triton+hipblaslt w4 sym | 62.5 / 2.1 / 0.55 | 61.6 / 8.7 / 0.56 | 64.9 / 33.1 / 0.54 | 61.2 / 70.2 / 0.58 | 70.2 / 122.3 / 0.52 | 75.4 / 227.8 / 0.51 | 107.7 / 319.1 / 0.40 | 174.6 / 787.1 / 0.39 | 396.3 / 1387.2 / 0.43 |
| dequant_triton+hipblaslt w4 zp | 65.1 / 2.1 / 0.54 | 62.5 / 8.6 / 0.56 | 64.8 / 33.2 / 0.55 | 66.5 / 64.6 / 0.54 | 70.2 / 122.4 / 0.53 | 80.9 / 212.4 / 0.48 | 104.2 / 329.6 / 0.41 | 181.7 / 756.4 / 0.38 | 398.8 / 1378.6 / 0.42 |
| dequant_triton+hipblaslt w8 sym | 66.1 / 2.0 / 1.03 | 66.5 / 8.1 / 1.03 | 68.9 / 31.2 / 1.00 | 73.0 / 58.9 / 0.95 | 74.9 / 114.8 / 0.94 | 86.7 / 198.1 / 0.83 | 110.9 / 309.7 / 0.69 | 185.9 / 739.4 / 0.55 | 399.9 / 1374.9 / 0.51 |
| dequant_triton+hipblaslt w8 zp | 63.7 / 2.1 / 1.08 | 63.4 / 8.5 / 1.09 | 65.7 / 32.7 / 1.05 | 67.4 / 63.8 / 1.03 | 71.6 / 120.0 / 0.99 | 81.3 / 211.2 / 0.90 | 105.5 / 325.8 / 0.73 | 180.4 / 762.0 / 0.57 | 397.6 / 1382.6 / 0.51 |
| aiter_a16w8_int8 w8 sym | 18.3 / 7.3 / 3.78 | 18.8 / 28.5 / 3.68 | 22.4 / 95.7 / 3.11 | 23.5 / 182.9 / 2.99 | 30.8 / 278.9 / 2.31 | 46.7 / 367.7 / 1.57 | 78.0 / 440.6 / 0.99 | 251.3 / 546.8 / 0.41 | 947.2 / 580.4 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 15.5 / 8.7 / 4.34 | 16.5 / 32.6 / 4.09 | 17.8 / 120.6 / 3.80 | 22.2 / 193.4 / 3.07 | 27.3 / 314.5 / 2.53 | 43.2 / 397.3 / 1.65 | 75.7 / 453.6 / 1.00 | 199.6 / 688.6 / 0.50 | 751.7 / 731.3 / 0.27 |
| humming w2 sym | 11.2 / 12.0 / 1.59 | 12.6 / 42.5 / 1.42 | 15.8 / 135.7 / 1.16 | 16.2 / 265.5 / 1.17 | 21.2 / 404.8 / 0.94 | 32.4 / 530.9 / 0.68 | 49.6 / 693.3 / 0.53 | 138.9 / 989.1 / 0.37 | 524.4 / 1048.3 / 0.29 |
| humming w2 zp | 12.0 / 11.2 / 1.53 | 13.1 / 41.0 / 1.41 | 16.5 / 130.5 / 1.15 | 16.9 / 253.9 / 1.15 | 21.4 / 400.5 / 0.95 | 32.6 / 526.5 / 0.69 | 50.6 / 679.5 / 0.53 | 143.7 / 956.5 / 0.36 | 540.4 / 1017.3 / 0.28 |
| humming w3 sym | 11.8 / 11.4 / 2.23 | 12.7 / 42.2 / 2.07 | 16.6 / 129.7 / 1.61 | 16.7 / 256.7 / 1.63 | 21.2 / 405.6 / 1.34 | 32.9 / 522.3 / 0.92 | 51.1 / 672.0 / 0.68 | 147.5 / 931.9 / 0.41 | 545.3 / 1008.2 / 0.29 |
| humming w3 zp | 12.9 / 10.4 / 2.08 | 13.7 / 39.2 / 1.96 | 16.8 / 127.5 / 1.62 | 17.2 / 250.1 / 1.62 | 22.2 / 386.2 / 1.30 | 33.6 / 510.8 / 0.92 | 52.2 / 658.2 / 0.67 | 145.3 / 945.9 / 0.41 | 543.9 / 1010.7 / 0.30 |
| humming w4 sym | 12.4 / 10.8 / 2.79 | 13.3 / 40.5 / 2.62 | 16.0 / 134.5 / 2.20 | 16.0 / 267.6 / 2.22 | 20.4 / 422.0 / 1.80 | 33.2 / 518.0 / 1.17 | 51.1 / 672.1 / 0.84 | 145.2 / 946.9 / 0.47 | 541.3 / 1015.6 / 0.31 |
| humming w4 zp | 13.3 / 10.1 / 2.64 | 14.2 / 37.8 / 2.48 | 16.2 / 132.9 / 2.21 | 16.5 / 260.3 / 2.19 | 21.3 / 403.4 / 1.75 | 33.4 / 514.1 / 1.18 | 52.9 / 649.8 / 0.82 | 146.4 / 939.0 / 0.47 | 551.1 / 997.6 / 0.31 |
| humming w8 sym | 17.1 / 7.8 / 3.98 | 17.7 / 30.4 / 3.86 | 18.2 / 118.0 / 3.77 | 18.5 / 232.0 / 3.74 | 22.9 / 374.8 / 3.06 | 36.4 / 471.4 / 1.99 | 56.1 / 612.5 / 1.36 | 148.8 / 923.5 / 0.68 | 563.0 / 976.4 / 0.36 |
| humming w8 zp | 17.3 / 7.7 / 3.96 | 18.0 / 29.8 / 3.82 | 18.3 / 117.6 / 3.79 | 18.7 / 229.9 / 3.73 | 22.4 / 382.9 / 3.15 | 36.9 / 465.2 / 1.97 | 57.9 / 593.1 / 1.33 | 151.3 / 908.6 / 0.68 | 573.2 / 959.1 / 0.35 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 156.5 / 6.0 / 6.00 | 151.7 / 24.8 / 6.20 | 156.1 / 96.3 / 6.03 | 153.4 / 196.0 / 6.15 | 155.4 / 386.9 / 6.10 | 168.1 / 715.4 / 5.69 | 215.4 / 1116.7 / 4.52 | 744.8 / 1291.7 / 1.44 | 2661.3 / 1446.0 / 0.56 |
| vllm_triton_w4a16 w4 sym | 196.9 / 4.8 / 1.23 | 200.8 / 18.7 / 1.21 | 223.7 / 67.2 / 1.09 | 226.9 / 132.5 / 1.09 | 218.5 / 275.2 / 1.15 | 260.7 / 461.3 / 0.99 | 492.6 / 488.2 / 0.56 | 1678.7 / 573.1 / 0.22 | 6486.1 / 593.3 / 0.12 |
| vllm_triton_w4a16 w4 zp | 214.9 / 4.4 / 1.14 | 220.5 / 17.0 / 1.11 | 236.9 / 63.5 / 1.04 | 247.8 / 121.3 / 1.00 | 239.2 / 251.4 / 1.05 | 273.5 / 439.7 / 0.95 | 524.1 / 458.9 / 0.53 | 1788.9 / 537.8 / 0.21 | 6872.4 / 560.0 / 0.11 |
| vllm_exllama_gptq w4 sym | 254.3 / 3.7 / 0.95 | 274.9 / 13.7 / 0.88 | 347.9 / 43.2 / 0.70 | 592.9 / 50.7 / 0.42 | 422.8 / 142.2 / 0.59 | 430.9 / 279.1 / 0.60 | 460.2 / 522.6 / 0.60 | 1052.7 / 913.9 / 0.36 | 3284.3 / 1171.7 / 0.24 |
| vllm_exllama_gptq w4 zp | 252.8 / 3.7 / 0.97 | 274.4 / 13.7 / 0.89 | 348.0 / 43.2 / 0.71 | 594.6 / 50.6 / 0.42 | 426.2 / 141.1 / 0.59 | 435.2 / 276.3 / 0.60 | 467.4 / 514.6 / 0.59 | 1051.7 / 914.8 / 0.36 | 3291.9 / 1169.0 / 0.24 |
| vllm_exllama_gptq w8 sym | 250.4 / 3.8 / 1.91 | 240.1 / 15.7 / 1.99 | 432.0 / 34.8 / 1.11 | 457.6 / 65.7 / 1.05 | 456.2 / 131.8 / 1.06 | 463.2 / 259.6 / 1.07 | 498.5 / 482.5 / 1.02 | 1090.3 / 882.4 / 0.56 | 3343.0 / 1151.1 / 0.30 |
| vllm_exllama_gptq w8 zp | 250.3 / 3.8 / 1.92 | 240.6 / 15.6 / 2.00 | 432.3 / 34.8 / 1.12 | 457.4 / 65.7 / 1.06 | 456.0 / 131.9 / 1.07 | 463.8 / 259.3 / 1.07 | 497.2 / 483.7 / 1.03 | 1090.5 / 882.2 / 0.56 | 3348.7 / 1149.2 / 0.30 |
| dequant_triton+hipblaslt w4 sym | 419.8 / 2.2 / 0.58 | 452.5 / 8.3 / 0.54 | 420.1 / 35.8 / 0.58 | 457.3 / 65.7 / 0.54 | 425.0 / 141.5 / 0.59 | 470.7 / 255.5 / 0.55 | 460.6 / 522.2 / 0.60 | 999.7 / 962.4 / 0.38 | 2881.8 / 1335.4 / 0.27 |
| dequant_triton+hipblaslt w4 zp | 443.4 / 2.1 / 0.55 | 463.2 / 8.1 / 0.53 | 430.9 / 34.9 / 0.57 | 469.4 / 64.0 / 0.53 | 435.2 / 138.2 / 0.58 | 483.8 / 248.6 / 0.54 | 486.9 / 494.0 / 0.57 | 1015.6 / 947.3 / 0.37 | 2902.4 / 1325.9 / 0.27 |
| dequant_triton+hipblaslt w8 sym | 449.5 / 2.1 / 1.06 | 484.0 / 7.8 / 0.99 | 455.5 / 33.0 / 1.05 | 491.2 / 61.2 / 0.98 | 461.4 / 130.3 / 1.05 | 509.1 / 236.2 / 0.97 | 504.5 / 476.8 / 1.01 | 1033.2 / 931.2 / 0.59 | 2936.5 / 1310.5 / 0.34 |
| dequant_triton+hipblaslt w8 zp | 454.1 / 2.1 / 1.06 | 486.6 / 7.7 / 0.99 | 459.6 / 32.7 / 1.05 | 488.6 / 61.5 / 0.99 | 464.4 / 129.5 / 1.05 | 501.7 / 239.7 / 0.99 | 517.6 / 464.7 / 0.99 | 1031.8 / 932.5 / 0.60 | 2948.1 / 1305.4 / 0.34 |
| aiter_a16w8_int8 w8 sym | 94.9 / 9.9 / 5.11 | 101.8 / 36.9 / 4.76 | 123.7 / 121.5 / 3.93 | 138.8 / 216.6 / 3.52 | 174.6 / 344.5 / 2.82 | 282.2 / 426.2 / 1.78 | 553.3 / 434.7 / 0.94 | 1729.8 / 556.2 / 0.36 | 6643.7 / 579.2 / 0.15 |
| aiter_a16w8_fp8_block128 w8 sym | 76.6 / 12.3 / 6.13 | 83.7 / 44.9 / 5.62 | 98.5 / 152.6 / 4.79 | 133.7 / 224.8 / 3.54 | 161.7 / 372.0 / 2.96 | 243.0 / 494.8 / 2.00 | 505.2 / 476.1 / 1.00 | 1371.5 / 701.5 / 0.44 | 5261.4 / 731.4 / 0.19 |
| humming w2 sym | 40.7 / 23.1 / 3.07 | 43.8 / 85.8 / 2.86 | 56.8 / 264.8 / 2.23 | 59.0 / 509.8 / 2.19 | 83.6 / 719.6 / 1.59 | 139.6 / 861.3 / 1.01 | 249.4 / 964.4 / 0.64 | 909.7 / 1057.6 / 0.28 | 3577.4 / 1075.7 / 0.18 |
| humming w2 zp | 43.5 / 21.6 / 2.95 | 45.7 / 82.2 / 2.82 | 59.6 / 252.2 / 2.19 | 61.5 / 489.0 / 2.16 | 83.8 / 717.9 / 1.63 | 136.1 / 883.8 / 1.07 | 258.2 / 931.6 / 0.63 | 938.3 / 1025.3 / 0.28 | 3697.0 / 1040.9 / 0.18 |
| humming w3 sym | 47.1 / 20.0 / 3.90 | 48.6 / 77.2 / 3.78 | 62.7 / 239.6 / 2.96 | 66.0 / 455.6 / 2.84 | 90.6 / 663.3 / 2.12 | 144.5 / 832.3 / 1.39 | 274.2 / 877.2 / 0.79 | 948.9 / 1013.8 / 0.34 | 3741.5 / 1028.5 / 0.19 |
| humming w3 zp | 49.6 / 19.0 / 3.78 | 51.3 / 73.3 / 3.66 | 63.6 / 236.5 / 2.98 | 65.5 / 459.2 / 2.92 | 90.3 / 666.0 / 2.17 | 146.5 / 820.8 / 1.39 | 275.0 / 874.7 / 0.80 | 953.3 / 1009.2 / 0.34 | 3725.2 / 1033.0 / 0.19 |
| humming w4 sym | 50.4 / 18.6 / 4.80 | 51.8 / 72.6 / 4.69 | 63.1 / 238.2 / 3.87 | 67.1 / 448.0 / 3.67 | 92.3 / 651.4 / 2.71 | 151.9 / 791.8 / 1.71 | 277.5 / 866.7 / 0.99 | 962.5 / 999.5 / 0.39 | 3771.3 / 1020.4 / 0.21 |
| humming w4 zp | 52.6 / 17.9 / 4.68 | 54.2 / 69.4 / 4.55 | 65.2 / 230.7 / 3.81 | 68.2 / 440.7 / 3.67 | 92.9 / 647.6 / 2.74 | 152.2 / 790.1 / 1.73 | 287.0 / 838.1 / 0.97 | 988.9 / 972.9 / 0.38 | 3915.8 / 982.7 / 0.20 |
| humming w8 sym | 85.2 / 11.0 / 5.60 | 84.9 / 44.3 / 5.62 | 89.5 / 168.0 / 5.36 | 91.3 / 329.3 / 5.27 | 106.5 / 564.8 / 4.56 | 163.5 / 735.6 / 3.02 | 313.8 / 766.5 / 1.63 | 1139.2 / 844.5 / 0.54 | 4578.6 / 840.5 / 0.22 |
| humming w8 zp | 87.4 / 10.8 / 5.50 | 85.2 / 44.1 / 5.65 | 90.3 / 166.5 / 5.35 | 94.1 / 319.5 / 5.15 | 108.7 / 553.3 / 4.50 | 170.9 / 703.7 / 2.91 | 327.1 / 735.3 / 1.57 | 1173.9 / 819.5 / 0.52 | 4681.7 / 822.0 / 0.22 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.7 / 6.2 / 6.21 | 74.2 / 25.3 / 6.34 | 77.4 / 97.1 / 6.08 | 78.0 / 192.7 / 6.05 | 85.1 / 353.4 / 5.58 | 94.4 / 636.8 / 5.08 | 123.7 / 972.6 / 3.95 | 364.8 / 1318.5 / 1.50 | 1275.6 / 1508.4 / 0.60 |
| vllm_triton_w4a16 w4 sym | 504.0 / 0.9 / 0.24 | 506.5 / 3.7 / 0.24 | 507.1 / 14.8 / 0.24 | 511.2 / 29.4 / 0.24 | 520.1 / 57.8 / 0.24 | 710.9 / 84.6 / 0.18 | 705.1 / 170.6 / 0.20 | 854.8 / 562.8 / 0.23 | 3299.1 / 583.2 / 0.13 |
| vllm_triton_w4a16 w4 zp | 544.9 / 0.9 / 0.22 | 542.5 / 3.5 / 0.23 | 545.9 / 13.8 / 0.23 | 548.7 / 27.4 / 0.23 | 572.5 / 52.5 / 0.22 | 759.5 / 79.2 / 0.17 | 754.0 / 159.5 / 0.19 | 913.7 / 526.4 / 0.22 | 3505.3 / 548.9 / 0.12 |
| vllm_exllama_gptq w4 sym | 1315.3 / 0.4 / 0.09 | 1354.3 / 1.4 / 0.09 | 1128.5 / 6.7 / 0.11 | 745.1 / 20.2 / 0.17 | 503.9 / 59.7 / 0.25 | 552.3 / 108.9 / 0.24 | 705.5 / 170.5 / 0.20 | 710.8 / 676.7 / 0.28 | 1600.6 / 1202.2 / 0.26 |
| vllm_exllama_gptq w4 zp | 1317.3 / 0.4 / 0.09 | 1351.9 / 1.4 / 0.09 | 1112.7 / 6.8 / 0.11 | 749.2 / 20.1 / 0.17 | 503.6 / 59.7 / 0.25 | 552.0 / 108.9 / 0.24 | 705.2 / 170.5 / 0.20 | 711.2 / 676.4 / 0.28 | 1601.7 / 1201.3 / 0.27 |
| vllm_exllama_gptq w8 sym | 1271.3 / 0.4 / 0.19 | 967.3 / 1.9 / 0.25 | 825.1 / 9.1 / 0.29 | 483.3 / 31.1 / 0.50 | 507.0 / 59.3 / 0.48 | 554.5 / 108.4 / 0.45 | 710.2 / 169.3 / 0.36 | 717.4 / 670.6 / 0.44 | 1619.9 / 1187.8 / 0.33 |
| vllm_exllama_gptq w8 zp | 1279.7 / 0.4 / 0.19 | 1029.5 / 1.8 / 0.23 | 826.0 / 9.1 / 0.29 | 483.1 / 31.1 / 0.50 | 508.3 / 59.1 / 0.48 | 555.9 / 108.2 / 0.45 | 711.1 / 169.1 / 0.36 | 715.9 / 671.9 / 0.44 | 1622.5 / 1185.9 / 0.33 |
| dequant_triton+hipblaslt w4 sym | 187.8 / 2.5 / 0.65 | 203.0 / 9.3 / 0.60 | 213.3 / 35.2 / 0.57 | 206.1 / 72.9 / 0.60 | 219.3 / 137.1 / 0.57 | 202.7 / 296.6 / 0.64 | 247.1 / 486.7 / 0.57 | 493.0 / 975.7 / 0.40 | 1385.6 / 1388.6 / 0.30 |
| dequant_triton+hipblaslt w4 zp | 201.8 / 2.3 / 0.60 | 221.4 / 8.5 / 0.55 | 229.8 / 32.7 / 0.54 | 224.8 / 66.9 / 0.55 | 225.2 / 133.5 / 0.56 | 216.6 / 277.6 / 0.61 | 264.5 / 454.7 / 0.53 | 501.0 / 960.1 / 0.39 | 1402.8 / 1371.7 / 0.30 |
| dequant_triton+hipblaslt w8 sym | 213.7 / 2.2 / 1.12 | 226.6 / 8.3 / 1.05 | 229.8 / 32.7 / 1.04 | 230.8 / 65.1 / 1.04 | 232.8 / 129.1 / 1.04 | 238.0 / 252.7 / 1.04 | 261.0 / 460.8 / 0.99 | 492.4 / 976.8 / 0.64 | 1405.5 / 1369.0 / 0.39 |
| dequant_triton+hipblaslt w8 zp | 233.6 / 2.0 / 1.03 | 235.2 / 8.0 / 1.02 | 215.8 / 34.8 / 1.12 | 226.4 / 66.4 / 1.07 | 244.9 / 122.8 / 1.00 | 227.8 / 264.0 / 1.10 | 274.9 / 437.4 / 0.94 | 514.2 / 935.4 / 0.61 | 1415.2 / 1359.6 / 0.38 |
| aiter_a16w8_int8 w8 sym | 48.7 / 9.6 / 4.97 | 50.1 / 37.5 / 4.84 | 57.8 / 130.1 / 4.21 | 66.5 / 226.2 / 3.68 | 82.6 / 363.8 / 2.99 | 130.6 / 460.5 / 1.93 | 261.6 / 459.7 / 1.00 | 868.1 / 554.1 / 0.37 | 3313.9 / 580.6 / 0.16 |
| aiter_a16w8_fp8_block128 w8 sym | 41.9 / 11.2 / 5.61 | 44.4 / 42.3 / 5.30 | 46.2 / 162.6 / 5.11 | 64.9 / 231.6 / 3.65 | 77.2 / 389.6 / 3.11 | 112.2 / 535.9 / 2.18 | 253.7 / 474.0 / 1.00 | 677.9 / 709.5 / 0.46 | 2596.4 / 741.1 / 0.21 |
| humming w2 sym | 28.7 / 16.4 / 2.18 | 31.8 / 59.2 / 1.97 | 37.8 / 198.9 / 1.68 | 39.2 / 383.4 / 1.65 | 52.1 / 577.4 / 1.29 | 80.9 / 743.7 / 0.89 | 145.7 / 825.7 / 0.56 | 467.8 / 1028.3 / 0.29 | 1795.8 / 1071.5 / 0.20 |
| humming w2 zp | 31.4 / 14.9 / 2.04 | 33.6 / 55.9 / 1.92 | 40.6 / 185.3 / 1.61 | 41.1 / 365.6 / 1.62 | 53.3 / 563.9 / 1.29 | 83.9 / 716.6 / 0.88 | 151.5 / 793.6 / 0.55 | 476.2 / 1010.2 / 0.29 | 1835.8 / 1048.1 / 0.20 |
| humming w3 sym | 31.6 / 14.9 / 2.90 | 33.2 / 56.6 / 2.77 | 40.2 / 186.8 / 2.31 | 41.6 / 360.9 / 2.26 | 55.2 / 545.1 / 1.75 | 85.3 / 704.9 / 1.19 | 153.3 / 784.3 / 0.72 | 484.5 / 992.9 / 0.34 | 1854.8 / 1037.4 / 0.21 |
| humming w3 zp | 34.1 / 13.8 / 2.74 | 35.4 / 53.0 / 2.65 | 40.9 / 183.8 / 2.32 | 42.8 / 351.6 / 2.24 | 56.0 / 537.2 / 1.76 | 87.2 / 689.8 / 1.18 | 155.6 / 773.1 / 0.72 | 479.8 / 1002.6 / 0.35 | 1852.0 / 1038.9 / 0.21 |
| humming w4 sym | 34.5 / 13.6 / 3.51 | 34.4 / 54.7 / 3.53 | 39.8 / 188.9 / 3.07 | 42.6 / 353.2 / 2.90 | 55.9 / 537.7 / 2.25 | 87.6 / 686.2 / 1.49 | 154.9 / 776.5 / 0.90 | 489.6 / 982.5 / 0.40 | 1861.4 / 1033.7 / 0.23 |
| humming w4 zp | 34.4 / 13.7 / 3.58 | 36.3 / 51.8 / 3.40 | 40.5 / 185.5 / 3.06 | 42.0 / 358.1 / 2.98 | 57.0 / 527.3 / 2.24 | 90.7 / 662.9 / 1.46 | 159.9 / 752.2 / 0.89 | 487.8 / 986.2 / 0.41 | 1862.2 / 1033.3 / 0.23 |
| humming w8 sym | 45.6 / 10.3 / 5.24 | 45.1 / 41.6 / 5.29 | 47.0 / 159.8 / 5.10 | 49.3 / 305.0 / 4.89 | 62.3 / 482.9 / 3.91 | 101.1 / 594.5 / 2.45 | 174.6 / 688.8 / 1.47 | 500.2 / 961.7 / 0.63 | 1952.8 / 985.3 / 0.28 |
| humming w8 zp | 46.1 / 10.2 / 5.21 | 47.8 / 39.3 / 5.03 | 48.2 / 155.9 / 5.01 | 50.0 / 300.8 / 4.86 | 63.9 / 470.8 / 3.84 | 103.2 / 582.6 / 2.42 | 179.5 / 670.0 / 1.44 | 505.6 / 951.5 / 0.62 | 1976.5 / 973.5 / 0.27 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 15.5 / 3.4 / 3.38 | 15.8 / 13.3 / 3.32 | 15.4 / 54.4 / 3.42 | 16.9 / 99.0 / 3.13 | 23.0 / 146.1 / 2.34 | 24.9 / 269.7 / 2.21 | 35.9 / 374.2 / 1.61 | 64.7 / 829.2 / 1.13 | 184.2 / 1166.0 / 0.74 |
| vllm_triton_w4a16 w4 sym | 92.7 / 0.6 / 0.15 | 93.5 / 2.2 / 0.14 | 93.8 / 8.9 / 0.15 | 94.5 / 17.7 / 0.15 | 95.9 / 35.0 / 0.15 | 130.8 / 51.3 / 0.12 | 129.5 / 103.6 / 0.14 | 150.7 / 356.3 / 0.23 | 404.8 / 530.6 / 0.24 |
| vllm_triton_w4a16 w4 zp | 100.0 / 0.5 / 0.14 | 100.3 / 2.1 / 0.14 | 100.7 / 8.3 / 0.14 | 101.3 / 16.6 / 0.14 | 105.2 / 31.9 / 0.14 | 139.5 / 48.1 / 0.12 | 138.8 / 96.7 / 0.14 | 162.1 / 331.2 / 0.21 | 430.9 / 498.3 / 0.23 |
| vllm_exllama_gptq w4 sym | 51.8 / 1.0 / 0.26 | 73.2 / 2.9 / 0.19 | 118.4 / 7.1 / 0.12 | 170.6 / 9.8 / 0.08 | 80.4 / 41.7 / 0.18 | 77.1 / 87.1 / 0.21 | 71.6 / 187.6 / 0.26 | 106.3 / 505.1 / 0.32 | 225.0 / 954.5 / 0.43 |
| vllm_exllama_gptq w4 zp | 51.7 / 1.0 / 0.26 | 73.3 / 2.9 / 0.19 | 118.3 / 7.1 / 0.12 | 170.4 / 9.8 / 0.08 | 80.4 / 41.7 / 0.19 | 76.4 / 87.8 / 0.21 | 71.5 / 187.6 / 0.26 | 106.4 / 504.6 / 0.33 | 225.5 / 952.2 / 0.43 |
| vllm_exllama_gptq w8 sym | 56.0 / 0.9 / 0.48 | 74.2 / 2.8 / 0.36 | 123.7 / 6.8 / 0.22 | 50.0 / 33.5 / 0.55 | 82.5 / 40.7 / 0.34 | 78.7 / 85.2 / 0.37 | 73.1 / 183.6 / 0.44 | 109.5 / 490.4 / 0.43 | 229.2 / 936.8 / 0.48 |
| vllm_exllama_gptq w8 zp | 55.7 / 0.9 / 0.48 | 74.4 / 2.8 / 0.36 | 123.9 / 6.8 / 0.22 | 49.8 / 33.7 / 0.55 | 82.5 / 40.7 / 0.34 | 78.5 / 85.5 / 0.38 | 72.9 / 184.2 / 0.44 | 109.3 / 491.1 / 0.44 | 227.3 / 944.6 / 0.49 |
| dequant_triton+hipblaslt w4 sym | 30.9 / 1.7 / 0.44 | 31.7 / 6.6 / 0.43 | 30.6 / 27.4 / 0.45 | 32.7 / 51.4 / 0.43 | 37.0 / 90.7 / 0.40 | 38.8 / 173.0 / 0.42 | 49.1 / 273.5 / 0.38 | 72.4 / 741.9 / 0.48 | 194.1 / 1106.6 / 0.50 |
| dequant_triton+hipblaslt w4 zp | 31.7 / 1.7 / 0.43 | 31.9 / 6.6 / 0.43 | 31.4 / 26.7 / 0.44 | 32.6 / 51.5 / 0.44 | 37.7 / 89.0 / 0.40 | 38.6 / 173.8 / 0.42 | 49.8 / 269.6 / 0.38 | 72.3 / 742.1 / 0.48 | 194.6 / 1103.5 / 0.50 |
| dequant_triton+hipblaslt w8 sym | 33.2 / 1.6 / 0.80 | 33.5 / 6.3 / 0.80 | 32.3 / 26.0 / 0.83 | 34.2 / 49.1 / 0.80 | 39.0 / 86.0 / 0.72 | 39.8 / 168.4 / 0.73 | 50.8 / 264.3 / 0.63 | 71.9 / 746.5 / 0.66 | 194.9 / 1101.8 / 0.57 |
| dequant_triton+hipblaslt w8 zp | 32.1 / 1.6 / 0.83 | 32.5 / 6.5 / 0.83 | 31.6 / 26.6 / 0.86 | 33.2 / 50.5 / 0.83 | 38.5 / 87.1 / 0.73 | 40.1 / 167.3 / 0.73 | 50.6 / 265.5 / 0.63 | 72.0 / 745.3 / 0.66 | 195.3 / 1099.5 / 0.57 |
| aiter_a16w8_int8 w8 sym | 12.2 / 4.3 / 2.22 | 12.9 / 16.3 / 2.11 | 13.7 / 61.2 / 2.00 | 16.7 / 100.7 / 1.66 | 20.6 / 162.9 / 1.38 | 32.8 / 204.6 / 0.90 | 47.5 / 282.6 / 0.68 | 163.6 / 328.2 / 0.29 | 391.5 / 548.5 / 0.28 |
| aiter_a16w8_fp8_block128 w8 sym | 10.4 / 5.0 / 2.51 | 11.0 / 19.0 / 2.38 | 12.0 / 70.2 / 2.22 | 15.5 / 108.0 / 1.73 | 19.5 / 172.0 / 1.41 | 29.3 / 229.4 / 0.99 | 47.3 / 283.6 / 0.67 | 122.6 / 437.8 / 0.39 | 305.8 / 702.3 / 0.36 |
| humming w2 sym | 9.1 / 5.8 / 0.77 | 9.9 / 21.1 / 0.71 | 11.4 / 73.5 / 0.64 | 11.7 / 143.6 / 0.65 | 15.1 / 222.2 / 0.55 | 25.3 / 265.0 / 0.38 | 32.6 / 411.4 / 0.37 | 74.1 / 724.6 / 0.38 | 219.5 / 978.3 / 0.41 |
| humming w2 zp | 9.4 / 5.6 / 0.76 | 10.4 / 20.2 / 0.70 | 11.6 / 72.0 / 0.64 | 11.8 / 142.0 / 0.66 | 15.5 / 216.5 / 0.55 | 25.5 / 263.6 / 0.39 | 33.0 / 406.2 / 0.38 | 76.3 / 703.6 / 0.37 | 229.1 / 937.3 / 0.40 |
| humming w3 sym | 9.4 / 5.6 / 1.09 | 10.0 / 20.9 / 1.03 | 11.7 / 71.4 / 0.90 | 12.1 / 138.6 / 0.90 | 15.8 / 212.7 / 0.73 | 26.7 / 251.5 / 0.48 | 33.7 / 398.6 / 0.46 | 76.8 / 698.9 / 0.41 | 229.3 / 936.5 / 0.41 |
| humming w3 zp | 9.7 / 5.4 / 1.07 | 10.5 / 20.0 / 1.00 | 11.8 / 70.9 / 0.91 | 12.4 / 135.4 / 0.90 | 16.0 / 209.6 / 0.73 | 26.3 / 255.0 / 0.50 | 33.6 / 399.2 / 0.47 | 76.8 / 699.2 / 0.41 | 224.8 / 955.4 / 0.42 |
| humming w4 sym | 9.5 / 5.5 / 1.43 | 10.1 / 20.8 / 1.35 | 11.7 / 71.8 / 1.19 | 12.3 / 136.3 / 1.15 | 16.1 / 208.0 / 0.92 | 26.0 / 257.9 / 0.62 | 32.9 / 408.2 / 0.57 | 77.6 / 691.6 / 0.44 | 226.1 / 949.6 / 0.43 |
| humming w4 zp | 9.9 / 5.3 / 1.38 | 10.7 / 19.7 / 1.29 | 11.8 / 70.9 / 1.19 | 12.1 / 138.6 / 1.19 | 16.5 / 203.8 / 0.91 | 26.7 / 251.2 / 0.61 | 33.5 / 401.2 / 0.57 | 79.9 / 672.0 / 0.43 | 231.5 / 927.6 / 0.42 |
| humming w8 sym | 11.0 / 4.8 / 2.42 | 11.5 / 18.2 / 2.32 | 13.0 / 64.4 / 2.07 | 14.0 / 120.2 / 1.95 | 17.9 / 187.4 / 1.56 | 28.1 / 238.8 / 1.04 | 35.6 / 376.9 / 0.90 | 79.5 / 675.3 / 0.60 | 232.8 / 922.4 / 0.47 |
| humming w8 zp | 11.2 / 4.7 / 2.39 | 11.8 / 17.8 / 2.28 | 13.3 / 63.3 / 2.05 | 14.0 / 120.0 / 1.97 | 17.7 / 189.8 / 1.59 | 28.0 / 239.8 / 1.05 | 36.2 / 370.9 / 0.89 | 81.6 / 657.8 / 0.59 | 239.8 / 895.5 / 0.46 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 79.3 / 6.6 / 6.61 | 81.6 / 25.7 / 6.43 | 82.8 / 101.3 / 6.35 | 85.3 / 196.6 / 6.19 | 85.5 / 392.6 / 6.22 | 105.0 / 638.8 / 5.13 | 131.6 / 1019.9 / 4.20 | 513.0 / 1046.5 / 1.25 | 1549.2 / 1386.2 / 0.64 |
| vllm_triton_w4a16 w4 sym | 120.2 / 4.4 / 1.12 | 122.7 / 17.1 / 1.10 | 131.7 / 63.7 / 1.04 | 134.6 / 124.6 / 1.03 | 135.7 / 247.4 / 1.05 | 172.3 / 389.6 / 0.87 | 307.2 / 436.9 / 0.53 | 980.4 / 547.6 / 0.26 | 3669.8 / 585.2 / 0.16 |
| vllm_triton_w4a16 w4 zp | 132.7 / 4.0 / 1.03 | 134.5 / 15.6 / 1.02 | 143.2 / 58.6 / 0.96 | 146.4 / 114.6 / 0.95 | 149.4 / 224.6 / 0.96 | 172.6 / 388.8 / 0.87 | 326.9 / 410.5 / 0.51 | 1037.3 / 517.6 / 0.24 | 3882.5 / 553.1 / 0.15 |
| vllm_exllama_gptq w4 sym | 134.5 / 3.9 / 1.01 | 166.8 / 12.6 / 0.81 | 227.7 / 36.8 / 0.60 | 351.8 / 47.7 / 0.39 | 243.7 / 137.7 / 0.58 | 250.6 / 267.8 / 0.60 | 272.7 / 492.3 / 0.60 | 654.6 / 820.1 / 0.38 | 1918.8 / 1119.2 / 0.31 |
| vllm_exllama_gptq w4 zp | 134.3 / 3.9 / 1.01 | 167.0 / 12.6 / 0.82 | 228.0 / 36.8 / 0.60 | 351.7 / 47.7 / 0.40 | 244.0 / 137.5 / 0.59 | 250.8 / 267.6 / 0.60 | 272.7 / 492.2 / 0.60 | 649.3 / 826.9 / 0.39 | 1922.6 / 1117.0 / 0.31 |
| vllm_exllama_gptq w8 sym | 147.1 / 3.6 / 1.81 | 147.2 / 14.2 / 1.81 | 262.6 / 31.9 / 1.02 | 261.1 / 64.3 / 1.03 | 262.7 / 127.7 / 1.04 | 271.6 / 247.1 / 1.03 | 291.2 / 460.9 / 1.01 | 683.8 / 785.1 / 0.56 | 1953.6 / 1099.2 / 0.37 |
| vllm_exllama_gptq w8 zp | 145.6 / 3.6 / 1.84 | 148.1 / 14.2 / 1.81 | 262.9 / 31.9 / 1.03 | 261.6 / 64.1 / 1.04 | 263.2 / 127.5 / 1.05 | 270.8 / 247.8 / 1.04 | 292.1 / 459.5 / 1.02 | 681.6 / 787.7 / 0.56 | 1955.1 / 1098.4 / 0.37 |
| dequant_triton+hipblaslt w4 sym | 215.2 / 2.4 / 0.63 | 216.6 / 9.7 / 0.63 | 215.3 / 39.0 / 0.64 | 218.2 / 76.9 / 0.64 | 219.5 / 152.9 / 0.65 | 235.3 / 285.2 / 0.64 | 257.1 / 522.0 / 0.64 | 643.5 / 834.4 / 0.39 | 1671.0 / 1285.2 / 0.36 |
| dequant_triton+hipblaslt w4 zp | 219.8 / 2.4 / 0.62 | 225.3 / 9.3 / 0.61 | 220.1 / 38.1 / 0.63 | 229.6 / 73.1 / 0.61 | 223.2 / 150.3 / 0.64 | 247.2 / 271.5 / 0.61 | 262.4 / 511.5 / 0.63 | 649.9 / 826.1 / 0.39 | 1685.9 / 1273.8 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 226.0 / 2.3 / 1.18 | 229.9 / 9.1 / 1.16 | 230.7 / 36.4 / 1.16 | 233.4 / 71.9 / 1.16 | 237.7 / 141.2 / 1.15 | 247.7 / 271.0 / 1.13 | 273.3 / 491.1 / 1.08 | 655.8 / 818.7 / 0.58 | 1699.3 / 1263.8 / 0.43 |
| dequant_triton+hipblaslt w8 zp | 236.7 / 2.2 / 1.13 | 236.6 / 8.9 / 1.14 | 234.8 / 35.7 / 1.15 | 236.5 / 70.9 / 1.15 | 239.5 / 140.1 / 1.15 | 257.7 / 260.4 / 1.10 | 278.1 / 482.7 / 1.07 | 667.7 / 804.1 / 0.57 | 1709.1 / 1256.5 / 0.43 |
| aiter_a16w8_int8 w8 sym | 59.7 / 8.8 / 4.53 | 66.0 / 31.8 / 4.10 | 79.6 / 105.4 / 3.42 | 92.6 / 181.2 / 2.96 | 117.0 / 286.7 / 2.37 | 190.5 / 352.3 / 1.50 | 320.2 / 419.2 / 0.93 | 1055.5 / 508.7 / 0.36 | 3881.1 / 553.3 / 0.19 |
| aiter_a16w8_fp8_block128 w8 sym | 48.5 / 10.8 / 5.41 | 54.7 / 38.3 / 4.80 | 66.3 / 126.5 / 3.98 | 85.8 / 195.6 / 3.10 | 109.7 / 305.8 / 2.46 | 162.8 / 412.2 / 1.70 | 319.1 / 420.6 / 0.91 | 814.3 / 659.3 / 0.46 | 3021.3 / 710.8 / 0.24 |
| humming w2 sym | 27.1 / 19.3 / 2.57 | 29.0 / 72.3 / 2.42 | 36.1 / 232.5 / 1.98 | 37.6 / 446.6 / 1.95 | 50.3 / 667.1 / 1.53 | 83.5 / 803.9 / 1.01 | 154.6 / 868.2 / 0.64 | 534.9 / 1003.6 / 0.35 | 2050.8 / 1047.2 / 0.26 |
| humming w2 zp | 29.2 / 18.0 / 2.46 | 30.9 / 67.9 / 2.33 | 37.8 / 222.2 / 1.95 | 38.7 / 433.7 / 1.95 | 53.1 / 632.4 / 1.49 | 86.8 / 773.5 / 0.99 | 160.6 / 835.8 / 0.63 | 554.8 / 967.6 / 0.34 | 2136.5 / 1005.2 / 0.25 |
| humming w3 sym | 29.9 / 17.5 / 3.43 | 31.3 / 66.9 / 3.28 | 38.9 / 215.7 / 2.68 | 40.9 / 410.3 / 2.59 | 54.4 / 616.3 / 2.01 | 88.8 / 755.9 / 1.32 | 162.4 / 826.6 / 0.81 | 557.5 / 963.0 / 0.39 | 2131.8 / 1007.4 / 0.26 |
| humming w3 zp | 31.3 / 16.8 / 3.34 | 32.6 / 64.3 / 3.21 | 39.6 / 211.6 / 2.68 | 41.3 / 406.5 / 2.62 | 56.0 / 599.0 / 1.99 | 88.8 / 756.0 / 1.34 | 161.7 / 829.8 / 0.82 | 554.3 / 968.5 / 0.40 | 2128.8 / 1008.8 / 0.27 |
| humming w4 sym | 31.4 / 16.7 / 4.31 | 33.2 / 63.2 / 4.09 | 40.5 / 207.3 / 3.38 | 42.1 / 398.8 / 3.30 | 58.1 / 577.7 / 2.45 | 94.3 / 711.6 / 1.59 | 164.5 / 815.8 / 1.00 | 558.2 / 961.7 / 0.45 | 2139.4 / 1003.8 / 0.28 |
| humming w4 zp | 33.3 / 15.8 / 4.13 | 34.8 / 60.2 / 3.95 | 39.4 / 212.9 / 3.53 | 42.0 / 399.1 / 3.35 | 57.5 / 583.1 / 2.51 | 92.5 / 725.3 / 1.64 | 170.5 / 787.4 / 0.97 | 574.3 / 934.8 / 0.44 | 2209.2 / 972.1 / 0.27 |
| humming w8 sym | 50.2 / 10.4 / 5.30 | 53.0 / 39.5 / 5.03 | 53.0 / 158.4 / 5.06 | 56.3 / 297.9 / 4.79 | 67.0 / 500.7 / 4.08 | 104.3 / 643.2 / 2.69 | 184.9 / 725.9 / 1.60 | 622.2 / 862.8 / 0.61 | 2370.3 / 906.0 / 0.31 |
| humming w8 zp | 51.1 / 10.3 / 5.25 | 53.2 / 39.4 / 5.05 | 53.6 / 156.4 / 5.04 | 57.0 / 294.5 / 4.77 | 68.3 / 491.2 / 4.03 | 109.8 / 611.0 / 2.57 | 186.2 / 720.7 / 1.59 | 650.7 / 825.0 / 0.59 | 2471.5 / 868.9 / 0.29 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 45.7 / 5.7 / 5.74 | 46.3 / 22.7 / 5.67 | 47.7 / 88.0 / 5.52 | 50.6 / 165.9 / 5.22 | 56.3 / 298.0 / 4.73 | 59.3 / 566.3 / 4.56 | 92.4 / 726.3 / 3.01 | 235.9 / 1138.1 / 1.38 | 857.1 / 1252.7 / 0.60 |
| vllm_triton_w4a16 w4 sym | 453.2 / 0.6 / 0.15 | 454.3 / 2.3 / 0.15 | 455.8 / 9.2 / 0.15 | 459.0 / 18.3 / 0.15 | 465.0 / 36.1 / 0.15 | 633.5 / 53.0 / 0.12 | 627.3 / 107.0 / 0.13 | 728.3 / 368.6 / 0.18 | 1971.4 / 544.7 / 0.16 |
| vllm_triton_w4a16 w4 zp | 487.7 / 0.5 / 0.14 | 488.2 / 2.1 / 0.14 | 489.6 / 8.6 / 0.14 | 492.2 / 17.0 / 0.14 | 513.7 / 32.7 / 0.14 | 679.4 / 49.4 / 0.11 | 674.5 / 99.5 / 0.12 | 782.5 / 343.1 / 0.17 | 2098.1 / 511.8 / 0.15 |
| vllm_exllama_gptq w4 sym | 500.3 / 0.5 / 0.14 | 547.3 / 1.9 / 0.12 | 648.1 / 6.5 / 0.11 | 695.0 / 12.1 / 0.10 | 381.9 / 43.9 / 0.19 | 458.3 / 73.2 / 0.17 | 590.5 / 113.6 / 0.14 | 539.0 / 498.0 / 0.24 | 1146.3 / 936.7 / 0.28 |
| vllm_exllama_gptq w4 zp | 501.1 / 0.5 / 0.14 | 547.2 / 1.9 / 0.12 | 646.8 / 6.5 / 0.11 | 695.2 / 12.1 / 0.10 | 383.4 / 43.8 / 0.19 | 461.1 / 72.8 / 0.17 | 590.0 / 113.7 / 0.14 | 540.4 / 496.8 / 0.24 | 1157.0 / 928.1 / 0.28 |
| vllm_exllama_gptq w8 sym | 507.5 / 0.5 / 0.26 | 536.3 / 2.0 / 0.25 | 500.9 / 8.4 / 0.27 | 373.8 / 22.4 / 0.36 | 408.0 / 41.1 / 0.34 | 498.1 / 67.4 / 0.28 | 624.3 / 107.5 / 0.24 | 559.3 / 479.9 / 0.35 | 1166.7 / 920.3 / 0.33 |
| vllm_exllama_gptq w8 zp | 509.3 / 0.5 / 0.26 | 534.6 / 2.0 / 0.25 | 499.6 / 8.4 / 0.27 | 372.8 / 22.5 / 0.36 | 403.9 / 41.5 / 0.34 | 496.8 / 67.5 / 0.29 | 624.3 / 107.5 / 0.24 | 558.9 / 480.3 / 0.35 | 1172.4 / 915.8 / 0.33 |
| dequant_triton+hipblaslt w4 sym | 112.2 / 2.3 / 0.60 | 115.5 / 9.1 / 0.59 | 114.1 / 36.8 / 0.60 | 117.9 / 71.2 / 0.59 | 122.3 / 137.2 / 0.58 | 128.9 / 260.3 / 0.58 | 166.4 / 403.4 / 0.50 | 299.7 / 895.7 / 0.43 | 924.0 / 1162.0 / 0.34 |
| dequant_triton+hipblaslt w4 zp | 119.9 / 2.2 / 0.57 | 124.0 / 8.5 / 0.55 | 122.1 / 34.4 / 0.57 | 127.3 / 65.9 / 0.55 | 128.6 / 130.5 / 0.56 | 138.8 / 241.7 / 0.55 | 173.9 / 385.9 / 0.48 | 308.5 / 870.2 / 0.42 | 933.6 / 1150.1 / 0.34 |
| dequant_triton+hipblaslt w8 sym | 116.1 / 2.3 / 1.15 | 116.5 / 9.0 / 1.15 | 117.9 / 35.6 / 1.14 | 120.9 / 69.4 / 1.12 | 126.3 / 132.8 / 1.08 | 130.9 / 256.3 / 1.08 | 168.1 / 399.2 / 0.89 | 304.6 / 881.2 / 0.64 | 929.3 / 1155.4 / 0.41 |
| dequant_triton+hipblaslt w8 zp | 120.1 / 2.2 / 1.12 | 120.9 / 8.7 / 1.11 | 122.2 / 34.3 / 1.11 | 123.7 / 67.8 / 1.10 | 129.7 / 129.3 / 1.06 | 135.8 / 247.2 / 1.05 | 173.0 / 387.9 / 0.87 | 308.6 / 870.0 / 0.64 | 936.4 / 1146.6 / 0.41 |
| aiter_a16w8_int8 w8 sym | 37.6 / 7.0 / 3.60 | 38.6 / 27.2 / 3.51 | 39.7 / 105.7 / 3.43 | 54.2 / 154.6 / 2.53 | 64.9 / 258.5 / 2.14 | 111.3 / 301.5 / 1.28 | 215.0 / 312.2 / 0.70 | 769.0 / 349.1 / 0.26 | 1856.6 / 578.3 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 29.7 / 8.8 / 4.41 | 30.4 / 34.4 / 4.32 | 32.5 / 129.0 / 4.06 | 53.2 / 157.7 / 2.50 | 61.5 / 272.6 / 2.19 | 94.0 / 356.9 / 1.48 | 216.4 / 310.1 / 0.68 | 573.2 / 468.3 / 0.34 | 1445.7 / 742.7 / 0.27 |
| humming w2 sym | 25.6 / 10.2 / 1.36 | 29.0 / 36.1 / 1.21 | 27.2 / 153.9 / 1.31 | 27.9 / 300.4 / 1.32 | 35.7 / 469.5 / 1.08 | 60.4 / 555.1 / 0.71 | 99.3 / 675.7 / 0.51 | 335.2 / 800.8 / 0.29 | 1028.5 / 1044.0 / 0.28 |
| humming w2 zp | 27.0 / 9.7 / 1.33 | 30.4 / 34.5 / 1.19 | 28.6 / 146.4 / 1.28 | 28.7 / 292.3 / 1.32 | 37.2 / 451.2 / 1.07 | 61.4 / 546.7 / 0.71 | 100.6 / 667.3 / 0.51 | 340.2 / 789.1 / 0.29 | 1056.4 / 1016.4 / 0.27 |
| humming w3 sym | 26.8 / 9.8 / 1.91 | 29.7 / 35.3 / 1.73 | 28.9 / 145.0 / 1.80 | 29.7 / 282.9 / 1.79 | 38.7 / 433.7 / 1.43 | 63.4 / 529.5 / 0.93 | 103.8 / 646.8 / 0.65 | 347.4 / 772.8 / 0.33 | 1067.6 / 1005.7 / 0.28 |
| humming w3 zp | 28.3 / 9.3 / 1.85 | 30.8 / 34.0 / 1.70 | 28.6 / 146.5 / 1.86 | 29.2 / 287.3 / 1.86 | 39.3 / 426.8 / 1.43 | 63.4 / 529.7 / 0.95 | 103.4 / 649.2 / 0.66 | 345.8 / 776.2 / 0.33 | 1063.6 / 1009.6 / 0.29 |
| humming w4 sym | 27.8 / 9.4 / 2.43 | 30.1 / 34.9 / 2.26 | 27.4 / 153.1 / 2.50 | 28.7 / 292.7 / 2.43 | 38.6 / 434.3 / 1.85 | 63.7 / 526.8 / 1.19 | 104.9 / 639.9 / 0.79 | 355.3 / 755.6 / 0.37 | 1066.3 / 1007.0 / 0.30 |
| humming w4 zp | 29.3 / 8.9 / 2.34 | 31.5 / 33.3 / 2.18 | 27.6 / 151.9 / 2.52 | 28.6 / 293.7 / 2.47 | 39.9 / 420.0 / 1.82 | 64.3 / 522.0 / 1.19 | 104.5 / 642.2 / 0.81 | 355.9 / 754.3 / 0.37 | 1075.3 / 998.6 / 0.30 |
| humming w8 sym | 36.6 / 7.2 / 3.64 | 37.6 / 27.9 / 3.55 | 34.6 / 121.2 / 3.88 | 37.0 / 226.5 / 3.65 | 46.9 / 357.5 / 2.92 | 73.4 / 457.4 / 1.92 | 112.2 / 598.4 / 1.33 | 362.4 / 740.8 / 0.54 | 1103.6 / 972.9 / 0.35 |
| humming w8 zp | 36.8 / 7.1 / 3.65 | 38.5 / 27.3 / 3.49 | 35.2 / 119.2 / 3.84 | 37.8 / 222.2 / 3.60 | 47.3 / 354.7 / 2.92 | 75.3 / 445.6 / 1.89 | 114.6 / 585.6 / 1.31 | 363.8 / 737.9 / 0.54 | 1115.1 / 962.9 / 0.35 |

## Full results (bf16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 / 2.7 / 2.72 | 9.6 / 14.0 / 3.51 | 10.3 / 51.9 / 3.27 | 12.7 / 84.8 / 2.69 | 18.4 / 116.9 / 1.88 | 19.3 / 222.4 / 1.85 | 20.2 / 425.9 / 1.87 | 47.0 / 731.6 / 1.07 | 94.3 / 1458.0 / 1.07 |
| vllm_triton_w4a16 w4 sym | 84.8 / 0.4 / 0.10 | 84.5 / 1.6 / 0.10 | 85.1 / 6.3 / 0.10 | 85.3 / 12.6 / 0.11 | 84.7 / 25.3 / 0.11 | 135.4 / 31.7 / 0.08 | 134.8 / 63.7 / 0.10 | 135.7 / 253.3 / 0.19 | 307.1 / 447.5 / 0.25 |
| vllm_triton_w4a16 w4 zp | 88.8 / 0.4 / 0.10 | 89.2 / 1.5 / 0.10 | 89.4 / 6.0 / 0.10 | 89.6 / 12.0 / 0.10 | 88.4 / 24.3 / 0.11 | 125.8 / 34.1 / 0.09 | 125.2 / 68.6 / 0.10 | 126.4 / 271.8 / 0.20 | 339.6 / 404.7 / 0.22 |
| dequant_triton+hipblaslt w4 sym | 24.9 / 1.3 / 0.35 | 21.5 / 6.2 / 0.41 | 22.9 / 23.4 / 0.39 | 24.4 / 43.9 / 0.38 | 31.5 / 68.2 / 0.31 | 30.2 / 142.1 / 0.36 | 32.3 / 265.6 / 0.40 | 58.0 / 592.2 / 0.44 | 106.7 / 1288.3 / 0.71 |
| dequant_triton+hipblaslt w4 zp | 25.0 / 1.3 / 0.35 | 21.0 / 6.4 / 0.42 | 23.4 / 23.0 / 0.38 | 24.5 / 43.8 / 0.38 | 31.2 / 68.8 / 0.31 | 30.2 / 142.1 / 0.36 | 32.5 / 264.5 / 0.40 | 57.7 / 595.9 / 0.44 | 106.8 / 1286.5 / 0.71 |
| dequant_triton+hipblaslt w8 sym | 24.9 / 1.3 / 0.69 | 22.3 / 6.0 / 0.77 | 23.3 / 23.1 / 0.74 | 24.8 / 43.2 / 0.71 | 32.1 / 67.0 / 0.56 | 31.2 / 137.6 / 0.61 | 34.1 / 252.2 / 0.62 | 58.0 / 592.7 / 0.58 | 106.9 / 1285.9 / 0.79 |
| dequant_triton+hipblaslt w8 zp | 25.6 / 1.3 / 0.67 | 23.3 / 5.8 / 0.74 | 23.5 / 22.8 / 0.74 | 25.0 / 43.0 / 0.71 | 30.5 / 70.4 / 0.60 | 31.0 / 138.4 / 0.62 | 34.9 / 246.3 / 0.61 | 58.6 / 586.8 / 0.58 | 108.6 / 1265.7 / 0.78 |
| aiter_a16w8_int8 w8 sym | 9.6 / 3.5 / 1.80 | 10.0 / 13.4 / 1.73 | 12.7 / 42.4 / 1.39 | 12.6 / 85.4 / 1.42 | 15.6 / 137.8 / 1.18 | 26.8 / 160.1 / 0.72 | 34.8 / 246.8 / 0.62 | 64.1 / 535.9 / 0.53 | 231.4 / 593.8 / 0.36 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.15 | 9.0 / 15.0 / 1.88 | 9.5 / 56.6 / 1.80 | 11.8 / 91.0 / 1.47 | 13.8 / 155.4 / 1.29 | 23.9 / 179.7 / 0.79 | 37.2 / 230.9 / 0.56 | 54.7 / 628.2 / 0.61 | 197.3 / 696.7 / 0.42 |
| humming w2 sym | 7.8 / 4.3 / 0.58 | 8.1 / 16.6 / 0.56 | 11.1 / 48.5 / 0.43 | 11.3 / 94.9 / 0.44 | 13.9 / 154.6 / 0.40 | 21.7 / 197.7 / 0.30 | 24.4 / 352.2 / 0.35 | 50.0 / 687.7 / 0.42 | 159.5 / 861.8 / 0.45 |
| humming w2 zp | 7.9 / 4.3 / 0.58 | 8.2 / 16.4 / 0.57 | 11.3 / 47.7 / 0.43 | 11.4 / 93.8 / 0.45 | 14.1 / 152.6 / 0.40 | 21.7 / 197.9 / 0.31 | 24.5 / 350.8 / 0.36 | 50.9 / 675.7 / 0.42 | 159.9 / 859.3 / 0.45 |
| humming w3 sym | 7.9 / 4.3 / 0.83 | 8.2 / 16.3 / 0.80 | 11.3 / 47.4 / 0.60 | 11.6 / 92.6 / 0.61 | 14.2 / 151.8 / 0.54 | 22.2 / 193.7 / 0.39 | 24.3 / 352.9 / 0.44 | 51.1 / 672.5 / 0.46 | 166.3 / 826.4 / 0.44 |
| humming w3 zp | 8.1 / 4.1 / 0.82 | 8.5 / 15.9 / 0.80 | 11.4 / 46.9 / 0.61 | 11.7 / 91.6 / 0.61 | 14.4 / 149.6 / 0.54 | 22.1 / 194.4 / 0.40 | 25.2 / 341.5 / 0.43 | 51.4 / 668.3 / 0.46 | 161.5 / 850.9 / 0.46 |
| humming w4 sym | 7.8 / 4.3 / 1.12 | 8.2 / 16.3 / 1.06 | 11.2 / 48.0 / 0.80 | 11.4 / 93.9 / 0.80 | 14.5 / 148.3 / 0.67 | 22.1 / 194.5 / 0.49 | 24.7 / 347.1 / 0.52 | 51.4 / 668.7 / 0.49 | 166.1 / 827.3 / 0.46 |
| humming w4 zp | 8.1 / 4.1 / 1.08 | 8.4 / 16.0 / 1.05 | 11.4 / 47.1 / 0.79 | 11.6 / 92.7 / 0.80 | 14.9 / 143.9 / 0.66 | 22.3 / 192.5 / 0.49 | 24.6 / 349.2 / 0.53 | 52.7 / 651.6 / 0.48 | 167.4 / 821.0 / 0.45 |
| humming w8 sym | 8.5 / 3.9 / 2.00 | 9.1 / 14.8 / 1.89 | 11.8 / 45.5 / 1.47 | 12.0 / 89.1 / 1.46 | 14.9 / 144.5 / 1.22 | 23.3 / 184.0 / 0.82 | 26.1 / 329.3 / 0.81 | 55.9 / 615.2 / 0.60 | 174.2 / 788.8 / 0.48 |
| humming w8 zp | 8.7 / 3.9 / 1.98 | 9.1 / 14.7 / 1.89 | 11.9 / 45.0 / 1.46 | 12.1 / 88.8 / 1.46 | 14.9 / 144.1 / 1.22 | 23.1 / 185.6 / 0.83 | 26.5 / 323.5 / 0.81 | 56.8 / 604.6 / 0.60 | 174.1 / 789.4 / 0.48 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 14.0 / 3.6 / 3.61 | 14.2 / 14.1 / 3.54 | 13.8 / 58.4 / 3.67 | 16.7 / 96.3 / 3.05 | 19.0 / 169.2 / 2.71 | 20.5 / 313.6 / 2.58 | 26.9 / 478.3 / 2.06 | 58.9 / 874.8 / 1.21 | 159.3 / 1294.1 / 0.84 |
| vllm_triton_w4a16 w4 sym | 85.3 / 0.6 / 0.15 | 85.3 / 2.4 / 0.15 | 85.3 / 9.4 / 0.16 | 85.6 / 18.8 / 0.16 | 84.7 / 38.0 / 0.17 | 135.6 / 47.5 / 0.12 | 134.8 / 95.6 / 0.14 | 158.6 / 324.9 / 0.21 | 454.8 / 453.3 / 0.21 |
| vllm_triton_w4a16 w4 zp | 89.3 / 0.6 / 0.15 | 89.8 / 2.2 / 0.15 | 89.8 / 9.0 / 0.15 | 90.2 / 17.9 / 0.15 | 88.8 / 36.3 / 0.16 | 125.7 / 51.2 / 0.12 | 125.3 / 102.8 / 0.15 | 174.4 / 295.6 / 0.20 | 499.7 / 412.6 / 0.19 |
| dequant_triton+hipblaslt w4 sym | 28.6 / 1.8 / 0.46 | 29.2 / 6.9 / 0.45 | 28.1 / 28.7 / 0.47 | 31.7 / 50.8 / 0.43 | 33.9 / 95.1 / 0.42 | 35.5 / 181.6 / 0.44 | 41.2 / 313.1 / 0.44 | 69.3 / 743.5 / 0.49 | 170.5 / 1209.1 / 0.57 |
| dequant_triton+hipblaslt w4 zp | 29.4 / 1.7 / 0.45 | 29.3 / 6.9 / 0.45 | 28.6 / 28.2 / 0.47 | 34.0 / 47.4 / 0.40 | 33.7 / 95.5 / 0.43 | 35.2 / 182.8 / 0.45 | 43.4 / 296.7 / 0.42 | 69.6 / 740.5 / 0.49 | 170.9 / 1206.4 / 0.57 |
| dequant_triton+hipblaslt w8 sym | 31.1 / 1.6 / 0.82 | 31.1 / 6.5 / 0.82 | 30.0 / 26.8 / 0.86 | 32.5 / 49.6 / 0.81 | 36.1 / 89.2 / 0.74 | 37.4 / 172.2 / 0.75 | 43.1 / 299.0 / 0.71 | 68.3 / 754.7 / 0.68 | 172.4 / 1195.7 / 0.64 |
| dequant_triton+hipblaslt w8 zp | 31.0 / 1.6 / 0.83 | 31.1 / 6.5 / 0.83 | 29.8 / 27.0 / 0.88 | 32.3 / 49.8 / 0.82 | 36.1 / 89.3 / 0.75 | 37.0 / 174.1 / 0.77 | 42.6 / 302.8 / 0.73 | 68.9 / 747.7 / 0.68 | 173.0 / 1191.4 / 0.63 |
| aiter_a16w8_int8 w8 sym | 11.3 / 4.5 / 2.30 | 11.5 / 17.5 / 2.26 | 14.7 / 54.7 / 1.78 | 14.9 / 108.1 / 1.79 | 19.2 / 167.8 / 1.42 | 29.9 / 215.1 / 0.95 | 38.6 / 333.5 / 0.81 | 117.1 / 440.3 / 0.40 | 345.2 / 597.2 / 0.32 |
| aiter_a16w8_fp8_block128 w8 sym | 8.9 / 5.6 / 2.82 | 9.8 / 20.6 / 2.58 | 11.0 / 72.9 / 2.31 | 13.9 / 116.0 / 1.86 | 17.3 / 186.2 / 1.53 | 26.7 / 241.5 / 1.04 | 38.8 / 331.9 / 0.78 | 100.4 / 513.5 / 0.46 | 294.5 / 700.0 / 0.37 |
| humming w2 sym | 9.1 / 5.5 / 0.73 | 9.9 / 20.3 / 0.68 | 12.0 / 66.9 / 0.58 | 12.2 / 132.2 / 0.60 | 15.5 / 208.2 / 0.52 | 25.1 / 256.3 / 0.37 | 30.8 / 418.8 / 0.39 | 74.3 / 694.1 / 0.37 | 232.3 / 887.6 / 0.39 |
| humming w2 zp | 9.5 / 5.3 / 0.73 | 10.2 / 19.8 / 0.69 | 12.2 / 66.2 / 0.59 | 12.4 / 130.0 / 0.61 | 15.8 / 203.9 / 0.52 | 25.7 / 251.0 / 0.37 | 31.6 / 407.9 / 0.38 | 75.2 / 685.1 / 0.37 | 234.3 / 880.1 / 0.39 |
| humming w3 sym | 9.4 / 5.4 / 1.05 | 10.0 / 20.2 / 0.99 | 12.3 / 65.7 / 0.83 | 12.5 / 128.4 / 0.84 | 15.8 / 203.7 / 0.70 | 25.5 / 252.8 / 0.49 | 30.7 / 419.6 / 0.49 | 75.4 / 683.2 / 0.41 | 244.6 / 842.7 / 0.38 |
| humming w3 zp | 9.8 / 5.2 / 1.03 | 10.3 / 19.5 / 0.98 | 12.5 / 64.6 / 0.83 | 12.8 / 126.0 / 0.84 | 16.3 / 198.2 / 0.70 | 25.5 / 252.7 / 0.50 | 31.1 / 414.4 / 0.49 | 75.8 / 679.7 / 0.41 | 240.9 / 855.7 / 0.39 |
| humming w4 sym | 9.3 / 5.4 / 1.39 | 10.1 / 20.0 / 1.29 | 12.2 / 66.1 / 1.09 | 12.4 / 129.5 / 1.10 | 16.0 / 201.0 / 0.89 | 25.2 / 255.6 / 0.62 | 31.0 / 415.6 / 0.59 | 75.8 / 679.6 / 0.45 | 241.9 / 852.3 / 0.40 |
| humming w4 zp | 9.7 / 5.2 / 1.36 | 10.4 / 19.4 / 1.28 | 12.4 / 65.0 / 1.09 | 12.6 / 128.3 / 1.10 | 16.8 / 192.0 / 0.86 | 25.7 / 250.4 / 0.61 | 31.5 / 408.7 / 0.58 | 77.0 / 669.4 / 0.44 | 245.3 / 840.4 / 0.40 |
| humming w8 sym | 11.1 / 4.5 / 2.31 | 11.9 / 16.9 / 2.15 | 13.3 / 60.4 / 1.94 | 13.7 / 117.7 / 1.92 | 17.9 / 180.2 / 1.50 | 27.2 / 236.9 / 1.04 | 33.8 / 381.1 / 0.91 | 79.0 / 652.1 / 0.59 | 248.6 / 829.4 / 0.44 |
| humming w8 zp | 11.3 / 4.5 / 2.29 | 12.1 / 16.7 / 2.14 | 13.6 / 59.3 / 1.92 | 14.0 / 115.2 / 1.89 | 17.6 / 182.6 / 1.53 | 27.4 / 235.0 / 1.03 | 33.7 / 382.8 / 0.92 | 80.4 / 640.8 / 0.58 | 250.9 / 821.8 / 0.44 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 47.9 / 4.9 / 4.90 | 48.4 / 19.4 / 4.86 | 45.2 / 83.1 / 5.22 | 48.4 / 155.3 / 4.90 | 48.3 / 311.5 / 4.96 | 56.0 / 536.4 / 4.34 | 67.1 / 896.4 / 3.75 | 191.0 / 1259.5 / 1.58 | 632.1 / 1521.9 / 0.80 |
| vllm_triton_w4a16 w4 sym | 95.2 / 2.5 / 0.64 | 95.3 / 9.9 / 0.64 | 95.4 / 39.4 / 0.65 | 95.7 / 78.5 / 0.66 | 95.3 / 157.7 / 0.68 | 137.0 / 219.5 / 0.50 | 162.7 / 369.7 / 0.47 | 557.4 / 431.5 / 0.23 | 2064.6 / 466.0 / 0.16 |
| vllm_triton_w4a16 w4 zp | 101.5 / 2.3 / 0.60 | 101.5 / 9.3 / 0.60 | 102.4 / 36.7 / 0.61 | 102.3 / 73.5 / 0.62 | 100.2 / 150.0 / 0.65 | 126.6 / 237.5 / 0.55 | 175.4 / 342.8 / 0.44 | 580.7 / 414.2 / 0.22 | 2247.2 / 428.1 / 0.15 |
| dequant_triton+hipblaslt w4 sym | 96.9 / 2.4 / 0.63 | 91.1 / 10.3 / 0.67 | 96.0 / 39.2 / 0.64 | 111.7 / 67.3 / 0.56 | 107.1 / 140.4 / 0.60 | 113.3 / 265.3 / 0.61 | 150.5 / 399.5 / 0.51 | 241.5 / 996.1 / 0.53 | 674.0 / 1427.4 / 0.49 |
| dequant_triton+hipblaslt w4 zp | 98.3 / 2.4 / 0.62 | 91.4 / 10.3 / 0.67 | 97.1 / 38.7 / 0.64 | 100.3 / 74.9 / 0.63 | 107.2 / 140.3 / 0.61 | 109.2 / 275.3 / 0.64 | 150.5 / 399.5 / 0.52 | 241.2 / 997.4 / 0.53 | 676.6 / 1421.9 / 0.49 |
| dequant_triton+hipblaslt w8 sym | 118.2 / 2.0 / 1.01 | 102.4 / 9.2 / 1.17 | 106.7 / 35.2 / 1.13 | 107.1 / 70.2 / 1.13 | 117.8 / 127.6 / 1.05 | 114.1 / 263.4 / 1.12 | 155.7 / 386.1 / 0.87 | 248.4 / 968.1 / 0.75 | 680.9 / 1413.0 / 0.57 |
| dequant_triton+hipblaslt w8 zp | 119.2 / 2.0 / 1.01 | 103.3 / 9.1 / 1.17 | 106.0 / 35.5 / 1.14 | 114.2 / 65.8 / 1.07 | 118.1 / 127.3 / 1.05 | 120.8 / 248.9 / 1.06 | 158.5 / 379.4 / 0.86 | 251.1 / 957.7 / 0.75 | 683.0 / 1408.6 / 0.57 |
| aiter_a16w8_int8 w8 sym | 31.6 / 7.4 / 3.83 | 33.9 / 27.7 / 3.58 | 44.9 / 83.7 / 2.72 | 46.2 / 162.9 / 2.67 | 59.1 / 254.5 / 2.12 | 93.1 / 323.0 / 1.39 | 148.4 / 405.1 / 0.93 | 417.8 / 575.7 / 0.45 | 1591.7 / 604.4 / 0.24 |
| aiter_a16w8_fp8_block128 w8 sym | 23.2 / 10.1 / 5.07 | 26.3 / 35.7 / 4.48 | 32.3 / 116.4 / 3.67 | 44.4 / 169.3 / 2.69 | 54.2 / 277.3 / 2.24 | 80.6 / 373.0 / 1.56 | 148.0 / 406.4 / 0.91 | 357.8 / 672.2 / 0.52 | 1360.2 / 707.3 / 0.28 |
| humming w2 sym | 18.9 / 12.4 / 1.65 | 19.4 / 48.5 / 1.62 | 24.5 / 153.5 / 1.32 | 24.8 / 303.0 / 1.34 | 32.0 / 469.8 / 1.11 | 49.6 / 606.4 / 0.80 | 78.7 / 763.7 / 0.61 | 268.2 / 896.7 / 0.37 | 1036.4 / 928.3 / 0.29 |
| humming w2 zp | 19.2 / 12.3 / 1.68 | 19.7 / 47.7 / 1.64 | 24.4 / 154.0 / 1.36 | 24.7 / 304.4 / 1.39 | 31.7 / 474.8 / 1.15 | 50.7 / 593.0 / 0.80 | 79.9 / 752.9 / 0.61 | 271.6 / 885.7 / 0.36 | 1051.4 / 915.1 / 0.29 |
| humming w3 sym | 19.9 / 11.8 / 2.30 | 19.8 / 47.4 / 2.33 | 25.4 / 147.8 / 1.84 | 26.0 / 289.6 / 1.85 | 32.8 / 458.0 / 1.52 | 52.0 / 578.6 / 1.04 | 81.0 / 742.3 / 0.77 | 278.3 / 864.2 / 0.41 | 1078.4 / 892.1 / 0.29 |
| humming w3 zp | 20.9 / 11.2 / 2.24 | 20.9 / 44.9 / 2.25 | 25.4 / 147.9 / 1.88 | 25.9 / 289.7 / 1.88 | 33.6 / 447.3 / 1.52 | 52.5 / 572.4 / 1.05 | 81.0 / 742.7 / 0.79 | 276.9 / 868.5 / 0.41 | 1082.3 / 888.9 / 0.29 |
| humming w4 sym | 19.7 / 11.9 / 3.07 | 20.0 / 47.0 / 3.04 | 24.7 / 152.1 / 2.49 | 24.9 / 301.4 / 2.51 | 34.1 / 441.2 / 1.90 | 52.3 / 575.3 / 1.32 | 81.7 / 736.1 / 0.95 | 277.8 / 865.9 / 0.46 | 1081.8 / 889.3 / 0.30 |
| humming w4 zp | 20.7 / 11.4 / 2.98 | 20.9 / 45.0 / 2.95 | 24.5 / 153.2 / 2.55 | 25.5 / 294.3 / 2.49 | 35.2 / 427.5 / 1.87 | 53.1 / 566.4 / 1.32 | 83.5 / 720.3 / 0.94 | 285.9 / 841.4 / 0.45 | 1098.2 / 876.0 / 0.30 |
| humming w8 sym | 27.9 / 8.4 / 4.28 | 28.3 / 33.2 / 4.23 | 30.4 / 123.8 / 3.96 | 31.6 / 238.2 / 3.85 | 38.5 / 390.7 / 3.21 | 59.2 / 507.8 / 2.16 | 89.6 / 671.4 / 1.52 | 302.0 / 796.4 / 0.62 | 1163.7 / 826.7 / 0.33 |
| humming w8 zp | 28.4 / 8.3 / 4.23 | 28.7 / 32.7 / 4.20 | 30.6 / 122.8 / 3.96 | 31.2 / 240.7 / 3.92 | 38.6 / 389.9 / 3.23 | 59.3 / 507.0 / 2.17 | 89.9 / 668.9 / 1.52 | 305.1 / 788.3 / 0.61 | 1167.5 / 824.0 / 0.33 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.4 / 4.8 / 4.82 | 24.7 / 19.0 / 4.75 | 26.4 / 71.3 / 4.48 | 29.7 / 126.5 / 3.99 | 29.4 / 255.2 / 4.07 | 34.1 / 440.9 / 3.58 | 51.2 / 587.1 / 2.48 | 114.1 / 1054.0 / 1.36 | 302.6 / 1589.6 / 0.89 |
| vllm_triton_w4a16 w4 sym | 290.4 / 0.4 / 0.10 | 289.7 / 1.6 / 0.10 | 289.3 / 6.5 / 0.11 | 289.6 / 13.0 / 0.11 | 288.6 / 26.0 / 0.11 | 464.5 / 32.4 / 0.07 | 462.4 / 65.0 / 0.09 | 465.2 / 258.5 / 0.15 | 1047.9 / 459.0 / 0.17 |
| vllm_triton_w4a16 w4 zp | 302.3 / 0.4 / 0.10 | 302.8 / 1.6 / 0.10 | 303.0 / 6.2 / 0.10 | 304.1 / 12.4 / 0.10 | 300.8 / 25.0 / 0.11 | 429.6 / 35.0 / 0.08 | 428.6 / 70.1 / 0.09 | 430.9 / 279.1 / 0.16 | 1168.9 / 411.5 / 0.15 |
| dequant_triton+hipblaslt w4 sym | 54.3 / 2.2 / 0.56 | 54.2 / 8.7 / 0.56 | 55.8 / 33.7 / 0.55 | 57.9 / 64.9 / 0.54 | 58.8 / 127.9 / 0.56 | 65.0 / 231.2 / 0.54 | 84.4 / 356.3 / 0.47 | 139.8 / 860.3 / 0.49 | 325.1 / 1479.7 / 0.56 |
| dequant_triton+hipblaslt w4 zp | 57.6 / 2.0 / 0.53 | 57.9 / 8.1 / 0.53 | 59.0 / 31.9 / 0.53 | 61.2 / 61.4 / 0.52 | 61.8 / 121.6 / 0.53 | 68.6 / 219.0 / 0.51 | 86.7 / 346.9 / 0.46 | 141.1 / 852.1 / 0.48 | 328.2 / 1465.5 / 0.55 |
| dequant_triton+hipblaslt w8 sym | 58.0 / 2.0 / 1.03 | 58.6 / 8.0 / 1.02 | 59.5 / 31.6 / 1.01 | 61.1 / 61.5 / 0.99 | 62.0 / 121.3 / 1.00 | 67.6 / 222.2 / 0.95 | 87.6 / 343.1 / 0.79 | 142.1 / 846.0 / 0.69 | 325.7 / 1476.9 / 0.65 |
| dequant_triton+hipblaslt w8 zp | 58.4 / 2.0 / 1.03 | 58.8 / 8.0 / 1.02 | 59.4 / 31.6 / 1.02 | 61.4 / 61.2 / 1.00 | 61.8 / 121.6 / 1.01 | 68.2 / 220.5 / 0.95 | 87.9 / 342.1 / 0.79 | 143.9 / 835.7 / 0.68 | 328.5 / 1464.4 / 0.64 |
| aiter_a16w8_int8 w8 sym | 19.6 / 6.0 / 3.09 | 20.2 / 23.3 / 3.01 | 28.5 / 65.9 / 2.15 | 28.1 / 133.9 / 2.20 | 35.7 / 210.3 / 1.76 | 69.4 / 216.6 / 0.94 | 112.8 / 266.6 / 0.62 | 204.4 / 588.3 / 0.48 | 769.3 / 625.3 / 0.28 |
| aiter_a16w8_fp8_block128 w8 sym | 16.7 / 7.0 / 3.52 | 17.6 / 26.7 / 3.34 | 18.5 / 101.8 / 3.21 | 27.2 / 138.1 / 2.20 | 30.6 / 245.5 / 2.00 | 56.6 / 265.7 / 1.12 | 120.8 / 248.9 / 0.56 | 173.4 / 693.5 / 0.56 | 649.2 / 740.9 / 0.32 |
| humming w2 sym | 13.0 / 9.1 / 1.21 | 13.5 / 34.9 / 1.17 | 18.4 / 102.2 / 0.88 | 18.7 / 201.0 / 0.90 | 22.7 / 331.6 / 0.79 | 35.8 / 420.1 / 0.57 | 50.1 / 599.7 / 0.50 | 157.9 / 761.4 / 0.34 | 525.6 / 915.3 / 0.32 |
| humming w2 zp | 13.5 / 8.7 / 1.19 | 14.1 / 33.2 / 1.15 | 19.0 / 98.8 / 0.88 | 19.2 / 195.3 / 0.90 | 23.7 / 317.8 / 0.78 | 36.0 / 417.0 / 0.58 | 51.7 / 582.1 / 0.49 | 161.6 / 744.4 / 0.33 | 533.9 / 901.0 / 0.31 |
| humming w3 sym | 13.3 / 8.8 / 1.73 | 13.9 / 33.8 / 1.66 | 18.7 / 100.3 / 1.26 | 19.5 / 192.3 / 1.23 | 23.2 / 323.8 / 1.09 | 36.2 / 414.7 / 0.76 | 52.2 / 576.3 / 0.62 | 161.3 / 745.5 / 0.38 | 538.3 / 893.7 / 0.32 |
| humming w3 zp | 14.4 / 8.2 / 1.63 | 15.0 / 31.3 / 1.57 | 19.3 / 97.4 / 1.24 | 19.8 / 189.8 / 1.24 | 24.0 / 313.6 / 1.07 | 36.0 / 417.4 / 0.78 | 52.0 / 577.9 / 0.63 | 160.0 / 751.7 / 0.38 | 536.9 / 896.0 / 0.33 |
| humming w4 sym | 12.9 / 9.1 / 2.35 | 13.5 / 34.9 / 2.26 | 18.6 / 100.9 / 1.66 | 18.6 / 201.9 / 1.69 | 24.0 / 313.1 / 1.36 | 35.9 / 418.3 / 0.97 | 52.5 / 572.4 / 0.76 | 161.0 / 747.0 / 0.42 | 544.2 / 884.0 / 0.33 |
| humming w4 zp | 14.4 / 8.1 / 2.13 | 14.9 / 31.5 / 2.07 | 18.9 / 99.4 / 1.66 | 19.4 / 193.3 / 1.64 | 24.9 / 301.9 / 1.33 | 36.2 / 414.9 / 0.98 | 53.4 / 562.9 / 0.75 | 163.8 / 734.0 / 0.42 | 545.4 / 882.0 / 0.33 |
| humming w8 sym | 14.6 / 8.0 / 4.08 | 15.5 / 30.3 / 3.86 | 19.8 / 95.0 / 3.04 | 20.2 / 186.4 / 3.02 | 25.2 / 298.3 / 2.46 | 39.2 / 383.3 / 1.64 | 57.7 / 521.5 / 1.20 | 176.7 / 680.8 / 0.55 | 563.0 / 854.4 / 0.37 |
| humming w8 zp | 15.1 / 7.8 / 3.97 | 15.7 / 30.0 / 3.84 | 19.7 / 95.2 / 3.07 | 20.0 / 188.1 / 3.07 | 25.4 / 295.4 / 2.45 | 39.7 / 379.0 / 1.63 | 58.1 / 517.1 / 1.20 | 179.2 / 671.2 / 0.55 | 576.3 / 834.6 / 0.37 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 26.1 / 5.1 / 5.15 | 26.4 / 20.3 / 5.08 | 27.5 / 78.1 / 4.90 | 29.3 / 146.7 / 4.62 | 30.5 / 281.6 / 4.47 | 40.3 / 426.2 / 3.43 | 56.4 / 609.1 / 2.53 | 142.2 / 966.8 / 1.18 | 350.3 / 1569.2 / 0.77 |
| vllm_triton_w4a16 w4 sym | 166.9 / 0.8 / 0.21 | 166.3 / 3.2 / 0.21 | 166.4 / 12.9 / 0.21 | 167.3 / 25.7 / 0.21 | 164.5 / 52.2 / 0.22 | 265.7 / 64.7 / 0.15 | 264.2 / 130.0 / 0.16 | 311.8 / 440.8 / 0.22 | 1188.8 / 462.5 / 0.14 |
| vllm_triton_w4a16 w4 zp | 174.7 / 0.8 / 0.20 | 176.0 / 3.0 / 0.20 | 175.6 / 12.2 / 0.20 | 175.1 / 24.5 / 0.20 | 173.9 / 49.4 / 0.21 | 247.7 / 69.4 / 0.16 | 246.3 / 139.5 / 0.18 | 345.5 / 397.8 / 0.20 | 1304.4 / 421.5 / 0.13 |
| dequant_triton+hipblaslt w4 sym | 58.8 / 2.3 / 0.59 | 58.9 / 9.1 / 0.59 | 63.9 / 33.6 / 0.55 | 66.0 / 65.1 / 0.54 | 69.3 / 124.0 / 0.53 | 80.0 / 214.9 / 0.48 | 97.0 / 354.3 / 0.44 | 170.5 / 806.0 / 0.40 | 377.3 / 1457.1 / 0.45 |
| dequant_triton+hipblaslt w4 zp | 65.4 / 2.1 / 0.53 | 60.7 / 8.8 / 0.58 | 64.9 / 33.1 / 0.55 | 64.3 / 66.8 / 0.56 | 70.1 / 122.5 / 0.53 | 83.5 / 205.7 / 0.47 | 91.9 / 374.1 / 0.47 | 172.3 / 797.7 / 0.40 | 376.8 / 1459.0 / 0.45 |
| dequant_triton+hipblaslt w8 sym | 65.4 / 2.1 / 1.04 | 66.1 / 8.1 / 1.03 | 68.5 / 31.3 / 1.00 | 73.5 / 58.4 / 0.94 | 74.3 / 115.7 / 0.95 | 87.9 / 195.4 / 0.82 | 104.5 / 328.9 / 0.73 | 177.5 / 774.4 / 0.57 | 382.2 / 1438.5 / 0.53 |
| dequant_triton+hipblaslt w8 zp | 68.1 / 2.0 / 1.01 | 63.8 / 8.4 / 1.08 | 70.9 / 30.3 / 0.98 | 73.8 / 58.2 / 0.94 | 76.7 / 112.1 / 0.92 | 88.3 / 194.5 / 0.82 | 100.9 / 340.6 / 0.76 | 181.3 / 758.0 / 0.56 | 384.5 / 1429.7 / 0.53 |
| aiter_a16w8_int8 w8 sym | 19.0 / 7.1 / 3.64 | 19.4 / 27.7 / 3.58 | 25.7 / 83.6 / 2.72 | 24.0 / 178.7 / 2.92 | 31.1 / 275.9 / 2.29 | 48.1 / 357.2 / 1.53 | 78.3 / 438.7 / 0.99 | 232.2 / 591.8 / 0.44 | 879.6 / 625.0 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 15.5 / 8.7 / 4.33 | 16.4 / 32.7 / 4.10 | 17.9 / 119.9 / 3.78 | 22.3 / 192.2 / 3.05 | 27.3 / 315.1 / 2.54 | 42.7 / 402.6 / 1.67 | 75.7 / 454.0 / 1.00 | 198.3 / 693.0 / 0.51 | 746.7 / 736.3 / 0.27 |
| humming w2 sym | 13.0 / 10.4 / 1.38 | 13.3 / 40.4 / 1.35 | 18.3 / 117.2 / 1.00 | 18.7 / 230.1 / 1.01 | 22.6 / 379.4 / 0.88 | 35.0 / 491.5 / 0.63 | 54.1 / 635.4 / 0.48 | 154.9 / 887.1 / 0.33 | 587.9 / 935.2 / 0.26 |
| humming w2 zp | 13.9 / 9.7 / 1.33 | 14.2 / 37.8 / 1.30 | 19.0 / 112.8 / 0.99 | 19.5 / 220.3 / 0.99 | 23.8 / 361.5 / 0.86 | 35.4 / 485.6 / 0.64 | 55.7 / 616.6 / 0.48 | 157.3 / 873.6 / 0.33 | 596.9 / 921.0 / 0.26 |
| humming w3 sym | 13.5 / 9.9 / 1.94 | 13.7 / 39.2 / 1.93 | 19.1 / 112.5 / 1.40 | 18.9 / 227.4 / 1.44 | 23.1 / 372.0 / 1.23 | 35.1 / 489.6 / 0.87 | 55.8 / 615.7 / 0.62 | 157.9 / 870.2 / 0.38 | 602.0 / 913.3 / 0.27 |
| humming w3 zp | 14.9 / 9.0 / 1.80 | 15.3 / 35.2 / 1.76 | 19.3 / 111.5 / 1.42 | 19.9 / 216.2 / 1.40 | 23.9 / 359.3 / 1.21 | 35.5 / 483.4 / 0.87 | 56.2 / 611.5 / 0.62 | 160.2 / 858.0 / 0.38 | 602.1 / 913.1 / 0.27 |
| humming w4 sym | 13.1 / 10.2 / 2.63 | 13.6 / 39.4 / 2.55 | 18.3 / 117.2 / 1.92 | 18.6 / 231.4 / 1.92 | 24.6 / 348.6 / 1.49 | 37.5 / 458.6 / 1.04 | 59.4 / 578.6 / 0.72 | 161.3 / 852.0 / 0.42 | 604.2 / 909.9 / 0.28 |
| humming w4 zp | 15.0 / 9.0 / 2.35 | 15.7 / 34.3 / 2.25 | 19.0 / 113.1 / 1.88 | 19.0 / 226.1 / 1.91 | 24.8 / 345.9 / 1.50 | 36.2 / 474.9 / 1.09 | 57.4 / 598.4 / 0.76 | 162.2 / 847.4 / 0.42 | 608.5 / 903.4 / 0.28 |
| humming w8 sym | 17.4 / 7.7 / 3.92 | 18.1 / 29.6 / 3.77 | 20.5 / 104.9 / 3.35 | 21.1 / 203.8 / 3.28 | 25.9 / 331.3 / 2.71 | 38.8 / 443.1 / 1.87 | 60.7 / 565.7 / 1.26 | 167.2 / 821.9 / 0.61 | 637.6 / 862.2 / 0.32 |
| humming w8 zp | 17.4 / 7.7 / 3.94 | 18.1 / 29.7 / 3.81 | 19.7 / 108.8 / 3.51 | 20.2 / 212.2 / 3.44 | 25.2 / 340.8 / 2.81 | 39.1 / 439.1 / 1.86 | 62.0 / 554.2 / 1.24 | 168.1 / 817.4 / 0.61 | 643.7 / 854.1 / 0.32 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 151.5 / 6.2 / 6.20 | 164.1 / 22.9 / 5.73 | 156.4 / 96.1 / 6.02 | 165.5 / 181.7 / 5.70 | 155.8 / 386.0 / 6.08 | 174.9 / 687.6 / 5.47 | 207.1 / 1161.1 / 4.70 | 723.3 / 1330.2 / 1.49 | 2544.0 / 1512.7 / 0.58 |
| vllm_triton_w4a16 w4 sym | 283.6 / 3.3 / 0.85 | 283.0 / 13.3 / 0.86 | 297.3 / 50.6 / 0.82 | 297.4 / 101.1 / 0.83 | 280.5 / 214.3 / 0.89 | 320.4 / 375.3 / 0.81 | 622.0 / 386.7 / 0.44 | 2092.8 / 459.7 / 0.18 | 8104.0 / 474.9 / 0.10 |
| vllm_triton_w4a16 w4 zp | 307.8 / 3.1 / 0.79 | 306.9 / 12.2 / 0.80 | 317.0 / 47.4 / 0.78 | 316.8 / 94.9 / 0.78 | 296.6 / 202.7 / 0.85 | 345.7 / 347.8 / 0.75 | 669.0 / 359.5 / 0.41 | 2267.8 / 424.2 / 0.17 | 8887.8 / 433.0 / 0.09 |
| dequant_triton+hipblaslt w4 sym | 418.0 / 2.2 / 0.58 | 430.4 / 8.7 / 0.56 | 417.8 / 36.0 / 0.58 | 434.4 / 69.2 / 0.57 | 423.1 / 142.1 / 0.59 | 448.0 / 268.4 / 0.58 | 471.0 / 510.6 / 0.58 | 968.6 / 993.3 / 0.39 | 2767.7 / 1390.4 / 0.28 |
| dequant_triton+hipblaslt w4 zp | 432.1 / 2.2 / 0.56 | 444.9 / 8.4 / 0.55 | 431.0 / 34.9 / 0.57 | 448.2 / 67.1 / 0.55 | 437.4 / 137.5 / 0.58 | 463.4 / 259.5 / 0.56 | 484.9 / 496.0 / 0.57 | 980.3 / 981.4 / 0.39 | 2795.6 / 1376.5 / 0.28 |
| dequant_triton+hipblaslt w8 sym | 467.0 / 2.0 / 1.02 | 462.7 / 8.1 / 1.03 | 471.5 / 31.9 / 1.02 | 489.9 / 61.4 / 0.98 | 473.3 / 127.1 / 1.03 | 502.4 / 239.4 / 0.98 | 496.1 / 484.8 / 1.03 | 1020.2 / 943.0 / 0.60 | 2809.4 / 1369.8 / 0.36 |
| dequant_triton+hipblaslt w8 zp | 466.6 / 2.0 / 1.03 | 472.7 / 8.0 / 1.02 | 467.6 / 32.1 / 1.03 | 477.4 / 63.0 / 1.02 | 476.9 / 126.1 / 1.03 | 488.3 / 246.3 / 1.02 | 496.4 / 484.5 / 1.04 | 1009.6 / 953.0 / 0.61 | 2810.1 / 1369.5 / 0.36 |
| aiter_a16w8_int8 w8 sym | 99.8 / 9.4 / 4.86 | 111.8 / 33.6 / 4.34 | 133.4 / 112.7 / 3.65 | 140.8 / 213.5 / 3.47 | 174.9 / 343.7 / 2.82 | 290.7 / 413.6 / 1.72 | 521.9 / 460.9 / 0.99 | 1597.1 / 602.4 / 0.39 | 6166.1 / 624.1 / 0.17 |
| aiter_a16w8_fp8_block128 w8 sym | 76.5 / 12.3 / 6.14 | 84.2 / 44.7 / 5.59 | 101.0 / 148.9 / 4.67 | 133.2 / 225.7 / 3.56 | 160.9 / 373.6 / 2.97 | 241.6 / 497.7 / 2.01 | 505.8 / 475.5 / 0.99 | 1358.5 / 708.2 / 0.45 | 5227.7 / 736.1 / 0.19 |
| humming w2 sym | 55.8 / 16.8 / 2.24 | 55.8 / 67.3 / 2.24 | 75.9 / 198.1 / 1.67 | 77.5 / 387.9 / 1.66 | 97.2 / 618.6 / 1.37 | 150.5 / 799.3 / 0.94 | 286.3 / 840.0 / 0.55 | 1017.9 / 945.1 / 0.25 | 4098.7 / 938.9 / 0.16 |
| humming w2 zp | 57.7 / 16.3 / 2.23 | 58.1 / 64.6 / 2.22 | 77.1 / 194.9 / 1.69 | 79.3 / 379.1 / 1.67 | 99.5 / 604.3 / 1.38 | 154.2 / 779.8 / 0.94 | 291.6 / 824.9 / 0.56 | 1032.6 / 931.7 / 0.25 | 4128.0 / 932.2 / 0.16 |
| humming w3 sym | 60.2 / 15.6 / 3.05 | 60.5 / 62.1 / 3.04 | 81.2 / 185.2 / 2.29 | 83.5 / 359.9 / 2.25 | 102.4 / 587.1 / 1.87 | 156.3 / 769.5 / 1.28 | 296.1 / 812.3 / 0.73 | 1064.2 / 904.1 / 0.30 | 4288.1 / 897.4 / 0.17 |
| humming w3 zp | 62.2 / 15.1 / 3.01 | 62.8 / 59.9 / 2.99 | 80.9 / 185.9 / 2.34 | 83.3 / 361.1 / 2.30 | 104.7 / 574.5 / 1.87 | 157.5 / 763.4 / 1.29 | 298.6 / 805.5 / 0.74 | 1067.0 / 901.7 / 0.30 | 4280.9 / 899.0 / 0.17 |
| humming w4 sym | 60.6 / 15.5 / 4.00 | 61.6 / 61.0 / 3.94 | 79.3 / 189.6 / 3.08 | 82.2 / 365.5 / 3.00 | 105.8 / 568.5 / 2.37 | 160.1 / 751.3 / 1.62 | 299.7 / 802.4 / 0.92 | 1083.2 / 888.1 / 0.35 | 4361.4 / 882.4 / 0.18 |
| humming w4 zp | 61.8 / 15.2 / 3.98 | 63.6 / 59.1 / 3.88 | 80.4 / 186.9 / 3.08 | 82.8 / 362.9 / 3.02 | 111.1 / 541.5 / 2.29 | 162.3 / 741.2 / 1.62 | 304.7 / 789.2 / 0.92 | 1103.3 / 872.0 / 0.34 | 4433.6 / 868.0 / 0.18 |
| humming w8 sym | 88.0 / 10.7 / 5.42 | 85.1 / 44.1 / 5.61 | 97.8 / 153.7 / 4.90 | 100.2 / 300.1 / 4.80 | 123.8 / 485.6 / 3.92 | 182.5 / 659.1 / 2.71 | 348.6 / 690.0 / 1.47 | 1271.1 / 756.9 / 0.48 | 5054.3 / 761.4 / 0.20 |
| humming w8 zp | 90.0 / 10.4 / 5.34 | 87.7 / 42.9 / 5.49 | 99.0 / 151.8 / 4.88 | 101.9 / 295.1 / 4.76 | 126.5 / 475.4 / 3.87 | 186.3 / 645.5 / 2.67 | 357.5 / 672.8 / 1.44 | 1292.8 / 744.2 / 0.48 | 5123.5 / 751.1 / 0.20 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.1 / 6.3 / 6.25 | 74.3 / 25.3 / 6.32 | 74.7 / 100.6 / 6.30 | 77.3 / 194.4 / 6.11 | 81.9 / 366.9 / 5.79 | 93.6 / 642.2 / 5.12 | 118.8 / 1012.2 / 4.11 | 355.1 / 1354.6 / 1.54 | 1218.1 / 1579.6 / 0.63 |
| vllm_triton_w4a16 w4 sym | 577.6 / 0.8 / 0.21 | 577.1 / 3.3 / 0.21 | 576.5 / 13.0 / 0.21 | 576.7 / 26.1 / 0.21 | 570.7 / 52.7 / 0.22 | 923.7 / 65.1 / 0.14 | 921.6 / 130.5 / 0.15 | 1061.4 / 453.2 / 0.18 | 4185.9 / 459.7 / 0.10 |
| vllm_triton_w4a16 w4 zp | 605.1 / 0.8 / 0.20 | 607.5 / 3.1 / 0.20 | 601.1 / 12.5 / 0.20 | 606.7 / 24.8 / 0.20 | 600.1 / 50.1 / 0.21 | 854.7 / 70.4 / 0.15 | 851.6 / 141.2 / 0.17 | 1187.4 / 405.1 / 0.17 | 4578.3 / 420.3 / 0.09 |
| dequant_triton+hipblaslt w4 sym | 186.8 / 2.5 / 0.65 | 202.9 / 9.3 / 0.60 | 201.7 / 37.3 / 0.61 | 206.1 / 72.9 / 0.60 | 207.5 / 144.9 / 0.61 | 214.9 / 279.8 / 0.61 | 237.1 / 507.2 / 0.59 | 476.2 / 1010.2 / 0.41 | 1333.2 / 1443.2 / 0.32 |
| dequant_triton+hipblaslt w4 zp | 201.5 / 2.3 / 0.61 | 228.1 / 8.2 / 0.54 | 219.5 / 34.2 / 0.56 | 232.2 / 64.7 / 0.54 | 226.1 / 133.0 / 0.56 | 238.1 / 252.6 / 0.55 | 257.3 / 467.3 / 0.55 | 494.9 / 972.1 / 0.40 | 1346.5 / 1429.0 / 0.32 |
| dequant_triton+hipblaslt w8 sym | 211.4 / 2.2 / 1.13 | 229.5 / 8.2 / 1.04 | 212.4 / 35.4 / 1.13 | 233.2 / 64.5 / 1.03 | 217.1 / 138.5 / 1.12 | 239.9 / 250.7 / 1.03 | 249.4 / 482.1 / 1.03 | 491.1 / 979.5 / 0.64 | 1343.5 / 1432.2 / 0.40 |
| dequant_triton+hipblaslt w8 zp | 225.9 / 2.1 / 1.06 | 217.1 / 8.7 / 1.11 | 230.3 / 32.6 / 1.05 | 241.7 / 62.2 / 1.00 | 233.8 / 128.6 / 1.05 | 235.6 / 255.2 / 1.06 | 260.9 / 461.0 / 0.99 | 492.8 / 976.1 / 0.64 | 1352.9 / 1422.3 / 0.40 |
| aiter_a16w8_int8 w8 sym | 49.3 / 9.5 / 4.92 | 50.5 / 37.2 / 4.80 | 70.4 / 106.7 / 3.46 | 67.0 / 224.3 / 3.65 | 83.1 / 361.8 / 2.97 | 133.5 / 450.5 / 1.89 | 260.6 / 461.5 / 1.00 | 795.0 / 605.1 / 0.40 | 3052.8 / 630.3 / 0.18 |
| aiter_a16w8_fp8_block128 w8 sym | 42.2 / 11.1 / 5.58 | 43.5 / 43.2 / 5.41 | 47.3 / 158.8 / 4.99 | 64.8 / 232.1 / 3.66 | 77.7 / 386.9 / 3.08 | 111.4 / 539.6 / 2.19 | 252.6 / 476.0 / 1.00 | 671.6 / 716.2 / 0.46 | 2568.3 / 749.2 / 0.21 |
| humming w2 sym | 33.6 / 14.0 / 1.86 | 33.7 / 55.7 / 1.86 | 45.5 / 165.2 / 1.40 | 46.7 / 322.2 / 1.39 | 57.4 / 524.2 / 1.17 | 89.2 / 674.3 / 0.81 | 162.5 / 739.9 / 0.50 | 527.1 / 912.6 / 0.26 | 2020.1 / 952.5 / 0.18 |
| humming w2 zp | 36.9 / 12.7 / 1.74 | 38.0 / 49.4 / 1.70 | 48.4 / 155.3 / 1.35 | 48.8 / 308.0 / 1.36 | 58.9 / 510.7 / 1.17 | 91.1 / 659.8 / 0.81 | 166.7 / 721.6 / 0.50 | 536.1 / 897.2 / 0.26 | 2045.7 / 940.6 / 0.18 |
| humming w3 sym | 35.4 / 13.3 / 2.59 | 35.7 / 52.6 / 2.58 | 47.0 / 160.0 / 1.98 | 48.1 / 312.5 / 1.96 | 59.8 / 502.8 / 1.61 | 92.4 / 651.1 / 1.10 | 167.0 / 720.1 / 0.66 | 536.1 / 897.3 / 0.31 | 2077.2 / 926.3 / 0.19 |
| humming w3 zp | 40.0 / 11.8 / 2.34 | 41.1 / 45.7 / 2.28 | 48.1 / 156.3 / 1.97 | 49.6 / 303.0 / 1.93 | 61.3 / 490.4 / 1.60 | 92.3 / 651.4 / 1.12 | 166.5 / 722.2 / 0.68 | 538.0 / 894.1 / 0.31 | 2087.6 / 921.7 / 0.19 |
| humming w4 sym | 35.1 / 13.4 / 3.45 | 35.9 / 52.4 / 3.38 | 46.5 / 161.6 / 2.63 | 48.6 / 309.1 / 2.54 | 62.6 / 480.0 / 2.01 | 93.6 / 642.2 / 1.39 | 168.3 / 714.7 / 0.83 | 537.7 / 894.5 / 0.37 | 2096.5 / 917.8 / 0.20 |
| humming w4 zp | 40.6 / 11.6 / 3.03 | 41.6 / 45.2 / 2.96 | 47.9 / 157.1 / 2.59 | 50.5 / 297.4 / 2.48 | 65.6 / 458.6 / 1.95 | 95.2 / 631.3 / 1.39 | 171.4 / 701.5 / 0.83 | 549.5 / 875.4 / 0.36 | 2117.2 / 908.8 / 0.20 |
| humming w8 sym | 45.5 / 10.3 / 5.25 | 45.7 / 41.1 / 5.22 | 54.8 / 137.1 / 4.37 | 57.7 / 260.7 / 4.18 | 72.3 / 416.0 / 3.37 | 108.0 / 556.8 / 2.30 | 189.2 / 635.6 / 1.36 | 563.3 / 854.0 / 0.56 | 2232.5 / 861.9 / 0.24 |
| humming w8 zp | 46.4 / 10.1 / 5.19 | 46.8 / 40.2 / 5.15 | 54.9 / 136.9 / 4.40 | 56.9 / 264.0 / 4.26 | 73.1 / 411.3 / 3.35 | 109.7 / 547.9 / 2.28 | 191.7 / 627.4 / 1.35 | 576.1 / 835.0 / 0.55 | 2282.8 / 842.9 / 0.24 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 16.1 / 3.3 / 3.27 | 16.4 / 12.8 / 3.20 | 16.5 / 50.8 / 3.20 | 16.6 / 101.2 / 3.20 | 23.0 / 145.7 / 2.33 | 25.1 / 266.9 / 2.19 | 35.4 / 378.9 / 1.63 | 65.5 / 819.6 / 1.12 | 179.7 / 1195.3 / 0.76 |
| vllm_triton_w4a16 w4 sym | 106.0 / 0.5 / 0.13 | 105.9 / 2.0 / 0.13 | 106.2 / 7.9 / 0.13 | 106.4 / 15.8 / 0.13 | 105.2 / 31.9 / 0.14 | 169.0 / 39.7 / 0.10 | 167.7 / 80.0 / 0.11 | 195.4 / 274.7 / 0.18 | 513.3 / 418.4 / 0.19 |
| vllm_triton_w4a16 w4 zp | 110.7 / 0.5 / 0.12 | 111.2 / 1.9 / 0.12 | 111.6 / 7.5 / 0.12 | 111.6 / 15.0 / 0.13 | 110.1 / 30.5 / 0.14 | 156.3 / 42.9 / 0.10 | 155.8 / 86.1 / 0.12 | 215.4 / 249.2 / 0.16 | 525.7 / 408.5 / 0.18 |
| dequant_triton+hipblaslt w4 sym | 31.5 / 1.7 / 0.43 | 31.8 / 6.6 / 0.43 | 31.2 / 26.9 / 0.44 | 33.2 / 50.6 / 0.43 | 37.7 / 89.1 / 0.39 | 38.9 / 172.4 / 0.41 | 49.8 / 269.5 / 0.38 | 71.5 / 750.4 / 0.48 | 190.4 / 1127.8 / 0.51 |
| dequant_triton+hipblaslt w4 zp | 31.8 / 1.6 / 0.43 | 32.1 / 6.5 / 0.43 | 31.4 / 26.7 / 0.44 | 33.4 / 50.3 / 0.43 | 38.1 / 88.1 / 0.39 | 38.9 / 172.5 / 0.42 | 50.3 / 266.8 / 0.38 | 70.8 / 758.3 / 0.49 | 187.8 / 1143.8 / 0.52 |
| dequant_triton+hipblaslt w8 sym | 32.7 / 1.6 / 0.81 | 33.1 / 6.3 / 0.81 | 32.1 / 26.1 / 0.84 | 34.4 / 48.8 / 0.79 | 39.0 / 86.0 / 0.72 | 40.1 / 167.3 / 0.73 | 50.7 / 264.6 / 0.63 | 70.3 / 763.9 / 0.68 | 190.3 / 1128.6 / 0.58 |
| dequant_triton+hipblaslt w8 zp | 32.5 / 1.6 / 0.83 | 32.8 / 6.4 / 0.82 | 31.6 / 26.5 / 0.86 | 33.8 / 49.7 / 0.81 | 38.8 / 86.5 / 0.73 | 40.1 / 167.2 / 0.73 | 50.5 / 265.8 / 0.64 | 70.4 / 762.7 / 0.68 | 190.7 / 1126.2 / 0.58 |
| aiter_a16w8_int8 w8 sym | 12.5 / 4.2 / 2.16 | 12.5 / 16.8 / 2.17 | 16.2 / 51.8 / 1.69 | 16.9 / 99.0 / 1.63 | 21.1 / 159.4 / 1.35 | 33.4 / 200.7 / 0.89 | 47.4 / 283.1 / 0.68 | 144.9 / 370.5 / 0.33 | 359.1 / 598.0 / 0.31 |
| aiter_a16w8_fp8_block128 w8 sym | 10.2 / 5.1 / 2.58 | 11.1 / 18.8 / 2.36 | 12.0 / 70.2 / 2.22 | 15.7 / 106.9 / 1.71 | 19.5 / 172.5 / 1.42 | 28.9 / 232.3 / 1.00 | 47.3 / 283.5 / 0.67 | 122.2 / 439.3 / 0.39 | 302.8 / 709.3 / 0.36 |
| humming w2 sym | 9.7 / 5.4 / 0.72 | 10.4 / 20.2 / 0.68 | 13.2 / 63.5 / 0.55 | 13.4 / 125.3 / 0.57 | 17.2 / 195.1 / 0.48 | 27.2 / 246.3 / 0.35 | 35.0 / 383.1 / 0.35 | 89.8 / 598.1 / 0.31 | 240.8 / 891.7 / 0.38 |
| humming w2 zp | 10.0 / 5.2 / 0.72 | 10.7 / 19.5 / 0.68 | 13.3 / 63.0 / 0.56 | 13.5 / 124.0 / 0.58 | 17.2 / 194.7 / 0.49 | 27.4 / 244.6 / 0.36 | 35.9 / 373.5 / 0.34 | 91.2 / 588.8 / 0.31 | 247.8 / 866.8 / 0.37 |
| humming w3 sym | 10.1 / 5.2 / 1.02 | 10.6 / 19.9 / 0.98 | 13.5 / 62.3 / 0.79 | 13.8 / 121.6 / 0.79 | 17.7 / 189.7 / 0.65 | 28.3 / 237.1 / 0.45 | 35.7 / 375.6 / 0.43 | 89.9 / 597.4 / 0.35 | 244.9 / 876.9 / 0.38 |
| humming w3 zp | 10.4 / 5.0 / 1.00 | 11.1 / 18.9 / 0.95 | 13.7 / 61.2 / 0.79 | 14.0 / 119.8 / 0.79 | 17.8 / 188.7 / 0.66 | 27.6 / 242.9 / 0.47 | 36.0 / 372.4 / 0.43 | 89.7 / 598.5 / 0.35 | 246.4 / 871.4 / 0.38 |
| humming w4 sym | 10.0 / 5.3 / 1.36 | 10.4 / 20.1 / 1.31 | 13.4 / 62.8 / 1.04 | 13.8 / 121.2 / 1.02 | 17.7 / 189.4 / 0.84 | 27.8 / 241.0 / 0.58 | 36.1 / 372.3 / 0.52 | 88.9 / 603.7 / 0.39 | 247.5 / 867.6 / 0.39 |
| humming w4 zp | 10.4 / 5.0 / 1.32 | 11.0 / 19.0 / 1.25 | 13.5 / 62.1 / 1.04 | 13.8 / 121.8 / 1.04 | 18.3 / 183.2 / 0.82 | 27.9 / 240.1 / 0.58 | 36.6 / 366.5 / 0.52 | 91.1 / 589.1 / 0.38 | 249.8 / 859.8 / 0.39 |
| humming w8 sym | 11.3 / 4.6 / 2.35 | 11.5 / 18.2 / 2.31 | 14.2 / 59.1 / 1.90 | 14.6 / 114.8 / 1.87 | 18.6 / 180.6 / 1.50 | 29.4 / 228.4 / 0.99 | 39.0 / 343.7 / 0.82 | 93.6 / 573.8 / 0.51 | 261.8 / 820.3 / 0.42 |
| humming w8 zp | 11.7 / 4.5 / 2.30 | 12.0 / 17.4 / 2.23 | 14.3 / 58.5 / 1.90 | 14.9 / 112.9 / 1.85 | 18.7 / 179.6 / 1.51 | 29.5 / 227.5 / 1.00 | 39.0 / 344.2 / 0.82 | 95.1 / 564.5 / 0.50 | 265.5 / 808.8 / 0.42 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 84.2 / 6.2 / 6.23 | 84.1 / 24.9 / 6.24 | 82.8 / 101.3 / 6.35 | 87.6 / 191.5 / 6.03 | 85.8 / 391.2 / 6.20 | 104.4 / 642.9 / 5.16 | 128.5 / 1044.3 / 4.30 | 503.1 / 1067.1 / 1.27 | 1489.6 / 1441.7 / 0.66 |
| vllm_triton_w4a16 w4 sym | 177.9 / 2.9 / 0.76 | 177.4 / 11.8 / 0.77 | 180.4 / 46.5 / 0.76 | 180.4 / 93.0 / 0.77 | 175.5 / 191.2 / 0.81 | 202.8 / 330.9 / 0.74 | 389.8 / 344.3 / 0.42 | 1217.6 / 440.9 / 0.21 | 4585.3 / 468.3 / 0.13 |
| vllm_triton_w4a16 w4 zp | 193.0 / 2.7 / 0.71 | 192.7 / 10.9 / 0.71 | 193.4 / 43.4 / 0.71 | 193.6 / 86.6 / 0.72 | 186.1 / 180.3 / 0.77 | 217.9 / 307.9 / 0.69 | 419.1 / 320.3 / 0.39 | 1310.9 / 409.6 / 0.19 | 4981.0 / 431.1 / 0.12 |
| dequant_triton+hipblaslt w4 sym | 216.2 / 2.4 / 0.63 | 216.4 / 9.7 / 0.63 | 216.5 / 38.7 / 0.63 | 218.8 / 76.7 / 0.63 | 218.7 / 153.4 / 0.65 | 236.1 / 284.2 / 0.63 | 257.2 / 521.8 / 0.64 | 622.1 / 863.0 / 0.40 | 1619.2 / 1326.2 / 0.37 |
| dequant_triton+hipblaslt w4 zp | 222.9 / 2.4 / 0.61 | 222.6 / 9.4 / 0.61 | 223.1 / 37.6 / 0.62 | 227.1 / 73.9 / 0.62 | 226.3 / 148.3 / 0.63 | 242.9 / 276.3 / 0.62 | 262.5 / 511.3 / 0.63 | 623.2 / 861.5 / 0.40 | 1622.9 / 1323.3 / 0.37 |
| dequant_triton+hipblaslt w8 sym | 230.1 / 2.3 / 1.16 | 232.2 / 9.0 / 1.15 | 231.0 / 36.3 / 1.16 | 235.1 / 71.4 / 1.15 | 233.9 / 143.4 / 1.17 | 253.4 / 264.9 / 1.11 | 272.8 / 492.0 / 1.08 | 632.9 / 848.3 / 0.60 | 1625.4 / 1321.2 / 0.45 |
| dequant_triton+hipblaslt w8 zp | 232.1 / 2.3 / 1.16 | 235.4 / 8.9 / 1.14 | 231.5 / 36.2 / 1.17 | 238.1 / 70.5 / 1.14 | 234.3 / 143.2 / 1.18 | 256.7 / 261.5 / 1.10 | 271.3 / 494.6 / 1.09 | 638.7 / 840.5 / 0.60 | 1635.1 / 1313.4 / 0.45 |
| aiter_a16w8_int8 w8 sym | 64.3 / 8.2 / 4.21 | 71.2 / 29.5 / 3.81 | 87.4 / 96.0 / 3.11 | 93.5 / 179.4 / 2.93 | 117.7 / 285.0 / 2.36 | 195.8 / 342.7 / 1.45 | 318.5 / 421.4 / 0.94 | 955.1 / 562.1 / 0.40 | 3541.8 / 606.3 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 48.5 / 10.8 / 5.40 | 54.9 / 38.2 / 4.78 | 66.5 / 126.1 / 3.97 | 85.8 / 195.6 / 3.10 | 109.4 / 306.8 / 2.46 | 162.5 / 413.1 / 1.70 | 318.5 / 421.4 / 0.91 | 808.1 / 664.3 / 0.47 | 2992.5 / 717.6 / 0.24 |
| humming w2 sym | 36.1 / 14.5 / 1.93 | 36.7 / 57.2 / 1.91 | 48.6 / 172.6 / 1.47 | 49.2 / 341.1 / 1.49 | 61.2 / 548.2 / 1.25 | 94.5 / 710.1 / 0.89 | 177.3 / 756.9 / 0.56 | 594.0 / 903.9 / 0.31 | 2296.4 / 935.2 / 0.23 |
| humming w2 zp | 37.6 / 13.9 / 1.91 | 37.7 / 55.6 / 1.91 | 49.3 / 170.3 / 1.49 | 50.1 / 335.1 / 1.50 | 62.5 / 536.8 / 1.26 | 96.7 / 694.3 / 0.89 | 180.1 / 745.4 / 0.56 | 601.9 / 891.9 / 0.31 | 2330.2 / 921.6 / 0.23 |
| humming w3 sym | 38.2 / 13.7 / 2.68 | 38.9 / 53.9 / 2.64 | 50.9 / 164.8 / 2.05 | 52.1 / 321.8 / 2.03 | 64.5 / 520.5 / 1.70 | 99.6 / 673.8 / 1.17 | 182.0 / 737.6 / 0.72 | 613.8 / 874.7 / 0.35 | 2388.6 / 899.1 / 0.24 |
| humming w3 zp | 40.3 / 13.0 / 2.59 | 40.1 / 52.3 / 2.62 | 50.9 / 164.7 / 2.09 | 51.9 / 323.0 / 2.08 | 65.9 / 508.9 / 1.69 | 98.7 / 679.9 / 1.20 | 183.1 / 732.9 / 0.73 | 615.6 / 872.1 / 0.36 | 2401.2 / 894.3 / 0.24 |
| humming w4 sym | 37.4 / 14.0 / 3.62 | 38.1 / 55.1 / 3.56 | 49.4 / 169.7 / 2.77 | 50.7 / 330.7 / 2.74 | 66.0 / 508.3 / 2.16 | 99.7 / 673.4 / 1.50 | 183.7 / 730.5 / 0.89 | 621.0 / 864.5 / 0.40 | 2411.2 / 890.6 / 0.25 |
| humming w4 zp | 39.3 / 13.3 / 3.50 | 39.1 / 53.6 / 3.52 | 50.1 / 167.3 / 2.77 | 51.2 / 327.7 / 2.75 | 67.4 / 497.7 / 2.14 | 101.3 / 662.7 / 1.50 | 187.3 / 716.7 / 0.89 | 628.7 / 854.0 / 0.40 | 2446.8 / 877.7 / 0.24 |
| humming w8 sym | 52.0 / 10.1 / 5.12 | 53.3 / 39.3 / 5.00 | 59.2 / 141.6 / 4.53 | 61.9 / 270.9 / 4.36 | 76.6 / 438.2 / 3.57 | 110.9 / 604.9 / 2.53 | 203.1 / 660.7 / 1.45 | 698.3 / 768.9 / 0.55 | 2679.2 / 801.5 / 0.27 |
| humming w8 zp | 53.7 / 9.8 / 5.00 | 55.3 / 37.9 / 4.86 | 60.5 / 138.6 / 4.46 | 63.2 / 265.5 / 4.30 | 77.7 / 431.6 / 3.54 | 114.4 / 586.7 / 2.47 | 206.6 / 649.6 / 1.44 | 710.2 / 755.9 / 0.54 | 2732.3 / 786.0 / 0.27 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 45.8 / 5.7 / 5.73 | 46.2 / 22.7 / 5.67 | 47.8 / 87.7 / 5.50 | 50.6 / 165.9 / 5.22 | 56.5 / 296.8 / 4.71 | 68.0 / 493.7 / 3.97 | 91.4 / 734.0 / 3.04 | 233.7 / 1148.9 / 1.39 | 836.4 / 1283.8 / 0.61 |
| vllm_triton_w4a16 w4 sym | 518.5 / 0.5 / 0.13 | 516.6 / 2.0 / 0.13 | 516.7 / 8.1 / 0.13 | 516.9 / 16.2 / 0.14 | 512.1 / 32.8 / 0.14 | 824.2 / 40.7 / 0.09 | 819.4 / 81.9 / 0.10 | 952.2 / 281.9 / 0.14 | 2535.3 / 423.5 / 0.13 |
| vllm_triton_w4a16 w4 zp | 541.1 / 0.5 / 0.13 | 543.7 / 1.9 / 0.13 | 543.0 / 7.7 / 0.13 | 543.3 / 15.4 / 0.13 | 536.6 / 31.3 / 0.13 | 763.6 / 43.9 / 0.10 | 761.3 / 88.2 / 0.11 | 1052.9 / 254.9 / 0.12 | 2610.2 / 411.4 / 0.12 |
| dequant_triton+hipblaslt w4 sym | 116.1 / 2.3 / 0.58 | 116.1 / 9.0 / 0.58 | 117.1 / 35.8 / 0.59 | 114.9 / 73.0 / 0.60 | 123.4 / 136.0 / 0.58 | 137.1 / 244.8 / 0.55 | 167.0 / 401.8 / 0.50 | 293.4 / 915.0 / 0.45 | 908.1 / 1182.4 / 0.35 |
| dequant_triton+hipblaslt w4 zp | 124.6 / 2.1 / 0.55 | 124.7 / 8.4 / 0.55 | 125.8 / 33.3 / 0.55 | 123.8 / 67.8 / 0.57 | 132.7 / 126.4 / 0.54 | 145.7 / 230.3 / 0.52 | 175.1 / 383.3 / 0.48 | 301.7 / 889.6 / 0.43 | 914.2 / 1174.5 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 116.0 / 2.3 / 1.15 | 117.1 / 9.0 / 1.14 | 118.6 / 35.4 / 1.13 | 121.3 / 69.1 / 1.11 | 125.9 / 133.2 / 1.09 | 140.8 / 238.2 / 1.00 | 170.0 / 394.7 / 0.88 | 299.0 / 897.6 / 0.66 | 904.6 / 1186.9 / 0.42 |
| dequant_triton+hipblaslt w8 zp | 123.6 / 2.1 / 1.09 | 119.0 / 8.8 / 1.13 | 123.7 / 33.9 / 1.09 | 122.4 / 68.5 / 1.11 | 130.8 / 128.2 / 1.05 | 142.3 / 235.8 / 1.00 | 173.3 / 387.3 / 0.86 | 302.5 / 887.5 / 0.65 | 915.3 / 1173.1 / 0.42 |
| aiter_a16w8_int8 w8 sym | 34.1 / 7.7 / 3.97 | 35.1 / 29.9 / 3.86 | 52.5 / 79.9 / 2.60 | 54.7 / 153.4 / 2.51 | 65.1 / 257.5 / 2.13 | 115.4 / 290.7 / 1.24 | 214.2 / 313.3 / 0.70 | 683.0 / 393.0 / 0.29 | 1697.6 / 632.5 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 29.7 / 8.8 / 4.42 | 30.7 / 34.2 / 4.28 | 32.5 / 129.0 / 4.06 | 53.0 / 158.3 / 2.51 | 61.2 / 274.2 / 2.21 | 93.9 / 357.2 / 1.48 | 215.8 / 311.0 / 0.68 | 572.1 / 469.2 / 0.34 | 1428.3 / 751.8 / 0.27 |
| humming w2 sym | 27.9 / 9.4 / 1.25 | 30.2 / 34.8 / 1.16 | 35.8 / 117.1 / 1.00 | 36.6 / 229.1 / 1.00 | 43.4 / 386.8 / 0.89 | 68.3 / 491.2 / 0.62 | 117.7 / 570.1 / 0.43 | 422.3 / 635.7 / 0.23 | 1156.4 / 928.5 / 0.25 |
| humming w2 zp | 28.9 / 9.1 / 1.24 | 31.3 / 33.5 / 1.15 | 35.8 / 117.2 / 1.03 | 36.0 / 233.1 / 1.05 | 43.4 / 386.1 / 0.92 | 70.0 / 479.3 / 0.62 | 120.0 / 559.4 / 0.43 | 428.0 / 627.2 / 0.23 | 1164.3 / 922.3 / 0.25 |
| humming w3 sym | 29.1 / 9.0 / 1.76 | 31.1 / 33.7 / 1.66 | 37.7 / 111.3 / 1.39 | 38.4 / 218.3 / 1.38 | 45.1 / 372.3 / 1.22 | 69.8 / 480.5 / 0.85 | 118.6 / 565.8 / 0.56 | 420.5 / 638.4 / 0.27 | 1169.8 / 917.9 / 0.26 |
| humming w3 zp | 30.6 / 8.6 / 1.71 | 32.9 / 31.9 / 1.60 | 37.2 / 112.7 / 1.43 | 38.6 / 217.2 / 1.40 | 46.7 / 359.6 / 1.20 | 69.6 / 482.2 / 0.86 | 118.6 / 566.1 / 0.57 | 420.1 / 639.0 / 0.27 | 1173.3 / 915.2 / 0.26 |
| humming w4 sym | 28.8 / 9.1 / 2.35 | 30.5 / 34.4 / 2.22 | 34.9 / 120.1 / 1.96 | 35.9 / 233.5 / 1.94 | 45.1 / 372.1 / 1.59 | 70.4 / 476.4 / 1.07 | 118.3 / 567.1 / 0.70 | 416.1 / 645.2 / 0.31 | 1168.5 / 918.9 / 0.27 |
| humming w4 zp | 30.8 / 8.5 / 2.23 | 33.2 / 31.5 / 2.07 | 35.7 / 117.4 / 1.95 | 36.2 / 232.0 / 1.95 | 46.4 / 361.7 / 1.56 | 71.6 / 468.8 / 1.07 | 120.8 / 555.7 / 0.70 | 428.4 / 626.6 / 0.31 | 1193.5 / 899.6 / 0.27 |
| humming w8 sym | 36.3 / 7.2 / 3.67 | 37.7 / 27.8 / 3.54 | 38.4 / 109.2 / 3.49 | 40.5 / 207.4 / 3.34 | 50.9 / 329.4 / 2.69 | 75.2 / 445.9 / 1.87 | 126.0 / 532.6 / 1.18 | 444.9 / 603.4 / 0.44 | 1245.8 / 861.9 / 0.31 |
| humming w8 zp | 37.3 / 7.0 / 3.59 | 39.0 / 26.9 / 3.45 | 38.8 / 108.0 / 3.48 | 41.0 / 204.4 / 3.32 | 51.7 / 324.5 / 2.67 | 76.8 / 436.8 / 1.85 | 129.5 / 518.1 / 1.16 | 442.3 / 606.8 / 0.45 | 1249.7 / 859.2 / 0.31 |
