# ROCm baseline GEMM benchmark (2026-10-28T16:41:10)

Y[M,N] = X[M,K] @ W[N,K]^T; weight-only quant unless noted; group size 128; zp = asymmetric zero point, sym = uint{4,8}b{8,128}.

## Environment

- device: `AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, HIP_VISIBLE_DEVICES=3`
- torch: `2.11.0+rocm7.2`
- triton: `3.6.0`
- hip: `7.2.26015`
- date: `2026-10-28T16:41:10`
- args: `--impls torch_dense vllm_triton_w4a16 vllm_exllama_gptq dequant_triton+hipblaslt aiter_a16w8_int8 aiter_a16w8_fp8_block128 humming --bits 4 8 --windows 3 --out-dir benchmarks/rocm/results --prefix humming_perf --tag 2026-10-28`

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
| measured: copy 4 GiB (read+write) TB/s | 4.95 |
| measured: sum 4 GiB (read-only) TB/s | 3.46 |
| measured: fp16 8192^3 F.linear TFLOPs | 1441.80 |
| measured: bf16 8192^3 F.linear TFLOPs | 1605.76 |

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
| vllm_exllama_gptq [fp16 w4 sym] | OK: ok (max rel err 5.94e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w4 zp] | OK: ok (max rel err 6.46e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 sym] | OK: ok (max rel err 7.90e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 zp] | OK: ok (max rel err 7.77e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| dequant_triton+hipblaslt [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 sym] | OK: ok (max rel err 3.17e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 zp] | OK: ok (max rel err 3.12e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| aiter_a16w8_int8 [fp16 w8 sym] | OK: ok (max rel err 2.12e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_int8 [fp16 w8 zp] | SKIP: unsupported: symmetric only (no zero-point input) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_fp8_block128 [fp16 w8 sym] | OK: ok (max rel err 2.13e-04) | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| aiter_a16w8_fp8_block128 [fp16 w8 zp] | SKIP: unsupported: fp8 weights: no zero point | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| humming [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w4 zp] | OK: ok (max rel err 2.85e-04) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w8 sym] | OK: ok (max rel err 2.93e-04) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [fp16 w8 zp] | OK: ok (max rel err 2.90e-04) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
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
| humming [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w4 zp] | OK: ok (max rel err 2.27e-03) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w8 sym] | OK: ok (max rel err 2.35e-03) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |
| humming [bf16 w8 zp] | OK: ok (max rel err 2.31e-03) | humming@1d082f19d1 HummingLayer (gfx950 MFMA GEMM, humming/kernel/humming_mfma.cuh) |

## Headline (fp16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.2 (2.75 TB/s) | 10.3 (3.29 TB/s) | 42.4 (811 TF) | 96.5 (1425 TF) | 157.5 (5.97 TB/s) | 155.4 (6.06 TB/s) | 743.6 (1294 TF) | 2659.0 (1447 TF) | 75.8 (6.19 TB/s) | 77.4 (6.09 TB/s) | 362.7 (1326 TF) | 1274.1 (1510 TF) |
| vllm_triton_w4a16 w4 sym | 73.8 (0.12 TB/s) | 74.9 (0.12 TB/s) | 104.4 (329 TF) | 246.9 (557 TF) | 196.1 (1.24 TB/s) | 223.7 (1.09 TB/s) | 1677.5 (574 TF) | 6488.6 (593 TF) | 499.9 (0.24 TB/s) | 505.5 (0.24 TB/s) | 853.7 (563 TF) | 3295.3 (584 TF) |
| vllm_triton_w4a16 w4 zp | 80.0 (0.11 TB/s) | 81.0 (0.11 TB/s) | 112.2 (306 TF) | 262.1 (524 TF) | 214.4 (1.14 TB/s) | 239.1 (1.03 TB/s) | 1780.4 (540 TF) | 6857.1 (561 TF) | 543.3 (0.23 TB/s) | 544.7 (0.23 TB/s) | 912.2 (527 TF) | 3501.8 (549 TF) |
| vllm_exllama_gptq w4 sym | 53.2 (0.16 TB/s) | 119.7 (0.07 TB/s) | 67.1 (512 TF) | 138.3 (994 TF) | 252.7 (0.96 TB/s) | 348.5 (0.70 TB/s) | 1059.0 (909 TF) | 3302.0 (1165 TF) | 1308.9 (0.09 TB/s) | 1140.0 (0.11 TB/s) | 723.4 (665 TF) | 1605.9 (1198 TF) |
| vllm_exllama_gptq w4 zp | 53.1 (0.17 TB/s) | 119.2 (0.07 TB/s) | 67.0 (513 TF) | 138.3 (994 TF) | 250.4 (0.97 TB/s) | 348.0 (0.71 TB/s) | 1052.3 (914 TF) | 3303.4 (1165 TF) | 1305.9 (0.09 TB/s) | 1068.6 (0.12 TB/s) | 723.7 (665 TF) | 1610.8 (1195 TF) |
| vllm_exllama_gptq w8 sym | 56.0 (0.30 TB/s) | 124.6 (0.14 TB/s) | 70.0 (491 TF) | 145.9 (942 TF) | 249.6 (1.91 TB/s) | 431.8 (1.11 TB/s) | 1109.1 (867 TF) | 3362.2 (1145 TF) | 1271.6 (0.19 TB/s) | 824.2 (0.29 TB/s) | 732.0 (657 TF) | 1628.8 (1181 TF) |
| vllm_exllama_gptq w8 zp | 59.3 (0.29 TB/s) | 127.8 (0.14 TB/s) | 70.0 (491 TF) | 145.7 (943 TF) | 250.5 (1.92 TB/s) | 432.8 (1.12 TB/s) | 1107.2 (869 TF) | 3363.4 (1144 TF) | 1253.8 (0.19 TB/s) | 822.6 (0.29 TB/s) | 732.6 (657 TF) | 1632.6 (1179 TF) |
| dequant_triton+hipblaslt w4 sym | 25.1 (0.34 TB/s) | 23.1 (0.39 TB/s) | 50.6 (679 TF) | 108.2 (1271 TF) | 427.9 (0.57 TB/s) | 425.1 (0.57 TB/s) | 1006.9 (955 TF) | 2886.4 (1333 TF) | 212.9 (0.57 TB/s) | 220.0 (0.56 TB/s) | 498.7 (965 TF) | 1388.5 (1386 TF) |
| dequant_triton+hipblaslt w4 zp | 25.2 (0.35 TB/s) | 23.2 (0.39 TB/s) | 52.2 (658 TF) | 107.9 (1274 TF) | 454.8 (0.54 TB/s) | 439.9 (0.56 TB/s) | 1022.1 (941 TF) | 2906.6 (1324 TF) | 229.8 (0.53 TB/s) | 235.9 (0.52 TB/s) | 506.6 (950 TF) | 1404.6 (1370 TF) |
| dequant_triton+hipblaslt w8 sym | 25.0 (0.68 TB/s) | 23.5 (0.74 TB/s) | 51.2 (671 TF) | 110.1 (1248 TF) | 472.1 (1.01 TB/s) | 471.2 (1.02 TB/s) | 1037.1 (928 TF) | 2943.7 (1307 TF) | 241.7 (0.99 TB/s) | 238.8 (1.00 TB/s) | 502.8 (957 TF) | 1410.3 (1364 TF) |
| dequant_triton+hipblaslt w8 zp | 25.3 (0.68 TB/s) | 23.6 (0.74 TB/s) | 51.0 (673 TF) | 110.5 (1244 TF) | 468.0 (1.03 TB/s) | 467.6 (1.03 TB/s) | 1035.0 (930 TF) | 2955.3 (1302 TF) | 249.8 (0.96 TB/s) | 251.5 (0.96 TB/s) | 517.0 (930 TF) | 1423.7 (1352 TF) |
| aiter_a16w8_int8 w8 sym | 9.2 (1.89 TB/s) | 10.6 (1.65 TB/s) | 68.1 (505 TF) | 245.2 (560 TF) | 97.1 (4.99 TB/s) | 129.1 (3.77 TB/s) | 1727.4 (557 TF) | 6621.0 (581 TF) | 48.5 (5.00 TB/s) | 60.0 (4.06 TB/s) | 867.8 (554 TF) | 3312.8 (581 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.16 TB/s) | 9.5 (1.79 TB/s) | 55.1 (624 TF) | 199.4 (689 TF) | 77.0 (6.10 TB/s) | 104.1 (4.53 TB/s) | 1370.6 (702 TF) | 5266.2 (731 TF) | 42.5 (5.53 TB/s) | 47.4 (4.99 TB/s) | 677.4 (710 TF) | 2597.2 (741 TF) |
| humming w4 sym | 7.3 (1.18 TB/s) | 9.4 (0.95 TB/s) | 42.8 (803 TF) | 120.3 (1142 TF) | 50.2 (4.83 TB/s) | 50.6 (4.83 TB/s) | 770.2 (1249 TF) | 2998.9 (1283 TF) | 36.0 (3.37 TB/s) | 30.7 (3.98 TB/s) | 432.1 (1113 TF) | 1482.0 (1298 TF) |
| humming w4 zp | 7.6 (1.16 TB/s) | 9.6 (0.95 TB/s) | 43.6 (788 TF) | 124.9 (1100 TF) | 52.8 (4.66 TB/s) | 52.6 (4.71 TB/s) | 801.5 (1200 TF) | 3119.8 (1233 TF) | 38.0 (3.23 TB/s) | 31.9 (3.89 TB/s) | 442.1 (1088 TF) | 1536.8 (1252 TF) |
| humming w8 sym | 8.5 (2.00 TB/s) | 9.9 (1.75 TB/s) | 46.8 (735 TF) | 138.7 (991 TF) | 87.1 (5.48 TB/s) | 89.0 (5.38 TB/s) | 1095.0 (879 TF) | 4403.2 (874 TF) | 43.8 (5.44 TB/s) | 44.5 (5.38 TB/s) | 458.8 (1049 TF) | 1785.9 (1077 TF) |
| humming w8 zp | 8.6 (2.00 TB/s) | 10.1 (1.73 TB/s) | 47.2 (728 TF) | 140.5 (978 TF) | 88.4 (5.44 TB/s) | 90.7 (5.32 TB/s) | 1088.9 (883 TF) | 4511.9 (853 TF) | 47.2 (5.09 TB/s) | 44.8 (5.39 TB/s) | 463.5 (1038 TF) | 1795.2 (1072 TF) |

## Headline (bf16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.4 (2.71 TB/s) | 10.3 (3.28 TB/s) | 46.9 (733 TF) | 94.3 (1457 TF) | 171.2 (5.49 TB/s) | 157.4 (5.98 TB/s) | 723.3 (1330 TF) | 2546.8 (1511 TF) | 75.9 (6.19 TB/s) | 77.6 (6.07 TB/s) | 353.5 (1361 TF) | 1215.5 (1583 TF) |
| vllm_triton_w4a16 w4 sym | 84.4 (0.10 TB/s) | 84.7 (0.10 TB/s) | 135.2 (254 TF) | 307.4 (447 TF) | 283.5 (0.85 TB/s) | 296.9 (0.82 TB/s) | 2093.9 (459 TF) | 8127.7 (473 TF) | 575.2 (0.21 TB/s) | 570.3 (0.21 TB/s) | 1061.6 (453 TF) | 4186.9 (460 TF) |
| vllm_triton_w4a16 w4 zp | 87.8 (0.10 TB/s) | 89.2 (0.10 TB/s) | 126.2 (272 TF) | 340.1 (404 TF) | 307.1 (0.80 TB/s) | 316.4 (0.78 TB/s) | 2268.5 (424 TF) | 8892.3 (433 TF) | 602.3 (0.20 TB/s) | 605.7 (0.20 TB/s) | 1186.2 (406 TF) | 4580.9 (420 TF) |
| dequant_triton+hipblaslt w4 sym | 24.9 (0.35 TB/s) | 22.9 (0.39 TB/s) | 57.9 (593 TF) | 105.0 (1309 TF) | 421.0 (0.58 TB/s) | 417.2 (0.59 TB/s) | 981.8 (980 TF) | 2766.8 (1391 TF) | 218.8 (0.55 TB/s) | 219.6 (0.56 TB/s) | 481.6 (999 TF) | 1337.5 (1439 TF) |
| dequant_triton+hipblaslt w4 zp | 25.1 (0.35 TB/s) | 23.0 (0.39 TB/s) | 58.4 (589 TF) | 105.7 (1300 TF) | 426.3 (0.57 TB/s) | 428.4 (0.57 TB/s) | 994.5 (967 TF) | 2783.6 (1383 TF) | 234.3 (0.52 TB/s) | 225.8 (0.55 TB/s) | 503.0 (956 TF) | 1345.5 (1430 TF) |
| dequant_triton+hipblaslt w8 sym | 25.5 (0.67 TB/s) | 23.4 (0.74 TB/s) | 58.3 (589 TF) | 106.7 (1288 TF) | 477.9 (1.00 TB/s) | 474.7 (1.01 TB/s) | 1021.4 (942 TF) | 2819.1 (1365 TF) | 239.3 (1.00 TB/s) | 238.8 (1.00 TB/s) | 498.8 (964 TF) | 1347.7 (1428 TF) |
| dequant_triton+hipblaslt w8 zp | 25.3 (0.68 TB/s) | 23.5 (0.74 TB/s) | 58.5 (587 TF) | 107.2 (1282 TF) | 473.7 (1.01 TB/s) | 479.4 (1.01 TB/s) | 1028.9 (935 TF) | 2835.0 (1357 TF) | 233.1 (1.03 TB/s) | 235.2 (1.03 TB/s) | 507.6 (948 TF) | 1363.4 (1411 TF) |
| aiter_a16w8_int8 w8 sym | 9.6 (1.80 TB/s) | 12.6 (1.39 TB/s) | 63.5 (541 TF) | 230.9 (595 TF) | 104.9 (4.62 TB/s) | 136.4 (3.57 TB/s) | 1596.6 (603 TF) | 6165.1 (624 TF) | 49.3 (4.92 TB/s) | 71.0 (3.43 TB/s) | 795.0 (605 TF) | 3054.1 (630 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.16 TB/s) | 9.5 (1.79 TB/s) | 54.8 (627 TF) | 197.5 (696 TF) | 77.1 (6.10 TB/s) | 104.4 (4.52 TB/s) | 1359.6 (708 TF) | 5216.3 (738 TF) | 42.6 (5.52 TB/s) | 47.6 (4.96 TB/s) | 670.4 (718 TF) | 2571.1 (748 TF) |
| humming w4 sym | 7.8 (1.12 TB/s) | 9.8 (0.91 TB/s) | 42.6 (806 TF) | 135.0 (1018 TF) | 55.4 (4.37 TB/s) | 56.5 (4.32 TB/s) | 878.6 (1095 TF) | 3447.4 (1116 TF) | 37.9 (3.20 TB/s) | 33.5 (3.65 TB/s) | 443.2 (1085 TF) | 1681.7 (1144 TF) |
| humming w4 zp | 8.1 (1.09 TB/s) | 10.0 (0.91 TB/s) | 43.8 (784 TF) | 140.5 (978 TF) | 58.0 (4.24 TB/s) | 58.0 (4.27 TB/s) | 906.8 (1061 TF) | 3568.4 (1078 TF) | 41.3 (2.98 TB/s) | 35.2 (3.53 TB/s) | 455.6 (1056 TF) | 1734.0 (1110 TF) |
| humming w8 sym | 8.5 (2.00 TB/s) | 10.3 (1.69 TB/s) | 48.2 (713 TF) | 141.7 (970 TF) | 89.2 (5.35 TB/s) | 90.7 (5.28 TB/s) | 1084.7 (887 TF) | 4389.0 (877 TF) | 46.0 (5.19 TB/s) | 44.8 (5.35 TB/s) | 466.1 (1032 TF) | 1810.8 (1063 TF) |
| humming w8 zp | 8.7 (1.98 TB/s) | 10.5 (1.66 TB/s) | 48.3 (712 TF) | 145.0 (948 TF) | 86.2 (5.58 TB/s) | 91.2 (5.29 TB/s) | 1099.6 (875 TF) | 4509.5 (853 TF) | 45.0 (5.35 TB/s) | 45.2 (5.34 TB/s) | 484.0 (994 TF) | 1870.8 (1028 TF) |

## Full results (fp16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.2 / 2.7 / 2.75 | 9.4 / 14.3 / 3.59 | 10.3 / 52.3 / 3.29 | 12.6 / 85.0 / 2.70 | 13.1 / 164.6 / 2.65 | 17.0 / 253.0 / 2.10 | 22.5 / 382.1 / 1.68 | 42.4 / 811.1 / 1.19 | 96.5 / 1424.9 / 1.04 |
| vllm_triton_w4a16 w4 sym | 73.8 / 0.5 / 0.12 | 74.5 / 1.8 / 0.12 | 74.9 / 7.2 / 0.12 | 75.7 / 14.2 / 0.12 | 76.5 / 28.1 / 0.13 | 104.8 / 41.0 / 0.10 | 103.5 / 83.0 / 0.12 | 104.4 / 329.0 / 0.24 | 246.9 / 556.7 / 0.31 |
| vllm_triton_w4a16 w4 zp | 80.0 / 0.4 / 0.11 | 80.1 / 1.7 / 0.11 | 81.0 / 6.6 / 0.11 | 81.6 / 13.2 / 0.11 | 84.7 / 25.3 / 0.12 | 111.9 / 38.4 / 0.10 | 111.2 / 77.2 / 0.12 | 112.2 / 306.4 / 0.23 | 262.1 / 524.4 / 0.29 |
| vllm_exllama_gptq w4 sym | 53.2 / 0.6 / 0.16 | 71.3 / 1.9 / 0.12 | 119.7 / 4.5 / 0.07 | 169.6 / 6.3 / 0.05 | 41.4 / 51.9 / 0.23 | 43.5 / 98.7 / 0.25 | 49.2 / 174.7 / 0.26 | 67.1 / 512.3 / 0.38 | 138.3 / 993.6 / 0.55 |
| vllm_exllama_gptq w4 zp | 53.1 / 0.6 / 0.17 | 71.6 / 1.9 / 0.12 | 119.2 / 4.5 / 0.07 | 161.0 / 6.7 / 0.06 | 41.4 / 51.9 / 0.24 | 43.3 / 99.1 / 0.25 | 49.2 / 174.7 / 0.26 | 67.0 / 512.5 / 0.38 | 138.3 / 993.9 / 0.55 |
| vllm_exllama_gptq w8 sym | 56.0 / 0.6 / 0.30 | 71.9 / 1.9 / 0.24 | 124.6 / 4.3 / 0.14 | 45.2 / 23.8 / 0.39 | 45.3 / 47.4 / 0.40 | 47.3 / 90.8 / 0.40 | 53.4 / 160.9 / 0.40 | 70.0 / 490.9 / 0.48 | 145.9 / 942.1 / 0.58 |
| vllm_exllama_gptq w8 zp | 59.3 / 0.6 / 0.29 | 72.1 / 1.9 / 0.24 | 127.8 / 4.2 / 0.14 | 45.1 / 23.8 / 0.39 | 45.3 / 47.4 / 0.40 | 47.3 / 90.8 / 0.41 | 52.8 / 162.8 / 0.41 | 70.0 / 490.8 / 0.48 | 145.7 / 943.3 / 0.58 |
| dequant_triton+hipblaslt w4 sym | 25.1 / 1.3 / 0.34 | 22.5 / 6.0 / 0.39 | 23.1 / 23.2 / 0.39 | 24.7 / 43.4 / 0.37 | 25.0 / 85.8 / 0.39 | 28.3 / 151.7 / 0.38 | 32.1 / 267.3 / 0.40 | 50.6 / 679.1 / 0.50 | 108.2 / 1270.7 / 0.70 |
| dequant_triton+hipblaslt w4 zp | 25.2 / 1.3 / 0.35 | 21.9 / 6.1 / 0.40 | 23.2 / 23.2 / 0.39 | 24.4 / 44.0 / 0.38 | 24.8 / 86.5 / 0.39 | 28.2 / 152.4 / 0.38 | 32.1 / 267.8 / 0.40 | 52.2 / 658.3 / 0.49 | 107.9 / 1273.8 / 0.70 |
| dequant_triton+hipblaslt w8 sym | 25.0 / 1.3 / 0.68 | 22.4 / 6.0 / 0.76 | 23.5 / 22.9 / 0.74 | 24.9 / 43.0 / 0.70 | 25.5 / 84.1 / 0.71 | 28.6 / 150.3 / 0.67 | 33.0 / 260.5 / 0.64 | 51.2 / 671.0 / 0.66 | 110.1 / 1248.3 / 0.76 |
| dequant_triton+hipblaslt w8 zp | 25.3 / 1.3 / 0.68 | 22.0 / 6.1 / 0.78 | 23.6 / 22.8 / 0.74 | 24.6 / 43.7 / 0.72 | 25.6 / 83.9 / 0.71 | 28.4 / 151.3 / 0.68 | 32.4 / 265.2 / 0.66 | 51.0 / 673.2 / 0.67 | 110.5 / 1244.0 / 0.76 |
| aiter_a16w8_int8 w8 sym | 9.2 / 3.7 / 1.89 | 10.0 / 13.4 / 1.73 | 10.6 / 50.5 / 1.65 | 12.4 / 86.9 / 1.44 | 15.5 / 138.6 / 1.18 | 26.2 / 164.2 / 0.74 | 34.9 / 246.2 / 0.62 | 68.1 / 504.6 / 0.50 | 245.2 / 560.5 / 0.34 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.16 | 8.6 / 15.6 / 1.95 | 9.5 / 56.5 / 1.79 | 11.8 / 90.8 / 1.46 | 13.8 / 155.1 / 1.29 | 23.9 / 180.1 / 0.79 | 37.2 / 230.7 / 0.56 | 55.1 / 623.6 / 0.61 | 199.4 / 689.2 / 0.42 |
| humming w4 sym | 7.3 / 4.6 / 1.18 | 8.1 / 16.7 / 1.08 | 9.4 / 57.3 / 0.95 | 8.7 / 123.8 / 1.06 | 10.9 / 196.9 / 0.89 | 20.6 / 208.3 / 0.52 | 21.6 / 397.1 / 0.59 | 42.8 / 802.9 / 0.59 | 120.3 / 1142.3 / 0.63 |
| humming w4 zp | 7.6 / 4.4 / 1.16 | 8.2 / 16.3 / 1.07 | 9.6 / 56.2 / 0.95 | 8.8 / 121.7 / 1.05 | 11.2 / 191.6 / 0.88 | 20.8 / 206.4 / 0.52 | 21.8 / 393.6 / 0.59 | 43.6 / 787.8 / 0.59 | 124.9 / 1100.1 / 0.61 |
| humming w8 sym | 8.5 / 3.9 / 2.00 | 9.0 / 14.9 / 1.90 | 9.9 / 54.4 / 1.75 | 9.5 / 113.0 / 1.85 | 11.7 / 183.3 / 1.54 | 22.0 / 194.9 / 0.87 | 23.4 / 367.9 / 0.91 | 46.8 / 734.8 / 0.72 | 138.7 / 990.8 / 0.61 |
| humming w8 zp | 8.6 / 3.9 / 2.00 | 9.1 / 14.8 / 1.90 | 10.1 / 53.3 / 1.73 | 9.5 / 112.6 / 1.86 | 11.9 / 181.0 / 1.53 | 22.0 / 195.2 / 0.88 | 23.4 / 367.3 / 0.91 | 47.2 / 727.5 / 0.72 | 140.5 / 978.4 / 0.60 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 13.8 / 3.6 / 3.65 | 14.1 / 14.3 / 3.59 | 13.7 / 58.6 / 3.69 | 14.4 / 112.1 / 3.55 | 17.9 / 179.6 / 2.88 | 22.3 / 289.0 / 2.38 | 27.0 / 477.6 / 2.06 | 65.4 / 788.1 / 1.09 | 164.2 / 1255.6 / 0.82 |
| vllm_triton_w4a16 w4 sym | 74.8 / 0.7 / 0.17 | 74.8 / 2.7 / 0.17 | 75.3 / 10.7 / 0.18 | 75.8 / 21.2 / 0.18 | 77.1 / 41.8 / 0.18 | 105.0 / 61.3 / 0.15 | 104.2 / 123.7 / 0.17 | 122.2 / 421.9 / 0.28 | 365.7 / 563.7 / 0.27 |
| vllm_triton_w4a16 w4 zp | 80.3 / 0.6 / 0.16 | 80.8 / 2.5 / 0.16 | 81.0 / 9.9 / 0.17 | 81.3 / 19.8 / 0.17 | 84.5 / 38.1 / 0.17 | 112.6 / 57.2 / 0.14 | 111.3 / 115.8 / 0.17 | 131.4 / 392.4 / 0.26 | 387.4 / 532.1 / 0.25 |
| vllm_exllama_gptq w4 sym | 47.3 / 1.1 / 0.28 | 71.3 / 2.8 / 0.18 | 140.3 / 5.7 / 0.10 | 161.0 / 10.0 / 0.09 | 44.6 / 72.2 / 0.32 | 50.3 / 128.1 / 0.31 | 61.6 / 209.1 / 0.30 | 96.8 / 532.7 / 0.35 | 211.8 / 973.5 / 0.46 |
| vllm_exllama_gptq w4 zp | 47.1 / 1.1 / 0.28 | 71.2 / 2.8 / 0.18 | 140.7 / 5.7 / 0.10 | 161.2 / 10.0 / 0.09 | 44.5 / 72.3 / 0.32 | 49.9 / 129.1 / 0.31 | 62.0 / 207.8 / 0.29 | 96.4 / 534.5 / 0.35 | 211.3 / 975.9 / 0.46 |
| vllm_exllama_gptq w8 sym | 51.8 / 1.0 / 0.49 | 72.6 / 2.8 / 0.35 | 146.3 / 5.5 / 0.18 | 45.3 / 35.6 / 0.58 | 46.6 / 69.2 / 0.58 | 52.1 / 123.6 / 0.54 | 64.2 / 200.6 / 0.48 | 99.5 / 517.8 / 0.47 | 211.7 / 973.8 / 0.52 |
| vllm_exllama_gptq w8 zp | 51.9 / 1.0 / 0.50 | 72.5 / 2.8 / 0.36 | 146.9 / 5.5 / 0.18 | 45.5 / 35.4 / 0.58 | 46.9 / 68.7 / 0.58 | 52.2 / 123.3 / 0.54 | 64.3 / 200.4 / 0.48 | 99.5 / 517.8 / 0.47 | 215.2 / 958.0 / 0.51 |
| dequant_triton+hipblaslt w4 sym | 28.6 / 1.8 / 0.46 | 29.2 / 6.9 / 0.45 | 29.1 / 27.7 / 0.46 | 29.2 / 55.2 / 0.47 | 35.0 / 92.1 / 0.41 | 35.4 / 181.9 / 0.44 | 41.6 / 309.8 / 0.44 | 78.0 / 660.6 / 0.43 | 176.7 / 1166.9 / 0.55 |
| dequant_triton+hipblaslt w4 zp | 29.8 / 1.7 / 0.44 | 29.4 / 6.9 / 0.45 | 28.5 / 28.3 / 0.47 | 30.1 / 53.5 / 0.46 | 34.1 / 94.4 / 0.42 | 36.3 / 177.6 / 0.43 | 42.2 / 305.4 / 0.43 | 79.0 / 652.3 / 0.43 | 177.0 / 1164.5 / 0.55 |
| dequant_triton+hipblaslt w8 sym | 31.1 / 1.6 / 0.82 | 31.0 / 6.5 / 0.83 | 29.8 / 27.1 / 0.87 | 31.0 / 52.0 / 0.85 | 35.3 / 91.2 / 0.76 | 37.9 / 170.1 / 0.74 | 42.9 / 300.5 / 0.72 | 81.1 / 635.4 / 0.57 | 179.2 / 1150.2 / 0.61 |
| dequant_triton+hipblaslt w8 zp | 31.0 / 1.6 / 0.83 | 31.3 / 6.4 / 0.82 | 29.7 / 27.1 / 0.88 | 31.1 / 51.7 / 0.85 | 35.5 / 90.8 / 0.76 | 37.9 / 169.9 / 0.75 | 42.9 / 300.6 / 0.72 | 81.5 / 632.6 / 0.57 | 179.7 / 1147.3 / 0.61 |
| aiter_a16w8_int8 w8 sym | 11.0 / 4.6 / 2.36 | 11.7 / 17.2 / 2.23 | 12.8 / 62.8 / 2.05 | 14.6 / 110.0 / 1.82 | 18.8 / 171.0 / 1.45 | 29.1 / 221.6 / 0.98 | 38.8 / 331.7 / 0.80 | 127.4 / 404.7 / 0.37 | 366.4 / 562.7 / 0.30 |
| aiter_a16w8_fp8_block128 w8 sym | 9.0 / 5.6 / 2.81 | 9.7 / 20.7 / 2.59 | 11.0 / 73.1 / 2.32 | 13.9 / 116.0 / 1.86 | 17.5 / 184.6 / 1.52 | 26.7 / 241.2 / 1.04 | 39.0 / 330.7 / 0.78 | 100.0 / 515.2 / 0.46 | 297.9 / 692.1 / 0.37 |
| humming w4 sym | 9.1 / 5.5 / 1.42 | 10.0 / 20.2 / 1.31 | 9.5 / 84.7 / 1.40 | 9.1 / 176.4 / 1.49 | 12.6 / 256.0 / 1.14 | 23.1 / 279.1 / 0.68 | 26.5 / 486.9 / 0.69 | 61.9 / 833.3 / 0.55 | 176.7 / 1166.5 / 0.55 |
| humming w4 zp | 9.4 / 5.4 / 1.41 | 10.2 / 19.7 / 1.30 | 9.7 / 83.1 / 1.39 | 9.3 / 172.9 / 1.48 | 13.0 / 247.5 / 1.11 | 23.6 / 273.5 / 0.67 | 26.9 / 479.1 / 0.69 | 62.7 / 822.0 / 0.55 | 182.6 / 1129.0 / 0.53 |
| humming w8 sym | 11.2 / 4.5 / 2.29 | 11.9 / 16.9 / 2.16 | 11.3 / 71.0 / 2.28 | 11.1 / 145.2 / 2.36 | 15.0 / 215.2 / 1.79 | 25.2 / 255.3 / 1.12 | 29.3 / 439.7 / 1.05 | 66.7 / 772.2 / 0.70 | 204.5 / 1008.2 / 0.54 |
| humming w8 zp | 11.2 / 4.5 / 2.29 | 12.0 / 16.7 / 2.15 | 11.4 / 70.5 / 2.28 | 11.3 / 142.6 / 2.34 | 15.3 / 211.0 / 1.77 | 26.4 / 244.0 / 1.07 | 29.9 / 430.4 / 1.04 | 67.0 / 769.1 / 0.70 | 205.5 / 1003.1 / 0.53 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 47.9 / 4.9 / 4.90 | 50.3 / 18.7 / 4.67 | 45.2 / 83.1 / 5.22 | 48.2 / 155.8 / 4.91 | 47.1 / 319.5 / 5.08 | 51.0 / 589.2 / 4.77 | 67.4 / 891.6 / 3.73 | 195.2 / 1232.5 / 1.55 | 662.8 / 1451.5 / 0.76 |
| vllm_triton_w4a16 w4 sym | 78.2 / 3.0 / 0.78 | 78.5 / 12.0 / 0.77 | 79.0 / 47.6 / 0.78 | 79.4 / 94.7 / 0.79 | 80.1 / 187.6 / 0.81 | 105.9 / 284.0 / 0.65 | 129.5 / 464.5 / 0.60 | 448.0 / 536.8 / 0.28 | 1660.7 / 579.3 / 0.20 |
| vllm_triton_w4a16 w4 zp | 86.2 / 2.7 / 0.71 | 87.4 / 10.8 / 0.70 | 87.9 / 42.8 / 0.71 | 88.0 / 85.4 / 0.72 | 89.2 / 168.4 / 0.73 | 113.5 / 264.9 / 0.61 | 137.4 / 437.7 / 0.57 | 473.2 / 508.3 / 0.27 | 1758.2 / 547.2 / 0.19 |
| vllm_exllama_gptq w4 sym | 79.3 / 3.0 / 0.76 | 138.6 / 6.8 / 0.44 | 194.4 / 19.3 / 0.32 | 195.4 / 38.5 / 0.32 | 106.3 / 141.4 / 0.61 | 112.8 / 266.5 / 0.61 | 131.3 / 458.0 / 0.59 | 276.6 / 869.6 / 0.46 | 863.5 / 1114.2 / 0.38 |
| vllm_exllama_gptq w4 zp | 79.4 / 3.0 / 0.77 | 136.4 / 6.9 / 0.45 | 193.4 / 19.4 / 0.32 | 195.1 / 38.5 / 0.32 | 106.0 / 141.9 / 0.61 | 112.6 / 266.9 / 0.62 | 131.1 / 458.5 / 0.59 | 275.8 / 872.0 / 0.47 | 863.4 / 1114.3 / 0.38 |
| vllm_exllama_gptq w8 sym | 87.3 / 2.7 / 1.37 | 136.8 / 6.9 / 0.87 | 182.1 / 20.6 / 0.66 | 122.2 / 61.5 / 0.99 | 112.7 / 133.4 / 1.10 | 120.0 / 250.6 / 1.06 | 144.3 / 416.8 / 0.94 | 287.6 / 836.4 / 0.65 | 877.0 / 1097.0 / 0.44 |
| vllm_exllama_gptq w8 zp | 87.1 / 2.7 / 1.38 | 136.8 / 6.9 / 0.88 | 182.1 / 20.6 / 0.67 | 122.3 / 61.5 / 1.00 | 112.7 / 133.4 / 1.10 | 119.9 / 250.8 / 1.07 | 145.1 / 414.4 / 0.94 | 287.8 / 835.8 / 0.65 | 878.1 / 1095.7 / 0.44 |
| dequant_triton+hipblaslt w4 sym | 100.6 / 2.3 / 0.60 | 103.3 / 9.1 / 0.59 | 100.5 / 37.4 / 0.61 | 113.7 / 66.1 / 0.55 | 112.3 / 133.8 / 0.58 | 114.2 / 263.4 / 0.60 | 150.1 / 400.6 / 0.52 | 252.3 / 953.2 / 0.51 | 712.1 / 1351.1 / 0.46 |
| dequant_triton+hipblaslt w4 zp | 102.8 / 2.3 / 0.59 | 92.2 / 10.2 / 0.67 | 102.4 / 36.7 / 0.61 | 117.1 / 64.2 / 0.54 | 100.3 / 149.8 / 0.65 | 118.0 / 254.7 / 0.59 | 128.8 / 466.9 / 0.60 | 254.7 / 944.4 / 0.50 | 706.9 / 1361.0 / 0.47 |
| dequant_triton+hipblaslt w8 sym | 120.2 / 2.0 / 0.99 | 120.0 / 7.8 / 1.00 | 108.2 / 34.7 / 1.11 | 121.1 / 62.1 / 1.00 | 119.4 / 126.0 / 1.03 | 122.0 / 246.5 / 1.05 | 163.2 / 368.5 / 0.83 | 260.1 / 924.8 / 0.72 | 715.7 / 1344.2 / 0.54 |
| dequant_triton+hipblaslt w8 zp | 120.0 / 2.0 / 1.00 | 122.4 / 7.7 / 0.98 | 107.8 / 34.9 / 1.12 | 121.3 / 62.0 / 1.01 | 118.6 / 126.7 / 1.05 | 124.2 / 242.1 / 1.03 | 162.9 / 369.1 / 0.84 | 261.8 / 918.7 / 0.71 | 720.7 / 1335.0 / 0.54 |
| aiter_a16w8_int8 w8 sym | 30.0 / 7.8 / 4.03 | 33.4 / 28.1 / 3.63 | 41.1 / 91.5 / 2.97 | 45.7 / 164.5 / 2.70 | 59.1 / 254.4 / 2.12 | 91.0 / 330.3 / 1.42 | 150.8 / 398.7 / 0.91 | 444.5 / 541.1 / 0.42 | 1696.0 / 567.3 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 23.1 / 10.2 / 5.09 | 26.3 / 35.7 / 4.47 | 33.3 / 112.8 / 3.56 | 44.5 / 168.8 / 2.69 | 54.5 / 275.6 / 2.23 | 81.1 / 370.9 / 1.55 | 147.7 / 407.2 / 0.91 | 361.0 / 666.2 / 0.51 | 1373.4 / 700.5 / 0.28 |
| humming w4 sym | 17.1 / 13.8 / 3.56 | 17.9 / 52.4 / 3.39 | 16.5 / 228.3 / 3.74 | 19.5 / 385.2 / 3.21 | 28.3 / 531.5 / 2.29 | 46.8 / 642.6 / 1.47 | 67.0 / 897.9 / 1.16 | 217.2 / 1107.2 / 0.59 | 765.8 / 1256.3 / 0.43 |
| humming w4 zp | 17.7 / 13.2 / 3.47 | 18.6 / 50.6 / 3.33 | 17.0 / 221.4 / 3.68 | 19.8 / 379.6 / 3.21 | 28.1 / 534.1 / 2.33 | 46.5 / 646.4 / 1.50 | 69.0 / 871.9 / 1.14 | 224.7 / 1070.6 / 0.57 | 795.8 / 1208.9 / 0.41 |
| humming w8 sym | 27.1 / 8.7 / 4.41 | 27.9 / 33.7 / 4.28 | 26.6 / 141.1 / 4.52 | 29.5 / 254.8 / 4.11 | 35.9 / 418.8 / 3.44 | 51.7 / 581.1 / 2.47 | 73.2 / 821.8 / 1.86 | 245.3 / 980.7 / 0.76 | 944.8 / 1018.3 / 0.41 |
| humming w8 zp | 28.1 / 8.4 / 4.28 | 28.6 / 32.8 / 4.21 | 27.5 / 136.7 / 4.41 | 30.4 / 247.3 / 4.02 | 36.9 / 407.3 / 3.37 | 52.3 / 574.7 / 2.46 | 74.1 / 812.0 / 1.85 | 246.9 / 974.1 / 0.76 | 949.7 / 1013.0 / 0.41 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.2 / 4.9 / 4.86 | 24.7 / 19.0 / 4.76 | 26.2 / 71.8 / 4.51 | 29.6 / 127.0 / 4.01 | 29.3 / 256.3 / 4.09 | 33.9 / 443.2 / 3.60 | 51.8 / 580.2 / 2.45 | 118.0 / 1018.7 / 1.31 | 314.1 / 1531.4 / 0.85 |
| vllm_triton_w4a16 w4 sym | 254.1 / 0.5 / 0.12 | 253.9 / 1.9 / 0.12 | 256.1 / 7.3 / 0.12 | 258.8 / 14.5 / 0.12 | 261.4 / 28.8 / 0.12 | 356.5 / 42.2 / 0.10 | 353.7 / 85.0 / 0.11 | 354.7 / 339.0 / 0.19 | 839.2 / 573.2 / 0.22 |
| vllm_triton_w4a16 w4 zp | 271.4 / 0.4 / 0.11 | 271.6 / 1.7 / 0.11 | 273.6 / 6.9 / 0.11 | 276.2 / 13.6 / 0.12 | 285.0 / 26.4 / 0.12 | 380.9 / 39.5 / 0.09 | 379.0 / 79.3 / 0.10 | 381.6 / 315.1 / 0.18 | 891.7 / 539.4 / 0.20 |
| vllm_exllama_gptq w4 sym | 335.7 / 0.3 / 0.09 | 395.6 / 1.2 / 0.08 | 562.4 / 3.3 / 0.06 | 769.1 / 4.9 / 0.04 | 160.2 / 46.9 / 0.20 | 177.2 / 84.8 / 0.20 | 237.0 / 126.9 / 0.17 | 329.7 / 364.8 / 0.21 | 426.1 / 1128.9 / 0.42 |
| vllm_exllama_gptq w4 zp | 331.0 / 0.4 / 0.09 | 394.2 / 1.2 / 0.08 | 550.7 / 3.4 / 0.06 | 749.1 / 5.0 / 0.04 | 160.2 / 46.9 / 0.20 | 177.5 / 84.7 / 0.20 | 236.8 / 127.0 / 0.17 | 329.6 / 364.8 / 0.21 | 427.7 / 1124.7 / 0.42 |
| vllm_exllama_gptq w8 sym | 336.6 / 0.3 / 0.18 | 391.7 / 1.2 / 0.15 | 556.1 / 3.4 / 0.11 | 163.0 / 23.0 / 0.37 | 173.8 / 43.3 / 0.36 | 191.1 / 78.7 / 0.34 | 249.9 / 120.3 / 0.28 | 343.0 / 350.6 / 0.28 | 446.0 / 1078.7 / 0.47 |
| vllm_exllama_gptq w8 zp | 336.7 / 0.3 / 0.18 | 395.6 / 1.2 / 0.15 | 584.2 / 3.2 / 0.10 | 163.3 / 23.0 / 0.38 | 173.9 / 43.2 / 0.36 | 191.1 / 78.6 / 0.34 | 249.9 / 120.3 / 0.28 | 343.2 / 350.4 / 0.28 | 447.3 / 1075.4 / 0.47 |
| dequant_triton+hipblaslt w4 sym | 55.1 / 2.1 / 0.55 | 55.7 / 8.4 / 0.55 | 57.0 / 33.0 / 0.54 | 59.2 / 63.5 / 0.53 | 59.8 / 125.6 / 0.55 | 66.0 / 227.8 / 0.53 | 85.0 / 353.7 / 0.47 | 142.8 / 842.1 / 0.48 | 339.2 / 1418.2 / 0.53 |
| dequant_triton+hipblaslt w4 zp | 58.3 / 2.0 / 0.52 | 58.4 / 8.0 / 0.53 | 59.7 / 31.5 / 0.52 | 62.2 / 60.5 / 0.51 | 62.4 / 120.4 / 0.53 | 68.8 / 218.6 / 0.51 | 88.0 / 341.8 / 0.45 | 144.9 / 830.2 / 0.47 | 342.8 / 1403.4 / 0.53 |
| dequant_triton+hipblaslt w8 sym | 58.2 / 2.0 / 1.02 | 58.3 / 8.1 / 1.02 | 59.1 / 31.8 / 1.02 | 61.2 / 61.4 / 0.99 | 61.7 / 121.8 / 1.00 | 67.8 / 221.6 / 0.95 | 88.0 / 341.7 / 0.79 | 146.7 / 819.6 / 0.66 | 341.9 / 1407.1 / 0.62 |
| dequant_triton+hipblaslt w8 zp | 58.9 / 2.0 / 1.02 | 59.8 / 7.9 / 1.01 | 60.4 / 31.1 / 1.00 | 62.2 / 60.4 / 0.98 | 62.7 / 119.9 / 1.00 | 69.0 / 217.7 / 0.94 | 88.8 / 338.7 / 0.78 | 147.4 / 815.7 / 0.66 | 344.4 / 1396.7 / 0.61 |
| aiter_a16w8_int8 w8 sym | 19.6 / 6.0 / 3.10 | 20.7 / 22.7 / 2.93 | 20.6 / 91.2 / 2.97 | 27.7 / 135.9 / 2.23 | 35.6 / 211.0 / 1.77 | 66.3 / 226.7 / 0.98 | 112.5 / 267.2 / 0.62 | 224.2 / 536.3 / 0.44 | 840.7 / 572.2 / 0.25 |
| aiter_a16w8_fp8_block128 w8 sym | 16.7 / 7.1 / 3.53 | 17.6 / 26.7 / 3.35 | 18.5 / 101.6 / 3.21 | 27.1 / 138.5 / 2.21 | 30.4 / 247.0 / 2.01 | 56.6 / 265.7 / 1.12 | 120.1 / 250.3 / 0.57 | 173.6 / 692.7 / 0.56 | 656.1 / 733.1 / 0.32 |
| humming w4 sym | 12.0 / 9.8 / 2.53 | 13.2 / 35.5 / 2.30 | 13.6 / 138.1 / 2.27 | 15.8 / 238.2 / 1.99 | 20.1 / 373.6 / 1.62 | 31.5 / 477.5 / 1.11 | 44.3 / 678.9 / 0.90 | 131.7 / 913.3 / 0.52 | 389.9 / 1233.9 / 0.47 |
| humming w4 zp | 12.7 / 9.2 / 2.41 | 13.7 / 34.2 / 2.25 | 13.9 / 135.2 / 2.25 | 15.9 / 235.9 / 2.00 | 20.1 / 373.4 / 1.64 | 32.1 / 468.8 / 1.11 | 44.8 / 671.0 / 0.90 | 131.9 / 911.6 / 0.52 | 405.5 / 1186.4 / 0.45 |
| humming w8 sym | 14.7 / 8.0 / 4.05 | 15.4 / 30.5 / 3.88 | 17.2 / 109.3 / 3.50 | 17.6 / 213.1 / 3.45 | 22.7 / 330.9 / 2.73 | 36.9 / 407.4 / 1.74 | 49.8 / 603.3 / 1.39 | 145.7 / 825.3 / 0.67 | 457.2 / 1052.1 / 0.46 |
| humming w8 zp | 15.0 / 7.8 / 4.00 | 15.7 / 29.9 / 3.84 | 17.4 / 108.0 / 3.49 | 17.7 / 211.9 / 3.46 | 23.0 / 327.3 / 2.72 | 36.6 / 411.0 / 1.77 | 50.1 / 600.7 / 1.39 | 147.5 / 815.3 / 0.66 | 459.5 / 1046.8 / 0.46 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 25.8 / 5.2 / 5.21 | 26.1 / 20.6 / 5.15 | 27.2 / 79.0 / 4.96 | 29.2 / 147.2 / 4.64 | 30.4 / 282.5 / 4.48 | 40.4 / 425.6 / 3.43 | 64.8 / 530.1 / 2.20 | 141.4 / 972.2 / 1.19 | 366.3 / 1500.7 / 0.73 |
| vllm_triton_w4a16 w4 sym | 145.9 / 0.9 / 0.24 | 146.3 / 3.7 / 0.24 | 147.1 / 14.6 / 0.24 | 146.7 / 29.3 / 0.24 | 148.5 / 57.8 / 0.25 | 204.3 / 84.1 / 0.19 | 202.6 / 169.6 / 0.21 | 249.4 / 551.1 / 0.27 | 947.5 / 580.2 / 0.18 |
| vllm_triton_w4a16 w4 zp | 157.4 / 0.9 / 0.22 | 157.6 / 3.4 / 0.22 | 158.3 / 13.6 / 0.22 | 157.8 / 27.2 / 0.23 | 163.9 / 52.4 / 0.23 | 219.2 / 78.4 / 0.18 | 217.0 / 158.3 / 0.20 | 266.9 / 515.0 / 0.26 | 1007.2 / 545.8 / 0.17 |
| vllm_exllama_gptq w4 sym | 165.2 / 0.8 / 0.21 | 185.3 / 2.9 / 0.19 | 305.9 / 7.0 / 0.12 | 284.5 / 15.1 / 0.12 | 155.1 / 55.4 / 0.24 | 167.9 / 102.3 / 0.23 | 224.3 / 153.2 / 0.19 | 229.9 / 597.7 / 0.30 | 493.8 / 1113.3 / 0.34 |
| vllm_exllama_gptq w4 zp | 164.8 / 0.8 / 0.21 | 185.8 / 2.9 / 0.19 | 292.7 / 7.3 / 0.12 | 282.8 / 15.2 / 0.13 | 154.9 / 55.5 / 0.24 | 167.5 / 102.5 / 0.23 | 224.0 / 153.4 / 0.19 | 230.1 / 597.3 / 0.30 | 494.1 / 1112.7 / 0.34 |
| vllm_exllama_gptq w8 sym | 156.9 / 0.9 / 0.43 | 183.7 / 2.9 / 0.37 | 297.2 / 7.2 / 0.23 | 117.0 / 36.7 / 0.59 | 168.3 / 51.1 / 0.42 | 180.9 / 95.0 / 0.40 | 236.9 / 145.0 / 0.32 | 243.4 / 564.6 / 0.42 | 512.8 / 1072.1 / 0.40 |
| vllm_exllama_gptq w8 zp | 156.7 / 0.9 / 0.44 | 182.9 / 2.9 / 0.38 | 291.8 / 7.4 / 0.24 | 116.7 / 36.8 / 0.60 | 168.5 / 51.0 / 0.42 | 181.0 / 94.9 / 0.40 | 237.3 / 144.8 / 0.33 | 244.0 / 563.2 / 0.42 | 513.7 / 1070.2 / 0.40 |
| dequant_triton+hipblaslt w4 sym | 62.1 / 2.2 / 0.56 | 64.8 / 8.3 / 0.54 | 64.5 / 33.3 / 0.54 | 64.6 / 66.5 / 0.55 | 69.7 / 123.2 / 0.53 | 78.7 / 218.3 / 0.49 | 106.2 / 323.5 / 0.41 | 176.3 / 779.8 / 0.39 | 395.9 / 1388.5 / 0.43 |
| dequant_triton+hipblaslt w4 zp | 66.0 / 2.0 / 0.53 | 64.2 / 8.4 / 0.55 | 65.9 / 32.6 / 0.54 | 68.1 / 63.1 / 0.53 | 71.4 / 120.3 / 0.52 | 82.0 / 209.5 / 0.48 | 106.4 / 322.9 / 0.41 | 180.8 / 760.1 / 0.38 | 401.2 / 1370.3 / 0.42 |
| dequant_triton+hipblaslt w8 sym | 65.8 / 2.0 / 1.04 | 67.3 / 8.0 / 1.01 | 68.9 / 31.1 / 1.00 | 73.3 / 58.6 / 0.94 | 75.2 / 114.2 / 0.93 | 87.5 / 196.3 / 0.83 | 114.4 / 300.4 / 0.67 | 185.8 / 739.5 / 0.55 | 400.1 / 1374.1 / 0.51 |
| dequant_triton+hipblaslt w8 zp | 64.3 / 2.1 / 1.07 | 65.1 / 8.2 / 1.06 | 66.9 / 32.1 / 1.03 | 68.9 / 62.4 / 1.01 | 72.4 / 118.7 / 0.98 | 82.9 / 207.4 / 0.88 | 106.5 / 322.5 / 0.72 | 180.2 / 762.8 / 0.57 | 398.3 / 1380.2 / 0.51 |
| aiter_a16w8_int8 w8 sym | 18.5 / 7.2 / 3.74 | 19.0 / 28.3 / 3.65 | 23.6 / 91.2 / 2.96 | 23.5 / 182.8 / 2.99 | 30.9 / 277.6 / 2.30 | 47.0 / 365.6 / 1.56 | 78.3 / 439.0 / 0.99 | 251.1 / 547.3 / 0.41 | 948.1 / 579.8 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 15.4 / 8.7 / 4.35 | 16.3 / 32.9 / 4.12 | 17.8 / 120.7 / 3.80 | 22.1 / 194.4 / 3.09 | 27.0 / 318.3 / 2.56 | 43.0 / 399.4 / 1.66 | 75.2 / 456.7 / 1.00 | 199.3 / 689.6 / 0.51 | 752.4 / 730.6 / 0.27 |
| humming w4 sym | 12.5 / 10.8 / 2.78 | 13.3 / 40.4 / 2.61 | 13.4 / 160.2 / 2.62 | 15.2 / 283.4 / 2.35 | 19.8 / 433.7 / 1.85 | 30.6 / 560.7 / 1.27 | 46.1 / 745.9 / 0.93 | 127.9 / 1074.4 / 0.53 | 435.4 / 1262.8 / 0.39 |
| humming w4 zp | 13.3 / 10.1 / 2.64 | 14.2 / 37.9 / 2.49 | 13.7 / 156.3 / 2.60 | 15.6 / 276.1 / 2.33 | 20.5 / 419.9 / 1.82 | 30.9 / 555.7 / 1.27 | 46.7 / 735.2 / 0.93 | 131.3 / 1047.1 / 0.52 | 451.7 / 1217.0 / 0.38 |
| humming w8 sym | 17.1 / 7.8 / 3.98 | 17.8 / 30.1 / 3.83 | 17.8 / 120.3 / 3.85 | 18.1 / 237.0 / 3.82 | 21.8 / 394.0 / 3.22 | 34.0 / 505.1 / 2.13 | 50.5 / 680.0 / 1.51 | 138.2 / 994.4 / 0.74 | 514.3 / 1068.9 / 0.39 |
| humming w8 zp | 17.2 / 7.8 / 3.99 | 18.0 / 29.8 / 3.82 | 17.9 / 119.9 / 3.87 | 18.3 / 234.3 / 3.80 | 22.4 / 384.3 / 3.17 | 34.6 / 496.7 / 2.11 | 51.6 / 666.0 / 1.49 | 140.7 / 976.8 / 0.73 | 516.1 / 1065.3 / 0.39 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 157.5 / 6.0 / 5.97 | 163.2 / 23.0 / 5.76 | 155.4 / 96.7 / 6.06 | 165.3 / 181.9 / 5.71 | 156.9 / 383.3 / 6.04 | 182.5 / 658.9 / 5.24 | 215.2 / 1117.8 / 4.52 | 743.6 / 1293.8 / 1.44 | 2659.0 / 1447.3 / 0.56 |
| vllm_triton_w4a16 w4 sym | 196.1 / 4.8 / 1.24 | 200.0 / 18.8 / 1.21 | 223.7 / 67.2 / 1.09 | 227.5 / 132.1 / 1.08 | 217.7 / 276.2 / 1.15 | 259.3 / 463.7 / 1.00 | 491.3 / 489.6 / 0.56 | 1677.5 / 573.5 / 0.22 | 6488.6 / 593.1 / 0.12 |
| vllm_triton_w4a16 w4 zp | 214.4 / 4.4 / 1.14 | 219.7 / 17.1 / 1.11 | 239.1 / 62.9 / 1.03 | 246.8 / 121.8 / 1.01 | 237.8 / 252.8 / 1.06 | 272.5 / 441.3 / 0.96 | 522.3 / 460.5 / 0.53 | 1780.4 / 540.4 / 0.21 | 6857.1 / 561.2 / 0.11 |
| vllm_exllama_gptq w4 sym | 252.7 / 3.7 / 0.96 | 273.1 / 13.8 / 0.89 | 348.5 / 43.1 / 0.70 | 594.2 / 50.6 / 0.41 | 447.3 / 134.4 / 0.56 | 452.4 / 265.8 / 0.57 | 479.5 / 501.6 / 0.57 | 1059.0 / 908.5 / 0.35 | 3302.0 / 1165.4 / 0.24 |
| vllm_exllama_gptq w4 zp | 250.4 / 3.8 / 0.97 | 273.3 / 13.8 / 0.90 | 348.0 / 43.2 / 0.71 | 594.6 / 50.6 / 0.42 | 450.4 / 133.5 / 0.56 | 455.7 / 263.9 / 0.57 | 484.9 / 496.0 / 0.57 | 1052.3 / 914.2 / 0.36 | 3303.4 / 1165.0 / 0.24 |
| vllm_exllama_gptq w8 sym | 249.6 / 3.8 / 1.91 | 241.2 / 15.6 / 1.98 | 431.8 / 34.8 / 1.11 | 480.8 / 62.5 / 1.00 | 481.7 / 124.8 / 1.01 | 487.0 / 246.9 / 1.01 | 516.4 / 465.8 / 0.99 | 1109.1 / 867.5 / 0.55 | 3362.2 / 1144.6 / 0.30 |
| vllm_exllama_gptq w8 zp | 250.5 / 3.8 / 1.92 | 242.1 / 15.5 / 1.99 | 432.8 / 34.7 / 1.12 | 480.9 / 62.5 / 1.01 | 481.8 / 124.8 / 1.01 | 487.8 / 246.5 / 1.02 | 517.2 / 465.0 / 0.99 | 1107.2 / 868.9 / 0.56 | 3363.4 / 1144.2 / 0.30 |
| dequant_triton+hipblaslt w4 sym | 427.9 / 2.2 / 0.57 | 449.1 / 8.4 / 0.54 | 425.1 / 35.4 / 0.57 | 452.2 / 66.5 / 0.55 | 434.5 / 138.4 / 0.58 | 467.9 / 257.0 / 0.55 | 479.1 / 502.1 / 0.58 | 1006.9 / 955.5 / 0.37 | 2886.4 / 1333.2 / 0.27 |
| dequant_triton+hipblaslt w4 zp | 454.8 / 2.1 / 0.54 | 464.7 / 8.1 / 0.53 | 439.9 / 34.2 / 0.56 | 465.4 / 64.6 / 0.53 | 445.7 / 134.9 / 0.57 | 481.7 / 249.7 / 0.54 | 513.2 / 468.7 / 0.54 | 1022.1 / 941.3 / 0.37 | 2906.6 / 1324.0 / 0.27 |
| dequant_triton+hipblaslt w8 sym | 472.1 / 2.0 / 1.01 | 465.5 / 8.1 / 1.03 | 471.2 / 31.9 / 1.02 | 469.6 / 64.0 / 1.02 | 473.9 / 126.9 / 1.02 | 483.8 / 248.5 / 1.02 | 541.1 / 444.5 / 0.94 | 1037.1 / 927.7 / 0.59 | 2943.7 / 1307.3 / 0.34 |
| dequant_triton+hipblaslt w8 zp | 468.0 / 2.0 / 1.03 | 479.6 / 7.8 / 1.00 | 467.6 / 32.1 / 1.03 | 479.8 / 62.7 / 1.01 | 473.5 / 127.0 / 1.03 | 495.4 / 242.8 / 1.00 | 532.2 / 451.9 / 0.97 | 1035.0 / 929.5 / 0.59 | 2955.3 / 1302.2 / 0.34 |
| aiter_a16w8_int8 w8 sym | 97.1 / 9.7 / 4.99 | 111.4 / 33.7 / 4.35 | 129.1 / 116.4 / 3.77 | 139.5 / 215.6 / 3.50 | 174.0 / 345.5 / 2.83 | 281.7 / 426.9 / 1.78 | 552.6 / 435.2 / 0.94 | 1727.4 / 557.0 / 0.36 | 6621.0 / 581.2 / 0.15 |
| aiter_a16w8_fp8_block128 w8 sym | 77.0 / 12.2 / 6.10 | 85.1 / 44.2 / 5.53 | 104.1 / 144.4 / 4.53 | 134.1 / 224.2 / 3.54 | 162.2 / 370.8 / 2.95 | 243.0 / 494.8 / 2.00 | 504.2 / 477.0 / 1.00 | 1370.6 / 701.9 / 0.44 | 5266.2 / 730.8 / 0.19 |
| humming w4 sym | 50.2 / 18.7 / 4.83 | 51.6 / 72.8 / 4.70 | 50.6 / 297.1 / 4.83 | 60.0 / 501.4 / 4.11 | 82.1 / 732.7 / 3.05 | 131.0 / 917.8 / 1.98 | 222.3 / 1081.7 / 1.24 | 770.2 / 1249.2 / 0.49 | 2998.9 / 1283.2 / 0.26 |
| humming w4 zp | 52.8 / 17.8 / 4.66 | 53.5 / 70.2 / 4.60 | 52.6 / 285.5 / 4.71 | 62.5 / 481.3 / 4.00 | 85.2 / 706.1 / 2.99 | 132.3 / 908.7 / 1.99 | 231.4 / 1039.3 / 1.21 | 801.5 / 1200.3 / 0.47 | 3119.8 / 1233.5 / 0.25 |
| humming w8 sym | 87.1 / 10.8 / 5.48 | 85.6 / 43.9 / 5.58 | 89.0 / 168.9 / 5.38 | 92.2 / 326.1 / 5.22 | 108.9 / 551.9 / 4.46 | 152.0 / 791.2 / 3.25 | 292.1 / 823.4 / 1.75 | 1095.0 / 878.6 / 0.56 | 4403.2 / 874.0 / 0.23 |
| humming w8 zp | 88.4 / 10.6 / 5.44 | 85.9 / 43.8 / 5.61 | 90.7 / 165.7 / 5.32 | 93.1 / 323.0 / 5.21 | 111.4 / 539.7 / 4.39 | 154.9 / 776.3 / 3.21 | 296.4 / 811.5 / 1.74 | 1088.9 / 883.5 / 0.56 | 4511.9 / 852.9 / 0.23 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.8 / 6.2 / 6.19 | 76.0 / 24.7 / 6.18 | 77.4 / 97.2 / 6.09 | 79.1 / 190.0 / 5.97 | 84.8 / 354.6 / 5.60 | 95.1 / 632.2 / 5.04 | 124.0 / 970.1 / 3.94 | 362.7 / 1326.3 / 1.50 | 1274.1 / 1510.2 / 0.61 |
| vllm_triton_w4a16 w4 sym | 499.9 / 0.9 / 0.24 | 503.1 / 3.7 / 0.24 | 505.5 / 14.9 / 0.24 | 510.7 / 29.4 / 0.24 | 516.8 / 58.2 / 0.24 | 707.9 / 84.9 / 0.18 | 700.5 / 171.7 / 0.20 | 853.7 / 563.5 / 0.23 | 3295.3 / 583.9 / 0.13 |
| vllm_triton_w4a16 w4 zp | 543.3 / 0.9 / 0.23 | 542.6 / 3.5 / 0.23 | 544.7 / 13.8 / 0.23 | 548.3 / 27.4 / 0.23 | 568.0 / 52.9 / 0.22 | 758.2 / 79.3 / 0.17 | 751.0 / 160.1 / 0.19 | 912.2 / 527.3 / 0.22 | 3501.8 / 549.5 / 0.12 |
| vllm_exllama_gptq w4 sym | 1308.9 / 0.4 / 0.09 | 1342.8 / 1.4 / 0.09 | 1140.0 / 6.6 / 0.11 | 752.0 / 20.0 / 0.16 | 514.3 / 58.5 / 0.24 | 562.9 / 106.8 / 0.23 | 715.3 / 168.1 / 0.20 | 723.4 / 665.0 / 0.27 | 1605.9 / 1198.2 / 0.26 |
| vllm_exllama_gptq w4 zp | 1305.9 / 0.4 / 0.09 | 1344.4 / 1.4 / 0.09 | 1068.6 / 7.0 / 0.12 | 756.1 / 19.9 / 0.17 | 515.5 / 58.3 / 0.25 | 563.4 / 106.7 / 0.23 | 718.5 / 167.4 / 0.20 | 723.7 / 664.7 / 0.27 | 1610.8 / 1194.5 / 0.26 |
| vllm_exllama_gptq w8 sym | 1271.6 / 0.4 / 0.19 | 965.5 / 1.9 / 0.25 | 824.2 / 9.1 / 0.29 | 496.9 / 30.3 / 0.48 | 520.4 / 57.8 / 0.47 | 570.0 / 105.5 / 0.43 | 724.4 / 166.0 / 0.35 | 732.0 / 657.2 / 0.43 | 1628.8 / 1181.3 / 0.33 |
| vllm_exllama_gptq w8 zp | 1253.8 / 0.4 / 0.19 | 969.0 / 1.9 / 0.25 | 822.6 / 9.1 / 0.29 | 497.4 / 30.2 / 0.49 | 521.2 / 57.7 / 0.47 | 570.9 / 105.3 / 0.44 | 726.5 / 165.5 / 0.36 | 732.6 / 656.6 / 0.43 | 1632.6 / 1178.5 / 0.33 |
| dequant_triton+hipblaslt w4 sym | 212.9 / 2.2 / 0.57 | 207.2 / 9.1 / 0.59 | 220.0 / 34.2 / 0.56 | 209.7 / 71.7 / 0.59 | 228.0 / 131.8 / 0.55 | 223.4 / 269.1 / 0.58 | 251.5 / 478.1 / 0.56 | 498.7 / 964.6 / 0.39 | 1388.5 / 1385.8 / 0.30 |
| dequant_triton+hipblaslt w4 zp | 229.8 / 2.0 / 0.53 | 227.9 / 8.2 / 0.54 | 235.9 / 31.9 / 0.52 | 230.9 / 65.1 / 0.54 | 244.9 / 122.8 / 0.52 | 240.0 / 250.6 / 0.55 | 274.4 / 438.3 / 0.51 | 506.6 / 949.5 / 0.39 | 1404.6 / 1369.9 / 0.30 |
| dequant_triton+hipblaslt w8 sym | 241.7 / 1.9 / 0.99 | 240.8 / 7.8 / 0.99 | 238.8 / 31.5 / 1.00 | 243.4 / 61.8 / 0.99 | 241.7 / 124.4 / 1.01 | 250.3 / 240.3 / 0.99 | 275.6 / 436.4 / 0.93 | 502.8 / 956.8 / 0.62 | 1410.3 / 1364.3 / 0.38 |
| dequant_triton+hipblaslt w8 zp | 249.8 / 1.9 / 0.96 | 247.6 / 7.6 / 0.97 | 251.5 / 29.9 / 0.96 | 254.0 / 59.2 / 0.96 | 254.5 / 118.1 / 0.96 | 260.9 / 230.5 / 0.96 | 290.0 / 414.7 / 0.89 | 517.0 / 930.5 / 0.61 | 1423.7 / 1351.5 / 0.38 |
| aiter_a16w8_int8 w8 sym | 48.5 / 9.7 / 5.00 | 50.5 / 37.2 / 4.80 | 60.0 / 125.3 / 4.06 | 66.5 / 226.1 / 3.68 | 82.0 / 366.5 / 3.01 | 129.6 / 463.8 / 1.94 | 261.5 / 459.9 / 1.00 | 867.8 / 554.3 / 0.37 | 3312.8 / 580.8 / 0.16 |
| aiter_a16w8_fp8_block128 w8 sym | 42.5 / 11.1 / 5.53 | 44.2 / 42.5 / 5.32 | 47.4 / 158.7 / 4.99 | 67.5 / 222.6 / 3.52 | 75.8 / 396.6 / 3.16 | 112.1 / 536.4 / 2.18 | 253.7 / 474.1 / 1.00 | 677.4 / 710.1 / 0.46 | 2597.2 / 740.9 / 0.21 |
| humming w4 sym | 36.0 / 13.1 / 3.37 | 34.3 / 54.7 / 3.54 | 30.7 / 244.5 / 3.98 | 36.7 / 409.2 / 3.36 | 51.1 / 588.3 / 2.46 | 79.9 / 752.2 / 1.63 | 138.7 / 866.8 / 1.01 | 432.1 / 1113.3 / 0.46 | 1482.0 / 1298.3 / 0.28 |
| humming w4 zp | 38.0 / 12.4 / 3.23 | 36.0 / 52.3 / 3.43 | 31.9 / 235.6 / 3.89 | 38.6 / 389.0 / 3.24 | 52.5 / 572.6 / 2.43 | 80.4 / 748.0 / 1.65 | 138.4 / 869.1 / 1.02 | 442.1 / 1088.0 / 0.45 | 1536.8 / 1252.0 / 0.28 |
| humming w8 sym | 43.8 / 10.7 / 5.44 | 45.1 / 41.6 / 5.29 | 44.5 / 168.8 / 5.38 | 47.1 / 319.0 / 5.11 | 58.2 / 516.6 / 4.18 | 94.0 / 640.0 / 2.64 | 157.7 / 762.6 / 1.63 | 458.8 / 1048.6 / 0.69 | 1785.9 / 1077.4 / 0.30 |
| humming w8 zp | 47.2 / 9.9 / 5.09 | 46.0 / 40.9 / 5.24 | 44.8 / 167.7 / 5.39 | 47.4 / 316.8 / 5.12 | 59.1 / 509.1 / 4.15 | 95.5 / 629.3 / 2.62 | 159.0 / 756.3 / 1.63 | 463.5 / 1037.7 / 0.68 | 1795.2 / 1071.8 / 0.30 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 15.5 / 3.4 / 3.38 | 15.8 / 13.3 / 3.33 | 15.4 / 54.6 / 3.43 | 16.8 / 99.7 / 3.16 | 23.0 / 146.0 / 2.34 | 25.1 / 267.7 / 2.20 | 36.1 / 372.0 / 1.60 | 64.7 / 829.8 / 1.14 | 183.7 / 1169.1 / 0.74 |
| vllm_triton_w4a16 w4 sym | 92.9 / 0.6 / 0.15 | 93.2 / 2.2 / 0.15 | 93.6 / 9.0 / 0.15 | 94.4 / 17.8 / 0.15 | 95.6 / 35.1 / 0.15 | 130.4 / 51.4 / 0.12 | 129.3 / 103.8 / 0.14 | 150.5 / 356.7 / 0.23 | 403.6 / 532.1 / 0.24 |
| vllm_triton_w4a16 w4 zp | 99.6 / 0.5 / 0.14 | 100.1 / 2.1 / 0.14 | 100.3 / 8.4 / 0.14 | 100.8 / 16.6 / 0.14 | 104.8 / 32.0 / 0.14 | 139.5 / 48.1 / 0.12 | 137.9 / 97.3 / 0.14 | 161.8 / 331.7 / 0.21 | 427.3 / 502.6 / 0.23 |
| vllm_exllama_gptq w4 sym | 51.8 / 1.0 / 0.26 | 73.7 / 2.8 / 0.18 | 115.3 / 7.3 / 0.12 | 171.3 / 9.8 / 0.08 | 80.7 / 41.6 / 0.18 | 77.3 / 86.9 / 0.21 | 71.4 / 187.9 / 0.26 | 106.9 / 502.2 / 0.32 | 224.5 / 956.5 / 0.43 |
| vllm_exllama_gptq w4 zp | 51.9 / 1.0 / 0.26 | 73.7 / 2.8 / 0.19 | 115.6 / 7.3 / 0.12 | 171.1 / 9.8 / 0.08 | 80.8 / 41.5 / 0.18 | 77.3 / 86.8 / 0.21 | 71.5 / 187.7 / 0.26 | 107.1 / 501.4 / 0.32 | 224.4 / 957.2 / 0.43 |
| vllm_exllama_gptq w8 sym | 55.8 / 0.9 / 0.48 | 74.4 / 2.8 / 0.36 | 120.6 / 7.0 / 0.22 | 50.5 / 33.2 / 0.54 | 82.6 / 40.6 / 0.34 | 79.0 / 84.9 / 0.37 | 73.6 / 182.4 / 0.43 | 109.6 / 489.8 / 0.43 | 228.3 / 940.5 / 0.48 |
| vllm_exllama_gptq w8 zp | 55.9 / 0.9 / 0.48 | 74.4 / 2.8 / 0.36 | 120.6 / 7.0 / 0.23 | 50.6 / 33.1 / 0.54 | 82.7 / 40.6 / 0.34 | 79.3 / 84.7 / 0.37 | 73.4 / 182.9 / 0.44 | 109.8 / 489.1 / 0.43 | 227.0 / 946.2 / 0.49 |
| dequant_triton+hipblaslt w4 sym | 31.5 / 1.7 / 0.43 | 31.9 / 6.6 / 0.43 | 31.2 / 26.9 / 0.44 | 32.8 / 51.2 / 0.43 | 37.8 / 88.7 / 0.39 | 39.0 / 172.0 / 0.41 | 49.6 / 270.6 / 0.38 | 71.8 / 748.0 / 0.48 | 194.1 / 1106.5 / 0.50 |
| dequant_triton+hipblaslt w4 zp | 32.0 / 1.6 / 0.43 | 32.3 / 6.5 / 0.42 | 31.5 / 26.6 / 0.44 | 32.9 / 50.9 / 0.43 | 38.2 / 88.0 / 0.39 | 39.3 / 171.0 / 0.41 | 50.2 / 267.1 / 0.38 | 72.2 / 743.4 / 0.48 | 194.5 / 1104.0 / 0.50 |
| dequant_triton+hipblaslt w8 sym | 33.0 / 1.6 / 0.81 | 33.3 / 6.3 / 0.80 | 32.1 / 26.2 / 0.84 | 34.0 / 49.3 / 0.80 | 39.1 / 85.9 / 0.71 | 40.4 / 166.0 / 0.72 | 50.8 / 264.2 / 0.63 | 72.1 / 745.0 / 0.66 | 195.3 / 1099.7 / 0.57 |
| dequant_triton+hipblaslt w8 zp | 32.9 / 1.6 / 0.81 | 33.2 / 6.3 / 0.81 | 31.9 / 26.3 / 0.85 | 33.9 / 49.5 / 0.81 | 39.1 / 85.9 / 0.72 | 40.3 / 166.4 / 0.73 | 50.9 / 263.5 / 0.63 | 72.7 / 738.3 / 0.66 | 195.6 / 1097.9 / 0.57 |
| aiter_a16w8_int8 w8 sym | 12.1 / 4.3 / 2.23 | 12.8 / 16.3 / 2.11 | 13.9 / 60.5 / 1.97 | 16.7 / 100.7 / 1.66 | 20.6 / 162.9 / 1.38 | 33.0 / 203.5 / 0.90 | 47.3 / 283.7 / 0.68 | 163.5 / 328.4 / 0.29 | 392.1 / 547.6 / 0.28 |
| aiter_a16w8_fp8_block128 w8 sym | 10.6 / 5.0 / 2.48 | 11.0 / 19.1 / 2.40 | 11.9 / 70.2 / 2.22 | 15.7 / 106.8 / 1.71 | 19.5 / 172.3 / 1.41 | 29.0 / 231.2 / 0.99 | 47.0 / 285.5 / 0.67 | 122.2 / 439.5 / 0.39 | 305.9 / 702.0 / 0.36 |
| humming w4 sym | 9.4 / 5.6 / 1.44 | 10.1 / 20.7 / 1.34 | 10.0 / 84.2 / 1.39 | 10.2 / 165.0 / 1.39 | 13.7 / 244.3 / 1.08 | 24.2 / 277.7 / 0.67 | 30.8 / 436.1 / 0.61 | 69.5 / 772.9 / 0.50 | 198.4 / 1082.4 / 0.49 |
| humming w4 zp | 9.9 / 5.3 / 1.38 | 10.6 / 19.7 / 1.30 | 10.1 / 83.2 / 1.39 | 10.2 / 164.0 / 1.41 | 14.2 / 236.8 / 1.06 | 24.3 / 275.9 / 0.67 | 30.8 / 435.7 / 0.62 | 71.1 / 755.2 / 0.49 | 205.4 / 1045.6 / 0.47 |
| humming w8 sym | 11.3 / 4.6 / 2.36 | 11.5 / 18.2 / 2.32 | 11.9 / 70.3 / 2.26 | 12.1 / 138.4 / 2.25 | 16.8 / 200.0 / 1.67 | 26.4 / 253.7 / 1.11 | 33.6 / 399.9 / 0.95 | 74.7 / 719.0 / 0.64 | 212.3 / 1011.7 / 0.52 |
| humming w8 zp | 11.3 / 4.7 / 2.39 | 11.8 / 17.8 / 2.28 | 12.1 / 69.1 / 2.24 | 12.2 / 137.3 / 2.25 | 17.0 / 196.9 / 1.65 | 26.4 / 254.4 / 1.12 | 33.6 / 398.9 / 0.95 | 75.0 / 715.5 / 0.64 | 214.7 / 1000.3 / 0.52 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 79.2 / 6.6 / 6.62 | 83.3 / 25.2 / 6.30 | 84.6 / 99.1 / 6.22 | 86.9 / 193.2 / 6.08 | 86.9 / 386.2 / 6.12 | 108.3 / 619.7 / 4.97 | 138.0 / 972.9 / 4.01 | 514.7 / 1043.0 / 1.24 | 1547.4 / 1387.8 / 0.64 |
| vllm_triton_w4a16 w4 sym | 119.7 / 4.4 / 1.13 | 121.9 / 17.2 / 1.11 | 131.1 / 64.0 / 1.04 | 133.7 / 125.5 / 1.04 | 134.9 / 248.7 / 1.05 | 169.7 / 395.5 / 0.88 | 305.7 / 439.0 / 0.54 | 980.4 / 547.6 / 0.26 | 3660.3 / 586.7 / 0.16 |
| vllm_triton_w4a16 w4 zp | 132.3 / 4.0 / 1.03 | 134.0 / 15.6 / 1.02 | 142.8 / 58.8 / 0.97 | 146.1 / 114.8 / 0.96 | 148.8 / 225.5 / 0.96 | 172.2 / 389.7 / 0.88 | 325.3 / 412.6 / 0.51 | 1033.6 / 519.4 / 0.24 | 3860.4 / 556.3 / 0.15 |
| vllm_exllama_gptq w4 sym | 133.8 / 3.9 / 1.01 | 166.8 / 12.6 / 0.81 | 227.7 / 36.8 / 0.60 | 351.7 / 47.7 / 0.40 | 261.5 / 128.3 / 0.54 | 267.8 / 250.6 / 0.56 | 284.1 / 472.5 / 0.58 | 651.7 / 823.9 / 0.38 | 1921.1 / 1117.8 / 0.31 |
| vllm_exllama_gptq w4 zp | 133.4 / 3.9 / 1.02 | 167.2 / 12.5 / 0.82 | 227.9 / 36.8 / 0.61 | 351.7 / 47.7 / 0.40 | 262.1 / 128.0 / 0.55 | 267.9 / 250.5 / 0.56 | 283.6 / 473.3 / 0.58 | 658.7 / 815.0 / 0.38 | 1925.6 / 1115.3 / 0.31 |
| vllm_exllama_gptq w8 sym | 146.0 / 3.6 / 1.82 | 148.8 / 14.1 / 1.79 | 263.1 / 31.9 / 1.02 | 280.1 / 59.9 / 0.96 | 281.3 / 119.3 / 0.97 | 287.8 / 233.2 / 0.97 | 304.2 / 441.2 / 0.97 | 687.3 / 781.1 / 0.56 | 1960.7 / 1095.3 / 0.37 |
| vllm_exllama_gptq w8 zp | 145.1 / 3.6 / 1.85 | 148.5 / 14.1 / 1.81 | 263.6 / 31.8 / 1.02 | 279.5 / 60.0 / 0.97 | 281.5 / 119.2 / 0.98 | 287.3 / 233.6 / 0.98 | 304.2 / 441.2 / 0.98 | 685.8 / 782.8 / 0.56 | 1962.6 / 1094.2 / 0.37 |
| dequant_triton+hipblaslt w4 sym | 219.7 / 2.4 / 0.62 | 219.5 / 9.6 / 0.62 | 220.2 / 38.1 / 0.62 | 222.9 / 75.3 / 0.62 | 223.1 / 150.4 / 0.64 | 240.5 / 279.0 / 0.62 | 262.9 / 510.6 / 0.62 | 646.5 / 830.4 / 0.39 | 1675.2 / 1281.9 / 0.36 |
| dequant_triton+hipblaslt w4 zp | 227.3 / 2.3 / 0.60 | 226.7 / 9.3 / 0.60 | 227.8 / 36.8 / 0.61 | 230.1 / 72.9 / 0.61 | 230.7 / 145.5 / 0.62 | 248.5 / 270.1 / 0.61 | 269.9 / 497.3 / 0.61 | 652.8 / 822.4 / 0.39 | 1687.5 / 1272.6 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 235.0 / 2.2 / 1.13 | 233.1 / 9.0 / 1.14 | 234.6 / 35.8 / 1.14 | 237.2 / 70.7 / 1.14 | 238.4 / 140.8 / 1.15 | 256.7 / 261.4 / 1.09 | 279.4 / 480.3 / 1.06 | 665.5 / 806.7 / 0.57 | 1700.2 / 1263.1 / 0.43 |
| dequant_triton+hipblaslt w8 zp | 238.9 / 2.2 / 1.12 | 237.9 / 8.8 / 1.13 | 239.0 / 35.1 / 1.13 | 241.2 / 69.6 / 1.13 | 242.1 / 138.6 / 1.14 | 260.8 / 257.3 / 1.08 | 283.5 / 473.5 / 1.05 | 667.1 / 804.8 / 0.57 | 1710.8 / 1255.3 / 0.43 |
| aiter_a16w8_int8 w8 sym | 60.1 / 8.7 / 4.50 | 69.4 / 30.2 / 3.90 | 82.0 / 102.4 / 3.32 | 92.9 / 180.6 / 2.95 | 117.8 / 284.7 / 2.35 | 190.9 / 351.5 / 1.49 | 320.0 / 419.4 / 0.94 | 1059.0 / 507.0 / 0.36 | 3895.3 / 551.3 / 0.19 |
| aiter_a16w8_fp8_block128 w8 sym | 48.7 / 10.8 / 5.39 | 54.9 / 38.2 / 4.78 | 69.7 / 120.4 / 3.79 | 87.0 / 192.8 / 3.06 | 109.9 / 305.2 / 2.45 | 162.9 / 412.0 / 1.70 | 318.0 / 422.1 / 0.92 | 813.7 / 659.8 / 0.46 | 3027.2 / 709.4 / 0.24 |
| humming w4 sym | 31.3 / 16.7 / 4.32 | 33.4 / 62.8 / 4.06 | 30.4 / 275.6 / 4.50 | 37.5 / 447.8 / 3.70 | 51.8 / 648.2 / 2.75 | 82.4 / 814.2 / 1.81 | 140.2 / 957.3 / 1.17 | 464.6 / 1155.5 / 0.54 | 1691.5 / 1269.6 / 0.35 |
| humming w4 zp | 33.0 / 15.9 / 4.16 | 34.6 / 60.6 / 3.98 | 32.5 / 257.8 / 4.27 | 39.5 / 424.9 / 3.57 | 54.0 / 621.3 / 2.67 | 84.6 / 793.4 / 1.79 | 138.4 / 969.6 / 1.20 | 482.2 / 1113.5 / 0.52 | 1753.3 / 1224.8 / 0.34 |
| humming w8 sym | 50.5 / 10.4 / 5.28 | 53.5 / 39.2 / 4.98 | 52.6 / 159.4 / 5.09 | 56.1 / 299.2 / 4.81 | 66.7 / 502.7 / 4.10 | 96.0 / 698.8 / 2.92 | 170.9 / 785.6 / 1.73 | 574.9 / 933.9 / 0.66 | 2192.4 / 979.5 / 0.33 |
| humming w8 zp | 54.6 / 9.6 / 4.91 | 54.0 / 38.8 / 4.98 | 53.7 / 156.3 / 5.03 | 56.8 / 295.6 / 4.79 | 68.1 / 492.8 / 4.05 | 98.6 / 680.8 / 2.87 | 172.7 / 777.2 / 1.72 | 580.9 / 924.2 / 0.66 | 2212.0 / 970.8 / 0.33 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 45.5 / 5.8 / 5.76 | 45.9 / 22.9 / 5.72 | 47.5 / 88.3 / 5.54 | 50.2 / 167.0 / 5.26 | 56.5 / 296.9 / 4.71 | 59.4 / 565.2 / 4.55 | 92.4 / 726.5 / 3.01 | 235.7 / 1138.9 / 1.38 | 855.7 / 1254.8 / 0.60 |
| vllm_triton_w4a16 w4 sym | 453.1 / 0.6 / 0.15 | 453.8 / 2.3 / 0.15 | 455.0 / 9.2 / 0.15 | 459.1 / 18.3 / 0.15 | 464.0 / 36.2 / 0.15 | 631.8 / 53.1 / 0.12 | 627.2 / 107.0 / 0.13 | 726.1 / 369.7 / 0.18 | 1964.6 / 546.6 / 0.16 |
| vllm_triton_w4a16 w4 zp | 486.0 / 0.5 / 0.14 | 487.1 / 2.2 / 0.14 | 488.6 / 8.6 / 0.14 | 491.5 / 17.1 / 0.14 | 511.1 / 32.8 / 0.14 | 678.9 / 49.4 / 0.11 | 673.1 / 99.7 / 0.12 | 781.1 / 343.7 / 0.17 | 2094.6 / 512.6 / 0.15 |
| vllm_exllama_gptq w4 sym | 499.0 / 0.5 / 0.14 | 553.7 / 1.9 / 0.12 | 657.7 / 6.4 / 0.10 | 692.4 / 12.1 / 0.10 | 382.1 / 43.9 / 0.19 | 460.4 / 72.9 / 0.16 | 592.4 / 113.3 / 0.14 | 537.0 / 499.9 / 0.24 | 1142.7 / 939.6 / 0.28 |
| vllm_exllama_gptq w4 zp | 500.4 / 0.5 / 0.14 | 555.1 / 1.9 / 0.12 | 657.2 / 6.4 / 0.10 | 696.7 / 12.0 / 0.10 | 384.4 / 43.6 / 0.19 | 461.0 / 72.8 / 0.17 | 592.1 / 113.3 / 0.14 | 536.9 / 500.0 / 0.24 | 1150.3 / 933.5 / 0.28 |
| vllm_exllama_gptq w8 sym | 505.1 / 0.5 / 0.26 | 528.9 / 2.0 / 0.25 | 502.7 / 8.3 / 0.27 | 373.5 / 22.5 / 0.36 | 407.6 / 41.2 / 0.34 | 497.7 / 67.4 / 0.28 | 624.0 / 107.5 / 0.24 | 556.4 / 482.5 / 0.35 | 1161.6 / 924.4 / 0.33 |
| vllm_exllama_gptq w8 zp | 508.0 / 0.5 / 0.26 | 528.1 / 2.0 / 0.25 | 501.4 / 8.4 / 0.27 | 374.3 / 22.4 / 0.36 | 408.1 / 41.1 / 0.34 | 498.2 / 67.3 / 0.28 | 624.5 / 107.5 / 0.24 | 555.8 / 482.9 / 0.35 | 1167.9 / 919.4 / 0.33 |
| dequant_triton+hipblaslt w4 sym | 118.5 / 2.2 / 0.57 | 118.6 / 8.8 / 0.57 | 119.8 / 35.0 / 0.57 | 122.0 / 68.8 / 0.57 | 126.1 / 133.1 / 0.57 | 135.1 / 248.5 / 0.56 | 171.8 / 390.6 / 0.48 | 303.3 / 884.9 / 0.43 | 926.8 / 1158.5 / 0.34 |
| dequant_triton+hipblaslt w4 zp | 126.7 / 2.1 / 0.54 | 126.9 / 8.3 / 0.54 | 128.1 / 32.7 / 0.54 | 130.3 / 64.4 / 0.54 | 134.9 / 124.4 / 0.53 | 142.8 / 234.9 / 0.53 | 180.0 / 372.9 / 0.47 | 311.8 / 861.0 / 0.42 | 936.4 / 1146.6 / 0.34 |
| dequant_triton+hipblaslt w8 sym | 118.6 / 2.2 / 1.12 | 118.9 / 8.8 / 1.12 | 120.5 / 34.8 / 1.11 | 123.3 / 68.0 / 1.09 | 128.4 / 130.6 / 1.07 | 135.0 / 248.6 / 1.04 | 172.5 / 389.1 / 0.86 | 306.7 / 875.2 / 0.64 | 931.8 / 1152.4 / 0.41 |
| dequant_triton+hipblaslt w8 zp | 124.1 / 2.1 / 1.08 | 124.6 / 8.4 / 1.08 | 126.2 / 33.2 / 1.07 | 128.8 / 65.2 / 1.06 | 133.7 / 125.5 / 1.03 | 140.9 / 238.2 / 1.01 | 178.9 / 375.2 / 0.84 | 312.7 / 858.3 / 0.63 | 940.4 / 1141.8 / 0.41 |
| aiter_a16w8_int8 w8 sym | 37.5 / 7.0 / 3.60 | 38.4 / 27.3 / 3.52 | 40.0 / 104.8 / 3.40 | 54.5 / 153.9 / 2.52 | 64.7 / 259.3 / 2.15 | 110.8 / 302.8 / 1.29 | 214.5 / 312.8 / 0.70 | 767.4 / 349.8 / 0.26 | 1852.4 / 579.6 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 29.6 / 8.8 / 4.43 | 30.6 / 34.3 / 4.30 | 32.4 / 129.3 / 4.07 | 52.9 / 158.4 / 2.51 | 61.1 / 274.5 / 2.21 | 93.6 / 358.6 / 1.49 | 215.4 / 311.5 / 0.68 | 571.0 / 470.1 / 0.34 | 1444.9 / 743.1 / 0.27 |
| humming w4 sym | 27.6 / 9.5 / 2.45 | 30.2 / 34.7 / 2.25 | 21.7 / 193.3 / 3.16 | 26.2 / 319.6 / 2.65 | 36.8 / 455.9 / 1.94 | 57.3 / 585.8 / 1.32 | 93.4 / 718.2 / 0.89 | 312.0 / 860.4 / 0.42 | 937.4 / 1145.5 / 0.34 |
| humming w4 zp | 29.1 / 9.0 / 2.36 | 31.4 / 33.4 / 2.19 | 22.9 / 183.2 / 3.04 | 27.3 / 307.1 / 2.58 | 37.4 / 448.6 / 1.94 | 57.3 / 586.0 / 1.34 | 95.7 / 701.3 / 0.88 | 320.5 / 837.7 / 0.41 | 963.7 / 1114.2 / 0.33 |
| humming w8 sym | 36.0 / 7.3 / 3.70 | 37.5 / 28.0 / 3.56 | 33.4 / 125.5 / 4.01 | 37.1 / 226.3 / 3.64 | 45.6 / 367.7 / 3.00 | 66.8 / 502.3 / 2.11 | 103.7 / 647.0 / 1.44 | 340.1 / 789.2 / 0.58 | 1004.3 / 1069.1 / 0.38 |
| humming w8 zp | 36.8 / 7.1 / 3.65 | 38.5 / 27.3 / 3.49 | 33.9 / 123.7 / 3.98 | 37.6 / 222.9 / 3.62 | 47.0 / 357.1 / 2.94 | 67.9 / 494.5 / 2.09 | 105.1 / 638.6 / 1.43 | 338.4 / 793.3 / 0.58 | 1017.8 / 1055.0 / 0.38 |

## Full results (bf16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.4 / 2.7 / 2.71 | 9.6 / 13.9 / 3.48 | 10.3 / 52.0 / 3.28 | 12.6 / 85.2 / 2.71 | 18.5 / 115.9 / 1.87 | 19.3 / 222.8 / 1.85 | 20.1 / 427.5 / 1.88 | 46.9 / 732.6 / 1.07 | 94.3 / 1457.5 / 1.07 |
| vllm_triton_w4a16 w4 sym | 84.4 / 0.4 / 0.10 | 84.5 / 1.6 / 0.10 | 84.7 / 6.3 / 0.10 | 85.0 / 12.6 / 0.11 | 83.8 / 25.6 / 0.12 | 134.8 / 31.9 / 0.08 | 134.0 / 64.1 / 0.10 | 135.2 / 254.1 / 0.19 | 307.4 / 447.1 / 0.25 |
| vllm_triton_w4a16 w4 zp | 87.8 / 0.4 / 0.10 | 88.7 / 1.5 / 0.10 | 89.2 / 6.0 / 0.10 | 89.0 / 12.1 / 0.10 | 87.8 / 24.5 / 0.11 | 125.6 / 34.2 / 0.09 | 125.1 / 68.7 / 0.10 | 126.2 / 272.3 / 0.20 | 340.1 / 404.2 / 0.22 |
| dequant_triton+hipblaslt w4 sym | 24.9 / 1.3 / 0.35 | 21.0 / 6.4 / 0.42 | 22.9 / 23.4 / 0.39 | 24.4 / 44.0 / 0.38 | 31.0 / 69.2 / 0.31 | 30.3 / 141.9 / 0.35 | 32.8 / 261.9 / 0.39 | 57.9 / 593.3 / 0.44 | 105.0 / 1309.4 / 0.72 |
| dequant_triton+hipblaslt w4 zp | 25.1 / 1.3 / 0.35 | 22.5 / 6.0 / 0.39 | 23.0 / 23.3 / 0.39 | 24.4 / 44.1 / 0.38 | 31.4 / 68.4 / 0.31 | 30.3 / 141.8 / 0.36 | 32.4 / 264.9 / 0.40 | 58.4 / 588.6 / 0.44 | 105.7 / 1299.7 / 0.72 |
| dequant_triton+hipblaslt w8 sym | 25.5 / 1.3 / 0.67 | 22.5 / 6.0 / 0.76 | 23.4 / 23.0 / 0.74 | 25.0 / 43.0 / 0.70 | 31.5 / 68.1 / 0.57 | 31.7 / 135.4 / 0.60 | 34.2 / 251.2 / 0.62 | 58.3 / 589.3 / 0.58 | 106.7 / 1288.1 / 0.79 |
| dequant_triton+hipblaslt w8 zp | 25.3 / 1.3 / 0.68 | 23.5 / 5.7 / 0.73 | 23.5 / 22.8 / 0.74 | 24.9 / 43.1 / 0.71 | 32.3 / 66.5 / 0.56 | 31.3 / 137.1 / 0.61 | 34.3 / 250.8 / 0.62 | 58.5 / 586.9 / 0.58 | 107.2 / 1281.6 / 0.79 |
| aiter_a16w8_int8 w8 sym | 9.6 / 3.5 / 1.80 | 10.0 / 13.4 / 1.73 | 12.6 / 42.5 / 1.39 | 12.6 / 85.4 / 1.42 | 15.6 / 138.0 / 1.18 | 26.8 / 160.1 / 0.72 | 34.9 / 245.8 / 0.61 | 63.5 / 540.9 / 0.54 | 230.9 / 595.1 / 0.36 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.16 | 8.7 / 15.5 / 1.94 | 9.5 / 56.4 / 1.79 | 11.8 / 90.8 / 1.46 | 13.8 / 155.3 / 1.29 | 24.0 / 179.1 / 0.79 | 37.1 / 231.5 / 0.56 | 54.8 / 627.0 / 0.61 | 197.5 / 696.1 / 0.42 |
| humming w4 sym | 7.8 / 4.3 / 1.12 | 8.3 / 16.2 / 1.05 | 9.8 / 55.0 / 0.91 | 8.9 / 120.2 / 1.03 | 11.8 / 182.2 / 0.82 | 21.5 / 199.7 / 0.50 | 22.5 / 381.3 / 0.57 | 42.6 / 806.4 / 0.60 | 135.0 / 1018.2 / 0.56 |
| humming w4 zp | 8.1 / 4.2 / 1.09 | 8.4 / 16.0 / 1.05 | 10.0 / 53.8 / 0.91 | 9.2 / 117.3 / 1.02 | 12.1 / 178.0 / 0.81 | 21.2 / 202.3 / 0.51 | 22.6 / 380.2 / 0.57 | 43.8 / 784.2 / 0.58 | 140.5 / 978.5 / 0.54 |
| humming w8 sym | 8.5 / 3.9 / 2.00 | 9.0 / 14.9 / 1.90 | 10.3 / 52.3 / 1.69 | 9.6 / 111.4 / 1.82 | 12.3 / 175.1 / 1.48 | 22.5 / 191.1 / 0.85 | 24.3 / 353.6 / 0.87 | 48.2 / 713.3 / 0.70 | 141.7 / 970.1 / 0.59 |
| humming w8 zp | 8.7 / 3.9 / 1.98 | 9.2 / 14.6 / 1.88 | 10.5 / 51.2 / 1.66 | 9.7 / 110.8 / 1.83 | 12.6 / 170.6 / 1.45 | 22.4 / 191.5 / 0.86 | 24.1 / 355.9 / 0.89 | 48.3 / 711.9 / 0.70 | 145.0 / 947.7 / 0.58 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 14.0 / 3.6 / 3.60 | 14.2 / 14.2 / 3.55 | 13.9 / 58.0 / 3.65 | 16.6 / 96.8 / 3.06 | 19.0 / 169.3 / 2.71 | 20.5 / 313.6 / 2.58 | 26.8 / 480.5 / 2.07 | 58.2 / 885.6 / 1.23 | 159.8 / 1290.2 / 0.84 |
| vllm_triton_w4a16 w4 sym | 85.0 / 0.6 / 0.15 | 85.0 / 2.4 / 0.15 | 85.2 / 9.5 / 0.16 | 85.2 / 18.9 / 0.16 | 84.6 / 38.1 / 0.17 | 135.6 / 47.5 / 0.12 | 134.7 / 95.7 / 0.14 | 158.3 / 325.6 / 0.21 | 454.9 / 453.2 / 0.21 |
| vllm_triton_w4a16 w4 zp | 89.0 / 0.6 / 0.15 | 89.5 / 2.3 / 0.15 | 89.3 / 9.0 / 0.15 | 89.9 / 17.9 / 0.15 | 88.6 / 36.4 / 0.16 | 125.5 / 51.3 / 0.12 | 125.3 / 102.9 / 0.15 | 174.5 / 295.3 / 0.20 | 501.2 / 411.3 / 0.19 |
| dequant_triton+hipblaslt w4 sym | 29.5 / 1.7 / 0.44 | 29.4 / 6.9 / 0.44 | 28.6 / 28.2 / 0.47 | 31.2 / 51.5 / 0.44 | 34.2 / 94.2 / 0.42 | 35.3 / 182.5 / 0.44 | 41.0 / 314.4 / 0.45 | 69.4 / 742.3 / 0.49 | 170.6 / 1208.2 / 0.57 |
| dequant_triton+hipblaslt w4 zp | 29.9 / 1.7 / 0.44 | 29.3 / 6.9 / 0.45 | 28.5 / 28.2 / 0.47 | 32.8 / 49.2 / 0.42 | 34.1 / 94.5 / 0.42 | 35.0 / 183.8 / 0.45 | 41.9 / 307.7 / 0.44 | 69.4 / 743.1 / 0.49 | 170.4 / 1209.9 / 0.57 |
| dequant_triton+hipblaslt w8 sym | 31.0 / 1.6 / 0.82 | 31.3 / 6.4 / 0.82 | 30.0 / 26.9 / 0.86 | 34.4 / 46.9 / 0.76 | 36.0 / 89.5 / 0.75 | 37.4 / 172.1 / 0.75 | 42.7 / 301.5 / 0.72 | 69.5 / 741.4 / 0.67 | 172.4 / 1195.8 / 0.64 |
| dequant_triton+hipblaslt w8 zp | 31.0 / 1.6 / 0.83 | 31.3 / 6.4 / 0.83 | 29.8 / 27.0 / 0.88 | 32.8 / 49.1 / 0.81 | 35.9 / 89.7 / 0.75 | 37.3 / 172.8 / 0.76 | 42.6 / 302.4 / 0.73 | 69.0 / 746.5 / 0.68 | 172.5 / 1195.2 / 0.64 |
| aiter_a16w8_int8 w8 sym | 11.2 / 4.5 / 2.32 | 11.6 / 17.4 / 2.25 | 14.7 / 54.8 / 1.79 | 15.0 / 107.5 / 1.78 | 19.0 / 169.8 / 1.44 | 29.7 / 217.2 / 0.96 | 38.6 / 333.9 / 0.81 | 116.6 / 442.0 / 0.40 | 344.5 / 598.4 / 0.32 |
| aiter_a16w8_fp8_block128 w8 sym | 9.0 / 5.6 / 2.81 | 9.8 / 20.6 / 2.58 | 11.1 / 72.4 / 2.29 | 13.8 / 116.5 / 1.87 | 17.6 / 183.4 / 1.51 | 26.7 / 241.0 / 1.04 | 38.7 / 333.0 / 0.79 | 100.0 / 515.2 / 0.46 | 294.7 / 699.5 / 0.37 |
| humming w4 sym | 9.3 / 5.4 / 1.40 | 10.1 / 19.9 / 1.29 | 9.9 / 81.4 / 1.34 | 9.5 / 168.7 / 1.43 | 13.4 / 241.2 / 1.07 | 24.2 / 266.2 / 0.64 | 27.3 / 472.7 / 0.67 | 61.8 / 834.6 / 0.55 | 197.3 / 1045.1 / 0.49 |
| humming w4 zp | 9.6 / 5.2 / 1.37 | 10.4 / 19.4 / 1.28 | 10.2 / 79.1 / 1.33 | 9.8 / 165.1 / 1.42 | 14.2 / 227.1 / 1.02 | 24.2 / 266.5 / 0.65 | 27.5 / 469.0 / 0.67 | 63.6 / 810.8 / 0.54 | 201.9 / 1021.1 / 0.48 |
| humming w8 sym | 11.2 / 4.5 / 2.29 | 12.0 / 16.8 / 2.14 | 11.6 / 69.3 / 2.23 | 11.3 / 142.6 / 2.32 | 15.6 / 207.1 / 1.73 | 25.6 / 251.4 / 1.10 | 29.8 / 432.6 / 1.03 | 65.4 / 788.3 / 0.71 | 206.8 / 997.0 / 0.53 |
| humming w8 zp | 11.3 / 4.4 / 2.28 | 12.1 / 16.6 / 2.13 | 11.9 / 67.7 / 2.19 | 11.5 / 140.4 / 2.30 | 15.8 / 203.3 / 1.71 | 26.0 / 248.3 / 1.09 | 30.3 / 425.6 / 1.02 | 66.7 / 772.3 / 0.70 | 210.9 / 977.5 / 0.52 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 48.7 / 4.8 / 4.83 | 49.0 / 19.2 / 4.80 | 45.4 / 82.7 / 5.19 | 48.8 / 154.0 / 4.86 | 47.6 / 316.0 / 5.03 | 55.9 / 538.1 / 4.35 | 66.7 / 901.0 / 3.77 | 191.2 / 1257.6 / 1.58 | 633.0 / 1519.8 / 0.80 |
| vllm_triton_w4a16 w4 sym | 95.1 / 2.5 / 0.64 | 94.9 / 9.9 / 0.64 | 95.0 / 39.5 / 0.65 | 94.9 / 79.2 / 0.66 | 95.3 / 157.7 / 0.68 | 136.6 / 220.1 / 0.51 | 161.3 / 372.9 / 0.48 | 556.3 / 432.3 / 0.23 | 2066.9 / 465.5 / 0.16 |
| vllm_triton_w4a16 w4 zp | 101.4 / 2.3 / 0.60 | 101.5 / 9.3 / 0.60 | 101.1 / 37.2 / 0.61 | 100.8 / 74.6 / 0.63 | 100.4 / 149.7 / 0.65 | 126.7 / 237.3 / 0.55 | 175.1 / 343.4 / 0.44 | 581.9 / 413.4 / 0.22 | 2251.6 / 427.3 / 0.15 |
| dequant_triton+hipblaslt w4 sym | 100.9 / 2.3 / 0.60 | 97.7 / 9.6 / 0.62 | 100.6 / 37.4 / 0.61 | 102.1 / 73.6 / 0.61 | 111.1 / 135.3 / 0.58 | 110.4 / 272.4 / 0.62 | 151.6 / 396.7 / 0.51 | 241.4 / 996.4 / 0.53 | 679.1 / 1416.7 / 0.48 |
| dequant_triton+hipblaslt w4 zp | 100.9 / 2.3 / 0.60 | 103.4 / 9.1 / 0.59 | 101.3 / 37.1 / 0.61 | 116.2 / 64.7 / 0.54 | 100.8 / 149.1 / 0.65 | 119.4 / 251.8 / 0.58 | 130.7 / 460.1 / 0.59 | 248.3 / 968.5 / 0.52 | 677.4 / 1420.3 / 0.49 |
| dequant_triton+hipblaslt w8 sym | 119.2 / 2.0 / 1.00 | 107.7 / 8.7 / 1.11 | 107.8 / 34.9 / 1.12 | 112.5 / 66.8 / 1.08 | 119.5 / 125.8 / 1.03 | 118.9 / 252.8 / 1.07 | 162.8 / 369.3 / 0.84 | 253.3 / 949.4 / 0.74 | 682.9 / 1408.8 / 0.57 |
| dequant_triton+hipblaslt w8 zp | 114.6 / 2.0 / 1.05 | 119.9 / 7.8 / 1.00 | 108.8 / 34.6 / 1.11 | 122.2 / 61.5 / 1.00 | 120.9 / 124.3 / 1.03 | 123.8 / 242.8 / 1.04 | 161.5 / 372.3 / 0.85 | 258.5 / 930.6 / 0.72 | 688.3 / 1397.8 / 0.56 |
| aiter_a16w8_int8 w8 sym | 32.2 / 7.3 / 3.77 | 35.1 / 26.8 / 3.46 | 44.8 / 83.8 / 2.72 | 46.4 / 161.8 / 2.65 | 59.5 / 252.7 / 2.11 | 93.1 / 322.8 / 1.39 | 147.3 / 408.2 / 0.94 | 416.5 / 577.5 / 0.45 | 1592.3 / 604.2 / 0.24 |
| aiter_a16w8_fp8_block128 w8 sym | 23.2 / 10.1 / 5.07 | 26.4 / 35.5 / 4.45 | 33.3 / 112.8 / 3.56 | 44.5 / 168.9 / 2.69 | 54.4 / 276.5 / 2.24 | 80.7 / 372.3 / 1.56 | 147.4 / 408.1 / 0.91 | 357.9 / 672.0 / 0.52 | 1364.0 / 705.4 / 0.28 |
| humming w4 sym | 18.3 / 12.9 / 3.32 | 18.6 / 50.5 / 3.27 | 18.0 / 209.4 / 3.43 | 21.6 / 348.7 / 2.91 | 29.3 / 513.6 / 2.21 | 47.8 / 629.3 / 1.44 | 69.3 / 868.0 / 1.12 | 228.8 / 1051.2 / 0.56 | 886.5 / 1085.3 / 0.37 |
| humming w4 zp | 19.1 / 12.3 / 3.22 | 19.5 / 48.3 / 3.17 | 18.7 / 200.5 / 3.33 | 22.5 / 334.7 / 2.83 | 31.0 / 484.2 / 2.12 | 47.3 / 635.5 / 1.48 | 71.6 / 840.4 / 1.09 | 236.9 / 1015.4 / 0.54 | 913.2 / 1053.5 / 0.36 |
| humming w8 sym | 28.0 / 8.4 / 4.26 | 28.7 / 32.7 / 4.17 | 27.5 / 136.7 / 4.38 | 30.3 / 247.7 / 4.00 | 37.0 / 405.7 / 3.33 | 52.1 / 577.1 / 2.45 | 74.2 / 810.4 / 1.83 | 250.0 / 962.1 / 0.75 | 958.5 / 1003.7 / 0.40 |
| humming w8 zp | 27.7 / 8.5 / 4.34 | 28.7 / 32.8 / 4.20 | 27.6 / 136.3 / 4.40 | 30.2 / 248.7 / 4.05 | 36.9 / 407.6 / 3.37 | 52.8 / 569.7 / 2.44 | 77.1 / 779.4 / 1.77 | 255.8 / 940.4 / 0.73 | 977.4 / 984.3 / 0.40 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.3 / 4.8 / 4.83 | 24.8 / 19.0 / 4.75 | 26.3 / 71.4 / 4.48 | 29.7 / 126.7 / 4.00 | 29.4 / 255.3 / 4.07 | 34.2 / 440.1 / 3.58 | 51.0 / 589.6 / 2.49 | 115.1 / 1045.0 / 1.35 | 302.0 / 1592.7 / 0.89 |
| vllm_triton_w4a16 w4 sym | 289.2 / 0.4 / 0.10 | 288.9 / 1.6 / 0.10 | 289.1 / 6.5 / 0.11 | 289.5 / 13.0 / 0.11 | 287.2 / 26.2 / 0.11 | 462.7 / 32.5 / 0.08 | 461.3 / 65.2 / 0.09 | 465.4 / 258.4 / 0.15 | 1047.4 / 459.3 / 0.17 |
| vllm_triton_w4a16 w4 zp | 300.9 / 0.4 / 0.10 | 302.0 / 1.6 / 0.10 | 304.5 / 6.2 / 0.10 | 303.3 / 12.4 / 0.10 | 300.0 / 25.1 / 0.11 | 429.1 / 35.0 / 0.08 | 428.2 / 70.2 / 0.09 | 430.9 / 279.1 / 0.16 | 1168.1 / 411.8 / 0.15 |
| dequant_triton+hipblaslt w4 sym | 55.6 / 2.1 / 0.55 | 55.8 / 8.4 / 0.55 | 57.1 / 32.9 / 0.54 | 59.2 / 63.4 / 0.53 | 60.0 / 125.3 / 0.54 | 65.9 / 228.2 / 0.53 | 85.1 / 353.4 / 0.47 | 138.7 / 866.8 / 0.49 | 325.2 / 1479.0 / 0.56 |
| dequant_triton+hipblaslt w4 zp | 58.6 / 2.0 / 0.52 | 58.8 / 8.0 / 0.52 | 59.9 / 31.4 / 0.52 | 62.2 / 60.4 / 0.51 | 62.7 / 119.9 / 0.52 | 68.6 / 219.1 / 0.51 | 87.6 / 343.1 / 0.46 | 141.7 / 848.9 / 0.48 | 329.0 / 1462.2 / 0.55 |
| dequant_triton+hipblaslt w8 sym | 58.6 / 2.0 / 1.02 | 58.8 / 8.0 / 1.02 | 59.6 / 31.5 / 1.01 | 61.1 / 61.5 / 0.99 | 62.0 / 121.2 / 1.00 | 67.9 / 221.5 / 0.95 | 88.1 / 341.4 / 0.78 | 141.9 / 847.3 / 0.69 | 326.8 / 1472.1 / 0.65 |
| dequant_triton+hipblaslt w8 zp | 59.5 / 2.0 / 1.01 | 60.0 / 7.8 / 1.00 | 60.7 / 31.0 / 1.00 | 62.4 / 60.2 / 0.98 | 63.0 / 119.4 / 0.99 | 68.9 / 218.2 / 0.94 | 88.4 / 340.0 / 0.79 | 144.9 / 830.0 / 0.68 | 329.5 / 1460.1 / 0.64 |
| aiter_a16w8_int8 w8 sym | 19.5 / 6.0 / 3.10 | 20.3 / 23.2 / 3.00 | 28.5 / 65.9 / 2.15 | 28.2 / 133.3 / 2.19 | 35.7 / 210.6 / 1.76 | 69.3 / 217.0 / 0.94 | 112.7 / 266.7 / 0.62 | 204.1 / 589.3 / 0.48 | 771.0 / 623.9 / 0.27 |
| aiter_a16w8_fp8_block128 w8 sym | 17.0 / 6.9 / 3.46 | 17.7 / 26.5 / 3.32 | 18.3 / 102.4 / 3.23 | 27.2 / 138.2 / 2.20 | 30.2 / 248.7 / 2.02 | 56.4 / 266.8 / 1.13 | 120.9 / 248.7 / 0.56 | 172.9 / 695.7 / 0.56 | 648.1 / 742.2 / 0.32 |
| humming w4 sym | 12.7 / 9.2 / 2.39 | 13.4 / 34.9 / 2.26 | 14.3 / 131.7 / 2.16 | 16.2 / 231.5 / 1.94 | 21.1 / 356.9 / 1.55 | 32.5 / 463.2 / 1.08 | 45.4 / 662.8 / 0.88 | 130.8 / 919.1 / 0.52 | 440.1 / 1093.0 / 0.41 |
| humming w4 zp | 13.9 / 8.4 / 2.21 | 14.8 / 31.8 / 2.09 | 14.9 / 125.9 / 2.10 | 16.8 / 223.7 / 1.90 | 22.2 / 338.4 / 1.49 | 33.2 / 453.1 / 1.07 | 45.9 / 654.8 / 0.88 | 134.8 / 891.9 / 0.51 | 451.7 / 1065.0 / 0.40 |
| humming w8 sym | 14.6 / 8.0 / 4.08 | 15.4 / 30.5 / 3.88 | 17.4 / 108.3 / 3.47 | 17.9 / 210.3 / 3.40 | 23.0 / 326.9 / 2.70 | 37.1 / 405.4 / 1.74 | 51.3 / 585.6 / 1.34 | 150.6 / 798.6 / 0.65 | 464.1 / 1036.5 / 0.45 |
| humming w8 zp | 15.1 / 7.8 / 3.98 | 15.8 / 29.7 / 3.81 | 17.6 / 106.6 / 3.44 | 18.3 / 205.2 / 3.35 | 23.5 / 319.7 / 2.66 | 36.9 / 407.3 / 1.76 | 50.8 / 592.2 / 1.37 | 150.1 / 801.3 / 0.65 | 476.4 / 1009.6 / 0.44 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 26.0 / 5.2 / 5.17 | 26.3 / 20.4 / 5.10 | 27.4 / 78.3 / 4.92 | 29.3 / 146.8 / 4.62 | 30.6 / 280.3 / 4.45 | 40.5 / 424.4 / 3.42 | 55.4 / 620.5 / 2.58 | 131.8 / 1043.1 / 1.27 | 350.8 / 1567.2 / 0.77 |
| vllm_triton_w4a16 w4 sym | 166.7 / 0.8 / 0.21 | 166.5 / 3.2 / 0.21 | 165.8 / 13.0 / 0.21 | 166.0 / 25.9 / 0.21 | 164.7 / 52.2 / 0.22 | 265.4 / 64.7 / 0.15 | 264.0 / 130.2 / 0.16 | 311.3 / 441.5 / 0.22 | 1189.1 / 462.3 / 0.14 |
| vllm_triton_w4a16 w4 zp | 175.6 / 0.8 / 0.20 | 175.6 / 3.1 / 0.20 | 175.3 / 12.2 / 0.20 | 175.4 / 24.5 / 0.20 | 172.4 / 49.8 / 0.21 | 247.7 / 69.4 / 0.16 | 246.1 / 139.6 / 0.18 | 346.5 / 396.7 / 0.20 | 1306.7 / 420.7 / 0.13 |
| dequant_triton+hipblaslt w4 sym | 60.1 / 2.2 / 0.58 | 65.4 / 8.2 / 0.53 | 61.9 / 34.7 / 0.57 | 69.4 / 61.9 / 0.51 | 67.4 / 127.5 / 0.55 | 84.5 / 203.3 / 0.46 | 92.5 / 371.4 / 0.47 | 174.7 / 786.6 / 0.39 | 376.1 / 1461.6 / 0.45 |
| dequant_triton+hipblaslt w4 zp | 61.3 / 2.2 / 0.57 | 66.5 / 8.1 / 0.53 | 63.3 / 34.0 / 0.56 | 68.2 / 62.9 / 0.53 | 70.2 / 122.3 / 0.53 | 85.5 / 200.9 / 0.46 | 90.8 / 378.4 / 0.48 | 171.1 / 803.4 / 0.40 | 378.0 / 1454.3 / 0.45 |
| dequant_triton+hipblaslt w8 sym | 62.9 / 2.1 / 1.08 | 69.8 / 7.7 / 0.98 | 69.5 / 30.9 / 0.99 | 73.5 / 58.4 / 0.94 | 71.5 / 120.2 / 0.98 | 88.3 / 194.5 / 0.82 | 95.4 / 360.0 / 0.80 | 178.5 / 770.1 / 0.57 | 377.7 / 1455.6 / 0.54 |
| dequant_triton+hipblaslt w8 zp | 70.2 / 1.9 / 0.98 | 70.4 / 7.6 / 0.98 | 71.9 / 29.9 / 0.96 | 74.1 / 58.0 / 0.94 | 78.1 / 110.1 / 0.91 | 88.5 / 194.1 / 0.82 | 103.4 / 332.2 / 0.74 | 179.1 / 767.5 / 0.57 | 380.8 / 1443.7 / 0.53 |
| aiter_a16w8_int8 w8 sym | 19.5 / 6.9 / 3.55 | 20.2 / 26.6 / 3.44 | 25.9 / 83.0 / 2.69 | 24.0 / 179.2 / 2.93 | 31.6 / 271.8 / 2.26 | 48.7 / 352.7 / 1.51 | 78.3 / 438.8 / 0.99 | 232.7 / 590.6 / 0.44 | 880.3 / 624.5 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 15.5 / 8.7 / 4.34 | 16.4 / 32.8 / 4.11 | 18.1 / 118.7 / 3.74 | 22.2 / 193.2 / 3.07 | 27.2 / 315.7 / 2.54 | 42.7 / 402.3 / 1.67 | 75.0 / 458.0 / 1.01 | 197.1 / 697.3 / 0.51 | 746.2 / 736.8 / 0.27 |
| humming w4 sym | 13.1 / 10.3 / 2.65 | 13.7 / 39.1 / 2.53 | 14.1 / 151.9 / 2.48 | 16.5 / 260.3 / 2.16 | 20.8 / 412.3 / 1.76 | 31.1 / 551.6 / 1.25 | 47.0 / 730.3 / 0.91 | 133.2 / 1031.7 / 0.51 | 493.4 / 1114.1 / 0.34 |
| humming w4 zp | 14.6 / 9.2 / 2.41 | 15.1 / 35.6 / 2.34 | 14.6 / 147.5 / 2.45 | 16.8 / 255.5 / 2.15 | 21.4 / 400.5 / 1.74 | 31.4 / 546.6 / 1.25 | 48.6 / 707.4 / 0.90 | 139.2 / 987.1 / 0.49 | 508.6 / 1080.8 / 0.33 |
| humming w8 sym | 17.4 / 7.7 / 3.92 | 18.0 / 29.9 / 3.80 | 17.8 / 120.4 / 3.85 | 18.3 / 234.3 / 3.77 | 22.7 / 378.4 / 3.10 | 34.8 / 494.0 / 2.08 | 52.0 / 660.8 / 1.47 | 143.0 / 960.8 / 0.71 | 518.9 / 1059.6 / 0.39 |
| humming w8 zp | 17.5 / 7.7 / 3.92 | 18.4 / 29.2 / 3.75 | 17.8 / 120.8 / 3.89 | 18.5 / 232.0 / 3.77 | 23.0 / 372.8 / 3.07 | 35.0 / 490.3 / 2.08 | 52.9 / 649.8 / 1.46 | 146.1 / 940.7 / 0.70 | 530.8 / 1035.7 / 0.38 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 171.2 / 5.5 / 5.49 | 161.1 / 23.3 / 5.83 | 157.4 / 95.5 / 5.98 | 173.4 / 173.3 / 5.44 | 164.8 / 364.8 / 5.75 | 169.7 / 708.8 / 5.64 | 223.2 / 1077.5 / 4.36 | 723.3 / 1330.1 / 1.48 | 2546.8 / 1511.0 / 0.58 |
| vllm_triton_w4a16 w4 sym | 283.5 / 3.3 / 0.85 | 282.3 / 13.3 / 0.86 | 296.9 / 50.6 / 0.82 | 297.0 / 101.2 / 0.83 | 280.4 / 214.5 / 0.89 | 320.1 / 375.7 / 0.81 | 619.6 / 388.2 / 0.45 | 2093.9 / 459.5 / 0.18 | 8127.7 / 473.5 / 0.10 |
| vllm_triton_w4a16 w4 zp | 307.1 / 3.1 / 0.80 | 306.3 / 12.3 / 0.80 | 316.4 / 47.5 / 0.78 | 316.5 / 95.0 / 0.78 | 296.0 / 203.1 / 0.85 | 345.9 / 347.6 / 0.75 | 669.3 / 359.4 / 0.41 | 2268.5 / 424.1 / 0.17 | 8892.3 / 432.8 / 0.09 |
| dequant_triton+hipblaslt w4 sym | 421.0 / 2.2 / 0.58 | 423.0 / 8.9 / 0.57 | 417.2 / 36.0 / 0.59 | 423.0 / 71.1 / 0.58 | 422.2 / 142.4 / 0.59 | 438.4 / 274.3 / 0.59 | 490.9 / 489.9 / 0.56 | 981.8 / 979.9 / 0.38 | 2766.8 / 1390.9 / 0.28 |
| dequant_triton+hipblaslt w4 zp | 426.3 / 2.2 / 0.57 | 434.4 / 8.7 / 0.56 | 428.4 / 35.1 / 0.57 | 434.1 / 69.3 / 0.57 | 445.7 / 134.9 / 0.57 | 449.3 / 267.7 / 0.58 | 508.1 / 473.4 / 0.55 | 994.5 / 967.4 / 0.38 | 2783.6 / 1382.5 / 0.28 |
| dequant_triton+hipblaslt w8 sym | 477.9 / 2.0 / 1.00 | 479.9 / 7.8 / 0.99 | 474.7 / 31.7 / 1.01 | 477.9 / 62.9 / 1.01 | 481.8 / 124.8 / 1.01 | 492.0 / 244.4 / 1.00 | 570.8 / 421.4 / 0.90 | 1021.4 / 941.9 / 0.60 | 2819.1 / 1365.1 / 0.36 |
| dequant_triton+hipblaslt w8 zp | 473.7 / 2.0 / 1.01 | 493.7 / 7.6 / 0.97 | 479.4 / 31.4 / 1.01 | 498.0 / 60.4 / 0.97 | 485.9 / 123.8 / 1.01 | 511.6 / 235.1 / 0.97 | 561.4 / 428.4 / 0.92 | 1028.9 / 935.1 / 0.60 | 2835.0 / 1357.4 / 0.36 |
| aiter_a16w8_int8 w8 sym | 104.9 / 9.0 / 4.62 | 114.1 / 32.9 / 4.25 | 136.4 / 110.2 / 3.57 | 140.7 / 213.7 / 3.47 | 174.5 / 344.6 / 2.83 | 290.3 / 414.3 / 1.73 | 520.1 / 462.5 / 1.00 | 1596.6 / 602.6 / 0.39 | 6165.1 / 624.2 / 0.17 |
| aiter_a16w8_fp8_block128 w8 sym | 77.1 / 12.2 / 6.10 | 85.0 / 44.2 / 5.54 | 104.4 / 144.1 / 4.52 | 134.1 / 224.3 / 3.54 | 160.4 / 374.9 / 2.98 | 240.4 / 500.3 / 2.02 | 502.3 / 478.8 / 1.00 | 1359.6 / 707.6 / 0.44 | 5216.3 / 737.7 / 0.19 |
| humming w4 sym | 55.4 / 17.0 / 4.37 | 54.9 / 68.5 / 4.42 | 56.5 / 266.0 / 4.32 | 67.2 / 447.7 / 3.67 | 89.6 / 670.8 / 2.80 | 135.6 / 887.1 / 1.91 | 251.6 / 955.9 / 1.10 | 878.6 / 1095.0 / 0.43 | 3447.4 / 1116.3 / 0.23 |
| humming w4 zp | 58.0 / 16.2 / 4.24 | 57.9 / 64.9 / 4.25 | 58.0 / 259.0 / 4.27 | 68.9 / 436.4 / 3.63 | 90.7 / 662.7 / 2.80 | 139.2 / 863.9 / 1.89 | 261.7 / 919.1 / 1.07 | 906.8 / 1061.0 / 0.42 | 3568.4 / 1078.4 / 0.22 |
| humming w8 sym | 89.2 / 10.5 / 5.35 | 86.1 / 43.7 / 5.55 | 90.7 / 165.7 / 5.28 | 93.7 / 320.8 / 5.13 | 111.8 / 537.7 / 4.34 | 156.2 / 770.0 / 3.16 | 296.3 / 811.8 / 1.72 | 1084.7 / 887.0 / 0.56 | 4389.0 / 876.8 / 0.23 |
| humming w8 zp | 86.2 / 10.9 / 5.58 | 87.2 / 43.1 / 5.52 | 91.2 / 164.8 / 5.29 | 94.8 / 317.1 / 5.12 | 114.8 / 523.7 / 4.26 | 159.1 / 755.7 / 3.13 | 304.5 / 789.8 / 1.69 | 1099.6 / 874.9 / 0.56 | 4509.5 / 853.4 / 0.23 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.9 / 6.2 / 6.19 | 76.2 / 24.6 / 6.17 | 77.6 / 96.8 / 6.07 | 79.1 / 190.0 / 5.97 | 84.2 / 357.2 / 5.64 | 94.3 / 637.4 / 5.08 | 123.3 / 975.3 / 3.96 | 353.5 / 1360.9 / 1.54 | 1215.5 / 1583.0 / 0.64 |
| vllm_triton_w4a16 w4 sym | 575.2 / 0.8 / 0.21 | 570.6 / 3.3 / 0.21 | 570.3 / 13.2 / 0.21 | 571.1 / 26.3 / 0.22 | 570.1 / 52.7 / 0.22 | 920.5 / 65.3 / 0.14 | 917.0 / 131.1 / 0.15 | 1061.6 / 453.1 / 0.18 | 4186.9 / 459.6 / 0.10 |
| vllm_triton_w4a16 w4 zp | 602.3 / 0.8 / 0.20 | 603.5 / 3.1 / 0.20 | 605.7 / 12.4 / 0.20 | 606.4 / 24.8 / 0.20 | 597.5 / 50.3 / 0.21 | 853.9 / 70.4 / 0.15 | 849.8 / 141.5 / 0.17 | 1186.2 / 405.5 / 0.17 | 4580.9 / 420.0 / 0.09 |
| dequant_triton+hipblaslt w4 sym | 218.8 / 2.1 / 0.55 | 214.0 / 8.8 / 0.57 | 219.6 / 34.2 / 0.56 | 217.0 / 69.3 / 0.57 | 227.1 / 132.4 / 0.55 | 224.7 / 267.6 / 0.58 | 258.1 / 465.9 / 0.54 | 481.6 / 998.8 / 0.41 | 1337.5 / 1438.6 / 0.32 |
| dequant_triton+hipblaslt w4 zp | 234.3 / 2.0 / 0.52 | 230.2 / 8.2 / 0.53 | 225.8 / 33.3 / 0.55 | 237.6 / 63.3 / 0.52 | 239.0 / 125.8 / 0.53 | 244.2 / 246.3 / 0.54 | 271.0 / 443.8 / 0.52 | 503.0 / 956.3 / 0.39 | 1345.5 / 1430.1 / 0.32 |
| dequant_triton+hipblaslt w8 sym | 239.3 / 2.0 / 1.00 | 240.8 / 7.8 / 0.99 | 238.8 / 31.5 / 1.00 | 242.9 / 61.9 / 0.99 | 242.0 / 124.2 / 1.00 | 250.3 / 240.2 / 0.99 | 274.7 / 437.7 / 0.94 | 498.8 / 964.4 / 0.63 | 1347.7 / 1427.8 / 0.40 |
| dequant_triton+hipblaslt w8 zp | 233.1 / 2.0 / 1.03 | 247.3 / 7.6 / 0.97 | 235.2 / 32.0 / 1.03 | 236.3 / 63.6 / 1.03 | 246.5 / 122.0 / 0.99 | 257.6 / 233.4 / 0.97 | 277.9 / 432.8 / 0.93 | 507.6 / 947.7 / 0.62 | 1363.4 / 1411.3 / 0.40 |
| aiter_a16w8_int8 w8 sym | 49.3 / 9.5 / 4.92 | 50.6 / 37.1 / 4.79 | 71.0 / 105.9 / 3.43 | 66.8 / 224.9 / 3.66 | 82.3 / 365.2 / 3.00 | 133.7 / 449.6 / 1.88 | 260.5 / 461.6 / 1.00 | 795.0 / 605.0 / 0.40 | 3054.1 / 630.0 / 0.18 |
| aiter_a16w8_fp8_block128 w8 sym | 42.6 / 11.0 / 5.52 | 43.7 / 43.0 / 5.38 | 47.6 / 157.8 / 4.96 | 63.9 / 235.4 / 3.72 | 75.7 / 397.2 / 3.17 | 110.9 / 542.1 / 2.20 | 252.4 / 476.5 / 1.01 | 670.4 / 717.6 / 0.46 | 2571.1 / 748.4 / 0.21 |
| humming w4 sym | 37.9 / 12.4 / 3.20 | 35.6 / 52.7 / 3.41 | 33.5 / 224.7 / 3.65 | 40.6 / 370.3 / 3.04 | 54.0 / 556.6 / 2.33 | 79.9 / 752.2 / 1.63 | 138.8 / 866.3 / 1.01 | 443.2 / 1085.3 / 0.44 | 1681.7 / 1144.2 / 0.25 |
| humming w4 zp | 41.3 / 11.4 / 2.98 | 39.8 / 47.2 / 3.10 | 35.2 / 213.6 / 3.53 | 42.0 / 357.6 / 2.98 | 55.2 / 545.1 / 2.31 | 81.6 / 736.8 / 1.62 | 141.9 / 847.7 / 1.00 | 455.6 / 1055.7 / 0.44 | 1734.0 / 1109.7 / 0.24 |
| humming w8 sym | 46.0 / 10.2 / 5.19 | 45.3 / 41.5 / 5.28 | 44.8 / 167.8 / 5.35 | 48.3 / 311.4 / 4.99 | 62.4 / 481.9 / 3.90 | 95.6 / 628.9 / 2.59 | 162.2 / 741.6 / 1.59 | 466.1 / 1032.0 / 0.67 | 1810.8 / 1062.6 / 0.30 |
| humming w8 zp | 45.0 / 10.4 / 5.35 | 46.4 / 40.5 / 5.19 | 45.2 / 166.1 / 5.34 | 50.1 / 300.2 / 4.85 | 61.6 / 488.2 / 3.98 | 95.3 / 630.9 / 2.62 | 162.8 / 738.7 / 1.59 | 484.0 / 993.9 / 0.65 | 1870.8 / 1028.5 / 0.29 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 16.3 / 3.2 / 3.21 | 16.6 / 12.6 / 3.16 | 15.4 / 54.4 / 3.42 | 16.5 / 101.9 / 3.22 | 23.1 / 145.2 / 2.33 | 25.1 / 267.7 / 2.20 | 36.1 / 371.5 / 1.60 | 64.4 / 833.2 / 1.14 | 179.8 / 1194.6 / 0.76 |
| vllm_triton_w4a16 w4 sym | 105.8 / 0.5 / 0.13 | 105.7 / 2.0 / 0.13 | 105.7 / 7.9 / 0.13 | 105.8 / 15.9 / 0.13 | 104.8 / 32.0 / 0.14 | 168.6 / 39.8 / 0.10 | 167.3 / 80.2 / 0.11 | 195.1 / 275.2 / 0.18 | 511.7 / 419.7 / 0.19 |
| vllm_triton_w4a16 w4 zp | 110.5 / 0.5 / 0.12 | 111.0 / 1.9 / 0.12 | 111.1 / 7.6 / 0.13 | 111.2 / 15.1 / 0.13 | 109.7 / 30.6 / 0.14 | 156.0 / 43.0 / 0.10 | 155.5 / 86.3 / 0.12 | 215.2 / 249.4 / 0.16 | 525.1 / 409.0 / 0.19 |
| dequant_triton+hipblaslt w4 sym | 31.7 / 1.7 / 0.43 | 32.1 / 6.5 / 0.42 | 31.3 / 26.8 / 0.44 | 33.3 / 50.4 / 0.42 | 38.0 / 88.4 / 0.39 | 39.2 / 171.0 / 0.41 | 49.7 / 270.2 / 0.38 | 71.4 / 751.6 / 0.48 | 189.8 / 1131.3 / 0.51 |
| dequant_triton+hipblaslt w4 zp | 32.1 / 1.6 / 0.43 | 32.4 / 6.5 / 0.42 | 31.5 / 26.6 / 0.44 | 33.5 / 50.1 / 0.43 | 38.3 / 87.5 / 0.39 | 39.5 / 169.8 / 0.41 | 50.2 / 267.2 / 0.38 | 72.1 / 744.4 / 0.48 | 190.1 / 1129.4 / 0.51 |
| dequant_triton+hipblaslt w8 sym | 32.7 / 1.6 / 0.81 | 33.3 / 6.3 / 0.80 | 32.1 / 26.1 / 0.84 | 34.1 / 49.1 / 0.80 | 39.1 / 85.8 / 0.71 | 40.6 / 165.3 / 0.72 | 50.7 / 264.7 / 0.63 | 70.6 / 760.9 / 0.68 | 190.6 / 1126.4 / 0.58 |
| dequant_triton+hipblaslt w8 zp | 32.7 / 1.6 / 0.82 | 33.2 / 6.3 / 0.81 | 31.9 / 26.3 / 0.85 | 34.1 / 49.2 / 0.81 | 39.2 / 85.5 / 0.72 | 40.5 / 165.8 / 0.73 | 50.8 / 264.4 / 0.63 | 71.5 / 750.8 / 0.67 | 191.1 / 1123.6 / 0.58 |
| aiter_a16w8_int8 w8 sym | 12.2 / 4.3 / 2.21 | 12.5 / 16.8 / 2.17 | 16.2 / 51.9 / 1.69 | 17.1 / 98.0 / 1.62 | 21.1 / 159.2 / 1.34 | 33.4 / 200.6 / 0.89 | 47.3 / 283.7 / 0.68 | 145.0 / 370.3 / 0.33 | 357.9 / 600.0 / 0.31 |
| aiter_a16w8_fp8_block128 w8 sym | 10.6 / 5.0 / 2.48 | 11.0 / 19.1 / 2.39 | 11.9 / 70.2 / 2.22 | 15.7 / 106.6 / 1.71 | 19.6 / 171.5 / 1.41 | 29.1 / 230.2 / 0.99 | 47.2 / 284.1 / 0.67 | 122.1 / 439.8 / 0.39 | 303.0 / 708.6 / 0.36 |
| humming w4 sym | 9.8 / 5.3 / 1.38 | 10.4 / 20.2 / 1.31 | 10.5 / 80.2 / 1.32 | 10.5 / 159.7 / 1.35 | 15.1 / 222.1 / 0.98 | 25.2 / 266.1 / 0.64 | 31.7 / 423.6 / 0.59 | 71.7 / 748.3 / 0.48 | 203.3 / 1056.1 / 0.48 |
| humming w4 zp | 10.2 / 5.1 / 1.34 | 10.9 / 19.3 / 1.27 | 10.6 / 79.0 / 1.32 | 10.7 / 157.3 / 1.35 | 15.7 / 213.7 / 0.96 | 24.6 / 272.9 / 0.67 | 31.6 / 424.1 / 0.60 | 73.7 / 728.8 / 0.47 | 209.7 / 1024.1 / 0.47 |
| humming w8 sym | 11.1 / 4.7 / 2.39 | 11.5 / 18.3 / 2.33 | 12.3 / 68.4 / 2.20 | 12.2 / 137.8 / 2.24 | 17.4 / 193.2 / 1.61 | 27.8 / 241.2 / 1.05 | 34.2 / 392.4 / 0.93 | 76.2 / 704.7 / 0.62 | 216.4 / 992.4 / 0.51 |
| humming w8 zp | 11.6 / 4.5 / 2.32 | 12.0 / 17.5 / 2.25 | 12.5 / 67.1 / 2.17 | 12.3 / 136.0 / 2.23 | 17.8 / 188.2 / 1.58 | 27.5 / 243.9 / 1.07 | 35.0 / 383.7 / 0.92 | 77.9 / 689.5 / 0.61 | 219.9 / 976.4 / 0.50 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 83.1 / 6.3 / 6.31 | 83.2 / 25.2 / 6.30 | 84.5 / 99.2 / 6.22 | 86.9 / 193.0 / 6.07 | 86.7 / 386.8 / 6.13 | 107.3 / 625.2 / 5.02 | 136.4 / 984.2 / 4.06 | 503.0 / 1067.3 / 1.27 | 1489.5 / 1441.8 / 0.66 |
| vllm_triton_w4a16 w4 sym | 177.8 / 2.9 / 0.76 | 177.1 / 11.8 / 0.77 | 180.1 / 46.6 / 0.76 | 179.9 / 93.3 / 0.77 | 175.2 / 191.6 / 0.81 | 202.6 / 331.3 / 0.74 | 389.2 / 344.9 / 0.42 | 1217.5 / 441.0 / 0.21 | 4569.2 / 470.0 / 0.13 |
| vllm_triton_w4a16 w4 zp | 192.7 / 2.7 / 0.71 | 192.4 / 10.9 / 0.71 | 193.2 / 43.4 / 0.71 | 193.1 / 86.9 / 0.72 | 185.8 / 180.5 / 0.77 | 217.6 / 308.4 / 0.69 | 419.9 / 319.6 / 0.39 | 1310.5 / 409.7 / 0.19 | 4988.0 / 430.5 / 0.12 |
| dequant_triton+hipblaslt w4 sym | 219.9 / 2.4 / 0.61 | 219.8 / 9.5 / 0.62 | 220.2 / 38.1 / 0.62 | 222.8 / 75.3 / 0.62 | 222.9 / 150.6 / 0.64 | 240.8 / 278.7 / 0.62 | 261.7 / 512.9 / 0.63 | 626.6 / 856.8 / 0.40 | 1620.7 / 1325.0 / 0.37 |
| dequant_triton+hipblaslt w4 zp | 226.9 / 2.3 / 0.60 | 227.4 / 9.2 / 0.60 | 227.8 / 36.8 / 0.61 | 230.4 / 72.8 / 0.61 | 230.5 / 145.5 / 0.62 | 248.4 / 270.2 / 0.61 | 269.4 / 498.2 / 0.61 | 626.2 / 857.3 / 0.40 | 1625.4 / 1321.2 / 0.37 |
| dequant_triton+hipblaslt w8 sym | 233.7 / 2.2 / 1.14 | 233.6 / 9.0 / 1.14 | 235.0 / 35.7 / 1.14 | 236.8 / 70.9 / 1.14 | 237.2 / 141.5 / 1.15 | 255.1 / 263.1 / 1.10 | 278.6 / 481.8 / 1.06 | 635.0 / 845.5 / 0.60 | 1636.0 / 1312.6 / 0.45 |
| dequant_triton+hipblaslt w8 zp | 238.6 / 2.2 / 1.12 | 237.5 / 8.8 / 1.13 | 238.4 / 35.2 / 1.13 | 243.5 / 68.9 / 1.12 | 241.6 / 138.9 / 1.14 | 261.0 / 257.1 / 1.08 | 283.3 / 473.8 / 1.05 | 639.6 / 839.4 / 0.60 | 1640.2 / 1309.3 / 0.45 |
| aiter_a16w8_int8 w8 sym | 65.3 / 8.0 / 4.14 | 71.9 / 29.2 / 3.77 | 87.9 / 95.4 / 3.10 | 93.9 / 178.8 / 2.92 | 118.5 / 283.2 / 2.34 | 196.5 / 341.5 / 1.45 | 318.8 / 421.0 / 0.94 | 956.3 / 561.4 / 0.40 | 3550.2 / 604.9 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 48.7 / 10.8 / 5.38 | 55.1 / 38.0 / 4.76 | 69.8 / 120.2 / 3.78 | 86.9 / 193.1 / 3.06 | 109.6 / 306.0 / 2.46 | 162.5 / 412.9 / 1.70 | 318.4 / 421.5 / 0.91 | 808.0 / 664.4 / 0.47 | 3002.4 / 715.3 / 0.24 |
| humming w4 sym | 33.6 / 15.6 / 4.02 | 34.8 / 60.2 / 3.89 | 34.8 / 241.0 / 3.94 | 41.6 / 403.4 / 3.34 | 55.9 / 600.1 / 2.55 | 85.1 / 788.9 / 1.76 | 149.4 / 898.4 / 1.10 | 509.4 / 1053.8 / 0.49 | 1954.0 / 1099.0 / 0.30 |
| humming w4 zp | 35.4 / 14.8 / 3.88 | 36.6 / 57.2 / 3.76 | 36.9 / 227.6 / 3.77 | 43.3 / 387.6 / 3.25 | 57.1 / 588.0 / 2.53 | 86.2 / 778.2 / 1.76 | 154.3 / 869.8 / 1.08 | 525.0 / 1022.7 / 0.48 | 2017.7 / 1064.3 / 0.30 |
| humming w8 sym | 51.2 / 10.2 / 5.20 | 53.9 / 38.9 / 4.95 | 53.3 / 157.3 / 5.03 | 57.0 / 294.3 / 4.73 | 68.5 / 490.0 / 3.99 | 99.0 / 678.0 / 2.83 | 175.7 / 763.8 / 1.68 | 586.4 / 915.5 / 0.65 | 2229.8 / 963.1 / 0.33 |
| humming w8 zp | 52.6 / 10.0 / 5.10 | 54.7 / 38.3 / 4.91 | 54.4 / 154.2 / 4.97 | 58.1 / 288.7 / 4.68 | 69.8 / 480.4 / 3.94 | 98.1 / 684.3 / 2.88 | 180.0 / 745.5 / 1.65 | 598.8 / 896.6 / 0.64 | 2281.6 / 941.2 / 0.32 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 45.7 / 5.7 / 5.74 | 46.1 / 22.7 / 5.69 | 47.8 / 87.8 / 5.51 | 50.6 / 165.9 / 5.22 | 56.3 / 297.7 / 4.72 | 67.9 / 494.0 / 3.98 | 90.5 / 741.9 / 3.07 | 230.9 / 1162.7 / 1.41 | 835.0 / 1285.9 / 0.61 |
| vllm_triton_w4a16 w4 sym | 517.6 / 0.5 / 0.13 | 516.1 / 2.0 / 0.13 | 516.4 / 8.1 / 0.13 | 515.0 / 16.3 / 0.14 | 510.1 / 32.9 / 0.14 | 823.0 / 40.8 / 0.09 | 818.6 / 82.0 / 0.10 | 950.3 / 282.5 / 0.14 | 2533.3 / 423.9 / 0.13 |
| vllm_triton_w4a16 w4 zp | 540.9 / 0.5 / 0.13 | 541.8 / 1.9 / 0.13 | 541.6 / 7.7 / 0.13 | 542.2 / 15.5 / 0.13 | 535.4 / 31.3 / 0.14 | 762.6 / 44.0 / 0.10 | 760.5 / 88.2 / 0.11 | 1052.1 / 255.1 / 0.12 | 2611.0 / 411.2 / 0.12 |
| dequant_triton+hipblaslt w4 sym | 118.3 / 2.2 / 0.57 | 118.6 / 8.8 / 0.57 | 119.8 / 35.0 / 0.57 | 121.8 / 68.9 / 0.57 | 126.3 / 132.8 / 0.57 | 142.8 / 235.1 / 0.53 | 171.7 / 390.8 / 0.48 | 299.9 / 895.0 / 0.43 | 907.0 / 1183.9 / 0.35 |
| dequant_triton+hipblaslt w4 zp | 127.0 / 2.1 / 0.54 | 127.2 / 8.2 / 0.54 | 128.2 / 32.7 / 0.54 | 130.3 / 64.4 / 0.54 | 135.3 / 124.0 / 0.53 | 150.7 / 222.6 / 0.50 | 179.2 / 374.4 / 0.47 | 306.3 / 876.4 / 0.43 | 914.2 / 1174.5 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 118.2 / 2.2 / 1.13 | 119.1 / 8.8 / 1.12 | 120.5 / 34.8 / 1.11 | 123.3 / 68.0 / 1.10 | 128.1 / 131.0 / 1.07 | 141.3 / 237.5 / 1.00 | 171.7 / 390.8 / 0.87 | 301.4 / 890.8 / 0.65 | 906.7 / 1184.2 / 0.42 |
| dequant_triton+hipblaslt w8 zp | 124.0 / 2.1 / 1.08 | 124.5 / 8.4 / 1.08 | 126.1 / 33.3 / 1.07 | 128.9 / 65.1 / 1.06 | 134.1 / 125.1 / 1.03 | 147.8 / 227.0 / 0.96 | 176.8 / 379.5 / 0.85 | 307.0 / 874.4 / 0.64 | 916.3 / 1171.9 / 0.42 |
| aiter_a16w8_int8 w8 sym | 34.8 / 7.5 / 3.89 | 35.3 / 29.7 / 3.84 | 52.3 / 80.2 / 2.60 | 54.6 / 153.8 / 2.51 | 65.1 / 257.6 / 2.14 | 115.3 / 291.1 / 1.24 | 214.2 / 313.2 / 0.70 | 682.3 / 393.4 / 0.29 | 1698.8 / 632.1 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 29.7 / 8.8 / 4.41 | 30.5 / 34.4 / 4.30 | 32.4 / 129.3 / 4.07 | 52.9 / 158.6 / 2.52 | 61.0 / 275.1 / 2.21 | 93.5 / 359.0 / 1.49 | 215.4 / 311.5 / 0.68 | 571.0 / 470.1 / 0.34 | 1429.5 / 751.1 / 0.27 |
| humming w4 sym | 28.5 / 9.2 / 2.37 | 30.7 / 34.1 / 2.21 | 24.1 / 174.3 / 2.85 | 28.7 / 292.4 / 2.42 | 38.5 / 435.4 / 1.86 | 58.5 / 573.1 / 1.29 | 96.3 / 697.0 / 0.86 | 321.9 / 834.0 / 0.41 | 961.5 / 1116.8 / 0.33 |
| humming w4 zp | 30.1 / 8.7 / 2.28 | 32.5 / 32.3 / 2.12 | 25.1 / 167.3 / 2.77 | 30.0 / 279.4 / 2.35 | 40.2 / 417.8 / 1.81 | 59.3 / 565.6 / 1.29 | 98.7 / 679.9 / 0.85 | 332.9 / 806.5 / 0.40 | 989.9 / 1084.7 / 0.32 |
| humming w8 sym | 36.1 / 7.3 / 3.69 | 37.3 / 28.1 / 3.58 | 33.8 / 124.0 / 3.96 | 37.9 / 221.1 / 3.56 | 46.7 / 358.9 / 2.93 | 67.3 / 498.4 / 2.09 | 105.2 / 638.1 / 1.42 | 344.9 / 778.4 / 0.57 | 1028.8 / 1043.7 / 0.37 |
| humming w8 zp | 37.2 / 7.0 / 3.60 | 38.6 / 27.1 / 3.48 | 34.5 / 121.6 / 3.92 | 38.5 / 217.9 / 3.54 | 47.9 / 350.1 / 2.88 | 67.4 / 497.8 / 2.11 | 106.1 / 632.4 / 1.41 | 350.0 / 766.9 / 0.56 | 1054.5 / 1018.3 / 0.37 |
