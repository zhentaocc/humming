# ROCm baseline GEMM benchmark (2026-10-28T05:28:52)

Y[M,N] = X[M,K] @ W[N,K]^T; weight-only quant unless noted; group size 128; zp = asymmetric zero point, sym = uint{4,8}b{8,128}.

## Environment

- device: `AMD Instinct MI355X (gfx950:sramecc+:xnack-), 256 CUs, HIP_VISIBLE_DEVICES=5`
- torch: `2.11.0+rocm7.2`
- triton: `3.6.0`
- hip: `7.2.26015`
- date: `2026-10-28T05:28:52`
- args: `--out-dir benchmarks/rocm/results --tag 2026-10-28`

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
| measured: copy 4 GiB (read+write) TB/s | 5.07 |
| measured: sum 4 GiB (read-only) TB/s | 2.79 |
| measured: fp16 8192^3 F.linear TFLOPs | 1471.70 |
| measured: bf16 8192^3 F.linear TFLOPs | 1593.38 |

## Methodology

- HIP-event timing; warmup >= 20; >= 100 timed calls; calls captured into a HIP graph (no host launch overhead) unless the `timing` column says eager.
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
| vllm_awq_triton_gemm [fp16 w4 sym] | SKIP: unsupported: AWQ format always carries zero points | vllm@5efbf0886a layers/quantization/awq_triton.py awq_gemm_triton (split_k=8, as auto_awq.py) |
| vllm_awq_triton_gemm [fp16 w4 zp] | OK: ok (max rel err 5.06e-04) | vllm@5efbf0886a layers/quantization/awq_triton.py awq_gemm_triton (split_k=8, as auto_awq.py) |
| vllm_awq_dequant_mm [fp16 w4 sym] | SKIP: unsupported: AWQ format always carries zero points | vllm@5efbf0886a awq_dequantize_triton + torch.matmul (vLLM AWQ path for M>=256) |
| vllm_awq_dequant_mm [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | vllm@5efbf0886a awq_dequantize_triton + torch.matmul (vLLM AWQ path for M>=256) |
| conch_mixed_precision [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [fp16 w8 sym] | OK: ok (max rel err 3.17e-04) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [fp16 w8 zp] | OK: ok (max rel err 3.12e-04) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| vllm_exllama_gptq [fp16 w4 sym] | OK: ok (max rel err 6.05e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w4 zp] | OK: ok (max rel err 6.39e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 sym] | OK: ok (max rel err 7.85e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| vllm_exllama_gptq [fp16 w8 zp] | OK: ok (max rel err 7.95e-04) | vllm@5efbf0886a csrc/libtorch_stable/quantization/gptq/q_gemm.cu gptq_gemm(use_exllama=True, v1 zeros) |
| dequant_triton+hipblaslt [fp16 w4 sym] | OK: ok (max rel err 2.74e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w4 zp] | OK: ok (max rel err 3.15e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 sym] | OK: ok (max rel err 3.17e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| dequant_triton+hipblaslt [fp16 w8 zp] | OK: ok (max rel err 3.12e-04) | this repo: simple Triton dequant kernel ([N,K/pf] int32 -> [N,K] fp16/bf16) + F.linear (hipBLASLt), timed together |
| aiter_a16w8_int8 [fp16 w8 sym] | OK: ok (max rel err 2.12e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_int8 [fp16 w8 zp] | SKIP: unsupported: symmetric only (no zero-point input) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16w8_blockscale.py with int8 W, scale block (1 x G) |
| aiter_a16w8_fp8_block128 [fp16 w8 sym] | OK: ok (max rel err 2.13e-04) | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| aiter_a16w8_fp8_block128 [fp16 w8 zp] | SKIP: unsupported: fp8 weights: no zero point | aiter@cdf01e2a4f gemm_a16w8_blockscale, FP8-e4m3 W with 128x128 block scales (own quant) |
| aiter_a16wfp4 [fp16 w4 sym] | OK: ok (max rel err 2.02e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16wfp4.py (MXFP4 W; NOTE: X is MXFP4-quantized in-kernel) |
| aiter_a16wfp4 [fp16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16wfp4.py (MXFP4 W; NOTE: X is MXFP4-quantized in-kernel) |
| aiter_afp4wfp4 [fp16 w4 sym] | OK: ok (max rel err 2.02e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_afp4wfp4.py (MXFP4 x MXFP4, X pre-quantized, quant not timed) |
| aiter_afp4wfp4 [fp16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_afp4wfp4.py (MXFP4 x MXFP4, X pre-quantized, quant not timed) |
| torch_scaled_mm_fp8 [fp16 w8 sym] | SKIP: error: ValueError: hipblaslt rowwise _scaled_mm only supports BFloat16 output but got Half | torch._scaled_mm (hipBLASLt) FP8-e4m3 rowwise X / per-channel W, torch 2.11.0+rocm7.2, X quant not timed |
| torch_scaled_mm_fp8 [fp16 w8 zp] | SKIP: unsupported: fp8: no zero point | torch._scaled_mm (hipBLASLt) FP8-e4m3 rowwise X / per-channel W, torch 2.11.0+rocm7.2, X quant not timed |
| aiter_a8w8_int8 [fp16 w8 sym] | OK: ok (max rel err 2.09e-04) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a8w8.py INT8 per-token X / per-channel W, X quant not timed |
| aiter_a8w8_int8 [fp16 w8 zp] | SKIP: unsupported: int8 W8A8 symmetric only | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a8w8.py INT8 per-token X / per-channel W, X quant not timed |
| aiter_gemm_a4w4_asm [fp16 w4 sym] | SKIP: unsupported: asm a4w4 kernels emit bf16 only | aiter@cdf01e2a4f ops/gemm_op_a4w4.py gemm_a4w4 (gfx950 asm MXFP4 x MXFP4, preshuffled W; X quant not timed) |
| aiter_gemm_a4w4_asm [fp16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/gemm_op_a4w4.py gemm_a4w4 (gfx950 asm MXFP4 x MXFP4, preshuffled W; X quant not timed) |
| torch_dense [bf16 dense] | OK: ok (max rel err 2.21e-03) | F.linear (hipBLASLt), torch 2.11.0+rocm7.2 |
| vllm_triton_w4a16 [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_triton_w4a16 [bf16 w4 zp] | OK: ok (max rel err 2.47e-03) | vllm@5efbf0886a kernels/linear/mixed_precision/triton_w4a16.py (TritonW4A16LinearKernel) |
| vllm_awq_triton_gemm [bf16 w4 sym] | SKIP: unsupported: AWQ format always carries zero points | vllm@5efbf0886a layers/quantization/awq_triton.py awq_gemm_triton (split_k=8, as auto_awq.py) |
| vllm_awq_triton_gemm [bf16 w4 zp] | SKIP: error: CompilationError: at 108:22: | vllm@5efbf0886a layers/quantization/awq_triton.py awq_gemm_triton (split_k=8, as auto_awq.py) |
| vllm_awq_dequant_mm [bf16 w4 sym] | SKIP: unsupported: AWQ format always carries zero points | vllm@5efbf0886a awq_dequantize_triton + torch.matmul (vLLM AWQ path for M>=256) |
| vllm_awq_dequant_mm [bf16 w4 zp] | OK: ok (max rel err 2.47e-03) | vllm@5efbf0886a awq_dequantize_triton + torch.matmul (vLLM AWQ path for M>=256) |
| conch_mixed_precision [bf16 w4 sym] | OK: ok (max rel err 2.21e-03) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [bf16 w4 zp] | OK: ok (max rel err 2.47e-03) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [bf16 w8 sym] | OK: ok (max rel err 2.40e-03) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
| conch_mixed_precision [bf16 w8 zp] | OK: ok (max rel err 2.53e-03) | conch-triton-kernels==1.3 mixed_precision_gemm (vLLM ConchLinearKernel layout) |
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
| aiter_a16wfp4 [bf16 w4 sym] | OK: ok (max rel err 1.69e-03) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16wfp4.py (MXFP4 W; NOTE: X is MXFP4-quantized in-kernel) |
| aiter_a16wfp4 [bf16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a16wfp4.py (MXFP4 W; NOTE: X is MXFP4-quantized in-kernel) |
| aiter_afp4wfp4 [bf16 w4 sym] | OK: ok (max rel err 1.69e-03) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_afp4wfp4.py (MXFP4 x MXFP4, X pre-quantized, quant not timed) |
| aiter_afp4wfp4 [bf16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_afp4wfp4.py (MXFP4 x MXFP4, X pre-quantized, quant not timed) |
| torch_scaled_mm_fp8 [bf16 w8 sym] | OK: ok (max rel err 1.70e-03) | torch._scaled_mm (hipBLASLt) FP8-e4m3 rowwise X / per-channel W, torch 2.11.0+rocm7.2, X quant not timed |
| torch_scaled_mm_fp8 [bf16 w8 zp] | SKIP: unsupported: fp8: no zero point | torch._scaled_mm (hipBLASLt) FP8-e4m3 rowwise X / per-channel W, torch 2.11.0+rocm7.2, X quant not timed |
| aiter_a8w8_int8 [bf16 w8 sym] | OK: ok (max rel err 1.72e-03) | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a8w8.py INT8 per-token X / per-channel W, X quant not timed |
| aiter_a8w8_int8 [bf16 w8 zp] | SKIP: unsupported: int8 W8A8 symmetric only | aiter@cdf01e2a4f ops/triton/gemm/basic/gemm_a8w8.py INT8 per-token X / per-channel W, X quant not timed |
| aiter_gemm_a4w4_asm [bf16 w4 sym] | OK: ok (max rel err 1.69e-03) | aiter@cdf01e2a4f ops/gemm_op_a4w4.py gemm_a4w4 (gfx950 asm MXFP4 x MXFP4, preshuffled W; X quant not timed) |
| aiter_gemm_a4w4_asm [bf16 w4 zp] | SKIP: unsupported: MXFP4: no zero point | aiter@cdf01e2a4f ops/gemm_op_a4w4.py gemm_a4w4 (gfx950 asm MXFP4 x MXFP4, preshuffled W; X quant not timed) |

## Headline (fp16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 (2.73 TB/s) | 10.3 (3.28 TB/s) | 44.0 (781 TF) | 100.3 (1371 TF) | 157.3 (5.97 TB/s) | 156.7 (6.01 TB/s) | 727.8 (1322 TF) | 2580.6 (1491 TF) | 75.3 (6.24 TB/s) | 76.6 (6.15 TB/s) | 366.0 (1314 TF) | 1469.8 (1309 TF) |
| vllm_triton_w4a16 w4 sym | 75.5 (0.12 TB/s) | 76.1 (0.12 TB/s) | 106.4 (323 TF) | 252.8 (544 TF) | 200.1 (1.21 TB/s) | 231.7 (1.05 TB/s) | 1696.1 (567 TF) | 6390.8 (602 TF) | 510.0 (0.24 TB/s) | 510.3 (0.24 TB/s) | 844.8 (569 TF) | 3254.5 (591 TF) |
| vllm_triton_w4a16 w4 zp | 81.2 (0.11 TB/s) | 81.8 (0.11 TB/s) | 114.2 (301 TF) | 270.5 (508 TF) | 221.1 (1.10 TB/s) | 241.9 (1.02 TB/s) | 1771.4 (543 TF) | 8907.0 (432 TF) | 541.6 (0.23 TB/s) | 546.7 (0.23 TB/s) | 902.2 (533 TF) | 3460.7 (556 TF) |
| vllm_awq_triton_gemm w4 zp | 40.4 (0.22 TB/s) | 41.5 (0.22 TB/s) | 1129.0 (30 TF) | 3727.9 (37 TF) | 959.8 (0.25 TB/s) | 905.3 (0.27 TB/s) | 22168.0 (43 TF) | 89745.6 (43 TF) | 420.6 (0.29 TB/s) | 387.1 (0.32 TB/s) | 10835.8 (44 TF) | 43183.4 (45 TF) |
| vllm_awq_dequant_mm w4 zp | 28.2 (0.31 TB/s) | 30.1 (0.30 TB/s) | 60.8 (565 TF) | 129.5 (1061 TF) | 382.2 (0.64 TB/s) | 383.0 (0.64 TB/s) | 997.9 (964 TF) | 3139.5 (1226 TF) | 192.2 (0.64 TB/s) | 192.6 (0.64 TB/s) | 492.2 (977 TF) | 1516.4 (1269 TF) |
| conch_mixed_precision w4 sym | 93.3 (0.09 TB/s) | 100.1 (0.09 TB/s) | 505.5 (68 TF) | 1987.3 (69 TF) | 490.4 (0.49 TB/s) | 623.8 (0.39 TB/s) | 13718.8 (70 TF) | 56003.1 (69 TF) | 615.3 (0.20 TB/s) | 671.3 (0.18 TB/s) | 6858.7 (70 TF) | 27287.1 (71 TF) |
| conch_mixed_precision w4 zp | 100.0 (0.09 TB/s) | 109.9 (0.08 TB/s) | 529.9 (65 TF) | 2086.2 (66 TF) | 773.6 (0.32 TB/s) | 851.7 (0.29 TB/s) | 14494.4 (66 TF) | 57843.2 (67 TF) | 665.4 (0.18 TB/s) | 738.0 (0.17 TB/s) | 7174.6 (67 TF) | 28602.7 (67 TF) |
| conch_mixed_precision w8 sym | 100.8 (0.17 TB/s) | 107.0 (0.16 TB/s) | 529.2 (65 TF) | 2083.2 (66 TF) | 598.9 (0.80 TB/s) | 703.4 (0.68 TB/s) | 14478.4 (66 TF) | 57728.8 (67 TF) | 674.0 (0.35 TB/s) | 724.2 (0.33 TB/s) | 7235.7 (66 TF) | 28805.3 (67 TF) |
| conch_mixed_precision w8 zp | 112.1 (0.15 TB/s) | 122.6 (0.14 TB/s) | 554.3 (62 TF) | 2215.1 (62 TF) | 872.6 (0.55 TB/s) | 959.3 (0.50 TB/s) | 15267.3 (63 TF) | 60904.8 (63 TF) | 762.1 (0.32 TB/s) | 822.3 (0.29 TB/s) | 7591.6 (63 TF) | 30283.9 (64 TF) |
| vllm_exllama_gptq w4 sym | 53.8 (0.16 TB/s) | 122.5 (0.07 TB/s) | 68.3 (503 TF) | 142.3 (965 TF) | 259.8 (0.93 TB/s) | 349.4 (0.70 TB/s) | 1026.5 (937 TF) | 3230.4 (1191 TF) | 1385.0 (0.09 TB/s) | 1135.0 (0.11 TB/s) | 710.5 (677 TF) | 1566.0 (1229 TF) |
| vllm_exllama_gptq w4 zp | 53.3 (0.16 TB/s) | 123.2 (0.07 TB/s) | 68.1 (504 TF) | 141.6 (971 TF) | 260.0 (0.94 TB/s) | 349.1 (0.70 TB/s) | 1031.8 (932 TF) | 3225.0 (1193 TF) | 1380.6 (0.09 TB/s) | 1113.9 (0.11 TB/s) | 712.0 (676 TF) | 1571.3 (1225 TF) |
| vllm_exllama_gptq w8 sym | 59.7 (0.29 TB/s) | 124.5 (0.14 TB/s) | 70.5 (488 TF) | 146.1 (940 TF) | 252.6 (1.89 TB/s) | 433.1 (1.11 TB/s) | 1080.0 (891 TF) | 3269.7 (1177 TF) | 1351.9 (0.18 TB/s) | 829.8 (0.29 TB/s) | 716.3 (672 TF) | 1584.1 (1215 TF) |
| vllm_exllama_gptq w8 zp | 59.8 (0.29 TB/s) | 124.8 (0.14 TB/s) | 70.4 (488 TF) | 146.5 (938 TF) | 255.1 (1.89 TB/s) | 431.1 (1.12 TB/s) | 1072.1 (897 TF) | 3274.9 (1175 TF) | 1345.7 (0.18 TB/s) | 824.6 (0.29 TB/s) | 714.0 (674 TF) | 1585.9 (1213 TF) |
| dequant_triton+hipblaslt w4 sym | 24.6 (0.35 TB/s) | 23.1 (0.39 TB/s) | 54.1 (635 TF) | 112.6 (1220 TF) | 414.0 (0.58 TB/s) | 404.8 (0.60 TB/s) | 969.1 (993 TF) | 2797.3 (1376 TF) | 191.0 (0.64 TB/s) | 214.3 (0.57 TB/s) | 491.9 (978 TF) | 1351.1 (1424 TF) |
| dequant_triton+hipblaslt w4 zp | 24.9 (0.35 TB/s) | 23.2 (0.39 TB/s) | 54.4 (631 TF) | 116.4 (1181 TF) | 428.8 (0.57 TB/s) | 443.6 (0.56 TB/s) | 997.9 (964 TF) | 2827.3 (1361 TF) | 215.3 (0.57 TB/s) | 234.8 (0.53 TB/s) | 506.1 (950 TF) | 1368.4 (1406 TF) |
| dequant_triton+hipblaslt w8 sym | 25.5 (0.67 TB/s) | 23.1 (0.75 TB/s) | 54.5 (631 TF) | 114.6 (1199 TF) | 457.1 (1.04 TB/s) | 454.8 (1.05 TB/s) | 1000.4 (962 TF) | 2846.7 (1352 TF) | 222.3 (1.07 TB/s) | 229.6 (1.04 TB/s) | 490.1 (981 TF) | 1363.2 (1412 TF) |
| dequant_triton+hipblaslt w8 zp | 25.3 (0.68 TB/s) | 23.5 (0.74 TB/s) | 53.9 (638 TF) | 114.5 (1200 TF) | 467.2 (1.03 TB/s) | 458.3 (1.05 TB/s) | 1026.0 (938 TF) | 2891.6 (1331 TF) | 236.0 (1.02 TB/s) | 224.1 (1.08 TB/s) | 495.4 (971 TF) | 1378.4 (1396 TF) |
| aiter_a16w8_int8 w8 sym | 9.2 (1.88 TB/s) | 10.6 (1.65 TB/s) | 71.6 (480 TF) | 256.7 (535 TF) | 96.5 (5.02 TB/s) | 126.4 (3.85 TB/s) | 1725.4 (558 TF) | 6637.1 (580 TF) | 51.5 (4.70 TB/s) | 59.0 (4.13 TB/s) | 866.4 (555 TF) | 3311.4 (581 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.17 TB/s) | 9.5 (1.79 TB/s) | 64.0 (537 TF) | 204.8 (671 TF) | 77.9 (6.04 TB/s) | 102.7 (4.60 TB/s) | 1362.0 (706 TF) | 5224.9 (737 TF) | 43.6 (5.39 TB/s) | 48.9 (4.83 TB/s) | 675.1 (713 TF) | 2570.9 (748 TF) |
| aiter_a16wfp4 w4 sym | 16.3 (0.55 TB/s) | 16.7 (0.55 TB/s) | 283.3 (121 TF) | 1119.9 (123 TF) | 51.9 (4.81 TB/s) | 194.3 (1.29 TB/s) | 7167.0 (134 TF) | 28994.6 (133 TF) | 104.7 (1.19 TB/s) | 105.7 (1.19 TB/s) | 3747.6 (128 TF) | 14864.5 (129 TF) |
| aiter_afp4wfp4 w4 sym | 7.3 (1.23 TB/s) | 7.0 (1.30 TB/s) | 48.8 (704 TF) | 55.9 (2458 TF) | 72.0 (3.47 TB/s) | 68.9 (3.65 TB/s) | 356.2 (2701 TF) | 1243.6 (3095 TF) | 36.5 (3.42 TB/s) | 36.0 (3.48 TB/s) | 292.5 (1645 TF) | 609.1 (3159 TF) |
| aiter_a8w8_int8 w8 sym | 8.5 (1.98 TB/s) | 9.9 (1.72 TB/s) | 38.8 (885 TF) | 105.4 (1304 TF) | 108.6 (4.33 TB/s) | 94.4 (5.00 TB/s) | 642.3 (1498 TF) | 2452.3 (1569 TF) | 58.5 (4.02 TB/s) | 68.2 (3.46 TB/s) | 331.3 (1452 TF) | 1201.1 (1602 TF) |

## Headline (bf16): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 | llama3-70b/down M=1 | llama3-70b/down M=16 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 (2.72 TB/s) | 10.4 (3.27 TB/s) | 48.0 (715 TF) | 96.0 (1432 TF) | 155.3 (6.05 TB/s) | 159.3 (5.91 TB/s) | 707.4 (1360 TF) | 2466.5 (1560 TF) | 75.4 (6.23 TB/s) | 76.9 (6.12 TB/s) | 350.1 (1374 TF) | 1182.5 (1627 TF) |
| vllm_triton_w4a16 w4 sym | 85.4 (0.10 TB/s) | 85.4 (0.10 TB/s) | 136.2 (252 TF) | 312.5 (440 TF) | 285.4 (0.85 TB/s) | 297.3 (0.82 TB/s) | 2088.4 (461 TF) | 8047.5 (478 TF) | 575.2 (0.21 TB/s) | 575.8 (0.21 TB/s) | 1061.5 (453 TF) | 4143.3 (464 TF) |
| vllm_triton_w4a16 w4 zp | 89.0 (0.10 TB/s) | 89.6 (0.10 TB/s) | 129.1 (266 TF) | 348.8 (394 TF) | 312.6 (0.78 TB/s) | 324.4 (0.76 TB/s) | 2241.4 (429 TF) | 8798.8 (437 TF) | 603.2 (0.20 TB/s) | 606.4 (0.20 TB/s) | 1175.4 (409 TF) | 4527.2 (425 TF) |
| vllm_awq_dequant_mm w4 zp | 27.6 (0.32 TB/s) | 30.2 (0.30 TB/s) | 56.6 (607 TF) | 117.9 (1166 TF) | 396.3 (0.62 TB/s) | 397.5 (0.62 TB/s) | 951.7 (1011 TF) | 2913.4 (1321 TF) | 192.2 (0.64 TB/s) | 192.8 (0.64 TB/s) | 450.9 (1067 TF) | 1414.6 (1360 TF) |
| conch_mixed_precision w4 sym | 102.8 (0.08 TB/s) | 109.9 (0.08 TB/s) | 521.7 (66 TF) | 2073.6 (66 TF) | 795.9 (0.30 TB/s) | 859.6 (0.28 TB/s) | 14195.5 (68 TF) | 56687.0 (68 TF) | 683.0 (0.18 TB/s) | 739.9 (0.17 TB/s) | 7051.5 (68 TF) | 28093.6 (68 TF) |
| conch_mixed_precision w4 zp | 113.4 (0.08 TB/s) | 121.2 (0.07 TB/s) | 544.9 (63 TF) | 2134.7 (64 TF) | 874.0 (0.28 TB/s) | 935.3 (0.26 TB/s) | 14926.0 (64 TF) | 59712.7 (64 TF) | 759.1 (0.16 TB/s) | 811.3 (0.15 TB/s) | 7364.2 (65 TF) | 29311.8 (66 TF) |
| conch_mixed_precision w8 sym | 105.9 (0.16 TB/s) | 113.7 (0.15 TB/s) | 540.9 (64 TF) | 2134.2 (64 TF) | 821.1 (0.58 TB/s) | 899.4 (0.53 TB/s) | 14844.4 (65 TF) | 59126.7 (65 TF) | 711.8 (0.34 TB/s) | 780.1 (0.31 TB/s) | 7409.8 (65 TF) | 29464.5 (65 TF) |
| conch_mixed_precision w8 zp | 127.1 (0.14 TB/s) | 136.5 (0.13 TB/s) | 576.7 (60 TF) | 2255.6 (61 TF) | 988.7 (0.49 TB/s) | 1066.0 (0.45 TB/s) | 15926.3 (60 TF) | 63526.6 (61 TF) | 860.6 (0.28 TB/s) | 925.2 (0.26 TB/s) | 7871.8 (61 TF) | 31410.9 (61 TF) |
| dequant_triton+hipblaslt w4 sym | 24.8 (0.35 TB/s) | 22.9 (0.39 TB/s) | 59.6 (577 TF) | 110.0 (1250 TF) | 417.4 (0.58 TB/s) | 423.8 (0.58 TB/s) | 979.4 (982 TF) | 2710.8 (1420 TF) | 202.9 (0.60 TB/s) | 210.6 (0.58 TB/s) | 473.8 (1015 TF) | 1303.7 (1476 TF) |
| dequant_triton+hipblaslt w4 zp | 25.4 (0.34 TB/s) | 23.0 (0.39 TB/s) | 60.4 (569 TF) | 113.2 (1215 TF) | 450.5 (0.54 TB/s) | 431.1 (0.57 TB/s) | 969.9 (992 TF) | 2725.8 (1412 TF) | 226.2 (0.54 TB/s) | 228.4 (0.54 TB/s) | 479.4 (1003 TF) | 1314.0 (1464 TF) |
| dequant_triton+hipblaslt w8 sym | 25.6 (0.67 TB/s) | 23.2 (0.74 TB/s) | 60.5 (568 TF) | 109.4 (1256 TF) | 479.7 (0.99 TB/s) | 451.8 (1.06 TB/s) | 989.8 (972 TF) | 2739.4 (1405 TF) | 221.3 (1.08 TB/s) | 226.7 (1.06 TB/s) | 483.9 (994 TF) | 1310.7 (1468 TF) |
| dequant_triton+hipblaslt w8 zp | 25.3 (0.68 TB/s) | 23.0 (0.76 TB/s) | 60.5 (568 TF) | 110.7 (1242 TF) | 461.8 (1.04 TB/s) | 448.1 (1.08 TB/s) | 991.4 (970 TF) | 2752.8 (1398 TF) | 239.5 (1.00 TB/s) | 239.2 (1.01 TB/s) | 492.8 (976 TF) | 1328.2 (1449 TF) |
| aiter_a16w8_int8 w8 sym | 9.7 (1.78 TB/s) | 12.6 (1.39 TB/s) | 66.9 (513 TF) | 241.8 (568 TF) | 104.2 (4.65 TB/s) | 135.7 (3.58 TB/s) | 1582.0 (608 TF) | 6078.4 (633 TF) | 50.7 (4.78 TB/s) | 72.2 (3.37 TB/s) | 787.5 (611 TF) | 3021.2 (637 TF) |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 (2.15 TB/s) | 9.5 (1.79 TB/s) | 57.7 (596 TF) | 206.8 (664 TF) | 77.4 (6.08 TB/s) | 102.6 (4.60 TB/s) | 1345.9 (715 TF) | 5174.2 (744 TF) | 44.6 (5.27 TB/s) | 49.1 (4.81 TB/s) | 665.4 (723 TF) | 2542.3 (757 TF) |
| aiter_a16wfp4 w4 sym | 16.3 (0.55 TB/s) | 16.5 (0.56 TB/s) | 284.7 (121 TF) | 1131.8 (121 TF) | 51.4 (4.85 TB/s) | 182.3 (1.38 TB/s) | 8161.1 (118 TF) | 165880.3 (23 TF) | 449.1 (0.28 TB/s) | 531.6 (0.24 TB/s) | 7192.0 (67 TF) | 14944.0 (129 TF) |
| aiter_afp4wfp4 w4 sym | 7.3 (1.22 TB/s) | 7.0 (1.30 TB/s) | 49.0 (701 TF) | 56.0 (2453 TF) | 70.2 (3.56 TB/s) | 68.4 (3.68 TB/s) | 358.4 (2684 TF) | 1235.9 (3114 TF) | 36.0 (3.46 TB/s) | 37.1 (3.38 TB/s) | 292.9 (1642 TF) | 610.4 (3152 TF) |
| torch_scaled_mm_fp8 w8 sym | 12.9 (1.31 TB/s) | 8.7 (1.95 TB/s) | 28.0 (1226 TF) | 61.2 (2246 TF) | 89.9 (5.23 TB/s) | 91.3 (5.17 TB/s) | 357.8 (2688 TF) | 1542.5 (2495 TF) | 41.8 (5.62 TB/s) | 42.9 (5.50 TB/s) | 182.8 (2631 TF) | 734.0 (2621 TF) |
| aiter_a8w8_int8 w8 sym | 8.5 (1.99 TB/s) | 9.9 (1.72 TB/s) | 38.4 (895 TF) | 105.1 (1307 TF) | 107.9 (4.36 TB/s) | 95.6 (4.94 TB/s) | 642.1 (1498 TF) | 2452.2 (1569 TF) | 58.2 (4.04 TB/s) | 69.2 (3.40 TB/s) | 330.7 (1455 TF) | 1200.8 (1602 TF) |
| aiter_gemm_a4w4_asm w4 sym | 7.7 (1.16 TB/s) | 8.1 (1.12 TB/s) | 14.8 (2324 TF) | 36.2 (3799 TF) | 43.6 (5.72 TB/s) | 45.0 (5.59 TB/s) | 215.4 (4466 TF) | 729.8 (5273 TF) | 43.8 (2.85 TB/s) | 47.8 (2.62 TB/s) | 111.2 (4325 TF) | 341.1 (5642 TF) |

## Full results (fp16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 / 2.7 / 2.73 | 9.4 / 14.3 / 3.58 | 10.3 / 52.1 / 3.28 | 27.8 / 38.6 / 1.22 | 13.2 / 162.7 / 2.62 | 17.1 / 251.1 / 2.08 | 23.4 / 366.4 / 1.61 | 44.0 / 781.2 / 1.14 | 100.3 / 1370.7 / 1.00 |
| vllm_triton_w4a16 w4 sym | 75.5 / 0.4 / 0.12 | 75.8 / 1.8 / 0.12 | 76.1 / 7.1 / 0.12 | 77.0 / 13.9 / 0.12 | 78.0 / 27.5 / 0.12 | 105.4 / 40.8 / 0.10 | 104.3 / 82.3 / 0.12 | 106.4 / 323.0 / 0.24 | 252.8 / 543.6 / 0.30 |
| vllm_triton_w4a16 w4 zp | 81.2 / 0.4 / 0.11 | 81.3 / 1.7 / 0.11 | 81.8 / 6.6 / 0.11 | 82.0 / 13.1 / 0.11 | 85.9 / 25.0 / 0.11 | 112.7 / 38.1 / 0.10 | 111.6 / 77.0 / 0.12 | 114.2 / 300.7 / 0.22 | 270.5 / 508.0 / 0.28 |
| vllm_awq_triton_gemm w4 zp | 40.4 / 0.8 / 0.22 | 40.3 / 3.3 / 0.22 | 41.5 / 12.9 / 0.22 | 41.8 / 25.7 / 0.22 | 70.3 / 30.6 / 0.14 | 121.8 / 35.3 / 0.09 | 226.3 / 38.0 / 0.06 | 1129.0 / 30.4 / 0.02 | 3727.9 / 36.9 / 0.02 |
| vllm_awq_dequant_mm w4 zp | 28.2 / 1.2 / 0.31 | 28.2 / 4.8 / 0.31 | 30.1 / 17.8 / 0.30 | 32.8 / 32.7 / 0.28 | 33.8 / 63.5 / 0.29 | 35.2 / 122.0 / 0.31 | 40.3 / 212.9 / 0.32 | 60.8 / 565.5 / 0.42 | 129.5 / 1061.2 / 0.58 |
| conch_mixed_precision w4 sym | 93.3 / 0.4 / 0.09 | 93.2 / 1.4 / 0.09 | 100.1 / 5.4 / 0.09 | 117.9 / 9.1 / 0.08 | 156.1 / 13.8 / 0.06 | 256.4 / 16.7 / 0.04 | 256.0 / 33.6 / 0.05 | 505.5 / 68.0 / 0.05 | 1987.3 / 69.2 / 0.04 |
| conch_mixed_precision w4 zp | 100.0 / 0.3 / 0.09 | 100.3 / 1.3 / 0.09 | 109.9 / 4.9 / 0.08 | 127.8 / 8.4 / 0.07 | 166.6 / 12.9 / 0.06 | 266.9 / 16.1 / 0.04 | 266.5 / 32.2 / 0.05 | 529.9 / 64.8 / 0.05 | 2086.2 / 65.9 / 0.04 |
| conch_mixed_precision w8 sym | 100.8 / 0.3 / 0.17 | 100.9 / 1.3 / 0.17 | 107.0 / 5.0 / 0.16 | 125.3 / 8.6 / 0.14 | 164.4 / 13.1 / 0.11 | 269.3 / 15.9 / 0.07 | 268.4 / 32.0 / 0.08 | 529.2 / 64.9 / 0.06 | 2083.2 / 66.0 / 0.04 |
| conch_mixed_precision w8 zp | 112.1 / 0.3 / 0.15 | 112.1 / 1.2 / 0.15 | 122.6 / 4.4 / 0.14 | 139.8 / 7.7 / 0.13 | 178.9 / 12.0 / 0.10 | 280.4 / 15.3 / 0.07 | 280.0 / 30.7 / 0.08 | 554.3 / 62.0 / 0.06 | 2215.1 / 62.0 / 0.04 |
| vllm_exllama_gptq w4 sym | 53.8 / 0.6 / 0.16 | 71.9 / 1.9 / 0.12 | 122.5 / 4.4 / 0.07 | 172.3 / 6.2 / 0.05 | 41.9 / 51.2 / 0.23 | 44.4 / 96.8 / 0.24 | 50.7 / 169.5 / 0.25 | 68.3 / 503.2 / 0.37 | 142.3 / 965.5 / 0.53 |
| vllm_exllama_gptq w4 zp | 53.3 / 0.6 / 0.16 | 71.6 / 1.9 / 0.12 | 123.2 / 4.4 / 0.07 | 157.1 / 6.8 / 0.06 | 41.7 / 51.5 / 0.23 | 44.6 / 96.3 / 0.24 | 50.7 / 169.4 / 0.26 | 68.1 / 504.4 / 0.37 | 141.6 / 970.9 / 0.54 |
| vllm_exllama_gptq w8 sym | 59.7 / 0.6 / 0.29 | 72.3 / 1.9 / 0.24 | 124.5 / 4.3 / 0.14 | 44.8 / 24.0 / 0.39 | 45.3 / 47.5 / 0.40 | 48.2 / 89.1 / 0.40 | 54.1 / 158.7 / 0.39 | 70.5 / 487.7 / 0.48 | 146.1 / 940.5 / 0.58 |
| vllm_exllama_gptq w8 zp | 59.8 / 0.6 / 0.29 | 74.7 / 1.8 / 0.23 | 124.8 / 4.3 / 0.14 | 45.1 / 23.8 / 0.39 | 45.6 / 47.1 / 0.40 | 47.7 / 90.0 / 0.40 | 54.3 / 158.1 / 0.39 | 70.4 / 488.1 / 0.48 | 146.5 / 938.2 / 0.57 |
| dequant_triton+hipblaslt w4 sym | 24.6 / 1.4 / 0.35 | 22.0 / 6.1 / 0.40 | 23.1 / 23.3 / 0.39 | 24.5 / 43.8 / 0.38 | 24.6 / 87.4 / 0.40 | 28.6 / 150.3 / 0.38 | 33.3 / 258.2 / 0.39 | 54.1 / 635.3 / 0.47 | 112.6 / 1220.3 / 0.67 |
| dequant_triton+hipblaslt w4 zp | 24.9 / 1.3 / 0.35 | 22.3 / 6.0 / 0.39 | 23.2 / 23.1 / 0.39 | 24.6 / 43.7 / 0.38 | 24.9 / 86.1 / 0.39 | 28.7 / 149.6 / 0.38 | 32.9 / 260.8 / 0.39 | 54.4 / 631.3 / 0.47 | 116.4 / 1180.6 / 0.65 |
| dequant_triton+hipblaslt w8 sym | 25.5 / 1.3 / 0.67 | 22.1 / 6.1 / 0.78 | 23.1 / 23.2 / 0.75 | 25.0 / 42.9 / 0.70 | 25.3 / 85.0 / 0.72 | 28.5 / 150.9 / 0.67 | 33.8 / 254.4 / 0.63 | 54.5 / 630.7 / 0.62 | 114.6 / 1199.4 / 0.73 |
| dequant_triton+hipblaslt w8 zp | 25.3 / 1.3 / 0.68 | 22.3 / 6.0 / 0.77 | 23.5 / 22.8 / 0.74 | 25.2 / 42.6 / 0.70 | 26.7 / 80.3 / 0.68 | 28.6 / 149.9 / 0.67 | 34.0 / 252.8 / 0.63 | 53.9 / 637.5 / 0.63 | 114.5 / 1199.9 / 0.74 |
| aiter_a16w8_int8 w8 sym | 9.2 / 3.6 / 1.88 | 10.2 / 13.1 / 1.70 | 10.6 / 50.4 / 1.65 | 12.4 / 86.7 / 1.44 | 15.5 / 138.3 / 1.18 | 26.3 / 163.3 / 0.74 | 35.2 / 243.8 / 0.61 | 71.6 / 480.0 / 0.48 | 256.7 / 535.5 / 0.33 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.17 | 8.7 / 15.4 / 1.93 | 9.5 / 56.5 / 1.79 | 11.9 / 90.1 / 1.45 | 14.1 / 152.7 / 1.27 | 24.1 / 178.0 / 0.78 | 37.7 / 228.0 / 0.56 | 64.0 / 537.0 / 0.52 | 204.8 / 671.0 / 0.41 |
| aiter_a16wfp4 w4 sym | 16.3 / 2.1 / 0.55 | 16.8 / 8.0 / 0.53 | 16.7 / 32.1 / 0.55 | 19.2 / 55.8 / 0.49 | 19.5 / 110.0 / 0.51 | 37.8 / 113.6 / 0.29 | 77.1 / 111.5 / 0.17 | 283.3 / 121.3 / 0.09 | 1119.9 / 122.7 / 0.07 |
| aiter_afp4wfp4 w4 sym | 7.3 / 4.6 / 1.23 | 7.5 / 18.0 / 1.20 | 7.0 / 76.7 / 1.30 | 7.3 / 146.6 / 1.26 | 7.9 / 271.0 / 1.21 | 10.4 / 411.5 / 0.98 | 16.5 / 522.1 / 0.70 | 48.8 / 704.0 / 0.40 | 55.9 / 2457.9 / 0.92 |
| aiter_a8w8_int8 w8 sym | 8.5 / 4.0 / 1.98 | 8.7 / 15.4 / 1.93 | 9.9 / 54.4 / 1.72 | 9.0 / 119.5 / 1.91 | 11.7 / 184.2 / 1.51 | 17.2 / 249.1 / 1.06 | 25.5 / 336.6 / 0.78 | 38.8 / 885.4 / 0.76 | 105.4 / 1303.9 / 0.64 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 13.9 / 3.6 / 3.62 | 14.1 / 14.3 / 3.59 | 13.8 / 58.4 / 3.67 | 14.4 / 111.5 / 3.53 | 18.0 / 178.6 / 2.86 | 22.6 / 285.5 / 2.35 | 28.1 / 457.8 / 1.98 | 64.4 / 799.9 / 1.11 | 165.3 / 1247.5 / 0.81 |
| vllm_triton_w4a16 w4 sym | 75.4 / 0.7 / 0.17 | 75.9 / 2.7 / 0.17 | 75.9 / 10.6 / 0.17 | 76.5 / 21.1 / 0.18 | 77.9 / 41.4 / 0.18 | 105.2 / 61.2 / 0.15 | 104.6 / 123.1 / 0.17 | 127.1 / 405.6 / 0.27 | 371.0 / 555.6 / 0.26 |
| vllm_triton_w4a16 w4 zp | 81.0 / 0.6 / 0.16 | 81.2 / 2.5 / 0.16 | 81.5 / 9.9 / 0.16 | 87.1 / 18.5 / 0.16 | 85.3 / 37.8 / 0.17 | 112.6 / 57.2 / 0.14 | 111.7 / 115.4 / 0.16 | 136.7 / 376.9 / 0.25 | 392.6 / 525.1 / 0.25 |
| vllm_awq_triton_gemm w4 zp | 61.1 / 0.8 / 0.21 | 60.7 / 3.3 / 0.22 | 56.6 / 14.2 / 0.24 | 58.3 / 27.6 / 0.24 | 94.6 / 34.0 / 0.15 | 172.2 / 37.4 / 0.09 | 327.4 / 39.4 / 0.06 | 1240.9 / 41.5 / 0.03 | 4913.4 / 42.0 / 0.02 |
| vllm_awq_dequant_mm w4 zp | 34.8 / 1.4 / 0.38 | 34.7 / 5.8 / 0.38 | 34.6 / 23.3 / 0.39 | 34.4 / 46.8 / 0.40 | 36.1 / 89.1 / 0.40 | 41.9 / 153.7 / 0.38 | 44.0 / 293.1 / 0.42 | 83.4 / 617.8 / 0.41 | 202.5 / 1018.2 / 0.48 |
| conch_mixed_precision w4 sym | 95.2 / 0.5 / 0.14 | 95.4 / 2.1 / 0.14 | 101.9 / 7.9 / 0.13 | 120.4 / 13.4 / 0.11 | 157.8 / 20.4 / 0.09 | 258.2 / 24.9 / 0.06 | 257.9 / 50.0 / 0.07 | 759.3 / 67.9 / 0.04 | 2977.7 / 69.2 / 0.03 |
| conch_mixed_precision w4 zp | 101.8 / 0.5 / 0.13 | 102.0 / 2.0 / 0.13 | 112.3 / 7.2 / 0.12 | 130.2 / 12.4 / 0.10 | 167.7 / 19.2 / 0.09 | 270.1 / 23.9 / 0.06 | 268.9 / 47.9 / 0.07 | 837.7 / 61.5 / 0.04 | 3141.5 / 65.6 / 0.03 |
| conch_mixed_precision w8 sym | 102.7 / 0.5 / 0.25 | 103.0 / 2.0 / 0.25 | 108.4 / 7.4 / 0.24 | 127.2 / 12.7 / 0.21 | 166.6 / 19.3 / 0.16 | 271.0 / 23.8 / 0.10 | 270.5 / 47.6 / 0.11 | 792.2 / 65.1 / 0.06 | 3119.4 / 66.1 / 0.04 |
| conch_mixed_precision w8 zp | 113.2 / 0.4 / 0.23 | 113.0 / 1.8 / 0.23 | 125.1 / 6.4 / 0.21 | 142.2 / 11.3 / 0.19 | 181.5 / 17.7 / 0.15 | 282.5 / 22.8 / 0.10 | 281.1 / 45.8 / 0.11 | 831.9 / 62.0 / 0.06 | 3283.1 / 62.8 / 0.03 |
| vllm_exllama_gptq w4 sym | 48.7 / 1.0 / 0.27 | 71.2 / 2.8 / 0.18 | 141.9 / 5.7 / 0.09 | 162.5 / 9.9 / 0.08 | 45.2 / 71.2 / 0.32 | 50.6 / 127.3 / 0.31 | 62.4 / 206.5 / 0.29 | 101.6 / 507.3 / 0.33 | 213.3 / 966.5 / 0.45 |
| vllm_exllama_gptq w4 zp | 48.6 / 1.0 / 0.27 | 73.6 / 2.7 / 0.18 | 139.6 / 5.8 / 0.10 | 160.9 / 10.0 / 0.09 | 45.0 / 71.6 / 0.32 | 51.0 / 126.2 / 0.31 | 63.4 / 203.3 / 0.29 | 101.7 / 506.7 / 0.34 | 213.6 / 965.4 / 0.45 |
| vllm_exllama_gptq w8 sym | 53.6 / 0.9 / 0.48 | 72.9 / 2.8 / 0.35 | 145.7 / 5.5 / 0.18 | 44.9 / 35.9 / 0.58 | 46.5 / 69.3 / 0.58 | 52.3 / 123.1 / 0.54 | 64.5 / 199.8 / 0.48 | 101.8 / 506.3 / 0.46 | 215.0 / 959.1 / 0.51 |
| vllm_exllama_gptq w8 zp | 51.8 / 1.0 / 0.50 | 74.4 / 2.7 / 0.35 | 147.4 / 5.5 / 0.18 | 45.1 / 35.7 / 0.58 | 46.4 / 69.4 / 0.58 | 52.6 / 122.5 / 0.54 | 65.2 / 197.6 / 0.47 | 102.8 / 501.1 / 0.45 | 214.8 / 959.8 / 0.51 |
| dequant_triton+hipblaslt w4 sym | 28.8 / 1.7 / 0.45 | 29.3 / 6.9 / 0.45 | 28.5 / 28.2 / 0.47 | 29.6 / 54.4 / 0.46 | 33.8 / 95.2 / 0.42 | 37.0 / 174.1 / 0.42 | 42.6 / 302.8 / 0.43 | 78.1 / 660.2 / 0.43 | 180.1 / 1144.9 / 0.54 |
| dequant_triton+hipblaslt w4 zp | 29.7 / 1.7 / 0.44 | 29.5 / 6.8 / 0.45 | 28.8 / 27.9 / 0.47 | 30.7 / 52.4 / 0.45 | 34.4 / 93.7 / 0.42 | 36.4 / 177.0 / 0.43 | 42.7 / 301.6 / 0.43 | 80.3 / 641.8 / 0.42 | 184.1 / 1119.6 / 0.53 |
| dequant_triton+hipblaslt w8 sym | 32.1 / 1.6 / 0.80 | 31.1 / 6.5 / 0.82 | 30.0 / 26.9 / 0.86 | 37.7 / 42.7 / 0.69 | 42.4 / 76.1 / 0.63 | 38.2 / 168.6 / 0.74 | 43.6 / 295.4 / 0.71 | 80.8 / 637.5 / 0.58 | 181.1 / 1138.1 / 0.60 |
| dequant_triton+hipblaslt w8 zp | 31.2 / 1.6 / 0.83 | 31.0 / 6.5 / 0.83 | 29.8 / 27.0 / 0.88 | 30.8 / 52.3 / 0.86 | 35.8 / 89.9 / 0.76 | 37.9 / 170.0 / 0.75 | 43.7 / 294.8 / 0.71 | 81.3 / 634.3 / 0.57 | 182.0 / 1132.6 / 0.60 |
| aiter_a16w8_int8 w8 sym | 11.0 / 4.6 / 2.35 | 11.7 / 17.2 / 2.22 | 12.7 / 63.5 / 2.07 | 14.7 / 109.4 / 1.81 | 19.0 / 169.2 / 1.43 | 29.5 / 218.1 / 0.97 | 40.9 / 314.9 / 0.76 | 130.4 / 395.2 / 0.36 | 373.8 / 551.5 / 0.29 |
| aiter_a16w8_fp8_block128 w8 sym | 9.0 / 5.6 / 2.80 | 9.8 / 20.4 / 2.56 | 11.2 / 72.1 / 2.28 | 14.0 / 114.8 / 1.84 | 17.7 / 181.8 / 1.50 | 27.3 / 235.7 / 1.02 | 40.6 / 317.3 / 0.75 | 104.2 / 494.7 / 0.44 | 302.2 / 682.2 / 0.36 |
| aiter_a16wfp4 w4 sym | 16.6 / 3.0 / 0.81 | 17.0 / 11.8 / 0.79 | 16.9 / 47.8 / 0.81 | 19.4 / 82.8 / 0.72 | 37.0 / 87.0 / 0.40 | 54.6 / 118.0 / 0.29 | 106.5 / 121.0 / 0.17 | 393.5 / 131.0 / 0.09 | 1532.9 / 134.5 / 0.06 |
| aiter_afp4wfp4 w4 sym | 8.1 / 6.2 / 1.65 | 8.7 / 23.1 / 1.54 | 7.3 / 109.9 / 1.85 | 7.7 / 208.8 / 1.79 | 8.7 / 371.9 / 1.65 | 11.0 / 586.9 / 1.39 | 17.1 / 752.8 / 1.00 | 50.2 / 1027.5 / 0.56 | 99.5 / 2072.9 / 0.73 |
| aiter_a8w8_int8 w8 sym | 9.3 / 5.4 / 2.73 | 9.2 / 21.8 / 2.73 | 10.5 / 76.8 / 2.43 | 9.5 / 168.9 / 2.70 | 12.2 / 263.9 / 2.15 | 18.1 / 356.6 / 1.51 | 26.3 / 490.0 / 1.12 | 54.0 / 954.1 / 0.78 | 153.8 / 1340.3 / 0.60 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 48.3 / 4.9 / 4.87 | 50.7 / 18.5 / 4.64 | 45.6 / 82.4 / 5.17 | 48.4 / 155.2 / 4.89 | 47.2 / 318.3 / 5.06 | 51.6 / 582.9 / 4.72 | 70.2 / 856.1 / 3.58 | 198.6 / 1211.3 / 1.52 | 645.5 / 1490.3 / 0.78 |
| vllm_triton_w4a16 w4 sym | 79.3 / 3.0 / 0.76 | 79.7 / 11.8 / 0.76 | 81.1 / 46.3 / 0.76 | 82.1 / 91.5 / 0.76 | 81.6 / 184.1 / 0.79 | 107.1 / 280.6 / 0.64 | 136.0 / 442.0 / 0.57 | 451.4 / 532.8 / 0.28 | 1648.1 / 583.7 / 0.20 |
| vllm_triton_w4a16 w4 zp | 86.8 / 2.7 / 0.70 | 88.3 / 10.6 / 0.69 | 88.9 / 42.3 / 0.70 | 89.0 / 84.5 / 0.71 | 90.2 / 166.6 / 0.72 | 114.1 / 263.5 / 0.61 | 143.7 / 418.5 / 0.54 | 473.2 / 508.2 / 0.27 | 1750.7 / 549.5 / 0.19 |
| vllm_awq_triton_gemm w4 zp | 192.8 / 1.2 / 0.32 | 193.8 / 4.8 / 0.32 | 195.4 / 19.2 / 0.32 | 205.7 / 36.5 / 0.31 | 382.7 / 39.3 / 0.17 | 739.0 / 40.7 / 0.09 | 1449.8 / 41.5 / 0.05 | 5779.2 / 41.6 / 0.02 | 23530.0 / 40.9 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 85.7 / 2.7 / 0.71 | 85.8 / 10.9 / 0.71 | 96.8 / 38.8 / 0.64 | 94.9 / 79.2 / 0.67 | 88.7 / 169.4 / 0.73 | 99.5 / 302.1 / 0.70 | 120.6 / 498.7 / 0.65 | 258.2 / 931.6 / 0.50 | 809.9 / 1187.9 / 0.41 |
| conch_mixed_precision w4 sym | 132.7 / 1.8 / 0.46 | 133.0 / 7.1 / 0.46 | 164.8 / 22.8 / 0.37 | 204.4 / 36.8 / 0.31 | 297.0 / 50.6 / 0.22 | 508.0 / 59.2 / 0.14 | 1001.1 / 60.1 / 0.08 | 3486.5 / 69.0 / 0.04 | 15134.0 / 63.6 / 0.02 |
| conch_mixed_precision w4 zp | 197.3 / 1.2 / 0.31 | 197.2 / 4.8 / 0.31 | 217.8 / 17.3 / 0.28 | 252.8 / 29.7 / 0.25 | 330.1 / 45.5 / 0.20 | 532.4 / 56.5 / 0.13 | 1053.0 / 57.1 / 0.07 | 3677.4 / 65.4 / 0.04 | 14516.3 / 66.3 / 0.02 |
| conch_mixed_precision w8 sym | 160.1 / 1.5 / 0.74 | 159.9 / 5.9 / 0.75 | 184.3 / 20.4 / 0.65 | 224.5 / 33.5 / 0.54 | 321.9 / 46.7 / 0.38 | 533.9 / 56.3 / 0.24 | 1053.2 / 57.1 / 0.13 | 3645.4 / 66.0 / 0.05 | 14488.7 / 66.4 / 0.03 |
| conch_mixed_precision w8 zp | 221.8 / 1.1 / 0.54 | 221.8 / 4.2 / 0.54 | 243.9 / 15.4 / 0.50 | 278.3 / 27.0 / 0.44 | 357.4 / 42.1 / 0.35 | 559.9 / 53.7 / 0.23 | 1106.7 / 54.3 / 0.12 | 3847.6 / 62.5 / 0.05 | 15309.2 / 62.8 / 0.03 |
| vllm_exllama_gptq w4 sym | 81.8 / 2.9 / 0.74 | 136.8 / 6.9 / 0.45 | 195.3 / 19.2 / 0.32 | 195.7 / 38.4 / 0.32 | 101.3 / 148.4 / 0.64 | 110.6 / 271.9 / 0.62 | 131.4 / 457.5 / 0.59 | 269.0 / 894.2 / 0.47 | 841.8 / 1142.9 / 0.39 |
| vllm_exllama_gptq w4 zp | 81.5 / 2.9 / 0.75 | 138.5 / 6.8 / 0.44 | 195.1 / 19.3 / 0.32 | 197.0 / 38.2 / 0.32 | 102.7 / 146.4 / 0.64 | 111.5 / 269.6 / 0.62 | 133.3 / 450.9 / 0.58 | 285.7 / 841.8 / 0.45 | 849.5 / 1132.5 / 0.39 |
| vllm_exllama_gptq w8 sym | 89.5 / 2.6 / 1.33 | 137.7 / 6.8 / 0.87 | 182.0 / 20.6 / 0.66 | 118.5 / 63.4 / 1.02 | 110.8 / 135.7 / 1.11 | 117.7 / 255.5 / 1.08 | 148.0 / 406.2 / 0.92 | 284.6 / 845.1 / 0.66 | 854.9 / 1125.4 / 0.45 |
| vllm_exllama_gptq w8 zp | 87.8 / 2.7 / 1.37 | 137.2 / 6.8 / 0.88 | 182.8 / 20.6 / 0.66 | 118.7 / 63.3 / 1.03 | 110.7 / 135.8 / 1.12 | 118.0 / 254.8 / 1.09 | 146.6 / 410.2 / 0.93 | 286.9 / 838.4 / 0.65 | 855.1 / 1125.2 / 0.46 |
| dequant_triton+hipblaslt w4 sym | 89.8 / 2.6 / 0.68 | 107.3 / 8.8 / 0.57 | 94.4 / 39.8 / 0.65 | 113.5 / 66.2 / 0.55 | 98.5 / 152.6 / 0.66 | 114.5 / 262.6 / 0.60 | 129.8 / 463.1 / 0.60 | 247.5 / 972.0 / 0.52 | 684.0 / 1406.6 / 0.48 |
| dequant_triton+hipblaslt w4 zp | 90.7 / 2.6 / 0.67 | 109.6 / 8.6 / 0.56 | 95.8 / 39.2 / 0.65 | 114.6 / 65.6 / 0.55 | 99.3 / 151.3 / 0.66 | 116.1 / 258.9 / 0.60 | 130.4 / 461.1 / 0.60 | 249.1 / 965.4 / 0.51 | 688.3 / 1397.8 / 0.48 |
| dequant_triton+hipblaslt w8 sym | 119.4 / 2.0 / 1.00 | 120.4 / 7.8 / 0.99 | 108.8 / 34.6 / 1.11 | 118.0 / 63.7 / 1.03 | 115.3 / 130.3 / 1.07 | 120.8 / 249.0 / 1.06 | 164.0 / 366.6 / 0.83 | 257.0 / 936.0 / 0.72 | 697.8 / 1378.7 / 0.56 |
| dequant_triton+hipblaslt w8 zp | 116.5 / 2.0 / 1.03 | 120.4 / 7.8 / 1.00 | 110.1 / 34.1 / 1.10 | 116.7 / 64.4 / 1.05 | 118.1 / 127.3 / 1.05 | 119.5 / 251.5 / 1.08 | 160.6 / 374.4 / 0.85 | 257.8 / 933.0 / 0.73 | 702.9 / 1368.7 / 0.55 |
| aiter_a16w8_int8 w8 sym | 30.5 / 7.7 / 3.98 | 34.2 / 27.4 / 3.54 | 41.0 / 91.6 / 2.98 | 47.5 / 158.3 / 2.60 | 62.4 / 241.0 / 2.01 | 96.1 / 312.8 / 1.35 | 157.8 / 381.0 / 0.87 | 450.4 / 534.1 / 0.42 | 1710.1 / 562.6 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 23.4 / 10.0 / 5.02 | 27.0 / 34.8 / 4.36 | 33.1 / 113.5 / 3.58 | 45.8 / 164.3 / 2.61 | 57.3 / 262.5 / 2.12 | 85.2 / 352.8 / 1.48 | 153.2 / 392.6 / 0.88 | 364.5 / 659.8 / 0.51 | 1371.2 / 701.6 / 0.28 |
| aiter_a16wfp4 w4 sym | 18.4 / 12.7 / 3.39 | 19.1 / 49.3 / 3.29 | 42.7 / 88.1 / 1.49 | 77.4 / 97.1 / 0.83 | 134.2 / 112.0 / 0.50 | 261.8 / 114.8 / 0.27 | 520.1 / 115.6 / 0.15 | 1813.2 / 132.6 / 0.07 | 7235.7 / 133.0 / 0.05 |
| aiter_afp4wfp4 w4 sym | 21.2 / 11.1 / 2.94 | 23.6 / 39.9 / 2.66 | 25.5 / 147.3 / 2.48 | 23.3 / 322.5 / 2.76 | 29.9 / 502.0 / 2.21 | 39.2 / 766.1 / 1.78 | 40.6 / 1481.9 / 1.91 | 104.0 / 2313.5 / 1.19 | 343.3 / 2802.1 / 0.89 |
| aiter_a8w8_int8 w8 sym | 33.2 / 7.1 / 3.55 | 34.2 / 27.5 / 3.45 | 25.3 / 148.3 / 4.68 | 34.4 / 218.6 / 3.48 | 45.5 / 330.7 / 2.67 | 49.2 / 610.8 / 2.55 | 97.1 / 619.1 / 1.37 | 183.2 / 1313.0 / 0.98 | 652.5 / 1474.3 / 0.57 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.2 / 4.8 / 4.85 | 24.6 / 19.1 / 4.77 | 26.2 / 71.6 / 4.50 | 29.6 / 126.9 / 4.00 | 29.9 / 251.3 / 4.01 | 34.2 / 439.8 / 3.57 | 53.1 / 566.6 / 2.39 | 118.2 / 1017.6 / 1.31 | 309.7 / 1553.1 / 0.87 |
| vllm_triton_w4a16 w4 sym | 255.4 / 0.5 / 0.12 | 256.4 / 1.8 / 0.12 | 259.0 / 7.3 / 0.12 | 261.5 / 14.4 / 0.12 | 264.9 / 28.4 / 0.12 | 356.3 / 42.2 / 0.10 | 354.8 / 84.7 / 0.11 | 363.6 / 330.7 / 0.19 | 830.6 / 579.2 / 0.22 |
| vllm_triton_w4a16 w4 zp | 275.2 / 0.4 / 0.11 | 274.9 / 1.7 / 0.11 | 276.5 / 6.8 / 0.11 | 278.7 / 13.5 / 0.11 | 289.6 / 26.0 / 0.11 | 380.5 / 39.5 / 0.09 | 381.8 / 78.8 / 0.10 | 381.5 / 315.2 / 0.18 | 881.1 / 546.0 / 0.21 |
| vllm_awq_triton_gemm w4 zp | 251.7 / 0.5 / 0.12 | 118.6 / 4.0 / 0.26 | 135.4 / 13.9 / 0.23 | 114.8 / 32.7 / 0.28 | 202.2 / 37.2 / 0.16 | 373.5 / 40.2 / 0.09 | 718.5 / 41.8 / 0.06 | 2760.3 / 43.6 / 0.03 | 10883.1 / 44.2 / 0.02 |
| vllm_awq_dequant_mm w4 zp | 51.8 / 2.3 / 0.59 | 51.0 / 9.2 / 0.60 | 53.6 / 35.1 / 0.58 | 53.2 / 70.7 / 0.60 | 54.6 / 137.8 / 0.60 | 60.9 / 246.8 / 0.58 | 75.6 / 397.7 / 0.53 | 149.9 / 802.3 / 0.46 | 385.6 / 1247.5 / 0.47 |
| conch_mixed_precision w4 sym | 314.0 / 0.4 / 0.10 | 314.9 / 1.5 / 0.10 | 342.7 / 5.5 / 0.09 | 402.4 / 9.3 / 0.08 | 535.3 / 14.0 / 0.06 | 876.6 / 17.1 / 0.04 | 879.2 / 34.2 / 0.04 | 1740.9 / 69.1 / 0.04 | 6858.6 / 70.1 / 0.03 |
| conch_mixed_precision w4 zp | 338.7 / 0.3 / 0.09 | 338.6 / 1.4 / 0.09 | 371.0 / 5.1 / 0.08 | 434.9 / 8.6 / 0.07 | 575.5 / 13.1 / 0.06 | 906.5 / 16.6 / 0.04 | 905.3 / 33.2 / 0.04 | 1808.6 / 66.5 / 0.04 | 7148.1 / 67.3 / 0.03 |
| conch_mixed_precision w8 sym | 339.9 / 0.3 / 0.18 | 339.3 / 1.4 / 0.18 | 362.6 / 5.2 / 0.17 | 425.6 / 8.8 / 0.14 | 561.8 / 13.4 / 0.11 | 921.5 / 16.3 / 0.07 | 920.9 / 32.6 / 0.07 | 1824.0 / 65.9 / 0.05 | 7216.5 / 66.7 / 0.03 |
| conch_mixed_precision w8 zp | 379.7 / 0.3 / 0.16 | 381.0 / 1.2 / 0.16 | 411.8 / 4.6 / 0.15 | 475.5 / 7.9 / 0.13 | 615.2 / 12.2 / 0.10 | 961.3 / 15.6 / 0.07 | 958.3 / 31.4 / 0.07 | 1904.0 / 63.2 / 0.05 | 7555.9 / 63.7 / 0.03 |
| vllm_exllama_gptq w4 sym | 363.6 / 0.3 / 0.08 | 390.3 / 1.2 / 0.08 | 563.0 / 3.3 / 0.06 | 760.5 / 4.9 / 0.04 | 159.7 / 47.1 / 0.20 | 177.7 / 84.6 / 0.20 | 240.6 / 125.0 / 0.17 | 329.8 / 364.7 / 0.21 | 419.8 / 1145.8 / 0.43 |
| vllm_exllama_gptq w4 zp | 334.1 / 0.4 / 0.09 | 394.7 / 1.2 / 0.08 | 565.9 / 3.3 / 0.06 | 710.6 / 5.3 / 0.04 | 160.1 / 46.9 / 0.20 | 177.9 / 84.5 / 0.20 | 238.9 / 125.8 / 0.17 | 329.7 / 364.7 / 0.21 | 419.4 / 1147.0 / 0.43 |
| vllm_exllama_gptq w8 sym | 336.0 / 0.3 / 0.18 | 419.0 / 1.1 / 0.14 | 556.6 / 3.4 / 0.11 | 162.1 / 23.2 / 0.38 | 172.6 / 43.5 / 0.36 | 190.5 / 78.9 / 0.34 | 253.5 / 118.6 / 0.27 | 342.3 / 351.3 / 0.28 | 437.3 / 1100.0 / 0.48 |
| vllm_exllama_gptq w8 zp | 336.6 / 0.3 / 0.18 | 412.6 / 1.1 / 0.15 | 557.8 / 3.4 / 0.11 | 162.4 / 23.1 / 0.38 | 172.7 / 43.5 / 0.36 | 190.6 / 78.9 / 0.34 | 253.7 / 118.5 / 0.27 | 342.3 / 351.4 / 0.29 | 436.3 / 1102.5 / 0.48 |
| dequant_triton+hipblaslt w4 sym | 55.2 / 2.1 / 0.55 | 55.3 / 8.5 / 0.55 | 56.2 / 33.4 / 0.55 | 58.9 / 63.8 / 0.53 | 59.9 / 125.4 / 0.55 | 66.5 / 226.1 / 0.53 | 85.9 / 349.8 / 0.46 | 145.6 / 826.2 / 0.47 | 337.1 / 1427.0 / 0.54 |
| dequant_triton+hipblaslt w4 zp | 58.1 / 2.0 / 0.53 | 58.3 / 8.1 / 0.53 | 59.0 / 31.8 / 0.53 | 61.6 / 61.1 / 0.52 | 62.5 / 120.3 / 0.53 | 69.0 / 217.9 / 0.51 | 88.2 / 340.9 / 0.45 | 149.7 / 803.1 / 0.46 | 343.2 / 1401.6 / 0.53 |
| dequant_triton+hipblaslt w8 sym | 58.0 / 2.0 / 1.03 | 59.1 / 7.9 / 1.01 | 59.2 / 31.7 / 1.02 | 61.7 / 60.9 / 0.99 | 61.8 / 121.6 / 1.00 | 68.4 / 219.8 / 0.94 | 88.3 / 340.5 / 0.78 | 148.5 / 809.6 / 0.66 | 335.2 / 1435.0 / 0.63 |
| dequant_triton+hipblaslt w8 zp | 59.8 / 2.0 / 1.01 | 59.7 / 7.9 / 1.01 | 60.3 / 31.1 / 1.01 | 62.2 / 60.4 / 0.98 | 63.6 / 118.1 / 0.98 | 69.2 / 217.2 / 0.94 | 88.6 / 339.5 / 0.79 | 149.8 / 802.8 / 0.65 | 345.9 / 1390.8 / 0.61 |
| aiter_a16w8_int8 w8 sym | 20.3 / 5.8 / 2.98 | 20.9 / 22.5 / 2.91 | 20.7 / 90.8 / 2.96 | 27.9 / 134.8 / 2.21 | 36.1 / 208.3 / 1.74 | 66.8 / 225.0 / 0.98 | 114.1 / 263.5 / 0.61 | 232.4 / 517.4 / 0.42 | 842.6 / 570.9 / 0.25 |
| aiter_a16w8_fp8_block128 w8 sym | 17.0 / 6.9 / 3.46 | 17.9 / 26.3 / 3.29 | 18.8 / 100.0 / 3.16 | 27.4 / 136.9 / 2.18 | 31.8 / 236.4 / 1.92 | 57.4 / 262.0 / 1.11 | 121.5 / 247.5 / 0.56 | 180.2 / 667.4 / 0.54 | 649.7 / 740.4 / 0.32 |
| aiter_a16wfp4 w4 sym | 53.0 / 2.2 / 0.59 | 54.3 / 8.6 / 0.58 | 53.4 / 35.2 / 0.59 | 61.6 / 61.1 / 0.53 | 61.5 / 122.1 / 0.55 | 121.5 / 123.7 / 0.30 | 239.9 / 125.3 / 0.17 | 948.3 / 126.8 / 0.07 | 3781.9 / 127.2 / 0.05 |
| aiter_afp4wfp4 w4 sym | 10.7 / 11.0 / 2.92 | 12.1 / 38.7 / 2.58 | 16.5 / 114.1 / 1.91 | 18.5 / 203.5 / 1.72 | 22.6 / 332.9 / 1.43 | 31.3 / 479.9 / 1.06 | 51.3 / 585.5 / 0.69 | 148.9 / 807.4 / 0.32 | 163.2 / 2947.3 / 0.59 |
| aiter_a8w8_int8 w8 sym | 24.7 / 4.8 / 2.38 | 24.8 / 18.9 / 2.37 | 29.7 / 63.3 / 1.99 | 25.4 / 148.0 / 2.34 | 33.0 / 228.0 / 1.83 | 53.3 / 281.9 / 1.16 | 79.8 / 376.9 / 0.81 | 113.8 / 1056.6 / 0.72 | 318.8 / 1509.1 / 0.47 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 26.1 / 5.1 / 5.15 | 26.2 / 20.5 / 5.12 | 27.5 / 78.2 / 4.91 | 29.1 / 147.6 / 4.65 | 30.8 / 279.3 / 4.43 | 40.6 / 423.1 / 3.41 | 64.4 / 533.8 / 2.22 | 145.2 / 946.3 / 1.16 | 356.1 / 1543.8 / 0.75 |
| vllm_triton_w4a16 w4 sym | 147.4 / 0.9 / 0.23 | 148.0 / 3.6 / 0.23 | 148.3 / 14.5 / 0.24 | 149.1 / 28.8 / 0.24 | 150.8 / 57.0 / 0.24 | 206.4 / 83.3 / 0.19 | 203.7 / 168.7 / 0.21 | 254.5 / 540.0 / 0.27 | 934.7 / 588.2 / 0.18 |
| vllm_triton_w4a16 w4 zp | 158.4 / 0.8 / 0.22 | 158.9 / 3.4 / 0.22 | 158.9 / 13.5 / 0.22 | 158.8 / 27.0 / 0.23 | 166.4 / 51.6 / 0.22 | 219.6 / 78.2 / 0.18 | 217.5 / 158.0 / 0.20 | 273.4 / 502.7 / 0.25 | 997.9 / 550.9 / 0.17 |
| vllm_awq_triton_gemm w4 zp | 120.9 / 1.1 / 0.29 | 121.7 / 4.4 / 0.29 | 124.3 / 17.3 / 0.28 | 126.5 / 33.9 / 0.28 | 224.2 / 38.3 / 0.17 | 422.5 / 40.7 / 0.09 | 824.1 / 41.7 / 0.05 | 3189.8 / 43.1 / 0.02 | 12740.5 / 43.1 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 56.6 / 2.4 / 0.62 | 56.4 / 9.5 / 0.62 | 55.9 / 38.4 / 0.63 | 60.6 / 70.9 / 0.59 | 62.2 / 138.1 / 0.59 | 70.3 / 244.2 / 0.56 | 86.9 / 395.3 / 0.50 | 178.6 / 769.5 / 0.38 | 446.9 / 1230.2 / 0.38 |
| conch_mixed_precision w4 sym | 179.8 / 0.7 / 0.19 | 180.0 / 3.0 / 0.19 | 193.5 / 11.1 / 0.18 | 228.3 / 18.8 / 0.16 | 302.3 / 28.4 / 0.12 | 504.8 / 34.0 / 0.08 | 504.0 / 68.2 / 0.09 | 1982.1 / 69.3 / 0.03 | 7840.7 / 70.1 / 0.02 |
| conch_mixed_precision w4 zp | 193.6 / 0.7 / 0.18 | 193.9 / 2.8 / 0.18 | 213.6 / 10.1 / 0.17 | 248.4 / 17.3 / 0.14 | 324.4 / 26.5 / 0.11 | 524.5 / 32.8 / 0.07 | 522.6 / 65.8 / 0.08 | 2060.2 / 66.7 / 0.03 | 8161.4 / 67.4 / 0.02 |
| conch_mixed_precision w8 sym | 197.3 / 0.7 / 0.35 | 197.4 / 2.7 / 0.35 | 209.3 / 10.3 / 0.33 | 245.2 / 17.5 / 0.28 | 322.3 / 26.7 / 0.22 | 532.6 / 32.3 / 0.14 | 530.5 / 64.8 / 0.14 | 2082.2 / 66.0 / 0.05 | 8291.3 / 66.3 / 0.02 |
| conch_mixed_precision w8 zp | 221.4 / 0.6 / 0.31 | 220.7 / 2.4 / 0.31 | 241.5 / 8.9 / 0.29 | 275.6 / 15.6 / 0.25 | 352.2 / 24.4 / 0.20 | 554.5 / 31.0 / 0.13 | 552.5 / 62.2 / 0.14 | 2180.0 / 63.0 / 0.05 | 8671.5 / 63.4 / 0.02 |
| vllm_exllama_gptq w4 sym | 165.1 / 0.8 / 0.21 | 185.5 / 2.9 / 0.19 | 304.4 / 7.1 / 0.12 | 289.1 / 14.9 / 0.12 | 154.6 / 55.6 / 0.24 | 167.6 / 102.5 / 0.23 | 223.9 / 153.5 / 0.19 | 236.9 / 580.1 / 0.29 | 483.3 / 1137.4 / 0.35 |
| vllm_exllama_gptq w4 zp | 164.8 / 0.8 / 0.21 | 186.2 / 2.9 / 0.19 | 298.4 / 7.2 / 0.12 | 286.8 / 15.0 / 0.12 | 153.1 / 56.1 / 0.24 | 166.1 / 103.4 / 0.23 | 222.5 / 154.4 / 0.19 | 237.2 / 579.4 / 0.29 | 483.1 / 1137.9 / 0.35 |
| vllm_exllama_gptq w8 sym | 166.1 / 0.8 / 0.41 | 182.7 / 2.9 / 0.37 | 300.5 / 7.1 / 0.23 | 116.5 / 36.9 / 0.59 | 166.5 / 51.6 / 0.42 | 180.6 / 95.1 / 0.40 | 235.6 / 145.8 / 0.33 | 245.1 / 560.6 / 0.41 | 499.7 / 1100.1 / 0.41 |
| vllm_exllama_gptq w8 zp | 165.2 / 0.8 / 0.42 | 183.5 / 2.9 / 0.38 | 285.6 / 7.5 / 0.24 | 116.5 / 36.9 / 0.60 | 166.6 / 51.6 / 0.42 | 180.7 / 95.1 / 0.40 | 236.0 / 145.6 / 0.33 | 244.8 / 561.4 / 0.42 | 499.4 / 1100.8 / 0.41 |
| dequant_triton+hipblaslt w4 sym | 64.8 / 2.1 / 0.53 | 64.3 / 8.4 / 0.54 | 66.5 / 32.3 / 0.53 | 68.1 / 63.1 / 0.52 | 71.9 / 119.5 / 0.51 | 83.0 / 207.1 / 0.47 | 109.7 / 313.2 / 0.39 | 178.2 / 771.2 / 0.38 | 386.7 / 1421.6 / 0.44 |
| dequant_triton+hipblaslt w4 zp | 61.8 / 2.2 / 0.56 | 66.5 / 8.1 / 0.53 | 63.5 / 33.8 / 0.56 | 70.1 / 61.3 / 0.51 | 69.5 / 123.7 / 0.53 | 82.2 / 209.0 / 0.47 | 104.9 / 327.4 / 0.41 | 177.8 / 773.2 / 0.39 | 389.4 / 1411.9 / 0.43 |
| dequant_triton+hipblaslt w8 sym | 68.5 / 2.0 / 1.00 | 62.8 / 8.5 / 1.09 | 70.7 / 30.4 / 0.97 | 73.2 / 58.7 / 0.94 | 76.1 / 112.8 / 0.92 | 87.4 / 196.7 / 0.83 | 115.3 / 298.1 / 0.66 | 181.9 / 755.4 / 0.56 | 389.4 / 1412.0 / 0.52 |
| dequant_triton+hipblaslt w8 zp | 64.5 / 2.1 / 1.06 | 69.3 / 7.8 / 0.99 | 66.4 / 32.4 / 1.04 | 73.7 / 58.2 / 0.95 | 72.3 / 118.7 / 0.98 | 87.9 / 195.4 / 0.83 | 107.5 / 319.7 / 0.72 | 182.4 / 753.6 / 0.56 | 388.8 / 1414.2 / 0.52 |
| aiter_a16w8_int8 w8 sym | 18.5 / 7.2 / 3.74 | 19.4 / 27.6 / 3.57 | 22.7 / 94.6 / 3.07 | 24.7 / 174.0 / 2.85 | 32.7 / 262.6 / 2.18 | 49.0 / 350.9 / 1.50 | 83.6 / 410.9 / 0.93 | 260.4 / 527.8 / 0.40 | 965.0 / 569.7 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 15.6 / 8.6 / 4.31 | 16.7 / 32.2 / 4.04 | 18.3 / 117.2 / 3.69 | 23.0 / 187.1 / 2.97 | 29.0 / 295.9 / 2.38 | 44.8 / 383.3 / 1.59 | 81.0 / 424.4 / 0.93 | 204.4 / 672.4 / 0.49 | 765.6 / 718.1 / 0.26 |
| aiter_a16wfp4 w4 sym | 31.8 / 4.2 / 1.12 | 32.5 / 16.5 / 1.10 | 32.4 / 66.3 / 1.12 | 37.6 / 114.1 / 0.97 | 72.9 / 117.9 / 0.52 | 142.7 / 120.4 / 0.28 | 281.9 / 121.9 / 0.16 | 1045.2 / 131.5 / 0.07 | 4167.7 / 131.9 / 0.04 |
| aiter_afp4wfp4 w4 sym | 12.9 / 10.4 / 2.77 | 11.6 / 46.4 / 3.09 | 11.6 / 185.2 / 3.10 | 12.5 / 343.7 / 2.91 | 16.2 / 530.0 / 2.28 | 20.9 / 820.2 / 1.83 | 33.0 / 1039.8 / 1.24 | 88.8 / 1547.4 / 0.64 | 183.6 / 2994.5 / 0.66 |
| aiter_a8w8_int8 w8 sym | 17.3 / 7.7 / 3.87 | 17.0 / 31.6 / 3.95 | 19.7 / 108.9 / 3.42 | 17.4 / 247.2 / 3.91 | 22.9 / 374.9 / 3.00 | 35.3 / 486.0 / 1.99 | 51.3 / 669.4 / 1.43 | 107.5 / 1278.8 / 0.86 | 380.3 / 1445.7 / 0.44 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 157.3 / 6.0 / 5.97 | 157.5 / 23.9 / 5.97 | 156.7 / 96.0 / 6.01 | 157.6 / 190.8 / 5.99 | 158.8 / 378.6 / 5.97 | 171.4 / 701.6 / 5.58 | 216.8 / 1109.2 / 4.49 | 727.8 / 1321.8 / 1.48 | 2580.6 / 1491.3 / 0.57 |
| vllm_triton_w4a16 w4 sym | 200.1 / 4.7 / 1.21 | 204.9 / 18.3 / 1.19 | 231.7 / 64.9 / 1.05 | 227.2 / 132.4 / 1.08 | 405.1 / 148.4 / 0.62 | 501.1 / 240.0 / 0.52 | 793.0 / 303.3 / 0.35 | 1696.1 / 567.2 / 0.22 | 6390.8 / 602.2 / 0.12 |
| vllm_triton_w4a16 w4 zp | 221.1 / 4.2 / 1.10 | 224.2 / 16.8 / 1.09 | 241.9 / 62.1 / 1.02 | 244.7 / 122.9 / 1.01 | 241.3 / 249.2 / 1.05 | 276.4 / 435.1 / 0.94 | 525.4 / 457.8 / 0.53 | 1771.4 / 543.1 / 0.21 | 8907.0 / 432.1 / 0.09 |
| vllm_awq_triton_gemm w4 zp | 959.8 / 1.0 / 0.25 | 960.6 / 3.9 / 0.26 | 905.3 / 16.6 / 0.27 | 726.5 / 41.4 / 0.34 | 1428.2 / 42.1 / 0.18 | 2798.9 / 43.0 / 0.09 | 5556.0 / 43.3 / 0.05 | 22168.0 / 43.4 / 0.02 | 89745.6 / 42.9 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 382.2 / 2.5 / 0.64 | 390.2 / 9.6 / 0.63 | 383.0 / 39.2 / 0.64 | 387.8 / 77.5 / 0.64 | 394.9 / 152.3 / 0.64 | 393.7 / 305.5 / 0.66 | 434.5 / 553.5 / 0.64 | 997.9 / 964.1 / 0.38 | 3139.5 / 1225.8 / 0.25 |
| conch_mixed_precision w4 sym | 490.4 / 1.9 / 0.49 | 490.9 / 7.7 / 0.49 | 623.8 / 24.1 / 0.39 | 775.9 / 38.7 / 0.32 | 1154.7 / 52.1 / 0.22 | 2335.4 / 51.5 / 0.11 | 3921.3 / 61.3 / 0.07 | 13718.8 / 70.1 / 0.03 | 56003.1 / 68.7 / 0.01 |
| conch_mixed_precision w4 zp | 773.6 / 1.2 / 0.32 | 774.2 / 4.9 / 0.32 | 851.7 / 17.7 / 0.29 | 991.5 / 30.3 / 0.25 | 1285.3 / 46.8 / 0.20 | 2092.5 / 57.5 / 0.12 | 3643.6 / 66.0 / 0.08 | 14494.4 / 66.4 / 0.03 | 57843.2 / 66.5 / 0.01 |
| conch_mixed_precision w8 sym | 598.9 / 1.6 / 0.80 | 598.9 / 6.3 / 0.80 | 703.4 / 21.4 / 0.68 | 857.6 / 35.1 / 0.56 | 1257.3 / 47.8 / 0.39 | 2095.3 / 57.4 / 0.24 | 3634.7 / 66.2 / 0.14 | 14478.4 / 66.4 / 0.04 | 57728.8 / 66.7 / 0.02 |
| conch_mixed_precision w8 zp | 872.6 / 1.1 / 0.55 | 875.1 / 4.3 / 0.55 | 959.3 / 15.7 / 0.50 | 1097.1 / 27.4 / 0.44 | 1403.9 / 42.8 / 0.35 | 2208.9 / 54.4 / 0.23 | 3841.0 / 62.6 / 0.13 | 15267.3 / 63.0 / 0.04 | 60904.8 / 63.2 / 0.02 |
| vllm_exllama_gptq w4 sym | 259.8 / 3.6 / 0.93 | 276.5 / 13.6 / 0.88 | 349.4 / 43.0 / 0.70 | 593.6 / 50.6 / 0.41 | 420.4 / 143.0 / 0.60 | 440.9 / 272.7 / 0.59 | 459.9 / 523.0 / 0.60 | 1026.5 / 937.3 / 0.37 | 3230.4 / 1191.3 / 0.24 |
| vllm_exllama_gptq w4 zp | 260.0 / 3.6 / 0.94 | 276.5 / 13.6 / 0.89 | 349.1 / 43.1 / 0.70 | 593.2 / 50.7 / 0.42 | 428.3 / 140.4 / 0.59 | 432.4 / 278.1 / 0.60 | 477.9 / 503.3 / 0.58 | 1031.8 / 932.5 / 0.37 | 3225.0 / 1193.3 / 0.24 |
| vllm_exllama_gptq w8 sym | 252.6 / 3.7 / 1.89 | 242.1 / 15.5 / 1.97 | 433.1 / 34.7 / 1.11 | 447.8 / 67.1 / 1.07 | 476.7 / 126.2 / 1.02 | 477.0 / 252.1 / 1.03 | 492.3 / 488.5 / 1.04 | 1080.0 / 890.8 / 0.57 | 3269.7 / 1177.0 / 0.31 |
| vllm_exllama_gptq w8 zp | 255.1 / 3.7 / 1.89 | 241.4 / 15.6 / 1.99 | 431.1 / 34.9 / 1.12 | 445.8 / 67.4 / 1.09 | 473.8 / 126.9 / 1.03 | 457.1 / 263.1 / 1.09 | 505.7 / 475.6 / 1.02 | 1072.1 / 897.4 / 0.57 | 3274.9 / 1175.1 / 0.31 |
| dequant_triton+hipblaslt w4 sym | 414.0 / 2.3 / 0.58 | 407.0 / 9.2 / 0.60 | 404.8 / 37.1 / 0.60 | 414.7 / 72.5 / 0.59 | 412.3 / 145.8 / 0.61 | 434.1 / 277.1 / 0.60 | 457.1 / 526.2 / 0.60 | 969.1 / 992.8 / 0.39 | 2797.3 / 1375.7 / 0.28 |
| dequant_triton+hipblaslt w4 zp | 428.8 / 2.2 / 0.57 | 424.7 / 8.8 / 0.58 | 443.6 / 33.9 / 0.56 | 427.4 / 70.3 / 0.58 | 449.0 / 133.9 / 0.56 | 456.8 / 263.3 / 0.57 | 484.7 / 496.2 / 0.57 | 997.9 / 964.1 / 0.38 | 2827.3 / 1361.1 / 0.28 |
| dequant_triton+hipblaslt w8 sym | 457.1 / 2.1 / 1.04 | 437.3 / 8.6 / 1.09 | 454.8 / 33.1 / 1.05 | 439.7 / 68.4 / 1.09 | 456.1 / 131.8 / 1.06 | 454.6 / 264.5 / 1.09 | 500.0 / 481.1 / 1.02 | 1000.4 / 961.7 / 0.61 | 2846.7 / 1351.8 / 0.36 |
| dequant_triton+hipblaslt w8 zp | 467.2 / 2.0 / 1.03 | 449.4 / 8.4 / 1.07 | 458.3 / 32.8 / 1.05 | 463.5 / 64.9 / 1.05 | 472.7 / 127.2 / 1.03 | 476.5 / 252.4 / 1.04 | 517.6 / 464.6 / 0.99 | 1026.0 / 937.6 / 0.60 | 2891.6 / 1330.9 / 0.35 |
| aiter_a16w8_int8 w8 sym | 96.5 / 9.7 / 5.02 | 106.2 / 35.4 / 4.57 | 126.4 / 119.0 / 3.85 | 144.7 / 207.8 / 3.38 | 181.4 / 331.5 / 2.72 | 289.0 / 416.1 / 1.73 | 548.5 / 438.5 / 0.94 | 1725.4 / 557.6 / 0.36 | 6637.1 / 579.8 / 0.15 |
| aiter_a16w8_fp8_block128 w8 sym | 77.9 / 12.1 / 6.04 | 84.1 / 44.7 / 5.59 | 102.7 / 146.4 / 4.60 | 131.0 / 229.5 / 3.62 | 167.7 / 358.5 / 2.85 | 249.0 / 482.9 / 1.95 | 510.3 / 471.3 / 0.99 | 1362.0 / 706.4 / 0.44 | 5224.9 / 736.5 / 0.19 |
| aiter_a16wfp4 w4 sym | 51.9 / 18.1 / 4.81 | 52.1 / 72.1 / 4.80 | 194.3 / 77.4 / 1.29 | 275.2 / 109.2 / 0.92 | 535.8 / 112.2 / 0.48 | 1070.1 / 112.4 / 0.25 | 2112.0 / 113.9 / 0.13 | 7167.0 / 134.2 / 0.05 | 28994.6 / 132.7 / 0.03 |
| aiter_afp4wfp4 w4 sym | 72.0 / 13.0 / 3.47 | 76.2 / 49.3 / 3.28 | 68.9 / 218.2 / 3.65 | 62.2 / 483.3 / 4.07 | 76.9 / 781.6 / 3.34 | 104.8 / 1147.4 / 2.53 | 141.2 / 1703.0 / 1.98 | 356.2 / 2700.8 / 1.04 | 1243.6 / 3094.5 / 0.59 |
| aiter_a8w8_int8 w8 sym | 108.6 / 8.7 / 4.33 | 107.2 / 35.0 / 4.39 | 94.4 / 159.3 / 5.00 | 108.3 / 277.6 / 4.38 | 142.8 / 421.1 / 3.35 | 173.1 / 694.9 / 2.81 | 317.0 / 758.8 / 1.58 | 642.3 / 1497.8 / 0.93 | 2452.3 / 1569.3 / 0.40 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.3 / 6.2 / 6.24 | 76.0 / 24.7 / 6.18 | 76.6 / 98.1 / 6.15 | 79.2 / 189.9 / 5.96 | 83.6 / 359.6 / 5.68 | 94.8 / 634.2 / 5.05 | 130.6 / 920.9 / 3.74 | 366.0 / 1314.3 / 1.49 | 1469.8 / 1309.1 / 0.53 |
| vllm_triton_w4a16 w4 sym | 510.0 / 0.9 / 0.24 | 511.3 / 3.7 / 0.24 | 510.3 / 14.7 / 0.24 | 513.9 / 29.3 / 0.24 | 522.5 / 57.5 / 0.24 | 709.9 / 84.7 / 0.18 | 705.0 / 170.6 / 0.20 | 844.8 / 569.4 / 0.23 | 3254.5 / 591.2 / 0.13 |
| vllm_triton_w4a16 w4 zp | 541.6 / 0.9 / 0.23 | 546.0 / 3.4 / 0.22 | 546.7 / 13.7 / 0.23 | 550.1 / 27.3 / 0.23 | 572.2 / 52.5 / 0.22 | 757.7 / 79.4 / 0.17 | 752.3 / 159.8 / 0.19 | 902.2 / 533.2 / 0.22 | 3460.7 / 556.0 / 0.12 |
| vllm_awq_triton_gemm w4 zp | 420.6 / 1.1 / 0.29 | 386.6 / 4.9 / 0.32 | 387.1 / 19.4 / 0.32 | 389.5 / 38.6 / 0.32 | 731.0 / 41.1 / 0.17 | 1413.3 / 42.5 / 0.09 | 2770.1 / 43.4 / 0.05 | 10835.8 / 44.4 / 0.02 | 43183.4 / 44.6 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 192.2 / 2.4 / 0.64 | 192.3 / 9.8 / 0.64 | 192.6 / 39.0 / 0.64 | 192.4 / 78.1 / 0.65 | 198.4 / 151.5 / 0.64 | 205.1 / 293.2 / 0.64 | 229.2 / 524.7 / 0.61 | 492.2 / 977.2 / 0.40 | 1516.4 / 1268.9 / 0.28 |
| conch_mixed_precision w4 sym | 615.3 / 0.8 / 0.20 | 615.6 / 3.1 / 0.20 | 671.3 / 11.2 / 0.18 | 799.4 / 18.8 / 0.15 | 1061.7 / 28.3 / 0.12 | 1750.9 / 34.3 / 0.07 | 1741.6 / 69.0 / 0.08 | 6858.7 / 70.1 / 0.03 | 27287.1 / 70.5 / 0.02 |
| conch_mixed_precision w4 zp | 665.4 / 0.7 / 0.18 | 666.1 / 2.8 / 0.18 | 738.0 / 10.2 / 0.17 | 864.9 / 17.4 / 0.14 | 1140.2 / 26.4 / 0.11 | 1811.3 / 33.2 / 0.07 | 1802.7 / 66.7 / 0.08 | 7174.6 / 67.0 / 0.03 | 28602.7 / 67.3 / 0.01 |
| conch_mixed_precision w8 sym | 674.0 / 0.7 / 0.35 | 674.2 / 2.8 / 0.35 | 724.2 / 10.4 / 0.33 | 850.7 / 17.7 / 0.28 | 1124.6 / 26.7 / 0.22 | 1849.4 / 32.5 / 0.13 | 1842.8 / 65.3 / 0.14 | 7235.7 / 66.5 / 0.04 | 28805.3 / 66.8 / 0.02 |
| conch_mixed_precision w8 zp | 762.1 / 0.6 / 0.32 | 761.1 / 2.5 / 0.32 | 822.3 / 9.1 / 0.29 | 952.3 / 15.8 / 0.26 | 1233.3 / 24.4 / 0.20 | 1927.7 / 31.2 / 0.13 | 1915.8 / 62.8 / 0.14 | 7591.6 / 63.4 / 0.04 | 30283.9 / 63.5 / 0.02 |
| vllm_exllama_gptq w4 sym | 1385.0 / 0.3 / 0.09 | 1344.5 / 1.4 / 0.09 | 1135.0 / 6.6 / 0.11 | 747.3 / 20.1 / 0.17 | 505.1 / 59.5 / 0.25 | 550.7 / 109.2 / 0.24 | 708.2 / 169.8 / 0.20 | 710.5 / 677.0 / 0.28 | 1566.0 / 1228.7 / 0.27 |
| vllm_exllama_gptq w4 zp | 1380.6 / 0.3 / 0.09 | 1345.5 / 1.4 / 0.09 | 1113.9 / 6.7 / 0.11 | 762.9 / 19.7 / 0.16 | 505.9 / 59.4 / 0.25 | 552.0 / 108.9 / 0.24 | 710.1 / 169.4 / 0.20 | 712.0 / 675.6 / 0.28 | 1571.3 / 1224.5 / 0.27 |
| vllm_exllama_gptq w8 sym | 1351.9 / 0.3 / 0.18 | 1022.3 / 1.8 / 0.23 | 829.8 / 9.1 / 0.29 | 485.8 / 30.9 / 0.50 | 510.0 / 59.0 / 0.48 | 555.9 / 108.2 / 0.45 | 711.9 / 168.9 / 0.36 | 716.3 / 671.6 / 0.44 | 1584.1 / 1214.6 / 0.34 |
| vllm_exllama_gptq w8 zp | 1345.7 / 0.3 / 0.18 | 1028.5 / 1.8 / 0.23 | 824.6 / 9.1 / 0.29 | 481.6 / 31.2 / 0.50 | 506.8 / 59.3 / 0.48 | 554.5 / 108.4 / 0.45 | 712.2 / 168.9 / 0.36 | 714.0 / 673.7 / 0.44 | 1585.9 / 1213.3 / 0.34 |
| dequant_triton+hipblaslt w4 sym | 191.0 / 2.5 / 0.64 | 217.5 / 8.6 / 0.56 | 214.3 / 35.1 / 0.57 | 210.9 / 71.3 / 0.59 | 226.0 / 133.0 / 0.56 | 227.4 / 264.4 / 0.57 | 252.0 / 477.2 / 0.56 | 491.9 / 978.0 / 0.40 | 1351.1 / 1424.1 / 0.31 |
| dequant_triton+hipblaslt w4 zp | 215.3 / 2.2 / 0.57 | 235.5 / 8.0 / 0.52 | 234.8 / 32.0 / 0.53 | 237.7 / 63.2 / 0.52 | 242.0 / 124.2 / 0.52 | 237.7 / 253.0 / 0.55 | 273.0 / 440.4 / 0.52 | 506.1 / 950.4 / 0.39 | 1368.4 / 1406.1 / 0.31 |
| dequant_triton+hipblaslt w8 sym | 222.3 / 2.1 / 1.07 | 221.3 / 8.5 / 1.08 | 229.6 / 32.7 / 1.04 | 223.4 / 67.3 / 1.08 | 235.0 / 127.9 / 1.03 | 230.5 / 260.9 / 1.08 | 254.6 / 472.4 / 1.01 | 490.1 / 981.4 / 0.64 | 1363.2 / 1411.5 / 0.40 |
| dequant_triton+hipblaslt w8 zp | 236.0 / 2.0 / 1.02 | 238.3 / 7.9 / 1.01 | 224.1 / 33.5 / 1.08 | 227.1 / 66.2 / 1.07 | 233.0 / 129.1 / 1.05 | 237.0 / 253.7 / 1.05 | 267.2 / 450.1 / 0.97 | 495.4 / 971.1 / 0.64 | 1378.4 / 1395.9 / 0.39 |
| aiter_a16w8_int8 w8 sym | 51.5 / 9.1 / 4.70 | 52.4 / 35.9 / 4.63 | 59.0 / 127.4 / 4.13 | 70.9 / 211.9 / 3.45 | 86.5 / 347.6 / 2.85 | 135.2 / 444.6 / 1.86 | 265.5 / 452.9 / 0.98 | 866.4 / 555.2 / 0.37 | 3311.4 / 581.1 / 0.16 |
| aiter_a16w8_fp8_block128 w8 sym | 43.6 / 10.8 / 5.39 | 46.0 / 40.8 / 5.11 | 48.9 / 153.7 / 4.83 | 67.4 / 223.1 / 3.52 | 82.4 / 364.7 / 2.91 | 118.6 / 506.8 / 2.06 | 260.6 / 461.4 / 0.97 | 675.1 / 712.5 / 0.46 | 2570.9 / 748.4 / 0.21 |
| aiter_a16wfp4 w4 sym | 104.7 / 4.5 / 1.19 | 106.6 / 17.6 / 1.17 | 105.7 / 71.1 / 1.19 | 121.8 / 123.4 / 1.04 | 242.6 / 123.9 / 0.53 | 484.1 / 124.2 / 0.28 | 970.7 / 123.9 / 0.15 | 3747.6 / 128.4 / 0.05 | 14864.5 / 129.4 / 0.03 |
| aiter_afp4wfp4 w4 sym | 36.5 / 12.9 / 3.42 | 40.0 / 47.0 / 3.13 | 36.0 / 209.1 / 3.48 | 39.9 / 376.6 / 3.15 | 42.4 / 708.4 / 2.99 | 63.9 / 941.4 / 2.02 | 90.9 / 1322.6 / 1.46 | 292.5 / 1644.6 / 0.54 | 609.1 / 3158.9 / 0.42 |
| aiter_a8w8_int8 w8 sym | 58.5 / 8.0 / 4.02 | 59.8 / 31.4 / 3.93 | 68.2 / 110.2 / 3.46 | 61.2 / 245.5 / 3.86 | 75.0 / 401.1 / 3.17 | 114.0 / 527.5 / 2.11 | 168.7 / 712.9 / 1.46 | 331.3 / 1452.0 / 0.85 | 1201.1 / 1602.0 / 0.35 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 16.7 / 3.1 / 3.15 | 16.4 / 12.8 / 3.19 | 16.5 / 50.7 / 3.19 | 18.4 / 91.0 / 2.88 | 31.5 / 106.4 / 1.70 | 34.5 / 194.5 / 1.59 | 36.5 / 368.0 / 1.58 | 66.2 / 811.1 / 1.11 | 183.4 / 1170.9 / 0.74 |
| vllm_triton_w4a16 w4 sym | 93.6 / 0.6 / 0.14 | 93.8 / 2.2 / 0.14 | 94.2 / 8.9 / 0.15 | 95.2 / 17.6 / 0.15 | 96.3 / 34.8 / 0.15 | 130.7 / 51.4 / 0.12 | 129.7 / 103.5 / 0.14 | 154.0 / 348.6 / 0.22 | 409.1 / 524.9 / 0.24 |
| vllm_triton_w4a16 w4 zp | 100.5 / 0.5 / 0.14 | 100.8 / 2.1 / 0.14 | 101.1 / 8.3 / 0.14 | 101.8 / 16.5 / 0.14 | 105.5 / 31.8 / 0.14 | 139.9 / 48.0 / 0.12 | 139.0 / 96.5 / 0.14 | 164.0 / 327.4 / 0.21 | 431.7 / 497.4 / 0.23 |
| vllm_awq_triton_gemm w4 zp | 59.0 / 0.9 / 0.23 | 58.1 / 3.6 / 0.24 | 60.0 / 14.0 / 0.23 | 61.5 / 27.3 / 0.23 | 100.4 / 33.4 / 0.15 | 178.8 / 37.5 / 0.09 | 350.9 / 38.3 / 0.05 | 1280.3 / 41.9 / 0.03 | 5056.8 / 42.5 / 0.02 |
| vllm_awq_dequant_mm w4 zp | 30.0 / 1.7 / 0.45 | 30.5 / 6.9 / 0.45 | 39.1 / 21.4 / 0.36 | 39.3 / 42.7 / 0.36 | 36.8 / 91.1 / 0.41 | 37.9 / 177.0 / 0.43 | 46.4 / 289.1 / 0.41 | 83.6 / 642.2 / 0.41 | 219.0 / 980.4 / 0.45 |
| conch_mixed_precision w4 sym | 117.5 / 0.4 / 0.12 | 117.7 / 1.8 / 0.12 | 127.2 / 6.6 / 0.11 | 150.2 / 11.2 / 0.09 | 199.6 / 16.8 / 0.07 | 322.1 / 20.8 / 0.05 | 321.4 / 41.8 / 0.06 | 940.7 / 57.1 / 0.04 | 3102.8 / 69.2 / 0.03 |
| conch_mixed_precision w4 zp | 126.0 / 0.4 / 0.11 | 126.1 / 1.7 / 0.11 | 138.4 / 6.1 / 0.10 | 167.5 / 10.0 / 0.09 | 218.6 / 15.3 / 0.07 | 333.1 / 20.1 / 0.05 | 332.6 / 40.4 / 0.06 | 981.8 / 54.7 / 0.04 | 3258.7 / 65.9 / 0.03 |
| conch_mixed_precision w8 sym | 127.3 / 0.4 / 0.21 | 127.5 / 1.6 / 0.21 | 137.5 / 6.1 / 0.20 | 160.2 / 10.5 / 0.17 | 209.6 / 16.0 / 0.13 | 338.4 / 19.8 / 0.09 | 338.7 / 39.6 / 0.09 | 995.5 / 53.9 / 0.05 | 3277.5 / 65.5 / 0.03 |
| conch_mixed_precision w8 zp | 140.2 / 0.4 / 0.19 | 140.2 / 1.5 / 0.19 | 152.4 / 5.5 / 0.18 | 176.2 / 9.5 / 0.16 | 227.3 / 14.8 / 0.12 | 352.8 / 19.0 / 0.08 | 352.3 / 38.1 / 0.09 | 1038.0 / 51.7 / 0.05 | 3420.5 / 62.8 / 0.03 |
| vllm_exllama_gptq w4 sym | 51.8 / 1.0 / 0.26 | 73.3 / 2.9 / 0.18 | 115.1 / 7.3 / 0.12 | 170.2 / 9.9 / 0.08 | 81.1 / 41.4 / 0.18 | 78.4 / 85.6 / 0.21 | 72.0 / 186.4 / 0.26 | 109.9 / 488.3 / 0.31 | 230.1 / 933.2 / 0.42 |
| vllm_exllama_gptq w4 zp | 52.1 / 1.0 / 0.26 | 73.3 / 2.9 / 0.19 | 115.5 / 7.3 / 0.12 | 170.4 / 9.8 / 0.08 | 81.0 / 41.4 / 0.18 | 79.0 / 84.9 / 0.21 | 72.0 / 186.4 / 0.26 | 110.5 / 485.7 / 0.31 | 229.4 / 935.9 / 0.42 |
| vllm_exllama_gptq w8 sym | 55.9 / 0.9 / 0.48 | 74.7 / 2.8 / 0.36 | 123.7 / 6.8 / 0.22 | 50.2 / 33.4 / 0.54 | 82.1 / 40.9 / 0.34 | 80.1 / 83.8 / 0.36 | 73.6 / 182.4 / 0.43 | 113.1 / 474.9 / 0.42 | 229.1 / 937.3 / 0.48 |
| vllm_exllama_gptq w8 zp | 55.8 / 0.9 / 0.48 | 74.5 / 2.8 / 0.36 | 123.8 / 6.8 / 0.22 | 50.1 / 33.5 / 0.55 | 82.5 / 40.7 / 0.34 | 81.5 / 82.3 / 0.36 | 73.9 / 181.7 / 0.43 | 113.0 / 475.1 / 0.42 | 229.8 / 934.4 / 0.48 |
| dequant_triton+hipblaslt w4 sym | 31.2 / 1.7 / 0.43 | 31.5 / 6.7 / 0.43 | 31.2 / 26.9 / 0.44 | 32.3 / 52.0 / 0.44 | 37.3 / 89.9 / 0.40 | 38.7 / 173.4 / 0.42 | 49.5 / 270.9 / 0.38 | 73.9 / 726.9 / 0.47 | 197.1 / 1089.4 / 0.49 |
| dequant_triton+hipblaslt w4 zp | 32.1 / 1.6 / 0.42 | 32.3 / 6.5 / 0.42 | 31.6 / 26.5 / 0.44 | 33.5 / 50.0 / 0.43 | 38.1 / 88.1 / 0.39 | 39.5 / 169.7 / 0.41 | 50.8 / 264.2 / 0.37 | 75.4 / 711.8 / 0.46 | 201.8 / 1063.9 / 0.48 |
| dequant_triton+hipblaslt w8 sym | 32.5 / 1.6 / 0.82 | 33.0 / 6.4 / 0.81 | 32.1 / 26.1 / 0.84 | 33.7 / 49.7 / 0.81 | 39.0 / 86.1 / 0.72 | 40.5 / 165.5 / 0.72 | 51.2 / 261.9 / 0.62 | 74.9 / 717.2 / 0.64 | 195.1 / 1100.7 / 0.57 |
| dequant_triton+hipblaslt w8 zp | 32.4 / 1.6 / 0.83 | 32.5 / 6.5 / 0.83 | 32.0 / 26.2 / 0.85 | 33.1 / 50.7 / 0.83 | 39.1 / 85.9 / 0.72 | 40.0 / 167.9 / 0.74 | 51.5 / 260.5 / 0.62 | 74.0 / 725.7 / 0.65 | 195.6 / 1097.7 / 0.57 |
| aiter_a16w8_int8 w8 sym | 12.3 / 4.3 / 2.21 | 13.5 / 15.6 / 2.02 | 13.7 / 61.3 / 2.00 | 16.9 / 99.4 / 1.64 | 20.9 / 160.3 / 1.35 | 33.0 / 203.1 / 0.90 | 48.6 / 276.3 / 0.66 | 173.7 / 309.1 / 0.28 | 397.4 / 540.4 / 0.28 |
| aiter_a16w8_fp8_block128 w8 sym | 10.5 / 5.0 / 2.51 | 11.1 / 18.9 / 2.38 | 12.1 / 69.5 / 2.20 | 15.7 / 106.8 / 1.71 | 19.8 / 169.1 / 1.39 | 29.2 / 229.7 / 0.99 | 48.3 / 277.8 / 0.65 | 124.9 / 429.7 / 0.38 | 311.0 / 690.5 / 0.35 |
| aiter_a16wfp4 w4 sym | 20.4 / 2.6 / 0.69 | 21.1 / 9.9 / 0.66 | 20.6 / 40.8 / 0.69 | 23.8 / 70.4 / 0.61 | 46.2 / 72.6 / 0.33 | 66.7 / 100.6 / 0.25 | 110.4 / 121.6 / 0.17 | 416.5 / 128.9 / 0.08 | 1625.5 / 132.1 / 0.06 |
| aiter_afp4wfp4 w4 sym | 11.2 / 4.7 / 1.25 | 11.8 / 17.8 / 1.19 | 8.5 / 98.9 / 1.67 | 9.0 / 185.8 / 1.59 | 10.5 / 320.2 / 1.41 | 13.5 / 498.4 / 1.16 | 21.1 / 636.4 / 0.82 | 63.2 / 849.2 / 0.43 | 124.0 / 1731.3 / 0.54 |
| aiter_a8w8_int8 w8 sym | 10.5 / 5.0 / 2.49 | 10.6 / 19.7 / 2.47 | 12.3 / 68.3 / 2.16 | 11.0 / 152.9 / 2.44 | 14.3 / 233.9 / 1.90 | 22.0 / 304.9 / 1.28 | 31.9 / 420.3 / 0.94 | 59.5 / 902.2 / 0.70 | 158.2 / 1357.6 / 0.56 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 81.5 / 6.4 / 6.43 | 81.8 / 25.6 / 6.41 | 83.6 / 100.4 / 6.29 | 86.5 / 193.9 / 6.10 | 85.8 / 391.0 / 6.19 | 105.5 / 636.4 / 5.11 | 133.1 / 1008.3 / 4.16 | 506.6 / 1059.9 / 1.26 | 1514.1 / 1418.3 / 0.65 |
| vllm_triton_w4a16 w4 sym | 123.0 / 4.3 / 1.10 | 124.4 / 16.9 / 1.09 | 134.3 / 62.4 / 1.02 | 137.1 / 122.4 / 1.01 | 137.5 / 244.0 / 1.03 | 169.3 / 396.4 / 0.88 | 309.0 / 434.3 / 0.53 | 974.0 / 551.2 / 0.26 | 3623.8 / 592.6 / 0.17 |
| vllm_triton_w4a16 w4 zp | 136.2 / 3.8 / 1.00 | 138.7 / 15.1 / 0.98 | 143.6 / 58.4 / 0.96 | 148.9 / 112.7 / 0.94 | 151.5 / 221.5 / 0.95 | 174.5 / 384.6 / 0.86 | 327.9 / 409.3 / 0.50 | 1026.5 / 523.0 / 0.24 | 3830.3 / 560.6 / 0.16 |
| vllm_awq_triton_gemm w4 zp | 405.1 / 1.3 / 0.34 | 407.1 / 5.2 / 0.34 | 414.4 / 20.2 / 0.33 | 422.4 / 39.7 / 0.33 | 815.0 / 41.2 / 0.18 | 1604.9 / 41.8 / 0.09 | 3185.8 / 42.1 / 0.05 | 12712.9 / 42.2 / 0.02 | 51620.7 / 41.6 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 228.6 / 2.3 / 0.60 | 231.2 / 9.1 / 0.59 | 231.7 / 36.2 / 0.60 | 246.4 / 68.1 / 0.57 | 483.3 / 69.4 / 0.30 | 502.0 / 133.7 / 0.30 | 265.6 / 505.3 / 0.62 | 621.9 / 863.3 / 0.40 | 1828.4 / 1174.5 / 0.33 |
| conch_mixed_precision w4 sym | 308.6 / 1.7 / 0.44 | 309.0 / 6.8 / 0.44 | 391.1 / 21.5 / 0.35 | 495.3 / 33.9 / 0.28 | 730.8 / 45.9 / 0.20 | 1255.3 / 53.5 / 0.12 | 2181.1 / 61.5 / 0.07 | 7747.3 / 69.3 / 0.03 | 30850.5 / 69.6 / 0.02 |
| conch_mixed_precision w4 zp | 487.6 / 1.1 / 0.28 | 489.5 / 4.3 / 0.28 | 536.3 / 15.6 / 0.26 | 631.1 / 26.6 / 0.22 | 834.6 / 40.2 / 0.17 | 1313.7 / 51.1 / 0.12 | 2287.2 / 58.7 / 0.07 | 8127.7 / 66.1 / 0.03 | 32507.1 / 66.1 / 0.02 |
| conch_mixed_precision w8 sym | 375.1 / 1.4 / 0.71 | 375.5 / 5.6 / 0.71 | 442.2 / 19.0 / 0.61 | 544.8 / 30.8 / 0.49 | 797.6 / 42.1 / 0.34 | 1324.5 / 50.7 / 0.21 | 2304.9 / 58.2 / 0.13 | 8233.4 / 65.2 / 0.05 | 32414.2 / 66.3 / 0.02 |
| conch_mixed_precision w8 zp | 544.5 / 1.0 / 0.49 | 544.8 / 3.9 / 0.49 | 591.4 / 14.2 / 0.46 | 686.3 / 24.4 / 0.40 | 889.6 / 37.7 / 0.31 | 1391.3 / 48.2 / 0.20 | 2423.0 / 55.4 / 0.12 | 8601.2 / 62.4 / 0.04 | 34405.8 / 62.4 / 0.02 |
| vllm_exllama_gptq w4 sym | 136.7 / 3.8 / 0.99 | 167.7 / 12.5 / 0.81 | 227.7 / 36.8 / 0.60 | 352.3 / 47.6 / 0.39 | 246.3 / 136.2 / 0.58 | 254.2 / 264.0 / 0.59 | 273.8 / 490.2 / 0.60 | 649.6 / 826.5 / 0.39 | 1877.1 / 1144.0 / 0.32 |
| vllm_exllama_gptq w4 zp | 136.7 / 3.8 / 1.00 | 167.6 / 12.5 / 0.81 | 228.0 / 36.8 / 0.60 | 352.4 / 47.6 / 0.40 | 245.7 / 136.5 / 0.58 | 254.5 / 263.7 / 0.59 | 275.7 / 486.9 / 0.60 | 638.1 / 841.3 / 0.39 | 1895.4 / 1133.0 / 0.32 |
| vllm_exllama_gptq w8 sym | 148.4 / 3.5 / 1.79 | 148.0 / 14.2 / 1.80 | 264.0 / 31.8 / 1.01 | 262.9 / 63.8 / 1.03 | 281.8 / 119.1 / 0.97 | 272.6 / 246.2 / 1.03 | 306.9 / 437.3 / 0.96 | 668.7 / 802.9 / 0.57 | 1955.7 / 1098.1 / 0.37 |
| vllm_exllama_gptq w8 zp | 148.2 / 3.5 / 1.81 | 147.7 / 14.2 / 1.82 | 265.1 / 31.6 / 1.02 | 280.0 / 59.9 / 0.97 | 264.9 / 126.7 / 1.04 | 287.9 / 233.1 / 0.98 | 296.2 / 453.1 / 1.00 | 669.2 / 802.3 / 0.57 | 1915.6 / 1121.1 / 0.38 |
| dequant_triton+hipblaslt w4 sym | 215.1 / 2.4 / 0.63 | 213.8 / 9.8 / 0.63 | 220.9 / 38.0 / 0.62 | 223.5 / 75.1 / 0.62 | 223.0 / 150.4 / 0.64 | 241.6 / 277.7 / 0.62 | 262.9 / 510.6 / 0.62 | 637.6 / 842.0 / 0.39 | 1642.8 / 1307.2 / 0.36 |
| dequant_triton+hipblaslt w4 zp | 227.1 / 2.3 / 0.60 | 228.1 / 9.2 / 0.60 | 228.1 / 36.8 / 0.60 | 229.9 / 73.0 / 0.61 | 230.8 / 145.4 / 0.62 | 248.4 / 270.1 / 0.61 | 269.5 / 498.0 / 0.61 | 641.3 / 837.2 / 0.39 | 1651.3 / 1300.5 / 0.36 |
| dequant_triton+hipblaslt w8 sym | 230.7 / 2.3 / 1.16 | 232.6 / 9.0 / 1.15 | 232.1 / 36.1 / 1.16 | 233.8 / 71.8 / 1.15 | 234.6 / 143.0 / 1.17 | 254.8 / 263.3 / 1.10 | 290.1 / 462.6 / 1.02 | 651.7 / 823.8 / 0.59 | 1664.4 / 1290.3 / 0.44 |
| dequant_triton+hipblaslt w8 zp | 232.7 / 2.3 / 1.15 | 232.8 / 9.0 / 1.15 | 235.4 / 35.6 / 1.15 | 237.3 / 70.7 / 1.15 | 238.9 / 140.5 / 1.15 | 256.0 / 262.1 / 1.10 | 277.5 / 483.7 / 1.07 | 654.2 / 820.6 / 0.59 | 1661.1 / 1292.8 / 0.44 |
| aiter_a16w8_int8 w8 sym | 61.4 / 8.5 / 4.41 | 69.8 / 30.1 / 3.88 | 81.3 / 103.2 / 3.35 | 95.7 / 175.3 / 2.86 | 121.2 / 276.9 / 2.29 | 196.5 / 341.5 / 1.45 | 325.8 / 412.0 / 0.92 | 1062.7 / 505.2 / 0.36 | 3905.8 / 549.8 / 0.19 |
| aiter_a16w8_fp8_block128 w8 sym | 49.6 / 10.6 / 5.29 | 56.5 / 37.1 / 4.65 | 70.8 / 118.4 / 3.73 | 89.1 / 188.3 / 2.98 | 115.9 / 289.5 / 2.33 | 168.9 / 397.3 / 1.64 | 324.3 / 413.9 / 0.90 | 808.5 / 664.0 / 0.47 | 2994.2 / 717.2 / 0.24 |
| aiter_a16wfp4 w4 sym | 30.5 / 17.2 / 4.57 | 29.5 / 71.2 / 4.74 | 104.6 / 80.2 / 1.35 | 166.8 / 100.6 / 0.86 | 310.5 / 108.1 / 0.47 | 599.0 / 112.0 / 0.26 | 1189.6 / 112.8 / 0.14 | 4057.4 / 132.3 / 0.06 | 16183.4 / 132.7 / 0.04 |
| aiter_afp4wfp4 w4 sym | 43.2 / 12.1 / 3.22 | 46.0 / 45.6 / 3.04 | 52.3 / 160.3 / 2.69 | 55.3 / 303.6 / 2.58 | 63.8 / 526.0 / 2.29 | 83.9 / 800.3 / 1.82 | 131.8 / 1018.7 / 1.26 | 250.9 / 2140.0 / 0.98 | 798.8 / 2688.5 / 0.71 |
| aiter_a8w8_int8 w8 sym | 70.5 / 7.4 / 3.72 | 71.0 / 29.6 / 3.70 | 60.0 / 139.9 / 4.40 | 72.5 / 231.4 / 3.67 | 94.6 / 354.7 / 2.85 | 109.0 / 615.8 / 2.53 | 204.7 / 655.6 / 1.42 | 383.5 / 1400.1 / 0.97 | 1410.5 / 1522.5 / 0.50 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 47.5 / 5.5 / 5.51 | 47.3 / 22.2 / 5.54 | 48.8 / 85.9 / 5.39 | 51.5 / 162.8 / 5.13 | 56.3 / 297.9 / 4.72 | 60.3 / 556.4 / 4.48 | 92.0 / 729.5 / 3.02 | 236.2 / 1136.7 / 1.38 | 843.1 / 1273.6 / 0.61 |
| vllm_triton_w4a16 w4 sym | 456.7 / 0.6 / 0.15 | 462.6 / 2.3 / 0.15 | 458.2 / 9.2 / 0.15 | 462.2 / 18.2 / 0.15 | 467.6 / 35.9 / 0.15 | 634.1 / 52.9 / 0.12 | 629.5 / 106.6 / 0.13 | 727.1 / 369.2 / 0.18 | 1958.4 / 548.3 / 0.16 |
| vllm_triton_w4a16 w4 zp | 490.0 / 0.5 / 0.14 | 490.8 / 2.1 / 0.14 | 536.4 / 7.8 / 0.13 | 494.2 / 17.0 / 0.14 | 514.6 / 32.6 / 0.14 | 679.4 / 49.4 / 0.11 | 676.2 / 99.2 / 0.12 | 782.2 / 343.2 / 0.17 | 2088.8 / 514.0 / 0.15 |
| vllm_awq_triton_gemm w4 zp | 392.9 / 0.7 / 0.17 | 257.1 / 4.1 / 0.27 | 244.6 / 17.1 / 0.28 | 245.4 / 34.2 / 0.28 | 426.5 / 39.3 / 0.17 | 803.8 / 41.7 / 0.10 | 1559.8 / 43.0 / 0.05 | 6053.0 / 44.3 / 0.02 | 24066.4 / 44.6 / 0.01 |
| vllm_awq_dequant_mm w4 zp | 104.9 / 2.5 / 0.65 | 104.5 / 10.0 / 0.65 | 105.9 / 39.6 / 0.65 | 109.1 / 76.9 / 0.64 | 115.8 / 144.9 / 0.62 | 127.2 / 263.9 / 0.60 | 154.3 / 434.9 / 0.54 | 333.6 / 804.7 / 0.39 | 1109.3 / 968.0 / 0.29 |
| conch_mixed_precision w4 sym | 569.4 / 0.5 / 0.12 | 568.6 / 1.8 / 0.12 | 615.4 / 6.8 / 0.11 | 730.5 / 11.5 / 0.10 | 973.2 / 17.2 / 0.07 | 1561.7 / 21.5 / 0.05 | 1558.2 / 43.1 / 0.05 | 4595.0 / 58.4 / 0.03 | 15267.9 / 70.3 / 0.02 |
| conch_mixed_precision w4 zp | 612.3 / 0.4 / 0.11 | 612.5 / 1.7 / 0.11 | 671.2 / 6.2 / 0.10 | 786.2 / 10.7 / 0.09 | 1038.7 / 16.2 / 0.07 | 1620.3 / 20.7 / 0.05 | 1613.0 / 41.6 / 0.05 | 4819.4 / 55.7 / 0.03 | 15956.6 / 67.3 / 0.02 |
| conch_mixed_precision w8 sym | 617.2 / 0.4 / 0.22 | 617.2 / 1.7 / 0.22 | 677.0 / 6.2 / 0.20 | 779.8 / 10.8 / 0.17 | 1014.7 / 16.5 / 0.14 | 1642.9 / 20.4 / 0.09 | 1638.7 / 41.0 / 0.09 | 4862.5 / 55.2 / 0.04 | 16024.8 / 67.0 / 0.02 |
| conch_mixed_precision w8 zp | 681.9 / 0.4 / 0.20 | 682.2 / 1.5 / 0.20 | 739.3 / 5.7 / 0.18 | 854.8 / 9.8 / 0.16 | 1119.4 / 15.0 / 0.12 | 1719.7 / 19.5 / 0.08 | 1710.1 / 39.2 / 0.09 | 5106.5 / 52.6 / 0.04 | 16894.0 / 63.6 / 0.02 |
| vllm_exllama_gptq w4 sym | 508.6 / 0.5 / 0.13 | 554.1 / 1.9 / 0.12 | 649.2 / 6.5 / 0.11 | 700.1 / 12.0 / 0.10 | 382.2 / 43.9 / 0.19 | 459.5 / 73.0 / 0.16 | 591.6 / 113.4 / 0.14 | 538.4 / 498.6 / 0.24 | 1126.9 / 952.8 / 0.28 |
| vllm_exllama_gptq w4 zp | 501.1 / 0.5 / 0.14 | 554.4 / 1.9 / 0.12 | 645.6 / 6.5 / 0.11 | 726.8 / 11.5 / 0.10 | 383.2 / 43.8 / 0.19 | 459.0 / 73.1 / 0.17 | 591.2 / 113.5 / 0.14 | 541.2 / 496.0 / 0.24 | 1127.1 / 952.6 / 0.28 |
| vllm_exllama_gptq w8 sym | 524.9 / 0.5 / 0.25 | 530.8 / 2.0 / 0.25 | 505.8 / 8.3 / 0.27 | 372.1 / 22.5 / 0.36 | 405.6 / 41.4 / 0.34 | 497.0 / 67.5 / 0.28 | 628.5 / 106.8 / 0.24 | 559.0 / 480.2 / 0.35 | 1141.8 / 940.4 / 0.34 |
| vllm_exllama_gptq w8 zp | 516.4 / 0.5 / 0.26 | 537.6 / 1.9 / 0.25 | 503.8 / 8.3 / 0.27 | 372.5 / 22.5 / 0.36 | 409.0 / 41.0 / 0.34 | 497.4 / 67.5 / 0.29 | 628.9 / 106.7 / 0.24 | 560.6 / 478.8 / 0.35 | 1144.7 / 938.0 / 0.34 |
| dequant_triton+hipblaslt w4 sym | 118.1 / 2.2 / 0.57 | 118.7 / 8.8 / 0.57 | 120.3 / 34.9 / 0.57 | 121.7 / 68.9 / 0.57 | 126.2 / 133.0 / 0.57 | 135.1 / 248.3 / 0.56 | 170.8 / 392.8 / 0.49 | 311.1 / 862.7 / 0.42 | 932.3 / 1151.7 / 0.34 |
| dequant_triton+hipblaslt w4 zp | 126.7 / 2.1 / 0.54 | 127.0 / 8.3 / 0.54 | 127.9 / 32.8 / 0.54 | 130.3 / 64.4 / 0.54 | 134.6 / 124.6 / 0.54 | 143.3 / 234.2 / 0.53 | 178.8 / 375.2 / 0.47 | 310.8 / 863.8 / 0.42 | 920.7 / 1166.2 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 118.8 / 2.2 / 1.12 | 108.8 / 9.6 / 1.23 | 120.1 / 34.9 / 1.12 | 113.1 / 74.2 / 1.19 | 127.5 / 131.6 / 1.07 | 134.8 / 248.9 / 1.05 | 160.4 / 418.3 / 0.93 | 303.0 / 886.0 / 0.65 | 914.3 / 1174.3 / 0.42 |
| dequant_triton+hipblaslt w8 zp | 122.6 / 2.1 / 1.09 | 118.2 / 8.9 / 1.14 | 123.5 / 34.0 / 1.09 | 122.4 / 68.5 / 1.11 | 131.4 / 127.6 / 1.05 | 135.9 / 246.9 / 1.04 | 173.5 / 386.7 / 0.86 | 307.7 / 872.4 / 0.64 | 919.9 / 1167.2 / 0.42 |
| aiter_a16w8_int8 w8 sym | 39.0 / 6.7 / 3.47 | 40.0 / 26.2 / 3.39 | 40.8 / 102.7 / 3.33 | 55.5 / 151.1 / 2.47 | 66.7 / 251.5 / 2.09 | 111.6 / 300.6 / 1.28 | 216.8 / 309.5 / 0.70 | 767.5 / 349.8 / 0.26 | 1854.3 / 579.0 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 30.8 / 8.5 / 4.26 | 31.3 / 33.5 / 4.20 | 33.7 / 124.6 / 3.92 | 53.3 / 157.3 / 2.50 | 63.4 / 264.6 / 2.13 | 95.5 / 351.5 / 1.46 | 216.9 / 309.4 / 0.68 | 572.3 / 469.0 / 0.34 | 1432.6 / 749.5 / 0.27 |
| aiter_a16wfp4 w4 sym | 93.2 / 2.8 / 0.75 | 95.6 / 11.0 / 0.73 | 92.8 / 45.2 / 0.76 | 106.7 / 78.6 / 0.67 | 213.1 / 78.7 / 0.34 | 312.8 / 107.3 / 0.25 | 535.7 / 125.3 / 0.16 | 2057.9 / 130.4 / 0.06 | 8082.2 / 132.9 / 0.04 |
| aiter_afp4wfp4 w4 sym | 38.3 / 6.8 / 1.82 | 41.0 / 25.5 / 1.70 | 28.5 / 147.4 / 2.46 | 31.9 / 263.1 / 2.21 | 41.7 / 402.6 / 1.71 | 55.8 / 601.6 / 1.30 | 90.5 / 741.9 / 0.84 | 292.1 / 919.0 / 0.32 | 542.4 / 1979.6 / 0.31 |
| aiter_a8w8_int8 w8 sym | 44.0 / 6.0 / 2.98 | 43.8 / 23.9 / 2.99 | 51.8 / 80.9 / 2.54 | 44.5 / 188.5 / 2.97 | 56.9 / 294.7 / 2.34 | 94.0 / 357.0 / 1.44 | 141.0 / 476.1 / 0.99 | 246.5 / 1089.0 / 0.68 | 682.6 / 1573.0 / 0.41 |

## Full results (bf16): `us / TFLOPs / TB/s`


### llama3-8b/o_proj (N=4096, K=4096)

| impl | llama3-8b/o_proj M=1 | llama3-8b/o_proj M=4 | llama3-8b/o_proj M=16 | llama3-8b/o_proj M=32 | llama3-8b/o_proj M=64 | llama3-8b/o_proj M=128 | llama3-8b/o_proj M=256 | llama3-8b/o_proj M=1024 | llama3-8b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 12.3 / 2.7 / 2.72 | 9.7 / 13.9 / 3.47 | 10.4 / 51.9 / 3.27 | 12.6 / 85.3 / 2.71 | 18.7 / 114.7 / 1.85 | 19.4 / 221.4 / 1.84 | 20.4 / 420.7 / 1.85 | 48.0 / 715.3 / 1.05 | 96.0 / 1432.3 / 1.05 |
| vllm_triton_w4a16 w4 sym | 85.4 / 0.4 / 0.10 | 85.5 / 1.6 / 0.10 | 85.4 / 6.3 / 0.10 | 85.7 / 12.5 / 0.11 | 85.4 / 25.2 / 0.11 | 135.4 / 31.7 / 0.08 | 134.4 / 63.9 / 0.10 | 136.2 / 252.2 / 0.19 | 312.5 / 439.8 / 0.24 |
| vllm_triton_w4a16 w4 zp | 89.0 / 0.4 / 0.10 | 89.1 / 1.5 / 0.10 | 89.6 / 6.0 / 0.10 | 89.7 / 12.0 / 0.10 | 88.4 / 24.3 / 0.11 | 125.9 / 34.1 / 0.09 | 125.3 / 68.6 / 0.10 | 129.1 / 266.2 / 0.20 | 348.8 / 394.1 / 0.22 |
| vllm_awq_dequant_mm w4 zp | 27.6 / 1.2 / 0.32 | 27.8 / 4.8 / 0.32 | 30.2 / 17.8 / 0.30 | 32.9 / 32.7 / 0.28 | 33.1 / 64.8 / 0.29 | 35.2 / 121.9 / 0.31 | 33.0 / 260.0 / 0.39 | 56.6 / 606.6 / 0.45 | 117.9 / 1165.7 / 0.64 |
| conch_mixed_precision w4 sym | 102.8 / 0.3 / 0.08 | 102.9 / 1.3 / 0.09 | 109.9 / 4.9 / 0.08 | 128.6 / 8.4 / 0.07 | 165.4 / 13.0 / 0.06 | 262.9 / 16.3 / 0.04 | 262.4 / 32.7 / 0.05 | 521.7 / 65.9 / 0.05 | 2073.6 / 66.3 / 0.04 |
| conch_mixed_precision w4 zp | 113.4 / 0.3 / 0.08 | 113.4 / 1.2 / 0.08 | 121.2 / 4.4 / 0.07 | 138.9 / 7.7 / 0.07 | 176.4 / 12.2 / 0.06 | 275.3 / 15.6 / 0.04 | 275.4 / 31.2 / 0.05 | 544.9 / 63.1 / 0.05 | 2134.7 / 64.4 / 0.04 |
| conch_mixed_precision w8 sym | 105.9 / 0.3 / 0.16 | 105.8 / 1.3 / 0.16 | 113.7 / 4.7 / 0.15 | 131.0 / 8.2 / 0.13 | 169.5 / 12.7 / 0.11 | 272.5 / 15.8 / 0.07 | 271.5 / 31.6 / 0.08 | 540.9 / 63.5 / 0.06 | 2134.2 / 64.4 / 0.04 |
| conch_mixed_precision w8 zp | 127.1 / 0.3 / 0.14 | 127.5 / 1.1 / 0.14 | 136.5 / 3.9 / 0.13 | 154.0 / 7.0 / 0.12 | 194.7 / 11.0 / 0.09 | 292.4 / 14.7 / 0.07 | 291.3 / 29.5 / 0.07 | 576.7 / 59.6 / 0.06 | 2255.6 / 60.9 / 0.04 |
| dequant_triton+hipblaslt w4 sym | 24.8 / 1.4 / 0.35 | 20.7 / 6.5 / 0.42 | 22.9 / 23.5 / 0.39 | 24.4 / 43.9 / 0.38 | 30.9 / 69.6 / 0.31 | 30.3 / 141.6 / 0.35 | 33.1 / 259.3 / 0.39 | 59.6 / 576.8 / 0.43 | 110.0 / 1249.7 / 0.69 |
| dequant_triton+hipblaslt w4 zp | 25.4 / 1.3 / 0.34 | 22.9 / 5.9 / 0.38 | 23.0 / 23.4 / 0.39 | 24.5 / 43.8 / 0.38 | 30.9 / 69.6 / 0.32 | 30.5 / 140.8 / 0.35 | 33.1 / 259.2 / 0.39 | 60.4 / 569.3 / 0.42 | 113.2 / 1214.6 / 0.67 |
| dequant_triton+hipblaslt w8 sym | 25.6 / 1.3 / 0.67 | 22.0 / 6.1 / 0.78 | 23.2 / 23.1 / 0.74 | 24.7 / 43.4 / 0.71 | 29.5 / 72.7 / 0.61 | 31.6 / 135.8 / 0.60 | 35.1 / 245.1 / 0.61 | 60.5 / 568.2 / 0.56 | 109.4 / 1255.8 / 0.77 |
| dequant_triton+hipblaslt w8 zp | 25.3 / 1.3 / 0.68 | 21.9 / 6.1 / 0.79 | 23.0 / 23.3 / 0.76 | 24.8 / 43.2 / 0.71 | 31.1 / 69.0 / 0.58 | 32.0 / 134.2 / 0.60 | 34.4 / 249.7 / 0.62 | 60.5 / 567.7 / 0.56 | 110.7 / 1241.9 / 0.76 |
| aiter_a16w8_int8 w8 sym | 9.7 / 3.5 / 1.78 | 10.0 / 13.5 / 1.74 | 12.6 / 42.6 / 1.39 | 12.6 / 85.1 / 1.41 | 15.6 / 138.0 / 1.18 | 26.8 / 160.1 / 0.72 | 35.4 / 242.5 / 0.61 | 66.9 / 513.2 / 0.51 | 241.8 / 568.5 / 0.35 |
| aiter_a16w8_fp8_block128 w8 sym | 7.8 / 4.3 / 2.15 | 8.7 / 15.3 / 1.93 | 9.5 / 56.4 / 1.79 | 12.0 / 89.7 / 1.45 | 14.0 / 153.3 / 1.27 | 24.1 / 178.3 / 0.78 | 37.6 / 228.3 / 0.56 | 57.7 / 595.6 / 0.58 | 206.8 / 664.5 / 0.41 |
| aiter_a16wfp4 w4 sym | 16.3 / 2.1 / 0.55 | 16.5 / 8.1 / 0.54 | 16.5 / 32.5 / 0.56 | 19.8 / 54.3 / 0.48 | 19.9 / 107.8 / 0.50 | 38.3 / 112.1 / 0.29 | 74.0 / 116.0 / 0.18 | 284.7 / 120.7 / 0.09 | 1131.8 / 121.4 / 0.07 |
| aiter_afp4wfp4 w4 sym | 7.3 / 4.6 / 1.22 | 7.6 / 17.8 / 1.19 | 7.0 / 77.0 / 1.30 | 7.3 / 147.0 / 1.27 | 7.9 / 271.2 / 1.21 | 10.4 / 412.7 / 0.98 | 16.5 / 521.2 / 0.70 | 49.0 / 701.4 / 0.40 | 56.0 / 2453.1 / 0.92 |
| torch_scaled_mm_fp8 w8 sym | 12.9 / 2.6 / 1.31 | 13.0 / 10.3 / 1.29 | 8.7 / 61.7 / 1.95 | 9.0 / 119.1 / 1.91 | 9.1 / 235.4 / 1.93 | 10.9 / 392.6 / 1.68 | 13.4 / 640.8 / 1.49 | 28.0 / 1226.4 / 1.05 | 61.2 / 2246.2 / 1.10 |
| aiter_a8w8_int8 w8 sym | 8.5 / 4.0 / 1.99 | 8.7 / 15.4 / 1.94 | 9.9 / 54.2 / 1.72 | 8.9 / 120.1 / 1.92 | 11.6 / 185.3 / 1.52 | 17.3 / 248.6 / 1.06 | 25.6 / 335.1 / 0.78 | 38.4 / 895.1 / 0.77 | 105.1 / 1307.3 / 0.64 |
| aiter_gemm_a4w4_asm w4 sym | 7.7 / 4.3 / 1.16 | 7.8 / 17.2 / 1.15 | 8.1 / 66.3 / 1.12 | 8.6 / 125.3 / 1.08 | 9.8 / 219.6 / 0.98 | 8.4 / 511.2 / 1.22 | 8.7 / 987.3 / 1.33 | 14.8 / 2323.8 / 1.32 | 36.2 / 3798.5 / 1.42 |

### llama3-8b/qkv (N=6144, K=4096)

| impl | llama3-8b/qkv M=1 | llama3-8b/qkv M=4 | llama3-8b/qkv M=16 | llama3-8b/qkv M=32 | llama3-8b/qkv M=64 | llama3-8b/qkv M=128 | llama3-8b/qkv M=256 | llama3-8b/qkv M=1024 | llama3-8b/qkv M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 14.0 / 3.6 / 3.60 | 14.3 / 14.1 / 3.53 | 14.0 / 57.6 / 3.62 | 16.9 / 95.5 / 3.02 | 19.1 / 168.8 / 2.71 | 20.7 / 311.0 / 2.56 | 27.3 / 472.5 / 2.04 | 60.1 / 857.4 / 1.19 | 158.4 / 1301.9 / 0.85 |
| vllm_triton_w4a16 w4 sym | 85.6 / 0.6 / 0.15 | 85.6 / 2.4 / 0.15 | 85.6 / 9.4 / 0.15 | 85.3 / 18.9 / 0.16 | 84.7 / 38.1 / 0.17 | 136.0 / 47.4 / 0.12 | 134.7 / 95.6 / 0.14 | 161.2 / 319.7 / 0.21 | 456.8 / 451.3 / 0.21 |
| vllm_triton_w4a16 w4 zp | 89.2 / 0.6 / 0.15 | 89.7 / 2.2 / 0.15 | 89.8 / 9.0 / 0.15 | 89.9 / 17.9 / 0.15 | 88.8 / 36.3 / 0.16 | 125.9 / 51.2 / 0.12 | 125.3 / 102.8 / 0.15 | 175.5 / 293.7 / 0.19 | 502.8 / 410.0 / 0.19 |
| vllm_awq_dequant_mm w4 zp | 33.7 / 1.5 / 0.39 | 34.2 / 5.9 / 0.39 | 34.4 / 23.4 / 0.39 | 34.6 / 46.6 / 0.40 | 36.0 / 89.5 / 0.40 | 41.5 / 155.2 / 0.38 | 43.4 / 296.7 / 0.42 | 79.7 / 646.7 / 0.43 | 174.8 / 1179.6 / 0.56 |
| conch_mixed_precision w4 sym | 104.6 / 0.5 / 0.12 | 104.6 / 1.9 / 0.12 | 111.6 / 7.2 / 0.12 | 130.4 / 12.4 / 0.10 | 166.9 / 19.3 / 0.09 | 264.2 / 24.4 / 0.06 | 263.7 / 48.9 / 0.07 | 779.0 / 66.2 / 0.04 | 3071.0 / 67.1 / 0.03 |
| conch_mixed_precision w4 zp | 114.4 / 0.4 / 0.11 | 114.4 / 1.8 / 0.12 | 123.2 / 6.5 / 0.11 | 141.7 / 11.4 / 0.10 | 178.4 / 18.1 / 0.08 | 277.0 / 23.3 / 0.06 | 276.7 / 46.6 / 0.07 | 818.6 / 63.0 / 0.04 | 3207.9 / 64.3 / 0.03 |
| conch_mixed_precision w8 sym | 106.9 / 0.5 / 0.24 | 107.2 / 1.9 / 0.24 | 115.3 / 7.0 / 0.22 | 132.5 / 12.2 / 0.20 | 171.4 / 18.8 / 0.16 | 274.5 / 23.5 / 0.10 | 273.2 / 47.2 / 0.11 | 809.2 / 63.7 / 0.06 | 3201.2 / 64.4 / 0.03 |
| conch_mixed_precision w8 zp | 128.0 / 0.4 / 0.20 | 128.2 / 1.6 / 0.20 | 137.9 / 5.8 / 0.19 | 154.9 / 10.4 / 0.17 | 196.4 / 16.4 / 0.14 | 294.6 / 21.9 / 0.10 | 293.4 / 43.9 / 0.11 | 866.2 / 59.5 / 0.05 | 3389.2 / 60.8 / 0.03 |
| dequant_triton+hipblaslt w4 sym | 29.5 / 1.7 / 0.44 | 29.4 / 6.9 / 0.45 | 28.6 / 28.2 / 0.47 | 33.1 / 48.7 / 0.41 | 34.2 / 94.1 / 0.42 | 35.5 / 181.6 / 0.44 | 42.7 / 302.0 / 0.43 | 72.8 / 708.2 / 0.47 | 172.4 / 1195.6 / 0.56 |
| dequant_triton+hipblaslt w4 zp | 29.4 / 1.7 / 0.45 | 29.3 / 6.9 / 0.45 | 28.2 / 28.6 / 0.48 | 32.3 / 49.9 / 0.42 | 33.9 / 95.1 / 0.42 | 35.3 / 182.4 / 0.44 | 41.8 / 308.0 / 0.44 | 74.1 / 695.8 / 0.46 | 175.5 / 1174.6 / 0.55 |
| dequant_triton+hipblaslt w8 sym | 31.4 / 1.6 / 0.81 | 31.4 / 6.4 / 0.81 | 30.0 / 26.8 / 0.86 | 32.7 / 49.3 / 0.80 | 35.7 / 90.3 / 0.75 | 37.0 / 173.9 / 0.76 | 43.5 / 296.2 / 0.71 | 73.0 / 705.7 / 0.64 | 172.7 / 1193.9 / 0.63 |
| dequant_triton+hipblaslt w8 zp | 31.1 / 1.6 / 0.83 | 31.3 / 6.4 / 0.82 | 29.9 / 26.9 / 0.87 | 32.3 / 49.9 / 0.82 | 35.8 / 90.0 / 0.76 | 36.6 / 175.8 / 0.77 | 42.5 / 303.4 / 0.73 | 72.7 / 708.5 / 0.64 | 172.5 / 1195.2 / 0.64 |
| aiter_a16w8_int8 w8 sym | 11.3 / 4.5 / 2.31 | 11.5 / 17.4 / 2.26 | 14.7 / 54.7 / 1.78 | 15.1 / 106.8 / 1.76 | 19.3 / 166.5 / 1.41 | 30.2 / 213.5 / 0.95 | 40.5 / 318.2 / 0.77 | 118.9 / 433.4 / 0.40 | 351.3 / 586.8 / 0.31 |
| aiter_a16w8_fp8_block128 w8 sym | 9.0 / 5.6 / 2.79 | 9.9 / 20.4 / 2.56 | 11.2 / 72.0 / 2.28 | 14.0 / 115.0 / 1.84 | 17.7 / 182.3 / 1.50 | 27.3 / 236.0 / 1.02 | 40.3 / 319.4 / 0.75 | 103.5 / 498.0 / 0.45 | 302.9 / 680.5 / 0.36 |
| aiter_a16wfp4 w4 sym | 16.6 / 3.0 / 0.81 | 16.7 / 12.0 / 0.80 | 16.6 / 48.5 / 0.82 | 20.0 / 80.5 / 0.70 | 38.0 / 84.7 / 0.39 | 55.9 / 115.2 / 0.29 | 108.9 / 118.4 / 0.17 | 395.7 / 130.2 / 0.09 | 1537.4 / 134.1 / 0.06 |
| aiter_afp4wfp4 w4 sym | 8.1 / 6.2 / 1.65 | 8.8 / 23.0 / 1.53 | 7.3 / 110.1 / 1.86 | 7.7 / 209.2 / 1.80 | 8.6 / 374.4 / 1.66 | 10.9 / 589.3 / 1.39 | 17.2 / 749.3 / 0.99 | 50.5 / 1020.5 / 0.56 | 100.0 / 2061.2 / 0.73 |
| torch_scaled_mm_fp8 w8 sym | 13.3 / 3.8 / 1.89 | 13.5 / 15.0 / 1.88 | 9.3 / 86.8 / 2.74 | 9.7 / 165.8 / 2.65 | 11.4 / 283.6 / 2.31 | 14.1 / 458.0 / 1.94 | 16.1 / 798.2 / 1.82 | 35.6 / 1447.6 / 1.18 | 115.4 / 1786.6 / 0.80 |
| aiter_a8w8_int8 w8 sym | 9.2 / 5.5 / 2.75 | 9.3 / 21.7 / 2.72 | 10.5 / 76.7 / 2.42 | 9.6 / 168.6 / 2.69 | 12.2 / 263.4 / 2.15 | 18.1 / 356.2 / 1.51 | 26.4 / 487.3 / 1.11 | 54.9 / 939.4 / 0.77 | 154.2 / 1336.9 / 0.60 |
| aiter_gemm_a4w4_asm w4 sym | 8.0 / 6.3 / 1.66 | 8.1 / 24.9 / 1.66 | 8.5 / 94.5 / 1.60 | 10.3 / 155.6 / 1.34 | 10.1 / 319.7 / 1.42 | 8.8 / 736.1 / 1.74 | 10.2 / 1259.3 / 1.67 | 20.6 / 2505.7 / 1.37 | 56.9 / 3621.2 / 1.27 |

### llama3-8b/gate_up (N=28672, K=4096)

| impl | llama3-8b/gate_up M=1 | llama3-8b/gate_up M=4 | llama3-8b/gate_up M=16 | llama3-8b/gate_up M=32 | llama3-8b/gate_up M=64 | llama3-8b/gate_up M=128 | llama3-8b/gate_up M=256 | llama3-8b/gate_up M=1024 | llama3-8b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 50.4 / 4.7 / 4.66 | 49.0 / 19.2 / 4.80 | 45.4 / 82.8 / 5.20 | 49.1 / 153.2 / 4.83 | 46.6 / 322.7 / 5.13 | 56.1 / 535.7 / 4.33 | 67.5 / 890.2 / 3.73 | 191.0 / 1259.4 / 1.58 | 628.6 / 1530.5 / 0.80 |
| vllm_triton_w4a16 w4 sym | 95.8 / 2.5 / 0.63 | 96.1 / 9.8 / 0.63 | 97.0 / 38.8 / 0.64 | 96.9 / 77.6 / 0.65 | 95.9 / 156.7 / 0.68 | 137.1 / 219.3 / 0.50 | 168.6 / 356.6 / 0.46 | 559.1 / 430.2 / 0.23 | 2049.6 / 469.4 / 0.16 |
| vllm_triton_w4a16 w4 zp | 102.6 / 2.3 / 0.59 | 102.1 / 9.2 / 0.60 | 103.4 / 36.4 / 0.60 | 103.8 / 72.4 / 0.61 | 100.9 / 149.1 / 0.65 | 127.4 / 235.9 / 0.55 | 181.5 / 331.3 / 0.43 | 578.0 / 416.1 / 0.22 | 2234.1 / 430.6 / 0.15 |
| vllm_awq_dequant_mm w4 zp | 84.1 / 2.8 / 0.73 | 84.7 / 11.1 / 0.72 | 95.5 / 39.4 / 0.65 | 94.4 / 79.6 / 0.67 | 91.9 / 163.6 / 0.71 | 98.9 / 303.9 / 0.70 | 117.6 / 511.4 / 0.66 | 241.6 / 995.6 / 0.53 | 738.2 / 1303.2 / 0.45 |
| conch_mixed_precision w4 sym | 203.0 / 1.2 / 0.30 | 202.8 / 4.6 / 0.30 | 219.0 / 17.2 / 0.28 | 254.8 / 29.5 / 0.25 | 327.8 / 45.9 / 0.20 | 522.2 / 57.6 / 0.13 | 1033.8 / 58.2 / 0.07 | 3582.6 / 67.1 / 0.04 | 14223.5 / 67.6 / 0.02 |
| conch_mixed_precision w4 zp | 223.4 / 1.1 / 0.27 | 231.5 / 4.1 / 0.27 | 239.0 / 15.7 / 0.26 | 275.1 / 27.3 / 0.23 | 349.8 / 43.0 / 0.19 | 549.0 / 54.8 / 0.13 | 1086.9 / 55.3 / 0.07 | 3767.3 / 63.8 / 0.03 | 14860.7 / 64.7 / 0.02 |
| conch_mixed_precision w8 sym | 227.2 / 1.0 / 0.53 | 209.8 / 4.5 / 0.57 | 229.1 / 16.4 / 0.53 | 260.9 / 28.8 / 0.47 | 338.4 / 44.4 / 0.36 | 543.9 / 55.3 / 0.23 | 1075.1 / 55.9 / 0.13 | 3733.3 / 64.4 / 0.05 | 14867.2 / 64.7 / 0.03 |
| conch_mixed_precision w8 zp | 251.3 / 0.9 / 0.48 | 251.4 / 3.7 / 0.48 | 270.7 / 13.9 / 0.45 | 304.6 / 24.7 / 0.40 | 386.6 / 38.9 / 0.32 | 582.4 / 51.6 / 0.22 | 1154.1 / 52.1 / 0.12 | 4004.6 / 60.1 / 0.05 | 15744.2 / 61.1 / 0.03 |
| dequant_triton+hipblaslt w4 sym | 95.1 / 2.5 / 0.64 | 103.0 / 9.1 / 0.59 | 93.7 / 40.1 / 0.66 | 112.4 / 66.9 / 0.56 | 106.5 / 141.1 / 0.61 | 115.5 / 260.4 / 0.60 | 149.2 / 403.1 / 0.52 | 240.9 / 998.5 / 0.53 | 656.6 / 1465.2 / 0.50 |
| dequant_triton+hipblaslt w4 zp | 94.1 / 2.5 / 0.65 | 99.8 / 9.4 / 0.61 | 94.9 / 39.6 / 0.65 | 103.9 / 72.3 / 0.61 | 104.4 / 144.0 / 0.62 | 111.7 / 269.2 / 0.62 | 143.5 / 419.0 / 0.54 | 240.1 / 1001.7 / 0.53 | 663.1 / 1450.8 / 0.50 |
| dequant_triton+hipblaslt w8 sym | 117.3 / 2.0 / 1.02 | 117.5 / 8.0 / 1.02 | 103.8 / 36.2 / 1.16 | 119.5 / 62.9 / 1.02 | 107.8 / 139.5 / 1.15 | 122.9 / 244.6 / 1.04 | 144.5 / 416.2 / 0.94 | 251.7 / 955.5 / 0.74 | 667.8 / 1440.6 / 0.58 |
| dequant_triton+hipblaslt w8 zp | 101.9 / 2.3 / 1.18 | 109.7 / 8.6 / 1.10 | 105.5 / 35.6 / 1.15 | 119.1 / 63.1 / 1.03 | 110.0 / 136.7 / 1.13 | 117.0 / 257.0 / 1.10 | 146.2 / 411.3 / 0.94 | 250.7 / 959.3 / 0.75 | 670.1 / 1435.8 / 0.58 |
| aiter_a16w8_int8 w8 sym | 32.9 / 7.1 / 3.68 | 35.7 / 26.3 / 3.40 | 46.1 / 81.6 / 2.65 | 47.7 / 157.5 / 2.58 | 62.1 / 242.0 / 2.02 | 97.0 / 309.9 / 1.33 | 154.4 / 389.4 / 0.89 | 421.7 / 570.4 / 0.45 | 1586.4 / 606.4 / 0.25 |
| aiter_a16w8_fp8_block128 w8 sym | 23.6 / 10.0 / 4.99 | 27.3 / 34.4 / 4.32 | 33.0 / 114.0 / 3.60 | 45.6 / 164.8 / 2.62 | 57.2 / 262.7 / 2.13 | 85.3 / 352.3 / 1.48 | 153.7 / 391.3 / 0.87 | 362.7 / 663.1 / 0.51 | 1361.4 / 706.7 / 0.28 |
| aiter_a16wfp4 w4 sym | 18.5 / 12.7 / 3.38 | 18.6 / 50.4 / 3.36 | 43.6 / 86.3 / 1.46 | 79.1 / 95.0 / 0.81 | 138.5 / 108.6 / 0.48 | 272.5 / 110.3 / 0.26 | 542.7 / 110.8 / 0.15 | 1855.8 / 129.6 / 0.07 | 7413.0 / 129.8 / 0.04 |
| aiter_afp4wfp4 w4 sym | 21.2 / 11.1 / 2.95 | 23.4 / 40.2 / 2.68 | 25.5 / 147.5 / 2.49 | 23.1 / 325.2 / 2.78 | 30.2 / 497.0 / 2.19 | 39.4 / 763.5 / 1.78 | 40.9 / 1469.1 / 1.90 | 104.4 / 2303.0 / 1.18 | 340.1 / 2828.5 / 0.90 |
| torch_scaled_mm_fp8 w8 sym | 23.7 / 9.9 / 4.96 | 24.1 / 39.0 / 4.89 | 24.1 / 155.8 / 4.91 | 26.3 / 286.0 / 4.55 | 27.3 / 550.2 / 4.45 | 30.9 / 972.4 / 4.06 | 40.6 / 1481.1 / 3.28 | 120.8 / 1990.3 / 1.49 | 396.3 / 2427.4 / 0.93 |
| aiter_a8w8_int8 w8 sym | 33.0 / 7.1 / 3.56 | 33.8 / 27.8 / 3.49 | 25.6 / 147.0 / 4.64 | 34.2 / 219.8 / 3.50 | 45.6 / 329.8 / 2.67 | 48.3 / 622.7 / 2.60 | 97.6 / 616.0 / 1.36 | 183.4 / 1311.6 / 0.98 | 650.6 / 1478.7 / 0.57 |
| aiter_gemm_a4w4_asm w4 sym | 14.0 / 16.8 / 4.46 | 13.9 / 67.6 / 4.51 | 14.2 / 265.6 / 4.48 | 14.4 / 523.5 / 4.48 | 15.4 / 977.1 / 4.30 | 18.7 / 1606.2 / 3.74 | 24.3 / 2473.4 / 3.19 | 66.0 / 3645.3 / 1.87 | 213.2 / 4513.4 / 1.44 |

### llama3-8b/down (N=4096, K=14336)

| impl | llama3-8b/down M=1 | llama3-8b/down M=4 | llama3-8b/down M=16 | llama3-8b/down M=32 | llama3-8b/down M=64 | llama3-8b/down M=128 | llama3-8b/down M=256 | llama3-8b/down M=1024 | llama3-8b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 24.5 / 4.8 / 4.79 | 24.9 / 18.9 / 4.72 | 35.2 / 53.4 / 3.36 | 29.8 / 125.9 / 3.97 | 29.5 / 255.0 / 4.06 | 34.1 / 440.6 / 3.58 | 52.4 / 574.2 / 2.42 | 115.1 / 1045.0 / 1.35 | 297.1 / 1618.9 / 0.90 |
| vllm_triton_w4a16 w4 sym | 298.7 / 0.4 / 0.10 | 322.6 / 1.5 / 0.09 | 290.8 / 6.5 / 0.11 | 291.6 / 12.9 / 0.11 | 288.9 / 26.0 / 0.11 | 463.7 / 32.4 / 0.07 | 461.8 / 65.1 / 0.09 | 464.7 / 258.8 / 0.15 | 1043.4 / 461.0 / 0.17 |
| vllm_triton_w4a16 w4 zp | 302.1 / 0.4 / 0.10 | 303.1 / 1.6 / 0.10 | 302.4 / 6.2 / 0.10 | 304.0 / 12.4 / 0.10 | 301.4 / 24.9 / 0.11 | 428.9 / 35.1 / 0.08 | 427.9 / 70.3 / 0.09 | 431.2 / 278.9 / 0.16 | 1162.2 / 413.9 / 0.16 |
| vllm_awq_dequant_mm w4 zp | 51.8 / 2.3 / 0.59 | 52.5 / 9.0 / 0.58 | 52.7 / 35.6 / 0.59 | 53.2 / 70.6 / 0.60 | 55.3 / 136.0 / 0.59 | 60.3 / 249.4 / 0.58 | 75.3 / 399.5 / 0.53 | 137.3 / 876.1 / 0.50 | 347.0 / 1386.2 / 0.52 |
| conch_mixed_precision w4 sym | 347.1 / 0.3 / 0.09 | 346.7 / 1.4 / 0.09 | 377.6 / 5.0 / 0.08 | 439.8 / 8.5 / 0.07 | 570.7 / 13.2 / 0.06 | 895.3 / 16.8 / 0.04 | 894.5 / 33.6 / 0.04 | 1784.1 / 67.4 / 0.04 | 7051.6 / 68.2 / 0.03 |
| conch_mixed_precision w4 zp | 383.6 / 0.3 / 0.08 | 384.1 / 1.2 / 0.08 | 413.3 / 4.5 / 0.07 | 472.1 / 8.0 / 0.07 | 606.4 / 12.4 / 0.05 | 936.0 / 16.1 / 0.04 | 935.1 / 32.1 / 0.04 | 1876.1 / 64.1 / 0.04 | 7333.7 / 65.6 / 0.03 |
| conch_mixed_precision w8 sym | 357.2 / 0.3 / 0.17 | 357.5 / 1.3 / 0.17 | 388.9 / 4.8 / 0.15 | 447.5 / 8.4 / 0.14 | 581.7 / 12.9 / 0.11 | 933.1 / 16.1 / 0.07 | 928.4 / 32.4 / 0.07 | 1854.5 / 64.8 / 0.05 | 7410.6 / 64.9 / 0.03 |
| conch_mixed_precision w8 zp | 431.5 / 0.3 / 0.14 | 431.4 / 1.1 / 0.14 | 463.2 / 4.1 / 0.13 | 526.6 / 7.1 / 0.12 | 668.6 / 11.2 / 0.09 | 997.8 / 15.1 / 0.07 | 993.1 / 30.3 / 0.07 | 1967.0 / 61.1 / 0.05 | 7737.3 / 62.2 / 0.03 |
| dequant_triton+hipblaslt w4 sym | 54.5 / 2.2 / 0.56 | 54.6 / 8.6 / 0.56 | 55.5 / 33.8 / 0.56 | 57.7 / 65.2 / 0.55 | 59.0 / 127.4 / 0.55 | 64.9 / 231.5 / 0.54 | 83.8 / 358.6 / 0.47 | 140.7 / 854.5 / 0.48 | 322.7 / 1490.6 / 0.56 |
| dequant_triton+hipblaslt w4 zp | 57.4 / 2.0 / 0.53 | 57.2 / 8.2 / 0.54 | 58.1 / 32.3 / 0.54 | 60.7 / 61.9 / 0.52 | 61.6 / 121.9 / 0.53 | 68.7 / 218.8 / 0.51 | 87.4 / 343.8 / 0.46 | 143.7 / 836.7 / 0.47 | 326.8 / 1472.0 / 0.56 |
| dequant_triton+hipblaslt w8 sym | 57.7 / 2.0 / 1.03 | 59.1 / 7.9 / 1.01 | 58.5 / 32.1 / 1.03 | 60.9 / 61.7 / 1.00 | 61.0 / 123.1 / 1.02 | 67.7 / 222.0 / 0.95 | 85.0 / 353.8 / 0.81 | 144.2 / 833.8 / 0.68 | 325.2 / 1479.2 / 0.65 |
| dequant_triton+hipblaslt w8 zp | 59.7 / 2.0 / 1.01 | 58.5 / 8.0 / 1.03 | 58.7 / 32.0 / 1.03 | 62.0 / 60.6 / 0.99 | 70.0 / 107.4 / 0.89 | 68.9 / 218.3 / 0.94 | 88.2 / 340.7 / 0.79 | 142.5 / 843.7 / 0.69 | 324.8 / 1481.1 / 0.65 |
| aiter_a16w8_int8 w8 sym | 20.1 / 5.8 / 3.01 | 20.3 / 23.1 / 2.99 | 28.6 / 65.7 / 2.14 | 28.5 / 132.1 / 2.17 | 36.3 / 207.2 / 1.73 | 69.7 / 215.6 / 0.94 | 113.3 / 265.3 / 0.62 | 210.0 / 572.7 / 0.47 | 762.5 / 630.8 / 0.28 |
| aiter_a16w8_fp8_block128 w8 sym | 17.1 / 6.9 / 3.44 | 17.9 / 26.2 / 3.29 | 18.7 / 100.7 / 3.18 | 27.4 / 137.1 / 2.19 | 31.3 / 240.4 / 1.95 | 57.2 / 262.7 / 1.11 | 121.5 / 247.4 / 0.56 | 178.5 / 673.8 / 0.54 | 644.3 / 746.6 / 0.33 |
| aiter_a16wfp4 w4 sym | 53.3 / 2.2 / 0.59 | 53.1 / 8.8 / 0.59 | 52.5 / 35.8 / 0.61 | 63.7 / 59.0 / 0.51 | 63.3 / 118.8 / 0.53 | 124.5 / 120.8 / 0.29 | 246.0 / 122.2 / 0.17 | 950.4 / 126.5 / 0.07 | 3771.0 / 127.6 / 0.05 |
| aiter_afp4wfp4 w4 sym | 10.7 / 11.0 / 2.92 | 12.1 / 38.7 / 2.57 | 16.5 / 114.2 / 1.91 | 18.5 / 203.0 / 1.71 | 22.5 / 333.5 / 1.43 | 31.3 / 480.1 / 1.06 | 51.4 / 585.5 / 0.69 | 149.0 / 806.9 / 0.32 | 162.2 / 2965.5 / 0.59 |
| torch_scaled_mm_fp8 w8 sym | 16.7 / 7.0 / 3.53 | 16.9 / 27.7 / 3.47 | 18.8 / 100.1 / 3.15 | 20.3 / 184.7 / 2.92 | 20.4 / 367.8 / 2.94 | 32.7 / 460.3 / 1.89 | 32.1 / 935.2 / 2.01 | 88.6 / 1357.8 / 0.92 | 188.1 / 2556.9 / 0.80 |
| aiter_a8w8_int8 w8 sym | 24.7 / 4.8 / 2.38 | 24.8 / 18.9 / 2.37 | 29.6 / 63.4 / 1.99 | 25.4 / 148.2 / 2.34 | 32.9 / 228.3 / 1.83 | 53.2 / 282.5 / 1.16 | 79.5 / 378.3 / 0.81 | 114.3 / 1051.9 / 0.72 | 319.7 / 1504.7 / 0.47 |
| aiter_gemm_a4w4_asm w4 sym | 21.2 / 5.5 / 1.47 | 21.5 / 21.8 / 1.45 | 22.6 / 83.0 / 1.39 | 24.3 / 154.9 / 1.31 | 32.1 / 233.9 / 1.00 | 23.2 / 647.9 / 1.43 | 23.7 / 1268.9 / 1.49 | 41.4 / 2907.6 / 1.15 | 100.5 / 4788.1 / 0.95 |

### llama3-70b/o_proj (N=8192, K=8192)

| impl | llama3-70b/o_proj M=1 | llama3-70b/o_proj M=4 | llama3-70b/o_proj M=16 | llama3-70b/o_proj M=32 | llama3-70b/o_proj M=64 | llama3-70b/o_proj M=128 | llama3-70b/o_proj M=256 | llama3-70b/o_proj M=1024 | llama3-70b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 26.1 / 5.1 / 5.13 | 26.5 / 20.3 / 5.08 | 27.5 / 78.1 / 4.90 | 29.3 / 146.5 / 4.62 | 30.7 / 279.5 / 4.43 | 41.0 / 418.8 / 3.37 | 57.8 / 594.0 / 2.46 | 130.8 / 1050.4 / 1.28 | 343.2 / 1601.6 / 0.78 |
| vllm_triton_w4a16 w4 sym | 168.2 / 0.8 / 0.21 | 167.2 / 3.2 / 0.21 | 168.1 / 12.8 / 0.21 | 167.0 / 25.7 / 0.21 | 164.8 / 52.1 / 0.22 | 271.7 / 63.2 / 0.14 | 261.5 / 131.4 / 0.16 | 317.8 / 432.4 / 0.21 | 1185.7 / 463.7 / 0.14 |
| vllm_triton_w4a16 w4 zp | 174.6 / 0.8 / 0.20 | 174.6 / 3.1 / 0.20 | 175.3 / 12.2 / 0.20 | 175.6 / 24.5 / 0.20 | 172.4 / 49.8 / 0.21 | 248.3 / 69.2 / 0.16 | 245.7 / 139.9 / 0.18 | 349.2 / 393.6 / 0.20 | 1338.8 / 410.6 / 0.13 |
| vllm_awq_dequant_mm w4 zp | 56.1 / 2.4 / 0.62 | 56.3 / 9.5 / 0.62 | 56.4 / 38.1 / 0.63 | 60.1 / 71.4 / 0.60 | 61.6 / 139.4 / 0.60 | 71.3 / 241.1 / 0.55 | 82.8 / 415.0 / 0.52 | 150.5 / 913.0 / 0.45 | 399.0 / 1377.7 / 0.42 |
| conch_mixed_precision w4 sym | 200.1 / 0.7 / 0.17 | 199.8 / 2.7 / 0.17 | 213.3 / 10.1 / 0.17 | 250.1 / 17.2 / 0.14 | 320.8 / 26.8 / 0.11 | 517.3 / 33.2 / 0.07 | 514.8 / 66.7 / 0.08 | 2033.8 / 67.6 / 0.03 | 8076.3 / 68.1 / 0.02 |
| conch_mixed_precision w4 zp | 229.0 / 0.6 / 0.15 | 222.8 / 2.4 / 0.16 | 237.2 / 9.1 / 0.15 | 271.3 / 15.8 / 0.13 | 346.5 / 24.8 / 0.11 | 541.6 / 31.7 / 0.07 | 538.4 / 63.8 / 0.08 | 2118.7 / 64.9 / 0.03 | 8364.3 / 65.7 / 0.02 |
| conch_mixed_precision w8 sym | 207.1 / 0.6 / 0.33 | 206.9 / 2.6 / 0.33 | 221.9 / 9.7 / 0.31 | 256.4 / 16.8 / 0.27 | 332.1 / 25.9 / 0.21 | 537.1 / 32.0 / 0.14 | 536.1 / 64.1 / 0.14 | 2116.1 / 65.0 / 0.05 | 8408.5 / 65.4 / 0.02 |
| conch_mixed_precision w8 zp | 249.3 / 0.5 / 0.28 | 249.6 / 2.2 / 0.28 | 267.0 / 8.0 / 0.26 | 305.2 / 14.1 / 0.23 | 382.2 / 22.5 / 0.18 | 577.7 / 29.7 / 0.13 | 574.5 / 59.8 / 0.13 | 2268.2 / 60.6 / 0.04 | 8908.8 / 61.7 / 0.02 |
| dequant_triton+hipblaslt w4 sym | 63.3 / 2.1 / 0.55 | 64.7 / 8.3 / 0.54 | 65.7 / 32.7 / 0.53 | 68.6 / 62.6 / 0.52 | 70.9 / 121.1 / 0.52 | 82.6 / 207.9 / 0.47 | 100.5 / 341.8 / 0.43 | 173.4 / 792.6 / 0.39 | 373.3 / 1472.8 / 0.45 |
| dequant_triton+hipblaslt w4 zp | 63.1 / 2.1 / 0.55 | 66.0 / 8.1 / 0.53 | 65.2 / 32.9 / 0.54 | 69.8 / 61.5 / 0.51 | 70.7 / 121.5 / 0.52 | 84.1 / 204.2 / 0.46 | 98.9 / 347.5 / 0.44 | 174.1 / 789.4 / 0.39 | 376.4 / 1460.4 / 0.45 |
| dequant_triton+hipblaslt w8 sym | 68.2 / 2.0 / 1.00 | 67.8 / 7.9 / 1.01 | 70.7 / 30.4 / 0.97 | 72.4 / 59.3 / 0.96 | 75.3 / 114.0 / 0.93 | 88.1 / 194.9 / 0.82 | 97.8 / 351.5 / 0.78 | 176.0 / 780.7 / 0.58 | 376.9 / 1458.6 / 0.54 |
| dequant_triton+hipblaslt w8 zp | 69.8 / 1.9 / 0.98 | 69.9 / 7.7 / 0.98 | 71.1 / 30.2 / 0.97 | 74.0 / 58.0 / 0.94 | 76.7 / 112.0 / 0.92 | 88.2 / 194.8 / 0.83 | 103.2 / 333.0 / 0.75 | 177.9 / 772.4 / 0.57 | 376.6 / 1459.8 / 0.54 |
| aiter_a16w8_int8 w8 sym | 19.4 / 6.9 / 3.57 | 20.1 / 26.7 / 3.45 | 26.1 / 82.4 / 2.68 | 25.1 / 171.2 / 2.80 | 33.2 / 258.3 / 2.14 | 50.2 / 342.4 / 1.46 | 83.1 / 413.5 / 0.93 | 238.8 / 575.5 / 0.43 | 869.9 / 632.0 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 15.7 / 8.5 / 4.28 | 16.6 / 32.3 / 4.05 | 18.3 / 117.4 / 3.70 | 23.1 / 186.0 / 2.95 | 29.3 / 293.4 / 2.37 | 45.0 / 382.0 / 1.59 | 80.4 / 427.3 / 0.94 | 203.4 / 675.7 / 0.49 | 741.2 / 741.7 / 0.27 |
| aiter_a16wfp4 w4 sym | 32.0 / 4.2 / 1.11 | 31.9 / 16.8 / 1.12 | 32.1 / 66.8 / 1.13 | 38.9 / 110.3 / 0.94 | 75.6 / 113.6 / 0.50 | 148.3 / 115.8 / 0.27 | 292.5 / 117.5 / 0.15 | 1055.7 / 130.2 / 0.07 | 4205.4 / 130.7 / 0.04 |
| aiter_afp4wfp4 w4 sym | 12.7 / 10.6 / 2.82 | 11.4 / 47.1 / 3.14 | 11.6 / 185.9 / 3.12 | 12.5 / 342.4 / 2.90 | 16.0 / 535.2 / 2.30 | 20.6 / 834.5 / 1.86 | 33.1 / 1037.6 / 1.24 | 88.6 / 1550.6 / 0.64 | 182.5 / 3012.6 / 0.66 |
| torch_scaled_mm_fp8 w8 sym | 15.9 / 8.5 / 4.23 | 16.0 / 33.6 / 4.20 | 16.9 / 127.2 / 4.00 | 16.2 / 265.7 / 4.20 | 19.1 / 450.4 / 3.60 | 23.0 / 745.6 / 3.05 | 33.1 / 1037.0 / 2.22 | 65.7 / 2093.0 / 1.41 | 215.1 / 2555.5 / 0.78 |
| aiter_a8w8_int8 w8 sym | 17.5 / 7.7 / 3.85 | 17.3 / 31.1 / 3.90 | 19.8 / 108.2 / 3.40 | 17.5 / 245.6 / 3.88 | 22.8 / 377.3 / 3.02 | 35.1 / 488.9 / 2.00 | 51.8 / 663.8 / 1.42 | 107.2 / 1282.0 / 0.86 | 361.6 / 1520.2 / 0.46 |
| aiter_gemm_a4w4_asm w4 sym | 14.4 / 9.3 / 2.48 | 14.5 / 37.1 / 2.47 | 15.1 / 142.1 / 2.38 | 16.1 / 266.2 / 2.25 | 15.8 / 542.1 / 2.33 | 15.7 / 1091.0 / 2.43 | 18.5 / 1858.4 / 2.21 | 40.1 / 3426.3 / 1.42 | 110.9 / 4956.5 / 1.09 |

### llama3-70b/gate_up (N=57344, K=8192)

| impl | llama3-70b/gate_up M=1 | llama3-70b/gate_up M=4 | llama3-70b/gate_up M=16 | llama3-70b/gate_up M=32 | llama3-70b/gate_up M=64 | llama3-70b/gate_up M=128 | llama3-70b/gate_up M=256 | llama3-70b/gate_up M=1024 | llama3-70b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 155.3 / 6.1 / 6.05 | 157.3 / 23.9 / 5.98 | 159.3 / 94.4 / 5.91 | 163.5 / 183.9 / 5.77 | 162.1 / 370.9 / 5.85 | 175.6 / 684.9 / 5.45 | 207.3 / 1160.2 / 4.69 | 707.4 / 1360.0 / 1.52 | 2466.5 / 1560.2 / 0.60 |
| vllm_triton_w4a16 w4 sym | 285.4 / 3.3 / 0.85 | 285.8 / 13.1 / 0.85 | 297.3 / 50.6 / 0.82 | 295.4 / 101.8 / 0.83 | 281.5 / 213.6 / 0.89 | 323.7 / 371.5 / 0.80 | 620.1 / 387.9 / 0.45 | 2088.4 / 460.7 / 0.18 | 8047.5 / 478.2 / 0.10 |
| vllm_triton_w4a16 w4 zp | 312.6 / 3.0 / 0.78 | 308.2 / 12.2 / 0.79 | 324.4 / 46.3 / 0.76 | 319.7 / 94.0 / 0.78 | 300.1 / 200.3 / 0.84 | 351.1 / 342.5 / 0.74 | 666.0 / 361.1 / 0.42 | 2241.4 / 429.2 / 0.17 | 8798.8 / 437.4 / 0.09 |
| vllm_awq_dequant_mm w4 zp | 396.3 / 2.4 / 0.62 | 391.1 / 9.6 / 0.62 | 397.5 / 37.8 / 0.62 | 396.6 / 75.8 / 0.63 | 399.3 / 150.6 / 0.63 | 404.0 / 297.7 / 0.65 | 427.4 / 562.8 / 0.65 | 951.7 / 1010.9 / 0.40 | 2913.4 / 1320.9 / 0.27 |
| conch_mixed_precision w4 sym | 795.9 / 1.2 / 0.30 | 796.7 / 4.7 / 0.30 | 859.6 / 17.5 / 0.28 | 996.6 / 30.2 / 0.25 | 1281.1 / 46.9 / 0.20 | 2053.8 / 58.6 / 0.13 | 3609.3 / 66.6 / 0.08 | 14195.5 / 67.8 / 0.03 | 56687.0 / 67.9 / 0.01 |
| conch_mixed_precision w4 zp | 874.0 / 1.1 / 0.28 | 873.8 / 4.3 / 0.28 | 935.3 / 16.1 / 0.26 | 1080.5 / 27.8 / 0.23 | 1372.2 / 43.8 / 0.18 | 2154.1 / 55.8 / 0.12 | 3759.2 / 64.0 / 0.07 | 14926.0 / 64.5 / 0.03 | 59712.7 / 64.4 / 0.01 |
| conch_mixed_precision w8 sym | 821.1 / 1.1 / 0.58 | 822.7 / 4.6 / 0.58 | 899.4 / 16.7 / 0.53 | 1023.8 / 29.4 / 0.47 | 1324.7 / 45.4 / 0.37 | 2135.6 / 56.3 / 0.23 | 3714.1 / 64.8 / 0.14 | 14844.4 / 64.8 / 0.04 | 59126.7 / 65.1 / 0.02 |
| conch_mixed_precision w8 zp | 988.7 / 0.9 / 0.49 | 989.3 / 3.8 / 0.49 | 1066.0 / 14.1 / 0.45 | 1200.2 / 25.0 / 0.40 | 1523.4 / 39.5 / 0.32 | 2304.0 / 52.2 / 0.22 | 4006.5 / 60.0 / 0.13 | 15926.3 / 60.4 / 0.04 | 63526.6 / 60.6 / 0.02 |
| dequant_triton+hipblaslt w4 sym | 417.4 / 2.3 / 0.58 | 427.8 / 8.8 / 0.57 | 423.8 / 35.5 / 0.58 | 423.9 / 70.9 / 0.58 | 415.8 / 144.6 / 0.60 | 438.3 / 274.4 / 0.59 | 462.9 / 519.6 / 0.60 | 979.4 / 982.3 / 0.38 | 2710.8 / 1419.6 / 0.29 |
| dequant_triton+hipblaslt w4 zp | 450.5 / 2.1 / 0.54 | 430.3 / 8.7 / 0.57 | 431.1 / 34.9 / 0.57 | 433.0 / 69.4 / 0.57 | 441.6 / 136.2 / 0.57 | 446.6 / 269.3 / 0.58 | 510.7 / 470.9 / 0.54 | 969.9 / 991.9 / 0.39 | 2725.8 / 1411.8 / 0.29 |
| dequant_triton+hipblaslt w8 sym | 479.7 / 2.0 / 0.99 | 471.2 / 8.0 / 1.01 | 451.8 / 33.3 / 1.06 | 472.3 / 63.7 / 1.02 | 458.3 / 131.2 / 1.06 | 487.5 / 246.7 / 1.01 | 510.6 / 471.0 / 1.00 | 989.8 / 971.9 / 0.62 | 2739.4 / 1404.8 / 0.37 |
| dequant_triton+hipblaslt w8 zp | 461.8 / 2.0 / 1.04 | 483.1 / 7.8 / 1.00 | 448.1 / 33.6 / 1.08 | 480.2 / 62.6 / 1.01 | 457.5 / 131.4 / 1.07 | 475.6 / 252.8 / 1.05 | 531.7 / 452.3 / 0.97 | 991.4 / 970.4 / 0.62 | 2752.8 / 1397.9 / 0.37 |
| aiter_a16w8_int8 w8 sym | 104.2 / 9.0 / 4.65 | 112.7 / 33.3 / 4.30 | 135.7 / 110.8 / 3.58 | 146.4 / 205.4 / 3.34 | 181.5 / 331.2 / 2.71 | 298.3 / 403.1 / 1.68 | 514.6 / 467.4 / 1.01 | 1582.0 / 608.1 / 0.39 | 6078.4 / 633.1 / 0.17 |
| aiter_a16w8_fp8_block128 w8 sym | 77.4 / 12.1 / 6.08 | 85.1 / 44.2 / 5.53 | 102.6 / 146.5 / 4.60 | 132.5 / 226.9 / 3.58 | 167.2 / 359.6 / 2.86 | 247.7 / 485.5 / 1.97 | 511.1 / 470.6 / 0.98 | 1345.9 / 714.8 / 0.45 | 5174.2 / 743.7 / 0.20 |
| aiter_a16wfp4 w4 sym | 51.4 / 18.3 / 4.85 | 53.8 / 69.9 / 4.65 | 182.3 / 82.5 / 1.38 | 282.6 / 106.4 / 0.90 | 555.4 / 108.3 / 0.46 | 1101.3 / 109.2 / 0.24 | 2195.3 / 109.6 / 0.13 | 8161.1 / 117.9 / 0.05 | 165880.3 / 23.2 / 0.01 |
| aiter_afp4wfp4 w4 sym | 70.2 / 13.4 / 3.56 | 75.6 / 49.7 / 3.31 | 68.4 / 219.8 / 3.68 | 61.1 / 491.8 / 4.14 | 77.8 / 773.3 / 3.31 | 98.9 / 1216.1 / 2.68 | 140.8 / 1708.3 / 1.99 | 358.4 / 2684.2 / 1.04 | 1235.9 / 3113.7 / 0.60 |
| torch_scaled_mm_fp8 w8 sym | 89.9 / 10.4 / 5.23 | 88.3 / 42.6 / 5.33 | 91.3 / 164.6 / 5.17 | 88.1 / 341.4 / 5.38 | 96.6 / 622.4 / 4.95 | 101.7 / 1182.5 / 4.78 | 133.6 / 1799.9 / 3.75 | 357.8 / 2688.5 / 1.67 | 1542.5 / 2494.9 / 0.63 |
| aiter_a8w8_int8 w8 sym | 107.9 / 8.7 / 4.36 | 116.8 / 32.2 / 4.03 | 95.6 / 157.3 / 4.94 | 109.3 / 275.1 / 4.34 | 143.2 / 419.8 / 3.34 | 167.7 / 716.9 / 2.90 | 320.6 / 750.3 / 1.56 | 642.1 / 1498.4 / 0.93 | 2452.2 / 1569.3 / 0.40 |
| aiter_gemm_a4w4_asm w4 sym | 43.6 / 21.5 / 5.72 | 44.3 / 84.8 / 5.64 | 45.0 / 334.1 / 5.59 | 45.7 / 657.5 / 5.54 | 48.9 / 1230.9 / 5.26 | 54.4 / 2212.4 / 4.87 | 72.8 / 3303.5 / 3.85 | 215.4 / 4466.5 / 1.73 | 729.8 / 5272.9 / 1.01 |

### llama3-70b/down (N=8192, K=28672)

| impl | llama3-70b/down M=1 | llama3-70b/down M=4 | llama3-70b/down M=16 | llama3-70b/down M=32 | llama3-70b/down M=64 | llama3-70b/down M=128 | llama3-70b/down M=256 | llama3-70b/down M=1024 | llama3-70b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 75.4 / 6.2 / 6.23 | 74.5 / 25.2 / 6.31 | 76.9 / 97.8 / 6.12 | 78.9 / 190.5 / 5.98 | 84.5 / 355.8 / 5.62 | 94.3 / 637.5 / 5.08 | 126.7 / 949.1 / 3.86 | 350.1 / 1373.9 / 1.56 | 1182.5 / 1627.1 / 0.65 |
| vllm_triton_w4a16 w4 sym | 575.2 / 0.8 / 0.21 | 572.8 / 3.3 / 0.21 | 575.8 / 13.1 / 0.21 | 571.7 / 26.3 / 0.22 | 572.3 / 52.5 / 0.22 | 923.5 / 65.1 / 0.14 | 919.0 / 130.9 / 0.15 | 1061.5 / 453.1 / 0.18 | 4143.3 / 464.4 / 0.10 |
| vllm_triton_w4a16 w4 zp | 603.2 / 0.8 / 0.20 | 605.2 / 3.1 / 0.20 | 606.4 / 12.4 / 0.20 | 603.9 / 24.9 / 0.21 | 599.2 / 50.2 / 0.21 | 860.6 / 69.9 / 0.15 | 851.0 / 141.3 / 0.17 | 1175.4 / 409.3 / 0.17 | 4527.2 / 425.0 / 0.09 |
| vllm_awq_dequant_mm w4 zp | 192.2 / 2.4 / 0.64 | 193.9 / 9.7 / 0.63 | 192.8 / 39.0 / 0.64 | 193.8 / 77.6 / 0.64 | 198.3 / 151.6 / 0.64 | 204.2 / 294.5 / 0.64 | 219.9 / 546.9 / 0.64 | 450.9 / 1066.8 / 0.44 | 1414.6 / 1360.2 / 0.30 |
| conch_mixed_precision w4 sym | 683.0 / 0.7 / 0.18 | 683.5 / 2.7 / 0.18 | 739.9 / 10.2 / 0.17 | 867.3 / 17.3 / 0.14 | 1131.7 / 26.6 / 0.11 | 1803.5 / 33.3 / 0.07 | 1783.9 / 67.4 / 0.08 | 7051.5 / 68.2 / 0.03 | 28093.6 / 68.5 / 0.01 |
| conch_mixed_precision w4 zp | 759.1 / 0.6 / 0.16 | 759.6 / 2.5 / 0.16 | 811.3 / 9.3 / 0.15 | 940.7 / 16.0 / 0.13 | 1203.6 / 25.0 / 0.10 | 1863.0 / 32.3 / 0.07 | 1862.1 / 64.6 / 0.08 | 7364.2 / 65.3 / 0.03 | 29311.8 / 65.6 / 0.01 |
| conch_mixed_precision w8 sym | 711.8 / 0.7 / 0.34 | 712.7 / 2.6 / 0.34 | 780.1 / 9.6 / 0.31 | 897.6 / 16.7 / 0.27 | 1163.3 / 25.8 / 0.21 | 1867.9 / 32.2 / 0.13 | 1856.5 / 64.8 / 0.14 | 7409.8 / 64.9 / 0.04 | 29464.5 / 65.3 / 0.02 |
| conch_mixed_precision w8 zp | 860.6 / 0.5 / 0.28 | 864.4 / 2.2 / 0.28 | 925.2 / 8.1 / 0.26 | 1050.2 / 14.3 / 0.23 | 1338.4 / 22.5 / 0.18 | 1998.6 / 30.1 / 0.12 | 1983.2 / 60.6 / 0.13 | 7871.8 / 61.1 / 0.04 | 31410.9 / 61.3 / 0.02 |
| dequant_triton+hipblaslt w4 sym | 202.9 / 2.3 / 0.60 | 209.8 / 9.0 / 0.58 | 210.6 / 35.7 / 0.58 | 214.6 / 70.1 / 0.57 | 216.7 / 138.7 / 0.58 | 224.4 / 268.0 / 0.58 | 245.7 / 489.5 / 0.57 | 473.8 / 1015.2 / 0.41 | 1303.7 / 1475.9 / 0.33 |
| dequant_triton+hipblaslt w4 zp | 226.2 / 2.1 / 0.54 | 223.7 / 8.4 / 0.55 | 228.4 / 32.9 / 0.54 | 225.7 / 66.6 / 0.55 | 234.7 / 128.1 / 0.54 | 234.6 / 256.3 / 0.56 | 260.4 / 461.9 / 0.54 | 479.4 / 1003.3 / 0.41 | 1314.0 / 1464.3 / 0.32 |
| dequant_triton+hipblaslt w8 sym | 221.3 / 2.1 / 1.08 | 219.1 / 8.6 / 1.09 | 226.7 / 33.2 / 1.06 | 233.9 / 64.3 / 1.03 | 215.3 / 139.6 / 1.13 | 244.6 / 245.8 / 1.01 | 249.3 / 482.4 / 1.03 | 483.9 / 994.0 / 0.65 | 1310.7 / 1468.0 / 0.41 |
| dequant_triton+hipblaslt w8 zp | 239.5 / 2.0 / 1.00 | 231.8 / 8.1 / 1.04 | 239.2 / 31.4 / 1.01 | 235.0 / 64.0 / 1.03 | 251.8 / 119.4 / 0.97 | 242.8 / 247.6 / 1.03 | 270.7 / 444.3 / 0.96 | 492.8 / 976.1 / 0.64 | 1328.2 / 1448.7 / 0.41 |
| aiter_a16w8_int8 w8 sym | 50.7 / 9.3 / 4.78 | 53.2 / 35.3 / 4.55 | 72.2 / 104.0 / 3.37 | 71.7 / 209.7 / 3.41 | 86.3 / 348.6 / 2.86 | 140.1 / 429.3 / 1.80 | 263.6 / 456.2 / 0.99 | 787.5 / 610.9 / 0.40 | 3021.2 / 636.9 / 0.18 |
| aiter_a16w8_fp8_block128 w8 sym | 44.6 / 10.5 / 5.27 | 44.9 / 41.9 / 5.24 | 49.1 / 153.1 / 4.81 | 68.6 / 219.2 / 3.46 | 81.3 / 370.0 / 2.95 | 117.1 / 513.6 / 2.09 | 259.3 / 463.9 / 0.98 | 665.4 / 722.9 / 0.47 | 2542.3 / 756.9 / 0.21 |
| aiter_a16wfp4 w4 sym | 449.1 / 1.0 / 0.28 | 489.1 / 3.8 / 0.26 | 531.6 / 14.1 / 0.24 | 154.1 / 97.6 / 0.82 | 286.0 / 105.1 / 0.45 | 532.0 / 113.0 / 0.25 | 2517.5 / 47.8 / 0.06 | 7192.0 / 66.9 / 0.03 | 14944.0 / 128.8 / 0.03 |
| aiter_afp4wfp4 w4 sym | 36.0 / 13.0 / 3.46 | 36.7 / 51.2 / 3.40 | 37.1 / 202.6 / 3.38 | 44.9 / 335.1 / 2.80 | 42.2 / 712.1 / 3.00 | 62.1 / 968.2 / 2.07 | 90.3 / 1332.4 / 1.47 | 292.9 / 1642.2 / 0.54 | 610.4 / 3152.1 / 0.42 |
| torch_scaled_mm_fp8 w8 sym | 41.8 / 11.2 / 5.62 | 41.5 / 45.3 / 5.66 | 42.9 / 175.3 / 5.50 | 44.5 / 337.5 / 5.31 | 54.3 / 553.4 / 4.38 | 57.7 / 1042.0 / 4.17 | 85.2 / 1410.7 / 2.89 | 182.8 / 2631.5 / 1.54 | 734.0 / 2621.4 / 0.57 |
| aiter_a8w8_int8 w8 sym | 58.2 / 8.1 / 4.04 | 58.6 / 32.1 / 4.01 | 69.2 / 108.6 / 3.40 | 58.8 / 255.4 / 4.02 | 83.6 / 359.4 / 2.84 | 114.4 / 525.6 / 2.10 | 170.4 / 705.8 / 1.45 | 330.7 / 1454.6 / 0.85 | 1200.8 / 1602.3 / 0.35 |
| aiter_gemm_a4w4_asm w4 sym | 43.8 / 10.7 / 2.85 | 46.5 / 40.5 / 2.69 | 47.8 / 157.1 / 2.62 | 49.9 / 301.4 / 2.52 | 48.5 / 619.5 / 2.61 | 48.6 / 1236.3 / 2.65 | 58.7 / 2048.4 / 2.26 | 111.2 / 4324.8 / 1.41 | 341.1 / 5641.7 / 0.75 |

### qwen3-32b/o_proj (N=5120, K=5120)

| impl | qwen3-32b/o_proj M=1 | qwen3-32b/o_proj M=4 | qwen3-32b/o_proj M=16 | qwen3-32b/o_proj M=32 | qwen3-32b/o_proj M=64 | qwen3-32b/o_proj M=128 | qwen3-32b/o_proj M=256 | qwen3-32b/o_proj M=1024 | qwen3-32b/o_proj M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 16.2 / 3.2 / 3.23 | 16.3 / 12.9 / 3.23 | 16.1 / 52.0 / 3.27 | 17.1 / 98.1 / 3.10 | 23.2 / 144.6 / 2.32 | 25.5 / 263.3 / 2.16 | 36.4 / 368.6 / 1.58 | 64.2 / 835.7 / 1.14 | 177.7 / 1208.3 / 0.77 |
| vllm_triton_w4a16 w4 sym | 106.0 / 0.5 / 0.13 | 106.2 / 2.0 / 0.13 | 105.7 / 7.9 / 0.13 | 106.3 / 15.8 / 0.13 | 105.1 / 31.9 / 0.14 | 168.8 / 39.8 / 0.10 | 167.5 / 80.1 / 0.11 | 196.3 / 273.5 / 0.18 | 541.6 / 396.5 / 0.18 |
| vllm_triton_w4a16 w4 zp | 110.8 / 0.5 / 0.12 | 111.2 / 1.9 / 0.12 | 111.1 / 7.5 / 0.13 | 111.5 / 15.0 / 0.13 | 109.9 / 30.5 / 0.14 | 156.1 / 43.0 / 0.10 | 155.6 / 86.3 / 0.12 | 215.3 / 249.4 / 0.16 | 526.8 / 407.6 / 0.18 |
| vllm_awq_dequant_mm w4 zp | 30.1 / 1.7 / 0.45 | 30.6 / 6.8 / 0.45 | 39.3 / 21.3 / 0.35 | 39.2 / 42.8 / 0.36 | 36.8 / 91.1 / 0.41 | 37.7 / 178.1 / 0.43 | 46.0 / 291.5 / 0.41 | 82.1 / 654.1 / 0.42 | 220.7 / 973.0 / 0.44 |
| conch_mixed_precision w4 sym | 129.2 / 0.4 / 0.10 | 129.1 / 1.6 / 0.10 | 139.6 / 6.0 / 0.10 | 162.9 / 10.3 / 0.09 | 211.4 / 15.9 / 0.07 | 328.8 / 20.4 / 0.05 | 327.9 / 40.9 / 0.06 | 966.7 / 55.5 / 0.04 | 3196.2 / 67.2 / 0.03 |
| conch_mixed_precision w4 zp | 141.5 / 0.4 / 0.10 | 142.1 / 1.5 / 0.10 | 152.0 / 5.5 / 0.09 | 175.4 / 9.6 / 0.08 | 223.7 / 15.0 / 0.07 | 345.5 / 19.4 / 0.05 | 344.1 / 39.0 / 0.06 | 1015.0 / 52.9 / 0.03 | 3333.6 / 64.4 / 0.03 |
| conch_mixed_precision w8 sym | 133.1 / 0.4 / 0.20 | 133.4 / 1.6 / 0.20 | 145.6 / 5.8 / 0.18 | 167.4 / 10.0 / 0.16 | 216.0 / 15.5 / 0.13 | 342.2 / 19.6 / 0.09 | 341.4 / 39.3 / 0.09 | 1012.8 / 53.0 / 0.05 | 3351.1 / 64.1 / 0.03 |
| conch_mixed_precision w8 zp | 159.1 / 0.3 / 0.17 | 158.7 / 1.3 / 0.17 | 171.4 / 4.9 / 0.16 | 192.5 / 8.7 / 0.14 | 245.5 / 13.7 / 0.12 | 366.3 / 18.3 / 0.08 | 364.9 / 36.8 / 0.09 | 1075.6 / 49.9 / 0.04 | 3544.2 / 60.6 / 0.03 |
| dequant_triton+hipblaslt w4 sym | 31.5 / 1.7 / 0.43 | 31.8 / 6.6 / 0.43 | 31.2 / 26.9 / 0.44 | 32.7 / 51.3 / 0.43 | 37.5 / 89.5 / 0.40 | 38.8 / 173.1 / 0.42 | 49.3 / 272.1 / 0.38 | 72.3 / 742.2 / 0.48 | 190.8 / 1125.5 / 0.51 |
| dequant_triton+hipblaslt w4 zp | 31.7 / 1.7 / 0.43 | 32.5 / 6.5 / 0.42 | 31.3 / 26.8 / 0.45 | 33.5 / 50.0 / 0.43 | 37.8 / 88.8 / 0.40 | 39.5 / 169.8 / 0.41 | 49.9 / 268.7 / 0.38 | 72.9 / 736.3 / 0.47 | 194.3 / 1105.3 / 0.50 |
| dequant_triton+hipblaslt w8 sym | 32.8 / 1.6 / 0.81 | 33.6 / 6.2 / 0.80 | 32.1 / 26.1 / 0.84 | 34.5 / 48.7 / 0.79 | 38.9 / 86.3 / 0.72 | 40.5 / 165.6 / 0.72 | 50.6 / 265.4 / 0.63 | 72.5 / 740.3 / 0.66 | 188.2 / 1141.2 / 0.59 |
| dequant_triton+hipblaslt w8 zp | 32.2 / 1.6 / 0.83 | 32.7 / 6.4 / 0.82 | 31.7 / 26.4 / 0.86 | 33.8 / 49.6 / 0.81 | 38.9 / 86.2 / 0.72 | 40.4 / 166.0 / 0.73 | 50.8 / 264.3 / 0.63 | 72.7 / 738.2 / 0.66 | 189.6 / 1132.5 / 0.58 |
| aiter_a16w8_int8 w8 sym | 12.2 / 4.3 / 2.21 | 12.6 / 16.7 / 2.16 | 16.2 / 51.7 / 1.69 | 17.3 / 96.7 / 1.60 | 21.3 / 157.3 / 1.33 | 33.7 / 199.1 / 0.88 | 48.2 / 278.6 / 0.67 | 147.8 / 363.1 / 0.33 | 364.9 / 588.5 / 0.30 |
| aiter_a16w8_fp8_block128 w8 sym | 10.2 / 5.1 / 2.56 | 11.4 / 18.3 / 2.30 | 12.1 / 69.3 / 2.19 | 15.7 / 107.2 / 1.72 | 19.7 / 170.2 / 1.40 | 29.2 / 229.8 / 0.99 | 48.1 / 278.8 / 0.65 | 124.9 / 429.9 / 0.38 | 310.7 / 691.2 / 0.35 |
| aiter_a16wfp4 w4 sym | 20.5 / 2.6 / 0.68 | 20.6 / 10.2 / 0.68 | 20.2 / 41.5 / 0.71 | 24.7 / 68.0 / 0.59 | 47.9 / 70.1 / 0.32 | 69.3 / 96.8 / 0.24 | 113.8 / 118.0 / 0.17 | 416.7 / 128.8 / 0.08 | 1618.2 / 132.7 / 0.06 |
| aiter_afp4wfp4 w4 sym | 11.2 / 4.7 / 1.24 | 11.7 / 17.9 / 1.20 | 8.6 / 98.0 / 1.65 | 9.0 / 186.1 / 1.59 | 10.5 / 318.9 / 1.40 | 13.5 / 498.7 / 1.16 | 21.1 / 637.2 / 0.82 | 63.3 / 847.8 / 0.43 | 123.9 / 1733.7 / 0.54 |
| torch_scaled_mm_fp8 w8 sym | 15.9 / 3.3 / 1.65 | 16.1 / 13.0 / 1.63 | 10.9 / 76.8 / 2.42 | 11.3 / 148.5 / 2.37 | 13.3 / 251.4 / 2.04 | 17.1 / 392.6 / 1.65 | 19.7 / 680.2 / 1.53 | 40.6 / 1321.9 / 1.03 | 104.4 / 2056.6 / 0.85 |
| aiter_a8w8_int8 w8 sym | 10.6 / 5.0 / 2.48 | 10.7 / 19.7 / 2.47 | 12.3 / 68.1 / 2.15 | 11.0 / 152.3 / 2.43 | 14.4 / 233.1 / 1.89 | 22.0 / 304.4 / 1.28 | 31.6 / 424.1 / 0.95 | 59.6 / 901.3 / 0.70 | 158.2 / 1357.6 / 0.56 |
| aiter_gemm_a4w4_asm w4 sym | 9.8 / 5.3 / 1.42 | 10.6 / 19.7 / 1.31 | 11.0 / 75.9 / 1.28 | 11.5 / 145.8 / 1.25 | 13.2 / 253.5 / 1.11 | 10.7 / 625.0 / 1.45 | 12.7 / 1055.8 / 1.36 | 23.1 / 2322.6 / 1.18 | 65.2 / 3291.9 / 1.03 |

### qwen3-32b/gate_up (N=51200, K=5120)

| impl | qwen3-32b/gate_up M=1 | qwen3-32b/gate_up M=4 | qwen3-32b/gate_up M=16 | qwen3-32b/gate_up M=32 | qwen3-32b/gate_up M=64 | qwen3-32b/gate_up M=128 | qwen3-32b/gate_up M=256 | qwen3-32b/gate_up M=1024 | qwen3-32b/gate_up M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 81.5 / 6.4 / 6.43 | 80.7 / 26.0 / 6.50 | 83.0 / 101.0 / 6.34 | 85.6 / 196.1 / 6.17 | 84.8 / 395.6 / 6.27 | 104.8 / 640.1 / 5.14 | 128.0 / 1048.9 / 4.32 | 496.4 / 1081.5 / 1.29 | 1469.4 / 1461.4 / 0.67 |
| vllm_triton_w4a16 w4 sym | 182.6 / 2.9 / 0.74 | 183.2 / 11.4 / 0.74 | 187.6 / 44.7 / 0.73 | 185.5 / 90.5 / 0.75 | 182.1 / 184.2 / 0.78 | 207.7 / 323.1 / 0.72 | 404.8 / 331.6 / 0.41 | 1216.3 / 441.4 / 0.21 | 4532.5 / 473.8 / 0.13 |
| vllm_triton_w4a16 w4 zp | 194.8 / 2.7 / 0.70 | 194.6 / 10.8 / 0.70 | 199.4 / 42.1 / 0.69 | 198.6 / 84.5 / 0.70 | 189.9 / 176.7 / 0.76 | 219.1 / 306.4 / 0.69 | 417.8 / 321.3 / 0.40 | 1301.7 / 412.4 / 0.19 | 4953.7 / 433.5 / 0.12 |
| vllm_awq_dequant_mm w4 zp | 226.2 / 2.3 / 0.60 | 227.9 / 9.2 / 0.60 | 229.0 / 36.6 / 0.60 | 228.9 / 73.3 / 0.61 | 229.8 / 146.0 / 0.62 | 234.2 / 286.6 / 0.64 | 253.2 / 530.2 / 0.65 | 622.0 / 863.2 / 0.40 | 1674.6 / 1282.4 / 0.36 |
| conch_mixed_precision w4 sym | 500.8 / 1.0 / 0.27 | 501.3 / 4.2 / 0.27 | 541.4 / 15.5 / 0.25 | 635.6 / 26.4 / 0.22 | 828.8 / 40.5 / 0.17 | 1307.8 / 51.3 / 0.11 | 2250.9 / 59.6 / 0.07 | 7986.0 / 67.2 / 0.03 | 31928.7 / 67.3 / 0.02 |
| conch_mixed_precision w4 zp | 549.3 / 1.0 / 0.25 | 550.3 / 3.8 / 0.25 | 591.7 / 14.2 / 0.23 | 684.9 / 24.5 / 0.20 | 878.1 / 38.2 / 0.16 | 1358.0 / 49.4 / 0.11 | 2367.0 / 56.7 / 0.07 | 8428.3 / 63.7 / 0.03 | 33540.1 / 64.0 / 0.02 |
| conch_mixed_precision w8 sym | 541.3 / 1.0 / 0.49 | 533.6 / 3.9 / 0.50 | 564.8 / 14.9 / 0.47 | 658.0 / 25.5 / 0.41 | 850.1 / 39.5 / 0.32 | 1354.2 / 49.6 / 0.21 | 2357.5 / 56.9 / 0.12 | 8363.9 / 64.2 / 0.05 | 33265.4 / 64.6 / 0.02 |
| conch_mixed_precision w8 zp | 620.8 / 0.8 / 0.43 | 621.2 / 3.4 / 0.43 | 669.6 / 12.5 / 0.40 | 750.9 / 22.3 / 0.36 | 962.4 / 34.9 / 0.29 | 1444.6 / 46.5 / 0.20 | 2513.1 / 53.4 / 0.12 | 8916.0 / 60.2 / 0.04 | 35578.2 / 60.4 / 0.02 |
| dequant_triton+hipblaslt w4 sym | 213.6 / 2.5 / 0.63 | 217.5 / 9.6 / 0.62 | 213.8 / 39.2 / 0.64 | 220.1 / 76.2 / 0.63 | 222.6 / 150.7 / 0.64 | 232.7 / 288.4 / 0.64 | 258.1 / 520.0 / 0.64 | 612.3 / 876.8 / 0.41 | 1595.3 / 1346.2 / 0.37 |
| dequant_triton+hipblaslt w4 zp | 226.6 / 2.3 / 0.60 | 226.1 / 9.3 / 0.60 | 225.7 / 37.2 / 0.61 | 221.4 / 75.8 / 0.63 | 220.2 / 152.4 / 0.65 | 246.0 / 272.8 / 0.61 | 263.4 / 509.6 / 0.63 | 620.6 / 865.1 / 0.41 | 1600.1 / 1342.1 / 0.37 |
| dequant_triton+hipblaslt w8 sym | 229.6 / 2.3 / 1.16 | 229.9 / 9.1 / 1.16 | 230.4 / 36.4 / 1.16 | 233.2 / 72.0 / 1.16 | 234.3 / 143.2 / 1.17 | 250.8 / 267.5 / 1.12 | 271.2 / 494.8 / 1.09 | 630.0 / 852.2 / 0.61 | 1609.0 / 1334.6 / 0.45 |
| dequant_triton+hipblaslt w8 zp | 228.4 / 2.3 / 1.18 | 234.0 / 9.0 / 1.15 | 234.3 / 35.8 / 1.15 | 236.2 / 71.0 / 1.15 | 236.5 / 141.9 / 1.17 | 250.9 / 267.4 / 1.13 | 273.6 / 490.5 / 1.09 | 629.4 / 853.0 / 0.61 | 1612.5 / 1331.8 / 0.45 |
| aiter_a16w8_int8 w8 sym | 66.5 / 7.9 / 4.07 | 72.8 / 28.8 / 3.72 | 88.7 / 94.6 / 3.07 | 95.6 / 175.6 / 2.87 | 123.1 / 272.6 / 2.25 | 200.7 / 334.4 / 1.42 | 324.0 / 414.3 / 0.92 | 949.2 / 565.6 / 0.41 | 3520.6 / 610.0 / 0.21 |
| aiter_a16w8_fp8_block128 w8 sym | 49.4 / 10.6 / 5.31 | 57.0 / 36.8 / 4.61 | 70.8 / 118.5 / 3.73 | 88.9 / 188.7 / 2.99 | 114.0 / 294.4 / 2.36 | 168.2 / 399.0 / 1.65 | 323.8 / 414.5 / 0.90 | 802.2 / 669.2 / 0.47 | 2959.1 / 725.7 / 0.24 |
| aiter_a16wfp4 w4 sym | 29.2 / 17.9 / 4.77 | 30.8 / 68.1 / 4.53 | 104.2 / 80.5 / 1.35 | 174.9 / 95.9 / 0.82 | 324.8 / 103.3 / 0.45 | 626.3 / 107.1 / 0.24 | 1246.0 / 107.7 / 0.14 | 4191.4 / 128.1 / 0.06 | 16709.9 / 128.5 / 0.04 |
| aiter_afp4wfp4 w4 sym | 42.7 / 12.3 / 3.27 | 45.1 / 46.5 / 3.10 | 51.9 / 161.7 / 2.72 | 54.9 / 305.8 / 2.60 | 64.2 / 522.6 / 2.27 | 83.9 / 799.9 / 1.82 | 131.6 / 1019.8 / 1.26 | 249.0 / 2155.9 / 0.99 | 792.9 / 2708.6 / 0.72 |
| torch_scaled_mm_fp8 w8 sym | 43.9 / 11.9 / 5.97 | 44.1 / 47.6 / 5.96 | 44.9 / 186.8 / 5.88 | 46.7 / 359.1 / 5.69 | 48.1 / 697.3 / 5.59 | 61.4 / 1092.5 / 4.50 | 82.7 / 1622.5 / 3.50 | 294.5 / 1823.2 / 1.26 | 948.1 / 2265.0 / 0.74 |
| aiter_a8w8_int8 w8 sym | 70.3 / 7.5 / 3.73 | 72.0 / 29.1 / 3.65 | 60.0 / 139.9 / 4.40 | 72.8 / 230.3 / 3.65 | 94.8 / 353.8 / 2.84 | 109.2 / 614.6 / 2.53 | 205.0 / 654.9 / 1.41 | 386.4 / 1389.4 / 0.96 | 1413.1 / 1519.7 / 0.50 |
| aiter_gemm_a4w4_asm w4 sym | 24.7 / 21.2 / 5.64 | 24.5 / 85.6 / 5.71 | 25.5 / 329.4 / 5.53 | 26.7 / 628.0 / 5.34 | 29.9 / 1123.4 / 4.89 | 35.6 / 1885.5 / 4.29 | 49.9 / 2689.9 / 3.33 | 148.5 / 3614.5 / 1.66 | 467.0 / 4598.6 / 1.22 |

### qwen3-32b/down (N=5120, K=25600)

| impl | qwen3-32b/down M=1 | qwen3-32b/down M=4 | qwen3-32b/down M=16 | qwen3-32b/down M=32 | qwen3-32b/down M=64 | qwen3-32b/down M=128 | qwen3-32b/down M=256 | qwen3-32b/down M=1024 | qwen3-32b/down M=4096 |
|---|---|---|---|---|---|---|---|---|---|
| torch_dense (dense) | 47.3 / 5.5 / 5.54 | 47.8 / 21.9 / 5.49 | 48.7 / 86.2 / 5.41 | 51.0 / 164.6 / 5.18 | 56.7 / 296.0 / 4.70 | 69.0 / 486.6 / 3.92 | 99.2 / 676.7 / 2.80 | 231.4 / 1159.9 / 1.41 | 821.3 / 1307.4 / 0.63 |
| vllm_triton_w4a16 w4 sym | 519.0 / 0.5 / 0.13 | 517.1 / 2.0 / 0.13 | 517.8 / 8.1 / 0.13 | 518.1 / 16.2 / 0.13 | 511.0 / 32.8 / 0.14 | 823.7 / 40.7 / 0.09 | 819.8 / 81.9 / 0.10 | 954.1 / 281.3 / 0.14 | 2524.9 / 425.3 / 0.13 |
| vllm_triton_w4a16 w4 zp | 539.6 / 0.5 / 0.13 | 543.4 / 1.9 / 0.13 | 542.5 / 7.7 / 0.13 | 543.2 / 15.4 / 0.13 | 535.0 / 31.4 / 0.14 | 763.0 / 44.0 / 0.10 | 760.0 / 88.3 / 0.11 | 1051.1 / 255.4 / 0.12 | 2586.1 / 415.2 / 0.12 |
| vllm_awq_dequant_mm w4 zp | 102.1 / 2.6 / 0.67 | 102.1 / 10.3 / 0.67 | 103.3 / 40.6 / 0.67 | 106.0 / 79.2 / 0.66 | 112.7 / 148.8 / 0.64 | 124.3 / 269.9 / 0.61 | 148.2 / 452.8 / 0.57 | 344.1 / 780.1 / 0.38 | 1132.9 / 947.8 / 0.28 |
| conch_mixed_precision w4 sym | 624.9 / 0.4 / 0.11 | 626.1 / 1.7 / 0.11 | 679.5 / 6.2 / 0.10 | 792.3 / 10.6 / 0.09 | 1028.5 / 16.3 / 0.07 | 1598.2 / 21.0 / 0.05 | 1591.5 / 42.2 / 0.05 | 4746.4 / 56.6 / 0.03 | 15743.5 / 68.2 / 0.02 |
| conch_mixed_precision w4 zp | 688.8 / 0.4 / 0.10 | 689.6 / 1.5 / 0.10 | 741.6 / 5.7 / 0.09 | 852.6 / 9.8 / 0.08 | 1095.2 / 15.3 / 0.07 | 1706.7 / 19.7 / 0.04 | 1670.5 / 40.2 / 0.05 | 4974.4 / 54.0 / 0.03 | 16401.2 / 65.5 / 0.02 |
| conch_mixed_precision w8 sym | 640.6 / 0.4 / 0.21 | 640.9 / 1.6 / 0.21 | 704.1 / 6.0 / 0.19 | 814.6 / 10.3 / 0.17 | 1049.5 / 16.0 / 0.13 | 1662.0 / 20.2 / 0.09 | 1654.1 / 40.6 / 0.09 | 4947.8 / 54.3 / 0.04 | 16385.6 / 65.5 / 0.02 |
| conch_mixed_precision w8 zp | 775.1 / 0.3 / 0.17 | 775.5 / 1.4 / 0.17 | 837.4 / 5.0 / 0.16 | 937.0 / 9.0 / 0.14 | 1196.0 / 14.0 / 0.12 | 1786.6 / 18.8 / 0.08 | 1774.7 / 37.8 / 0.08 | 5300.1 / 50.6 / 0.04 | 17433.0 / 61.6 / 0.02 |
| dequant_triton+hipblaslt w4 sym | 116.7 / 2.2 / 0.58 | 117.5 / 8.9 / 0.58 | 118.2 / 35.5 / 0.58 | 121.3 / 69.2 / 0.57 | 124.5 / 134.8 / 0.57 | 141.0 / 238.0 / 0.54 | 168.1 / 399.3 / 0.50 | 293.1 / 915.7 / 0.45 | 892.2 / 1203.5 / 0.36 |
| dequant_triton+hipblaslt w4 zp | 120.5 / 2.2 / 0.57 | 124.7 / 8.4 / 0.55 | 116.7 / 35.9 / 0.59 | 124.2 / 67.5 / 0.56 | 132.4 / 126.7 / 0.54 | 148.7 / 225.6 / 0.51 | 175.1 / 383.2 / 0.48 | 301.9 / 889.1 / 0.43 | 900.6 / 1192.3 / 0.35 |
| dequant_triton+hipblaslt w8 sym | 118.1 / 2.2 / 1.13 | 115.9 / 9.0 / 1.15 | 118.7 / 35.3 / 1.13 | 119.8 / 70.0 / 1.13 | 126.7 / 132.4 / 1.08 | 139.6 / 240.4 / 1.01 | 168.9 / 397.4 / 0.88 | 294.6 / 911.1 / 0.67 | 896.4 / 1197.8 / 0.43 |
| dequant_triton+hipblaslt w8 zp | 119.6 / 2.2 / 1.12 | 122.3 / 8.6 / 1.10 | 122.9 / 34.1 / 1.10 | 122.4 / 68.6 / 1.11 | 131.2 / 127.9 / 1.05 | 140.9 / 238.2 / 1.01 | 171.1 / 392.2 / 0.88 | 299.0 / 897.7 / 0.66 | 899.3 / 1194.0 / 0.43 |
| aiter_a16w8_int8 w8 sym | 36.5 / 7.2 / 3.71 | 37.1 / 28.3 / 3.65 | 53.3 / 78.7 / 2.56 | 55.7 / 150.7 / 2.46 | 66.7 / 251.5 / 2.08 | 116.6 / 287.7 / 1.23 | 216.6 / 309.8 / 0.70 | 682.6 / 393.2 / 0.29 | 1680.0 / 639.1 / 0.23 |
| aiter_a16w8_fp8_block128 w8 sym | 30.4 / 8.6 / 4.31 | 31.2 / 33.6 / 4.20 | 33.8 / 124.2 / 3.91 | 53.1 / 158.0 / 2.51 | 62.7 / 267.4 / 2.15 | 95.7 / 350.8 / 1.45 | 216.2 / 310.4 / 0.68 | 572.4 / 469.0 / 0.34 | 1412.5 / 760.2 / 0.27 |
| aiter_a16wfp4 w4 sym | 93.8 / 2.8 / 0.74 | 93.6 / 11.2 / 0.75 | 91.6 / 45.8 / 0.77 | 110.5 / 75.9 / 0.65 | 221.4 / 75.8 / 0.33 | 324.2 / 103.5 / 0.24 | 553.0 / 121.4 / 0.15 | 2052.8 / 130.8 / 0.07 | 8188.0 / 131.1 / 0.04 |
| aiter_afp4wfp4 w4 sym | 38.4 / 6.8 / 1.81 | 41.0 / 25.6 / 1.70 | 28.6 / 146.8 / 2.45 | 31.6 / 265.5 / 2.23 | 41.8 / 401.6 / 1.70 | 55.4 / 605.3 / 1.31 | 90.4 / 742.5 / 0.84 | 274.3 / 978.7 / 0.34 | 539.6 / 1990.0 / 0.31 |
| torch_scaled_mm_fp8 w8 sym | 27.9 / 9.4 / 4.69 | 28.6 / 36.7 / 4.59 | 30.1 / 139.3 / 4.37 | 32.9 / 255.1 / 4.02 | 32.0 / 524.6 / 4.17 | 39.2 / 854.9 / 3.46 | 67.0 / 1000.9 / 2.09 | 131.5 / 2040.9 / 1.28 | 400.0 / 2684.4 / 0.69 |
| aiter_a8w8_int8 w8 sym | 43.8 / 6.0 / 2.99 | 43.6 / 24.0 / 3.01 | 52.1 / 80.4 / 2.52 | 44.8 / 187.2 / 2.95 | 57.4 / 292.1 / 2.32 | 93.0 / 360.6 / 1.46 | 139.6 / 480.7 / 1.00 | 247.5 / 1084.6 / 0.68 | 682.7 / 1572.7 / 0.41 |
| aiter_gemm_a4w4_asm w4 sym | 36.3 / 7.2 / 1.92 | 36.8 / 28.5 / 1.89 | 39.1 / 107.3 / 1.79 | 42.2 / 198.9 / 1.67 | 40.8 / 410.9 / 1.74 | 40.0 / 839.0 / 1.82 | 47.1 / 1423.9 / 1.61 | 82.9 / 3239.0 / 1.14 | 236.2 / 4545.2 / 0.71 |
