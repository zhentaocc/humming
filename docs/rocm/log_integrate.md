# ROCm integration log (branch `rocm-integrate`)

Goal: Humming's public API (`HummingLayer`, `humming_forward`, `humming.ops.humming_gemm`) runs the
gfx950 MFMA GEMM end to end on MI355X, benchmarked against the existing ROCm baselines.
Results: [RESULTS.md](RESULTS.md).

## Step 1: merge `rocm-mfma` (3c1ae72)

The HIP compatibility overlap was resolved in favour of `utils/platform.cuh`, which is now the single source of HIP compat:
- `base.cuh`, `base_conversion.cuh`, `ptx/math.cuh` and `all.cuh` keep the platform versions.
- `utils/rocm/compat.cuh` was dropped. All of its aliases, `__dp4a` and `CUtensorMap` already live in `platform.cuh`.
- `utils/hip/math.cuh`: `lop3` lowers to the gfx950 `v_bitop3_b32`, which the MFMA dequant loop needs. `lop3_and_or` and `lop3_and_xor` go through it.
- `mfma.cuh` and `dequant_native.cuh` key on `HUMMING_HIP`, and `kWaveSize = kWarpSize`.

Both branches' `tests/rocm` suites merged: 583 passed, 40 skipped.

## Step 2: wiring (ad081fc, e9c12e4, e4f5da9)

Call path on ROCm:

```
HummingLayer.transform()
  prepare_layer_config      pads N to a multiple of 64 and K to a multiple of 128 (lcm with the user's padding)
  transform_humming_tensors -> transform_mfma_tensors
      check_mfma_layer_config      raises NotImplementedError for unsupported configs
      weight: layout.repack_weight_for_mfma, chunked over N to bound int64 temporaries
      scale:  [groups, N] in the activation dtype
      zp:     unpacked from Humming's N-packed int32 to uint8 [groups, N]
layer(x) -> humming_forward -> humming_gemm -> _humming_gemm_rocm
  pads K on the input; runs into a padded N output and slices it; adds the bias outside the kernel
  HummingKernel.prepare_kernels -> prepare_mfma_kernels
      Gfx950Heuristics.get_configs   one MfmaTile per M interval (0,8], (8,32], (32,64], (64,inf)
      HummingMfmaKernel (KernelRuntime) x4 via compile_many: HIPCCCompiler, -O3, no fast-math
      ops.register_kernel: the C++ launcher sees the MFMA_* __constant__ symbols -> MfmaKernelData
  torch.ops.humming.launch_kernel -> launch_mfma_kernel (csrc/launcher/mfma.h)
      split-K chosen at launch time from the runtime M and the CU count
      hipModuleLaunchKernel on torch's current stream
```

Design notes:
- The torch op schema is unchanged. The launcher recognises an MFMA code object by the `MFMA_BLOCK_M` symbol and branches before the CUDA `KernelData`/TMA plumbing. All of this is under `HUMMING_HIP`.
- Split-K workspace:
  - Outside capture: a grow-only fp32 buffer per (device, stream), at least 4 MiB. Launches on one stream are ordered, so it is safe to reuse.
  - Under stream capture: every launch takes a fresh allocation from the graph's private pool. A later eager grow can therefore never free memory that a graph still uses.
- Tile counters:
  - Normally the layer's `locks` buffer (2048 int32, zeroed at construction) is used. The kernel's last block resets each counter, so replays stay valid.
  - The fallback is a per-device 64K-entry hipMalloc buffer. Its first allocation is refused during capture.
- Symmetric integer weights (`{"dtype": "int4"}`) stay offset-binary on ROCm: `convert_symmetric_integer_weight` is a no-op there. The kernel subtracts the implicit 2^(bits-1) itself.
- The CUDA path is behaviourally unchanged:
  - Every Python branch is gated on `current_device.is_rocm`, `DeviceInfo(...).is_rocm` or `is_rocm_build()`.
  - Every C++ change is under `HUMMING_HIP`, except `MmaType::MFMA = 4`, the `"mfma"` enum value and `CubinReader::hasSymbol`, which are additive.

Supported:
- activations: fp16 / bf16;
- weights: uint1..8, either symmetric (implicit zero point) or with uint8 zero points;
- scales: per-channel or group scales, where the group size is a multiple of 32 that divides K;
- optional bias, any N and K (padded), dense GEMM.

Raises NotImplementedError (message starts "Humming on ROCm ..."):
- FP weights;
- input quantization (int8/fp8/fp4 activations);
- MoE;
- secondary or tensor scales;
- int or fused-e8m0 scales;
- FP zero points;
- group sizes such as 16;
- the compute options `gemm_type` other than dense, `use_f16_accum`, `use_batch_invariant` and `use_m_major_input_scale`;
- non-gfx950 AMD GPUs.

Validation before writing tests (probe scripts, GPU 3):
- bits 1–8 × {channel, g32 zp, g64, g128, g128 zp} × fp16/bf16 at M ∈ {1, 7, 64, 255}: worst relative error 2.4e-3, against an fp32 GEMM with the layer's own dequantized weight.
- Padding + bias at (1056, 1056), (4128, 3008) and (96, 224): fp16 ~3e-4, bf16 ~2.8e-3.
- Graph capture at M ∈ {1, 16, 200}: OK.
- README int6 example: relative error vs the *unquantized* fp16 weight is 0.036, which is int6 quantization error.

Problem: `tests/rocm/test_layer.py` asserted the old "MFMA GEMM backend not yet integrated" error.
Fix: it now checks that int4/int8 layers run and match the reference, and that FP8/MXFP4/int8-activation/g16 layers raise NotImplementedError.

Problem: in eager mode, `layer(x)` costs ~30 us of host time against ~15 us for a raw `mfma_gemm` call.
Investigation: cProfile showed a `json.loads` of the layer config on every call.
Fix (e4f5da9): `get_mfma_gemm_meta` is now cached. The remaining overhead is the generic `humming_forward` Python layering, which the CUDA path shares.

## Step 3: tests (355a62d, `tests/rocm/test_layer_e2e.py`)

`test_layer_e2e.py` has 124 tests, all through `HummingLayer`. The reference is an fp32 GEMM with `weight_schema.dequant_tensors` of the quantized tensors, taken before `transform()`. Tolerances are 2e-3 for fp16 and 1e-2 for bf16.

| test | coverage |
|---|---|
| `test_bits_and_scales` | bits 1–8 × {channel, g32+zp, g64, g128, g128+zp} × fp16/bf16, at M ∈ {1, 7, 64, 255, 1024} |
| `test_real_shapes` | (4096, 4096), (14336, 4096), (4096, 14336), (28672, 8192) × {w4 g128, w4 g128 zp, w8 channel} × fp16/bf16 |
| `test_padding_and_bias` | unaligned N/K plus bias |
| `test_outputs_argument`, `test_readme_int6_example` | the `outputs=` argument; the README int6 example |
| `test_cuda_graph_capture` | M ∈ {1, 16, 200} × fp16/bf16 × warm/cold (cold = the first forward of a fresh layer happens inside the capture), replayed 3x with new inputs |

A cold capture also passes with an empty `HUMMING_CACHE_DIR`. In that case hipcc runs and the module loads inside the capture.

Full suite command:

```bash
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD .venv-rocm/bin/python -m pytest -q tests/rocm -p no:cacheprovider
```

Result: 709 passed, 40 skipped. The skips are the zp × per-channel combinations in `test_mfma_gemm.py`, which test zp only group-wise.

## Step 4: benchmark

Changes to the harness:
- `benchmarks/rocm/baselines.py` registers `humming`, which builds a `HummingLayer` from the canonical `Problem`:
  - packs via `humming.ops.pack_weight`; zp is N-packed as Humming expects;
  - then runs `load_from_tensors` and `transform()`;
  - the rotating copies deep-copy the transformed layer, so it is transformed only once per shape.
- `bench_baselines.py` changes:
  - `--prefix` sets the output name.
  - `--windows N` keeps the best of N timing windows.
  - Bug fix: the correctness gate used `max(bits, 4)`, so 2/3-bit configs were checked as 4-bit. It now uses the real bit width; the existing 4/8-bit baselines are unaffected.

The sweep (GPU 3, best of 3 windows) took about 9 min and produced 3420 rows, with no errors.

Integration overhead vs standalone (GPU 4, same process, same timing, int4 g128):
- graph mode: layer/standalone ratio is 0.97–1.05 across 5 shapes × 6 M × fp16/bf16, i.e. noise.
- The standalone numbers match log_mfma.md, e.g. 28672×8192 fp16 M=1024 is 476 us against the logged ~477.

## Known limitations

- M ≥ 1024 is 0.7–0.9x of hipBLASLt dense and of dequant+hipBLASLt. Causes:
  - the dequant VALU cost;
  - the 16×16 MFMA;
  - a single tile bucket for M > 64.
- bf16 large M is a further ~10–15% behind fp16.
- W8 symmetric decode is ~10% behind AITER FP8 block128. That kernel reads far fewer scale bytes and uses a different format.
- The heuristic was tuned on 5 shapes up to M=1024. There is no autotuning and no stream-K.
- Eager-mode host overhead is ~30 us per call; use graphs.
- Only gfx950 is supported. Other AMD archs raise NotImplementedError.
