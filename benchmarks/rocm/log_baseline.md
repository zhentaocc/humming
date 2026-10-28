# ROCm baseline GEMM benchmark: log

The goal is a fair baseline of the existing ROCm low-bit GEMMs (W4A16 and W8A16, plus FP8/MXFP4 references) on MI355X (gfx950). The Humming ROCm kernel can then be registered into the same harness and compared on identical shapes and with the same timing method.

Results from the full sweep, run on 2026-10-28 on GPU 5:

- `results/baselines_2026-10-28.csv` has 3690 ok rows.
- `results/baselines_2026-10-28.md` has the environment, the status table, the headline tables and full per-shape tables.

## Layout

| file | purpose |
|---|---|
| `shapes.py` | Single source of truth for the M list, model (N, K) shapes, headline subset and correctness-check shape. |
| `common.py` | `Problem` (canonical quantized weight `w = (q - z) * s`), packing helpers (GPTQ, AWQ and N-packed layouts), the `REGISTRY` and `@register`, and the `Unsupported` exception. |
| `baselines.py` | All registered implementations. |
| `bench_baselines.py` | CLI: correctness gate, timing, HBM/TFLOPs probe, CSV and markdown output. |
| `third_party/vllm_awq_triton.py`, `third_party/vllm_triton_w4a16.py` | Verbatim pure-Triton kernels copied from vLLM. Only the vLLM imports are replaced. |
| `ext/vllm_gptq/src/` | vLLM `csrc/libtorch_stable/quantization/gptq` copied unchanged, plus a new `bindings.cu`. It is built standalone with `torch.utils.cpp_extension.load` (hipified, gfx950). |

Reproduce with:

```bash
export HIP_VISIBLE_DEVICES=5
PY=/home/zhenchen/projects/humming-rocm/.venv-rocm/bin/python
$PY benchmarks/rocm/bench_baselines.py --out-dir benchmarks/rocm/results 2>&1 | grep -v "^Warning: Latency"
# subset:
$PY benchmarks/rocm/bench_baselines.py --impls torch_dense aiter_a16w8_int8 --dtypes fp16 \
    --bits 8 --zp sym --shapes llama3-70b/down --m 1 16 1024
```

The full sweep takes about 25 min. It covers 2 dtypes, bits 4 and 8, sym and zp, 10 shapes and 9 M values. The `Warning: Latency not found` lines come from hipBLASLt or `_scaled_mm` stderr and are harmless.

## Environment findings

- torch 2.11.0+rocm7.2, Triton 3.6.0 (not 3.7), Python 3.10, HIP 7.2. hipcc comes from /opt/rocm-7.1.0.
- **AITER** (`/home/zhenchen/projects/aiter` @ `cdf01e2a4f`, installed editable):
  - Problem: the JIT `.so` files in the tree were built for Python 3.12 with a newer libstdc++. Importing fails with `GLIBCXX_3.4.31 not found`. An `LD_PRELOAD` of a newer libstdc++ only moves the failure on, to an undefined `PyObject_Vectorcall`.
  - Fix: `bench_baselines.py` sets `AITER_JIT_DIR=benchmarks/rocm/.aiter_jit` (gitignored) before importing aiter. `module_aiter_core` then rebuilds in about 10 s. The asm `gemm_a4w4` path also JIT-built fine.
  - The aiter tree is not modified.
- **vLLM** (`/home/zhenchen/projects/vllm` @ `5efbf0886a`):
  - Problem: the package is not importable in this venv. Its `_C.abi3.so` was built for py3.12 and a different torch.
  - Consequence: every op that lives in `_C` is unavailable, including `awq_gemm` HIP, `gptq_marlin`, `machete`, the cutlass scaled_mm and `rocm_skinny` (wvSplitK).
  - Workaround: the pure-Triton kernels are copied into `third_party/` (unchanged code, with a header that cites the source), and the exllama `q_gemm.cu` is built as a standalone extension.
- **conch-triton-kernels 1.3**: this is the backend of vLLM's `ConchLinearKernel`. It was installed into the venv with `pip install --no-deps`.

## What runs

| impl | dtypes | bits | zp | source |
|---|---|---|---|---|
| `torch_dense` | fp16, bf16 | 16 | n/a | F.linear (hipBLASLt). This is the roofline-ish reference. |
| `vllm_triton_w4a16` | fp16, bf16 | 4 | sym, zp | vLLM `TritonW4A16LinearKernel` (MI300 tile table) |
| `vllm_awq_triton_gemm` | fp16 | 4 | zp | vLLM `awq_gemm_triton`, split_k=8. vLLM uses it for M < 256. |
| `vllm_awq_dequant_mm` | fp16, bf16 | 4 | zp | vLLM `awq_dequantize_triton` + matmul. vLLM uses it for M >= 256. |
| `conch_mixed_precision` | fp16, bf16 | 4, 8 | sym, zp | conch `mixed_precision_gemm` |
| `vllm_exllama_gptq` | fp16 | 4, 8 | sym, zp | vLLM exllama `gptq_gemm` (HIP), GPTQ v1 zeros |
| `dequant_triton+hipblaslt` | fp16, bf16 | 4, 8 | sym, zp | Our own Triton dequant (one group per program) followed by F.linear, timed together. This is the "naive" fallback. |
| `aiter_a16w8_int8` | fp16, bf16 | 8 | sym | AITER Triton `gemm_a16w8_blockscale`, int8 W with 1 x G scales |
| `aiter_a16w8_fp8_block128` | fp16, bf16 | 8 (fp8) | sym | Same kernel, FP8 e4m3 W with 128x128 block scales (own quant) |
| `aiter_a16wfp4` | fp16, bf16 | 4 (MXFP4) | sym | AITER Triton `gemm_a16wfp4`. X is MXFP4-quantized inside the kernel, so this is not true A16. |
| `aiter_afp4wfp4` | fp16, bf16 | 4 (MXFP4) | sym | AITER Triton `gemm_afp4wfp4`. Kind w4a4; X quantization is not timed. |
| `aiter_gemm_a4w4_asm` | bf16 | 4 (MXFP4) | sym | AITER gfx950 asm `gemm_a4w4` with preshuffled W. Kind w4a4; X quantization is not timed. |
| `aiter_a8w8_int8` | fp16, bf16 | 8 | sym | AITER Triton `gemm_a8w8`: per-token int8 X, per-channel W. Kind w8a8. |
| `torch_scaled_mm_fp8` | bf16 | 8 (fp8) | sym | `torch._scaled_mm`: rowwise fp8 X, per-channel W. Kind w8a8. |

Every listed config passes the fp32 correctness gate. The relative Frobenius error is about 2e-4 for fp16 and about 2e-3 for bf16, against a tolerance of 2e-2 (3e-2 for MXFP4).

The MXFP4 and FP8 implementations use their own weight quantization, so their reference weight is that format dequantized. They are format references, not drop-in replacements for INT4/INT8 GPTQ/AWQ.

## What does not run, and why

**vLLM `_C` HIP ops (AWQ HIP `awq_gemm`, `gptq_marlin`, `rocm_skinny` wvSplitK and others).** vLLM's `_C` does not load in this environment (py3.12 build). Rebuilding all of vLLM was out of scope. Only `q_gemm.cu` was extracted, because it is self-contained.

**Marlin, Machete, CUTLASS W4A8, AllSpark, Bitblas, Exllama v2-format.** These are CUDA-only (or need `_C`). There is no ROCm path.

**vLLM `rdna3` / `rdna_hybrid` mixed-precision kernels.** These are gfx11/gfx12 (WMMA) only and are gated off on gfx950.

**`vllm_awq_triton_gemm` with bf16.** Triton 3.6 fails to compile `tl.dot(..., out_dtype=bfloat16)` (CompilationError). The kernel's accumulator type follows the input dtype, so bf16 is broken on this Triton. It is reported as an error.

**`vllm_exllama_gptq` with bf16.** The kernel is fp16-only by design.

**`torch_scaled_mm_fp8` with fp16.** hipBLASLt rowwise `_scaled_mm` only supports BFloat16 output (ValueError). A bf16 output followed by a cast was not added, because that would not be the same kernel.

**`aiter_gemm_a4w4_asm` with fp16.** The asm kernels emit bf16 only.

**AWQ with sym (no zero point).** The AWQ format always carries zeros.

**AITER int8/fp8/MXFP4 kernels with zp.** These kernels have no zero-point input. AITER has no W4A16 INT4-with-zp GEMM (the int4 paths are MoE-only, e.g. `fused_moe` w4a16), so INT4 dense W4A16 on AITER is not covered.

## Methodology

- **Correctness gate.** Each (impl, dtype, bits, zp) is checked first, on (N, K) = (512, 1024) with M in {1, 33, 300}, so that M-dependent dispatch is exercised. The check is the relative Frobenius error against `x_ref @ w_ref^T` computed in fp32. A failing or erroring config is skipped and reported in the status table.
- **Timing.**
  - HIP events, warmup 20, at least 100 timed calls.
  - L calls (cycling through the rotating copies) are captured into one HIP graph, and the graph is replayed. This removes Python and launch overhead, which matters at M = 1 (around 10 us kernels).
  - If capture fails, timing falls back to eager and the `timing` column says so. No implementation needed the fallback in this sweep.
- **L2/MALL defeat.**
  - MI355X has 4 MB L2 per XCD and a 256 MB Infinity Cache (MALL). Many W4 weights (8-30 MB) would otherwise be MALL-resident across iterations.
  - Each config builds `n_copies` fully independent states (X, packed W, scales, zeros, workspace, all via `setup()`). The cyclic footprint is at least 1 GiB (4 x MALL), capped at 256 copies.
  - Consecutive calls use different copies, so weights are always read cold from HBM.
  - Measured effect for dense fp16 M=1, 1 copy vs rotating:
    - 4096x4096: 10.1 us vs 12.5 us (+24%).
    - 4096x14336: 23.0 us vs 25.1 us.
- **Metrics.**
  - TFLOPs = 2MNK / t.
  - Effective TB/s = (quantized weight bytes incl. scales/zeros + activation bytes + output bytes) / t.
  - For the dequant+GEMM fallbacks, the intermediate fp16 weight write and read is **not** counted. TB/s is therefore "useful bytes" and comparable across implementations.
- **Peaks.**
  - Vendor peaks: HBM 8 TB/s, FP16 2.5 PF, FP8 5.0 PF, MXFP4 10.1 PF dense.
  - Measured probes:

    | probe | result |
    |---|---|
    | 4 GiB copy | 5.07 TB/s (read+write) |
    | 4 GiB `torch.sum` | 2.79 TB/s |
    | fp16 8192^3 F.linear | 1472 TF |
    | bf16 8192^3 F.linear | 1593 TF |

  - The sum probe is a poor bandwidth kernel; trust the copy probe instead.
  - The best achieved weight-streaming rate in the sweep is about 6.2 TB/s, from dense GEMV on the large 70B shapes. That is the practical ceiling to quote for memory-bound M.
- **Canonical quantization.** The data is random `q`; scales are about 1/sqrt(K); zeros are uniform in [1, 2^bits) for zp and 2^(bits-1) for sym. Group size is 128.

## Observations (fp16, group 128)

- **No W4A16 INT4 baseline gets anywhere near the HBM roofline at small M.**
  - The best INT4 at M=1 on 70B gate_up (57344x8192) is `vllm_triton_w4a16` at 200 us, about 1.2 TB/s effective. Dense fp16 takes 157 us at 6.0 TB/s.
  - So on ROCm today, W4A16 is slower than plain fp16 for decode on this shape. This is the gap Humming has to close. The ideal is about 4x faster than dense, roughly 40 us at 6 TB/s.
- **At large M (1024/4096), dequant+hipBLASLt-style paths win** (`dequant_triton+hipblaslt`, `vllm_awq_dequant_mm`), at about 1.2-1.4 PF, close to dense.
  - The fused Triton kernels plateau at about 550-600 TF (`vllm_triton_w4a16`).
  - conch runs at about 65 TF.
  - `awq_gemm` (split_k=8) runs at about 40 TF.
- **exllama `gptq_gemm`** is competitive at large M, about 1.2 PF; it switches to its dequant + hipBLAS path for M > 50 (4-bit) or M > 24 (8-bit). Its small-M kernel is slow, especially for large K: 70B down at M=1 takes 1.38 ms.
- **The strongest small-M baselines are not INT4 W4A16:**
  - AITER `gemm_a16w8_blockscale`: int8 4.7-5.0 TB/s, fp8 5.4-6.0 TB/s on the 70B shapes.
  - AITER MXFP4 `afp4wfp4` (w4a4) at about 3.5 TB/s.
  - At large M, `afp4wfp4` reaches about 3.1 PF.

## Adding a new implementation (e.g. Humming)

Add one function to `baselines.py`, or to any module imported by `bench_baselines.py`:

```python
@register("humming_w4a16", kind="wNa16", bits=(4, 8), source="humming@<commit> ...")
def humming(M, N, K, dtype, bits, group_size, zp):
    if <config not supported>:
        raise Unsupported("why")
    def setup(p: Problem):          # called once per rotating copy; allocate NEW tensors
        qw, qz, s = gptq_pack(p)    # or n_packed_pack / pack_last_dim / own repack of p.q, p.scales, p.zeros
        return {"x": p.fresh_x(), "qw": qw, "qz": qz, "s": s}
    def run(state):                 # timed; must be graph-capturable (no host sync)
        return humming_gemm(state["x"], state["qw"], state["s"], state["qz"])
    return setup, run
```

Then run `bench_baselines.py --impls humming_w4a16 torch_dense ...`.

Optional state keys:

- `w_ref` / `x_ref`: an fp32 override of the reference, for implementations with their own quant format.
- `weight_bytes` / `act_bytes`: overrides for the TB/s accounting.
