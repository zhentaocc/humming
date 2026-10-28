# ROCm platform port log (branch `rocm-platform`)

Scope: device detection, HIP JIT compiler, runtime/launch, device-header portability
and the non-MMA kernels (pack/unpack/quant/dequant/repack weight, process_mxfp4,
process_input) on MI355X (gfx950). The MFMA GEMM is out of scope.

Environment: ROCm 7.1 at `/opt/rocm`, torch 2.11.0+rocm7.2 in
`/home/zhenchen/projects/humming-rocm/.venv-rocm`, GPU 3 only (`HIP_VISIBLE_DEVICES=3`).

## Step 1: device detection

Changes:
- `humming/utils/rocm.py` (new): `is_rocm_build()` (from `torch.version.hip`), ROCm
  path/version/hipcc discovery, `normalize_gcn_arch`.
- `humming/csrc/device_info.cpp`: `#ifdef HUMMING_HIP` branch using the HIP runtime
  (`hipGetDeviceProperties` + `hipDeviceGetAttribute`). New attributes on both
  platforms: `is_rocm`, `arch` (`sm_XY` on CUDA, `gfx950` on ROCm), `warp_size`.
  The CUDA repr is unchanged; ROCm appends the three new fields.
- `humming/utils/device.py`: ROCm build (`g++ -DHUMMING_HIP=1 -D__HIP_PLATFORM_AMD__
  -I/opt/rocm/include -lamdhip64`), precompiled CUDA artifact ignored on ROCm, ROCm
  version folded into the build hash.
- `humming/device.py`: `is_rocm`, `arch`, `warp_size` properties.

Decisions:
- `sm_major = sm_minor = sm_version = 0` on ROCm. HIP reports gfx950 as compute
  capability 9.5, which every `sm_version >= 90` gate would read as Hopper
  (WGMMA/TMA/PDL/cluster). 0 makes all such gates take the oldest path; code that
  needs the AMD arch keys on `current_device.is_rocm` / `current_device.arch`.
- Memory bandwidth on CDNA uses 4 transfers per reported memory clock
  (2000 MHz x 8192 bit x 4 / 8 = 8.19 TB/s on MI355X, matches spec).
- `tensorcore_tops` on ROCm: dense MFMA rate table (gfx950: 4096 FP16 ops/CU/clk,
  FP8/INT8 x2, FP6/FP4 x4, no INT4/E3M4/E0M3).

Result (`HIP_VISIBLE_DEVICES=3 python -c "from humming.device import current_device; current_device.print()"`):
`name='AMD Instinct MI355X', sm_count=256, sm_version=0, l2=4 MB, l1=32 KB,
default_smem_size=max_smem_size=163840, sm_clock_khz=2400000,
memory_bandwidth_gbps=8192.0, float16 2516 TOPS, is_rocm=True, arch='gfx950', warp_size=64`.
