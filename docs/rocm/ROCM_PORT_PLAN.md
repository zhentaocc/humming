# Humming on ROCm：可行性分析与移植计划

> 目标：让蚂蚁的 [vllm-project/humming](https://github.com/vllm-project/humming)（JIT 量化 GEMM 库）在 AMD Instinct（首发 MI355X / gfx950）上运行，并接入 vLLM ROCm。
> 基线：humming `f970d0b`（2026-10-28），ROCm 7.1 / hipcc，torch 2.11.0+rocm7.2，单卡 MI355X。
> 状态（2026-10-28，分支 `rocm-integrate`）：**gfx950 MFMA weight-only GEMM 已接入公开 API**（`HummingLayer` / `humming_forward`），支持 fp16/bf16 × INT1–8（对称或 uint8 zp）× per-channel/group scale；FP 权重、量化激活、MoE 仍为 NotImplementedError。测试见 `tests/rocm/`，性能见 [RESULTS.md](RESULTS.md)，集成记录见 [log_integrate.md](log_integrate.md)。

## TL;DR

- **可行，但这不是 hipify 能解决的“移植”，而是“新增一个 MFMA 后端”。**
  Humming 的 Python 层（schema / config / transform / tune 框架 / vLLM 接入）和 C++ 模板骨架（scheduler、主循环结构、dequant 位运算）可复用；
  而直接与 NVIDIA 硬件绑定的部分（mma.sync/WGMMA/UMMA 片段布局、ldmatrix、cp.async、TMA、mbarrier、cluster、PDL、cubin patch）需要按 CDNA 重写或删除。
- **最大的结构性障碍是 wave64。** gfx950 只有 64-lane wavefront；Humming 的 s2r loader / MMA / epilogue 全部按 32-lane warp 的 `mma.sync` 片段布局编写（`lane_id = threadIdx.x % 32` 等 32-lane 假设在 `include/` 中约 240 处）。这些层要按 MFMA 布局重新推导，而不是逐行替换。
- **价值点：** ROCm 上现有的 AITER/Triton 已覆盖 FP8、MXFP4、W4A16；Humming 能补上的空缺是**任意位宽 weight-only 量化**（INT1–8 对称/带 zero point、FP6、e5m3 scale 等）、**W4A8/W8A8 混合精度**，以及**同一套 schema 下的 MoE GEMM**。gfx950 的 `v_mfma_scale_f32_*_f8f6f4` 原生支持 FP8/FP6/FP4 混合 + E8M0 block scale，和 Humming 的 MXMMA 路径语义高度吻合。
- **工作量估算（1 人，熟悉 CDNA）：** MVP（FP16/BF16 激活 × INTx/FPx 权重，Dense + MoE，正确性通过）约 **5–7 周**；加上 8-bit 激活与 MX 路径、tuner、vLLM 接入，总计约 **10–14 周**。这是基于代码量的粗估，Phase 1 结束后需重新校准。
- **建议：** 先给上游提 RFC（上游已有 `USE_PPU` 这种非 NVIDIA 后端的先例），然后按下文 Phase 0 → 3 推进，每个 Phase 作为独立 PR。

---

## 1. Humming 的结构

| 层 | 代码 | 规模 | 与 NVIDIA 的耦合 |
|---|---|---|---|
| Python API / schema / layer | `layer.py`, `schema/*`, `ops/*`, `forward.py` | ~5k 行 | 低（`torch.cuda`，ROCm torch 天然兼容） |
| 权重变换 | `transform.py`, `kernel/repack_weight.py` | ~1.5k 行 | **高**：产出的 “humming 格式” 是按 mma.sync B 片段排布的 |
| 配置 / MMA 代码生成 | `config/*` | ~2k 行 | **高**：`MmaOpClassImpl.generate_ptx()` 直接生成 `mma.sync` / `wgmma` / `tcgen05` 内联 PTX |
| Tuner | `tune/*` | ~6k 行 | 中：框架可复用；`sm90*/sm100/sm120` 启发式表不适用 |
| JIT 编译 / 运行时 | `jit/*`, `utils/nvrtc.py`, `utils/cuda.py`, `utils/cubin.py` | ~1k 行 | **高**：NVRTC/NVCC、`cuda.bindings.driver`、cubin 解析与 patch |
| C++ launcher 扩展 | `csrc/launcher/*`, `csrc/device_info.cpp` | ~2.5k 行 | **高**：`cuModuleLoadData`、`cuLaunchKernelEx`、`cuTensorMapEncodeTiled` |
| 设备端头文件 | `include/humming/**` | ~13.8k 行 | 混合，见下 |

设备端头文件细分：

| 子目录 | 内容 | ROCm 处理方式 |
|---|---|---|
| `datatype/` | dequant 位运算（Marlin/lmdeploy 风格的 magic-number int→fp） | **基本可移植**（见 §3 probe），`prmt/lop3/cvt` 换成 AMDGCN builtin |
| `scheduler.cuh` | tile 调度、stream-K、MoE expert 映射 | 可移植；`ld.global.acquire` / `red.relaxed` 换成 `__hip_atomic_*` |
| `arith/` | scale / zero point 应用、epilogue 算术 | 大部分可移植；依赖累加器寄存器布局的部分要改 |
| `memory/g2s_*` | global→smem：cp.async / TMA / mbarrier 流水线 | **重写**：同步 load 或 gfx950 `global_load_lds_dwordx4`，`__syncthreads` 多 stage |
| `memory/s2r_*` | smem→reg：ldmatrix、按 MMA 片段取数 | **重写**：按 MFMA 布局；可用 gfx950 `ds_read_b64_tr_b16`（还有 `ds_read_b64_tr_b4/_b8`、`ds_read_b96_tr_b6`）|
| `mma/` | `wmma.cuh`（mma.sync）、`wgmma.cuh`、`umma.cuh`、`mxmma.cuh` | **新增 `mfma.cuh`**；wgmma/umma 不移植 |
| `epilogue/` | smem 写回（stmatrix）、gmem 写回、split-K 规约 | **部分重写**：累加器布局不同 |
| `kernel/humming.cuh` | 主循环骨架（非 warp-specialized） | **可复用**：骨架对 Ctx/MMA 是模板化的，`barrier.cluster`、PDL 用 `if constexpr` 关掉 |
| `kernel/humming_ws.cuh`, `humming_umma.cuh` | SM90 warp-specialized / SM100 tcgen05 | 不移植 |
| `kernel/process_input/*` | 激活动态量化、hadamard | 可移植；`cvt.*.e4m3x2/e2m1x2` 换成 gfx950 `cvt_pk_fp8` / `cvt_scalef32_pk_fp4` builtin |

内联 PTX 统计（`include/` 内）：`cp.async.bulk.tensor`（TMA）×20+、`tcgen05.*` ×17、`stmatrix` ×6、`ldmatrix` ×8、`mbarrier.*` ×3、`wgmma.*` ×3、`barrier.cluster` ×6、`griddepcontrol` ×2、`setmaxnreg` ×2、`cvt.*` ×36、`prmt/lop3` ×7，外加 `config/mma.py` 运行时生成的 `mma.sync`。

### 已有的非 NVIDIA 先例：`USE_PPU`

上游已支持 `is_ppu`（一款 CUDA 兼容但非 NVIDIA 的加速器）：`USE_PPU` / `is_ppu` 分支分布在约 25 个文件（Python + C++ + 头文件），`transform.py` 里按 PPU MMA 的 B 操作数通道顺序（`[0,2,4,6,1,3,5,7]`）对权重做行置换，编译器强制走 NVCC。说明**维护者接受“设备 flag + 布局置换 + 宏分支”这种后端扩展方式**，ROCm 后端可以沿用这个模式（`is_rocm` / `USE_HIP`），只是改动面比 PPU 大得多（PPU 依然是 32-lane、依然吃 PTX）。

---

## 2. CUDA → gfx950 映射

| Humming 依赖 | gfx950 对应 | 备注 |
|---|---|---|
| warp = 32 | wavefront = 64 | **核心差异**：引入 `kWarpSize`；`__shfl_sync(0xffffffff, …)` 的 mask 必须是 64-bit（probe 已触发该 static_assert） |
| `mma.sync.m16n8k16` (f16/bf16) | `v_mfma_f32_16x16x32_{f16,bf16}`, `v_mfma_f32_32x32x16_{f16,bf16}` | 片段布局完全不同 → s2r / dequant 输出顺序 / epilogue 重推 |
| `mma.sync` e4m3/e5m2 | `v_mfma_f32_16x16x128_f8f6f4`（OCP fp8） | gfx950 是 OCP e4m3；**gfx942 (MI300X) 是 FNUZ**，编码不同，首发不覆盖 |
| `mma.sync` s8 | `v_mfma_i32_16x16x64_i8` | |
| `mma.sync` s4 (INT4 激活) | **无** | INT4 激活只能退化为 INT8 MFMA 或不支持 |
| MXMMA `kind::mxf8f6f4` + ue8m0 | `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` + E8M0 | A/B 可独立选 fp8/fp6/fp4，**语义最接近**；NVFP4 的 ue4m3 scale 无硬件支持 → 走软件 scale |
| WGMMA / UMMA(tcgen05) / TMEM | 无 | 不移植；对应 tuner 分支在 ROCm 上禁用 |
| `cp.async` (g2s) | 同步 `global_load` → `ds_write`，或 gfx950 `global_load_lds_dwordx4`（`__builtin_amdgcn_global_load_lds`）| 后者是 gfx950 新增的 128-bit 直写 LDS，最接近 cp.async |
| TMA / tensor map / multicast / cluster | 无（buffer load + 软件地址计算） | `kUseTma*` 全部为 false |
| `mbarrier` 流水线 | `__syncthreads` / `s_barrier`，或 LDS 原子做简易计数 | 非 warp-specialized 的 `humming.cuh` 本来就主要用 `__syncthreads`，影响可控 |
| `ldmatrix` / `ldmatrix.m8n16.s4` | `ds_read_b128`；`ds_read_b64_tr_b{4,8,16}` / `ds_read_b96_tr_b6`（gfx950） | 转置读对 A/B 片段都有用 |
| `stmatrix` | `ds_write_b64/b128` | |
| `prmt.b32` / `__byte_perm` | `v_perm_b32`（`__builtin_amdgcn_perm`） | 选择子编码不同，需要一张映射表 |
| `lop3.b32` | 普通位运算（编译器会生成 `v_bfi_b32`/`v_and_or_b32`） | |
| `cvt.rn.f16x2.e4m3x2` 等 | `__builtin_amdgcn_cvt_pk_f32_fp8` / `cvt_scalef32_pk_*` | |
| `__dp4a` | `__builtin_amdgcn_sdot4` | |
| PDL (`griddepcontrol`) | 无 | `kUsePdl = false` |
| `setmaxnreg` | 无（VGPR+AGPR 共 512/lane，静态分配） | |
| 动态 smem 上限 | LDS 160 KB/CU | 和 H100（227 KB）不同，tuner 需要重新约束 |
| NVRTC / NVCC JIT | hiprtc（`/opt/rocm/lib/libhiprtc.so`）或 hipcc → `.hsaco` | 都可用；hiprtc 支持 name expression |
| `cuda.bindings.driver`（cuModuleLoad / cuLaunchKernelEx） | `hipModuleLoadData` / `hipModuleLaunchKernel`（C++ launcher 里用；Python 侧可换 `hip-python` 或统一走 C++ launcher） | |
| cubin 解析 / patch（undocumented e3m4/e0m3） | 不适用 | 该特性在 ROCm 上关闭 |

---

## 3. 实测 probe：datatype/dequant 层

`docs/rocm/probe/run_probe.sh`：用 hipcc 为 gfx950 编译 `dequant<UInt4, Float16>`，`shim/` 只做两件事：(1) `cuda*.h` → `hip*.h` 别名；(2) 把 TMA / mbarrier / WGMMA / tcgen05 / cp.async 头文件换成空 stub。

结果：**32 个错误，约 10 个根因**，全部是机械性的：

```
11  dequant_native.cuh   asm 约束 'h'（PTX cvt.*x2）       → AMDGCN cvt builtin
 13  base_conversion/dequant  nv_bfloat16/nv_bfloat162 类型名、__float2bfloat162_rn 等  → 类型/函数别名
  3  ptx/shared.cuh       __cvta_generic_to_shared         → LDS 地址空间 cast
  1  warp shuffle         32-bit mask                      → wave64 64-bit mask
  1  dequant_fused.cuh    __dp4a                           → __builtin_amdgcn_sdot4
  ...
```

结论：**dequant 位运算核心（Humming 的主要 IP）可以几乎原样复用**。真正的工作量在 MFMA 片段布局、内存流水线和 epilogue。

---

## 4. 分阶段计划

每个 Phase 都能独立测试，作为单独 PR 提交。

### Phase 0 — 平台层与非 GEMM kernel（约 1–1.5 周）

- `device.py` / `csrc/device_info.cpp`：`is_rocm`、`arch = "gfx950"`、CU 数、LDS 上限、`warp_size = 64`；`sm_version` 对 ROCm 返回哨兵值，并审计所有 `sm_version // 10 == 9` 一类的判断。
- `jit/compiler.py`：新增 `HIPRTCCompiler`（以及调试用的 `HIPCCCompiler`），产物 `.hsaco`，cache key 包含 ROCm 版本与 arch。
- `jit/runtime.py` + `csrc/launcher`：module 加载与 launch 走 HIP；Python 侧 `cbd.cuLaunchKernelEx` 的 7 处调用收敛到一个 launcher 抽象。
- `include/humming/utils/base.cuh`：新增 `platform.cuh`（类型别名、`kWarpSize`、shuffle 包装、LDS 指针 cast）。
- 先跑通不涉及 MMA 的 kernel：`pack/unpack_weight`、`quant_weight`、`dequant_weight`、`process_input`（激活量化/hadamard）。
- **验收**：`tests/` 中这些 op 的单测在 MI355X 上通过。

### Phase 1 — MFMA GEMM MVP：FP16/BF16 激活（约 4–5 周）

- `config/mma.py`：新增 `MmaType.MFMA` 和 `MfmaOpClassImpl`，生成 `__builtin_amdgcn_mfma_f32_16x16x32_{f16,bf16}`（先只做 16×16；32×32 留到调优阶段）。
- `mma/mfma.cuh`：对应 `wmma.cuh` 的 `transform_b / run / final_regs_c_as_ptr`。
- **布局设计（核心工作）**：确定每个 lane 持有的 B 片段，然后
  - `transform.py` / `repack_weight` 新增 ROCm interleave 模式，使每个 lane 加载的 packed 32-bit 字经 `dequant()` 后**直接落在 MFMA B 操作数布局上**（Marlin 思路：离线重排权重，在线零 shuffle）；
  - `s2r_loader_a`：A 片段用 `ds_read_b128` 或 `ds_read_b64_tr_b16`；
  - `epilogue/smem_writer`：MFMA 16×16 累加器布局（每个 lane 持有一列中连续 4 行）。
- `g2s`：先做同步 load + `__syncthreads` 双/多 stage，再换成 `global_load_lds_dwordx4`。
- Stream-K 锁与 MoE 调度：替换原子操作，其余复用。
- 覆盖范围：INT1–8（对称 / 动态 zero point）、任意 FPx（kBits ≤ 8）权重；group / channel / tensor scale；Dense + MoE。
- **验收**：`tests/` 中 FP16/BF16 激活的 GEMM / MoE 正确性矩阵通过；在 decode 典型 shape（M ≤ 64）下与 rocBLAS FP16 GEMM、vLLM ROCm 现有 W4A16（Triton）kernel 对比带宽利用率。

### Phase 2 — 8-bit 激活与 MX 路径（约 3–4 周）

- FP8（OCP e4m3/e5m2）× INTx/FPx 权重：`v_mfma_f32_16x16x128_f8f6f4`，权重 dequant 到 fp8（复用 `get_mma_b_dtype` 的选择逻辑）。
- INT8 激活：`v_mfma_i32_16x16x64_i8`。
- MXFP4 / MXFP6 / MXFP8：`v_mfma_scale_f32_16x16x128_f8f6f4` + E8M0，原生混合 A/B 格式（对应 Humming 的 `kNativeMixed` / MXMMA 路径）。NVFP4（ue4m3 scale）走软件 scale。
- `process_input` 的激活量化输出布局与上述 MMA 对齐。

### Phase 3 — Tuner、vLLM 接入、打包（约 2–3 周）

- `tune/gfx950.py`：先基于 measured autotune（`testing/tuning.py` 已有框架）收集配置，再沉淀成启发式；约束为 LDS 160 KB、VGPR/AGPR 预算、occupancy。
- vLLM：`vllm/model_executor/layers/quantization/humming.py` 及 MoE oracle 在 `current_platform.is_rocm()` 时放行 humming 后端；验证 GPTQ / AWQ / compressed-tensors / Quark checkpoint 端到端（上游已支持 Quark schema，对 AMD 用户是加分项）。
- 打包：`humming-kernels[rocm]` extra，ROCm 下不依赖 `cuda-bindings`。
- CI：至少一条 MI355X 正确性 job（具体放在哪套 CI 需和上游商量）。

### 不移植（ROCm 上显式关闭）

WGMMA、UMMA/tcgen05/TMEM、TMA 与 multicast/cluster、PDL、`setmaxnreg`、warp-specialized kernel（`humming_ws.cuh`）、cubin patch 及 undocumented e3m4/e0m3、INT4 激活、PPU 路径。

---

## 5. 风险

1. **wave64 片段布局重推**是工作量的大头，也是最容易出错的地方。缓解：先写 Python 参考实现，生成并校验 “lane → (row, k)” 映射，再写 kernel；每一层都做单独单测（只测 dequant 输出布局、只测 s2r、只测 epilogue）。
2. **dequant 的 VALU 开销**：MI355X 的 MFMA 吞吐相对 VALU 更高，in-register dequant 在大 M（compute-bound）时可能成为瓶颈。Humming 的主场景是 decode / 小 M（memory-bound），影响有限；大 M 需要评估“每个 block dequant 一次到 LDS”的变体。
3. **权重格式与设备绑定**：ROCm 上 `transform()` 后的张量与 CUDA 版本不同。vLLM 在 load 时做变换，不受影响；但任何离线序列化的 humming 格式权重不能跨平台复用，需要文档说明，并在格式里打平台标记。
4. **FP8 编码**：首发只做 gfx950（OCP）。gfx942（MI300X/MI325X）的 FNUZ fp8 需要额外的 dtype 分支，单独排期。
5. **hiprtc 编译时间**：重模板代码在 hiprtc 上可能比 NVRTC 慢；Humming 已有磁盘 cache 和并行预编译（`get_parallel_build_workers`），需要实测首次编译耗时。
6. **上游接受度**：改动面大，涉及 Python / C++ / 头文件三层。建议先发 RFC issue（附本文档和 probe），确认维护者（主要是蚂蚁团队）接受 `is_rocm` 分支方式后再提代码；否则维护成独立 fork / 插件后端。

---

## 6. 备选方案（不推荐作为主线）

- **Triton / Gluon / FlyDSL 重写 kernel，只复用 Humming 的 Python 层**：开发更快、天然支持 wave64，但会丢掉 Humming C++ 模板生成的灵活性（任意位宽 dequant 的 magic-number 路径需要在 DSL 里重写），也与上游形成两套 kernel。可以作为 Phase 1 之前的**性能基线 / 正确性参考**。
- **只在 vLLM 侧把 humming 量化路由到 AITER/Triton**：只能覆盖 FP8 / MXFP4 / W4A16 这些已有 kernel 的格式，补不上 Humming 真正独有的任意位宽支持。

## 7. 下一步

1. 向 vllm-project/humming 提 RFC issue（需要你确认后再发）。
2. 启动 Phase 0；同时在 Python 里写 MFMA 16×16 B 片段的 lane 映射参考实现，作为 Phase 1 的布局设计输入。
