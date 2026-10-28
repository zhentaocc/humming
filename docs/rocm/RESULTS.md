# Humming on ROCm（MI355X / gfx950）性能结果

数据：`benchmarks/rocm/results/humming_vs_baselines_2026-10-28.{csv,md}`（3420 行，全部通过正确性检查）。
Humming 走公开 API：`HummingLayer.load_from_tensors → transform() → layer(x)`，即 `humming_forward` 背后 JIT 编译的 gfx950 MFMA GEMM。
对比对象：`torch_dense`（F.linear / hipBLASLt，fp16 权重）、`vllm_triton_w4a16`、`vllm_exllama_gptq`（仅 fp16）、`dequant_triton+hipblaslt`、`aiter_a16w8_int8`、`aiter_a16w8_fp8_block128`。
“最佳现有实现”指同一配置（dtype / bits / zp）下以上量化实现中最快的一个（不含 torch_dense；dense 单独列出）。

> 第 0 节是第二轮（大 M 优化 + FP8/MX/INT8 激活 + MoE，合并到 `rocm` 分支）的结果；第 1–7 节保留第一轮（`4288c98`）的数据作对照。

## 0. 第二轮：大 M 优化、低比特激活、MoE（2026-10-28）

三条线并行开发（分支 `rocm-perf` / `rocm-f8` / `rocm-moe`），合并后 `pytest tests/rocm`：**1670 passed, 87 skipped**。
详细过程见 `log_perf.md`、`log_f8.md`、`log_moe.md`。

### 0.1 Dense W4A16/W8A16：大 M 提速 1.14–1.33x

数据：`benchmarks/rocm/results/humming_perf_2026-10-28.{csv,md}`。10 个 shape 的几何平均，vs 最佳现有量化实现（括号：胜出 shape 数；“自身”= 相对第一轮 Humming 的提速）。

| 配置 | M=1 | M=16 | M=64 | M=256 | M=1024 | M=4096 |
|---|---|---|---|---|---|---|
| fp16 W4 sym | 4.13x (10/10; 自身 1.00x) | 4.20x (10/10; 1.22x) | 2.98x (10/10; 1.11x) | 1.84x (10/10; 1.12x) | 1.19x (9/10; 1.14x) | 0.95x (0/10; 1.23x) |
| fp16 W4 zp | 4.14x (10/10; 0.99x) | 4.26x (10/10; 1.19x) | 3.05x (10/10; 1.11x) | 1.85x (10/10; 1.13x) | 1.18x (9/10; 1.14x) | 0.92x (0/10; 1.21x) |
| fp16 W8 sym | 0.91x (1/10; 1.00x) | 1.07x (6/10; 1.06x) | 1.33x (10/10; 1.05x) | 1.59x (10/10; 1.08x) | 1.07x (7/10; 1.08x) | 0.80x (0/10; 1.09x) |
| fp16 W8 zp | 3.28x (10/10; 0.99x) | 3.60x (10/10; 1.05x) | 2.99x (10/10; 1.04x) | 1.69x (10/10; 1.09x) | 1.07x (6/10; 1.10x) | 0.80x (0/10; 1.10x) |
| bf16 W4 sym | 4.31x (10/10; 1.03x) | 4.19x (10/10; 1.33x) | 3.05x (10/10; 1.18x) | 1.78x (10/10; 1.18x) | 1.12x (8/10; 1.23x) | 0.82x (0/10; 1.23x) |
| bf16 W4 zp | 4.25x (10/10; 1.04x) | 4.20x (10/10; 1.30x) | 3.05x (10/10; 1.17x) | 1.75x (10/10; 1.17x) | 1.10x (8/10; 1.21x) | 0.80x (0/10; 1.21x) |
| bf16 W8 sym | 0.91x (1/10; 1.00x) | 1.06x (6/10; 1.14x) | 1.28x (10/10; 1.12x) | 1.56x (10/10; 1.15x) | 1.02x (6/10; 1.20x) | 0.76x (0/10; 1.21x) |
| bf16 W8 zp | 3.80x (10/10; 1.01x) | 3.58x (10/10; 1.13x) | 3.02x (10/10; 1.10x) | 1.67x (10/10; 1.15x) | 1.02x (6/10; 1.20x) | 0.74x (0/10; 1.20x) |

- 有效的改动：A 走 `global_load_lds` 直接进 LDS；更省的 int dequant（15→13 VALU/fragment）；bf16 用 `v_fma_mix_f32` dequant（25→17）；fp16 ≤4bit 在大 M 用 128×256 tile；M≤64 用 3 级流水；M≤16/≤32 的小 tile。
- 无效的改动：32×32 MFMA 布局、更大的 12×4/16×4 tile、bf16 或 8bit 用 128×256 tile（寄存器溢出）。
- 目标（M=1024 ≥1.1x、M=4096 ≥0.9x hipBLASLt fp16 dense）**大部分未达到**：fp16 W4 vs hipBLASLt 在 M=1024 为 0.85–1.06x，M=4096 为 0.82–0.89x。
  - 原因：M≥1024 已是 compute-bound。去掉 dequant 的同一 kernel 跑 1243 us，hipBLASLt 1283 us（8192×28672，M=4096）；实际 kernel 比它慢 17–19%，差距就是 dequant（每个 MFMA 1.66 条 VALU，无法完全被掩盖）。
  - bf16 在 M=4096 仍比 fp16 慢 14%：dequant 更贵（17 vs 13 VALU），128×256 tile 会溢出，只能用 128×128。

### 0.2 低比特激活：FP8 / MXFP4 / INT8（新增）

数据：`benchmarks/rocm/results/humming_f8_2026-10-28.{csv,md}`（合并后复测 `humming_f8_merged_2026-10-28.*`，FP8/MX 路径不变）。

支持：FP8 e4m3/e5m2 × FP8/FP6/FP4/INT1–5 权重（W8A8、W4A8）；MXFP4 × MXFP4/6/8 和 INT 权重；INT8 × INT1–8。
weight scale：per-tensor/channel/group、128×128 block、MX E8M0；activation scale：静态 per-tensor、动态 per-token、per-group、E8M0。
未支持（抛 `NotImplementedError`）：NVFP4、fp16/bf16 激活 × FP 权重、低比特激活的 MoE。

vs 同格式最佳 ROCm kernel（llama3-70b/down，bf16 输出，只计 GEMM）：

| 格式 | M=1 | M=16 | M=1024 | M=4096 | 对手 |
|---|---|---|---|---|---|
| FP8 per-token × FP8 per-channel | 1.00x | 0.94x | 0.66x | 0.57x | `torch._scaled_mm` / AITER CK/bpreshuffle |
| FP8 g128 × FP8 block-128 | 2.34x | 2.16x | 0.97x | 1.00x | AITER blockscale |
| FP8 × MXFP4（W4A8） | **4.41x** | 1.61x | **2.78x** | 1.97x | AITER a8wfp4 |
| FP8 × INT4 g128（W4A8） | 1.13x | 1.16x | 1.82x | 1.56x | 最快的 4bit 权重 kernel |
| MXFP4 × MXFP4 | 1.28x | 1.09x | 0.31x | 0.26x | AITER afp4wfp4 / a4w4 asm |
| INT8 × INT8 | 1.42x | 1.56x | 1.16x | 0.95x | AITER a8w8 int8 |

- W4A8（INT4 / MXFP4 权重 + FP8 激活）在所有 shape、所有 M 上比任何现有 ROCm 4bit 权重 kernel 快 1.1–4.4x。注：表中 “FP8 × INT4” 对比的 Humming W4A16 是第一轮的数字；合并后 W4A16 bf16 快了 1.1–1.4x，这一行在 M≥1024 的优势会相应缩小。
- 短板：FP8 per-channel 大 M 只有 hipBLASLt 的 ~0.6x（约 1.7–1.8 PFLOPs vs 2.8）；MXFP4×MXFP4 大 M 只有 AITER 手写汇编的 ~0.3x。两者都需要把权重经 LDS 分级、用更大 tile 重构 kernel。
- 动态 per-token 量化端到端多 6–8 us（小 M）。

### 0.3 MoE（新增）

数据：基线 `benchmarks/rocm/results/humming_moe_2026-10-28.{csv,md}`；合并后的 Humming `humming_moe_merged_2026-10-28.{csv,md}`。
单位：一个 MoE block（align + w13 + SiLU·mul + w2 + top-k 加权求和），us。

支持：fp16/bf16 × INT1–8（sym/zp，channel/group），indexed（vLLM 用的）/ grouped_contiguous / grouped_masked 三种布局，w13 + w2，per-expert bias，split-K，graph capture；HIP 单 kernel `moe_align_block_size`（decode 4–5 us）。

W4 vs 最佳现有 ROCm W4 MoE（bf16 为 AITER a16wi4；fp16 无 AITER int4，为 vLLM Triton wna16 调优版）：

| 模型 | dtype | 1 | 16 | 64 | 1024 | 2048 | 8192 tokens |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | bf16 | 20.7 us, 1.20x | 1.44x | 1.27x | 1.69x | 1.33x | 983 us, 1.35x |
| Mixtral-8x7B | bf16 | 71.5 us, 2.25x | 1.64x | 1.56x | 2.01x | 1.65x | 5889 us, 1.81x |
| DeepSeek-V3 | bf16 | 47.2 us, 1.47x | 1.41x | 1.29x | 1.58x | 1.40x | 7506 us, 1.53x |
| Qwen3-30B-A3B | fp16 | 2.12x | 3.12x | 3.12x | 2.43x | 2.08x | 1.92x |
| Mixtral-8x7B | fp16 | 3.29x | 3.91x | 3.86x | 2.43x | 2.47x | 2.57x |
| DeepSeek-V3 | fp16 | 3.79x | 4.06x | 4.15x | 3.13x | 3.21x | 2.51x |

- W8：比调优过的 Triton int8 快 1.2–2.4x。
- 对 16bit 未量化 MoE：合并大 M 优化后，bf16 8192 tokens 下 Mixtral 0.98x、DeepSeek-V3 1.01x、Qwen3 1.10x（合并前 Mixtral/DeepSeek-V3 落后 3–16%）。
- 合并时的问题：MoE kernel 复用 dense 的 load/mainloop，需跟着改成 direct-to-LDS。改完后 2 级流水的小 tile 每个 chunk 前都要 `vmcnt(0)`，decode 慢 4–9%；MoE 小 tile 改 3 级后比合并前快 0–14%（见 `log_moe.md` 最后一节）。
- **上游也有的 bug**：`quant_weight.cuh` / `pack_weight.cuh` 用 32 位 offset，>2^32 元素的权重（DeepSeek-V3 w13，7.5G 元素）量化错误（expert ≥146 的 rel_err 0.99），所有平台都受影响；已改 64 位。
- vLLM 侧要在 ROCm 上启用：`fused_humming_moe.py` 的 `_supports_current_device` 目前只认 CUDA ≥7.5，需加 gfx950；其余（align、activation、`moe_fused_mul_sum`、tile 约定）不用改。

## 1. 主表：fp16 W4A16 对称 g128

单元格格式：Humming 时间，括号内 M≤16 为有效 TB/s、M≥1024 为 TFLOPs；随后是最佳现有实现的时间与 Humming 加速比；最后是 dense 时间与加速比。

| shape (N×K) | M | Humming | 最佳现有实现 | vs 最佳 | dense | vs dense |
|---|---|---|---|---|---|---|
| llama3-8b/o_proj 4096×4096 | 1 | 7.3 us (1.18 TB/s) | dequant+hipBLASLt 25.2 | **3.43x** | 12.3 | 1.67x |
| | 16 | 10.6 us (0.84 TB/s) | dequant+hipBLASLt 22.9 | **2.17x** | 10.3 | 0.97x |
| | 1024 | 46.3 us (742 TF) | dequant+hipBLASLt 51.2 | 1.11x | 42.5 | 0.92x |
| | 4096 | 150.3 us (915 TF) | dequant+hipBLASLt 108.3 | 0.72x | 96.9 | 0.64x |
| llama3-70b/gate_up 57344×8192 | 1 | 50.4 us (4.80 TB/s) | vllm_triton_w4a16 196.9 | **3.90x** | 156.5 | 3.10x |
| | 16 | 63.1 us (3.87 TB/s) | vllm_triton_w4a16 223.7 | **3.54x** | 156.1 | 2.47x |
| | 1024 | 962.5 us (1000 TF) | dequant+hipBLASLt 999.7 | 1.04x | 744.8 | 0.77x |
| | 4096 | 3771 us (1020 TF) | dequant+hipBLASLt 2882 | 0.76x | 2661 | 0.71x |
| llama3-70b/down 8192×28672 | 1 | 34.5 us (3.51 TB/s) | dequant+hipBLASLt 187.8 | **5.44x** | 75.7 | 2.19x |
| | 16 | 39.8 us (3.07 TB/s) | dequant+hipBLASLt 213.3 | **5.36x** | 77.4 | 1.95x |
| | 1024 | 489.6 us (982 TF) | dequant+hipBLASLt 493.0 | 1.01x | 364.8 | 0.75x |
| | 4096 | 1861 us (1034 TF) | dequant+hipBLASLt 1386 | 0.74x | 1276 | 0.69x |

## 2. 全部 10 个 shape 的几何平均加速比（Humming vs 最佳现有实现）

| 配置 | M=1 | M=16 | M=64 | M=256 | M=1024 | M=4096 |
|---|---|---|---|---|---|---|
| fp16 W4 sym | 4.05x (10/10) | 3.39x | 2.65x | 1.62x | 1.03x (6/10) | 0.77x (0/10) |
| fp16 W4 zp | 4.08x | 3.52x | 2.71x | 1.61x | 1.02x (6/10) | 0.76x |
| fp16 W8 sym | 0.91x (1/10) | 1.00x (4/10) | 1.27x | 1.46x | 0.99x (3/10) | 0.73x |
| fp16 W8 zp | 3.26x | 3.32x | 2.83x | 1.52x | 0.97x (4/10) | 0.72x |
| bf16 W4 sym | 4.10x | 3.12x | 2.56x | 1.48x | 0.91x (3/10) | 0.67x |
| bf16 W4 zp | 4.04x | 3.20x | 2.57x | 1.50x | 0.90x (2/10) | 0.66x |
| bf16 W8 sym | 0.90x (1/10) | 0.92x (3/10) | 1.15x | 1.34x | 0.85x (1/10) | 0.63x |
| bf16 W8 zp | 3.73x | 3.12x | 2.66x | 1.41x | 0.85x (2/10) | 0.62x |

括号内为胜出 shape 数；未标注的均为 10/10。M≤256 时 W4 与 W8-zp 全部胜出。

## 3. W8A16 与 bf16 要点

- **W8 对称**：最强对手是 AITER `gemm_a16w8_blockscale`。
  - 对同格式的 INT8（1×128 scale）：Humming 全面更快，decode 1.07–1.13x，prefill 1.2–1.7x。
  - 对 FP8 128×128 block scale：decode M=1 慢约 10%（0.81–0.93x），M=1024/4096 快 1.03–1.36x。
  - FP8 block scale 每 128×128 只有一个 scale，读的 scale 字节远少于 g128 per-row 格式，且量化格式不同，不完全同口径。
- **W8 zp**：AITER 两种 kernel 都不支持 zero point，最佳现有实现退化为 dequant+hipBLASLt 或 exllama，因此 Humming 在 M≤256 时领先 3x 左右。
- **bf16**：decode 与 fp16 基本持平（llama3-70b/down M=1：35.1 vs 34.5 us）。M≥1024 比 fp16 慢 10–15%（每 2 个值多 3 次转换，见 log_mfma.md iteration 9），因此大 M 的劣势更明显（0.85–0.91x）。

## 4. Humming 独有：INT2 / INT3（fp16，g128）

| shape | M | W2 sym | W2 zp | W3 sym | W3 zp | W4 sym | dense |
|---|---|---|---|---|---|---|---|
| llama3-70b/gate_up | 1 | 40.7 | 43.5 | 47.1 | 49.6 | 50.4 | 156.5 |
| | 16 | 56.8 | 59.6 | 62.7 | 63.6 | 63.1 | 156.1 |
| | 1024 | 910 | 938 | 949 | 953 | 963 | 745 |
| llama3-70b/down | 1 | 28.7 | 31.4 | 31.6 | 34.1 | 34.5 | 75.7 |
| | 16 | 37.8 | 40.6 | 40.2 | 40.9 | 39.8 | 77.4 |

单位均为 us。现有 ROCm 库都不支持 INT2/INT3 weight-only GEMM。
INT2 的字节数是 INT4 的一半，但 decode 只快 1.2x 左右（gate_up M=1 约 2.9 TB/s）。原因是这时瓶颈已经从 HBM 转到发射和延迟，详见第 5 节。

## 5. 胜负与原因

- **赢（M≤256）：** 权重直接从 HBM 流入 MFMA，中间不落地 fp16。
  - 原因：dequant+hipBLASLt 要先写再读一份 fp16 权重；vLLM Triton W4A16 / exllama 在 gfx950 上没有针对 wave64 MFMA 的布局。
  - Humming 在 transform 阶段就把权重重排成 MFMA fragment 顺序，加载后无需 shuffle。
  - decode 再配合 split-K（最多 8 路，按约 1 block/CU），大 shape M=1 能到 4.8 TB/s（fp16 W4，llama3-70b/gate_up）。
- **持平（M≈1024）：** 胜负各半。
- **输（M=4096 全部，bf16 M=1024 大多数）：** 大 M 已是 compute-bound，hipBLASLt 的 fp16 GEMM 达到 1.3–1.4 PFLOPs，而 Humming 约 1.0 PFLOPs。原因有三：
  1. dequant 的 VALU 开销：每次 16-cycle MFMA 约 1.9 条 VALU。
  2. 只用了 16×16×16 MFMA，LDS 读 A 的次数偏多。
  3. 启发式在 M>64 时只有一档 tile（block 16×128，tiles_m=8），M=4096 也没有更大的 tile。
  - 这一段 dequant+hipBLASLt 主要受 hipBLASLt 主体决定，所以它反而领先。
- **W8 sym decode 小负于 AITER FP8 block128：** 见第 3 节（scale 字节更少，格式不同）。
- **集成开销：** 在 graph 模式下，`HummingLayer` 路径与独立 `mfma_gemm` 在同一 GPU、同一进程、同一计时方法下的比值为 0.97–1.05x，即在噪声范围内（第 7 节）。

## 6. 方法与噪声说明

- 测量环境：
  - 机器：MI355X（256 CU），`HIP_VISIBLE_DEVICES=3`；torch 2.11.0+rocm7.2，hipcc 来自 ROCm 7.1；默认 `-O3`，MFMA kernel 关闭 fast-math。
  - 实测带宽与算力：copy 4.65 TB/s，read 3.18 TB/s；F.linear 8192³ 为 1443 TFLOPs（fp16）、1591 TFLOPs（bf16）。
- 计时：
  - 用 HIP event；先 warmup ≥20 次，再在 HIP graph 中重放 ≥100 次调用，排除 host 开销。
  - 每个点取 **3 个计时窗口的最小值**，因为节点共享、GPU 0/1/2/6 上跑着他人的生产任务。
- L2/MALL 规避：使用轮换缓冲，总 footprint ≥1 GiB（4 倍 256 MB MALL），保证每次调用都从 HBM 读冷权重。Humming 的副本通过 deepcopy 已 transform 的 layer 得到。
- 指标定义：
  - TB/s 只计有用字节：量化权重 + scale/zp + 激活 + 输出。Humming 的 zp 是 uint8，按实际存储字节计。
  - TFLOPs = 2MNK/t。
- 正确性门槛：每个配置先在 N,K=512,1024、M∈{1,33,300} 上对比 fp32 参考，相对误差：Humming fp16 ≤2.9e-4，bf16 ≤2.4e-3。
- 噪声：
  - log_mfma.md 记录过整轮慢 2–8x 的现象。本轮未观察到，单点仍可能有约 ±5% 抖动。
  - 本轮 torch_dense 与 2026-10-28 GPU 5 的 baseline 数据一致（例如 4096² M=1 为 12.3 us）。
- 复现：
  ```bash
  HIP_VISIBLE_DEVICES=3 .venv-rocm/bin/python benchmarks/rocm/bench_baselines.py \
    --impls torch_dense vllm_triton_w4a16 vllm_exllama_gptq dequant_triton+hipblaslt \
            aiter_a16w8_int8 aiter_a16w8_fp8_block128 humming \
    --bits 2 3 4 8 --windows 3 --prefix humming_vs_baselines --tag 2026-10-28
  ```

## 7. 集成开销（vs log_mfma.md 独立 kernel）

int4 g128，同进程、同 `time_rotating`（best of 5）：

| shape | M=1 | M=16 | M=64 | M=255 | M=1024 |
|---|---|---|---|---|---|
| 28672×8192 fp16 独立 / layer (us) | 28.6 / 28.8 | 33.3 / 33.0 | 47.4 / 47.4 | 131.2 / 131.1 | 476.0 / 473.4 |
| 28672×8192 bf16 独立 / layer (us) | 33.6 / 33.7 | 40.9 / 41.0 | 55.1 / 54.8 | 147.9 / 147.4 | 514.7 / 511.5 |
| 4096×4096 fp16 独立 / layer (us) | 7.4 / 7.4 | 10.4 / 10.6 | 13.1 / 13.5 | 21.7 / 22.8 | 45.1 / 46.6 |

这些数值与 log_mfma.md 一致：28672×8192 fp16 的 M=1024 约 477 us，bf16 的 M=1024 为 512.9 us。

eager 模式（不用 graph）下，layer 路径每次调用约 30 us host 时间，独立调用约 15 us。M=1 小 shape 在 eager 下会被 Python 开销主导；实际部署应使用 CUDA/HIP graph。

## 8. 下一步

第二轮之后仍未解决的：

1. 大 M 的 dense / FP8 per-channel / MXFP4×MXFP4：需要权重经 LDS 分级 + 更大 tile 的 kernel 结构（现在每个 wave 各自在寄存器里 dequant B），以摊薄 dequant 并接近 MFMA 峰值。
2. bf16 大 M：dequant 比 fp16 贵，128×256 tile 放不下；需要更省寄存器的 dequant 或上面的 LDS 方案。
3. 低比特激活的 MoE、NVFP4、fp16/bf16 × FP 权重。
4. MoE：align kernel 只用一个 workgroup（64k 行时约 55 us）；tile 仅靠启发式，没有 autotune。
5. 上游：64 位 offset 修复可以作为独立 PR；ROCm 支持需先发 RFC（待确认）。

以下是第一轮的下一步（1–4 已在第二轮尝试，结果见第 0 节和 log_perf.md）。前四项来自 log_mfma.md，第五项来自第一轮数据。

1. 大 M 的 VALU 瓶颈：
   - 用 Marlin 式免移位 int4 dequant（mask + pk_fma）去掉约 48 条 `lshr`；
   - 改用 32×32×16 MFMA，减少 LDS 读 A；
   - A 走 `global_load_lds` 直接进 LDS。
2. 为 M≥1024 增加更大的 tile 档位，例如 warps_m=2 或 tiles_n=4，并把启发式扩展到 M=4096。
3. bf16：对称情形直接把位模式拼进 bf16 指数域，去掉 3 次转换。
4. decode 提速：
   - 每 CU 放更多在途字节（更宽的 tiles_n、更深的 stage），但受 VGPR 限制；
   - 改用 stream-K 式 split，避免 1/2/4/8 的粗粒度切分。
5. eager host 开销：`humming_forward` 的 Python 层约 15 us，可以合并 ROCm 分支的 padding/bias 判断，或直接走 torch op。
