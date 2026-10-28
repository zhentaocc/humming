# Humming on ROCm（MI355X / gfx950）性能结果

数据：`benchmarks/rocm/results/humming_vs_baselines_2026-10-28.{csv,md}`（3420 行，全部通过正确性检查）。
Humming 走公开 API：`HummingLayer.load_from_tensors → transform() → layer(x)`，即 `humming_forward` 背后 JIT 编译的 gfx950 MFMA GEMM。
对比对象：`torch_dense`（F.linear / hipBLASLt，fp16 权重）、`vllm_triton_w4a16`、`vllm_exllama_gptq`（仅 fp16）、`dequant_triton+hipblaslt`、`aiter_a16w8_int8`、`aiter_a16w8_fp8_block128`。
“最佳现有实现”指同一配置（dtype / bits / zp）下以上量化实现中最快的一个（不含 torch_dense；dense 单独列出）。

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

前四项来自 log_mfma.md，第五项来自本轮新数据。

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
