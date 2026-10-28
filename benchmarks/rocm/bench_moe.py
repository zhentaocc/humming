#!/usr/bin/env python
"""ROCm (MI355X / gfx950) MoE layer benchmark: Humming MFMA MoE vs vLLM Triton / AITER.

Usage (always pin the GPU):
    HIP_VISIBLE_DEVICES=5 python benchmarks/rocm/bench_moe.py                       # full sweep
    ... --models qwen3-30b-a3b --dtypes bf16 --bits 4 --impls humming triton_wna16 --tokens 1 64
    ... --from-csv benchmarks/rocm/results/humming_moe_2026-10-28.csv               # redo markdown

The timed unit is the routed-expert part of one MoE block (see ``moe_baselines.py``): routing
prep + w13 + SiLU-and-mul + w2 + weighted top-k sum. Methodology is in ``docs/rocm/log_moe.md``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import math
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
# AITER's MoE needs flydsl >= 0.1.8 (pinned 0.2.3); the venv / user site ships 0.1.3.1, so a private
# copy is installed into benchmarks/rocm/.pydeps (gitignored, see log_moe.md) and shadows it here.
_PYDEPS = os.path.join(_HERE, ".pydeps")
if os.path.isdir(_PYDEPS):
    sys.path.insert(0, _PYDEPS)
os.environ.setdefault("AITER_JIT_DIR", os.path.join(_HERE, ".aiter_jit"))
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

import torch  # noqa: E402

from moe_baselines import IMPLS, QUANTIZED, Routing, TritonWNA16TunedMoE, Unsupported  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# name -> (num_experts, top_k, hidden, moe_intermediate)
MODELS = {
    "mixtral-8x7b": (8, 2, 4096, 14336),
    "qwen3-30b-a3b": (128, 8, 2048, 768),
    "deepseek-v3": (256, 8, 7168, 2048),
}
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64]
PREFILL_TOKENS = [512, 1024, 2048, 4096, 8192]
CHECK_TOKENS = [3, 40]
ROTATE_BYTES = 1 << 30  # rotate weight copies until >= 1 GiB is touched (MALL is 256 MB)
MAX_COPIES = 8

CSV_FIELDS = ["model", "impl", "dtype", "bits", "group_size", "tokens", "layer_us", "align_us", "w13_us", "w2_us",
              "weight_bytes", "n_copies", "config", "rel_err", "status"]


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------
def time_graph(fn, n_copies: int, reps: int, target_ms: float = 20.0) -> float:
    """Best-of-``reps`` device time per call (us); each rep replays a graph of L calls cycling
    through the rotating weight copies."""
    for i in range(max(2, n_copies)):
        fn(i % n_copies)
    torch.cuda.synchronize()
    calls = n_copies * max(1, math.ceil(8 / n_copies))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(calls):
            fn(j % n_copies)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    replays = max(1, min(200, int(target_ms / max(start.elapsed_time(end), 1e-3))))
    best = float("inf")
    for _ in range(reps):
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / (replays * calls))
    del graph
    return best


# --------------------------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------------------------
def master_weights(model, dtype, seed=0):
    num_experts, _, hidden, inter = MODELS[model]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.empty(num_experts, 2 * inter, hidden, dtype=dtype, device="cuda")
    w2 = torch.empty(num_experts, hidden, inter, dtype=dtype, device="cuda")
    for e in range(num_experts):  # chunked: keeps the fp32 temporaries small
        w13[e] = (torch.randn(2 * inter, hidden, generator=gen, device="cuda") / hidden**0.5).to(dtype)
        w2[e] = (torch.randn(hidden, inter, generator=gen, device="cuda") / inter**0.5).to(dtype)
    return w13, w2


def make_routing(model, tokens, dtype, seed=1) -> Routing:
    num_experts, top_k, hidden, _ = MODELS[model]
    gen = torch.Generator(device="cuda").manual_seed(seed + tokens)
    x = torch.randn(tokens, hidden, generator=gen, device="cuda").to(dtype)
    logits = torch.randn(tokens, num_experts, generator=gen, device="cuda")
    weights, ids = torch.softmax(logits, dim=-1).topk(top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return Routing(x, ids.to(torch.int32).contiguous(), weights.float().contiguous())


def reference(impl, routing) -> torch.Tensor:
    experts = routing.topk_ids.unique().tolist()
    w13, w2 = impl.dequant(experts)
    out = torch.zeros(routing.tokens, w2.size(1), device="cuda")
    x = routing.x.float()
    for local, expert in enumerate(experts):
        token, slot = (routing.topk_ids == expert).nonzero(as_tuple=True)
        h = x[token] @ w13[local].T
        inter = h.size(1) // 2
        act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        out.index_add_(0, token, (act @ w2[local].T) * routing.topk_weights[token, slot, None])
    return out


def check(impl, model, dtype) -> float:
    worst = 0.0
    for tokens in CHECK_TOKENS:
        routing = make_routing(model, tokens, dtype, seed=7)
        out = impl.prepare(routing)(0)
        ref = reference(impl, routing)
        worst = max(worst, ((out.float() - ref).norm() / ref.norm()).item())
    return worst


# --------------------------------------------------------------------------------------------
# Triton config search
# --------------------------------------------------------------------------------------------
def tune_triton(impl: TritonWNA16TunedMoE, routing, n_copies, reps):
    """Per BLOCK_M: best (BLOCK_N, BLOCK_K, warps) for w13 and w2 separately; pick BLOCK_M by sum."""
    best = None
    for block_m in impl.block_m_candidates(routing.tokens, routing.top_k):
        per_gemm = {}
        for cfg in impl.candidate_configs(routing.tokens, routing.top_k):
            if cfg["BLOCK_SIZE_M"] != block_m:
                continue
            try:
                run = impl.prepare(routing, (cfg, cfg))
                for stage in ("w13", "w2"):
                    us = time_graph(run.stages[stage], n_copies, max(1, reps // 2), target_ms=5.0)
                    if stage not in per_gemm or us < per_gemm[stage][0]:
                        per_gemm[stage] = (us, cfg)
            except Exception as e:  # noqa: BLE001  (OutOfResources etc.)
                print(f"      skip {cfg}: {type(e).__name__}", flush=True)
                torch.cuda.synchronize()
        if len(per_gemm) == 2:
            total = per_gemm["w13"][0] + per_gemm["w2"][0]
            if best is None or total < best[0]:
                best = (total, per_gemm["w13"][1], per_gemm["w2"][1])
    assert best is not None, "no Triton config ran"
    return best[1], best[2]


# --------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------
def bench_impl(name, model, dtype_name, bits, w13, w2, tokens_list, args, writer):
    dtype = DTYPES[dtype_name]
    group_size = 128 if name in QUANTIZED and name != "aiter_a16wi4" else (32 if name == "aiter_a16wi4" else 0)
    row_base = {"model": model, "impl": name, "dtype": dtype_name, "bits": bits, "group_size": group_size}
    try:
        t0 = time.time()
        impl = IMPLS[name](w13, w2, dtype, bits, group_size or 128)
        rel_err = check(impl, model, dtype)
        tol = 1e-2 if dtype == torch.float16 else 2e-2
        if not rel_err < tol:
            raise RuntimeError(f"correctness check failed: rel_err={rel_err:.3e} > {tol}")
        weight_bytes = impl.weight_bytes()
        n_copies = max(1, min(MAX_COPIES, math.ceil(ROTATE_BYTES / weight_bytes)))
        impl.add_copies(n_copies)
        print(f"  {name:20s} {dtype_name} w{bits}: built+checked in {time.time() - t0:.0f}s, rel_err={rel_err:.2e}, "
              f"weights {weight_bytes / 2**20:.0f} MiB x{n_copies}", flush=True)
    except Unsupported as e:
        print(f"  {name:20s} {dtype_name} w{bits}: unsupported ({e})", flush=True)
        return
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        writer({**row_base, "status": f"error: {type(e).__name__}: {str(e)[:150]}"})
        torch.cuda.empty_cache()
        return

    for tokens in tokens_list:
        routing = make_routing(model, tokens, dtype)
        row = {**row_base, "tokens": tokens, "weight_bytes": weight_bytes, "n_copies": n_copies,
               "rel_err": f"{rel_err:.2e}"}
        try:
            config = None
            if isinstance(impl, TritonWNA16TunedMoE):
                config = tune_triton(impl, routing, n_copies, args.reps)
            run = impl.prepare(routing, config)
            row["layer_us"] = f"{time_graph(run, n_copies, args.reps):.2f}"
            for stage, fn in getattr(run, "stages", {}).items():
                row[f"{stage}_us"] = f"{time_graph(fn, n_copies, args.reps):.2f}"
            row["config"] = impl.last_config
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            row["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
            torch.cuda.synchronize()
        print(f"    T={tokens:5d} layer {row.get('layer_us', '-'):>9s} us  w13 {row.get('w13_us', '-'):>9s}  "
              f"w2 {row.get('w2_us', '-'):>9s}  align {row.get('align_us', '-'):>7s}  {row.get('config', '')}",
              flush=True)
        writer(row)
    del impl
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"], choices=list(DTYPES))
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--impls", nargs="+", default=list(IMPLS), choices=list(IMPLS))
    ap.add_argument("--tokens", nargs="+", type=int, default=DECODE_TOKENS + PREFILL_TOKENS)
    ap.add_argument("--reps", type=int, default=5, help="best-of-N timing repetitions")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--tag", default=f"humming_moe_{datetime.date.today().isoformat()}")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV instead of overwriting")
    ap.add_argument("--from-csv", help="only regenerate the markdown from this CSV")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.tag}.csv")
    md_path = os.path.join(args.out_dir, f"{args.tag}.md")
    if args.from_csv:
        with open(args.from_csv) as f:
            rows = list(csv.DictReader(f))
        write_markdown(md_path, rows)
        print(f"wrote {md_path}")
        return

    print(f"device: {torch.cuda.get_device_name()} (HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')})")
    mode = "a" if args.append and os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        out = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            out.writeheader()

        def writer(row):
            out.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            f.flush()

        for model in args.models:
            for dtype_name in args.dtypes:
                print(f"== {model} {dtype_name} {MODELS[model]}", flush=True)
                w13, w2 = master_weights(model, DTYPES[dtype_name])
                for name in args.impls:
                    bits_list = args.bits if name in QUANTIZED else [16]
                    for bits in bits_list:
                        bench_impl(name, model, dtype_name, bits, w13, w2, args.tokens, args, writer)
                del w13, w2
                gc.collect()
                torch.cuda.empty_cache()
    with open(csv_path) as f:
        write_markdown(md_path, list(csv.DictReader(f)))
    print(f"wrote {csv_path}\nwrote {md_path}")


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------
COLUMNS = [  # (label, impl, bits)
    ("Humming W4", "humming", "4"),
    ("Triton W4 (vLLM default)", "triton_wna16", "4"),
    ("Triton W4 (tuned)", "triton_wna16_tuned", "4"),
    ("AITER a16wi4", "aiter_a16wi4", "4"),
    ("Humming W8", "humming", "8"),
    ("Triton W8 (vLLM default)", "triton_wna16", "8"),
    ("Triton W8 (tuned)", "triton_wna16_tuned", "8"),
    ("Triton 16-bit", "triton_fp16", "16"),
    ("AITER 16-bit", "aiter_bf16", "16"),
]


def write_markdown(path, rows):
    ok = {}
    for r in rows:
        if r["status"] == "ok":
            ok[(r["model"], r["dtype"], r["impl"], r["bits"], int(r["tokens"]))] = float(r["layer_us"])
    lines = [f"# Humming MoE (ROCm gfx950) vs ROCm MoE baselines, {datetime.date.today().isoformat()}", ""]
    lines += [
        "Generated by `benchmarks/rocm/bench_moe.py` (methodology: `docs/rocm/log_moe.md`).",
        "",
        "- Unit: one MoE block's routed experts, i.e. routing prep (align / sort) + w13 + SiLU-and-mul + w2 + "
        "weighted top-k sum. Router GEMM / top-k excluded. Times in us, HIP-graph replay, best of N; weight "
        "copies rotate so that >= 1 GiB of weights is touched per cycle (no MALL reuse).",
        "- W4 / W8: symmetric int, group 128 (Humming, Triton); AITER a16wi4 is int4 group 32, bf16 only.",
        "- `speedup W4` = best non-Humming W4 time / Humming W4 time (likewise W8); `vs 16-bit` = best 16-bit "
        "time / best Humming time.",
        "",
    ]
    for model in MODELS:
        for dtype in ["bf16", "fp16"]:
            tokens = sorted({k[4] for k in ok if k[0] == model and k[1] == dtype})
            if not tokens:
                continue
            num_experts, top_k, hidden, inter = MODELS[model]
            cols = [c for c in COLUMNS if any((model, dtype, c[1], c[2], t) in ok for t in tokens)]
            lines.append(f"## {model} ({dtype}): E={num_experts}, top_k={top_k}, hidden={hidden}, inter={inter}")
            lines.append("")
            header = ["tokens"] + [c[0] for c in cols] + ["speedup W4", "speedup W8", "vs 16-bit"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            for t in tokens:
                cells = [str(t)]
                for _, impl, bits in cols:
                    v = ok.get((model, dtype, impl, bits, t))
                    cells.append(f"{v:.1f}" if v is not None else "-")

                def speedup(bits):
                    mine = ok.get((model, dtype, "humming", bits, t))
                    others = [ok[(model, dtype, c[1], bits, t)] for c in COLUMNS
                              if c[2] == bits and c[1] != "humming" and (model, dtype, c[1], bits, t) in ok]
                    return f"{min(others) / mine:.2f}x" if mine and others else "-"

                mine = [ok[(model, dtype, "humming", b, t)] for b in ("4", "8") if (model, dtype, "humming", b, t) in ok]
                dense = [ok[(model, dtype, i, "16", t)] for i in ("triton_fp16", "aiter_bf16")
                         if (model, dtype, i, "16", t) in ok]
                vs16 = f"{min(dense) / min(mine):.2f}x" if mine and dense else "-"
                cells += [speedup("4"), speedup("8"), vs16]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        lines += ["## Failures", "", "| model | impl | dtype | bits | tokens | status |", "|---|---|---|---|---|---|"]
        for r in errors:
            lines.append(f"| {r['model']} | {r['impl']} | {r['dtype']} | {r['bits']} | {r['tokens']} | {r['status']} |")
        lines.append("")
    lines += ["## Configs", "", "| model | dtype | impl | bits | tokens | config | align us | w13 us | w2 us |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "ok" and r["impl"] in ("humming", "triton_wna16", "triton_wna16_tuned"):
            lines.append(f"| {r['model']} | {r['dtype']} | {r['impl']} | {r['bits']} | {r['tokens']} | {r['config']} | "
                         f"{r['align_us']} | {r['w13_us']} | {r['w2_us']} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
