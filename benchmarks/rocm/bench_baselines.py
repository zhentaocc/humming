#!/usr/bin/env python
"""ROCm (MI355X / gfx950) baseline benchmark for weight-only quantized GEMM.

Usage (always pin the GPU):
    HIP_VISIBLE_DEVICES=5 python benchmarks/rocm/bench_baselines.py            # full sweep
    HIP_VISIBLE_DEVICES=5 python benchmarks/rocm/bench_baselines.py --quick    # smoke
    ... --impls torch_dense vllm_triton_w4a16 --dtypes fp16 --bits 4 --shapes llama3-8b/o_proj --m 1 16

Implementations live in ``baselines.py`` (registry contract in ``common.py``).
Shapes live in ``shapes.py``. Methodology is documented in ``log_baseline.md``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import math
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# AITER's shipped JIT .so files in the shared checkout were built for another
# python/libstdc++; build our own copy locally (gitignored).
os.environ.setdefault("AITER_JIT_DIR", os.path.join(_HERE, ".aiter_jit"))
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

import torch  # noqa: E402

import baselines  # noqa: E402,F401  (populates REGISTRY)
from common import REGISTRY, Unsupported, make_problem, nominal_weight_bytes  # noqa: E402
from shapes import CHECK_M_VALUES, CHECK_SHAPE, HEADLINE_M, HEADLINE_SHAPES, M_VALUES, MODEL_SHAPES  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
MALL_BYTES = 256 * 2**20  # MI355X Infinity Cache (MALL)

# Vendor peak numbers (AMD Instinct MI355X data sheet, dense, no sparsity).
PEAKS = {
    "HBM3E bandwidth (TB/s)": 8.0,
    "FP16/BF16 dense MFMA (TFLOPs)": 2516.6,
    "FP8 dense (TFLOPs)": 5033.2,
    "MXFP4 dense (TFLOPs)": 10066.3,
    "Infinity Cache / MALL (MB)": 256,
    "L2 per XCD (MB)": 4,
    "CUs": 256,
}

CSV_FIELDS = [
    "impl", "kind", "dtype", "bits", "group_size", "zp", "shape", "M", "N", "K",
    "us", "tflops", "tbps", "weight_bytes", "total_bytes", "n_copies", "timing", "status",
]


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def _state_tensors(state):
    if isinstance(state, dict):
        return [v for v in state.values() if isinstance(v, torch.Tensor)]
    if isinstance(state, (list, tuple)):
        return [v for v in state if isinstance(v, torch.Tensor)]
    return []


def _footprint(state) -> int:
    return sum(t.numel() * t.element_size() for t in _state_tensors(state))


def time_run(run, states, warmup, iters, mode):
    """Average device time per call (us) cycling through ``states``.

    graph mode: capture L calls (cycling through all rotating copies) into one
    HIP graph and replay it until >= iters calls; removes host launch overhead.
    """
    n = len(states)
    for i in range(max(warmup, n)):
        run(states[i % n])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if mode == "graph":
        try:
            L = n * max(1, math.ceil(min(iters, 20) / n))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for j in range(L):
                    run(states[j % n])
            g.replay()
            torch.cuda.synchronize()
            replays = max(1, math.ceil(iters / L))
            start.record()
            for _ in range(replays):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            del g
            return ms * 1e3 / (replays * L), "graph"
        except Exception as e:  # fall back to eager
            torch.cuda.synchronize()
            print(f"    graph capture failed ({type(e).__name__}: {str(e)[:120]}); eager timing", flush=True)
    start.record()
    for i in range(iters):
        run(states[i % n])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters, "eager"


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------
def hbm_probe():
    out = {}
    nbytes = 4 * 2**30
    a = torch.empty(nbytes // 2, dtype=torch.float16, device="cuda").normal_()
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps = 20
    s.record()
    for _ in range(reps):
        b.copy_(a)
    e.record()
    torch.cuda.synchronize()
    out["copy 4 GiB (read+write) TB/s"] = 2 * nbytes * reps / (s.elapsed_time(e) * 1e-3) / 1e12
    s.record()
    for _ in range(reps):
        torch.sum(a, dtype=torch.float32)
    e.record()
    torch.cuda.synchronize()
    out["sum 4 GiB (read-only) TB/s"] = nbytes * reps / (s.elapsed_time(e) * 1e-3) / 1e12
    del a, b
    for name, dt in DTYPES.items():
        n = 8192
        x = torch.randn(n, n, device="cuda", dtype=dt)
        w = torch.randn(n, n, device="cuda", dtype=dt)
        for _ in range(5):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        s.record()
        for _ in range(50):
            torch.nn.functional.linear(x, w)
        e.record()
        torch.cuda.synchronize()
        out[f"{name} 8192^3 F.linear TFLOPs"] = 2 * n**3 * 50 / (s.elapsed_time(e) * 1e-3) / 1e12
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------
def check_correctness(spec, dtype, bits, group_size, zp):
    """Returns (ok, worst_rel_err, message)."""
    N, K = CHECK_SHAPE
    worst = 0.0
    for M in CHECK_M_VALUES:
        p = make_problem(M, N, K, dtype, bits, group_size, zp, seed=1234)
        setup, run = spec.factory(M, N, K, dtype, bits, group_size, zp)
        state = setup(p)
        out = run(state)
        torch.cuda.synchronize()
        x_ref = state.get("x_ref") if isinstance(state, dict) else None
        w_ref = state.get("w_ref") if isinstance(state, dict) else None
        x_ref = p.x.float() if x_ref is None else x_ref.float()
        w_ref = p.w_dequant(torch.float32) if w_ref is None else w_ref.float()
        ref = x_ref @ w_ref.t()
        if tuple(out.shape) != (M, N):
            return False, float("nan"), f"bad output shape {tuple(out.shape)} at M={M}"
        err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        if not math.isfinite(err):
            return False, err, f"non-finite output at M={M}"
        worst = max(worst, err)
        if err > spec.tol:
            return False, err, f"rel err {err:.3e} > tol {spec.tol} at M={M}"
    return True, worst, f"ok (max rel err {worst:.2e})"


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    impl: str
    dtype: str
    bits: int
    zp: bool


def build_configs(args):
    cfgs = []
    zps = {"both": [False, True], "sym": [False], "asym": [True]}[args.zp]
    for dname in args.dtypes:
        for name in args.impls:
            spec = REGISTRY[name]
            if spec.kind == "dense":
                cfgs.append(Config(name, dname, 16, False))
                continue
            for b in args.bits:
                if b not in spec.bits:
                    continue
                for z in zps:
                    cfgs.append(Config(name, dname, b, z))
    return cfgs


def fmt_bytes_cfg(c: Config):
    return f"{c.impl} [{c.dtype} {'dense' if c.bits == 16 else f'w{c.bits} ' + ('zp' if c.zp else 'sym')}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], choices=list(DTYPES))
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--zp", default="both", choices=["both", "sym", "asym"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--shapes", nargs="+", default=list(MODEL_SHAPES), choices=list(MODEL_SHAPES))
    ap.add_argument("--m", nargs="+", type=int, default=M_VALUES)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--flush-bytes", type=int, default=4 * MALL_BYTES,
                    help="rotating-buffer footprint target (default 4x the 256 MB MALL)")
    ap.add_argument("--max-copies", type=int, default=256)
    ap.add_argument("--timing", default="graph", choices=["graph", "eager"])
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 shapes x M in {1,16,1024}, fp16 only")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--tag", default=datetime.date.today().isoformat())
    ap.add_argument("--prefix", default="baselines", help="output file prefix: <prefix>_<tag>.{csv,md}")
    ap.add_argument("--windows", type=int, default=1,
                    help="timing windows per point; the best (min) is kept (noisy shared node)")
    args = ap.parse_args()
    if args.quick:
        args.shapes = ["llama3-8b/o_proj", "llama3-70b/gate_up"]
        args.m = [1, 16, 1024]
        args.dtypes = ["fp16"]

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.prefix}_{args.tag}")
    csv_path, md_path = base + ".csv", base + ".md"

    dev = torch.cuda.get_device_properties(0)
    env = {
        "device": f"{dev.name} ({dev.gcnArchName}), {dev.multi_processor_count} CUs, HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}",
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "hip": torch.version.hip,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": " ".join(sys.argv[1:]),
    }
    print(env, flush=True)
    probe = {} if args.no_probe else hbm_probe()
    for k, v in probe.items():
        print(f"probe: {k} = {v:.2f}", flush=True)

    cfgs = build_configs(args)
    status = {}  # cfg -> (ok, msg)
    rows = []
    fcsv = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
    writer.writeheader()

    for c in cfgs:
        spec = REGISTRY[c.impl]
        dtype = DTYPES[c.dtype]
        key = fmt_bytes_cfg(c)
        try:
            ok, err, msg = check_correctness(spec, dtype, c.bits if c.bits != 16 else 4, args.group_size, c.zp)
        except Unsupported as e:
            ok, msg = False, f"unsupported: {e}"
        except Exception as e:
            ok, msg = False, f"error: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            traceback.print_exc()
        status[key] = (ok, msg, spec.source)
        print(f"[check] {key}: {msg}", flush=True)
        if not ok:
            continue

        for sname in args.shapes:
            N, K = MODEL_SHAPES[sname]
            pbits = c.bits if c.bits != 16 else 4
            full = make_problem(max(args.m), N, K, dtype, pbits, args.group_size, c.zp, seed=0)
            for M in args.m:
                p = dataclasses.replace(full, M=M, x=full.x[:M])
                row = dict(impl=c.impl, kind=spec.kind, dtype=c.dtype, bits=c.bits, group_size=args.group_size,
                           zp=int(c.zp), shape=sname, M=M, N=N, K=K)
                try:
                    setup, run = spec.factory(M, N, K, dtype, pbits, args.group_size, c.zp)
                    states = []
                    s0 = setup(p)
                    for k in ("w_ref", "x_ref"):
                        if isinstance(s0, dict):
                            s0.pop(k, None)
                    states.append(s0)
                    fp = max(_footprint(s0), 1)
                    n_copies = min(args.max_copies, max(1, math.ceil(args.flush_bytes / fp)))
                    for _ in range(n_copies - 1):
                        s = setup(p)
                        if isinstance(s, dict):
                            s.pop("w_ref", None)
                            s.pop("x_ref", None)
                        states.append(s)
                    timings = [time_run(run, states, args.warmup, args.iters, args.timing) for _ in range(args.windows)]
                    us, mode = min(timings)
                    sd = s0 if isinstance(s0, dict) else {}
                    wb = sd.get("weight_bytes") or nominal_weight_bytes(N, K, c.bits, args.group_size, c.zp)
                    ab = sd.get("act_bytes") or M * K * 2
                    total = wb + ab + M * N * 2
                    row.update(us=round(us, 3), tflops=round(2 * M * N * K / (us * 1e-6) / 1e12, 3),
                               tbps=round(total / (us * 1e-6) / 1e12, 3), weight_bytes=wb, total_bytes=total,
                               n_copies=n_copies, timing=mode, status="ok")
                    del states, s0
                except Unsupported as e:
                    row.update(status=f"unsupported: {e}")
                except Exception as e:
                    row.update(status=f"error: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}")
                    traceback.print_exc()
                    torch.cuda.synchronize()
                print(f"  {key} {sname} M={M}: {row.get('us', '-')} us  {row.get('tflops', '-')} TF  "
                      f"{row.get('tbps', '-')} TB/s  copies={row.get('n_copies', '-')} {row['status']}", flush=True)
                writer.writerow(row)
                fcsv.flush()
                rows.append(row)
            del full
            torch.cuda.empty_cache()
    fcsv.close()
    write_markdown(md_path, env, probe, status, rows, args)
    print(f"wrote {csv_path}\nwrote {md_path}")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def _cfg_label(r):
    if int(r["bits"]) == 16:
        return f"{r['impl']} (dense)"
    return f"{r['impl']} w{r['bits']}{' zp' if int(r['zp']) else ' sym'}"


def write_markdown(path, env, probe, status, rows, args):
    L = []
    L.append(f"# ROCm baseline GEMM benchmark ({env['date']})\n")
    L.append("Y[M,N] = X[M,K] @ W[N,K]^T; weight-only quant unless noted; group size "
             f"{args.group_size}; zp = asymmetric zero point, sym = uint{{4,8}}b{{8,128}}.\n")
    L.append("## Environment\n")
    for k, v in env.items():
        L.append(f"- {k}: `{v}`")
    L.append("\n## MI355X peaks (vendor) and measured probes\n")
    L.append("| item | value |\n|---|---|")
    for k, v in PEAKS.items():
        L.append(f"| peak: {k} | {v} |")
    for k, v in probe.items():
        L.append(f"| measured: {k} | {v:.2f} |")
    L.append("\n## Methodology\n")
    L.append(f"- HIP-event timing; warmup >= {args.warmup}; >= {args.iters} timed calls; calls captured into a "
             "HIP graph (no host launch overhead) unless the `timing` column says eager. "
             f"Best of {args.windows} timing window(s) per point.")
    L.append(f"- L2/MALL defeat: rotating buffers. Each config builds n_copies independent (X, packed W, scales, "
             f"zeros, workspace) sets so the cyclic footprint is >= {args.flush_bytes / 2**20:.0f} MiB "
             "(4x the 256 MB MALL, capped at "
             f"{args.max_copies} copies); calls cycle through the copies, so every call reads cold weights from HBM.")
    L.append("- TFLOPs = 2MNK / t. Effective TB/s = (quantized weight bytes incl. scales/zeros + activation bytes "
             "+ fp16 output bytes) / t. For dequant+GEMM fallbacks the intermediate fp16 weight traffic is NOT "
             "counted (so TB/s is comparable across implementations as 'useful bytes').")
    L.append("- w8a8 / w4a4 rows time the GEMM only (activation quantization done in setup, not timed).")
    L.append("- Correctness: each config is checked against an fp32 torch reference on "
             f"N,K={CHECK_SHAPE}, M in {CHECK_M_VALUES} before timing (relative Frobenius error).\n")
    L.append("## Implementation status\n")
    L.append("| config | status | source |\n|---|---|---|")
    for k, (ok, msg, src) in status.items():
        L.append(f"| {k} | {'OK' if ok else 'SKIP'}: {msg} | {src} |")

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def table(dtype, shapes, ms, cell):
        cfgs = []
        for r in ok_rows:
            lab = _cfg_label(r)
            if r["dtype"] == dtype and lab not in cfgs:
                cfgs.append(lab)
        idx = {(_cfg_label(r), r["shape"], r["M"]): r for r in ok_rows if r["dtype"] == dtype}
        cols = [(s, m) for s in shapes for m in ms]
        out = ["| impl | " + " | ".join(f"{s} M={m}" for s, m in cols) + " |",
               "|---|" + "---|" * len(cols)]
        for lab in cfgs:
            cells = []
            for s, m in cols:
                r = idx.get((lab, s, m))
                cells.append(cell(r) if r else "-")
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
        return out

    dtypes = list(dict.fromkeys(r["dtype"] for r in ok_rows))
    hs = [s for s in HEADLINE_SHAPES if s in args.shapes]
    hm = [m for m in HEADLINE_M if m in args.m]
    for dt in dtypes:
        L.append(f"\n## Headline ({dt}): time in us (effective TB/s for M<=16, TFLOPs for M>=1024)\n")
        L += table(dt, hs, hm, lambda r: f"{r['us']:.1f} ({r['tbps']:.2f} TB/s)" if r["M"] <= 64
                   else f"{r['us']:.1f} ({r['tflops']:.0f} TF)")
    for dt in dtypes:
        L.append(f"\n## Full results ({dt}): `us / TFLOPs / TB/s`\n")
        for s in args.shapes:
            N, K = MODEL_SHAPES[s]
            L.append(f"\n### {s} (N={N}, K={K})\n")
            L += table(dt, [s], args.m, lambda r: f"{r['us']:.1f} / {r['tflops']:.1f} / {r['tbps']:.2f}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
