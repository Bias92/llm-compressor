"""Time the shipped MSE Triton kernel against the eager search.

Not part of the PR - #2948 ships the kernel and its tests only. This exists
because the numbers from the exploratory benchmark do not describe this
kernel: those variants have a different neighbor count and none of them
carry the per-group patience counter.

Accuracy is checked before anything is timed, but not with torch.equal. The
kernel searches the same twenty ranges eager does and scores them with
scale_0 * p in fp32, which below fp32 is not the arithmetic that will be
applied, so it can pick a different one. What has to hold is that the range
it picks is about as good, measured against the best range the grid contains
rather than against eager's pick - eager is a search too, not the answer.

Usage:
    python bench_mse_triton.py
    python bench_mse_triton.py --rows 8192 --cols 8192 --num-bits 4
"""

import argparse
import time

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams

from llmcompressor.observers.helpers import flatten_for_calibration
from llmcompressor.observers.mse_quant import _grid_search_eager
from llmcompressor.observers.mse_triton import (
    can_use_triton,
    grid_search_triton,
    neighbors_for_config,
)

PATIENCE_OFF = 10**6


def eager(observed, args, patience, maxshrink, grid, norm):
    token_args = args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    return _grid_search_eager(
        observed, args, token_args, min_val, max_val,
        torch.full_like(min_val, torch.finfo(min_val.dtype).max),
        min_val.clone(), max_val.clone(),
        int(maxshrink * grid), patience, grid, norm,
    )


def objective(observed, args, min_v, max_v, norm):
    """Per-group quantization error for a range, scored in double."""
    token_args = args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )
    scale, zp = calculate_qparams(
        min_vals=min_v, max_vals=max_v, quantization_args=args,
        global_scale=None,
    )
    q = fake_quantize(
        observed, scale.unsqueeze(-1), zp.unsqueeze(-1), token_args
    )
    return (q.double() - observed.double()).abs().pow(norm).sum(dim=(0, -1))


def best_on_grid(observed, args, maxshrink, grid, norm):
    """The lowest error any step of the grid reaches, per group."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best = None
    for i in range(int(maxshrink * grid)):
        p = 1.0 - i / grid
        err = objective(observed, args, min_val * p, max_val * p, norm)
        best = err if best is None else torch.minimum(best, err)
    return best


def timed(fn, warmup, iters):
    """Median wall time and the peak allocation the call itself adds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = (torch.cuda.max_memory_allocated() - before) / 1024**2
    return sorted(times)[len(times) // 2] * 1e3, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--num-bits", type=int, default=8)
    ap.add_argument("--qtype", choices=["int", "float"], default="int")
    ap.add_argument(
        "--asymmetric",
        action="store_true",
        help="use asymmetric integer quantization",
    )
    ap.add_argument(
        "--weight-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    ap.add_argument("--maxshrink", type=float, default=0.20)
    ap.add_argument("--grid", type=float, default=100.0)
    ap.add_argument("--norm", type=float, default=2.4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--regret-bar", type=float, default=5e-2,
        help="mean regret above which the timing is refused",
    )
    cli = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    dtype = getattr(torch, cli.weight_dtype)
    torch.manual_seed(42)
    weight = torch.nn.Linear(
        cli.cols, cli.rows, bias=False, device="cuda", dtype=dtype
    ).weight.detach()
    args = QuantizationArgs(
        num_bits=cli.num_bits,
        type=cli.qtype,
        symmetric=not cli.asymmetric,
        strategy=QuantizationStrategy.GROUP, group_size=cli.group_size,
    )
    observed = flatten_for_calibration(weight, "weight", args)

    if not can_use_triton(observed, args):
        raise SystemExit("this config falls back to eager; nothing to time")

    n = neighbors_for_config(args, cli.grid, cli.maxshrink)
    symmetry = "asymmetric" if cli.asymmetric else "symmetric"
    print(f"{cli.rows}x{cli.cols} {cli.weight_dtype} {symmetry} {cli.qtype}"
          f"{cli.num_bits} group={cli.group_size}  observed="
          f"{tuple(observed.shape)}  N={n}\n")

    variants = [
        (f"eager (patience={cli.patience})",
         lambda p=cli.patience: eager(
             observed, args, p, cli.maxshrink, cli.grid, cli.norm)),
        ("eager (full grid)",
         lambda: eager(
             observed, args, PATIENCE_OFF, cli.maxshrink, cli.grid, cli.norm)),
        (f"triton (patience={cli.patience})",
         lambda p=cli.patience: grid_search_triton(
             observed, args, cli.maxshrink, p, cli.grid, cli.norm)),
        ("triton (full grid)",
         lambda: grid_search_triton(
             observed, args, cli.maxshrink, PATIENCE_OFF, cli.grid, cli.norm)),
    ]

    # accuracy before speed: a fast kernel that picks a worse range is not a
    # result, and the reference is the grid's own optimum, not eager's pick
    floor = best_on_grid(observed, args, cli.maxshrink, cli.grid, cli.norm)
    print(
        f"{'variant':>28} {'mean regret':>12} {'global':>10} "
        f"{'worst':>10}"
    )
    regrets = {}
    for name, fn in variants:
        got = objective(observed, args, *fn(), cli.norm)
        rel = (got - floor) / floor.abs().clamp_min(1e-30)
        regrets[name] = float(rel.mean())
        global_regret = (got.sum() - floor.sum()) / floor.sum().abs().clamp_min(
            1e-30
        )
        print(
            f"{name:>28} {float(rel.mean()):>12.2e} "
            f"{float(global_regret):>10.2e} {float(rel.max()):>10.2e}"
        )

    over = {k: v for k, v in regrets.items() if v > cli.regret_bar}
    if over:
        raise SystemExit(
            "not timing a search that gave up this much accuracy: "
            + ", ".join(f"{k} at {v:.2e}" for k, v in over.items())
        )

    print(f"\n{'variant':>28} {'median':>10} {'peak mem':>10} {'vs eager':>9}")
    base = None
    for name, fn in variants:
        ms, mem = timed(fn, cli.warmup, cli.iters)
        if base is None:
            base = ms
        print(f"{name:>28} {ms:>9.2f}ms {mem:>9.1f}MB {base / ms:>8.2f}x")

    print(
        "\nvs eager is against the first row, which is the default path "
        f"today (patience={cli.patience}). regret is how far above the best "
        "range on the grid each variant landed, averaged over groups."
    )


if __name__ == "__main__":
    main()
