"""Time the shipped MSE Triton kernel against the eager search.

Not part of the PR - #2948 ships the kernel and its tests only. This exists
because the numbers from the exploratory benchmark do not describe this
kernel: those variants run with patience on, and none of them carry the
per-step scale reconstruction, the observed-dtype rounding chain or the tie
rule that this one needs to match eager exactly. Quoting 6.46x from there
would be quoting a different, wrong kernel.

Correctness is checked before anything is timed. A fast kernel that picks a
different range is not a result.

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
    cli = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    dtype = getattr(torch, cli.weight_dtype)
    torch.manual_seed(42)
    weight = torch.nn.Linear(
        cli.cols, cli.rows, bias=False, device="cuda", dtype=dtype
    ).weight.detach()
    args = QuantizationArgs(
        num_bits=cli.num_bits, type=cli.qtype, symmetric=True,
        strategy=QuantizationStrategy.GROUP, group_size=cli.group_size,
    )
    observed = flatten_for_calibration(weight, "weight", args)

    if not can_use_triton(observed, args):
        raise SystemExit("this config falls back to eager; nothing to time")

    n = neighbors_for_config(args, cli.grid, cli.maxshrink, dtype)
    print(f"{cli.rows}x{cli.cols} {cli.weight_dtype} {cli.qtype}"
          f"{cli.num_bits} group={cli.group_size}  observed="
          f"{tuple(observed.shape)}  N={n}\n")

    # correctness before speed: the kernel searches the whole grid, so the
    # honest reference is eager with early stopping disabled
    got = grid_search_triton(
        observed, args, cli.maxshrink, cli.grid, cli.norm
    )
    want = eager(
        observed, args, PATIENCE_OFF, cli.maxshrink, cli.grid, cli.norm
    )
    if not (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])):
        diff = int((got[0] != want[0]).sum())
        raise SystemExit(
            f"kernel disagrees with eager on {diff} groups; not timing a "
            "kernel that returns a different answer"
        )
    print("kernel matches eager full grid exactly\n")

    rows = [
        (f"eager (patience={cli.patience})",
         lambda: eager(observed, args, cli.patience, cli.maxshrink, cli.grid,
                       cli.norm)),
        ("eager (full grid)",
         lambda: eager(observed, args, PATIENCE_OFF, cli.maxshrink, cli.grid,
                       cli.norm)),
        ("triton (full grid)",
         lambda: grid_search_triton(observed, args, cli.maxshrink, cli.grid,
                                    cli.norm)),
    ]

    print(f"{'variant':>26} {'median':>10} {'peak mem':>10} {'vs eager':>9}")
    base = None
    for name, fn in rows:
        ms, mem = timed(fn, cli.warmup, cli.iters)
        if base is None:
            base = ms
        print(f"{name:>26} {ms:>9.2f}ms {mem:>9.1f}MB {base / ms:>8.2f}x")

    print(
        "\nvs eager is against the first row, which is the default path "
        f"today (patience={cli.patience}). The full-grid eager row is the "
        "like-for-like comparison, since the kernel does not early stop."
    )


if __name__ == "__main__":
    main()
