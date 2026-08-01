"""Model-level accuracy check for the MSE observer Triton path.

HDCharles asked for accuracy numbers across patience settings and offered to
run the scripts. This is that script.

It never substitutes the search implementation. An earlier draft
monkeypatched a kernel from a scratch benchmark into the observer, which
measured a kernel that is not the one being shipped and used a hardcoded
neighbor count. What is patched here is only which path gets taken and
whether it was taken: ``can_use_triton`` is forced False for the eager rows,
and ``grid_search_triton`` is wrapped in a counter for the triton row. The
search that runs is production's either way.

The Triton path searches the whole grid and has no patience, so the
comparison that matters is against eager at today's default patience - that
is what a user's numbers would change from.

Every config runs in its own process, so no compiled kernel cache, allocator
state or observer registration carries between them.

Usage:
    python bench_mse_accuracy.py --all
    python bench_mse_accuracy.py --config triton     # one config, in-process
"""

import argparse
import gc
import json
import math
import subprocess
import sys
import time

import torch

# (path, patience). "eager" forces the fallback so the two rows differ only
# in which search ran.
CONFIGS = {
    # the model is loaded in bfloat16, so do not label this fp16
    "unquantized_bf16": (None, None),
    # the patience sweep. HDCharles asked whether the default should move
    # from 5 to 10 or thereabouts, so these are the rows that answer it.
    # eager_full is the reference the others approximate: it minimises the
    # observer's own error over the whole grid, which is not the same as
    # giving the best model perplexity.
    "eager_p5": ("eager", 5),
    "eager_p10": ("eager", 10),
    "eager_p15": ("eager", 15),
    "eager_full": ("eager", 10**6),
    # the kernel has no patience; it searches the whole grid, so it should
    # land on eager_full and the patience column is not meaningful for it
    "triton": ("triton", None),
}


def force_eager():
    """Make can_use_triton say no, without touching the search itself."""
    from llmcompressor.observers import mse_quant

    mse_quant.can_use_triton = lambda *a, **k: False


def took_triton_path():
    """Did the observer actually reach the kernel? Counts calls."""
    from llmcompressor.observers import mse_quant

    hits = []
    real = mse_quant.grid_search_triton

    def counted(*a, **k):
        hits.append(1)
        return real(*a, **k)

    mse_quant.grid_search_triton = counted
    return lambda: len(hits)


def build_recipe(num_bits, group_size, patience):
    return f"""
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    targets: ["Linear"]
                    weights:
                        num_bits: {num_bits}
                        type: "int"
                        symmetric: true
                        strategy: "group"
                        group_size: {group_size}
                        observer: "mse"
                        observer_kwargs:
                            patience: {patience}
"""


def perplexity(model, enc, n_windows, seqlen):
    """Fixed-window perplexity, same windows for every config."""
    device = next(model.parameters()).device
    total = 0.0
    for i in range(n_windows):
        batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
        with torch.no_grad():
            out = model(batch, labels=batch)
        total += out.loss.float().item() * seqlen
    # math.exp, not torch.exp on a fresh tensor: that would default to
    # float32 and truncate the result after the json rounding was already
    # removed for the same reason
    return math.exp(total / (n_windows * seqlen))


def run_one(name, cli):
    path, patience = CONFIGS[name]
    torch.manual_seed(cli.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, dtype=torch.bfloat16, device_map="auto"
    )

    hits = None
    if path == "eager":
        force_eager()
    elif path == "triton":
        hits = took_triton_path()

    quant_s = 0.0
    peak_mb = 0.0
    if path is not None:
        # the kernel ignores patience, but the recipe still needs a value;
        # use the full-grid setting so the eager fallback inside the same
        # run would not early stop either
        effective_patience = 10**6 if patience is None else patience
        from datasets import load_dataset

        from llmcompressor import oneshot

        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split=f"train_sft[:{cli.calib_samples}]",
        ).shuffle(seed=cli.seed)
        ds = ds.map(
            lambda ex: {
                "text": tokenizer.apply_chat_template(
                    ex["messages"], tokenize=False
                )
            }
        )
        ds = ds.map(
            lambda s: tokenizer(
                s["text"],
                padding=False,
                max_length=cli.calib_seqlen,
                truncation=True,
                add_special_tokens=False,
            ),
            remove_columns=ds.column_names,
        )

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        oneshot(
            model=model,
            dataset=ds,
            num_calibration_samples=cli.calib_samples,
            max_seq_length=cli.calib_seqlen,
            recipe=build_recipe(
                cli.num_bits, cli.group_size, effective_patience
            ),
        )
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        quant_s = time.perf_counter() - t0

    # A triton config that never reached the kernel would be an eager run
    # wearing another label, and every row would agree for the wrong reason.
    n_hits = hits() if hits else None
    if path == "triton" and not n_hits:
        raise RuntimeError(
            f"{name}: the kernel never ran, so this would report the eager "
            "path under a triton label"
        )

    from datasets import load_dataset

    # namespaced id: huggingface_hub 1.x rejects a bare "wikitext"
    test = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
    windows = min(enc.numel() // cli.eval_seqlen, cli.eval_windows)
    ppl = perplexity(model, enc, windows, cli.eval_seqlen)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "config": name,
        "path": path or "none",
        "patience": patience,
        "quant_s": round(quant_s, 2),
        "peak_mb": round(peak_mb, 1),
        # full precision: rounding here would hide exactly the small
        # differences this script exists to detect
        "ppl": ppl,
        "kernel_calls": n_hits,
        "eval_windows": windows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--config", choices=list(CONFIGS))
    what.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--num-bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib-samples", type=int, default=64)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--eval-seqlen", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    if cli.all:
        passthrough = [
            a
            for a in sys.argv[1:]
            if a != "--all" and not a.startswith("--config")
        ]
        results = []
        for name in CONFIGS:
            print(f"--- {name} ---", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--config", name, "--json"]
                + passthrough,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(proc.stdout[-2000:])
                print(proc.stderr[-4000:])
                results.append({"config": name, "error": "failed"})
                continue
            row = json.loads(
                [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
            )
            results.append(row)
            print(json.dumps(row), flush=True)

        # only a row that actually produced a number can be the baseline;
        # a failed eager_p5 would otherwise KeyError on base["ppl"] below
        base = next(
            (r for r in results
             if r.get("config") == "eager_p5" and "ppl" in r),
            None,
        )
        print()
        if base is None:
            print("")
            print("no usable eager_p5 baseline, d ppl left blank")
        print(f"{'config':>12} {'path':>7} {'patience':>9} {'ppl':>12} "
              f"{'d ppl':>11} {'quant_s':>9} {'peak_mb':>9} {'calls':>7}")
        for r in results:
            if "ppl" not in r:
                print(f"{r['config']:>12} {'failed':>50}")
                continue
            d = (
                f"{r['ppl'] - base['ppl']:+.6f}"
                if base and r is not base
                else "-"
            )
            print(f"{r['config']:>12} {r['path']:>7} "
                  f"{str(r['patience']):>9} {r['ppl']:>12.6f} {d:>11} "
                  f"{r['quant_s']:>9.2f} {r['peak_mb']:>9.1f} "
                  f"{str(r['kernel_calls']):>7}")
        print(
            "\nd ppl is against eager at the default patience, which is what "
            "the triton path replaces. calls is how many times the kernel "
            "actually ran; a triton row showing None or 0 did not measure "
            "what its name says."
        )
        failed = [r["config"] for r in results if "ppl" not in r]
        if failed:
            # a partial run printed as a table reads like a complete one
            raise SystemExit(
                f"{len(failed)} config(s) failed: "
                + ", ".join(failed)
            )
        return

    print(json.dumps(run_one(cli.config, cli)))


if __name__ == "__main__":
    main()
