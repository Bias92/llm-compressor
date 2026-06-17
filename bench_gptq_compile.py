"""
Scratch A100 benchmark + diagnostic for the GPTQ torch.compile path
(modern branch: torch-compile-observers).

NOT part of any PR. Keep OUT OF TREE. Copy onto the pod, run, copy numbers back,
then TERMINATE the pod (never reuse a pod — migration bug).

Two stages:

  # 1) cheap viability check FIRST — decides the kernel shape before burning hours
  TORCH_LOGS="graph_breaks,recompiles" python bench_gptq_compile.py diagnose \
      --model Qwen/Qwen3-8B

  # 2) full e2e matrix once the kernel forms a useful graph
  python bench_gptq_compile.py bench --model Qwen/Qwen3-8B --strategy group --runs 3
  python bench_gptq_compile.py bench --model Qwen/Qwen3-8B --strategy group \
      --runs 3 --activation

Anti-distortion measures (A100 numbers still drift — these are mandatory):
  * model + dataset are downloaded/cached BEFORE any timing (the "7.5x" mistake
    was download time landing inside the eager run)
  * explicit warmup; >=3 timed runs; median + all runs printed
  * cold (first compile) reported separately from warm
  * torch.cuda.synchronize() around the timed region; peak memory via reset stats
  * refuses torch 2.11 (unbacked dynamic-shapes perf regression)
  * asserts the global compile flag actually flips the path before trusting results

Env: this branch runs on the modern stack already on the pod (modern
compressed_tensors / transformers). Keep the RunPod PyTorch template's torch
(do NOT install 2.10 — it does not exist on the real index; never 2.11).
"""

import argparse
import gc
import logging
import statistics
import time

import torch

# Modern branch: the GPTQ compile flag lives in the gptq module itself
# (mirrors observers/mse_quant.py's set_torch_compile convention).
from llmcompressor.modifiers.gptq.gptq_quantize import (
    get_gptq_compile,
    set_gptq_compile,
)


def _guard_env():
    ver = torch.__version__
    if ver.startswith("2.11"):
        raise SystemExit(
            f"Refusing to benchmark on torch {ver}: 2.11 has the unbacked "
            "dynamic-shapes perf regression. Numbers would be invalid."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. Real compile-vs-eager timing needs a GPU; CPU "
            "fallback numbers are meaningless."
        )
    print(f"torch {ver} | device {torch.cuda.get_device_name(0)}")


def _recipe(strategy: str, group_size: int, activation: bool) -> str:
    acts = (
        """
                        input_activations:
                            num_bits: 8
                            type: "int"
                            symmetric: true
                            strategy: "tensor"
                            dynamic: false
        """
        if activation
        else ""
    )
    gs = (
        f"\n                            group_size: {group_size}"
        if strategy == "group"
        else ""
    )
    return f"""
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    targets: ["Linear"]
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: "{strategy}"{gs}
{acts}
    """


def _prep(model_id: str, num_samples: int, max_seq_len: int):
    """Download + tokenize BEFORE timing so no I/O lands in a timed run."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{num_samples}]"
    )
    ds = ds.map(
        lambda ex: {
            "text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        }
    )
    ds = ds.map(
        lambda s: tokenizer(
            s["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=ds.column_names,
    )
    return tokenizer, ds


def _run_oneshot(model_id, ds, tokenizer, recipe, num_samples, max_seq_len, enable):
    from transformers import AutoModelForCausalLM

    from llmcompressor import oneshot

    set_gptq_compile(enable)
    assert get_gptq_compile() is enable, "compile flag did not flip"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        tokenizer=tokenizer,
        recipe=recipe,
        num_calibration_samples=num_samples,
        max_seq_length=max_seq_len,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024**2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, peak


def cmd_diagnose(args):
    """Surface graph breaks + recompiles for the compiled column kernel. If it
    graph-breaks into a no-op, switch to a block-vectorized kernel before
    spending A100 time on the full matrix."""
    _guard_env()
    logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
    print(
        'Run with TORCH_LOGS="graph_breaks,recompiles" to see what the compiled '
        "_quantize_column kernel does. Watch for:\n"
        "  * repeated recompiles per layer (-> mark_dynamic / clone+mark_unbacked)\n"
        "  * a graph break making the kernel a no-op (-> kernel must become the "
        "block-vectorized form; per-column fake_quantize is not enough)\n"
    )
    tokenizer, ds = _prep(args.model, args.num_samples, args.max_seq_len)
    recipe = _recipe(args.strategy, args.group_size, args.activation)
    t, mem = _run_oneshot(
        args.model, ds, tokenizer, recipe, args.num_samples, args.max_seq_len, True
    )
    print(f"diagnose (compiled): {t:.1f}s | peak {mem:.0f} MB")


def cmd_bench(args):
    _guard_env()
    tokenizer, ds = _prep(args.model, args.num_samples, args.max_seq_len)
    recipe = _recipe(args.strategy, args.group_size, args.activation)

    label = (
        f"{args.model} | {args.strategy}"
        f"{'+act' if args.activation else ' (weight-only)'} | "
        f"seq={args.max_seq_len} n={args.num_samples}"
    )
    print("=" * 72)
    print(label)
    print("=" * 72)

    print("warmup (eager)...")
    _run_oneshot(
        args.model, ds, tokenizer, recipe, args.num_samples, args.max_seq_len, False
    )

    def timed(enable, n):
        times, mems = [], []
        for i in range(n):
            t, m = _run_oneshot(
                args.model, ds, tokenizer, recipe,
                args.num_samples, args.max_seq_len, enable,
            )
            times.append(t)
            mems.append(m)
            print(f"  run {i + 1}/{n}: {t:.1f}s | peak {m:.0f} MB")
        return times, mems

    print("\neager (this branch, flag off):")
    eager_t, eager_m = timed(False, args.runs)

    print("\ncompiled COLD (first compile included):")
    cold_t, cold_m = timed(True, 1)

    print("\ncompiled WARM (graph cached):")
    warm_t, warm_m = timed(True, args.runs)

    e = statistics.median(eager_t)
    c = cold_t[0]
    w = statistics.median(warm_t)
    print("\n" + "-" * 72)
    print(f"{'config':<26}{'median s':<12}{'peak MB':<12}{'vs eager':<12}")
    print(f"{'eager':<26}{e:<12.1f}{statistics.median(eager_m):<12.0f}{'':<12}")
    print(
        f"{'compiled cold':<26}{c:<12.1f}{cold_m[0]:<12.0f}"
        f"{(e / c if c else 0):.2f}x"
    )
    print(
        f"{'compiled warm':<26}{w:<12.1f}{statistics.median(warm_m):<12.0f}"
        f"{(e / w if w else 0):.2f}x"
    )
    print("-" * 72)
    print(
        "Reminder: also capture an eager-on-MAIN run (without these GPTQ-compile "
        "commits) for the third leg of the matrix. And TERMINATE the pod."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="Qwen/Qwen3-8B")
    common.add_argument("--strategy", choices=["group", "channel"], default="group")
    common.add_argument("--group-size", type=int, default=128)
    common.add_argument("--activation", action="store_true")
    common.add_argument("--num-samples", type=int, default=64)
    common.add_argument("--max-seq-len", type=int, default=4096)

    d = sub.add_parser("diagnose", parents=[common])
    d.set_defaults(func=cmd_diagnose)

    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--runs", type=int, default=3)
    b.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
