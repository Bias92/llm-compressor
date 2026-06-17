"""
Scratch profiler: where does GPTQ quantize_weight spend its time?

Answers HDCharles's question — "what % is cholesky vs error-feedback vs fake_quant" —
BEFORE deciding what to compile. Run EAGER (no compile). Wraps the three sections of
the modern `quantize_weight` in torch.profiler.record_function regions and reports the
CUDA-time breakdown. NOT for the PR.

Usage (pod):
    python profile_gptq_sections.py --model Qwen/Qwen3-8B --num-samples 16 --max-seq-len 2048
    # small model is fine — the % breakdown is structural:
    python profile_gptq_sections.py --model Qwen/Qwen2.5-3B-Instruct --num-samples 16 --max-seq-len 2048
"""

import argparse
from copy import copy

import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationStrategy,
    fake_quantize,
)
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function

import llmcompressor.modifiers.gptq.base as gptq_base
from llmcompressor.modifiers.gptq.gptq_quantize import (
    GPTQ_PRECISION,
    _apply_activation_ordering,
)
from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.pytorch.utils.helpers import tensor_sparsity


def quantize_weight_profiled(
    module, quant_args, hessian, blocksize: int = 128, percdamp: float = 0.01
):
    """Faithful copy of the modern eager quantize_weight, with record_function
    labels on cholesky / fake_quant / error_feedback / block_error sections."""
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessian
    observer = module.weight_observer

    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    if actorder == ActivationOrdering.GROUP and strategy not in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        actorder = None

    if actorder:
        W, H, perm = _apply_activation_ordering(W, H)
    if actorder == ActivationOrdering.GROUP:
        observer(W)

    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // divisor
        if actorder == ActivationOrdering.WEIGHT:
            g_idx = g_idx[perm]

    with record_function("gptq/observer"):
        qparams = observer.get_qparams()
        scale, zero_point, global_scale = (
            qparams["scale"],
            qparams["zero_point"],
            qparams["global_scale"],
        )

    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (
        (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
        if preserve_zeros
        else None
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    with record_function("gptq/cholesky"):
        try:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[0], device=H.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except torch._C._LinAlgError:
            logger.warning("hessian inversion failed; round-to-nearest fallback")
            Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if preserve_zeros:
            W1_nz_mask = W_nz_mask[:, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            with record_function("gptq/fake_quant"):
                if strategy == QuantizationStrategy.TENSOR:
                    q = fake_quantize(
                        q, scale, zero_point, quant_args, global_scale=global_scale
                    )
                elif strategy == QuantizationStrategy.CHANNEL:
                    q = fake_quantize(
                        q, scale[:, 0], zero_point[:, 0], quant_args,
                        global_scale=global_scale,
                    )
                elif strategy in (
                    QuantizationStrategy.GROUP,
                    QuantizationStrategy.TENSOR_GROUP,
                ):
                    gi = g_idx[i1 + i]
                    altered = copy(quant_args)
                    altered.strategy = QuantizationStrategy.CHANNEL
                    q = fake_quantize(
                        q, scale[:, gi], zero_point[:, gi], altered,
                        global_scale=global_scale,
                    )
                elif strategy == QuantizationStrategy.BLOCK:
                    bi = g_idx[i1 + i]
                    q = fake_quantize(
                        q.unsqueeze(1),
                        scale[:, bi : bi + 1],
                        zero_point[:, bi : bi + 1],
                        quant_args,
                        global_scale=global_scale,
                    ).squeeze(1)
                else:
                    raise ValueError(f"unsupported strategy: {strategy}")

            with record_function("gptq/error_feedback"):
                Q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2
                err1 = (w - q) / d
                w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                if preserve_zeros:
                    W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
                else:
                    W1[:, i:] -= w1_err
                Err1[:, i] = err1

        with record_function("gptq/block_error"):
            W[:, i1:i2] = Q1
            losses += torch.sum(losses1, 1) / 2
            w_err = Err1.matmul(Hinv[i1:i2, i2:])
            if preserve_zeros:
                W[:, i2:] -= w_err * W_nz_mask[:, i2:]
            else:
                W[:, i2:] -= w_err

    if actorder:
        invperm = torch.argsort(perm)
        W = W[:, invperm]

    W = W.reshape(final_shape).to(final_dtype)
    loss = torch.sum(losses).item()
    q_param_dict = {
        "weight": W,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if global_scale:
        q_param_dict["weight_global_scale"] = global_scale.to(dtype=final_dtype)
    if actorder == ActivationOrdering.GROUP:
        q_param_dict["weight_g_idx"] = g_idx[invperm]
    return (loss, q_param_dict)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=2048)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("need CUDA for a meaningful breakdown")

    # monkeypatch the eager quantize_weight with the instrumented copy
    gptq_base.quantize_weight = quantize_weight_profiled

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.num_samples}]"
    )
    ds = ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)})
    ds = ds.map(
        lambda s: tok(s["text"], padding=False, max_length=args.max_seq_len,
                      truncation=True, add_special_tokens=False),
        remove_columns=ds.column_names,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")

    recipe = """
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
                        strategy: "group"
                        group_size: 128
    """

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        oneshot(
            model=model, dataset=ds, tokenizer=tok, recipe=recipe,
            num_calibration_samples=args.num_samples, max_seq_length=args.max_seq_len,
        )

    print("\n" + "=" * 64)
    print("GPTQ quantize_weight section breakdown (CUDA time)")
    print("=" * 64)
    rows = {}
    for evt in prof.key_averages():
        if evt.key.startswith("gptq/"):
            rows[evt.key] = evt.cuda_time_total  # microseconds
    total = sum(rows.values()) or 1
    sections = (
        "gptq/observer",
        "gptq/cholesky",
        "gptq/fake_quant",
        "gptq/error_feedback",
        "gptq/block_error",
    )
    for k in sections:
        us = rows.get(k, 0)
        print(f"  {k:<24} {us/1e6:8.2f} s   {100*us/total:5.1f}%")
    print(f"  {'(sum of labeled)':<24} {total/1e6:8.2f} s")
    print("\nNote: this is time INSIDE quantize_weight only (not calibration forward).")
    print("Decide compile target from the dominant section.")


if __name__ == "__main__":
    main()
