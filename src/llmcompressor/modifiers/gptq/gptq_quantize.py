import math
from copy import copy

import torch
import torch._dynamo.config
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from loguru import logger

from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

GPTQ_PRECISION = torch.float32

__all__ = ["make_empty_hessian", "accumulate_hessian", "quantize_weight"]

_enable_gptq_compile = False


def set_gptq_compile(enabled: bool):
    global _enable_gptq_compile
    _enable_gptq_compile = enabled


def get_gptq_compile() -> bool:
    return _enable_gptq_compile


# Stage-2 experiment: compile the whole per-column inner loop (quant +
# error-feedback) as one helper, vs. just the per-column fake_quantize above.
_enable_gptq_block_compile = False


def set_gptq_block_compile(enabled: bool):
    global _enable_gptq_block_compile
    _enable_gptq_block_compile = enabled


def get_gptq_block_compile() -> bool:
    return _enable_gptq_block_compile


# Allow torch.compile to handle scalar conversions inside compressed_tensors'
# calculate_qparams (float(bit_range)). Same approach as the MSE observer
# compile path (observers/mse_quant.py) and GPTQ commit a4f9ba2e.
torch._dynamo.config.capture_scalar_outputs = True


def _quantize_column(
    column: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_args: QuantizationArgs,
    global_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Fake-quantize a single weight column.

    Extracted as a module-level function so it can be wrapped by torch.compile.
    All quantization-strategy branching and qparam slicing happens in the eager
    caller, so this kernel always receives channel-shaped ``(scale, zero_point)``
    and a matching ``quant_args``. It delegates to ``fake_quantize`` rather than
    inlining the quant/dequant math, which keeps ``global_scale`` and every
    strategy numerically identical to the eager path.
    """
    return fake_quantize(
        column, scale, zero_point, quant_args, global_scale=global_scale
    )


# Compiled variant of the per-column kernel. The outer GPTQ block/column loop
# stays eager to preserve the sequential error-feedback recurrence and the
# Cholesky inverse (data-dependent control flow).
_quantize_column_compiled = torch.compile(_quantize_column, dynamic=True)


def _quantize_block(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    scale_cols: torch.Tensor,
    zero_point_cols: torch.Tensor,
    quant_args: QuantizationArgs,
    global_scale: torch.Tensor | None,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a whole block's columns in one call (Stage-2 compile target).

    Runs the full per-column GPTQ inner loop — fake_quantize + the sequential
    error-feedback recurrence — so torch.compile can unroll the ``count``
    iterations and fuse across them. The caller precomputes the per-column
    channel-shaped qparams (``scale_cols``/``zero_point_cols``, e.g.
    ``scale[:, g_idx[i1:i2]]`` for GROUP) so no data-dependent g_idx indexing
    happens inside. ``W1`` is mutated in place for the recurrence but is dead
    after this call (the block-error step uses Q1/Err1), so it isn't returned.

    :return: (Q1, Err1, losses1)
    """
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    losses1 = torch.zeros_like(W1)
    for i in range(count):
        w = W1[:, i]
        d = Hinv1[i, i]
        q = fake_quantize(
            w,
            scale_cols[:, i],
            zero_point_cols[:, i],
            quant_args,
            global_scale=global_scale,
        )
        Q1[:, i] = q
        losses1[:, i] = (w - q) ** 2 / d**2
        err1 = (w - q) / d
        w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        W1[:, i:] -= w1_err
        Err1[:, i] = err1
    return Q1, Err1, losses1


# Compiled variant of the whole-inner-loop kernel. dynamic=True because count
# (blocksize) and column dims vary; Dynamo unrolls the `count` loop.
_quantize_block_compiled = torch.compile(_quantize_block, dynamic=True)


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    num_samples += num_added

    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2) * inp
    H += inp.matmul(inp.t())

    return H, num_samples


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian: preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: loss, quantized_weight, scale, zero_point, g_idx
    """
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
        logger.warning(
            "ActivationOrdering.GROUP requires a grouped quantization strategy; "
            "falling back to actorder=None for this module."
        )
        actorder = None

    # handle activation ordering
    if actorder:
        W, H, perm = _apply_activation_ordering(W, H)

    # handle g_idx and activation ordering
    if actorder == ActivationOrdering.GROUP:
        # actually need scale/zp for permuted weight for this format
        observer(W)
        # use identity g_idx (invert permutation later)

    # handle g_idx
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        # mapping from column index to group index
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // divisor

        if actorder == ActivationOrdering.WEIGHT:
            g_idx = g_idx[perm]

    qparams = observer.get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )

    # sparsity mask
    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (
        (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
        if preserve_zeros
        else None
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
    try:
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing GPTQModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    # Select the compiled or eager per-column kernel once. The flag is a global
    # set via set_gptq_compile (mirrors the MSE observer compile path), read here
    # at call time so it isn't threaded through modifier layers.
    quantize_column = (
        _quantize_column_compiled if get_gptq_compile() else _quantize_column
    )

    # Pre-build the per-column quantization args once instead of copying per
    # column inside the loop (which both allocates and breaks the compile graph).
    # GROUP/TENSOR_GROUP quantize each column as a channelwise slice.
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        column_quant_args = copy(quant_args)
        column_quant_args.strategy = QuantizationStrategy.CHANNEL
    else:
        column_quant_args = quant_args

    # Stage-2 path: compile the whole inner loop instead of just fake_quantize.
    # Limited to GROUP/TENSOR_GROUP without sparsity for the first experiment
    # (per-column scale gather is vectorizable; other strategies / preserve_zeros
    # fall through to the proven per-column path).
    block_mode = (
        get_gptq_block_compile()
        and strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)
        and not preserve_zeros
    )
    quantize_block = (
        _quantize_block_compiled if get_gptq_block_compile() else _quantize_block
    )

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]

        if block_mode:
            # precompute per-column channel qparams (vectorized g_idx gather),
            # then run the whole quant + error-feedback loop in one call
            cols = g_idx[i1:i2]
            Q1, Err1, losses1 = quantize_block(
                W1,
                Hinv1,
                scale[:, cols],
                zero_point[:, cols],
                column_quant_args,
                global_scale,
                count,
            )
        else:
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            losses1 = torch.zeros_like(W1)

            if preserve_zeros:
                W1_nz_mask = W_nz_mask[:, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()

                # select the channel-shaped qparams for this column (eager)
                if strategy == QuantizationStrategy.TENSOR:
                    q = quantize_column(
                        q, scale, zero_point, column_quant_args, global_scale
                    )
                elif strategy == QuantizationStrategy.CHANNEL:
                    q = quantize_column(
                        q,
                        scale[:, 0],
                        zero_point[:, 0],
                        column_quant_args,
                        global_scale,
                    )
                elif strategy in (
                    QuantizationStrategy.GROUP,
                    QuantizationStrategy.TENSOR_GROUP,
                ):
                    group_index = g_idx[i1 + i]
                    q = quantize_column(
                        q,
                        scale[:, group_index],
                        zero_point[:, group_index],
                        column_quant_args,
                        global_scale,
                    )
                elif strategy == QuantizationStrategy.BLOCK:
                    block_column_idx = g_idx[i1 + i]
                    q = quantize_column(
                        q.unsqueeze(1),
                        scale[:, block_column_idx : block_column_idx + 1],
                        zero_point[:, block_column_idx : block_column_idx + 1],
                        column_quant_args,
                        global_scale,
                    ).squeeze(1)
                else:
                    raise ValueError(
                        f"Quantization strategy is not supported for GPTQ: {strategy}"
                    )

                # propagate column error
                Q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                if preserve_zeros:
                    W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
                else:
                    W1[:, i:] -= w1_err
                Err1[:, i] = err1

        # propagate block error
        W[:, i1:i2] = Q1
        losses += torch.sum(losses1, 1) / 2

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        if preserve_zeros:
            W[:, i2:] -= w_err * W_nz_mask[:, i2:]
        else:
            W[:, i2:] -= w_err

    if actorder:
        # restore original permutation
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


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute weight and hessian in order of greatest output activations

    :param W: weight to permute
    :param H: hessian used to determine activation ordering
    :return: permuted weight, permuted hessian, permutation map
    """
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
