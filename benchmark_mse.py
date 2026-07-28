"""
Benchmark for MSE observer grid search.

Profiles the grid search that finds optimal min/max ranges for quantization.
The hot path is `_calculate_error` (called once per shrink step: default 20
steps), which runs `calculate_qparams` + `fake_quantize` on the full weight
tensor each time.

Usage:
    python benchmarks/bench_mse_observer.py [--device cuda] [--rows 4096] [--cols 4096]
"""

import argparse
import math
import time

import torch
import triton
import triton.language as tl
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams

from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    observe,
)
from llmcompressor.observers.helpers import flatten_for_calibration

WARMUP = 2
ITERS = 5


# ── Inlined grid search variants ──────────────────────────────────────────────


def grid_search_eager(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Eager grid search — direct port from mse_quant.py."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    total_steps = int(maxshrink * grid)
    no_improve_count = 0

    for i in range(total_steps):
        p = 1 - i / grid
        shrinked_min = min_val * p
        shrinked_max = max_val * p

        candidate_scales, candidate_zero_points = calculate_qparams(
            min_vals=shrinked_min,
            max_vals=shrinked_max,
            quantization_args=args,
            global_scale=None,
        )
        q = fake_quantize(
            observed,
            candidate_scales.unsqueeze(-1),
            candidate_zero_points.unsqueeze(-1),
            token_args,
        ).to(observed.dtype)
        err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
        del q

        improved = err < best_error
        if torch.any(improved):
            best_error[improved] = err[improved]
            best_min_val[improved] = shrinked_min[improved]
            best_max_val[improved] = shrinked_max[improved]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return best_min_val, best_max_val


@torch.compile(dynamic=True)
def _compute_chunk_compiled(
    observed, args, token_args, min_val, max_val, ps, chunk_size, norm,
    best_error, best_min_val, best_max_val,
):
    for j in range(chunk_size):
        shrinked_min = min_val * ps[j]
        shrinked_max = max_val * ps[j]

        candidate_scales, candidate_zero_points = calculate_qparams(
            min_vals=shrinked_min,
            max_vals=shrinked_max,
            quantization_args=args,
            global_scale=None,
        )
        q = fake_quantize(
            observed,
            candidate_scales.unsqueeze(-1),
            candidate_zero_points.unsqueeze(-1),
            token_args,
        ).to(observed.dtype)
        err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
        del q

        improved = err < best_error
        best_error = torch.where(improved, err, best_error)
        best_min_val = torch.where(improved, shrinked_min, best_min_val)
        best_max_val = torch.where(improved, shrinked_max, best_max_val)

    return best_error, best_min_val, best_max_val


def grid_search_compiled(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Compiled grid search — chunked torch.compile inner loop."""
    import torch._dynamo.config
    import torch._dynamo.decorators

    torch._dynamo.config.capture_scalar_outputs = True

    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    total_steps = int(maxshrink * grid)
    no_improve_count = 0

    observed = observed.clone()
    torch._dynamo.decorators.mark_unbacked(observed, observed.ndim - 1)

    idx = 0
    while idx < total_steps:
        chunk_end = min(idx + chunk_size, total_steps)
        current_chunk = chunk_end - idx

        ps = torch.tensor(
            [1.0 - (idx + j) / grid for j in range(current_chunk)],
            dtype=observed.dtype,
            device=observed.device,
        )

        prev_best = best_error.clone()
        best_error, best_min_val, best_max_val = _compute_chunk_compiled(
            observed, args, token_args, min_val, max_val,
            ps, current_chunk, norm, best_error, best_min_val, best_max_val,
        )

        if torch.equal(prev_best, best_error):
            no_improve_count += current_chunk
            if no_improve_count >= patience:
                break
        else:
            no_improve_count = 0
        idx = chunk_end

    return best_min_val, best_max_val


# ── Triton fused implementation ───────────────────────────────────────────────


@triton.jit
def _fused_grid_search_kernel(
    observed_ptr,
    all_scales_ptr,
    all_zps_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    stride_obs_row,
    stride_obs_group,
    stride_scale_step,
    stride_scale_row,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
    norm,
    BLOCK_G: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    best_err = float("inf")
    best_s = 0

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            scale = tl.load(
                all_scales_ptr
                + step * stride_scale_step
                + row * stride_scale_row
                + group
            )
            zp = tl.load(
                all_zps_ptr
                + step * stride_scale_step
                + row * stride_scale_row
                + group
            )

            # fused fake_quantize: round(x/scale + zp), clamp, dequant
            q_float = obs / scale + zp
            q_int = tl.extra.cuda.libdevice.nearbyint(q_float)
            q_int = tl.minimum(tl.maximum(q_int, q_min), q_max)
            q = (q_int - zp) * scale

            diff = tl.abs(q - obs)
            norm_f = norm.to(tl.float32)
            diff_pow = tl.extra.cuda.libdevice.pow(diff, norm_f)
            err = tl.sum(tl.where(g_mask, diff_pow, 0.0), axis=0)

            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, step, best_s)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


def grid_search_triton(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Fused Triton grid search — one kernel per (row, group), all steps in registers."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    # Precompute all candidate scales and zero_points
    all_scales = []
    all_zps = []
    for i in range(total_steps):
        p = 1 - i / grid
        s, zp = calculate_qparams(
            min_vals=min_val * p,
            max_vals=max_val * p,
            quantization_args=args,
            global_scale=None,
        )
        all_scales.append(s)
        all_zps.append(zp)

    all_scales = torch.stack(all_scales).contiguous()  # (total_steps, *qparam_shape)
    all_zps = torch.stack(all_zps).to(dtype=torch.float32).contiguous()

    _, num_rows, num_groups, group_size = observed.shape

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups),
        float("inf"),
        device=observed.device,
        dtype=torch.float32,
    )

    observed_contig = observed.contiguous()

    BLOCK_G = triton.next_power_of_2(group_size)
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    num_programs = num_rows * num_groups
    grid_launch = (num_programs,)

    bit_range = 2**args.num_bits - 1
    if args.symmetric:
        q_min_val = -(2 ** (args.num_bits - 1))
        q_max_val = 2 ** (args.num_bits - 1) - 1
    else:
        q_min_val = 0
        q_max_val = 2**args.num_bits - 1

    _fused_grid_search_kernel[grid_launch](
        observed_contig,
        all_scales,
        all_zps,
        best_step,
        best_error,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        observed_contig.stride(1),
        observed_contig.stride(2),
        all_scales.stride(0),
        all_scales.stride(1),
        float(q_min_val),
        float(q_max_val),
        norm,
        BLOCK_G=BLOCK_G,
        TOTAL_STEPS=TOTAL_STEPS,
    )

    # Reconstruct best_min/max from best step indices
    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step.long()]
    best_min_val = min_val * best_p
    best_max_val = max_val * best_p

    return best_min_val, best_max_val


# ── Triton codebook/cutoff implementation (format-agnostic) ──────────────────


@triton.jit
def _fused_grid_search_cutoff_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    cutoffs_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Format-agnostic grid search using codebook + cutoffs.

    The codebook contains the normalized representable values for the
    quantization format (e.g. integers for INT, irregular values for FP4).
    Cutoffs are midpoints between adjacent codebook entries.
    Shrinking by p just scales both codebook and cutoffs — equivalently,
    we divide observed by (scale_base * p) and bin against the fixed cutoffs.

    Binning uses binary search over sorted cutoffs: O(log n) comparisons
    per element instead of O(n).
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size
    num_cutoffs = num_codes - 1

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            p = (1.0 - step * ig).to(tl.float32)
            eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)

            obs_norm = obs / eff_scale  # [BLOCK_G]

            # Binary search over sorted cutoffs to find bin index
            lo = tl.zeros([BLOCK_G], dtype=tl.int32)
            hi = tl.full([BLOCK_G], num_cutoffs, dtype=tl.int32)
            for _ in range(LOG_C):
                mid = (lo + hi) >> 1
                cut_mid = tl.load(cutoffs_ptr + mid)  # gather [BLOCK_G]
                ge = obs_norm >= cut_mid
                lo = tl.where(ge, mid + 1, lo)
                hi = tl.where(ge, hi, mid)
            bin_idx = lo

            q_norm = tl.load(codes_ptr + bin_idx)  # gather [BLOCK_G]
            q = q_norm * eff_scale

            diff = tl.abs(q - obs).to(tl.float32)
            diff_pow = tl.extra.cuda.libdevice.pow(diff, norm_f)
            err = tl.sum(tl.where(g_mask, diff_pow, 0.0), axis=0)

            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, step, best_s)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


def _build_normalized_codebook(args):
    """Build normalized codebook and cutoffs for any quantization format.

    Returns (codes, cutoffs) where codes are the normalized representable
    values and cutoffs are midpoints between adjacent codes.  For INT these
    are just integers; for FP4/FP8 they would be the format's representable
    values.
    """
    if args.symmetric:
        q_min = -(2 ** (args.num_bits - 1))
        q_max = 2 ** (args.num_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2**args.num_bits - 1

    codes = torch.arange(q_min, q_max + 1, dtype=torch.float32)
    cutoffs = (codes[:-1] + codes[1:]) * 0.5
    return codes, cutoffs


def grid_search_triton_cutoff(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Format-agnostic Triton grid search using codebook + cutoffs."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    # Compute per-group scale at p=1.0
    scale_base, _ = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )

    # Build format-agnostic codebook
    codes, cutoffs = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)
    cutoffs = cutoffs.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups),
        float("inf"),
        device=observed.device,
        dtype=torch.float32,
    )

    import math

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    num_cutoffs = num_codes - 1
    LOG_C = max(1, math.ceil(math.log2(num_cutoffs))) if num_cutoffs > 0 else 1
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    _fused_grid_search_cutoff_kernel[grid_launch](
        observed_contig,
        scale_base,
        codes,
        cutoffs,
        best_step,
        best_error,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        num_codes,
        observed_contig.stride(1),
        observed_contig.stride(2),
        1.0 / grid,
        norm,
        BLOCK_G=BLOCK_G,
        LOG_C=LOG_C,
        TOTAL_STEPS=TOTAL_STEPS,
    )

    # Reconstruct best_min/max from best step indices
    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step.long()]
    best_min_val = min_val * best_p
    best_max_val = max_val * best_p

    return best_min_val, best_max_val


# ── Triton codebook min-distance (format-agnostic, no cutoffs) ───────────────


@triton.jit
def _fused_grid_search_mindist_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Format-agnostic grid search via binary-search nearest in codebook.

    For each element, binary searches the sorted codebook to find the
    insertion point, then compares distances to the two adjacent codes.
    No cutoffs needed — just the codebook itself.
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            p = (1.0 - step * ig).to(tl.float32)
            eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)

            obs_norm = obs / eff_scale  # [BLOCK_G]

            # Binary search for insertion point in sorted codes
            lo = tl.zeros([BLOCK_G], dtype=tl.int32)
            hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
            for _ in range(LOG_C):
                mid = (lo + hi) >> 1
                code_mid = tl.load(codes_ptr + mid)  # gather [BLOCK_G]
                ge = obs_norm >= code_mid
                lo = tl.where(ge, mid + 1, lo)
                hi = tl.where(ge, hi, mid)

            # Nearest is codes[lo-1] or codes[lo], check both
            idx_left = tl.maximum(lo - 1, 0)
            idx_right = tl.minimum(lo, num_codes - 1)
            code_left = tl.load(codes_ptr + idx_left)
            code_right = tl.load(codes_ptr + idx_right)
            dist_left = tl.abs(obs_norm - code_left)
            dist_right = tl.abs(obs_norm - code_right)
            min_dist = tl.minimum(dist_left, dist_right) * eff_scale

            min_dist_pow = tl.extra.cuda.libdevice.pow(
                min_dist.to(tl.float32), norm_f
            )
            err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, step, best_s)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


def grid_search_triton_mindist(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Format-agnostic Triton grid search using codebook min-distance."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    # Compute per-group scale at p=1.0
    scale_base, _ = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )

    # Build format-agnostic codebook (only need codes, not cutoffs)
    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups),
        float("inf"),
        device=observed.device,
        dtype=torch.float32,
    )

    import math

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    _fused_grid_search_mindist_kernel[grid_launch](
        observed_contig,
        scale_base,
        codes,
        best_step,
        best_error,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        num_codes,
        observed_contig.stride(1),
        observed_contig.stride(2),
        1.0 / grid,
        norm,
        BLOCK_G=BLOCK_G,
        LOG_C=LOG_C,
        TOTAL_STEPS=TOTAL_STEPS,
    )

    # Reconstruct best_min/max from best step indices
    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step.long()]
    best_min_val = min_val * best_p
    best_max_val = max_val * best_p

    return best_min_val, best_max_val


# ── Triton incremental codebook search (2-check and 3-check variants) ────────


@triton.jit
def _fused_grid_search_incr2_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Incremental grid search: full binary search at step 0, then check
    2 codes (current + 1 directional neighbor) per subsequent step.

    Since p decreases monotonically, |obs_norm| increases — positive values
    shift right in the codebook, negative values shift left.
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    # Precompute shift direction: +1 for positive obs, -1 for negative
    shift_dir = tl.where(obs >= 0, 1, -1)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0
    bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            p = (1.0 - step * ig).to(tl.float32)
            eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)
            obs_norm = obs / eff_scale

            if step == 0:
                # Full binary search for insertion point
                lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                for _ in range(LOG_C):
                    mid = (lo + hi) >> 1
                    code_mid = tl.load(codes_ptr + mid)
                    ge = obs_norm >= code_mid
                    lo = tl.where(ge, mid + 1, lo)
                    hi = tl.where(ge, hi, mid)

                idx_left = tl.maximum(lo - 1, 0)
                idx_right = tl.minimum(lo, num_codes - 1)
                d_left = tl.abs(obs_norm - tl.load(codes_ptr + idx_left))
                d_right = tl.abs(obs_norm - tl.load(codes_ptr + idx_right))
                left_wins = d_left <= d_right
                bin_idx = tl.where(left_wins, idx_left, idx_right)
                min_dist = tl.minimum(d_left, d_right) * eff_scale
            else:
                # Incremental: check current and 1 neighbor in shift direction
                d_cur = tl.abs(obs_norm - tl.load(codes_ptr + bin_idx))
                shift_idx = tl.maximum(tl.minimum(bin_idx + shift_dir, num_codes - 1), 0)
                d_shift = tl.abs(obs_norm - tl.load(codes_ptr + shift_idx))
                better = d_shift < d_cur
                bin_idx = tl.where(better, shift_idx, bin_idx)
                min_dist = tl.minimum(d_cur, d_shift) * eff_scale

            min_dist_pow = tl.extra.cuda.libdevice.pow(
                min_dist.to(tl.float32), norm_f
            )
            err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, step, best_s)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


@triton.jit
def _fused_grid_search_incr3_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Incremental grid search: full binary search at step 0, then check
    3 codes (current + 2 in shift direction) per subsequent step.

    Since |obs_norm| only increases, the shift is always away from zero:
    positive values shift right, negative shift left.
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    shift_dir = tl.where(obs >= 0, 1, -1)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0
    bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            p = (1.0 - step * ig).to(tl.float32)
            eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)
            obs_norm = obs / eff_scale

            if step == 0:
                lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                for _ in range(LOG_C):
                    mid = (lo + hi) >> 1
                    code_mid = tl.load(codes_ptr + mid)
                    ge = obs_norm >= code_mid
                    lo = tl.where(ge, mid + 1, lo)
                    hi = tl.where(ge, hi, mid)

                idx_left = tl.maximum(lo - 1, 0)
                idx_right = tl.minimum(lo, num_codes - 1)
                d_left = tl.abs(obs_norm - tl.load(codes_ptr + idx_left))
                d_right = tl.abs(obs_norm - tl.load(codes_ptr + idx_right))
                left_wins = d_left <= d_right
                bin_idx = tl.where(left_wins, idx_left, idx_right)
                min_dist = tl.minimum(d_left, d_right) * eff_scale
            else:
                # Check current + 2 neighbors in shift direction
                idx_s1 = tl.maximum(tl.minimum(bin_idx + shift_dir, num_codes - 1), 0)
                idx_s2 = tl.maximum(tl.minimum(bin_idx + shift_dir * 2, num_codes - 1), 0)
                d_cur = tl.abs(obs_norm - tl.load(codes_ptr + bin_idx))
                d_s1 = tl.abs(obs_norm - tl.load(codes_ptr + idx_s1))
                d_s2 = tl.abs(obs_norm - tl.load(codes_ptr + idx_s2))

                new_bin = bin_idx
                new_d = d_cur
                better_s1 = d_s1 < new_d
                new_bin = tl.where(better_s1, idx_s1, new_bin)
                new_d = tl.where(better_s1, d_s1, new_d)
                better_s2 = d_s2 < new_d
                new_bin = tl.where(better_s2, idx_s2, new_bin)
                new_d = tl.where(better_s2, d_s2, new_d)

                bin_idx = new_bin
                min_dist = new_d * eff_scale

            min_dist_pow = tl.extra.cuda.libdevice.pow(
                min_dist.to(tl.float32), norm_f
            )
            err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

            is_better = err < best_err
            best_err = tl.where(is_better, err, best_err)
            best_s = tl.where(is_better, step, best_s)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


@triton.jit
def _fused_grid_search_incr2_patience_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    patience,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Incremental 2-check with per-program early stopping."""
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    shift_dir = tl.where(obs >= 0, 1, -1)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0
    bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)
    patience_ctr = 0

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            if patience_ctr < patience:
                p = (1.0 - step * ig).to(tl.float32)
                eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)
                obs_norm = obs / eff_scale

                if step == 0:
                    lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                    hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                    for _ in range(LOG_C):
                        mid = (lo + hi) >> 1
                        code_mid = tl.load(codes_ptr + mid)
                        ge = obs_norm >= code_mid
                        lo = tl.where(ge, mid + 1, lo)
                        hi = tl.where(ge, hi, mid)

                    idx_left = tl.maximum(lo - 1, 0)
                    idx_right = tl.minimum(lo, num_codes - 1)
                    d_left = tl.abs(obs_norm - tl.load(codes_ptr + idx_left))
                    d_right = tl.abs(obs_norm - tl.load(codes_ptr + idx_right))
                    left_wins = d_left <= d_right
                    bin_idx = tl.where(left_wins, idx_left, idx_right)
                    min_dist = tl.minimum(d_left, d_right) * eff_scale
                else:
                    d_cur = tl.abs(obs_norm - tl.load(codes_ptr + bin_idx))
                    shift_idx = tl.maximum(
                        tl.minimum(bin_idx + shift_dir, num_codes - 1), 0
                    )
                    d_shift = tl.abs(obs_norm - tl.load(codes_ptr + shift_idx))
                    better = d_shift < d_cur
                    bin_idx = tl.where(better, shift_idx, bin_idx)
                    min_dist = tl.minimum(d_cur, d_shift) * eff_scale

                min_dist_pow = tl.extra.cuda.libdevice.pow(
                    min_dist.to(tl.float32), norm_f
                )
                err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

                is_better = err < best_err
                best_err = tl.where(is_better, err, best_err)
                best_s = tl.where(is_better, step, best_s)
                patience_ctr = tl.where(is_better, 0, patience_ctr + 1)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


def grid_search_triton_incr2p(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Incremental 2-check with per-program patience-based early stopping."""
    import math

    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    scale_base, _ = calculate_qparams(
        min_vals=min_val, max_vals=max_val,
        quantization_args=args, global_scale=None,
    )

    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups), float("inf"),
        device=observed.device, dtype=torch.float32,
    )

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    _fused_grid_search_incr2_patience_kernel[grid_launch](
        observed_contig, scale_base, codes,
        best_step, best_error,
        num_rows, num_groups, group_size, total_steps, num_codes,
        observed_contig.stride(1), observed_contig.stride(2),
        1.0 / grid, norm, patience,
        BLOCK_G=BLOCK_G, LOG_C=LOG_C, TOTAL_STEPS=TOTAL_STEPS,
    )

    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype, device=min_val.device,
    )
    best_p = ps[best_step.long()]
    return min_val * best_p, max_val * best_p


@triton.jit
def _fused_grid_search_chunked_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    NUM_NEIGHBORS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Chunked incremental search: process CHUNK_SIZE steps in an unrolled
    inner loop, check patience between chunks. Stops after one full chunk
    with zero improvement (equivalent to patience=CHUNK_SIZE per-step).
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    shift_dir = tl.where(obs >= 0, 1, -1)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0
    bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)
    converged = 0

    for chunk_start in range(0, TOTAL_STEPS, CHUNK_SIZE):
        if converged == 0:
            chunk_improved = 0

            for local in tl.static_range(0, CHUNK_SIZE):
                step = chunk_start + local
                if step < total_steps:
                    p = (1.0 - step * ig).to(tl.float32)
                    eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)
                    obs_norm = obs / eff_scale

                    if step == 0:
                        lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                        hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                        for _ in range(LOG_C):
                            mid = (lo + hi) >> 1
                            code_mid = tl.load(codes_ptr + mid)
                            ge = obs_norm >= code_mid
                            lo = tl.where(ge, mid + 1, lo)
                            hi = tl.where(ge, hi, mid)

                        idx_left = tl.maximum(lo - 1, 0)
                        idx_right = tl.minimum(lo, num_codes - 1)
                        d_left = tl.abs(obs_norm - tl.load(codes_ptr + idx_left))
                        d_right = tl.abs(obs_norm - tl.load(codes_ptr + idx_right))
                        left_wins = d_left <= d_right
                        bin_idx = tl.where(left_wins, idx_left, idx_right)
                        min_dist = tl.minimum(d_left, d_right) * eff_scale
                    else:
                        new_bin = bin_idx
                        new_d = tl.abs(obs_norm - tl.load(codes_ptr + bin_idx))
                        for k in tl.static_range(1, NUM_NEIGHBORS + 1):
                            idx_k = tl.maximum(
                                tl.minimum(bin_idx + shift_dir * k, num_codes - 1), 0
                            )
                            d_k = tl.abs(obs_norm - tl.load(codes_ptr + idx_k))
                            better_k = d_k < new_d
                            new_bin = tl.where(better_k, idx_k, new_bin)
                            new_d = tl.where(better_k, d_k, new_d)

                        bin_idx = new_bin
                        min_dist = new_d * eff_scale

                    min_dist_pow = tl.extra.cuda.libdevice.pow(
                        min_dist.to(tl.float32), norm_f
                    )
                    err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

                    is_better = err < best_err
                    best_err = tl.where(is_better, err, best_err)
                    best_s = tl.where(is_better, step, best_s)
                    chunk_improved = tl.where(is_better, 1, chunk_improved)

            # After chunk: if no step improved, converge
            converged = tl.where(chunk_improved > 0, 0, 1)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)


def _launch_chunked(observed, args, maxshrink, grid, norm, num_neighbors, chunk_size):
    import math

    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    scale_base, _ = calculate_qparams(
        min_vals=min_val, max_vals=max_val,
        quantization_args=args, global_scale=None,
    )

    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups), float("inf"),
        device=observed.device, dtype=torch.float32,
    )

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    _fused_grid_search_chunked_kernel[grid_launch](
        observed_contig, scale_base, codes,
        best_step, best_error,
        num_rows, num_groups, group_size, total_steps, num_codes,
        observed_contig.stride(1), observed_contig.stride(2),
        1.0 / grid, norm,
        BLOCK_G=BLOCK_G, LOG_C=LOG_C, TOTAL_STEPS=TOTAL_STEPS,
        NUM_NEIGHBORS=num_neighbors, CHUNK_SIZE=chunk_size,
    )

    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype, device=min_val.device,
    )
    best_p = ps[best_step.long()]
    return min_val * best_p, max_val * best_p


def grid_search_triton_chunk5(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_chunked(observed, args, maxshrink, grid, norm, 1, 5)


def grid_search_triton_chunk10(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_chunked(observed, args, maxshrink, grid, norm, 1, 10)


def grid_search_triton_chunk20(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_chunked(observed, args, maxshrink, grid, norm, 1, 20)


@triton.jit
def _fused_grid_search_incrN_patience_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    steps_run_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    patience,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    NUM_NEIGHBORS: tl.constexpr,
    RECORD_STEPS: tl.constexpr,
):
    """Incremental search with variable neighbor count and early stopping.

    NUM_NEIGHBORS controls how many codes to check in the shift direction
    after the initial binary search at step 0.

    RECORD_STEPS is a compile-time flag; when False the steps_run store is
    eliminated at compile time, so the timing path generates the same code
    as before this instrumentation was added.
    """
    pid = tl.program_id(0)
    row = pid // num_groups
    group = pid % num_groups

    if row >= num_rows:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    obs = tl.load(
        observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
        mask=g_mask,
        other=0.0,
    )

    scale_base = tl.load(scale_base_ptr + row * num_groups + group)

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    shift_dir = tl.where(obs >= 0, 1, -1)

    best_err = tl.full([], float("inf"), dtype=tl.float32)
    best_s = 0
    bin_idx = tl.zeros([BLOCK_G], dtype=tl.int32)
    patience_ctr = 0
    steps_run = 0

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            if patience_ctr < patience:
                steps_run += 1
                p = (1.0 - step * ig).to(tl.float32)
                eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)
                obs_norm = obs / eff_scale

                if step == 0:
                    lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                    hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                    for _ in range(LOG_C):
                        mid = (lo + hi) >> 1
                        code_mid = tl.load(codes_ptr + mid)
                        ge = obs_norm >= code_mid
                        lo = tl.where(ge, mid + 1, lo)
                        hi = tl.where(ge, hi, mid)

                    idx_left = tl.maximum(lo - 1, 0)
                    idx_right = tl.minimum(lo, num_codes - 1)
                    d_left = tl.abs(obs_norm - tl.load(codes_ptr + idx_left))
                    d_right = tl.abs(obs_norm - tl.load(codes_ptr + idx_right))
                    left_wins = d_left <= d_right
                    bin_idx = tl.where(left_wins, idx_left, idx_right)
                    min_dist = tl.minimum(d_left, d_right) * eff_scale
                else:
                    # Check current + NUM_NEIGHBORS in shift direction
                    new_bin = bin_idx
                    new_d = tl.abs(obs_norm - tl.load(codes_ptr + bin_idx))
                    for k in tl.static_range(1, NUM_NEIGHBORS + 1):
                        idx_k = tl.maximum(
                            tl.minimum(bin_idx + shift_dir * k, num_codes - 1), 0
                        )
                        d_k = tl.abs(obs_norm - tl.load(codes_ptr + idx_k))
                        better_k = d_k < new_d
                        new_bin = tl.where(better_k, idx_k, new_bin)
                        new_d = tl.where(better_k, d_k, new_d)

                    bin_idx = new_bin
                    min_dist = new_d * eff_scale

                min_dist_pow = tl.extra.cuda.libdevice.pow(
                    min_dist.to(tl.float32), norm_f
                )
                err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

                is_better = err < best_err
                best_err = tl.where(is_better, err, best_err)
                best_s = tl.where(is_better, step, best_s)
                patience_ctr = tl.where(is_better, 0, patience_ctr + 1)

    tl.store(best_step_ptr + row * num_groups + group, best_s)
    tl.store(best_error_ptr + row * num_groups + group, best_err)
    if RECORD_STEPS:
        tl.store(steps_run_ptr + row * num_groups + group, steps_run)


def _launch_incrN_patience(
    observed, args, maxshrink, patience, grid, norm, num_neighbors,
    record_steps=False,
):
    """Shared launcher for incrN patience kernel with variable neighbor count.

    ``record_steps=True`` additionally returns the per-program count of grid
    steps actually executed before early stopping. The store is guarded by a
    ``tl.constexpr`` so the timing path (record_steps=False) compiles to the
    same code as without the instrumentation.
    """
    import math

    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    scale_base, _ = calculate_qparams(
        min_vals=min_val, max_vals=max_val,
        quantization_args=args, global_scale=None,
    )

    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups), float("inf"),
        device=observed.device, dtype=torch.float32,
    )

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    steps_run = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )

    _fused_grid_search_incrN_patience_kernel[grid_launch](
        observed_contig, scale_base, codes,
        best_step, best_error, steps_run,
        num_rows, num_groups, group_size, total_steps, num_codes,
        observed_contig.stride(1), observed_contig.stride(2),
        1.0 / grid, norm, patience,
        BLOCK_G=BLOCK_G, LOG_C=LOG_C, TOTAL_STEPS=TOTAL_STEPS,
        NUM_NEIGHBORS=num_neighbors, RECORD_STEPS=record_steps,
    )

    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype, device=min_val.device,
    )
    best_p = ps[best_step.long()]
    bmin, bmax = min_val * best_p, max_val * best_p
    if record_steps:
        return bmin, bmax, steps_run
    return bmin, bmax


def grid_search_triton_incrNp1(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_incrN_patience(observed, args, maxshrink, patience, grid, norm, 1)


def grid_search_triton_incrNp2(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_incrN_patience(observed, args, maxshrink, patience, grid, norm, 2)


def grid_search_triton_incrNp3(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_incrN_patience(observed, args, maxshrink, patience, grid, norm, 3)


def grid_search_triton_incrNp4(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    return _launch_incrN_patience(observed, args, maxshrink, patience, grid, norm, 4)


def _launch_incremental_kernel(kernel, observed, args, maxshrink, grid, norm):
    """Shared launcher for incremental kernels."""
    import math

    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    scale_base, _ = calculate_qparams(
        min_vals=min_val, max_vals=max_val,
        quantization_args=args, global_scale=None,
    )

    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups), float("inf"),
        device=observed.device, dtype=torch.float32,
    )

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)
    grid_launch = (num_rows * num_groups,)

    kernel[grid_launch](
        observed_contig, scale_base, codes,
        best_step, best_error,
        num_rows, num_groups, group_size, total_steps, num_codes,
        observed_contig.stride(1), observed_contig.stride(2),
        1.0 / grid, norm,
        BLOCK_G=BLOCK_G, LOG_C=LOG_C, TOTAL_STEPS=TOTAL_STEPS,
    )

    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype, device=min_val.device,
    )
    best_p = ps[best_step.long()]
    return min_val * best_p, max_val * best_p


def grid_search_triton_incr2(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Incremental search: binary search at step 0, 2 checks after."""
    return _launch_incremental_kernel(
        _fused_grid_search_incr2_kernel, observed, args, maxshrink, grid, norm
    )


def grid_search_triton_incr3(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Incremental search: binary search at step 0, 3 checks after."""
    return _launch_incremental_kernel(
        _fused_grid_search_incr3_kernel, observed, args, maxshrink, grid, norm
    )


# ── Triton multi-group codebook min-distance ─────────────────────────────────


@triton.jit
def _fused_grid_search_multigroup_kernel(
    observed_ptr,
    scale_base_ptr,
    codes_ptr,
    best_step_ptr,
    best_error_ptr,
    num_rows,
    num_groups,
    group_size,
    total_steps,
    num_codes,
    stride_obs_row,
    stride_obs_group,
    inv_grid,
    norm,
    total_work,
    BLOCK_G: tl.constexpr,
    LOG_C: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    GROUPS_PER_PID: tl.constexpr,
):
    """Same as mindist but each program handles multiple groups."""
    pid = tl.program_id(0)
    base_work = pid * GROUPS_PER_PID

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < group_size

    ig = inv_grid.to(tl.float32)
    norm_f = norm.to(tl.float32)

    for g_off in range(GROUPS_PER_PID):
        work_idx = base_work + g_off
        valid = work_idx < total_work

        if valid:
            row = work_idx // num_groups
            group = work_idx % num_groups

            obs = tl.load(
                observed_ptr + row * stride_obs_row + group * stride_obs_group + g_idx,
                mask=g_mask,
                other=0.0,
            )

            scale_base = tl.load(scale_base_ptr + row * num_groups + group)

            best_err = tl.full([], float("inf"), dtype=tl.float32)
            best_s = 0

            for step in range(TOTAL_STEPS):
                if step < total_steps:
                    p = (1.0 - step * ig).to(tl.float32)
                    eff_scale = tl.maximum(scale_base * p, 1e-38).to(tl.float32)

                    obs_norm = obs / eff_scale

                    lo = tl.zeros([BLOCK_G], dtype=tl.int32)
                    hi = tl.full([BLOCK_G], num_codes, dtype=tl.int32)
                    for _ in range(LOG_C):
                        mid = (lo + hi) >> 1
                        code_mid = tl.load(codes_ptr + mid)
                        ge = obs_norm >= code_mid
                        lo = tl.where(ge, mid + 1, lo)
                        hi = tl.where(ge, hi, mid)

                    idx_left = tl.maximum(lo - 1, 0)
                    idx_right = tl.minimum(lo, num_codes - 1)
                    code_left = tl.load(codes_ptr + idx_left)
                    code_right = tl.load(codes_ptr + idx_right)
                    dist_left = tl.abs(obs_norm - code_left)
                    dist_right = tl.abs(obs_norm - code_right)
                    min_dist = tl.minimum(dist_left, dist_right) * eff_scale

                    min_dist_pow = tl.extra.cuda.libdevice.pow(
                        min_dist.to(tl.float32), norm_f
                    )
                    err = tl.sum(tl.where(g_mask, min_dist_pow, 0.0), axis=0)

                    is_better = err < best_err
                    best_err = tl.where(is_better, err, best_err)
                    best_s = tl.where(is_better, step, best_s)

            tl.store(best_step_ptr + row * num_groups + group, best_s)
            tl.store(best_error_ptr + row * num_groups + group, best_err)


def grid_search_triton_multigroup(
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
):
    """Format-agnostic Triton grid search — multiple groups per program."""
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    scale_base, _ = calculate_qparams(
        min_vals=min_val,
        max_vals=max_val,
        quantization_args=args,
        global_scale=None,
    )

    codes, _ = _build_normalized_codebook(args)
    codes = codes.to(device=observed.device)

    _, num_rows, num_groups, group_size = observed.shape
    observed_contig = observed.contiguous()
    scale_base = scale_base.contiguous()

    best_step = torch.zeros(
        num_rows, num_groups, dtype=torch.int32, device=observed.device
    )
    best_error = torch.full(
        (num_rows, num_groups),
        float("inf"),
        device=observed.device,
        dtype=torch.float32,
    )

    import math

    num_codes = codes.shape[0]
    BLOCK_G = triton.next_power_of_2(group_size)
    LOG_C = max(1, math.ceil(math.log2(num_codes)))
    TOTAL_STEPS = triton.next_power_of_2(total_steps)

    total_work = num_rows * num_groups
    GROUPS_PER_PID = 4
    num_pids = math.ceil(total_work / GROUPS_PER_PID)
    grid_launch = (num_pids,)

    _fused_grid_search_multigroup_kernel[grid_launch](
        observed_contig,
        scale_base,
        codes,
        best_step,
        best_error,
        num_rows,
        num_groups,
        group_size,
        total_steps,
        num_codes,
        observed_contig.stride(1),
        observed_contig.stride(2),
        1.0 / grid,
        norm,
        total_work,
        BLOCK_G=BLOCK_G,
        LOG_C=LOG_C,
        TOTAL_STEPS=TOTAL_STEPS,
        GROUPS_PER_PID=GROUPS_PER_PID,
    )

    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    best_p = ps[best_step.long()]
    best_min_val = min_val * best_p
    best_max_val = max_val * best_p

    return best_min_val, best_max_val


# ── Quality diagnostics ──────────────────────────────────────────────────────
#
# Everything below is only used by --quality. It never runs inside the timed
# loop, so latency measurements are unaffected.


def exact_error(observed, args, token_args, min_v, max_v, norm):
    """Per-group quantization error for a candidate min/max range.

    Recomputed through the real ``calculate_qparams`` + ``fake_quantize``
    path (identical to ``_calculate_error`` in mse_quant.py), so it is
    comparable across every variant regardless of how that variant
    internally approximates the quantization.
    """
    scales, zps = calculate_qparams(
        min_vals=min_v,
        max_vals=max_v,
        quantization_args=args,
        global_scale=None,
    )
    q = fake_quantize(
        observed, scales.unsqueeze(-1), zps.unsqueeze(-1), token_args
    ).to(observed.dtype)
    return torch.sum((q - observed).abs().pow(norm), dim=(0, -1))


def build_oracle(observed, args, token_args, maxshrink, grid, norm):
    """Eager full-grid search with NO early stopping — the ground truth.

    Evaluates every shrink step through the exact error path and keeps the
    full error tensor so ties can be detected.

    :return: dict with per-step errors, the per-group minimum, its argmin,
        the unshrunk min/max, and the shrink factors.
    """
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    total_steps = int(maxshrink * grid)

    errs = []
    for i in range(total_steps):
        p = 1 - i / grid
        errs.append(
            exact_error(observed, args, token_args, min_val * p, max_val * p, norm)
        )
    errs = torch.stack(errs)  # (steps, *qparams_shape)

    best_error, best_step = errs.min(dim=0)
    ps = torch.tensor(
        [1.0 - i / grid for i in range(total_steps)],
        dtype=min_val.dtype,
        device=min_val.device,
    )
    return {
        "errs": errs,
        "best_error": best_error,
        "best_step": best_step,
        "min_val": min_val,
        "max_val": max_val,
        "ps": ps,
        "total_steps": total_steps,
    }


def derive_num_buckets(args, grid, maxshrink):
    """Number of codebook buckets an element can shift by in one grid step.

    As ``p`` shrinks, ``obs_norm = obs / (scale * p) + zp`` moves away from
    ``zp`` by a factor ``1/p`` per step, so the largest index shift is

        N = ceil( R / (grid * p_min) ),  R = max(|q_min - zp|, |q_max - zp|)

    with ``p_min = 1 - maxshrink`` as the conservative bound. This is a
    *uniform-integer* bound: it assumes evenly spaced codes. Irregular
    codebooks (FP4/FP8) must derive N from their actual cutoff spacing
    instead.

    :return: (N, detail-string) or (None, reason) when unsupported.
    """
    q_type = str(getattr(args, "type", "int")).lower()
    if "int" not in q_type:
        return None, (
            f"non-integer type ({q_type}): irregular codebook, "
            "derive N from actual cutoffs"
        )
    if not args.symmetric:
        # zero_point is data dependent and the scratch codebook path does not
        # apply it at all, so asymmetric is not supported here yet.
        return None, "asymmetric: codebook path ignores zero_point, unsupported"

    zp = 0
    q_min = -(2 ** (args.num_bits - 1))
    q_max = 2 ** (args.num_bits - 1) - 1
    R = max(abs(q_min - zp), abs(q_max - zp))
    p_min = 1.0 - maxshrink
    n = math.ceil(R / (grid * p_min))
    return n, f"R={R} q=[{q_min},{q_max}] zp={zp} grid={grid:g} p_min={p_min:.2f}"


def analyze(bmin, bmax, observed, args, token_args, norm, oracle, rtol=1e-6):
    """Compare a variant's chosen range against the oracle.

    ``regret`` is the primary metric: how much worse the variant's *actual*
    quantization error is than the best achievable on the grid. Step-index
    agreement is reported separately and is only informational, since
    several steps can produce identical error.
    """
    v_err = exact_error(observed, args, token_args, bmin, bmax, norm)
    best = oracle["best_error"]

    denom = best.abs().clamp_min(torch.finfo(best.dtype).tiny)
    rel = (v_err - best) / denom
    worse = rel > rtol

    # informational: did it land on the same step index the oracle picked?
    min_val, max_val, ps = oracle["min_val"], oracle["max_val"], oracle["ps"]
    use_min = min_val.abs() >= max_val.abs()
    denom_p = torch.where(use_min, min_val, max_val)
    numer_p = torch.where(use_min, bmin, bmax)
    ok = denom_p.abs() > 0
    safe_denom = torch.where(ok, denom_p, torch.ones_like(denom_p))
    p_hat = torch.where(ok, numer_p / safe_denom, torch.ones_like(denom_p))
    shape = (-1,) + (1,) * p_hat.ndim
    chosen = (ps.reshape(shape) - p_hat.unsqueeze(0)).abs().argmin(0)
    same_step = (chosen == oracle["best_step"]) & ok

    return {
        "mean_rel": rel.clamp_min(0).mean().item(),
        "max_rel": rel.max().item(),
        "worse_pct": worse.float().mean().item() * 100.0,
        "match_pct": (~worse).float().mean().item() * 100.0,
        "same_step_pct": same_step.float().mean().item() * 100.0,
    }


_QUALITY_HDR = (
    f"{'variant':>22} {'mean_regret':>12} {'max_regret':>12} "
    f"{'worse%':>8} {'match%':>8} {'same_step%':>11} {'steps med/p95/max':>18}"
)


def _fmt_quality(name, m, steps=None):
    if steps is None:
        s = "-"
    else:
        f = steps.float()
        s = (
            f"{f.median().item():.0f}/"
            f"{f.quantile(0.95).item():.0f}/"
            f"{f.max().item():.0f}"
        )
    return (
        f"{name:>22} {m['mean_rel']:>12.3e} {m['max_rel']:>12.3e} "
        f"{m['worse_pct']:>8.2f} {m['match_pct']:>8.2f} "
        f"{m['same_step_pct']:>11.2f} {s:>18}"
    )


def run_quality(observed, args, token_args, maxshrink, patience, grid, norm,
                chunk_size, n_override, patience_sweep):
    """Quality-only pass: no timing, so diagnostics cannot skew latency."""
    total_steps = int(maxshrink * grid)

    print("Building oracle (eager full grid, early stopping disabled) ...")
    oracle = build_oracle(observed, args, token_args, maxshrink, grid, norm)
    ties = (
        (oracle["errs"] <= oracle["best_error"].unsqueeze(0) * (1 + 1e-6))
        .sum(0)
        .float()
    )
    print(
        f"  steps={total_steps}  groups={oracle['best_error'].numel()}  "
        f"tied-with-best steps: mean {ties.mean().item():.2f}, "
        f"max {ties.max().item():.0f}"
    )

    n_derived, detail = derive_num_buckets(args, grid, maxshrink)
    if n_derived is None:
        print(f"\nderived N: unsupported — {detail}")
    else:
        print(f"\nderived N = {n_derived}   ({detail})")
    n_sel = n_override if n_override is not None else (n_derived or 1)
    print(f"using N = {n_sel}" + (" (--n-buckets override)" if n_override else ""))

    def _an(bmin, bmax):
        return analyze(bmin, bmax, observed, args, token_args, norm, oracle)

    # Phase 0 — full-grid variants. These do no bucket approximation and no
    # early stopping, so any regret here comes from the codebook /
    # scale_base*p linearisation itself.
    print("\n=== phase 0: full-grid (isolates codebook + linearised scale) ===")
    print(_QUALITY_HDR)
    for name, fn in [
        ("eager(+patience)", grid_search_eager),
        ("compiled", grid_search_compiled),
        ("triton", grid_search_triton),
        ("triton_cutoff", grid_search_triton_cutoff),
        ("triton_mindist", grid_search_triton_mindist),
        ("triton_multigrp", grid_search_triton_multigroup),
    ]:
        bmin, bmax = fn(
            observed, args, token_args, maxshrink, patience, grid, norm, chunk_size
        )
        print(_fmt_quality(name, _an(bmin, bmax)))

    # Phase A — bucket approximation only. Patience is disabled by setting it
    # above the step count so the gate can never trigger.
    print("\n=== phase A: bucket approximation, patience OFF ===")
    print(_QUALITY_HDR)
    no_patience = total_steps + 1
    for n in (1, 2, 3, 4):
        bmin, bmax, steps = _launch_incrN_patience(
            observed, args, maxshrink, no_patience, grid, norm, n, record_steps=True
        )
        print(_fmt_quality(f"N={n} patience=off", _an(bmin, bmax), steps))

    # Phase B — early stopping only, at the selected N.
    print(f"\n=== phase B: patience sweep at N={n_sel} ===")
    print(_QUALITY_HDR)
    for pat in [no_patience] + list(patience_sweep):
        bmin, bmax, steps = _launch_incrN_patience(
            observed, args, maxshrink, pat, grid, norm, n_sel, record_steps=True
        )
        label = "off" if pat == no_patience else str(pat)
        print(_fmt_quality(f"N={n_sel} patience={label}", _an(bmin, bmax), steps))

    print(
        "\nregret = (variant error - oracle best error) / oracle best error, "
        "recomputed via calculate_qparams + fake_quantize."
        "\nmatch% counts groups within rtol of the oracle best (tie-aware); "
        "same_step% is informational only."
    )


# ── Benchmark infrastructure ─────────────────────────────────────────────────


def make_observer_inputs(rows, cols, strategy, group_size, num_bits, device):
    """Create a weight tensor and flatten it for calibration."""
    torch.manual_seed(42)
    module = torch.nn.Linear(cols, rows, bias=False, device=device)

    if strategy == "group":
        quant_args = QuantizationArgs(
            num_bits=num_bits,
            symmetric=True,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
        )
    elif strategy == "channel":
        quant_args = QuantizationArgs(
            num_bits=num_bits,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL,
        )
    elif strategy == "tensor":
        quant_args = QuantizationArgs(
            num_bits=num_bits,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    token_args = quant_args.model_copy(
        update={"strategy": QuantizationStrategy.TOKEN}
    )

    observed = flatten_for_calibration(module.weight, "weight", quant_args)

    return observed, quant_args, token_args


def time_fn(fn, inputs, warmup, iters):
    """Time a grid search function."""
    observed, args, token_args, maxshrink, patience, grid, norm, chunk_size = inputs
    is_cuda = observed.is_cuda

    for _ in range(warmup):
        fn(observed, args, token_args, maxshrink, patience, grid, norm, chunk_size)
        if is_cuda:
            torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()

    times = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(observed, args, token_args, maxshrink, patience, grid, norm, chunk_size)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_delta_mb = 0.0
    if is_cuda:
        peak_delta_mb = (
            torch.cuda.max_memory_allocated() - mem_before
        ) / (1024 * 1024)

    return times, peak_delta_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark MSE observer grid search")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--rows", type=int, default=4096*2)
    parser.add_argument("--cols", type=int, default=4096*2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument(
        "--strategy", default="group", choices=["group", "channel", "tensor"]
    )
    parser.add_argument("--maxshrink", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grid", type=float, default=100.0)
    parser.add_argument("--norm", type=float, default=2.4)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--quality",
        action="store_true",
        help="run accuracy diagnostics against an eager full-grid oracle "
        "instead of timing (the two never run together)",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=None,
        help="override the derived number of buckets checked per step",
    )
    parser.add_argument(
        "--patience-sweep",
        type=int,
        nargs="+",
        default=[15, 10, 5],
        help="patience values to sweep in --quality phase B",
    )
    args = parser.parse_args()

    total_steps = int(args.maxshrink * args.grid)

    print(f"Device:      {args.device}")
    print(f"Weight:      ({args.rows}, {args.cols})")
    print(f"Strategy:    {args.strategy}")
    print(f"Group size:  {args.group_size}")
    print(f"Bits:        {args.num_bits}")
    print(f"Maxshrink:   {args.maxshrink}")
    print(f"Grid:        {args.grid}")
    print(f"Total steps: {total_steps}")
    print(f"Patience:    {args.patience}")
    print(f"Norm:        {args.norm}")
    print(f"Chunk size:  {args.chunk_size}")
    print(f"Warmup:      {args.warmup}  Iters: {args.iters}")
    print()

    observed, quant_args, token_args = make_observer_inputs(
        args.rows, args.cols, args.strategy, args.group_size, args.num_bits, args.device
    )
    print(f"Observed shape: {observed.shape}")
    print()

    if args.quality:
        run_quality(
            observed, quant_args, token_args,
            args.maxshrink, args.patience, args.grid, args.norm,
            args.chunk_size, args.n_buckets, args.patience_sweep,
        )
        return

    inputs = (
        observed, quant_args, token_args,
        args.maxshrink, args.patience, args.grid, args.norm, args.chunk_size,
    )

    variants = [
        ("eager", grid_search_eager),
        ("compiled", grid_search_compiled),
        ("triton", grid_search_triton),
        ("triton_cutoff", grid_search_triton_cutoff),
        ("triton_mindist", grid_search_triton_mindist),
        ("triton_multigrp", grid_search_triton_multigroup),
        ("triton_incr2", grid_search_triton_incr2),
        ("triton_incr2p", grid_search_triton_incr2p),
        ("triton_incr3", grid_search_triton_incr3),
        ("triton_incrNp1", grid_search_triton_incrNp1),
        ("triton_incrNp2", grid_search_triton_incrNp2),
        ("triton_incrNp3", grid_search_triton_incrNp3),
        ("triton_incrNp4", grid_search_triton_incrNp4),
        ("triton_chunk5", grid_search_triton_chunk5),
        ("triton_chunk10", grid_search_triton_chunk10),
        ("triton_chunk20", grid_search_triton_chunk20),
    ]

    results = {}
    for name, fn in variants:
        print(f"Running {name} ...")
        times, peak_mb = time_fn(fn, inputs, args.warmup, args.iters)
        results[name] = {"times": times, "peak_mb": peak_mb}

    medians = {n: sorted(r["times"])[len(r["times"]) // 2] for n, r in results.items()}

    print()
    print(f"{'':>16} {'median':>10} {'min':>10} {'max':>10} {'peak_mem':>10}")
    for name in results:
        t = results[name]["times"]
        med = medians[name]
        peak = results[name]["peak_mb"]
        mem_str = f"{peak:.1f}MB" if peak > 0 else "n/a"
        print(
            f"{name:>16} {med:>10.4f}s "
            f"{min(t):>10.4f}s {max(t):>10.4f}s "
            f"{mem_str:>10}"
        )

    if len(results) > 1:
        print()
        base = medians["eager"]
        for name in list(results)[1:]:
            if medians[name] > 0:
                print(f"Speedup {name} vs eager: {base / medians[name]:.2f}x")


if __name__ == "__main__":
    with torch.no_grad():
        main()
