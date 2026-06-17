import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.observers import Observer
from llmcompressor.observers.compile_config import set_gptq_compile


def _build_hessian(module, num_samples=32, seed=0):
    """Accumulate a realistic (PSD) Hessian from random calibration inputs."""
    gen = torch.Generator().manual_seed(seed)
    H = make_empty_hessian(module)
    num_accumulated = 0
    in_features = module.weight.shape[1]
    for _ in range(num_samples):
        inp = torch.randn(8, in_features, generator=gen)
        H, num_accumulated = accumulate_hessian(inp, module, H, num_accumulated)
    return H


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        ("tensor", None),
        ("channel", None),
        ("group", 16),
    ],
)
def test_gptq_quantize_weight_torch_compile(strategy, group_size):
    """quantize_weight must produce the same result through the compiled column
    kernel as through the eager path, with the flag controlling which is used."""
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 32, bias=False)

    quant_args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy=strategy,
        group_size=group_size,
    )
    hessian = _build_hessian(module)

    # eager baseline
    set_gptq_compile(False)
    loss_e, W_e, scale_e, zp_e, gidx_e = quantize_weight(
        module, quant_args, {module: hessian.clone()}, blocksize=16
    )

    # compiled column kernel (same inputs)
    set_gptq_compile(True)
    try:
        loss_c, W_c, scale_c, zp_c, gidx_c = quantize_weight(
            module, quant_args, {module: hessian.clone()}, blocksize=16
        )
    finally:
        # reset global so other tests are not polluted (mirrors test_mse)
        set_gptq_compile(False)

    # GPTQ accumulates float32 error feedback and compile is non-bitwise, so use
    # explicit (loose) tolerances on the weight rather than assert_close defaults.
    torch.testing.assert_close(W_c, W_e, rtol=1e-3, atol=1e-3)
    # qparams are computed eagerly before the loop, so they must match exactly
    torch.testing.assert_close(scale_c, scale_e, rtol=0, atol=0)
    torch.testing.assert_close(zp_c, zp_e, rtol=0, atol=0)
    assert loss_c == pytest.approx(loss_e, rel=1e-3)
    assert (gidx_c is None) == (gidx_e is None)
    if gidx_e is not None:
        torch.testing.assert_close(gidx_c, gidx_e, rtol=0, atol=0)


def test_gptq_quantize_weight_compile_preserves_global_scale():
    """The compiled kernel must preserve ``global_scale`` (dropping it is what
    silently broke an earlier hand-rolled GPTQ-compile attempt) — exercise the
    fp4 ``tensor_group`` path, which is only correct when global_scale is used."""
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 32, bias=False)

    quant_args = QuantizationArgs(
        num_bits=4,
        type="float",
        symmetric=True,
        strategy="tensor_group",
        group_size=16,
    )

    # tensor_group requires a precomputed global scale on the module
    observer = Observer.load_from_registry(
        quant_args.observer if quant_args.observer else "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )
    module.weight_global_scale = observer.get_global_scale(module.weight)
    hessian = _build_hessian(module)

    set_gptq_compile(False)
    loss_e, W_e, *_ = quantize_weight(
        module, quant_args, {module: hessian.clone()}, blocksize=16
    )

    set_gptq_compile(True)
    try:
        loss_c, W_c, *_ = quantize_weight(
            module, quant_args, {module: hessian.clone()}, blocksize=16
        )
    finally:
        set_gptq_compile(False)

    torch.testing.assert_close(W_c, W_e, rtol=1e-3, atol=1e-3)
    assert loss_c == pytest.approx(loss_e, rel=1e-3)
