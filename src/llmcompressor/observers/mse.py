import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch import distributed as dist

from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import lerp
from llmcompressor.observers.mse_quant import _grid_search_mse

__all__ = ["MovingAverageMSEObserver"]


@Observer.register("memoryless_mse")
class MemorylessMSEObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared

    ```psuedocode
    mse_quant_error := mean((x - fake_quant(x))**2)
    global_scale <- min[min_vals, max_vals, global_scale](mse_quant_error(x))
    scale, zp <- min[min_vals, max_vals](mse_quant_error(x, global_scale))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        patience: number of consecutive search steps without improvement before
            early stopping\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.chunk_size = observer_kwargs.get("chunk_size", 5)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.chunk_size,
        )


@Observer.register("mse")
class MovingAverageMSEObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared, with moving average smoothing.

    ```psuedocode
    mse_quant_error := mean((x - fake_quant(x))**2)
    global_scale <- min[min_vals, max_vals, global_scale](mse_quant_error(x))
    scale, zp <- min[min_vals, max_vals](mse_quant_error(x, global_scale))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        patience: number of consecutive search steps without improvement before
            early stopping\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.chunk_size = observer_kwargs.get("chunk_size", 5)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.chunk_size,
        )

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals
