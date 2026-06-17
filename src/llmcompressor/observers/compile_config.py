"""
Global configuration for torch.compile support in calibration paths.

The compile flags are set by the oneshot entrypoint and read at call time by
the consuming code (observer instances, the GPTQ quantize routine). This avoids
threading the flags through recipe and modifier layers.
"""

_enable_observer_compile: bool = False
_enable_gptq_compile: bool = False


def set_observer_compile(enabled: bool) -> None:
    global _enable_observer_compile
    _enable_observer_compile = enabled


def get_observer_compile() -> bool:
    return _enable_observer_compile


def set_gptq_compile(enabled: bool) -> None:
    global _enable_gptq_compile
    _enable_gptq_compile = enabled


def get_gptq_compile() -> bool:
    return _enable_gptq_compile
