_enable_torch_compile = False


def set_torch_compile(enabled: bool):
    global _enable_torch_compile
    _enable_torch_compile = enabled


def get_torch_compile() -> bool:
    return _enable_torch_compile
