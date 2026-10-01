import functools
import gc
from typing import Optional, Union

import torch


try:
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa
except Exception:
    pass


try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    # torch_npu registers torch.npu; importing it is required for the NPU device
    import torch_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except Exception:
    HAS_NPU = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_NPU:
        torch.npu.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "npu":
        torch.npu.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_NPU:
        device = torch.device("npu")
    elif HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device


def is_oom_error(error: BaseException) -> bool:
    """
    Return True when the exception represents an out-of-memory error on any supported
    accelerator. Ascend NPU (torch_npu) raises RuntimeError rather than a dedicated class.
    """
    oom_classes = tuple(
        cls
        for cls in (
            getattr(torch.cuda, "OutOfMemoryError", None),
            getattr(getattr(torch, "npu", None), "OutOfMemoryError", None),
        )
        if isinstance(cls, type)
    )
    if oom_classes and isinstance(error, oom_classes):
        return True
    if isinstance(error, RuntimeError):
        return "out of memory" in str(error).lower()
    return False


def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)
