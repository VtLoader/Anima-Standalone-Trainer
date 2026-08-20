import os
import platform

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def default_arch_list() -> str:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if os.environ.get("ANIMA_CUDA_ARCH_LIST"):
        return os.environ["ANIMA_CUDA_ARCH_LIST"]

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}+PTX"
    except Exception:
        pass

    # Portable fallback for Turing, Ampere, Ada, and Hopper. Override with
    # ANIMA_CUDA_ARCH_LIST or TORCH_CUDA_ARCH_LIST for older/newer toolchains.
    return "7.5;8.0;8.6;8.9;9.0+PTX"


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_arch_list())


def cxx_args():
    if os.name == "nt":
        return ["/O2"]
    return ["-O3", "-std=c++17"]


def nvcc_args():
    args = ["-O3", "--use_fast_math"]
    if platform.system() != "Windows":
        args.extend(["-Xcompiler", "-fPIC"])
    return args


setup(
    name="anima-cuda-ops",
    version="0.1.0",
    description="CUDA operators for Anima LoRA non-attention training hot paths",
    py_modules=["anima_cuda_ops"],
    ext_modules=[
        CUDAExtension(
            name="anima_cuda_ops_ext",
            sources=["anima_ops.cpp", "anima_ops_kernel.cu"],
            extra_compile_args={
                "cxx": cxx_args(),
                "nvcc": nvcc_args(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
