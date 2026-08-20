# Anima CUDA Ops

This folder contains non-FlashAttention CUDA operator experiments for the Anima LoRA training path.

Implemented operators:

- `rmsnorm`: fused forward for Anima/QK/timestep RMSNorm-like usage.
- `rope_qk`: fused non-interleaved RoPE for q/k tensors shaped `(B, S, H, D)`.
- `noisy_input`: rectified-flow blend `(1 - t) * latents + t * noise`.
- `rectified_flow_mse_loss`: fused `target = noise - latents`, MSE, per-sample reduction, weighting, and final mean.

The extension is loaded via `torch.utils.cpp_extension.load`, so it does not modify the main trainer by default.

## Environment

The local machine detected during implementation:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: `8.9`
- VRAM: about 8 GiB
- Reference venv: `D:\anima_trainer_ref\venv`
- PyTorch: `2.10.0+cu130`
- CUDA Toolkit: default Windows install path

## Architecture Support

The wheel is built for the architecture list in `TORCH_CUDA_ARCH_LIST`. You can also set `ANIMA_CUDA_ARCH_LIST`; `setup.py` maps it to `TORCH_CUDA_ARCH_LIST` when the PyTorch variable is not already set.

Default behavior:

- If `TORCH_CUDA_ARCH_LIST` is set, it is used unchanged.
- Else if `ANIMA_CUDA_ARCH_LIST` is set, it is used unchanged.
- Else if a CUDA GPU is visible, the wheel targets the current GPU with PTX fallback, for example `8.9+PTX`.
- Else it falls back to `7.5;8.0;8.6;8.9;9.0+PTX`.

Common NVIDIA compute capabilities:

- Turing RTX 20xx/T4: `7.5`
- Ampere A100: `8.0`
- Ampere RTX 30xx/A10/A40: `8.6`
- Ada RTX 40xx/L4/L40: `8.9`
- Hopper H100/H200: `9.0`

Examples:

```powershell
$env:ANIMA_CUDA_ARCH_LIST="8.6;8.9+PTX"
```

```bash
export ANIMA_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
```

## Wheel Build And Install

Install build dependencies first if needed:

```bash
python -m pip install --upgrade pip setuptools wheel ninja
```

### Windows

Build the wheel with the Visual Studio and CUDA toolchain:

```powershell
cmd.exe /d /c "call `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" && set DISTUTILS_USE_SDK=1 && `"D:\anima_trainer_ref\venv\Scripts\python.exe`" setup.py bdist_wheel"
```

Install the built wheel into the reference venv:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" -m pip install --force-reinstall "D:\Anima-Standalone-Trainer\cuda\dist\anima_cuda_ops-0.1.0-cp311-cp311-win_amd64.whl"
```

Shortcut script:

```powershell
$env:PYTHON_BIN="D:\anima_trainer_ref\venv\Scripts\python.exe"
.\build_wheel_windows.ps1
```

### Linux

Build with the active Python environment:

```bash
cd cuda
python setup.py bdist_wheel
```

Or use the helper script:

```bash
cd cuda
PYTHON_BIN=/path/to/venv/bin/python ./build_wheel_linux.sh
```

Install the built wheel:

```bash
/path/to/venv/bin/python -m pip install --force-reinstall dist/anima_cuda_ops-0.1.0-*.whl
```

There is intentionally no JIT fallback in `anima_cuda_ops.py`. Importing `anima_cuda_ops` must load the installed extension directly (`.pyd` on Windows, `.so` on Linux), otherwise the import fails.

Installed-package smoke test from outside the repository:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" -c "import torch, anima_cuda_ops, anima_cuda_ops_ext; print(anima_cuda_ops.__file__); print(anima_cuda_ops_ext.__file__); x=torch.randn(2,4,device='cuda',dtype=torch.bfloat16); w=torch.randn(4,device='cuda',dtype=torch.bfloat16); y=anima_cuda_ops.rmsnorm(x,w,1e-6); torch.cuda.synchronize(); print(y.shape)"
```

Linux smoke test:

```bash
/path/to/venv/bin/python -c "import torch, anima_cuda_ops, anima_cuda_ops_ext; print(anima_cuda_ops.__file__); print(anima_cuda_ops_ext.__file__); x=torch.randn(2,4,device='cuda',dtype=torch.bfloat16); w=torch.randn(4,device='cuda',dtype=torch.bfloat16); y=anima_cuda_ops.rmsnorm(x,w,1e-6); torch.cuda.synchronize(); print(y.shape)"
```

## Benchmark

Use the requested reference environment:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --dtype bf16
```

The benchmark enforces:

- warmup time per operator: at least 1 second
- measured benchmark time per operator: at least 5 seconds

Optional full dtype run:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --dtype all
```

Realistic bf16 Anima LoRA non-attention simulation:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --mode realistic --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536
```

The realistic mode simulates the non-FlashAttention part of a DiT LoRA training step:

- bf16 tensors by default.
- WanVAE latent shape derived from image size with 8x downscale.
- Patch-spatial size assumed to be 2, so 1024px maps to latent `128x128` and DiT sequence length `4096`.
- Repeats q/k RMSNorm and RoPE for `blocks` transformer blocks.
- Runs rectified-flow noisy input and fused MSE loss on realistic latent tensors.
- Excludes attention itself, qkv GEMM, MLP GEMM, and LoRA GEMM because those are cuBLAS/FlashAttention domains or need training integration/backward to measure fairly.

## Memory Comparison

Run PyTorch reference and installed wheel mode as separate commands/processes so CUDA allocator statistics do not contaminate each other:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\memory_compare_anima_ops.py" --mode ref --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --retain-activations
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\memory_compare_anima_ops.py" --mode cuda --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --retain-activations
```

`--mode ref` does not import `anima_cuda_ops_ext`. `--mode cuda` removes the source `cuda` folder from `sys.path` before import, forcing the installed venv package to be used.

Important limitation: this is a training-like forward activation-retention memory simulation. The current custom kernels do not implement autograd backward, so this script is not a full replacement for an end-to-end training memory profile.

## Trainer Integration

The training UI exposes `Enable CUDA acceleration`, saved as:

```toml
[training_arguments]
enable_cuda_acceleration = true
```

When enabled, Anima training imports the installed `anima-cuda-ops` wheel and accelerates supported non-FlashAttention paths:

- RMSNorm in `library/anima_models.py`
- self-attention RoPE q/k rotation in `library/anima_models.py`
- rectified-flow noisy input in `library/anima_train_utils.py`
- simple full-finetune L2 rectified-flow loss in `anima_train.py` and `anima_train_muon.py`

LoRA training through `anima_train_network.py` uses the accelerated RMSNorm, RoPE, and noisy input paths. Its loss path remains conservative because NetworkTrainer supports differential output preservation and masked loss variants.

## Notes

These kernels intentionally exclude FlashAttention because the training code already supports mature FlashAttention backends. The target is the remaining smaller operators that can become visible after attention is optimized: RMSNorm, RoPE, rectified-flow elementwise work, and loss reduction.
