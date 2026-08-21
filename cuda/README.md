# Anima CUDA Ops

This folder contains non-FlashAttention CUDA operator experiments for the Anima LoRA training path.

Implemented operators:

- `rmsnorm`: fused forward **and backward** for Anima/QK/timestep RMSNorm-like usage.
- `rope_qk`: fused non-interleaved RoPE for q/k tensors shaped `(B, S, H, D)`, forward and backward.
- `noisy_input`: rectified-flow blend `(1 - t) * latents + t * noise`.
- `rectified_flow_mse_loss`: fused `target = noise - latents`, MSE, per-sample reduction, weighting, and final mean, forward and backward.
- `adaln_norm`: fused LayerNorm(affine=False) + AdaLN scale/shift modulation, forward and backward.
- `adamw_step`: fused AdamW optimizer step (decoupled weight decay).

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

The benchmark uses `triton.testing.do_bench` for timing, with defaults:

- `--warmup-ms 500` (warmup 500 ms)
- `--rep-ms 5000` (measurement rep 5000 ms)

`do_bench` flushes the CUDA cache between iterations, so per-iteration times are
comparable across operators. Both values may be overridden:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --dtype bf16 --warmup-ms 1000 --rep-ms 10000
```

To enlarge the micro-benchmark shapes, pass `--scale N` (linearly scales the sampled
dimension of each operator) and add `--profile-memory` to report peak GPU memory usage:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --dtype bf16 --scale 16 --profile-memory
```

Large-data spot checks on the RTX 4060 8GB (bf16):

- `--scale 16 --profile-memory`: rmsnorm 10.3x, rope 6.63x (S=65536), loss 5.0x; peak 5.26 GiB allocated (65.8% VRAM).
- `--mode realistic --image-size 2048 --batch-size 2 --blocks 28 --profile-memory`: 3.55x (1353 -> 381 ms); peak 2.53 GiB (31.6% VRAM).

Both stay well under the 8 GiB VRAM budget.

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

## Precision Validation

Correctness of the fused operators is checked three ways:

1. **`verify_optimized.py` / `verify_new_ops.py`** — fused forward/backward vs the pure-torch
   `*_ref` implementations, bf16/fp16/fp32.
2. **`gradcheck_ops.py`** — independent **finite-difference** gradient check of each fused
   backward (ground truth not derived from any shared reference).
3. **`precision_compare.py`** — full-pipeline "optimized path vs the original torch path" on a
   real `Block` (self/cross attention, AdaLN, QK-RMSNorm, RoPE) + noisy_input + flow loss,
   comparing forward outputs, `x` grad, and every parameter grad.

### Bug found & fixed: fused RoPE backward

The full-pipeline fp32 comparison initially exposed that the fused **RoPE backward was wrong**:
the self-attention q/k weight gradients differed from the original torch path by ~0.9 relative
error in fp32 (not a precision issue — it reproduced exactly in fp32 while other gradients were
~1e-6). The kernel reused the forward rotation where the crossed term uses `sin[d]`, but the
transpose requires `sin[mate]`. This silently corrupted self-attention q/k weight gradients
during training. Fixed by adding a `use_mate_sin` flag to `rope_kernel` (forward `0`, backward
`1`) and correcting `rope_qk_backward_ref`.

After the fix, `precision_compare.py` reports (optimized vs original, a real `Block`):

- fp32: forward and all gradients at ~1e-6 (machine precision).
- fp16: ~1e-3 relative error (normal mixed-precision noise).
- bf16: ~1e-2 relative error (normal bf16 rounding).

and `gradcheck_ops.py` shows the fused backward matches finite differences to ~1e-3 relative
for rmsnorm, rope (q/k), and adaln_norm.

## Training hook-up

With `--enable_cuda_acceleration` set, the fused operators are wired into the real model and
training path via `anima_cuda_accel.is_enabled()`:

| Operator | Hook point |
| --- | --- |
| `rmsnorm` (QK-norm / timestep-norm) | `anima_models.py` RMSNorm.forward |
| `rope_qk` (self-attention) | `anima_models.py` Attention.compute_qkv |
| `adaln_norm` (Block LayerNorm+AdaLN) | `anima_models.py` Block `_adaln_fn` |
| `noisy_input` | `anima_train_utils.py` rectified-flow path |
| `rectified_flow_mse_loss` | `anima_train.py` / `anima_train_muon.py` loss |
| `fused_adamw_step` via `FusedAdamW` | `train_util.py` `--optimizer_type AdamW` |

`FusedAdamW` subclasses `torch.optim.AdamW`; when acceleration is enabled and the master
weights are fp32/CUDA its `step()` calls the fused CUDA kernel, otherwise it falls back to the
stock AdamW step. Backward passes of the autograd ops are fused automatically with the forward.
The TP/SP trainers reuse `AnimaTrainer().train()`, so the same model-path hooks apply there.

## Fused global-norm grad clip (in the optimizer step)

When `--enable_cuda_acceleration` is on and `--max_grad_norm > 0` with `--optimizer_type AdamW`,
`FusedAdamW` is constructed with `max_grad_norm` and its `step()` computes the global gradient
norm once and folds `clip_scale = min(1, max_norm / grad_norm)` into every fused kernel launch,
so the trainer can skip its own `clip_grad_norm_` pass (`optimizer_handles_clip()`).
`amsgrad`/`maximize` groups and sparse grads make the optimizer fall back to stock AdamW.

## CUDA-graph training step (experimental, `--cuda_graph`)

`--cuda_graph` captures the fixed-shape DiT forward + fused loss + backward as one CUDA graph
(`library/cuda_graph_util.py::MaybeGraphedStep`) to strip per-kernel launch overhead.
It engages only for: single GPU, `--enable_cuda_acceleration`, l2 unmasked loss, and no
block swap / CPU-unsloth offload / gradient checkpointing / fused-backward. It re-captures on
shape change and falls back to eager permanently on any capture failure. Because CUDA graphs
record only kernel launches (not autograd's `param.grad` rebinding), the trainer zeroes
gradients in place (`set_to_none=False`) while the graph step is active.

## Loss backward saves only the residual

`rectified_flow_mse_loss`'s autograd wrapper now stores the single residual
`diff = pred - (noise - latents)` for backward instead of three full tensors, halving the
activation kept for the fused loss while keeping the exact fp32 gradient.

## Fusion skip detection

`anima_cuda_accel.rmsnorm` warns once when fusion is silently bypassed (e.g. mixed activation /
weight dtypes or CPU inputs), so "acceleration enabled but not applied" is visible in the log.
