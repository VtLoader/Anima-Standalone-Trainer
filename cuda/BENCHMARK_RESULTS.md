# Benchmark Results

Command:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --dtype all --warmup-seconds 1 --bench-seconds 5
```

Environment:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: 8.9
- VRAM: 8.00 GiB
- PyTorch: 2.10.0+cu130 for the installed-wheel verification. Earlier micro-benchmark rows were collected before the venv was updated.
- CUDA runtime: 13.0
- CUDA Toolkit: default Windows install path

Each operator was warmed up for at least 1 second and benchmarked for at least 5 seconds.

| Op | Dtype | Shape | PyTorch ref ms | CUDA op ms | Speedup | Max abs | Linf rel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| rmsnorm | bfloat16 | `(2, 1, 64, 64, 1536)` | 2.2297 | 0.2259 | 9.87x | 0.0625 | 0.00467 |
| rope_qk | bfloat16 | `(2, 4096, 16, 96)` | 2.9055 | 0.4824 | 6.02x | 0.03125 | 0.00565 |
| noisy_input | bfloat16 | `(2, 16, 1, 128, 128)` | 0.3528 | 0.0750 | 4.70x | 0.03125 | 0.00752 |
| rectified_flow_mse_loss | bfloat16 | `(2, 16, 1, 128, 128)` | 0.9776 | 0.1334 | 7.33x | 0.00000823 | 0.00000754 |
| rmsnorm | float16 | `(2, 1, 64, 64, 1536)` | 2.2291 | 0.2244 | 9.93x | 0.0078125 | 0.000586 |
| rope_qk | float16 | `(2, 4096, 16, 96)` | 2.9093 | 0.4836 | 6.02x | 0.00390625 | 0.000735 |
| noisy_input | float16 | `(2, 16, 1, 128, 128)` | 0.0470 | 0.0109 | 4.31x | 0.00195313 | 0.000419 |
| rectified_flow_mse_loss | float16 | `(2, 16, 1, 128, 128)` | 0.1342 | 0.1345 | 1.00x | 0.000000954 | 0.00000117 |
| rmsnorm | float32 | `(2, 1, 64, 64, 1536)` | 1.4965 | 0.4391 | 3.41x | 0.00000191 | 0.000000135 |
| rope_qk | float32 | `(2, 4096, 16, 96)` | 5.3112 | 0.9051 | 5.87x | 0.00000203 | 0.000000373 |
| noisy_input | float32 | `(2, 16, 1, 128, 128)` | 0.3711 | 0.0754 | 4.92x | 0.000000238 | 0.0000000558 |
| rectified_flow_mse_loss | float32 | `(2, 16, 1, 128, 128)` | 0.7130 | 0.1345 | 5.30x | 0.0 | 0.0 |

Notes:

- FlashAttention is intentionally excluded.
- The kernels are forward/benchmark interfaces for the non-attention hot-path candidates. They are not wired into the trainer by default.
- The reference functions mirror the relevant PyTorch expressions from `library/anima_models.py`, `library/anima_train_utils.py`, and the Anima training loss path.

## Realistic bf16 Anima LoRA simulation

The following commands simulate a bf16 Anima LoRA training step's non-FlashAttention hot path by repeating Q/K RMSNorm and RoPE for 28 DiT blocks, plus rectified-flow noisy input and loss. This intentionally excludes attention, qkv GEMMs, MLP GEMMs, and LoRA GEMMs.

### 1024px, batch 1

Command:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --mode realistic --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --warmup-seconds 1 --bench-seconds 5
```

Result:

- latent shape: `[1, 16, 1, 128, 128]`
- sequence length: `4096`
- PyTorch reference: `136.30 ms`
- CUDA ops: `42.38 ms`
- speedup: `3.22x`
- worst `linf_rel`: `0.0112`

After installing the wheel into the `cu130` venv, the same command produced:

- PyTorch: `2.10.0+cu130`
- CUDA runtime: `13.0`
- PyTorch reference: `134.81 ms`
- CUDA ops: `42.41 ms`
- speedup: `3.18x`
- worst `linf_rel`: `0.0000117` for loss and `0.00637` for tensor outputs

Installed package paths verified from outside the repository:

```text
D:\anima_trainer_ref\venv\Lib\site-packages\anima_cuda_ops.py
D:\anima_trainer_ref\venv\Lib\site-packages\anima_cuda_ops_ext.cp311-win_amd64.pyd
```

### 1024px, batch 2

Command:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --mode realistic --dtype bf16 --batch-size 2 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --warmup-seconds 1 --bench-seconds 5
```

Result:

- latent shape: `[2, 16, 1, 128, 128]`
- sequence length: `4096`
- PyTorch reference: `332.95 ms`
- CUDA ops: `84.95 ms`
- speedup: `3.92x`
- worst `linf_rel`: `0.0176`

### 1536px, batch 1

Command:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\benchmark_anima_ops.py" --mode realistic --dtype bf16 --batch-size 1 --image-size 1536 --blocks 28 --heads 16 --model-dim 1536 --warmup-seconds 1 --bench-seconds 5
```

Result:

- latent shape: `[1, 16, 1, 192, 192]`
- sequence length: `9216`
- PyTorch reference: `376.69 ms`
- CUDA ops: `95.84 ms`
- speedup: `3.93x`
- worst `linf_rel`: `0.0102`

## Memory comparison

The memory comparison was run in separate Python processes for the PyTorch reference and installed `anima-cuda-ops` package to avoid CUDA allocator statistics leaking between modes.

Script:

```powershell
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\memory_compare_anima_ops.py" --mode ref --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --retain-activations
& "D:\anima_trainer_ref\venv\Scripts\python.exe" "D:\Anima-Standalone-Trainer\cuda\memory_compare_anima_ops.py" --mode cuda --dtype bf16 --batch-size 1 --image-size 1024 --blocks 28 --heads 16 --model-dim 1536 --retain-activations
```

Important limitation: these kernels currently expose forward operators only. The comparison simulates the training hot-path memory pressure by retaining forward activations, but it is not a full autograd/backward training run.

### 1024px, batch 1, retain activations

| Mode | Wrapper | Extension | Peak allocated MiB | Peak reserved MiB | Peak delta MiB | Final allocated MiB | Time ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch ref | none | none | 2126.75 | 2144.00 | 2076.50 | 2066.75 | 221.63 |
| anima-cuda-ops | venv site-packages | venv `.pyd` | 2066.75 | 2068.00 | 2016.50 | 2066.75 | 55.38 |

Delta:

- Peak allocated reduction: `60.00 MiB`
- Peak reserved reduction: `76.00 MiB`
- Runtime speedup in this simulation: `4.00x`

### 1024px, batch 2, retain activations

| Mode | Wrapper | Extension | Peak allocated MiB | Peak reserved MiB | Peak delta MiB | Final allocated MiB | Time ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch ref | none | none | 4252.75 | 4280.00 | 4153.00 | 4132.75 | 424.80 |
| anima-cuda-ops | venv site-packages | venv `.pyd` | 4132.75 | 4134.00 | 4033.00 | 4132.75 | 101.75 |

Delta:

- Peak allocated reduction: `120.00 MiB`
- Peak reserved reduction: `146.00 MiB`
- Runtime speedup in this simulation: `4.17x`

### 1024px, batch 1, no activation retention

This isolates temporary operator memory rather than training-like activation retention.

| Mode | Peak allocated MiB | Peak reserved MiB | Peak delta MiB | Final allocated MiB | Time ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch ref | 194.75 | 224.00 | 144.50 | 122.75 | 213.68 |
| anima-cuda-ops | 146.75 | 148.00 | 96.50 | 122.75 | 47.64 |

Delta:

- Peak allocated reduction: `48.00 MiB`
- Peak reserved reduction: `76.00 MiB`
- Runtime speedup in this simulation: `4.49x`
