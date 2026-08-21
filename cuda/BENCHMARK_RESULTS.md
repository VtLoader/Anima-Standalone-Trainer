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

## Optimized results (after GPU-targeted changes)

The initial results above predate the GPU-targeted optimizations (vectorized RMSNorm,
pair-based RoPE with shared sincos, multi-block loss reduction, and fused backward
kernels). Re-run on the same RTX 4060 Laptop (CC 8.9) with the rebuilt wheel, timed with
`triton.testing.do_bench` (`--warmup-ms 500 --rep-ms 5000`).

### Forward micro-benchmark (same script, `--dtype all`, triton `do_bench`)

| Op | Dtype | Before (wall-clock) | After (do_bench) |
| --- | --- | ---: | ---: |
| rmsnorm | bfloat16 | 9.87x | 9.98x |
| rmsnorm | float16 | 9.93x | 9.91x |
| rmsnorm | float32 | 3.41x | 3.34x |
| rope_qk | bfloat16 | 6.02x | 5.72x |
| rope_qk | float16 | 6.02x | 5.71x |
| rope_qk | float32 | 5.87x | 5.95x |
| noisy_input | bfloat16 | 4.70x | 1.88x |
| noisy_input | float16 | 4.31x | 1.91x |
| noisy_input | float32 | 4.92x | 1.43x |
| rectified_flow_mse_loss | bfloat16 | 7.33x | 2.42x |
| rectified_flow_mse_loss | float16 | **1.00x** | **2.40x** |
| rectified_flow_mse_loss | float32 | 5.30x | 1.72x |

`do_bench` flushes the CUDA cache between every iteration, so the tiny elementwise
operators (`noisy_input`, `rectified_flow_mse_loss`, ~0.03-0.11 ms) are dominated by
launch/cache overhead and show compressed ratios. The memory-bound operators (`rmsnorm`,
`rope_qk`) are stable at ~10x and ~5.7x. The loss fp16 operator previously gave no
speedup under the old methodology (`1.00x`, one block per sample) and now beats the
reference; its absolute win is best seen in the realistic multi-block simulation.

### Backward micro-benchmark (`bench_backward.py`, CUDA vs Python reference, triton `do_bench`)

The backward pass was previously pure Python (fp32 upcast + several tensor kernels); it is
now fused CUDA.

| Op | Dtype | Speedup |
| --- | --- | ---: |
| rmsnorm_backward | bfloat16 | 4.0x |
| rmsnorm_backward | float16 | 4.1x |
| rmsnorm_backward | float32 | 2.6x |
| rope_backward | bfloat16 | 5.8x |
| rope_backward | float16 | 5.7x |
| rope_backward | float32 | 5.9x |
| rectified_flow_mse_loss_backward | bfloat16 | 3.6x |
| rectified_flow_mse_loss_backward | float16 | 3.5x |
| rectified_flow_mse_loss_backward | float32 | 1.6x |

Correctness of all forward and backward operators (vs `*_ref` reference implementations)
is verified for bf16/fp16/fp32 by `verify_optimized.py`.

## Large-data scaling (bf16, RTX 4060 8GB)

The benchmark script supports `--scale N` (enlarge micro shapes) and `--profile-memory`
(report peak GPU usage). Larger workloads amortize launch/cache overhead and hold or
improve speedups.

| Micro op | Size | Speedup |
| --- | --- | ---: |
| rmsnorm `--scale 16` | rows 32·4096, cols 1536 (~402M elems) | 10.26x |
| rope_qk `--scale 16` | S=65536 long sequence | 6.63x |
| noisy_input `--scale 16` | batch 32 | 1.82x |
| rectified_flow_mse_loss `--scale 16` | batch 32 | 4.98x |

Worst micro peak (`--scale 16 --profile-memory`): 5.26 GiB allocated (65.8% VRAM).

| Realistic | Config | Speedup | Peak VRAM |
| --- | --- | ---: | ---: |
| 1024px b2 | seq=4096, 28 blocks | 3.23x | -- |
| 2048px b2 | seq=16384, 28 blocks | 3.55x | 2.53 GiB (31.6%) |

All runs stay well under the 8 GiB budget.

## New fused operators (fused LayerNorm+AdaLN, fused AdamW)

Two more hot training operators were added, timed with triton `do_bench` on the RTX 4060
(8GB). Correctness is verified by `verify_new_ops.py` (bf16/fp16/fp32 forward+backward,
autograd, and the AdamW step vs the torch reference).

| Op | Shape / workload | Speedup |
| --- | --- | ---: |
| adaln_norm forward | (B,T,H,W,D)=(2,1,64,64,2048) fp32 | 4.8x |
| adaln_norm backward | same; includes grad_x + grad_scale/grad_shift | 5.4x |
| adamw_step | 8M-param fp32 master weights | 3.0x |

The `adaln_norm` backward reduces grad_scale/grad_shift by chunking each batch-timestep's
spatial rows across many blocks with `atomicAdd`, since `B*T` alone (e.g. 2) would leave
the GPU under-occupied.

## Precision validation of the optimized ops

Beyond timing, the optimized operators are checked against the **original torch path a real
`Block` actually executes** (not just a shared hand-written reference): `precision_compare.py`
runs a real `Block` (self/cross attention, AdaLN, QK-RMSNorm, RoPE) plus `noisy_input` and the
flow loss under `--enable_cuda_acceleration` off (original torch path) vs on (fused), and
`gradcheck_ops.py` verifies each fused backward with independent finite differences.

This full-pipeline fp32 comparison caught a **real fused-RoPE-backward bug**: self-attention
q/k weight gradients disagreed with the original torch path by ~0.9 relative error. The kernel
reused the forward rotation's `sin[d]` for the backward's crossed term; the transpose requires
`sin[mate]`. Fixed with a `use_mate_sin` flag (`rope_forward` uses `sin[d]`, `rope_backward`
uses `sin[mate]`) and the reference was corrected to match.

After the fix, the full-pipeline comparison is at machine precision in fp32:

- fp32: forward outputs and all parameter gradients ~1e-6 relative.
- fp16: ~1e-3 relative (normal mixed-precision noise).
- bf16: ~1e-2 relative (normal bf16 rounding).
- `gradcheck_ops.py` finite-difference check passes to ~1e-3 relative for rmsnorm, rope (q/k),
  and adaln_norm backward.
