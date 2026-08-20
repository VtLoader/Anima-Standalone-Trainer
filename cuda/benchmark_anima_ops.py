import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import torch

import anima_cuda_ops as ops


@dataclass
class BenchResult:
    name: str
    dtype: str
    shape: str
    ref_ms: float
    cuda_ms: float
    speedup: float
    max_abs: float
    linf_rel: float


@dataclass
class ScenarioResult:
    name: str
    dtype: str
    config: str
    ref_ms: float
    cuda_ms: float
    speedup: float
    checks: dict


def _sync():
    torch.cuda.synchronize()


def run_for_time(fn: Callable, warmup_seconds: float, bench_seconds: float) -> float:
    end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < end:
        fn()
    _sync()

    iters = 0
    start = time.perf_counter()
    end = start + bench_seconds
    while time.perf_counter() < end:
        fn()
        iters += 1
    _sync()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(iters, 1)


def error(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    linf_rel = max_abs / max(bf.abs().max().item(), 1e-6)
    return max_abs, linf_rel


def bench_rmsnorm(dtype, warmup, bench) -> BenchResult:
    B, T, H, W, D = 2, 1, 64, 64, 1536
    x = torch.randn(B, T, H, W, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype)
    ref = lambda: ops.rmsnorm_ref(x, w, 1e-6)
    cur = lambda: ops.rmsnorm(x, w, 1e-6)
    y_ref = ref()
    y_cur = cur()
    _sync()
    max_abs, max_rel = error(y_cur, y_ref)
    ref_ms = run_for_time(ref, warmup, bench)
    cuda_ms = run_for_time(cur, warmup, bench)
    return BenchResult("rmsnorm", str(dtype).replace("torch.", ""), str(tuple(x.shape)), ref_ms, cuda_ms, ref_ms / cuda_ms, max_abs, max_rel)


def bench_rope(dtype, warmup, bench) -> BenchResult:
    B, S, H, D = 2, 4096, 16, 96
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    freqs = torch.randn(S, D, device="cuda", dtype=dtype)
    ref = lambda: ops.rope_qk_ref(q, k, freqs)
    cur = lambda: ops.rope_qk(q, k, freqs)
    rq, rk = ref()
    cq, ck = cur()
    _sync()
    max_abs_q, max_rel_q = error(cq, rq)
    max_abs_k, max_rel_k = error(ck, rk)
    ref_ms = run_for_time(ref, warmup, bench)
    cuda_ms = run_for_time(cur, warmup, bench)
    return BenchResult("rope_qk", str(dtype).replace("torch.", ""), str(tuple(q.shape)), ref_ms, cuda_ms, ref_ms / cuda_ms, max(max_abs_q, max_abs_k), max(max_rel_q, max_rel_k))


def bench_noisy(dtype, warmup, bench) -> BenchResult:
    B, C, T, H, W = 2, 16, 1, 128, 128
    latents = torch.randn(B, C, T, H, W, device="cuda", dtype=dtype)
    noise = torch.randn_like(latents)
    t = torch.rand(B, device="cuda", dtype=dtype).clamp(1e-5, 1 - 1e-5)
    ref = lambda: ops.noisy_input_ref(latents, noise, t)
    cur = lambda: ops.noisy_input(latents, noise, t)
    y_ref = ref()
    y_cur = cur()
    _sync()
    max_abs, max_rel = error(y_cur, y_ref)
    ref_ms = run_for_time(ref, warmup, bench)
    cuda_ms = run_for_time(cur, warmup, bench)
    return BenchResult("noisy_input", str(dtype).replace("torch.", ""), str(tuple(latents.shape)), ref_ms, cuda_ms, ref_ms / cuda_ms, max_abs, max_rel)


def bench_loss(dtype, warmup, bench) -> BenchResult:
    B, C, T, H, W = 2, 16, 1, 128, 128
    model_pred = torch.randn(B, C, T, H, W, device="cuda", dtype=dtype)
    latents = torch.randn_like(model_pred)
    noise = torch.randn_like(model_pred)
    weighting = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)
    loss_weights = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)
    ref = lambda: ops.rectified_flow_mse_loss_ref(model_pred, latents, noise, weighting, loss_weights)
    cur = lambda: ops.rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights)
    y_ref = ref()
    y_cur = cur()
    _sync()
    max_abs, max_rel = error(y_cur.view(1), y_ref.view(1))
    ref_ms = run_for_time(ref, warmup, bench)
    cuda_ms = run_for_time(cur, warmup, bench)
    return BenchResult("rectified_flow_mse_loss", str(dtype).replace("torch.", ""), str(tuple(model_pred.shape)), ref_ms, cuda_ms, ref_ms / cuda_ms, max_abs, max_rel)


def bench_realistic_anima_lora(dtype, warmup, bench, batch_size, image_size, blocks, heads, model_dim) -> ScenarioResult:
    # Anima default image training path: image -> WanVAE latent with 8x downscale,
    # then DiT patch_spatial=2 makes S=(H/8/2)*(W/8/2). For 1024px, S=4096.
    latent_h = image_size // 8
    latent_w = image_size // 8
    patch_spatial = 2
    seq_len = (latent_h // patch_spatial) * (latent_w // patch_spatial)
    head_dim = model_dim // heads
    if model_dim % heads != 0:
        raise ValueError("model_dim must be divisible by heads")

    B = batch_size
    q = torch.randn(B, seq_len, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    q_cross = torch.randn_like(q)
    k_cross = torch.randn_like(q)
    w_head = torch.randn(head_dim, device="cuda", dtype=dtype)
    freqs = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)

    latents = torch.randn(B, 16, 1, latent_h, latent_w, device="cuda", dtype=dtype)
    noise = torch.randn_like(latents)
    model_pred = torch.randn_like(latents)
    t = torch.rand(B, device="cuda", dtype=dtype).clamp(1e-5, 1 - 1e-5)
    weighting = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)
    loss_weights = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)

    def ref():
        noisy = ops.noisy_input_ref(latents, noise, t)
        rq = rk = None
        cq = ck = None
        for _ in range(blocks):
            # Self-attention QK RMSNorm + RoPE, excluding attention itself.
            qn = ops.rmsnorm_ref(q, w_head, 1e-6)
            kn = ops.rmsnorm_ref(k, w_head, 1e-6)
            rq, rk = ops.rope_qk_ref(qn, kn, freqs)
            # Cross-attention QK RMSNorm, excluding attention itself.
            cq = ops.rmsnorm_ref(q_cross, w_head, 1e-6)
            ck = ops.rmsnorm_ref(k_cross, w_head, 1e-6)
        loss = ops.rectified_flow_mse_loss_ref(model_pred, latents, noise, weighting, loss_weights)
        return noisy, rq, rk, cq, ck, loss

    def cur():
        noisy = ops.noisy_input(latents, noise, t)
        rq = rk = None
        cq = ck = None
        for _ in range(blocks):
            qn = ops.rmsnorm(q, w_head, 1e-6)
            kn = ops.rmsnorm(k, w_head, 1e-6)
            rq, rk = ops.rope_qk(qn, kn, freqs)
            cq = ops.rmsnorm(q_cross, w_head, 1e-6)
            ck = ops.rmsnorm(k_cross, w_head, 1e-6)
        loss = ops.rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights)
        return noisy, rq, rk, cq, ck, loss

    ref_out = ref()
    cur_out = cur()
    _sync()
    labels = ("noisy", "self_q", "self_k", "cross_q", "cross_k", "loss")
    checks = {}
    for label, c, r in zip(labels, cur_out, ref_out):
        max_abs, linf_rel = error(c.view(-1), r.view(-1))
        checks[label] = {"max_abs": max_abs, "linf_rel": linf_rel}

    ref_ms = run_for_time(ref, warmup, bench)
    cuda_ms = run_for_time(cur, warmup, bench)
    config = json.dumps(
        {
            "batch_size": B,
            "image_size": image_size,
            "latent_shape": [B, 16, 1, latent_h, latent_w],
            "seq_len": seq_len,
            "blocks": blocks,
            "heads": heads,
            "head_dim": head_dim,
            "model_dim": model_dim,
        },
        ensure_ascii=False,
    )
    return ScenarioResult("realistic_anima_lora_non_attention", str(dtype).replace("torch.", ""), config, ref_ms, cuda_ms, ref_ms / cuda_ms, checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--bench-seconds", type=float, default=5.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "all"], default="bf16")
    parser.add_argument("--mode", choices=["micro", "realistic", "both"], default="micro")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=28)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=1536)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.warmup_seconds < 1.0:
        raise ValueError("Each operator warmup must be at least 1s")
    if args.bench_seconds < 5.0:
        raise ValueError("Each operator benchmark must be at least 5s")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.backends.cuda.matmul.allow_tf32 = True
    ops.load_extension(verbose=args.verbose_build)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total memory GiB: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtypes = list(dtype_map.values()) if args.dtype == "all" else [dtype_map[args.dtype]]

    results = []
    for dtype in dtypes:
        if args.mode in ("micro", "both"):
            for bench_fn in (bench_rmsnorm, bench_rope, bench_noisy, bench_loss):
                result = bench_fn(dtype, args.warmup_seconds, args.bench_seconds)
                results.append(result)
                print(json.dumps(result.__dict__, ensure_ascii=False))
        if args.mode in ("realistic", "both"):
            result = bench_realistic_anima_lora(
                dtype,
                args.warmup_seconds,
                args.bench_seconds,
                args.batch_size,
                args.image_size,
                args.blocks,
                args.heads,
                args.model_dim,
            )
            results.append(result)
            print(json.dumps(result.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
