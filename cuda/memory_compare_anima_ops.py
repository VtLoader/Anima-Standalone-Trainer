import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _bytes_to_mib(value: int) -> float:
    return value / 1024 / 1024


def _memory_snapshot(prefix: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        f"{prefix}_allocated_mib": _bytes_to_mib(torch.cuda.memory_allocated()),
        f"{prefix}_reserved_mib": _bytes_to_mib(torch.cuda.memory_reserved()),
        f"{prefix}_driver_free_mib": _bytes_to_mib(free),
        f"{prefix}_driver_total_mib": _bytes_to_mib(total),
    }


def _load_ops(mode: str):
    if mode == "cuda":
        script_dir = str(Path(__file__).resolve().parent)
        sys.path = [p for p in sys.path if str(Path(p or os.getcwd()).resolve()) != script_dir]
        import anima_cuda_ops as ops
        import anima_cuda_ops_ext

        return ops, getattr(ops, "__file__", None), getattr(anima_cuda_ops_ext, "__file__", None)
    return None, None, None


def rmsnorm_ref(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (input.float() * torch.rsqrt(input.float().pow(2).mean(-1, keepdim=True) + eps)).to(input.dtype) * weight


def rope_qk_ref(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_dim = freqs.shape[-1]
    cos = torch.cos(freqs[: q.shape[1]]).to(q.dtype).view(1, q.shape[1], 1, rot_dim)
    sin = torch.sin(freqs[: q.shape[1]]).to(q.dtype).view(1, q.shape[1], 1, rot_dim)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        out = x_rot * cos + rotated * sin
        return torch.cat((out, x_pass), dim=-1)

    return apply(q), apply(k)


def noisy_input_ref(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
    return (1 - t) * latents + t * noise


def rectified_flow_mse_loss_ref(
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    weighting: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    target = noise - latents
    loss = (model_pred.float() - target.float()).pow(2)
    loss = loss.mean(list(range(1, loss.ndim)))
    return (loss * weighting.float().view(-1) * loss_weights.float().view(-1)).mean()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)

    dtype = _dtype(args.dtype)
    ops, module_path, ext_path = _load_ops(args.mode)

    latent_h = args.image_size // 8
    latent_w = args.image_size // 8
    patch_spatial = args.patch_spatial
    seq_len = (latent_h // patch_spatial) * (latent_w // patch_spatial)
    if args.model_dim % args.heads != 0:
        raise ValueError("model_dim must be divisible by heads")
    head_dim = args.model_dim // args.heads
    B = args.batch_size
    device = torch.device(f"cuda:{args.device}")

    q = torch.randn(B, seq_len, args.heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    q_cross = torch.randn_like(q)
    k_cross = torch.randn_like(q)
    w_head = torch.randn(head_dim, device=device, dtype=dtype)
    freqs = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    latents = torch.randn(B, 16, 1, latent_h, latent_w, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    model_pred = torch.randn_like(latents)
    t = torch.rand(B, device=device, dtype=dtype).clamp(1e-5, 1 - 1e-5)
    weighting = torch.rand(B, device=device, dtype=dtype).clamp_min(0.1)
    loss_weights = torch.rand(B, device=device, dtype=dtype).clamp_min(0.1)

    torch.cuda.synchronize()
    baseline = _memory_snapshot("baseline")
    torch.cuda.reset_peak_memory_stats()

    retained = []
    started = time.perf_counter()
    for _ in range(args.iterations):
        if args.mode == "ref":
            noisy = noisy_input_ref(latents, noise, t)
            for _block in range(args.blocks):
                qn = rmsnorm_ref(q, w_head, 1e-6)
                kn = rmsnorm_ref(k, w_head, 1e-6)
                rq, rk = rope_qk_ref(qn, kn, freqs)
                cq = rmsnorm_ref(q_cross, w_head, 1e-6)
                ck = rmsnorm_ref(k_cross, w_head, 1e-6)
                if args.retain_activations:
                    retained.extend((qn, kn, rq, rk, cq, ck))
            loss = rectified_flow_mse_loss_ref(model_pred, latents, noise, weighting, loss_weights)
        else:
            noisy = ops.noisy_input(latents, noise, t)
            for _block in range(args.blocks):
                qn = ops.rmsnorm(q, w_head, 1e-6)
                kn = ops.rmsnorm(k, w_head, 1e-6)
                rq, rk = ops.rope_qk(qn, kn, freqs)
                cq = ops.rmsnorm(q_cross, w_head, 1e-6)
                ck = ops.rmsnorm(k_cross, w_head, 1e-6)
                if args.retain_activations:
                    retained.extend((qn, kn, rq, rk, cq, ck))
            loss = ops.rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights)
        if args.retain_activations:
            retained.extend((noisy, loss))

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / args.iterations
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final = _memory_snapshot("final")

    result = {
        "mode": args.mode,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "module_path": module_path,
        "extension_path": ext_path,
        "config": {
            "batch_size": B,
            "image_size": args.image_size,
            "latent_shape": [B, 16, 1, latent_h, latent_w],
            "seq_len": seq_len,
            "blocks": args.blocks,
            "heads": args.heads,
            "head_dim": head_dim,
            "model_dim": args.model_dim,
            "iterations": args.iterations,
            "retain_activations": args.retain_activations,
        },
        "elapsed_ms_per_iter": elapsed_ms,
        "peak_allocated_mib": _bytes_to_mib(peak_alloc),
        "peak_reserved_mib": _bytes_to_mib(peak_reserved),
        "peak_delta_allocated_mib": _bytes_to_mib(peak_alloc) - baseline["baseline_allocated_mib"],
        "retained_tensors": len(retained),
    }
    result.update(baseline)
    result.update(final)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ref", "cuda"], required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=28)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=1536)
    parser.add_argument("--patch-spatial", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--retain-activations", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
