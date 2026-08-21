"""Item 3 timing: CUDA-graph replay vs eager for the 28-block fused non-attention
loop at realistic shape (S=4096, 1024px). Measures launch-overhead savings.
"""
import sys
from pathlib import Path

import torch
from triton.testing import do_bench

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import anima_cuda_ops as ops
from library.cuda_graph_util import capture_forward, MaybeGraphedStep


def main():
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16
    B, S, H, D = 1, 4096, 16, 96
    blocks = 28
    q = torch.randn(B, S, H, D, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    w = torch.randn(D, device=dev, dtype=dtype) + 0.5
    freqs = torch.randn(S, D, device=dev, dtype=dtype)
    lat = torch.randn(B, 16, 1, 64, 64, device=dev, dtype=dtype)
    noise = torch.randn_like(lat)
    t = torch.rand(B, device=dev, dtype=dtype).clamp(1e-5, 1 - 1e-5)

    def loop_fn(q, k, freqs, lat, noise, t, w):
        for _ in range(blocks):
            qn = ops.rmsnorm(q, w, 1e-6)
            kn = ops.rmsnorm(k, w, 1e-6)
            q, k = ops.rope_qk(qn, kn, freqs)
        return ops.noisy_input(lat, noise, t) + q.sum() * 0.0 + k.sum() * 0.0

    # warm once
    loop_fn(q, k, freqs, lat, noise, t, w)
    torch.cuda.synchronize()
    t_eager = do_bench(lambda: loop_fn(q, k, freqs, lat, noise, t, w), warmup=200, rep=3000)

    static = [q.clone(), k.clone(), freqs.clone(), lat.clone(), noise.clone(), t.clone(), w.clone()]
    cap = capture_forward(loop_fn, *static)
    torch.cuda.synchronize()
    t_graph = do_bench(lambda: cap.replay(q, k, freqs, lat, noise, t, w), warmup=200, rep=3000)

    # correctness: replay on same input == eager
    out_eager = loop_fn(q, k, freqs, lat, noise, t, w)
    out_graph = cap.replay(q, k, freqs, lat, noise, t, w)
    torch.cuda.synchronize()
    diff = (out_eager - out_graph).abs().max().item()

    print(f"28-block fused non-attn loop (S={S}): eager={t_eager:.4f}ms  graph={t_graph:.4f}ms  "
          f"speedup x{t_eager / t_graph:.2f}  (replay==eager max_abs={diff:.3e})")


if __name__ == "__main__":
    main()
