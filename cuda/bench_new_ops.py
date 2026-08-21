import json
import sys
from pathlib import Path

import torch
from triton.testing import do_bench

import anima_cuda_ops as ops

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def bm(name, ref, cur, warmup_ms=500, rep_ms=5000):
    do_bench(ref, warmup=50, rep=100)
    do_bench(cur, warmup=50, rep=100)
    r = do_bench(ref, warmup=warmup_ms, rep=rep_ms)
    c = do_bench(cur, warmup=warmup_ms, rep=rep_ms)
    out = {"name": name, "ref_ms": round(r, 4), "cuda_ms": round(c, 4), "speedup": round(r / c, 2)}
    print(json.dumps(out, ensure_ascii=False), flush=True)


def main():
    dtype = torch.float32
    warmup, rep = 500, 5000

    # --- adaln_norm: realistic Anima block shape (B,T,H,W,D), fp32 ---
    B, T, H, W, D = 2, 1, 64, 64, 2048  # 2.9B model_channels=2048, 64x64 grid
    x = torch.randn(B, T, H, W, D, device="cuda", dtype=dtype)
    scale = torch.randn(B * T, D, device="cuda", dtype=dtype) * 0.1
    shift = torch.randn(B * T, D, device="cuda", dtype=dtype) * 0.1
    bm(
        "adaln_norm_fwd",
        lambda: ops.adaln_norm_ref(x, scale, shift, 1e-6),
        lambda: ops.adaln_norm(x, scale, shift, 1e-6),
        warmup, rep,
    )
    g = torch.randn_like(x)
    bt_total, hw = B * T, H * W
    bm(
        "adaln_norm_bwd",
        lambda: ops.adaln_norm_backward_ref(g, x, scale, bt_total, hw, 1e-6),
        lambda: ops.adaln_norm_backward(g, x, scale, bt_total, hw, 1e-6),
        warmup, rep,
    )

    # --- adamw_step over a large param (fp32) ---
    n = 1 << 23  # 8M params, ~2.9B model scale
    p = torch.randn(n, device="cuda", dtype=torch.float32) * 0.1
    gt = torch.randn(n, device="cuda", dtype=torch.float32) * 0.05
    m = torch.zeros(n, device="cuda", dtype=torch.float32)
    v = torch.zeros(n, device="cuda", dtype=torch.float32)
    m2 = torch.zeros_like(m); v2 = torch.zeros_like(v); p2 = p.clone(); g2 = gt.clone()
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    bm(
        "adamw_step",
        lambda: ops.adamw_step_ref(g2, p2, m2, v2, lr, b1, b2, eps, wd, 1.0, 1.0),
        lambda: ops.adamw_step(gt, p, m, v, lr, b1, b2, eps, wd, 1.0, 1.0),
        warmup, rep,
    )


if __name__ == "__main__":
    main()
