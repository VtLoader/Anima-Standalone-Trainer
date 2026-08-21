"""Large-scale accuracy check: fused ops vs pure-torch reference at the big
shapes used for the speedup measurements (fp32 -> machine precision, plus
bf16 noise band). Covers rmsnorm, rope (fwd/bwd), noisy, loss (fwd/bwd),
adamw fused step.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import anima_cuda_ops as ops


def check(name, a, b, tol_abs, tol_rel):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    d = (a - b).abs().max().item()
    denom = max(b.abs().max().item(), 1e-6)
    rel = d / denom
    ok = (d <= tol_abs) or (rel <= tol_rel)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max_abs={d:.3e} linf_rel={rel:.3e}")
    return ok


def main():
    ok = True
    for dtype, name, ta, tr in [
        (torch.bfloat16, "bf16", 1e-3, 5e-2),
        (torch.float32, "fp32", 1e-6, 1e-6),
    ]:
        torch.manual_seed(0)
        # --- realistic 2048/batch2 shapes ---
        B, S, H, D = 2, 16384, 16, 96
        x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        w = torch.randn(D, device="cuda", dtype=dtype) + 0.5
        ok &= check(f"rmsnorm_fwd_{name}", ops.rmsnorm(x, w, 1e-6), ops.rmsnorm_ref(x, w, 1e-6), ta, tr)

        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        freqs = torch.randn(S, D, device="cuda", dtype=dtype)
        qo, ko = ops.rope_qk(q, k, freqs)
        qr, kr = ops.rope_qk_ref(q, k, freqs)
        ok &= check(f"rope_fwd_{name}", qo, qr, ta, tr)
        gqo = torch.randn_like(q); gko = torch.randn_like(k)
        gq, gk = ops.rope_qk_backward(gqo, gko, freqs)
        gqr, gkr = ops.rope_qk_backward_ref(gqo, gko, freqs)
        ok &= check(f"rope_bwd_{name}", gq, gqr, ta, tr)

        lat = torch.randn(B, 16, 1, 128, 128, device="cuda", dtype=dtype)
        noi = torch.randn_like(lat)
        mp = torch.randn_like(lat) * 0.3
        t = torch.rand(B, device="cuda", dtype=dtype).clamp(1e-5, 1 - 1e-5)
        wgt = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)
        lw = torch.rand(B, device="cuda", dtype=dtype).clamp_min(0.1)
        ok &= check(f"noisy_{name}", ops.noisy_input(lat, noi, t), ops.noisy_input_ref(lat, noi, t), ta, tr)
        ok &= check(f"loss_fwd_{name}", ops.rectified_flow_mse_loss(mp, lat, noi, wgt, lw).view(1),
                    ops.rectified_flow_mse_loss_ref(mp, lat, noi, wgt, lw).view(1), ta, tr)
        go = torch.tensor([1.3], device="cuda", dtype=torch.float32)
        ok &= check(f"loss_bwd_{name}", ops.rectified_flow_mse_loss_backward(go, mp, lat, noi, wgt, lw),
                    ops.rectified_flow_mse_loss_backward_ref(go, mp, lat, noi, wgt, lw), ta, tr)

    # --- adamw at scale-16 param size (fp32), with and without clip_scale ---
    n = 1 << 20
    for clip in (1.0, 0.5):
        p = torch.randn(n, device="cuda", dtype=torch.float32) * 0.1
        g = torch.randn(n, device="cuda", dtype=torch.float32) * 0.05
        p2 = p.clone(); g2 = g.clone()
        m1 = torch.zeros(n, device="cuda", dtype=torch.float32); v1 = torch.zeros_like(m1)
        m2 = torch.zeros_like(m1); v2 = torch.zeros_like(v1)
        ops.adamw_step(g, p, m1, v1, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0, clip_scale=clip)
        ops.adamw_step_ref(g2.mul_(clip), p2, m2, v2, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0, 1.0)
        ok &= check(f"adamw_1M_fp32_clip{clip}", p, p2, 1e-6, 1e-6)

    print("LARGE-SCALE ACCURACY ALL PASS" if ok else "LARGE-SCALE ACCURACY SOME FAILED")


if __name__ == "__main__":
    main()
