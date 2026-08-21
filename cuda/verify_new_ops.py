import sys
from pathlib import Path

import torch
import anima_cuda_ops as ops

# Ensure the repo `library` package (with anima_cuda_accel) wins over any
# installed copy.
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def check(name, a, b, tol_abs=1e-3, tol_rel=5e-2):
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    diff = (aa - bb).abs()
    maxabs = diff.max().item()
    linf_rel = maxabs / max(bb.abs().max().item(), 1e-6)
    ok = (maxabs <= tol_abs) or (linf_rel <= tol_rel)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: max_abs={maxabs:.3e} linf_rel={linf_rel:.3e}")
    return ok


def test_dtype(dtype, name):
    ok = True
    torch.manual_seed(0)

    # --- adaln_norm forward ---
    B, T, H, W, D = 2, 1, 16, 16, 512
    x = torch.randn(B, T, H, W, D, device="cuda", dtype=dtype) * 2
    scale = torch.randn(B * T, D, device="cuda", dtype=dtype) * 0.1
    shift = torch.randn(B * T, D, device="cuda", dtype=dtype) * 0.1
    y = ops.adaln_norm(x, scale, shift, 1e-6)
    y_ref = ops.adaln_norm_ref(x, scale, shift, 1e-6)
    ok &= check(f"adaln_norm_fwd_{name}", y, y_ref)

    # --- adaln_norm backward (grad w.r.t. x, scale, shift) ---
    g = torch.randn_like(x)
    bt_total = B * T
    hw = H * W
    dx, ds, dh = ops.adaln_norm_backward(g, x, scale, bt_total, hw, 1e-6)
    dx_r, ds_r, dh_r = ops.adaln_norm_backward_ref(g, x, scale, bt_total, hw, 1e-6)
    ok &= check(f"adaln_norm_bwd_x_{name}", dx, dx_r)
    ok &= check(f"adaln_norm_bwd_scale_{name}", ds, ds_r)
    ok &= check(f"adaln_norm_bwd_shift_{name}", dh, dh_r)

    # --- autograd through the accel entry (fp32; exact match check) ---
    if dtype == torch.float32:
        from library import anima_cuda_accel as acc
        acc.set_enabled(True)
        xf = x.clone().requires_grad_(True)
        sf = scale.clone().requires_grad_(True)
        hf = shift.clone().requires_grad_(True)
        yf = acc.adaln_norm(xf, sf, hf, 1e-6)
        yf.backward(g)  # backprop with the SAME upstream grad used for dx/ds/dh
        ok &= check(f"adaln_autograd_x_{name}", xf.grad, dx_r, tol_abs=1e-4, tol_rel=1e-4)
        ok &= check(f"adaln_autograd_s_{name}", sf.grad, ds_r, tol_abs=1e-4, tol_rel=1e-4)
        ok &= check(f"adaln_autograd_h_{name}", hf.grad, dh_r, tol_abs=1e-4, tol_rel=1e-4)

    # --- adamw_step (fp32 only) ---
    if dtype == torch.float32:
        n = 1 << 20
        p = torch.randn(n, device="cuda", dtype=torch.float32) * 0.1
        g_t = torch.randn(n, device="cuda", dtype=torch.float32) * 0.05
        m1 = torch.zeros(n, device="cuda", dtype=torch.float32)
        v1 = torch.zeros(n, device="cuda", dtype=torch.float32)
        p2 = p.clone(); m2 = torch.zeros_like(m1); v2 = torch.zeros_like(v1)
        g2 = g_t.clone()
        lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
        bc1, bc2 = 1.0, 1.0
        ops.adamw_step(g_t, p, m1, v1, lr, b1, b2, eps, wd, bc1, bc2)
        ops.adamw_step_ref(g2, p2, m2, v2, lr, b1, b2, eps, wd, bc1, bc2)
        ok &= check("adamw_step_p", p, p2)
        ok &= check("adamw_step_m", m1, m2)
        ok &= check("adamw_step_v", v1, v2)
    return ok


def main():
    all_ok = True
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        all_ok &= test_dtype(dtype, name)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")


if __name__ == "__main__":
    main()
