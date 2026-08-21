import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import anima_cuda_ops as ops


def fd_grad(fn, x, eps=1e-3):
    """Finite-difference gradient of scalar fn(x) w.r.t. x (no autograd needed)."""
    xf = x.detach().float().clone()
    g = torch.zeros_like(xf)
    xflat = xf.view(-1)
    n = xflat.numel()
    with torch.no_grad():
        for i in range(n):
            orig = xflat[i].item()
            xflat[i] = orig + eps
            fp = fn(xf).item()
            xflat[i] = orig - eps
            fm = fn(xf).item()
            xflat[i] = orig
            g.view(-1)[i] = (fp - fm) / (2 * eps)
    return g.to(x.dtype)


def check_grad(name, fused_backward_fn, forward_fn, ref_shape, dtype=torch.float32, eps=1e-3, tol=2e-2):
    torch.manual_seed(0)
    x = torch.randn(*ref_shape, device="cuda", dtype=dtype)
    # wheel forward (fused)
    outs = forward_fn(x)
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    # downstream scalar = sum of outputs
    def scalar(*args):
        return sum(o.float().sum() for o in args)
    # fused analytic grads (all wrt x)
    if isinstance(outs[0], torch.Tensor):
        # rmsnorm uses weight; handled via functor below
        pass
    return x


def main():
    torch.set_printoptions(precision=6)

    # --- 1) RMSNorm backward vs finite diff ---
    print("=== rmsnorm backward (gradcheck fd) ===")
    torch.manual_seed(0)
    D = 96
    x = torch.randn(4, D, device="cuda", dtype=torch.float32)
    w = torch.randn(D, device="cuda", dtype=torch.float32) + 0.5
    y = ops.rmsnorm(x, w, 1e-6)
    gy = torch.randn_like(y)

    # fd grad wrt x
    def fn_x(xx):
        yy = ops.rmsnorm(xx, w, 1e-6)
        return (yy * gy).sum()
    gx_fd = fd_grad(fn_x, x)
    gxx_fused, _gw = ops.rmsnorm_backward(gy, x, w, 1e-6)
    errx = (gx_fd.float() - gxx_fused.float()).abs().max().item()
    relx = errx / gx_fd.float().abs().max().item()
    print(f"grad_x fd vs fused: max_abs={errx:.3e} rel={relx:.3e}")

    # fd grad wrt w
    def fn_w(ww):
        yy = ops.rmsnorm(x, ww, 1e-6)
        return (yy * gy).sum()
    gw_fd = fd_grad(fn_w, w)
    _gx, gw_fused = ops.rmsnorm_backward(gy, x, w, 1e-6)
    errw = (gw_fd.float() - gw_fused.float()).abs().max().item()
    relw = errw / gw_fd.float().abs().max().item()
    print(f"grad_w fd vs fused: max_abs={errw:.3e} rel={relw:.3e}")

    # --- 2) RoPE backward vs finite diff ---
    print("=== rope backward (gradcheck fd) ===")
    torch.manual_seed(1)
    B, S, H, D = 2, 16, 4, 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    freqs = torch.randn(S, D, device="cuda", dtype=torch.float32)
    qo, ko = ops.rope_qk(q, k, freqs)
    gq = torch.randn_like(qo)
    gk = torch.randn_like(ko)

    def fn_q(qq):
        qo2, ko2 = ops.rope_qk(qq, k, freqs)
        return (qo2 * gq).sum() + (ko2 * gk).sum()
    gx_fd = fd_grad(fn_q, q)
    gq_fused, _ = ops.rope_qk_backward(gq, gk, freqs)
    errq = (gx_fd.float() - gq_fused.float()).abs().max().item()
    relq = errq / gx_fd.float().abs().max().item()
    print(f"grad_q fd vs fused: max_abs={errq:.3e} rel={relq:.3e}")

    def fn_k(kk):
        qo2, ko2 = ops.rope_qk(q, kk, freqs)
        return (qo2 * gq).sum() + (ko2 * gk).sum()
    gk_fd = fd_grad(fn_k, k)
    _, gk_fused = ops.rope_qk_backward(gq, gk, freqs)
    errk = (gk_fd.float() - gk_fused.float()).abs().max().item()
    relk = errk / gk_fd.float().abs().max().item()
    print(f"grad_k fd vs fused: max_abs={errk:.3e} rel={relk:.3e}")

    # --- 3) adaln_norm backward vs finite diff ---
    print("=== adaln_norm backward (gradcheck fd) ===")
    torch.manual_seed(2)
    B2, T, H2, W2, D2 = 2, 1, 4, 4, 64
    x = torch.randn(B2, T, H2, W2, D2, device="cuda", dtype=torch.float32)
    scale = torch.randn(B2 * T, D2, device="cuda", dtype=torch.float32) * 0.1
    shift = torch.randn(B2 * T, D2, device="cuda", dtype=torch.float32) * 0.1
    y = ops.adaln_norm(x, scale, shift, 1e-6)
    gy = torch.randn_like(y)
    bt_total, hw = B2 * T, H2 * W2

    def fn_x(xx):
        return (ops.adaln_norm(xx, scale, shift, 1e-6) * gy).sum()
    gx_fd = fd_grad(fn_x, x)
    dx, ds, dh = ops.adaln_norm_backward(gy, x, scale, bt_total, hw, 1e-6)
    errx = (gx_fd.float() - dx.float()).abs().max().item()
    relx = errx / gx_fd.float().abs().max().item()
    print(f"grad_x fd vs fused: max_abs={errx:.3e} rel={relx:.3e}")

    def fn_s(ss):
        return (ops.adaln_norm(x, ss, shift, 1e-6) * gy).sum()
    gs_fd = fd_grad(fn_s, scale)
    _, ds, _ = ops.adaln_norm_backward(gy, x, scale, bt_total, hw, 1e-6)
    errs = (gs_fd.float() - ds.float()).abs().max().item()
    rels = errs / gs_fd.float().abs().max().item()
    print(f"grad_scale fd vs fused: max_abs={errs:.3e} rel={rels:.3e}")


if __name__ == "__main__":
    main()
