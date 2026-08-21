"""Runtime test for the FusedAdamW fixes.

1) default (plain AdamW): fused path == stock torch path (fp32/CUDA)
2) amsgrad=True: falls back to stock (no silent divergence / no crash)
3) maximize=True: falls back to stock
4) sparse grads: whole-optimizer fallback, no partial double-step / no crash
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_cuda_accel as acc


def clone_param(p):
    return p.detach().clone().requires_grad_(True)


def run(fused_enabled, params, **opt_kw):
    acc.set_enabled(fused_enabled)
    opt = acc.FusedAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **opt_kw)
    opt.step()
    return [p.detach().clone() for p in params]


def main():
    torch.manual_seed(0)
    dev = "cuda"

    # --- 1) default plain AdamW: fused == stock ---
    pa1 = torch.randn(64, 64, device=dev, dtype=torch.float32, requires_grad=True)
    pa1.grad = torch.randn_like(pa1)
    pa2 = clone_param(pa1); pa2.grad = pa1.grad.clone()
    with torch.no_grad():
        out_fused = run(True, [pa1])[0]
        out_stock = run(False, [pa2])[0]
    d = (out_fused - out_stock).abs().max().item()
    ok1 = d < 1e-6
    print(f"[{'OK' if ok1 else 'FAIL'}] plain AdamW fused vs stock: max_abs={d:.3e}")

    # --- 2) amsgrad=True with accel on: must NOT silently diverge, must match stock ---
    pb1 = torch.randn(64, 64, device=dev, dtype=torch.float32, requires_grad=True)
    pb1.grad = torch.randn_like(pb1)
    pb2 = clone_param(pb1); pb2.grad = pb1.grad.clone()
    out_a = run(True, [pb1], amsgrad=True)[0]
    out_b = run(False, [pb2], amsgrad=True)[0]
    d = (out_a - out_b).abs().max().item()
    ok2 = d < 1e-12  # both use stock torch path -> bit-identical
    print(f"[{'OK' if ok2 else 'FAIL'}] amsgrad=True fused-enable == stock: max_abs={d:.3e}")

    # --- 3) maximize=True with accel on ---
    pc1 = torch.randn(64, 64, device=dev, dtype=torch.float32, requires_grad=True)
    pc1.grad = torch.randn_like(pc1)
    pc2 = clone_param(pc1); pc2.grad = pc1.grad.clone()
    out_m1 = run(True, [pc1], maximize=True)[0]
    out_m2 = run(False, [pc2], maximize=True)[0]
    d = (out_m1 - out_m2).abs().max().item()
    ok3 = d < 1e-12
    print(f"[{'OK' if ok3 else 'FAIL'}] maximize=True fused-enable == stock: max_abs={d:.3e}")

    # --- 4) sparse grad: fallback to stock -> identical behavior/crash, and
    #        no partial fused update (dense params are not double-stepped) ---
    def run_mixed(fused_enabled, dense, sp):
        acc.set_enabled(fused_enabled)
        opt = acc.FusedAdamW([dense, sp], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        try:
            opt.step()
            raise SystemExit("expected RuntimeError for sparse AdamW step")
        except RuntimeError as e:
            return dense.detach().clone(), str(e)

    torch.manual_seed(1234)
    dense0 = torch.randn(32, 32, device=dev, dtype=torch.float32, requires_grad=True)
    dense0.grad = torch.randn_like(dense0)
    sp0 = torch.randn(16, 16, device=dev, dtype=torch.float32, requires_grad=True)
    sp0.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]], device=dev),
        torch.randn(2, device=dev, dtype=torch.float32), sp0.shape)

    d_on = clone_param(dense0); d_on.grad = dense0.grad.clone()
    s_on = clone_param(sp0); s_on.grad = sp0.grad.clone()
    d_off = clone_param(dense0); d_off.grad = dense0.grad.clone()
    s_off = clone_param(sp0); s_off.grad = sp0.grad.clone()

    d_on_out, msg_on = run_mixed(True, d_on, s_on)
    d_off_out, msg_off = run_mixed(False, d_off, s_off)
    ok4a = "sparse" in msg_on.lower() and "sparse" in msg_off.lower()
    delta = (d_on_out - d_off_out).abs().max().item()
    ok4b = delta == 0.0  # dense never double-stepped in either path
    print(f"[{'OK' if ok4a else 'FAIL'}] mixed sparse: both paths raise stock AdamW sparse error")
    print(f"[{'OK' if ok4b else 'FAIL'}] mixed sparse: dense param identical fused-enable vs disabled (no double-step), max_abs={delta:.3e}")
    ok4 = ok4a and ok4b

    acc.set_enabled(False)
    print("FUSED ADAMW TEST ALL PASS" if (ok1 and ok2 and ok3 and ok4) else "FUSED ADAMW TEST SOME FAILED")


if __name__ == "__main__":
    main()
