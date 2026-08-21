"""Item 2a: fused global-norm grad clip inside FusedAdamW.

Checks that:
  A) plain fused step (no clip) still matches previous behaviour (clip_scale=1).
  B) manual clip_grad_norm_ + step == fused max_grad_norm step (numerically).
  C) optimizer_handles_clip() reports correctly for the relevant optimizers.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_cuda_accel as acc


def make_params(seed=0):
    torch.manual_seed(seed)
    p = torch.randn(1 << 18, device="cuda", dtype=torch.float32).requires_grad_()
    p.grad = torch.randn_like(p) * 3.0  # large grads -> clipping kicks in
    return p


def run_fused(fused_enabled, p, *, max_grad_norm=None, manual_clip=None):
    acc.set_enabled(fused_enabled)
    opt = acc.FusedAdamW([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                         max_grad_norm=max_grad_norm)
    if manual_clip is not None:
        torch.nn.utils.clip_grad_norm_([p], manual_clip)
    opt.step()
    return p.detach().clone()


def main():
    dev = "cuda"

    # A) clip_scale=1 path unchanged vs stock
    p1 = make_params(1); p2 = make_params(1)
    r_fused = run_fused(True, p1)
    r_stock = run_fused(False, p2)
    dA = (r_fused - r_stock).abs().max().item()
    okA = dA < 1e-6
    print(f"[{'OK' if okA else 'FAIL'}] A) plain fused (clip_scale=1) == stock, max_abs={dA:.3e}")

    # B) manual clip + step == fused max_grad_norm step
    max_norm = 1.0
    pB1 = make_params(2); pB2 = make_params(2)
    r_manual = run_fused(True, pB1, manual_clip=max_norm)
    r_fusedclip = run_fused(True, pB2, max_grad_norm=max_norm)
    dB = (r_manual - r_fusedclip).abs().max().item()
    okB = dB < 1e-6
    print(f"[{'OK' if okB else 'FAIL'}] B) manual-clip+step == fused max_grad_norm step, max_abs={dB:.3e}")

    # B2) with a moderate norm (clip_scale < 1 actually triggered)
    pC1 = make_params(3); pC2 = make_params(3)
    gn = acc.global_grad_norm([pC1])
    r_m2 = run_fused(True, pC1, manual_clip=max_norm)
    r_f2 = run_fused(True, pC2, max_grad_norm=max_norm)
    dC = (r_m2 - r_f2).abs().max().item()
    okC = dC < 1e-6
    print(f"[{'OK' if okC else 'FAIL'}] B2) grad_norm={gn:.3f}; manual==fused(max_norm={max_norm}), max_abs={dC:.3e}")

    # C) optimizer_handles_clip
    pD = make_params(4)
    acc.set_enabled(True)
    o_fused = acc.FusedAdamW([pD], lr=1e-3, max_grad_norm=1.0)
    o_no = acc.FusedAdamW([pD], lr=1e-3, max_grad_norm=None)
    okD1 = acc.optimizer_handles_clip(o_fused) is True
    okD2 = acc.optimizer_handles_clip(o_no) is False
    acc.set_enabled(False)
    okD3 = acc.optimizer_handles_clip(o_fused) is False  # accel off -> falls back
    okD = okD1 and okD2 and okD3
    print(f"[{'OK' if okD else 'FAIL'}] C) optimizer_handles_clip: fused-clip-on=True({okD1}) no-clip=False({okD2}) off=False({okD3})")

    acc.set_enabled(False)
    print("FUSED CLIP TEST ALL PASS" if (okA and okB and okC and okD) else "FUSED CLIP TEST SOME FAILED")


if __name__ == "__main__":
    main()
