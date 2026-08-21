"""Item 4: fused loss path (autograd Function, save-diff-only) vs torch reference.

Verifies the DIFF-SAVING autograd path (used in training) produces the same loss
and same model_pred gradient as the pure-torch reference autograd.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_cuda_accel as acc


def ref_autograd(mp, lat, noi, wgt, lw):
    mp2 = mp.detach().clone().requires_grad_(True)
    target = noi - lat
    loss = (mp2.float() - target.float()).pow(2)
    loss = loss.mean(list(range(1, loss.ndim)))
    loss = (loss * wgt.float().view(-1) * lw.float().view(-1)).mean()
    loss.backward()
    return loss.item(), mp2.grad


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        for B in (1, 2):
            C, T, H, W = 16, 1, 32, 32
            mp = torch.randn(B, C, T, H, W, device=dev, dtype=dtype).requires_grad_(True)
            lat = torch.randn_like(mp).detach()
            noi = torch.randn_like(mp).detach()
            wgt = torch.rand(B, device=dev, dtype=dtype).clamp_min(0.1)
            lw = torch.rand(B, device=dev, dtype=dtype).clamp_min(0.1)

            l_ref, g_ref = ref_autograd(mp, lat, noi, wgt, lw)

            acc.set_enabled(True)
            mp2 = mp.detach().clone().requires_grad_(True)
            loss = acc.rectified_flow_mse_loss(mp2, lat, noi, wgt, lw)
            loss.backward()
            acc.set_enabled(False)
            l_fused = loss.item()
            d_loss = abs(l_fused - l_ref)
            d_grad = (mp2.grad.float() - g_ref.float()).abs().max().item()
            rel = d_grad / g_ref.float().abs().max().item()
            ok = d_loss < 1e-4 and rel < (0.05 if dtype != torch.float32 else 1e-5)
            print(f"[{'OK' if ok else 'FAIL'}] {name} B={B}: loss_diff={d_loss:.3e} grad_max_abs={d_grad:.3e} grad_rel={rel:.3e}")
            if not ok:
                raise SystemExit(1)
    print("LOSS SAVE-DIFF TEST ALL PASS")


if __name__ == "__main__":
    main()
