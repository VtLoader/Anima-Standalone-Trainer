import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_cuda_accel as acc
from library.anima_models import Block


def report(name, a, b, tag=""):
    """max_abs, Linf_rel, and relative RMS between optimized (a) and original (b)."""
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    linf_rel = max_abs / max(bf.abs().max().item(), 1e-9)
    rms_err = diff.square().mean().sqrt().item()
    rms_ref = bf.square().mean().sqrt().item()
    rms_rel = rms_err / max(rms_ref, 1e-9)
    print(f"[{name}{tag}] max_abs={max_abs:.3e} linf_rel={linf_rel:.3e} rms_rel={rms_rel:.3e}")


def run_block(path, enabled, dtype, ctrl_seed):
    from library import anima_cuda_accel as a
    torch.manual_seed(ctrl_seed)
    B, T, H, W, D = 1, 1, 8, 8, 256
    num_heads = 8
    head_dim = D // num_heads
    Cx, N = 384, 32
    blk = Block(x_dim=D, context_dim=Cx, num_heads=num_heads, mlp_ratio=4.0, use_adaln_lora=False).to(path).to(dtype)
    x = torch.randn(B, T, H, W, D, device=path, dtype=dtype, requires_grad=True)
    emb = torch.randn(B, T, D, device=path, dtype=dtype)
    cross = torch.randn(B, N, Cx, device=path, dtype=dtype)
    L = T * H * W
    rope = torch.randn(L, 1, 1, head_dim, device=path, dtype=dtype)

    a.set_enabled(bool(enabled))
    out = blk._forward(x, emb, cross, rope_emb_L_1_1_D=rope, use_fp32=False)
    # full backward through the block
    grad = torch.randn_like(out)
    out.backward(grad)
    grads = {
        "out": out.clone(),
        "x_grad": x.grad.clone() if x.grad is not None else None,
    }
    for name, p in blk.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.clone()
    a.set_enabled(False)
    return grads


def compare_block(dtype, tag):
    g0 = run_block("cuda", False, dtype, 1234)  # original torch path
    torch.cuda.empty_cache()
    g1 = run_block("cuda", True, dtype, 1234)   # fused CUDA path
    assert g0.keys() == g1.keys(), (g0.keys(), g1.keys())
    for k in g0.keys():
        if g0[k] is None or g1[k] is None:
            continue
        report(f"block.{k}", g1[k], g0[k], tag)


def run_elementwise(dtype):
    torch.manual_seed(7)
    B3, C, T, H, W = 2, 16, 1, 64, 64
    lat = torch.randn(B3, C, T, H, W, device="cuda", dtype=dtype)
    noi = torch.randn_like(lat)
    t = torch.rand(B3, device="cuda", dtype=dtype).clamp(1e-5, 1 - 1e-5)

    acc.set_enabled(False)
    n_orig = acc.noisy_input(lat, noi, t)
    acc.set_enabled(True)
    n_fused = acc.noisy_input(lat, noi, t)
    acc.set_enabled(False)
    report("noisy_input", n_fused, n_orig, f"[{str(dtype).split('.')[-1]}]")

    mp = torch.randn_like(lat) * 0.3
    wgt = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
    lw = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
    acc.set_enabled(False)
    loss_orig = acc.rectified_flow_mse_loss(mp, lat, noi, wgt, lw)
    acc.set_enabled(True)
    loss_fused = acc.rectified_flow_mse_loss(mp, lat, noi, wgt, lw)
    acc.set_enabled(False)
    report("loss_fwd", loss_fused, loss_orig, f"[{str(dtype).split('.')[-1]}]")

    # loss backward grad w.r.t. model_pred (linear, so compare directly)
    import anima_cuda_ops as ops
    acc.set_enabled(False)
    mp_o = mp.detach().requires_grad_(True)
    target = (noi - lat).detach()
    lo = ((mp_o.float() - target.float()) ** 2).mean(list(range(1, mp.ndim)))
    lo = (lo * wgt.float().view(-1) * lw.float().view(-1)).mean()
    lo.backward()
    g_orig = mp_o.grad
    go = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    g_fused = ops.rectified_flow_mse_loss_backward(go, mp, lat, noi, wgt, lw)
    report("loss_bwd", g_fused, g_orig, f"[{str(dtype).split('.')[-1]}]")


def main():
    for dtype, tag in [(torch.bfloat16, "[bf16]"), (torch.float16, "[fp16]"), (torch.float32, "[fp32]")]:
        print(f"===== dtype {tag} =====")
        compare_block(dtype, tag)
        run_elementwise(dtype)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
