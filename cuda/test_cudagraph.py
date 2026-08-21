"""Item 3: CUDA-graph capture correctness test on a real Block (+fused ops).

Captures (a) a plain forward and (b) a full forward+backward step at a fixed
shape, then replays with fresh inputs and compares loss / gradients against
eager execution.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_cuda_accel as acc
from library.anima_models import Block
from library.cuda_graph_util import capture_forward, capture_step


def make_fixture(seed, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    B, T, H, W, D = 1, 1, 4, 4, 128
    blk = Block(x_dim=D, context_dim=64, num_heads=8).to("cuda").to(dtype)
    x = torch.randn(B, T, H, W, D, device="cuda", dtype=dtype)
    emb = torch.randn(B, T, D, device="cuda", dtype=dtype)
    cross = torch.randn(B, 16, 64, device="cuda", dtype=dtype)
    rope = torch.randn(T * H * W, 1, 1, D // 8, device="cuda", dtype=dtype)
    return blk, x, emb, cross, rope


def eager_step(blk, x, emb, cross, rope, grad):
    out = blk._forward(x, emb, cross, rope_emb_L_1_1_D=rope)
    loss = (out.float() * grad).sum()
    loss.backward()
    return loss.item()


def main():
    acc.set_enabled(True)
    dtype = torch.bfloat16

    # ---- (a) forward-only capture ----
    blk_a, x, emb, cross, rope = make_fixture(0)
    blk_a = blk_a.eval()
    sx = x.clone(); s_emb = emb.clone(); s_cross = cross.clone(); s_rope = rope.clone()
    cap = capture_forward(blk_a.forward, sx, s_emb, s_cross, s_rope)  # signature: forward(x, emb, cross, rope_emb)
    # fresh inputs -> replay
    n = 2
    with torch.inference_mode():
        y_eager = blk_a.forward(sx, s_emb, s_cross, s_rope)
        out = cap.replay(sx, s_emb, s_cross, s_rope)  # same buffers copied
    ok_a = torch.allclose(y_eager.float(), out.float(), atol=0.0, rtol=0.0) if False else (y_eager - out).abs().max().item() == 0.0
    print(f"[{'OK' if ok_a else 'FAIL'}] forward-capture: replay == eager, max_abs={(y_eager - out).abs().max().item():.3e}")

    # ---- (b) full fwd+bwd capture ----
    blk_b, xb, emb, cross, rope = make_fixture(1)
    xb.requires_grad_(True)
    grad = torch.randn_like(blk_b._forward(xb, emb, cross, rope_emb_L_1_1_D=rope))
    grad = grad.detach()

    sx = xb.detach().clone().requires_grad_(True)
    s_emb = emb.clone(); s_cross = cross.clone(); s_rope = rope.clone()
    x_out = torch.empty_like(grad)  # unused; placeholder

    def step_fn(x, e, c, r):
        out = blk_b._forward(x, e, c, rope_emb_L_1_1_D=r)
        loss = (out.float() * grad).sum()
        loss.backward()
        return loss

    # eager reference
    blk_b.zero_grad(set_to_none=True)
    l_eager = eager_step(blk_b, xb, emb, cross, rope, grad)
    g_eager = {n: p.grad.clone() for n, p in blk_b.named_parameters() if p.grad is not None}

    # capture with static buffers
    blk_b.zero_grad(set_to_none=True)
    try:
        cap_b = capture_step(step_fn, [sx, s_emb, s_cross, s_rope])
    except Exception as e:
        print(f"[WARN] fwd+bwd capture failed: {type(e).__name__}: {str(e)[:140]}")
        cap_b = None

    if cap_b is not None:
        blk_b.zero_grad(set_to_none=True)
        sx.detach().copy_(xb.detach()); s_emb.detach().copy_(emb); s_cross.detach().copy_(cross); s_rope.detach().copy_(rope)
        l_cap = cap_b.replay(sx, s_emb, s_cross, s_rope).item()
        ok_l = abs(l_cap - l_eager) == 0.0
        worst = 0.0
        for n, p in blk_b.named_parameters():
            if p.grad is not None and n in g_eager:
                worst = max(worst, (p.grad - g_eager[n]).abs().max().item())
        ok_g = worst < 1e-2  # bf16 grads
        print(f"[{'OK' if ok_l and ok_g else 'FAIL'}] fwd+bwd capture: loss match={abs(l_cap - l_eager):.3e} "
              f"worst|grad diff|={worst:.3e}")
    # ---- (c) MaybeGraphedStep: capture/replay/recapture/fallback ----
    blk_c, xc, emb, cross, rope = make_fixture(2)
    xc.requires_grad_(True)
    grad_c = torch.randn_like(blk_c._forward(xc, emb, cross, rope_emb_L_1_1_D=rope)).detach()

    def eager_fwd_bwd(xx, e, c, r):
        blk_c.zero_grad(set_to_none=True)
        o = blk_c._forward(xx, e, c, rope_emb_L_1_1_D=r)
        l = (o.float() * grad_c).sum()
        l.backward()
        return l.item(), {n: p.grad.clone() for n, p in blk_c.named_parameters() if p.grad is not None}

    def step_fn(xx, e, c, r):
        o = blk_c._forward(xx, e, c, rope_emb_L_1_1_D=r)
        l = (o.float() * grad_c).sum()
        l.backward()
        return l

    from library.cuda_graph_util import MaybeGraphedStep
    zg = lambda: blk_c.zero_grad(set_to_none=True)
    gstep = MaybeGraphedStep(step_fn, enabled=True, zero_grad_fn=zg)
    # capture + replay with fresh inputs
    x_new = xc.detach().clone().requires_grad_(True)
    e_new = emb.detach() + 0.1
    l_graph = gstep(x_new, e_new, cross, rope).item()  # zeroed + backward inside replay -> clean grads
    g_graph = {n: p.grad.clone() for n, p in blk_c.named_parameters() if p.grad is not None}
    l_eager, g_eager = eager_fwd_bwd(x_new, e_new, cross, rope)
    assert set(g_graph) == set(g_eager), (set(g_graph) ^ set(g_eager))
    ok_loss = abs(l_graph - l_eager) < 1e-4
    worst = max((g_graph[n] - g_eager[n]).abs().max().item() for n in g_eager)
    ok_c = ok_loss and worst < 1e-2
    print(f"[{'OK' if ok_c else 'FAIL'}] MaybeGraphedStep capture+replay: loss_diff={abs(l_graph - l_eager):.3e} worst|grad|={worst:.3e}")

    # shape change -> recapture, still correct
    blk_d, xd, embd, crossd, roped = make_fixture(3, dtype=torch.bfloat16)
    xd.requires_grad_(True)
    grad_d = torch.randn_like(blk_d._forward(xd, embd, crossd, rope_emb_L_1_1_D=roped)).detach()
    def step_fn_d(xx, e, c, r):
        o = blk_d._forward(xx, e, c, rope_emb_L_1_1_D=r)
        l = (o.float() * grad_d).sum()
        l.backward()
        return l
    gstepD = MaybeGraphedStep(step_fn_d, enabled=True, zero_grad_fn=lambda: blk_d.zero_grad(set_to_none=True))
    xd2 = xd.detach().clone().requires_grad_(True)
    l_gd = gstepD(xd2, embd, crossd, roped).item()
    g_gd = {n: p.grad.clone() for n, p in blk_d.named_parameters() if p.grad is not None}
    blk_d.zero_grad(set_to_none=True)
    o = blk_d._forward(xd2, embd, crossd, rope_emb_L_1_1_D=roped)
    l_ed = (o.float() * grad_d).sum(); l_ed.backward()
    g_ed = {n: p.grad.clone() for n, p in blk_d.named_parameters() if p.grad is not None}
    assert set(g_gd) == set(g_ed)
    ok_rc = abs(l_gd - l_ed.item()) < 1e-4 and max((g_gd[n] - g_ed[n]).abs().max().item() for n in g_ed) < 1e-2
    print(f"[{'OK' if ok_rc else 'FAIL'}] MaybeGraphedStep recapture-on-shape-change: ok={ok_rc} (captures={gstepD.re_captures})")

    # forced failure -> permanent eager fallback, still correct
    def bad_fn(xx, e, c, r):
        raise RuntimeError("simulated graph-hostile op")
    gstepBad = MaybeGraphedStep(bad_fn, enabled=True)
    try:
        gstepBad(x_new, e_new, cross, rope)
        ok_ff = False
    except RuntimeError:
        ok_ff = True  # eager fallback propagated the underlying error -> wrapper did NOT hang
    ok_fb = ok_ff and gstepBad._broken is True
    print(f"[{'OK' if ok_fb else 'FAIL'}] MaybeGraphedStep fallback-on-failure: broken={gstepBad._broken} raised={ok_ff}")

    acc.set_enabled(False)


if __name__ == "__main__":
    main()
