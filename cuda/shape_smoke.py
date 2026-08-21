"""Shape-flow smoke test for the Anima generation/training forward path.

Builds a small MiniTrainDIT from the real model code and checks the latent
shape round-trip (B,C,T,H,W -> DiT -> unpatchify -> same shape), the CFG
batch-doubling path, and the fused-CUDA path against the torch reference.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from library import anima_models as M


def main():
    device = "cuda"
    torch.manual_seed(0)

    # Small representative config (patch_spatial=2, patch_temporal=1, image)
    def make_dit(**overrides):
        cfg = dict(
            max_img_h=128,
            max_img_w=128,
            max_frames=1,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=False,
            model_channels=64,
            num_blocks=1,
            num_heads=4,
            mlp_ratio=4.0,
            crossattn_emb_channels=32,
            pos_emb_cls="rope3d",
            use_llm_adapter=False,
        )
        cfg.update(overrides)
        return M.MiniTrainDIT(**cfg)

    dit = make_dit().to(device).to(torch.bfloat16).eval()

    # === 1) single-image latent round trip ===
    B, C, T, H, W = 1, 16, 1, 16, 16  # 128px image -> latent 16x16
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
    t = torch.rand(B, device=device, dtype=torch.bfloat16).clamp(1e-5, 1 - 1e-5)
    cross = torch.randn(B, 8, 32, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = dit(x, t, cross)
    ok1 = tuple(out.shape) == (B, C, T, H, W)
    print(f"[{'OK' if ok1 else 'FAIL'}] single latent round-trip: {tuple(out.shape)} == {B, C, T, H, W}")
    if tuple(out.shape) != (B, C, T, H, W):
        raise SystemExit(1)

    # === 2) CFG batch-doubling (B=2) -> same round trip ===
    B2 = 2
    x2 = torch.randn(B2, C, T, H, W, device=device, dtype=torch.bfloat16)
    t2 = torch.rand(B2, device=device, dtype=torch.bfloat16).clamp(1e-5, 1 - 1e-5)
    cross2 = torch.randn(B2, 8, 32, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out2 = dit(x2, t2, cross2)
    ok2 = tuple(out2.shape) == (B2, C, T, H, W)
    print(f"[{'OK' if ok2 else 'FAIL'}] CFG-doubled latent round-trip: {tuple(out2.shape)} == {B2, C, T, H, W}")

    # === 3) concat_padding_mask variant (model appends the mask channel) ===
    dit3 = make_dit(concat_padding_mask=True).to(device).to(torch.bfloat16).eval()
    x3 = torch.randn(B, 16, T, H, W, device=device, dtype=torch.bfloat16)  # base latents; model adds +1
    pm = torch.zeros(B, 1, H, W, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out3 = dit3(x3, t, cross, padding_mask=pm)
    ok3 = tuple(out3.shape) == (B, 16, T, H, W)
    print(f"[{'OK' if ok3 else 'FAIL'}] concat_padding_mask round-trip: {tuple(out3.shape)} == {B, 16, T, H, W}")

    # === 4) fused-CUDA path (accel on) vs reference Block, autograd through ===
    from library import anima_cuda_accel as acc
    blk = M.Block(x_dim=64, context_dim=32, num_heads=4).to(device).to(torch.float32)
    xf = torch.randn(B, T, H, W, 64, device=device, dtype=torch.float32, requires_grad=True)
    emb = torch.randn(B, T, 64, device=device, dtype=torch.float32)
    crossf = torch.randn(B, 8, 32, device=device, dtype=torch.float32)
    L = T * H * W
    rope = torch.randn(L, 1, 1, 16, device=device, dtype=torch.float32)  # head_dim = 64//4

    acc.set_enabled(False)
    out_ref = blk._forward(xf, emb, crossf, rope_emb_L_1_1_D=rope)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    gx_ref = xf.grad.clone()
    gq_ref = blk.self_attn.q_proj.weight.grad.clone()
    acc.set_enabled(True)
    gx = None
    try:
        xf2 = xf.detach().clone().requires_grad_(True)
        blk2 = M.Block(x_dim=64, context_dim=32, num_heads=4).to(device).to(torch.float32)
        blk2.load_state_dict(blk.state_dict())
        out_fus = blk2._forward(xf2, emb, crossf, rope_emb_L_1_1_D=rope)
        out_fus.backward(g)
        gx = (xf2.grad - gx_ref).abs().max().item()
    except Exception as e:
        print("  (fused path error, likely wheel not rebuilt yet):", type(e).__name__, str(e)[:120])
        gx = None
    ok4 = gx is not None and gx < 1e-4
    print(f"[{'OK' if ok4 else 'FAIL'}] fused Block autograd vs reference (max|x_grad err|) = {gx}")
    acc.set_enabled(False)

    print("SHAPE SMOKE ALL PASS" if (ok1 and ok2 and ok3 and ok4) else "SHAPE SMOKE SOME FAILED")


if __name__ == "__main__":
    main()
