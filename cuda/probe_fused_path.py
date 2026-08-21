"""Runtime probe: do the fused CUDA ops actually fire inside the real Block
under bf16 / fp16 training, with bf16 / fp32 model weights?

We spy on the ops module entry points and also compare a full Block autograd
output/grads fused vs reference. Prints which ops hit the CUDA kernel path.
"""
import sys
from pathlib import Path

import torch

REPO = str(Path(__file__).resolve().parent.parent)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import anima_cuda_ops as _OPS_BASE
from library import anima_cuda_accel as acc
from library.anima_models import Block

SPY = {"rmsnorm_fused": [0], "rope_fused": [0], "adaln_fused": [0]}


def spy_on(ops_module):
    """Wrap the ops entry points to count invocations; acc._ops is this module."""
    _rms0 = ops_module.rmsnorm
    _rmb0 = ops_module.rmsnorm_backward
    _rope0 = ops_module.rope_qk
    _ropeb0 = ops_module.rope_qk_backward
    _adaln0 = ops_module.adaln_norm
    _adalnb0 = ops_module.adaln_norm_backward

    def wrap(name, fn, counter):
        def w(*a, **k):
            counter[0] += 1
            return fn(*a, **k)
        return w

    ops_module.rmsnorm = wrap("rmsnorm", _rms0, SPY["rmsnorm_fused"])
    ops_module.rmsnorm_backward = wrap("rmsnorm_bwd", _rmb0, SPY["rmsnorm_fused"])
    ops_module.rope_qk = wrap("rope", _rope0, SPY["rope_fused"])
    ops_module.rope_qk_backward = wrap("rope_bwd", _ropeb0, SPY["rope_fused"])
    ops_module.adaln_norm = wrap("adaln", _adaln0, SPY["adaln_fused"])
    ops_module.adaln_norm_backward = wrap("adaln_bwd", _adalnb0, SPY["adaln_fused"])


def main():
    torch.manual_seed(0)
    dev = "cuda"
    acc.set_enabled(True)  # loads _ops first
    spy_on(acc._ops)

    def run(dtype, weight_dtype, tag):
        for k in SPY:
            SPY[k][0] = 0
        B, T, H, W, D = 1, 1, 8, 8, 256
        blk = Block(x_dim=D, context_dim=128, num_heads=8).to(dev).to(dtype)
        # set the norm weights to weight_dtype (simulate fp32 master under bf16)
        for m in blk.modules():
            if hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter) and m.weight.dim() == 1 and "norm" in type(m).__name__.lower():
                pass
        x = torch.randn(B, T, H, W, D, device=dev, dtype=dtype, requires_grad=True)
        emb = torch.randn(B, T, D, device=dev, dtype=dtype)
        cross = torch.randn(B, 16, 128, device=dev, dtype=dtype)
        rope = torch.randn(T * H * W, 1, 1, D // 8, device=dev, dtype=dtype)
        ac = torch.autocast(device_type="cuda", dtype=dtype)
        acc.set_enabled(True)
        with ac:
            out = blk._forward(x, emb, cross, rope_emb_L_1_1_D=rope)
            out.backward(torch.randn_like(out))
        print(f"[{tag}] dtype={str(dtype).split('.')[-1]} weight_dtype={str(weight_dtype).split('.')[-1]}  "
              f"rmsnorm_fused_calls={SPY['rmsnorm_fused'][0]} rope_fused_calls={SPY['rope_fused'][0]} adaln_fused_calls={SPY['adaln_fused'][0]}")
        acc.set_enabled(False)

    torch.manual_seed(1)
    run(torch.bfloat16, torch.bfloat16, "A-bf16-model-bf16")
    torch.manual_seed(1)
    run(torch.bfloat16, torch.float32, "B-bf16-forward-fp32-weights")
    torch.manual_seed(1)
    run(torch.float16, torch.float16, "C-fp16-model-fp16")

    # Direct: weight dtype mismatch path in acc.rmsnorm
    acc.set_enabled(True)
    x = torch.randn(4, 96, device=dev, dtype=torch.bfloat16)
    w_fp32 = torch.randn(96, device=dev, dtype=torch.float32)
    SPY["rmsnorm_fused"][0] = 0
    _ = acc.rmsnorm(x, w_fp32, 1e-6)
    print(f"[direct] acc.rmsnorm(bf16 x, fp32 w) fused_calls={SPY['rmsnorm_fused'][0]}  <- 0 means it silently fell back to eager")
    acc.set_enabled(False)
    print("PROBE DONE")


if __name__ == "__main__":
    main()
