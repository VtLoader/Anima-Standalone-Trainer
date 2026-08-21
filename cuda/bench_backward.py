import json

import torch
from triton.testing import do_bench

import anima_cuda_ops as ops


def run_for_time(fn, warmup_ms, rep_ms):
    # triton.testing.do_bench: warmup/rep in ms, returns mean per-iteration ms.
    return float(do_bench(fn, warmup=int(warmup_ms), rep=int(rep_ms)))


def bench(name, ref, cur, warmup_ms=500.0, rep_ms=5000.0):
    # Warm both first so one-time JIT/alloc effects are excluded.
    run_for_time(ref, 50, 100)
    run_for_time(cur, 50, 100)
    ref_ms = run_for_time(ref, warmup_ms, rep_ms)
    cuda_ms = run_for_time(cur, warmup_ms, rep_ms)
    out = {
        "name": name,
        "ref_ms": round(ref_ms, 4),
        "cuda_ms": round(cuda_ms, 4),
        "speedup": round(ref_ms / cuda_ms, 2),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main():
    warmup_ms, rep_ms = 500.0, 5000.0
    for dtype, dname in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        torch.manual_seed(0)
        B, S, H, D = 2, 4096, 16, 96
        x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        w = torch.randn(D, device="cuda", dtype=dtype) + 0.5
        g = torch.randn_like(x)
        eps = 1e-5

        bench(
            f"rmsnorm_backward_{dname}",
            lambda: ops.rmsnorm_backward_ref(g, x, w, eps),
            lambda: ops.rmsnorm_backward(g, x, w, eps),
            warmup_ms, rep_ms,
        )

        q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        freqs = torch.randn(S, D, device="cuda", dtype=dtype)
        gq = torch.randn_like(q)
        gk = torch.randn_like(k)
        bench(
            f"rope_backward_{dname}",
            lambda: ops.rope_qk_backward_ref(gq, gk, freqs),
            lambda: ops.rope_qk_backward(gq, gk, freqs),
            warmup_ms, rep_ms,
        )

        B3, C, T, H3, W3 = 2, 16, 1, 128, 128
        mp = torch.randn(B3, C, T, H3, W3, device="cuda", dtype=dtype) * 0.3
        lat = torch.randn_like(mp)
        noi = torch.randn_like(mp)
        wgt = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
        lw = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
        go = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        bench(
            f"loss_backward_{dname}",
            lambda: ops.rectified_flow_mse_loss_backward_ref(go, mp, lat, noi, wgt, lw),
            lambda: ops.rectified_flow_mse_loss_backward(go, mp, lat, noi, wgt, lw),
            warmup_ms, rep_ms,
        )


if __name__ == "__main__":
    main()
