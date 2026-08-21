"""Quantify Fused-QKV: 3 separate linears vs 1 merged GEMM at Anima shapes.

- q/k/v proj each hidden->inner (model_dim == inner for Anima).
- Measures both the ideal fused (weight pre-concatenated) and the per-call
  concat variant actually available inside a training step where weights change.
- bf16, S = 4096 (1024px) and S = 16384 (2048px).
"""
import torch
from triton.testing import do_bench

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def bench(bs, S, D, inner, dtype=torch.bfloat16):
    x = torch.randn(bs, S, D, device=dev, dtype=dtype)
    wq = torch.randn(inner, D, device=dev, dtype=dtype)
    wk = torch.randn(inner, D, device=dev, dtype=dtype)
    wv = torch.randn(inner, D, device=dev, dtype=dtype)
    wqkv = torch.cat([wq, wk, wv], dim=0)  # (3*inner, D)

    def three():
        q = torch.nn.functional.linear(x, wq)
        k = torch.nn.functional.linear(x, wk)
        v = torch.nn.functional.linear(x, wv)
        return q, k, v

    def three_cat():  # concat weights each call, then one GEMM
        w = torch.cat([wq, wk, wv], dim=0)
        out = torch.nn.functional.linear(x, w)
        return out.chunk(3, dim=-1)

    def fused():
        out = torch.nn.functional.linear(x, wqkv)
        return out.chunk(3, dim=-1)

    three()
    three_cat()
    fused()
    torch.cuda.synchronize()
    t3 = do_bench(three, warmup=200, rep=2000)
    tc = do_bench(three_cat, warmup=200, rep=2000)
    tf = do_bench(fused, warmup=200, rep=2000)
    print(f"S={S:6d} D={D:5d} inner={inner:5d}: 3xGEMM={t3:.4f}ms  fused(idl)={tf:.4f}ms (x{t3 / tf:.2f})  "
          f"3xGEMM+cat={tc:.4f}ms (x{t3 / tc:.2f} vs 3x)")


print("bf16, model_dim=1536, inner=1536")
for S in (4096, 16384):
    bench(1, S, 1536, 1536)
print("bf16, model_dim=3072, head_dim=96 -> inner=1536, k/v same")
for S in (4096, 16384):
    bench(1, S, 1536, 1536)
