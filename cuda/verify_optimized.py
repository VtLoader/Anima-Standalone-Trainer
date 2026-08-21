import torch
import anima_cuda_ops as ops


def check(name, a, b, tol_abs=1e-3, tol_rel=5e-2):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    rel = (diff / (b.abs() + 1e-8)).max().item()
    maxabs = diff.max().item()
    denom = max(b.abs().max().item(), 1e-6)
    linf_rel = maxabs / denom
    ok = (maxabs <= tol_abs) or (linf_rel <= tol_rel)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max_abs={maxabs:.3e} linf_rel={linf_rel:.3e}")
    return ok


def test_dtype(dtype, name):
    ok = True
    torch.manual_seed(0)

    # --- RMSNorm forward + backward ---
    B, S, H, D = 2, 256, 16, 96
    x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) + 0.5
    eps = 1e-5
    y = ops.rmsnorm(x, w, eps)
    y_ref = ops.rmsnorm_ref(x, w, eps)
    ok &= check(f"rmsnorm_fwd_{name}", y, y_ref)

    g = torch.randn_like(x)
    gx, gw = ops.rmsnorm_backward(g, x, w, eps)
    gx_ref, gw_ref = ops.rmsnorm_backward_ref(g, x, w, eps)
    ok &= check(f"rmsnorm_bwd_gx_{name}", gx, gx_ref)
    ok &= check(f"rmsnorm_bwd_gw_{name}", gw, gw_ref)

    # --- RoPE forward + backward ---
    B2, S2, H2, D2 = 2, 128, 8, 96
    q = torch.randn(B2, S2, H2, D2, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    freqs = torch.randn(S2, D2, device="cuda", dtype=dtype)
    qo, ko = ops.rope_qk(q, k, freqs)
    qor, kor = ops.rope_qk_ref(q, k, freqs)
    ok &= check(f"rope_fwd_q_{name}", qo, qor)
    ok &= check(f"rope_fwd_k_{name}", ko, kor)

    gqo = torch.randn_like(q)
    gko = torch.randn_like(k)
    gq_c, gk_c = ops.rope_qk_backward(gqo, gko, freqs)
    gq_r, gk_r = ops.rope_qk_backward_ref(gqo, gko, freqs)
    ok &= check(f"rope_bwd_q_{name}", gq_c, gq_r)
    ok &= check(f"rope_bwd_k_{name}", gk_c, gk_r)

    # partial rot_dim (pass-through) case
    Dp = 128
    qp = torch.randn(B2, S2, H2, Dp, device="cuda", dtype=dtype)
    kp = torch.randn_like(qp)
    freqs_p = torch.randn(S2, 96, device="cuda", dtype=dtype)
    qp_c, kp_c = ops.rope_qk(qp, kp, freqs_p)
    qp_r, kp_r = ops.rope_qk_ref(qp, kp, freqs_p)
    ok &= check(f"rope_fwd_partial_q_{name}", qp_c, qp_r)
    ok &= check(f"rope_fwd_partial_k_{name}", kp_c, kp_r)
    gqo2 = torch.randn_like(qp)
    gko2 = torch.randn_like(kp)
    gq_c2, gk_c2 = ops.rope_qk_backward(gqo2, gko2, freqs_p)
    gq_r2, gk_r2 = ops.rope_qk_backward_ref(gqo2, gko2, freqs_p)
    ok &= check(f"rope_bwd_partial_q_{name}", gq_c2, gq_r2)
    ok &= check(f"rope_bwd_partial_k_{name}", gk_c2, gk_r2)

    # --- noisy input ---
    B3, C, T, H3, W3 = 2, 16, 1, 64, 64
    lat = torch.randn(B3, C, T, H3, W3, device="cuda", dtype=dtype)
    noi = torch.randn_like(lat)
    t = torch.rand(B3, device="cuda", dtype=dtype).clamp(1e-5, 1 - 1e-5)
    ok &= check(f"noisy_{name}", ops.noisy_input(lat, noi, t), ops.noisy_input_ref(lat, noi, t))

    # --- MSE loss forward + backward ---
    mp = torch.randn_like(lat) * 0.3
    wgt = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
    lw = torch.rand(B3, device="cuda", dtype=dtype).clamp_min(0.1)
    loss_c = ops.rectified_flow_mse_loss(mp, lat, noi, wgt, lw).float()
    loss_r = ops.rectified_flow_mse_loss_ref(mp, lat, noi, wgt, lw).float()
    ok &= check(f"loss_fwd_{name}", loss_c.view(1), loss_r.view(1), tol_abs=1e-2, tol_rel=5e-2)
    go = torch.tensor([1.3], device="cuda", dtype=torch.float32)
    gmp_c = ops.rectified_flow_mse_loss_backward(go, mp, lat, noi, wgt, lw)
    gmp_r = ops.rectified_flow_mse_loss_backward_ref(go, mp, lat, noi, wgt, lw)
    ok &= check(f"loss_bwd_{name}", gmp_c, gmp_r)

    return ok


def main():
    all_ok = True
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        all_ok &= test_dtype(dtype, name)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")


if __name__ == "__main__":
    main()
