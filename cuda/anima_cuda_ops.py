import torch

import anima_cuda_ops_ext as _EXT


def load_extension(verbose: bool = False):
    return _EXT


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return load_extension().rmsnorm_forward(input.contiguous(), weight.contiguous(), float(eps))


def rmsnorm_backward(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        load_extension().rmsnorm_backward(
            grad_output.contiguous(), input.contiguous(), weight.contiguous(), float(eps)
        )
    )


def rope_qk(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(load_extension().rope_forward(q.contiguous(), k.contiguous(), freqs.contiguous()))


def rope_qk_backward(grad_q: torch.Tensor, grad_k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        load_extension().rope_backward(grad_q.contiguous(), grad_k.contiguous(), freqs.contiguous())
    )


def noisy_input(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    return load_extension().noisy_input(latents.contiguous(), noise.contiguous(), timesteps.contiguous())


def rectified_flow_mse_loss(
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    weighting: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    return load_extension().rectified_flow_mse_loss(
        model_pred.contiguous(),
        latents.contiguous(),
        noise.contiguous(),
        weighting.contiguous(),
        loss_weights.contiguous(),
    )


def rectified_flow_mse_loss_backward(
    grad_output: torch.Tensor,
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    weighting: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    return load_extension().rectified_flow_mse_loss_backward(
        grad_output.contiguous(),
        model_pred.contiguous(),
        latents.contiguous(),
        noise.contiguous(),
        weighting.contiguous(),
        loss_weights.contiguous(),
    )


def adaln_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return load_extension().adaln_norm_forward(x.contiguous(), scale.contiguous(), shift.contiguous(), float(eps))


def adaln_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    bt_total: int,
    hw: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        load_extension().adaln_norm_backward(
            grad_output.contiguous(), x.contiguous(), scale.contiguous(), int(bt_total), int(hw), float(eps)
        )
    )


def adamw_step(
    grad: torch.Tensor,
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    bias_correction1: float,
    bias_correction2: float,
    clip_scale: float = 1.0,
) -> None:
    load_extension().adamw_step(
        grad.contiguous(), param.contiguous(), exp_avg.contiguous(), exp_avg_sq.contiguous(),
        float(lr), float(beta1), float(beta2), float(eps), float(weight_decay),
        float(bias_correction1), float(bias_correction2), float(clip_scale),
    )


def rmsnorm_ref(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (input.float() * torch.rsqrt(input.float().pow(2).mean(-1, keepdim=True) + eps)).to(input.dtype) * weight


def rope_qk_ref(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_dim = freqs.shape[-1]
    cos = torch.cos(freqs[: q.shape[1]]).to(q.dtype).view(1, q.shape[1], 1, rot_dim)
    sin = torch.sin(freqs[: q.shape[1]]).to(q.dtype).view(1, q.shape[1], 1, rot_dim)

    def apply(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        out = x_rot * cos + rotated * sin
        return torch.cat((out, x_pass), dim=-1)

    return apply(q), apply(k)


def noisy_input_ref(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
    return (1 - t) * latents + t * noise


def rectified_flow_mse_loss_ref(
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    weighting: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    target = noise - latents
    loss = (model_pred.float() - target.float()).pow(2)
    loss = loss.mean(list(range(1, loss.ndim)))
    return (loss * weighting.float().view(-1) * loss_weights.float().view(-1)).mean()


def rmsnorm_backward_ref(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    xf = input.float()
    wf = weight.float()
    gf = grad_output.float()
    n = xf.shape[-1]
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    gw = gf * wf
    dot = (gw * xf).sum(dim=-1, keepdim=True)
    grad_x = (inv * gw - xf * (inv.pow(3) / n) * dot).to(input.dtype)
    grad_w = (gf * (xf * inv)).sum(dim=tuple(range(grad_output.ndim - 1))).to(weight.dtype)
    return grad_x, grad_w


def rope_qk_backward_ref(grad_q: torch.Tensor, grad_k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_dim = freqs.shape[-1]
    cos = torch.cos(freqs[: grad_q.shape[1]]).to(grad_q.dtype).view(1, grad_q.shape[1], 1, rot_dim)
    sin = torch.sin(freqs[: grad_q.shape[1]]).to(grad_q.dtype).view(1, grad_q.shape[1], 1, rot_dim)

    def apply_inverse(g: torch.Tensor) -> torch.Tensor:
        g_rot = g[..., :rot_dim]
        g_pass = g[..., rot_dim:]
        # True transpose of non-interleaved RoPE:
        #   out[0:half] = cos1*g1 + sin2*g2 ; out[half:] = cos2*g2 - sin1*g1
        cos1, cos2 = cos.chunk(2, dim=-1)
        sin1, sin2 = sin.chunk(2, dim=-1)
        g1, g2 = g_rot.chunk(2, dim=-1)
        out = torch.cat((cos1 * g1 + sin2 * g2, cos2 * g2 - sin1 * g1), dim=-1)
        return torch.cat((out, g_pass), dim=-1)

    return apply_inverse(grad_q), apply_inverse(grad_k)


def rectified_flow_mse_loss_backward_ref(
    grad_output: torch.Tensor,
    model_pred: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    weighting: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    elems_per_sample = model_pred[0].numel()
    batch = model_pred.shape[0]
    target = noise - latents
    scale = 2.0 / (elems_per_sample * batch)
    w = (
        weighting.view(-1, *([1] * (model_pred.ndim - 1))).float()
        * loss_weights.view(-1, *([1] * (model_pred.ndim - 1))).float()
    )
    grad = (model_pred.float() - target.float()) * w * scale * grad_output.float()
    return grad.to(model_pred.dtype)


def adaln_norm_ref(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (B,T,H,W,D); scale/shift: (B*T, D) broadcast over H,W.
    B, T, H, W, D = x.shape
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    n = (xf - mean) / torch.sqrt(var + eps)
    s = scale.view(B, T, 1, 1, D).float()
    h = shift.view(B, T, 1, 1, D).float()
    return (n * (1.0 + s) + h).to(x.dtype)


def adaln_norm_backward_ref(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    bt_total: int,
    hw: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, W, D = x.shape
    xf = x.float()
    gy = grad_output.float()
    s = scale.view(B, T, 1, 1, D).float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    n = (xf - mean) / torch.sqrt(var + eps)  # normalized (B,T,H,W,D)
    dn = gy * (1.0 + s)

    sum_dn = dn.sum(-1, keepdim=True)
    sum_dn_n = (dn * n).sum(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    grad_x = rstd * (dn - sum_dn / D - n * (sum_dn_n / D))

    grad_scale = (gy * n).sum(dim=(2, 3))  # sum over H,W -> (B,T,D)
    grad_shift = gy.sum(dim=(2, 3))        # -> (B,T,D)
    grad_x = grad_x.to(x.dtype)
    grad_scale = grad_scale.to(scale.dtype)
    grad_shift = grad_shift.to(scale.dtype)
    grad_scale = grad_scale.reshape(bt_total, D)
    grad_shift = grad_shift.reshape(bt_total, D)
    return grad_x, grad_scale, grad_shift


def adamw_step_ref(
    grad: torch.Tensor,
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    bias_correction1: float,
    bias_correction2: float,
) -> None:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    step_size = lr / bias_correction1
    param.mul_(1.0 - lr * weight_decay).addcdiv_(exp_avg, denom, value=-step_size)
