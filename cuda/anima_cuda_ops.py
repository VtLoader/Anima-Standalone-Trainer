import torch

import anima_cuda_ops_ext as _EXT


def load_extension(verbose: bool = False):
    return _EXT


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return load_extension().rmsnorm_forward(input.contiguous(), weight.contiguous(), float(eps))


def rope_qk(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(load_extension().rope_forward(q.contiguous(), k.contiguous(), freqs.contiguous()))


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
