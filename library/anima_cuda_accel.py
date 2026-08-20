import logging
from typing import Optional

import torch


logger = logging.getLogger(__name__)

_enabled = False
_ops = None


def set_enabled(enabled: bool) -> None:
    global _enabled, _ops
    _enabled = bool(enabled)
    if not _enabled:
        return
    if _ops is not None:
        return
    try:
        import anima_cuda_ops as ops
    except Exception as e:
        raise RuntimeError(
            "--enable_cuda_acceleration requires the anima-cuda-ops wheel to be installed "
            "in the active Python environment."
        ) from e
    _ops = ops
    logger.info("Anima CUDA acceleration enabled via anima-cuda-ops")


def is_enabled() -> bool:
    return _enabled and _ops is not None


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        y = _ops.rmsnorm(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.needs_weight_grad = ctx.needs_input_grad[1]
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        xf = x.float()
        wf = weight.float()
        gf = grad_output.float()
        n = xf.shape[-1]
        inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        gw = gf * wf
        dot = (gw * xf).sum(dim=-1, keepdim=True)
        grad_x = inv * gw - xf * (inv.pow(3) / n) * dot
        grad_w = None
        if ctx.needs_weight_grad:
            grad_w = (gf * (xf * inv)).sum(dim=tuple(range(grad_output.ndim - 1))).to(weight.dtype)
        return grad_x.to(x.dtype), grad_w, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not is_enabled() or not x.is_cuda or not weight.is_cuda or x.dtype != weight.dtype:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
    return _RMSNormFn.apply(x, weight, float(eps))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _RopeQKFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
        freqs = freqs.to(dtype=q.dtype)
        q_out, k_out = _ops.rope_qk(q, k, freqs)
        ctx.save_for_backward(freqs)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        (freqs,) = ctx.saved_tensors
        rot_dim = freqs.shape[-1]
        cos = torch.cos(freqs[: grad_q.shape[1]]).to(grad_q.dtype).view(1, grad_q.shape[1], 1, rot_dim)
        sin = torch.sin(freqs[: grad_q.shape[1]]).to(grad_q.dtype).view(1, grad_q.shape[1], 1, rot_dim)

        def apply_inverse(g: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if g is None:
                return None
            g_rot = g[..., :rot_dim]
            g_pass = g[..., rot_dim:]
            out = g_rot * cos - _rotate_half(g_rot) * sin
            return torch.cat((out, g_pass), dim=-1)

        return apply_inverse(grad_q), apply_inverse(grad_k), None


def rope_qk(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_enabled() or not q.is_cuda or not k.is_cuda or not freqs.is_cuda:
        raise RuntimeError("rope_qk should only be called when CUDA acceleration is enabled")
    if freqs.ndim == 4:
        freqs = freqs[:, 0, 0, :]
    return _RopeQKFn.apply(q, k, freqs)


def noisy_input(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    if not is_enabled() or not latents.is_cuda or latents.requires_grad or noise.requires_grad or timesteps.requires_grad:
        t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
        return (1 - t) * latents + t * noise
    return _ops.noisy_input(latents, noise, timesteps)


class _RectifiedFlowMSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_pred, latents, noise, weighting, loss_weights):
        loss = _ops.rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights)
        ctx.save_for_backward(model_pred, latents, noise, weighting, loss_weights)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        model_pred, latents, noise, weighting, loss_weights = ctx.saved_tensors
        elems_per_sample = model_pred[0].numel()
        batch = model_pred.shape[0]
        target = noise - latents
        scale = (2.0 / (elems_per_sample * batch))
        w = (weighting.view(-1, *([1] * (model_pred.ndim - 1))).float() *
             loss_weights.view(-1, *([1] * (model_pred.ndim - 1))).float())
        grad = (model_pred.float() - target.float()) * w * scale * grad_output.float()
        return grad.to(model_pred.dtype), None, None, None, None


def rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights) -> torch.Tensor:
    if not is_enabled() or not model_pred.is_cuda:
        target = noise - latents
        loss = (model_pred.float() - target.float()).pow(2)
        loss = loss.mean(list(range(1, loss.ndim)))
        return (loss * weighting.float().view(-1) * loss_weights.float().view(-1)).mean()
    return _RectifiedFlowMSELossFn.apply(model_pred, latents, noise, weighting, loss_weights)
