import logging

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


_warned_fallback = False


def _warn_once_fallback(detail: str) -> None:
    """Surface silent fusion skips so 'acceleration enabled but not applied' is
    visible in the log instead of silently running eager ops."""
    global _warned_fallback
    if _warned_fallback:
        return
    _warned_fallback = True
    logger.warning(
        "anima-cuda-ops fusion skipped for %s (eager fallback). The fused kernel "
        "requires CUDA tensors with matching dtype; mixed x/weight dtypes or CPU "
        "inputs silently bypass it. Keep model weights in the same dtype as "
        "activations (e.g. bf16) to get the full acceleration.",
        detail,
    )


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
        grad_x, grad_w = _ops.rmsnorm_backward(grad_output, x, weight, eps)
        if not ctx.needs_weight_grad:
            grad_w = None
        return grad_x, grad_w, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not is_enabled() or not x.is_cuda or not weight.is_cuda:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
    if x.dtype != weight.dtype:
        _warn_once_fallback("rmsnorm (mixed x/weight dtype)")
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight
    return _RMSNormFn.apply(x, weight, float(eps))


class _RopeQKFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
        # Angles are passed in fp32 (already fp32 from the positional embedder,
        # so this is a no-op): the CUDA kernel computes cos/sin from fp32 angles
        # exactly like the torch reference. Casting to q.dtype here would round
        # the angles to bf16 and also waste a conversion every call.
        freqs = freqs.to(dtype=torch.float32)
        q_out, k_out = _ops.rope_qk(q, k, freqs)
        ctx.save_for_backward(freqs)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        (freqs,) = ctx.saved_tensors
        if freqs.ndim == 4:
            freqs = freqs[:, 0, 0, :]
        # freqs is already fp32; the kernel and the host-side fp32 conversion are
        # no-ops when the forward and backward use the same angles.
        grad_q, grad_k = _ops.rope_qk_backward(grad_q, grad_k, freqs)
        return grad_q, grad_k, None


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
        # Save only the residual diff = pred - (noise - latents) instead of the
        # three full tensors: halves the activation kept for backward and the
        # backward still produces the exact fp32 gradient. `diff` is kept in the
        # model_pred dtype (bf16) to stay memory-light; the final gradient is
        # accumulated in fp32 as before.
        diff = model_pred - (noise - latents)
        ctx.save_for_backward(diff, weighting, loss_weights)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        diff, weighting, loss_weights = ctx.saved_tensors
        elems_per_sample = diff[0].numel()
        batch = diff.shape[0]
        scale = 2.0 / (elems_per_sample * batch)
        coef = weighting.float().view(-1) * loss_weights.float().view(-1)
        coef = (coef * scale * grad_output.float().view(-1)).view(batch, *([1] * (diff.ndim - 1)))
        grad = (diff.float() * coef).to(diff.dtype)
        return grad, None, None, None, None


def rectified_flow_mse_loss(model_pred, latents, noise, weighting, loss_weights) -> torch.Tensor:
    if not is_enabled() or not model_pred.is_cuda:
        target = noise - latents
        loss = (model_pred.float() - target.float()).pow(2)
        loss = loss.mean(list(range(1, loss.ndim)))
        return (loss * weighting.float().view(-1) * loss_weights.float().view(-1)).mean()
    return _RectifiedFlowMSELossFn.apply(model_pred, latents, noise, weighting, loss_weights)


class _AdaLNNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, eps):
        y = _ops.adaln_norm(x, scale, shift, eps)
        B, T, H, W, D = x.shape
        ctx.bt_total = B * T
        ctx.hw = H * W
        ctx.eps = eps
        ctx.save_for_backward(x, scale)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        dx, dscale, dshift = _ops.adaln_norm_backward(
            grad_output, x, scale, ctx.bt_total, ctx.hw, ctx.eps
        )
        # dshift is computed but shift is a leaf input (constant buffer-ish in
        # training), so we return it for completeness; callers may drop it.
        return dx, dscale, dshift, None


def adaln_norm(x, scale, shift, eps=1e-6):
    """Fused LayerNorm(affine=False) + AdaLN scale/shift modulation.

    x: (B,T,H,W,D); scale/shift: (B*T, D) broadcast over H,W.
    """
    if is_enabled() and x.is_cuda and scale.is_cuda and shift.is_cuda and x.dtype == scale.dtype == shift.dtype:
        return _AdaLNNormFn.apply(x, scale, shift, float(eps))
    B, T, H, W, D = x.shape
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)
    n = (xf - mean) / torch.sqrt(var + eps)
    s = scale.view(B, T, 1, 1, D)
    h = shift.view(B, T, 1, 1, D)
    return (n * (1.0 + s) + h).to(x.dtype)


def fused_adamw_step(grad, param, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, clip_scale=1.0):
    if (
        is_enabled()
        and grad.is_cuda
        and param.is_cuda
        and grad.dtype == torch.float32
        and param.dtype == torch.float32
        and param.dtype == exp_avg.dtype == exp_avg_sq.dtype
    ):
        return _ops.adamw_step(grad, param, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, float(clip_scale))
    # Non-fused fallback mirrors the fused kernel exactly (including clip_scale).
    clipped = grad if clip_scale >= 1.0 else grad.mul_(clip_scale)
    exp_avg.mul_(beta1).add_(clipped, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(clipped, clipped, value=1.0 - beta2)
    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    step_size = lr / bias_correction1
    param.mul_(1.0 - lr * weight_decay).addcdiv_(exp_avg, denom, value=-step_size)
    return None


def global_grad_norm(params) -> float:
    """Fused global L2 grad norm (one pass over nonzero grads; no cat/copy)."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads)
    total = 0.0
    for n in norms:
        total += float(n.item()) * float(n.item())
    return total ** 0.5


class FusedAdamW(torch.optim.AdamW):
    """AdamW whose step() uses the fused CUDA kernel when acceleration is enabled
    (fp32 master weights on CUDA), else falls back to the stock torch AdamW step.

    If ``max_grad_norm`` is set and fusion is active, the global gradient norm is
    computed once inside step() and folded into the CUDA kernel (clip_scale), so
    the separate clip_grad_norm_ pass can be skipped by the trainer.
    """

    def __init__(self, params, *, max_grad_norm=None, **kwargs):
        super().__init__(params, **kwargs)
        self.fused_grad_clip = max_grad_norm is not None and max_grad_norm > 0.0
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    def step(self, closure=None):
        # Eligibility is decided up front (params + group options) so the fused
        # loop below never has to bail out mid-way: a fallback here is performed
        # before any parameter is touched, so no partial double-update occurs.
        fused = is_enabled() and _fused_eligible(self)
        if not fused:
            # Fallback path: honour torch's own gradient clipping if a closure
            # chain expects it — no clipping is applied here (trainer does it).
            return super().step(closure)
        # Fused global-norm clipping: one reduction over grads, folded as
        # clip_scale = min(1, max_norm / grad_norm) into each kernel launch.
        clip_scale = 1.0
        if self.fused_grad_clip:
            gn = global_grad_norm([p for group in self.param_groups for p in group["params"]])
            if gn > 0.0:
                clip_scale = min(1.0, self.max_grad_norm / gn)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                stepn = state["step"]
                fused_adamw_step(
                    p.grad, p, exp_avg, exp_avg_sq,
                    lr, beta1, beta2, eps, wd,
                    1.0 - beta1 ** stepn, 1.0 - beta2 ** stepn,
                    clip_scale=clip_scale,
                )
        return loss


def _fused_eligible(optimizer) -> bool:
    for group in optimizer.param_groups:
        # The fused kernel implements only plain (amsgrad=False, maximize=False)
        # AdamW. amsgrad/maximize would silently change the update vs stock
        # torch.optim.AdamW, so fall back to torch for any group that sets them.
        if group.get("amsgrad", False) or group.get("maximize", False):
            return False
        for p in group["params"]:
            if p.grad is not None:
                if p.dtype != torch.float32 or not p.is_cuda or not p.grad.is_cuda:
                    return False
                if p.is_sparse or p.grad.is_sparse:
                    return False
    return True


def optimizer_handles_clip(optimizer) -> bool:
    """True when ``optimizer.step()`` already applies the global-norm grad clip
    (fused into the CUDA AdamW kernel), so the trainer can skip its own
    ``clip_grad_norm_`` pass. False for every other optimizer/configuration."""
    return (
        isinstance(optimizer, FusedAdamW)
        and getattr(optimizer, "fused_grad_clip", False)
        and is_enabled()
        and _fused_eligible(optimizer)
    )
