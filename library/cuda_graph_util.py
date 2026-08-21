"""CUDA-graph capture utilities for Anima (opt-in, experimental).

Provides two capture modes over a fixed-shape callable:

- ``capture_step(callable)`` wraps ``callable(*inputs) -> loss`` (which itself
  runs forward + backward) and lets the caller replay it with new input buffers
  via ``replay(*inputs)``.  Built for the training loop when shapes/activations
  are static for many consecutive steps (e.g. fixed-resolution buckets).

- ``capture_forward(module, static_inputs)`` captures a plain forward pass and
  replays it by copying into the static input buffers (used for sampling/inference
  and for profiling).

Safety: capture is strictly gated — caller must guarantee static shapes, no
blocks_to_swap / offload / dynamic control flow, and out-of-graph optimizer step.
Any capture failure is propagated (callers should fall back to eager).
"""
from __future__ import annotations

import torch


class GraphCapturedCallable:
    def __init__(self, fn, static_inputs: list[torch.Tensor], outputs=None, graph=None, zero_grad_fn=None):
        self._fn = fn
        self._static_inputs = static_inputs
        self._outputs = outputs
        self._graph = graph
        self._zero_grad_fn = zero_grad_fn

    def copy_static_inputs(self, inputs):
        if len(inputs) != len(self._static_inputs):
            raise ValueError(f"static-capture expects {len(self._static_inputs)} inputs, got {len(inputs)}")
        for src, dst in zip(inputs, self._static_inputs):
            if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype:
                raise ValueError(
                    f"static-capture input mismatch: expected {tuple(dst.shape)}/{dst.dtype}, got {tuple(src.shape)}/{src.dtype}"
                )
            # detach() view keeps autograd leaves (requires_grad=True static
            # inputs) usable for in-place refresh before each graph replay.
            dst.detach().copy_(src)

    def replay(self, *inputs):
        self.copy_static_inputs(list(inputs))
        # NOTE: do NOT zero grads here. CUDA graphs capture kernel launches only:
        # the autograd `param.grad = <buffer>` binding happens at CAPTURE time and
        # is not part of the graph. zeroing between replays would detach
        # param.grad from the capture-bound buffer. Grads are zeroed by the
        # trainer's optimizer.zero_grad() between steps.
        self._graph.replay()
        return self._outputs


def _pack_leaf_inputs(args, kwargs):
    """Flatten tensors that participate as graph inputs (leaf, no grad_fn)."""
    leaves = []

    def walk(x):
        if isinstance(x, torch.Tensor):
            leaves.append(x)
        elif isinstance(x, (list, tuple)):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        return None

    for a in args:
        walk(a)
    for v in kwargs.values():
        walk(v)
    return leaves


def capture_forward(fn, *static_inputs, warmup_iters: int = 3, zero_grad_fn=None) -> GraphCapturedCallable:
    """Capture ``fn(*static_inputs) -> tensor(s)`` as a CUDA graph.

    ``static_inputs`` are the actual buffers that will be mutated on each replay
    via ``.copy_``; the returned wrapper copies fresh values then replays.

    If ``fn`` runs ``loss.backward()`` inside (training step), pass
    ``zero_grad_fn=lambda: <zero the model grads>`` so warmup/capture replays do
    not accumulate gradients across runs.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA graph capture requires CUDA")
    # Ensure outputs are not part of autograd / allocate once.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if zero_grad_fn is not None:
            zero_grad_fn()
        for _ in range(warmup_iters):
            out = fn(*static_inputs)
            torch.cuda.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    # Allocate static outputs from the final forward's shape.
    if zero_grad_fn is not None:
        zero_grad_fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*static_inputs)
    torch.cuda.synchronize()
    # NOTE: no zero_grad_fn call here — leaving param.grad bound to the buffer the
    # capture-time backward created is exactly what lets later replays overwrite
    # that same buffer with fresh gradients.
    return GraphCapturedCallable(fn, list(static_inputs), outputs=out, graph=g, zero_grad_fn=zero_grad_fn)


def capture_step(fn, static_inputs: list[torch.Tensor], warmup_iters: int = 3, zero_grad_fn=None) -> GraphCapturedCallable:
    """Capture ``fn(*static_inputs) -> loss`` (forward+backward inside) as a graph.

    The returned wrapper copies fresh input values into the static buffers,
    replays, and returns the loss tensor.  Parameters' ``.grad`` are filled by the
    in-graph backward; the optimizer is stepped OUTSIDE the graph.
    """
    return capture_forward(fn, *static_inputs, warmup_iters=warmup_iters, zero_grad_fn=zero_grad_fn)


class MaybeGraphedStep:
    """Optional CUDA-graph wrapper around a fixed-shape training step.

    ``step_fn(*inputs)`` must run forward + loss + ``loss.backward()`` and return
    the scalar loss (or a list of outputs that includes the loss).

    Behaviour:
    - first call captures at the current tensor-layout key; later calls replay
      (cheap refresh copy).  If the tensor layout changes, it re-captures.
    - any capture/replay failure disables the wrapper permanently and falls back
      to eager, so a grapher bug can never break the run silently.
    """

    def __init__(self, step_fn, enabled: bool = True, logger=None, zero_grad_fn=None):
        self._fn = step_fn
        self._enabled = bool(enabled)
        self._broken = False
        self._cache: dict = {}
        self._logger = logger
        self._zero_grad_fn = zero_grad_fn
        self.re_captures = 0

    def _key(self, inputs):
        out = []
        for t in inputs:
            if isinstance(t, torch.Tensor):
                out.append((tuple(t.shape), t.dtype, t.device.index))
            else:
                out.append(type(t).__name__)
        return tuple(out)

    def __call__(self, *inputs):
        if not self._enabled or self._broken:
            return self._fn(*inputs)
        key = self._key(inputs)
        try:
            cap = self._cache.get(key)
            if cap is None:
                static = []
                for t in inputs:
                    if isinstance(t, torch.Tensor):
                        s = t.detach().clone()
                        if t.requires_grad and t.dtype.is_floating_point:
                            s = s.requires_grad_(True)
                        static.append(s)
                    else:
                        static.append(t)
                torch.cuda.synchronize()
                cap = capture_step(self._fn, static, zero_grad_fn=self._zero_grad_fn)
                self._cache[key] = cap
                self.re_captures += 1
                if self._logger is not None:
                    self._logger.info(f"CUDA-graph step captured (shape key {key}), total captures={self.re_captures}")
            else:
                cap.copy_static_inputs(list(inputs))
            out = cap.replay(*[t for t in inputs])
            # Return copies (not the static buffers) so eager consumers (optimizer,
            # logging) never alias memory that the next replay will overwrite.
            if isinstance(out, torch.Tensor):
                return out.clone()
            if isinstance(out, (tuple, list)):
                return [o.clone() if isinstance(o, torch.Tensor) else o for o in out]
            return out
        except Exception as e:  # capture broke -> permanent eager fallback
            self._broken = True
            self._cache = {}
            if self._logger is not None:
                self._logger.warning(f"CUDA-graph step disabled after failure ({type(e).__name__}: {e})")
            return self._fn(*inputs)
