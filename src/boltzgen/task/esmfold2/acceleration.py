# Copyright 2026 Anthropic, PBC; Copyright 2026 Biohub. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference acceleration adapted from Anthropic's public ESMFold2 kit.

This adapter targets the pinned native ESM implementation, not the older
Transformers fork used by the kit. See NOTICE.md for sources and licenses.
It does not change precision, kernels, sampling settings, or ESMC inputs.
"""

import ast
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
import logging
import sys
import textwrap
from types import FunctionType, MethodType

import torch
from torch import Tensor
import torch.nn.functional as F

from boltzgen.task.esmfold2.sampling import sample_without_scalar_sync
from boltzgen.task.esmfold2.contract import ACCELERATION_REVISION, validate_acceleration

logger = logging.getLogger(__name__)

_SOURCE_HASHES = {
    "sample": "0046a1ab71e1c3418cdb9137c4dd5552fc595c2f762541648f436936f1363a98",
    "forward": "54fada346abf221ec26482aa52f27a7ab801766cbee9d92740a383049026ab56",
}


@contextmanager
def acceleration_context(model, options: dict):
    """Scope all optimized state to one request, with native fallback."""
    mode = options.get("acceleration", "off")
    validate_acceleration(mode)
    if (mode == "fused") != getattr(model, "_boltzgen_fused_backend", False):
        raise ValueError(
            "Fused and native requests need separate ESMFold2 model instances"
        )
    execution = {
        "requested": mode,
        "revision": ACCELERATION_REVISION,
        "effective": "off",
        "kernel_backend": "fused" if mode == "fused" else "native",
    }
    if mode == "off":
        yield execution
        return
    if options.get("acceleration_revision") != ACCELERATION_REVISION:
        raise ValueError(
            "ESMFold2 acceleration request revision does not match this worker"
        )
    # On the native backend, graph setup outweighs replay savings for larger
    # crops. Fused kernels remain launch-bound over a wider measured range.
    controller = AcceleratedInference(model, max_tokens=512 if mode == "fused" else 256)
    try:
        try:
            controller.configure()
        except (ValueError, OSError) as exc:
            controller.close()
            execution["fallback_reason"] = str(exc)
            logger.warning(
                "ESMFold2 adapter unavailable; using the unwrapped %s backend: %s",
                execution["kernel_backend"],
                exc,
            )
        else:
            execution["effective"] = "cached_with_graphs"
        yield execution
        execution["graph_stats"] = dict(controller.stats)
        if controller.ready and not any(
            key.endswith("replays") for key in controller.stats
        ):
            execution["effective"] = "cached"
    finally:
        controller.close()


def check_source(
    function: Callable, expected: str, *, inference_mode: bool = False
) -> None:
    """Refuse a port against different upstream code, including local patches."""
    if type(function) is not FunctionType:
        raise ValueError(
            "Cannot inspect ESMFold2 source; leave its callable unchanged"
        )
    # getsource() follows __wrapped__ automatically. Validate the executed
    # wrapper first so functools.wraps cannot hide a modified implementation.
    if inference_mode:
        wrapped = getattr(function, "__wrapped__", None)
        native_code = torch.inference_mode()(lambda: None).__code__
        if function.__code__ is not native_code or type(wrapped) is not FunctionType:
            raise ValueError("Unsupported ESMFold2 inference wrapper")
        closure = inspect.getclosurevars(function).nonlocals
        factory = closure.get("ctx_factory")
        context = getattr(factory, "__self__", None)
        if (
            type(context) is not torch.inference_mode
            or context.mode is not True
            or type(factory) is not MethodType
            or factory.__func__ is not torch.inference_mode.clone
            or closure.get("func") is not wrapped
        ):
            raise ValueError("Unsupported ESMFold2 inference wrapper")
        function = wrapped
    if hasattr(function, "__wrapped__"):
        raise ValueError("Unsupported extra wrapper around ESMFold2 source")
    try:
        source = inspect.getsource(function)
    except TypeError as exc:
        # Instrumentation may wrap a valid native method in a callable object.
        # Its execution can still work even though Python cannot inspect it.
        raise ValueError(
            "Cannot inspect ESMFold2 source; leave its callable unchanged"
        ) from exc
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        # A valid lambda can occupy only part of a larger source expression.
        raise ValueError(
            "Cannot parse ESMFold2 source; leave its callable unchanged"
        ) from exc
    # Python 3.13+ omits empty fields by default; retain the pinned 3.12 digest.
    dump_options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    digest = hashlib.sha256(
        ast.dump(tree.body[0], include_attributes=False, **dump_options)
        .replace(", type_params=[]", "")
        .encode()
    ).hexdigest()
    if digest != expected:
        raise ValueError(f"Unsupported ESMFold2 source for {function.__qualname__}")


def build_mask(
    params: tuple, batch: int, atoms: int, window: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build the native SDPA mask, including holes and invalid-atom diagonals."""
    if len(params) > 2:
        valid = torch.zeros(batch * atoms, dtype=torch.bool, device=device)
        valid[params[2]] = True
        valid = valid.view(batch, atoms)
    else:
        valid = torch.ones(batch, atoms, dtype=torch.bool, device=device)
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(atoms, dtype=torch.bool, device=device)
    return allowed.unsqueeze(1), valid


@dataclass
class MaskCache:
    """Request-owned masks; retaining indices prevents allocator pointer reuse."""

    entries: dict = field(default_factory=dict)
    bytes_used: int = 0
    max_bytes: int = 256 * 1024**2

    def clear(self) -> None:
        self.entries.clear()
        self.bytes_used = 0

    def get(self, params: tuple, x: Tensor, window: int) -> tuple[Tensor, Tensor]:
        batch, atoms = x.shape[:2]
        indices = params[2] if len(params) > 2 else None
        key = (batch, atoms, window, x.device, id(indices))
        if key not in self.entries:
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Atom layout was not warmed before graph capture")
            allowed, valid = build_mask(params, batch, atoms, window, x.device)
            size = allowed.numel() + valid.numel()
            if self.bytes_used + size > self.max_bytes:
                return allowed, valid
            self.entries[key] = (indices, allowed, valid)
            self.bytes_used += size
        _, allowed, valid = self.entries[key]
        return allowed, valid


def cached_attention(
    module, x: Tensor, attention_params: tuple, cache: MaskCache
) -> Tensor:
    """Native SWA attention with its constant SDPA mask hoisted (kit U1)."""
    from esm.models.esmfold2.layers import apply_rotary_emb_3d, qk_norm

    batch, atoms = x.shape[:2]
    cos, sin = attention_params[:2]
    qkv = module.Wqkv(x).view(batch, atoms, 3, module.n_heads, module.head_dim)
    q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)
    q, k = qk_norm(q), qk_norm(k)
    q, k = apply_rotary_emb_3d(q, cos, sin), apply_rotary_emb_3d(k, cos, sin)
    input_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    allowed, valid = cache.get(attention_params, q, module.half_window)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed,
        scale=module.scale,
    ).transpose(1, 2)
    out = out * valid.unsqueeze(-1).unsqueeze(-1)
    out = out.to(input_dtype).reshape(batch, atoms, -1)
    return module.out_proj(out * torch.sigmoid(module.gate_proj(x)))


def _signature(value):
    if isinstance(value, Tensor):
        return (value.shape, value.dtype, value.device, value.stride())
    return value


def _clone_output(value):
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_output(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"Unsupported graph output: {type(value)}")


class _Graph:
    """A deterministic forward with private buffers and a private memory pool."""

    def __init__(
        self,
        forward: Callable,
        kwargs: dict,
        dynamic: tuple[str, ...],
        stream: torch.cuda.Stream,
    ):
        self.kwargs = dict(kwargs)
        self.dynamic = dynamic
        for name in dynamic:
            value = kwargs[name]
            self.kwargs[name] = value.clone() if isinstance(value, Tensor) else value
        device = next(
            value.device for value in kwargs.values() if isinstance(value, Tensor)
        )
        # Explicit streams/device also work for workers assigned cuda:1, etc.
        with torch.cuda.device(device):
            rng = torch.cuda.get_rng_state(device)
            cpu_rng = torch.get_rng_state()
            stream.wait_stream(torch.cuda.current_stream(device))
            try:
                with torch.cuda.stream(stream):
                    forward(**self.kwargs)
                    forward(**self.kwargs)
                torch.cuda.current_stream(device).wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                # The outer context restores the stream even if capture_end fails.
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(self.graph, stream=stream):
                        self.output = forward(**self.kwargs)
                torch.cuda.synchronize(device)
                if not torch.equal(
                    rng, torch.cuda.get_rng_state(device)
                ) or not torch.equal(cpu_rng, torch.get_rng_state()):
                    raise RuntimeError("Graph forward consumed random numbers")
            finally:
                torch.cuda.set_rng_state(rng, device)
                torch.set_rng_state(cpu_rng)

    def matches(self, kwargs: dict) -> bool:
        if kwargs.keys() != self.kwargs.keys():
            return False
        for name, old in self.kwargs.items():
            new = kwargs[name]
            if name in self.dynamic:
                if _signature(old) != _signature(new):
                    return False
            elif isinstance(old, (Tensor, dict)):
                if old is not new:
                    return False
            elif old != new:
                return False
        return True

    def replay(self, kwargs: dict):
        for name in self.dynamic:
            value = self.kwargs[name]
            if isinstance(value, Tensor):
                value.copy_(kwargs[name])
        self.graph.replay()
        # Callers may keep results alive across the next replay (notably Kabsch).
        return _clone_output(self.output)


class AcceleratedInference:
    """Own the kit's exact caches/graphs for one sequential inference worker.

    Reset before every independent input, even when shapes match. The first
    diffusion step stays eager to populate native conditioning/atom caches.
    Batched confidence and stochastic LM-dropout forwards are never captured.
    """

    def __init__(self, model, *, graphs: bool = True, max_tokens: int = 512):
        self.model = model
        self.graphs_enabled = graphs
        self.max_tokens = max_tokens
        self.masks = MaskCache()
        self.graphs: dict[str, _Graph] = {}
        self.disabled: set[str] = set()
        self.originals: list[tuple[object, str, bool, object]] = []
        self.stats = Counter()
        self.ready = False
        self.capture_stream = getattr(model, "_boltzgen_capture_stream", None)

    def _patch(self, module, name: str, function: Callable) -> None:
        attributes = vars(module)
        self.originals.append((module, name, name in attributes, attributes.get(name)))
        setattr(module, name, function)

    def configure(self) -> None:
        """Validate copied source before installing any instance-local wrappers."""
        if self.ready:
            return
        from esm.models.esmfold2 import layers

        check_source(
            layers.DiffusionStructureHead.sample,
            _SOURCE_HASHES["sample"],
            inference_mode=True,
        )
        check_source(layers.SWA3DRoPEAttention.forward, _SOURCE_HASHES["forward"])
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("ESMFold2 acceleration requires eval() and frozen weights")
        sample = self.model.structure_head.sample
        if (
            type(sample) is not MethodType
            or sample.__func__ is not layers.DiffusionStructureHead.sample
            or getattr(sample, "__self__", None) is not self.model.structure_head
        ):
            raise ValueError("Unsupported ESMFold2 sample method override or wrapper")
        attention_modules = [
            module
            for module in self.model.modules()
            if not layers.FLASH_ATTN_AVAILABLE
            and isinstance(module, layers.SWA3DRoPEAttention)
        ]
        for module in attention_modules:
            forward = module.forward
            if (
                type(forward) is not MethodType
                or forward.__func__ is not layers.SWA3DRoPEAttention.forward
                or getattr(forward, "__self__", None) is not module
            ):
                raise ValueError("Unsupported ESMFold2 attention method override or wrapper")

        def sample_wrapper(*args, **kwargs):
            # Recycling has finished. Its graph pool is no longer needed, and
            # releasing it makes room for the larger diffusion activations.
            self.graphs.clear()
            self.masks.clear()
            try:
                return sample_without_scalar_sync(
                    self.model.structure_head, *args, **kwargs
                )
            finally:
                # Every diffusion output is cloned; confidence needs no graph
                # storage. Free that storage before the five-sample head runs.
                self.graphs.clear()
                self.masks.clear()

        self._patch(self.model.structure_head, "sample", sample_wrapper)
        for module in attention_modules:

            def forward(x, attention_params, module=module):
                return cached_attention(module, x, attention_params, self.masks)

            self._patch(module, "forward", forward)
        trunk = self.model.folding_trunk
        trunk_forward = trunk.forward

        def trunk_wrapper(pair, pair_attention_mask=None):
            return self._run(
                "trunk",
                trunk_forward,
                dict(pair=pair, pair_attention_mask=pair_attention_mask),
                ("pair", "pair_attention_mask"),
                pair.shape[1],
            )

        self._patch(trunk, "forward", trunk_wrapper)
        diffusion = self.model.structure_head.diffusion_module
        diffusion_forward = diffusion.forward

        def diffusion_wrapper(**kwargs):
            cache = kwargs.get("inference_cache")
            if not cache or kwargs.get("return_atom_repr"):
                return diffusion_forward(**kwargs)
            return self._run(
                "diffusion",
                diffusion_forward,
                kwargs,
                ("x_noisy", "t_hat"),
                kwargs["z_trunk"].shape[1],
            )

        self._patch(diffusion, "forward", diffusion_wrapper)
        self.ready = True

    def _run(
        self,
        name: str,
        forward: Callable,
        kwargs: dict,
        dynamic: tuple[str, ...],
        tokens: int,
    ):
        first = kwargs[dynamic[0]]
        if (
            not self.graphs_enabled
            or name in self.disabled
            or torch.is_grad_enabled()
            or not first.is_cuda
            or tokens > self.max_tokens
        ):
            return forward(**kwargs)
        graph = self.graphs.get(name)
        if graph is not None and not graph.matches(kwargs):
            # Shape/context changes must never reuse stale constant buffers.
            del self.graphs[name]
            graph = None
        if graph is None:
            try:
                if (
                    self.capture_stream is None
                    or self.capture_stream.device != first.device
                ):
                    # PyTorch retains a cuBLAS workspace for each stream. Reuse
                    # one stream across requests instead of accumulating those
                    # workspaces throughout a long design campaign.
                    self.capture_stream = torch.cuda.Stream(device=first.device)
                    if self.model is not None:
                        self.model._boltzgen_capture_stream = self.capture_stream
                graph = _Graph(forward, kwargs, dynamic, self.capture_stream)
            except RuntimeError as exc:
                if not any(
                    word in str(exc).lower()
                    for word in ("capture", "graph", "out of memory")
                ):
                    raise
                self.disabled.add(name)
                self.stats[f"{name}_fallbacks"] += 1
                logger.warning(
                    "ESMFold2 %s graph unavailable; using eager execution: %s",
                    name,
                    exc,
                )
            if graph is None:
                # Release a failed capture's traceback/buffers before eager retry.
                torch.cuda.empty_cache()
                return forward(**kwargs)
            self.graphs[name] = graph
            self.stats[f"{name}_captures"] += 1
        self.stats[f"{name}_replays"] += 1
        return graph.replay(kwargs)

    def reset(self) -> None:
        """Release graph owners before masks and inputs they reference."""
        self.graphs.clear()
        self.masks.clear()
        self.disabled.clear()
        self.stats.clear()

    def close(self) -> None:
        """Restore native execution, including after a failed request."""
        self.reset()
        for module, name, had_override, original in reversed(self.originals):
            if had_override:
                setattr(module, name, original)
            else:
                delattr(module, name)
        self.originals.clear()
        self.ready = False
