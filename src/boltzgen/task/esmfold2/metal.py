"""Apple Metal (MPS) compatibility shim for the pinned ESMFold2 implementation.

The pinned `esm` package targets CUDA: it enters ``torch.autocast("cuda")`` and
uses operators whose MPS kernels lack some dtypes. Like a CUDA-API shim that
lets CUDA-shaped code run on Metal, this module leaves the model code untouched
and adapts the few torch entry points it relies on. It is installed only by the
worker, only when scoring on MPS, and is idempotent.
"""

import torch

_INSTALLED = False
_INTEGER_FALLBACK = ("scatter_reduce_", "scatter_reduce", "index_reduce_", "index_reduce")


def _cuda_to_mps(device_type):
    return "mps" if device_type == "cuda" else device_type


def _patch_autocast() -> None:
    """Run the model's hard-coded CUDA bf16 autocast regions on Metal."""

    class MetalAutocast(torch.autocast):
        def __init__(self, device_type="mps", *args, **kwargs):
            super().__init__(_cuda_to_mps(device_type), *args, **kwargs)

    torch.autocast = MetalAutocast
    torch.amp.autocast = MetalAutocast
    torch.amp.autocast_mode.autocast = MetalAutocast


def _wrap_integer_fallback(name: str) -> None:
    """MPS rejects integer dtypes for these reductions: compute them on the CPU."""
    original = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
        if self.device.type == "mps" and not (
            self.is_floating_point() or self.is_complex()
        ):
            args = tuple(
                a.cpu() if isinstance(a, torch.Tensor) else a for a in args
            )
            kwargs = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            if name.endswith("_"):
                cpu = self.cpu()
                getattr(cpu, name)(*args, **kwargs)
                return self.copy_(cpu)
            return getattr(self.cpu(), name)(*args, **kwargs).to(self.device)
        return original(self, *args, **kwargs)

    setattr(torch.Tensor, name, wrapper)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_autocast()
    for name in _INTEGER_FALLBACK:
        _wrap_integer_fallback(name)
    _INSTALLED = True
