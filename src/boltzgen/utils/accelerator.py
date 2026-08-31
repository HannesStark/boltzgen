"""Runtime accelerator compatibility helpers."""

from __future__ import annotations

import os
import warnings

import torch

_NPU_COMPAT_ENABLED = False


def enable_npu_compat() -> bool:
    """Enable torch-npu's CUDA compatibility bridge when an NPU is present."""
    global _NPU_COMPAT_ENABLED

    if _NPU_COMPAT_ENABLED:
        return True
    if os.environ.get("BOLTZGEN_NPU_COMPAT", "1").lower() in {"0", "false", "no"}:
        return False

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return False

    # Lightning's CUDA compatibility path expects a ``(major, minor)`` tuple.
    # torch-npu intentionally returns ``None`` unless this documented
    # compatibility value is configured. It is not used to select NPU kernels.
    os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "8.0")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        import torch_npu.contrib.transfer_to_npu  # noqa: F401

    _NPU_COMPAT_ENABLED = True
    return True


def npu_compat_enabled() -> bool:
    """Return whether the torch-npu compatibility bridge is active."""
    return _NPU_COMPAT_ENABLED


def available_device_count() -> int:
    """Return the number of devices exposed by the active accelerator."""
    if _NPU_COMPAT_ENABLED:
        return torch.npu.device_count()
    return torch.cuda.device_count()
