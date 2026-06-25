"""CUDA preflight checks for BoltzGen CLI entry points."""

from __future__ import annotations

from typing import Any

import torch


def _as_device_count(devices: Any) -> int:
    """Return the number of requested CUDA devices for common Hydra values."""
    if devices in (None, "auto"):
        return torch.cuda.device_count()
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    try:
        return int(devices)
    except (TypeError, ValueError):
        return torch.cuda.device_count()


def validate_cuda_runtime(devices: Any = None) -> tuple[int, tuple[int, int]]:
    """Fail early with actionable CUDA diagnostics before Lightning/DDP starts.

    PyTorch can report visible CUDA devices even when the installed CUDA runtime is
    newer than the host NVIDIA driver. In that state Lightning reaches NCCL/DDP
    initialization and emits a long distributed traceback. Touching each requested
    device here surfaces the real incompatibility before pipeline configuration or
    subprocess execution begins.
    """
    requested = _as_device_count(devices)
    if requested <= 0:
        raise RuntimeError(
            "No CUDA devices are visible to PyTorch. BoltzGen prediction steps "
            "require at least one CUDA device. Check the host GPU, NVIDIA "
            "driver, and CUDA_VISIBLE_DEVICES."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA devices were requested, but PyTorch reports that CUDA is not "
            "available. Run this pipeline on a CUDA-capable host or install a "
            "PyTorch build that matches the system NVIDIA driver."
        )

    visible = torch.cuda.device_count()
    if requested > visible:
        raise RuntimeError(
            f"Requested {requested} CUDA device(s), but only {visible} are visible. "
            "Reduce --devices or adjust CUDA_VISIBLE_DEVICES."
        )

    try:
        capability = torch.cuda.get_device_capability(0)
        for index in range(requested):
            probe = torch.empty(1, device=f"cuda:{index}")
            # Force the allocation to be realized so driver/runtime mismatches are
            # reported here instead of later inside NCCL initialization.
            probe.fill_(0)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - preserve the low-level CUDA reason.
        raise RuntimeError(
            "CUDA runtime validation failed before starting the BoltzGen pipeline. "
            "The most common cause is an NVIDIA driver that is too old for the "
            "installed PyTorch CUDA runtime. Update the NVIDIA driver, or install "
            "a PyTorch wheel built for a CUDA version supported by this driver. "
            "If single-GPU CUDA validation works but multi-GPU execution fails, "
            "retry with --devices 1 to avoid NCCL/DDP. Original CUDA error: "
            f"{exc}"
        ) from exc

    return requested, capability
