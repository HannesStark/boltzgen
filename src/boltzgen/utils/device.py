"""Device utilities for PyTorch device selection (CUDA, MPS, CPU)."""
import torch
from typing import Tuple, Optional


def get_device_type() -> str:
    """
    Get the best available device type.

    Returns
    -------
    str
        Device type string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device() -> torch.device:
    """
    Get the best available PyTorch device.

    Returns
    -------
    torch.device
        PyTorch device object
    """
    return torch.device(get_device_type())


def get_device_count() -> int:
    """
    Get the number of available devices.

    Returns
    -------
    int
        Number of devices (1 for MPS/CPU, cuda.device_count() for CUDA)
    """
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.device_count()
    else:
        # MPS and CPU only support single device
        return 1


def get_device_capability() -> Tuple[int, int]:
    """
    Get device capability (compute capability for CUDA, version for MPS/CPU).

    Returns
    -------
    Tuple[int, int]
        Device capability tuple. For CUDA, returns compute capability.
        For MPS/CPU, returns (8, 0) to indicate modern device support.
    """
    device_type = get_device_type()
    if device_type == "cuda":
        return torch.cuda.get_device_capability()
    else:
        # MPS and modern CPUs support most features, return (8, 0) as default
        return (8, 0)


def empty_cache():
    """
    Empty device cache if supported.
    Works for CUDA and MPS devices.
    """
    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
    # CPU doesn't need cache clearing


def get_autocast_device_type() -> Optional[str]:
    """
    Get the device type for autocast context manager.

    Returns
    -------
    Optional[str]
        "cuda", "mps", or "cpu" for autocast. Returns None if autocast not supported.
    """
    device_type = get_device_type()
    if device_type in ["cuda", "cpu"]:
        return device_type
    elif device_type == "mps":
        # MPS support for autocast was added in PyTorch 2.1
        # Check if available
        try:
            with torch.autocast(device_type="mps"):
                pass
            return "mps"
        except (RuntimeError, TypeError):
            # Fallback to CPU autocast if MPS autocast not supported
            return "cpu"
    return "cpu"
