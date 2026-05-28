"""XPU utilities.

By importing this module, the accelerator and strategy are registered in lightning.
"""

from .single_xpu_strategy import SingleXPUStrategy
from .xpu_accelerator import XPUAccelerator
from .xpu_precision import XPUMixedPrecision

__all__ = ["SingleXPUStrategy", "XPUAccelerator", "XPUMixedPrecision"]
