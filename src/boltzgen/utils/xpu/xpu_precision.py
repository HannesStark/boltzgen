"""XPU Precision Plugin for PyTorch Lightning."""

from contextlib import contextmanager
from typing import Any, Generator, Literal

import torch
from lightning_fabric.plugins.precision import MixedPrecision
from torch import Tensor


class XPUMixedPrecision(MixedPrecision):
    """Mixed precision plugin for XPU devices.

    This overrides the default MixedPrecision plugin to use 'xpu' as the
    device type for torch.autocast instead of 'cuda'.
    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"] = "bf16-mixed",
    ) -> None:
        """Initialize XPU mixed precision.

        Parameters
        ----------
        precision : Literal["16-mixed", "bf16-mixed"]
            The precision mode. "16-mixed" uses float16, "bf16-mixed" uses bfloat16.

        """
        # Determine dtype from precision string
        if precision == "16-mixed":
            dtype = torch.float16
        elif precision == "bf16-mixed":
            dtype = torch.bfloat16
        else:
            msg = f"Invalid precision: {precision}. Must be '16-mixed' or 'bf16-mixed'"
            raise ValueError(msg)

        # Initialize with xpu device type
        super().__init__(precision=precision, device="xpu")
        self._desired_input_dtype = dtype

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Context manager for forward pass with XPU autocast."""
        with torch.autocast(device_type="xpu", dtype=self._desired_input_dtype):
            yield

    def convert_input(self, data: Any) -> Any:
        """Convert input data to the appropriate precision."""
        return self._convert_fp_tensor(data)

    def _convert_fp_tensor(self, data: Any) -> Any:
        """Convert floating point tensors to the desired dtype."""
        if isinstance(data, Tensor) and data.is_floating_point():
            return data.to(self._desired_input_dtype)
        return data

