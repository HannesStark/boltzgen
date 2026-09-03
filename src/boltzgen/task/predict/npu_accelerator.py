"""PyTorch Lightning accelerator adapter for torch-npu (Ascend NPU)."""

import torch
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from typing_extensions import override


class NPUAccelerator(Accelerator):
    """Expose torch.npu as a regular Lightning accelerator."""

    @override
    def setup_device(self, device: torch.device) -> None:
        if getattr(device, "type", None) != "npu":
            raise ValueError(f"Selected device must be an NPU, got {device}")
        torch.npu.set_device(device)

    @override
    def get_device_stats(self, device: _DEVICE) -> dict:
        try:
            return dict(torch.npu.memory_stats(device))
        except Exception:
            return {}

    @override
    def teardown(self) -> None:
        if torch.npu.is_available():
            torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices):
        if isinstance(devices, (list, tuple)):
            return [int(device) for device in devices]
        if isinstance(devices, str):
            if devices == "auto":
                return list(range(NPUAccelerator.auto_device_count()))
            if devices.startswith("npu"):
                return [int(devices.split(":")[1])]
            return [int(devices)]
        if isinstance(devices, int):
            return list(range(devices))
        raise ValueError(f"Unsupported devices input for NPUAccelerator: {devices!r}")

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("npu", device) for device in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        return torch.npu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.npu.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "npu"
