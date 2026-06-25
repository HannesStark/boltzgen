import sys
import types

import pytest


class FakeCuda:
    def __init__(self, *, available=True, count=2):
        self.available = available
        self.count = count
        self.synchronized = False

    def device_count(self):
        return self.count

    def is_available(self):
        return self.available

    def get_device_capability(self, index=0):
        return (7, 5)

    def synchronize(self):
        self.synchronized = True


class FakeProbe:
    def fill_(self, value):
        return self


@pytest.fixture
def cuda_validation(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        empty=lambda *args, **kwargs: FakeProbe(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("boltzgen.cli.cuda_validation", None)

    from boltzgen.cli import cuda_validation as module

    return module


def test_validate_cuda_runtime_fails_before_nccl_when_cuda_allocation_fails(
    cuda_validation, monkeypatch
):
    fake_cuda = FakeCuda()

    def fake_empty(*args, **kwargs):
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    monkeypatch.setattr(cuda_validation.torch, "cuda", fake_cuda)
    monkeypatch.setattr(cuda_validation.torch, "empty", fake_empty)

    with pytest.raises(RuntimeError, match="CUDA runtime validation failed") as exc_info:
        cuda_validation.validate_cuda_runtime(2)

    message = str(exc_info.value)
    assert "NVIDIA driver that is too old" in message
    assert "--devices 1" in message
    assert "CUDA driver version is insufficient" in message


def test_validate_cuda_runtime_rejects_request_above_visible_devices(
    cuda_validation, monkeypatch
):
    fake_cuda = FakeCuda(count=1)
    monkeypatch.setattr(cuda_validation.torch, "cuda", fake_cuda)
    monkeypatch.setattr(cuda_validation.torch, "empty", lambda *args, **kwargs: FakeProbe())

    with pytest.raises(RuntimeError, match="Requested 2 CUDA device"):
        cuda_validation.validate_cuda_runtime(2)


def test_validate_cuda_runtime_returns_device_count_and_capability(
    cuda_validation, monkeypatch
):
    fake_cuda = FakeCuda(count=2)
    monkeypatch.setattr(cuda_validation.torch, "cuda", fake_cuda)
    monkeypatch.setattr(cuda_validation.torch, "empty", lambda *args, **kwargs: FakeProbe())

    devices, capability = cuda_validation.validate_cuda_runtime("auto")

    assert devices == 2
    assert capability == (7, 5)
    assert fake_cuda.synchronized
