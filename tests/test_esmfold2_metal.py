import pytest
import torch

from boltzgen.task.esmfold2 import metal

mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)


@pytest.fixture
def patched(monkeypatch):
    """Apply the shim, restoring every torch attribute it replaces."""
    for owner, name in [
        (torch, "autocast"),
        (torch.amp, "autocast"),
        (torch.amp.autocast_mode, "autocast"),
        *[(torch.Tensor, n) for n in metal._INTEGER_FALLBACK],
    ]:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(metal, "_INSTALLED", False)
    metal.install()


def test_install_is_idempotent(patched):
    wrapped = torch.Tensor.scatter_reduce_
    metal.install()
    assert torch.Tensor.scatter_reduce_ is wrapped


def test_cuda_autocast_maps_to_mps(patched):
    assert metal._cuda_to_mps("cuda") == "mps"
    assert metal._cuda_to_mps("cpu") == "cpu"
    with torch.autocast(device_type="cuda", enabled=False) as ctx:
        assert ctx.device == "mps"
    with torch.amp.autocast("cpu", enabled=False) as ctx:
        assert ctx.device == "cpu"


def test_cpu_integer_reductions_untouched(patched):
    out = torch.full((3,), 9, dtype=torch.long)
    out.scatter_reduce_(0, torch.tensor([0, 0, 2]), torch.tensor([5, 1, 7]), "amin")
    assert out.tolist() == [1, 9, 7]


@mps
def test_integer_scatter_reduce_falls_back_on_mps(patched):
    out = torch.full((3,), 9, dtype=torch.long, device="mps")
    index = torch.tensor([0, 0, 2], device="mps")
    src = torch.tensor([5, 1, 7], device="mps")
    out.scatter_reduce_(0, index, src, reduce="amin", include_self=True)
    assert out.device.type == "mps" and out.tolist() == [1, 9, 7]
    result = torch.full((3,), 9, dtype=torch.long, device="mps").scatter_reduce(
        0, index, src, "amax"
    )
    assert result.device.type == "mps" and result.tolist() == [9, 9, 9]
