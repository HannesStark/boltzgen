"""Real Trainer configuration checks; CUDA availability is simulated, never executed."""
# ruff: noqa: CPY001, INP001

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning import Callback, Trainer

from boltzgen.task.predict import predict


def build_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    devices: int | str | list[int],
    records: int = 100,
) -> Trainer:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    for key, value in {
        "SLURM_NTASKS": "1",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
        "SLURM_JOB_NAME": "prediction-regression",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("TORCHELASTIC_RUN_ID", "LOCAL_RANK", "RANK", "NODE_RANK", "GROUP_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        predict.Boltz,
        "load_from_checkpoint",
        lambda *_a, **_k: SimpleNamespace(eval=lambda: None),
    )
    task = predict.Predict(
        data=SimpleNamespace(predict_set=range(records), num_workers=0),
        writer=Callback(),
        checkpoint="unused",
        output=str(tmp_path),
        name="toy",
        recycling_steps=0,
        sampling_steps=0,
        trainer={
            "accelerator": "cuda",
            "devices": devices,
            "logger": False,
            "enable_checkpointing": False,
        },
    )
    with torch.enable_grad():
        task.run(run_prediction=False)
    return task.lightning_trainer


@pytest.mark.parametrize("devices", [-1, "auto", "0,2", [0, 2]])
def test_device_formats_select_local_workers_in_one_task_slurm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, devices: int | str | list[int]
) -> None:
    trainer = build_trainer(monkeypatch, tmp_path, devices=devices)
    expected_count = 4 if devices in (-1, "auto") else 2
    assert trainer.num_devices == expected_count
    assert trainer.distributed_sampler_kwargs == {
        "num_replicas": expected_count,
        "rank": 0,
    }
    environment = trainer.strategy.cluster_environment
    assert isinstance(environment, LightningEnvironment)
    assert not environment.creates_processes_externally


def test_device_list_truncation_preserves_device_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = build_trainer(monkeypatch, tmp_path, devices=[2, 3], records=1)
    assert trainer.num_devices == 1
    assert trainer.device_ids == [2]


@pytest.mark.parametrize("devices", [-1, "auto", "0,2"])
def test_resolved_device_count_fails_clearly_before_creating_empty_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, devices: int | str
) -> None:
    with pytest.raises(ValueError, match="Set fewer devices or num_nodes"):
        build_trainer(monkeypatch, tmp_path, devices=devices, records=1)
