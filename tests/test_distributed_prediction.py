"""Run the prediction launcher and sampler with real CPU worker processes."""
# ruff: noqa: CPY001, INP001, S603

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.plugins.precision import Precision
from torch.utils.data import DataLoader


class IntegerData(LightningDataModule):
    """Supply integer record identities without model inputs or weights."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.predict_set = list(range(count))
        self.num_workers = 0

    def predict_dataloader(self) -> DataLoader:
        """Return the loader that Lightning shards across workers."""
        return DataLoader(self.predict_set, batch_size=1, num_workers=0)


class IdentityModel(LightningModule):
    """Return record IDs through the real Lightning prediction loop."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Keep identities unchanged so tests can check exact coverage."""
        return batch


class RecordWriter(Callback):
    """Write multiple outputs per record, delaying one worker deliberately."""

    def __init__(self, output: Path, multiplicity: int) -> None:
        self.output = output
        self.multiplicity = multiplicity

    def on_predict_batch_end(  # noqa: PLR0917
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: torch.Tensor,
        batch: torch.Tensor,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Use synchronous writes, as the production prediction writers do."""
        if trainer.global_rank == 1:
            time.sleep(0.1)
        for item in outputs.tolist():
            for sample in range(self.multiplicity):
                (self.output / f"{item}_{sample}.record").write_text(
                    str(trainer.global_rank)
                )


def run_worker() -> None:
    """Entry point re-executed by both Lightning and torchrun workers."""
    from boltzgen.task.predict import predict  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--external-slurm", action="store_true")
    args = parser.parse_args()
    if args.external_slurm:
        # torchrun supplies a real process group rendezvous. Expose those ranks
        # through SLURM's interface to test externally launched worker selection.
        os.environ.update(
            SLURM_NTASKS=os.environ["WORLD_SIZE"],
            SLURM_NTASKS_PER_NODE=os.environ["LOCAL_WORLD_SIZE"],
            SLURM_PROCID=os.environ["RANK"],
            SLURM_LOCALID=os.environ["LOCAL_RANK"],
            SLURM_NODEID="0",
        )
        for key in (
            "RANK",
            "GROUP_RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
        ):
            os.environ.pop(key, None)

    predict.Boltz.load_from_checkpoint = lambda *_args, **_kwargs: IdentityModel()
    task = predict.Predict(
        data=IntegerData(7),
        writer=RecordWriter(args.output, 3),
        checkpoint="unused",
        output=args.output,
        name="integer_records",
        recycling_steps=0,
        sampling_steps=0,
        slurm=args.external_slurm,
        trainer={
            "accelerator": "cpu",
            "devices": 2,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )
    task.run()
    trainer = task.lightning_trainer
    receipt = {
        "world_size": trainer.world_size,
        "environment": type(trainer.strategy.cluster_environment).__name__,
        "sampler_replicas": trainer.distributed_sampler_kwargs["num_replicas"],
        "records_seen_at_return": len(list(args.output.glob("*.record"))),
    }
    (args.output / f"rank_{trainer.global_rank}.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize("launcher", ["local", "allocation", "torchrun", "slurm"])
def test_every_rank_finishes_writes_before_predict_returns(
    tmp_path: Path, launcher: str
) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "TORCHELASTIC_"))
        and key
        not in {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NODE_RANK",
        }
    }
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if launcher != "local":
        env.update(
            SLURM_NTASKS="1",
            SLURM_PROCID="0",
            SLURM_LOCALID="0",
            SLURM_NODEID="0",
            SLURM_JOB_NAME="prediction-regression",
        )
    command = [sys.executable]
    if launcher in {"torchrun", "slurm"}:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
        ]
    command += [str(Path(__file__).resolve()), str(tmp_path)]
    if launcher == "slurm":
        command.append("--external-slurm")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {f"{item}_{sample}.record" for item in range(7) for sample in range(3)}
    assert {item.name for item in tmp_path.glob("*.record")} == expected
    assert {item.read_text() for item in tmp_path.glob("*.record")} == {"0", "1"}
    environments = {
        "local": "LightningEnvironment",
        "allocation": "LightningEnvironment",
        "torchrun": "TorchElasticEnvironment",
        "slurm": "SLURMEnvironment",
    }
    for rank in range(2):
        receipt: dict[str, Any] = json.loads(
            (tmp_path / f"rank_{rank}.json").read_text()
        )
        assert receipt == {
            "world_size": 2,
            "environment": environments[launcher],
            "sampler_replicas": 2,
            "records_seen_at_return": len(expected),
        }


@pytest.mark.parametrize(
    "form", ["single", "list", "tuple", "set", "generator", "dict_values", "precision"]
)
def test_explicit_environment_plugins_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form: str
) -> None:
    from boltzgen.task.predict import predict  # noqa: PLC0415

    for key, value in {
        "SLURM_NTASKS": "1",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
        "SLURM_JOB_NAME": "regression",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    explicit = LightningEnvironment()
    plugins = {
        "single": explicit,
        "list": [explicit],
        "tuple": (explicit,),
        "set": {explicit},
        "generator": (item for item in [explicit]),
        "dict_values": {"environment": explicit}.values(),
        "precision": [Precision()],
    }[form]
    monkeypatch.setattr(
        predict.Boltz, "load_from_checkpoint", lambda *_args, **_kwargs: IdentityModel()
    )
    task = predict.Predict(
        data=IntegerData(7),
        writer=RecordWriter(tmp_path, 3),
        checkpoint="unused",
        output=str(tmp_path),
        name="integer_records",
        recycling_steps=0,
        sampling_steps=0,
        trainer={
            "accelerator": "cpu",
            "devices": 2,
            "plugins": plugins,
            "logger": False,
            "enable_checkpointing": False,
        },
    )
    # Predict sets global grad mode; contain that side effect within this test.
    with torch.enable_grad():
        task.run(run_prediction=False)
    actual = task.lightning_trainer.strategy.cluster_environment
    if form == "precision":
        assert isinstance(actual, LightningEnvironment)
    else:
        assert actual is explicit


if __name__ == "__main__":
    run_worker()
