"""CPU-only configuration regression tests; no model or input data are loaded."""
# ruff: noqa: CPY001, INP001, PLR2004

from argparse import Namespace
from pathlib import Path

import pytest

from boltzgen.cli import boltzgen as cli


def pipeline_args(tmp_path: Path, *extra: str) -> Namespace:
    checkpoint = tmp_path / "unused-checkpoint.pt"
    checkpoint.touch()
    return cli.build_parser().parse_args(
        [
            "configure",
            "unused-input.yaml",
            "--output",
            str(tmp_path / "output"),
            "--protocol",
            "peptide-anything",
            "--skip_inverse_folding",
            "--design_checkpoints",
            str(checkpoint),
            "--folding_checkpoint",
            str(checkpoint),
            "--config",
            "design",
            "trainer.accelerator=cpu",
            "trainer.precision=32",
            *extra,
        ]
    )


@pytest.mark.parametrize("mode", ["auto", "false"])
@pytest.mark.parametrize("failure", [AssertionError, RuntimeError])
def test_cpu_configuration_does_not_initialize_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    failure: type[Exception],
) -> None:
    args = pipeline_args(tmp_path, "--use_kernels", mode)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 0)

    def unavailable_capability() -> tuple[int, int]:
        raise failure("CUDA is unavailable")

    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", unavailable_capability)
    pipeline = cli.BinderDesignPipeline(args, tmp_path)
    design = next(step for step in pipeline.steps if step.name == "design").get_config()
    assert design.trainer.accelerator == "cpu"
    assert design.trainer.devices == 1
    assert design.override.use_kernels is False


def test_default_cpu_device_count_is_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A neutral capability result isolates the second, independent failure."""
    args = pipeline_args(tmp_path)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (0, 0))
    pipeline = cli.BinderDesignPipeline(args, tmp_path)
    design = next(step for step in pipeline.steps if step.name == "design").get_config()
    assert design.trainer.devices == 1


@pytest.mark.parametrize(
    ("capability", "expected_kernels"), [((7, 5), False), ((8, 0), True)]
)
def test_cuda_configuration_preserves_auto_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_kernels: bool,
) -> None:
    args = pipeline_args(tmp_path)
    args.config = None
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: capability)
    pipeline = cli.BinderDesignPipeline(args, tmp_path)
    design = next(step for step in pipeline.steps if step.name == "design").get_config()
    assert design.trainer.accelerator == "gpu"
    assert design.trainer.devices == 2
    assert design.override.use_kernels is expected_kernels


def test_cpu_explicit_device_count_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = pipeline_args(tmp_path, "--devices", "3")
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (0, 0))
    pipeline = cli.BinderDesignPipeline(args, tmp_path)
    design = next(step for step in pipeline.steps if step.name == "design").get_config()
    assert design.trainer.devices == 3


def test_cpu_step_device_override_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = pipeline_args(tmp_path)
    args.config[0].append("trainer.devices=2")
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (0, 0))
    pipeline = cli.BinderDesignPipeline(args, tmp_path)
    design = next(step for step in pipeline.steps if step.name == "design").get_config()
    assert design.trainer.devices == 2
