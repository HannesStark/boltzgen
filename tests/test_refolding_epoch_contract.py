"""Portable regressions for refolding validation epochs with skipped affinity output."""
# ruff: noqa: INP001

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import torch
from test_atom_confidence_export import _real_confidence_features
from test_folding_export_consistency import _forward_output, _InferenceBoundary

from boltzgen.model.models.boltz import Boltz
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.predict.writer import AffinityWriter, FoldingWriter

if TYPE_CHECKING:
    from boltzgen.model.validation.refolding import RefoldingValidator


def _make_validator(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RefoldingValidator, Analyze, dict[str, Any], SimpleNamespace]:
    from boltzgen.model.validation.refolding import RefoldingValidator  # noqa: PLC0415

    features = _real_confidence_features(None)
    features["design_mask"] = torch.ones_like(features["design_mask"], dtype=torch.bool)
    features["chain_design_mask"] = features["design_mask"].clone()

    def get_feat(path: Path, design_mask: torch.Tensor | None = None) -> dict[str, Any]:  # noqa: ARG001
        feat = dict(features)
        feat["id"] = path.stem
        feat["path"] = path
        return feat

    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"(contract)"),
        return_native=False,
        predict_set=SimpleNamespace(
            get_feat=get_feat,
            get_sample=lambda design_dir, sample_id: get_feat(
                Path(design_dir) / f"{sample_id}.cif"
            ),
        ),
        transfer_batch_to_device=Mock(),
        init_dataset=Mock(),
    )
    # Analyze configures PyTorch's process-wide interop pool in its constructor.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "set_num_interop_threads", lambda _: None)
        analyzer = Analyze(
            name="actual-analysis",
            data=data,
            design_dir=None,
            allatom_fold_metrics=True,
            affinity_metrics=True,
            compute_lddts=False,
        )

    validator = object.__new__(RefoldingValidator)
    validator.dataset_to_logname = {0: "val_monomer"}
    validator.inverse_fold = True
    validator.design_dir = None
    validator.writer = FoldingWriter(None)
    validator.aff_writer = AffinityWriter(None)
    validator.all_refold_metrics = defaultdict(list)
    validator.all_refolding_data = defaultdict(list)
    validator.design_val_step = Mock(return_value=True)
    validator.init_folding_model = Mock()
    validator.init_affinity_model = Mock()
    validator.ligand_plip = False
    validator.analyze_task = analyzer
    model = SimpleNamespace(
        device=torch.device("cpu"),
        current_epoch=0,
        global_step=0,
        log=Mock(),
        logger=None,
        trainer=SimpleNamespace(
            default_root_dir=root,
            global_rank=0,
            world_size=1,
        ),
    )
    return validator, analyzer, features, model


def _process(
    validator: RefoldingValidator,
    features: dict[str, Any],
    model: SimpleNamespace,
    *,
    status: str,
    batch_idx: int,
) -> None:
    def fold(batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return Boltz.predict_step(
            _InferenceBoundary(_forward_output(batch, 1), 1, mask=True),
            batch,
            batch_idx,
        )

    def affinity(batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:  # noqa: ARG001
        if status == "success":
            return {"exception": False, "affinity_pred_value": torch.tensor([1.25])}
        return Boltz.predict_step(
            _InferenceBoundary({}, 1, mask=False),
            {status: [True]},
            batch_idx,
        )

    validator.folding_model = SimpleNamespace(predict_step=fold)
    validator.affinity_model = SimpleNamespace(predict_step=affinity)
    validator.process(
        model=model,
        batch={"id": ["contract"]},
        out={"feat_masked": {"design_mask": features["design_mask"].unsqueeze(0)}},
        idx_dataset=0,
        dataloader_idx=0,
        n_samples=1,
        batch_idx=batch_idx,
    )


@pytest.mark.parametrize(
    ("statuses", "expected_targets"),
    [
        pytest.param(("exception",), 0, id="single-failed-affinity"),
        pytest.param(("exception", "skip"), 0, id="multiple-failed-affinities"),
        pytest.param(("success",), 1, id="successful-affinity"),
        pytest.param(("exception", "success"), 1, id="mixed-failed-and-successful"),
    ],
)
def test_refolding_epoch_end_handles_empty_and_mixed_results(
    statuses: tuple[str, ...],
    expected_targets: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip(
        "wandb", reason="Refolding validation uses optional dev dependencies"
    )
    validator, analyzer, features, model = _make_validator(tmp_path, monkeypatch)
    for batch_idx, status in enumerate(statuses):
        _process(
            validator,
            features,
            model,
            status=status,
            batch_idx=batch_idx,
        )

    sample_metrics = list(analyzer.metrics_dir.glob("*.npz"))
    assert len(sample_metrics) == 2 * int(expected_targets == 1)

    validator.on_epoch_end_refolding(model, "val_monomer")

    logged_targets = [
        call.args[1]
        for call in model.log.call_args_list
        if call.args[0] == "val_monomer/num_targets"
    ]
    assert logged_targets == [expected_targets]
    assert validator.all_refold_metrics["val_monomer"] == []
    assert validator.all_refolding_data["val_monomer"] == []


def test_refolding_epoch_gathers_remote_success_before_empty_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip(
        "wandb", reason="Refolding validation uses optional dev dependencies"
    )
    validator, _analyzer, features, model = _make_validator(tmp_path, monkeypatch)
    _process(
        validator,
        features,
        model,
        status="success",
        batch_idx=0,
    )
    remote_metrics = list(validator.all_refold_metrics["val_monomer"])
    remote_data = list(validator.all_refolding_data["val_monomer"])
    validator.all_refold_metrics["val_monomer"] = []
    validator.all_refolding_data["val_monomer"] = []

    model.trainer.world_size = 2
    gathered = iter([remote_metrics, remote_data])

    def all_gather_object(object_list: list[Any], local_object: list[Any]) -> None:
        object_list[0] = local_object
        object_list[1] = next(gathered)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    validator.on_epoch_end_refolding(model, "val_monomer")

    logged_targets = [
        call.args[1]
        for call in model.log.call_args_list
        if call.args[0] == "val_monomer/num_targets"
    ]
    assert logged_targets == [1]
    assert validator.all_refold_metrics["val_monomer"] == []
    assert validator.all_refolding_data["val_monomer"] == []
