"""Real analysis success/failure contract through validation export."""
# ruff: noqa: INP001

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from test_atom_confidence_export import _real_confidence_features
from test_folding_export_consistency import (
    _features,
    _forward_output,
    _InferenceBoundary,
)

from boltzgen.data.data import Structure
from boltzgen.model.models.boltz import Boltz
from boltzgen.task.analyze import analyze_utils
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.predict.writer import AffinityWriter, FoldingWriter


@pytest.mark.parametrize("affinity_status", ["exception", "skip", "success"])
@pytest.mark.parametrize("novelty", [False, True])
def test_validator_honors_real_analysis_return_and_cleanup(  # noqa: PLR0915
    affinity_status: str,
    novelty: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip(
        "wandb", reason="Refolding validation uses optional dev dependencies"
    )
    from boltzgen.model.validation.refolding import RefoldingValidator  # noqa: PLC0415

    features = _real_confidence_features(None)
    # FromGenerated.get_feat supplies bool design masks to analysis.
    features["design_mask"] = torch.ones_like(features["design_mask"], dtype=torch.bool)
    features["chain_design_mask"] = features["design_mask"].clone()
    if novelty:
        features["str_gen"], _, _ = Structure.from_feat(
            _features(padded=False, missing_atom=False, ligand=False)
        )

    def run_foldseek(command: list[str], *, check: bool) -> None:
        assert check
        queries = sorted(Path(command[2]).glob("*.pdb"))
        assert queries
        Path(command[4]).write_text(
            "".join(f"{path.stem}\ttarget\t0.5\t0.8\t0.6\n" for path in queries)
        )

    monkeypatch.setattr(analyze_utils.subprocess, "run", run_foldseek)

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
    # Analyze sets this process-wide option in its constructor. Keep the test
    # independent of previous Torch work without replacing any analysis logic.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "set_num_interop_threads", lambda _: None)
        analyzer = Analyze(
            name="actual-analysis",
            data=data,
            design_dir=None,
            allatom_fold_metrics=True,
            affinity_metrics=True,
            compute_lddts=False,
            novelty_original=novelty,
            novelty_refolded=novelty,
            novelty_per_target_original=novelty,
            novelty_per_target_refolded=novelty,
        )
    analysis_returns = []
    real_compute_metrics = analyzer.compute_metrics

    def observe_compute_metrics(**kwargs: object) -> str | None:
        result = real_compute_metrics(**kwargs)
        analysis_returns.append(result)
        return result

    analyzer.compute_metrics = observe_compute_metrics
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

    def folding_predict(batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return Boltz.predict_step(
            _InferenceBoundary(_forward_output(batch, 1), 1, mask=True),
            batch,
            batch_idx,
        )

    def affinity_predict(batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:  # noqa: ARG001
        if affinity_status == "success":
            return {"exception": False, "affinity_pred_value": torch.tensor([1.25])}
        return Boltz.predict_step(
            _InferenceBoundary({}, 1, mask=False),
            {affinity_status: [True]},
            batch_idx,
        )

    validator.folding_model = SimpleNamespace(predict_step=folding_predict)
    validator.affinity_model = SimpleNamespace(predict_step=affinity_predict)
    model = SimpleNamespace(
        device=torch.device("cpu"),
        current_epoch=0,
        global_step=0,
        log=Mock(),
        trainer=SimpleNamespace(default_root_dir=tmp_path, global_rank=0),
    )
    validator.process(
        model=model,
        batch={"id": ["contract"]},
        out={"feat_masked": {"design_mask": features["design_mask"].unsqueeze(0)}},
        idx_dataset=0,
        dataloader_idx=0,
        n_samples=1,
        batch_idx=0,
    )
    sample_id = "sample0_batch0_rank0_contract"
    succeeded = affinity_status == "success"
    assert analysis_returns == ([sample_id] if succeeded else [None])
    assert validator.writer.failed == 0
    assert validator.aff_writer.failed == int(affinity_status == "exception")
    assert (validator.writer.refold_cif_dir / f"{sample_id}.cif").is_file()
    assert (validator.aff_writer.outdir / f"{sample_id}.npz").exists() == succeeded
    assert not (validator.writer.outdir / f"{sample_id}.npz").exists()
    assert len(list(analyzer.metrics_dir.glob("*.npz"))) == (2 if succeeded else 0)
    assert len(validator.all_refold_metrics["val_monomer"]) == int(succeeded)
    assert len(validator.all_refolding_data["val_monomer"]) == int(succeeded)
    if succeeded:
        assert (
            validator.all_refold_metrics["val_monomer"][0]["affinity_pred_value"]
            == 1.25  # noqa: PLR2004
        )
        assert validator.all_refolding_data["val_monomer"][0]["sample_id"] == sample_id
        if novelty:
            metrics, _ = analyzer.compute_novelty(suffix=Path("rank0"))
            assert len(metrics) == 6  # noqa: PLR2004
            assert all(value == pytest.approx(0.7) for value in metrics.values())
