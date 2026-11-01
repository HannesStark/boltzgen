"""Analysis scratch structures must describe the current sample."""

# Pytest injects the fixture imported from the shared SASA tests.
# ruff: noqa: F811

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_sasa_integrity import (
    CASES,
    ResidueSpec,
    dataset,  # noqa: F401
    prepare,
    structure_from_specs,
)

from boltzgen.data import const
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.analyze.analyze import Analyze


@pytest.mark.parametrize("refolded", [False, True])
@pytest.mark.parametrize("stale_content", ["missing", "empty", "truncated", "other"])
def test_analysis_rebuilds_scratch_from_current_features(
    dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refolded: bool,
    stale_content: str,
) -> None:
    feat, directory = prepare(dataset, tmp_path / "sample", CASES["full"])
    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"^(.*)$"),
        predict_set=SimpleNamespace(get_sample=lambda **_: feat),
    )
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    task = Analyze(
        "test",
        data,
        design_dir=str(directory / "analysis"),
        allatom_fold_metrics=False,
        backbone_fold_metrics=False,
        largest_hydrophobic=not refolded,
        largest_hydrophobic_refolded=refolded,
        compute_lddts=False,
    )
    if refolded:
        task.fold_metrics = True
        folder = task.design_dir / const.folding_dirname
        folder.mkdir(parents=True)
        np.savez(
            folder / f"{feat['id']}.npz",
            res_type=feat["res_type"].numpy()[None],
            coords=feat["coords"].numpy(),
            design_to_target_iptm=np.array([0.8]),
            design_ptm=np.array([0.8]),
        )
        scratch = task.des_refold_pdb_dir / f"{feat['id']}_des_refold.cif"
    else:
        scratch = task.des_pdb_dir / f"{feat['id']}_des.cif"
    key = "design_largest_hydrophobic_patch" + ("_refolded" if refolded else "")
    assert task.compute_metrics(sample_id=feat["id"]) == feat["id"]
    with np.load(task.metrics_dir / f"metrics_{feat['id']}.npz") as output:
        expected = output[key].item()
    assert np.isfinite(expected)
    assert not scratch.exists()

    if stale_content == "empty":
        scratch.write_text("")
    elif stale_content == "truncated":
        scratch.write_text(
            "data_interrupted\nloop_\n_atom_site.id\n_atom_site.Cartn_x\n1"
        )
    elif stale_content == "other":
        other = structure_from_specs(
            [ResidueSpec("A", "GLY", (0, 0, 0), designed=True)]
        )
        scratch.write_text(to_mmcif(other))

    assert task.compute_metrics(sample_id=feat["id"]) == feat["id"]
    with np.load(task.metrics_dir / f"metrics_{feat['id']}.npz") as output:
        assert output[key].item() == pytest.approx(expected)
    assert not scratch.exists()
