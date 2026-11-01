"""Binding-site fractions include a single annotated hotspot."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_multichain_analysis import _features

from boltzgen.task.analyze.analyze import Analyze


@pytest.mark.parametrize("site_count", [0, 1, 2])
def test_binding_site_fraction_uses_strict_distance_to_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site_count: int
) -> None:
    feat = _features(tmp_path / "site.cif", ("GGG", "GGG"))
    feat["design_mask"] = torch.tensor([False, False, False, True, True, True])
    feat["chain_design_mask"] = feat["design_mask"].clone()
    feat["binding_type"][:site_count] = 1
    feat["center_coords"] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    )
    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"^(.*)$"),
        predict_set=SimpleNamespace(get_sample=lambda **_: feat),
    )
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    task = Analyze(
        "test",
        data,
        design_dir=str(tmp_path),
        allatom_fold_metrics=False,
        compute_lddts=False,
    )
    assert task.compute_metrics(sample_id=feat["id"]) == feat["id"]
    with np.load(task.metrics_dir / f"metrics_{feat['id']}.npz") as metrics:
        for threshold in task.bindsite_adherence_thresholds:
            key = f"bindsite_under_{threshold}rmsd"
            if site_count == 0:
                assert key not in metrics
            else:
                expected = (1.0 / site_count) if threshold > 4 else 0.0
                assert metrics[key].item() == pytest.approx(expected)
