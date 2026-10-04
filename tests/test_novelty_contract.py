"""Empty-rank novelty preserves the helper's DataFrame contract."""

# ruff: noqa: INP001
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from boltzgen.task.analyze import analyze_utils
from boltzgen.task.analyze.analyze import Analyze


def test_empty_foldseek_returns_typed_frame_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Mock(side_effect=AssertionError("Foldseek must not run with no input"))
    monkeypatch.setattr(analyze_utils.subprocess, "run", run)
    result = analyze_utils.compute_novelty_foldseek(
        indir=tmp_path / "missing-input",
        outdir=tmp_path,
        reference_db=tmp_path / "missing-db",
        files=[],
    )
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["query", "novelty"]
    assert result["query"].dtype == np.dtype("object")
    assert result["novelty"].dtype == np.dtype("float64")
    run.assert_not_called()


@pytest.mark.parametrize(
    ("flag", "metric_keys"),
    [
        ("novelty_original", ["novelty_original"]),
        ("novelty_refolded", ["novelty_refolded"]),
        (
            "novelty_per_target_original",
            ["mean_novelty_per_target_original", "median_novelty_per_target_original"],
        ),
        (
            "novelty_per_target_refolded",
            ["mean_novelty_per_target_refolded", "median_novelty_per_target_refolded"],
        ),
    ],
)
def test_actual_analyze_empty_local_rank_reports_undefined_novelty(
    flag: str,
    metric_keys: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(side_effect=AssertionError("Foldseek must not run with no input"))
    monkeypatch.setattr(analyze_utils.subprocess, "run", run)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "set_num_interop_threads", lambda _: None)
        analyzer = Analyze(
            name="empty-local-rank",
            data=SimpleNamespace(),
            design_dir=str(tmp_path),
            compute_lddts=False,
            **{flag: True},
        )
    metrics, data = analyzer.compute_novelty(suffix=Path("rank0"))
    assert set(metrics) == set(metric_keys)
    assert all(np.isnan(value) for value in metrics.values())
    assert all(frame.empty for frame in data.values())
    run.assert_not_called()


@pytest.mark.parametrize("scenario", ["all_hit", "partial_hit", "empty_alignment"])
def test_nonempty_foldseek_preserves_search_and_hit_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    calls = []

    def run(command: list[str], check: bool) -> None:
        calls.append((command, check))
        rows = "design1\ttargetA\t0.5\t0.8\t0.4\ndesign1\ttargetB\t0.5\t0.9\t0.5\n"
        if scenario == "all_hit":
            rows += "design2\ttargetC\t0.5\t0.2\t0.4\n"
        if scenario == "empty_alignment":
            rows = ""
        Path(command[4]).write_text(rows)

    monkeypatch.setattr(analyze_utils.subprocess, "run", run)
    result = analyze_utils.compute_novelty_foldseek(
        indir=tmp_path / "input",
        outdir=tmp_path,
        reference_db=tmp_path / "reference-db",
        files=["design1.pdb", "design2.pdb"],
        foldseek_binary="fixture-foldseek",
    )
    assert len(calls) == 1
    assert calls[0][0][:4] == [
        "fixture-foldseek",
        "easy-search",
        str(tmp_path / "input"),
        str(tmp_path / "reference-db"),
    ]
    assert calls[0][1] is True
    assert list(result["query"]) == ["design1", "design2"]
    expected = {
        "all_hit": [0.7, 0.3],
        "partial_hit": [0.7, 0.0],
        "empty_alignment": [0.0, 0.0],
    }[scenario]
    np.testing.assert_allclose(result["novelty"].to_numpy(dtype=float), expected)
    assert result["novelty"].mean() == pytest.approx(float(np.mean(expected)))
