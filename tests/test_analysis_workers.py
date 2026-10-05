"""Exercise analysis transport and recovery with real spawned processes."""

# ruff: noqa: INP001, PLR2004, CPY001

from __future__ import annotations

import copy
import json
import os
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from rdkit import Chem
from test_sasa_integrity import (
    CASES,
    dataset,  # noqa: F401 -- pytest fixture
    prepare,
)

from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.esmfold2.contract import (
    ESM_VERSION,
    ESMC_REVISION,
    MODEL_REVISION,
    SCORE_DIR,
    SCORE_KEY,
    file_sha256,
    fingerprint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from boltzgen.task.predict.data_from_generated import FromGeneratedDataset


@dataclass
class _CountedDataset:
    generated_paths: list[Path]
    pickles: int = field(default=0, init=False)

    def __len__(self) -> int:
        return len(self.generated_paths)

    def __getstate__(self) -> dict:
        self.pickles += 1
        return self.__dict__.copy()


class _TransportAnalyze(Analyze):
    """Observe process boundaries without expensive structure calculations."""

    def compute_metrics(self, idx: int) -> tuple | None:
        if self.failure == "once":
            with self.marker.with_name(f"attempt-{idx}").open("a") as attempts:
                attempts.write("attempt\n")
        if idx == 2 and self.failure == "once" and not self.marker.exists():
            self.marker.touch()
            os._exit(17)
        if self.failure == "always":
            os._exit(17)
        if idx == 2 and self.failure == "exception":
            raise ValueError("metric failed")
        if idx == 1 and self.failure == "skip":
            return None
        atom = self.molecule.GetAtomWithIdx(0)
        return (
            idx,
            self.label,
            os.getpid(),
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            atom.GetProp("name"),
            Chem.GetDefaultPickleProperties(),
        )


def _transport(tmp_path: Path, count: int = 8) -> _TransportAnalyze:
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    analyzer = object.__new__(_TransportAnalyze)
    analyzer.data = SimpleNamespace(
        predict_set=_CountedDataset(
            [tmp_path / f"sample_{i}.cif" for i in range(count)]
        )
    )
    analyzer.label = "first"
    analyzer.marker = tmp_path / "crashed"
    analyzer.failure = None
    analyzer.num_processes = 2
    analyzer.molecule = Chem.MolFromSmiles("CC")
    analyzer.molecule.GetAtomWithIdx(0).SetProp("name", "C1")
    return analyzer


@pytest.mark.parametrize(
    ("setting", "threads"),
    [(None, "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "3")],
)
def test_spawn_serializes_dataset_once_per_worker_and_sets_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str | None, threads: str
) -> None:
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    if setting is not None:
        monkeypatch.setenv(setting, threads)
    analyzer = _transport(tmp_path)
    before_threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    results = analyzer.run_parallel(8, 2)
    assert sorted(result[0] for result in results) == list(range(8))
    assert analyzer.data.predict_set.pickles == 2
    assert all(result[2] != os.getpid() for result in results)
    assert all(
        result[3:] == (int(threads), 1, "C1", Chem.PropertyPickleOptions.AllProps)
        for result in results
    )
    assert before_threads == (torch.get_num_threads(), torch.get_num_interop_threads())

    # A later pool gets the current task, without retaining the earlier state.
    analyzer.label = "second"
    results = analyzer.run_parallel(2, 1)
    assert all(result[1] == "second" for result in results)
    assert analyzer.data.predict_set.pickles == 3


def test_spawn_retries_crashed_work_without_losing_successes(tmp_path: Path) -> None:
    analyzer = _transport(tmp_path, 4)
    analyzer.failure = "once"
    results = analyzer.run_parallel(4, 1)
    assert analyzer.marker.exists()
    assert sorted(result[0] for result in results) == list(range(4))
    assert [
        len((tmp_path / f"attempt-{i}").read_text().splitlines()) for i in range(4)
    ] == [1, 1, 2, 1]


def test_spawn_distinguishes_skips_from_errors(tmp_path: Path) -> None:
    analyzer = _transport(tmp_path, 3)
    analyzer.failure = "skip"
    assert sorted(result[0] for result in analyzer.run_parallel(3, 1)) == [0, 2]
    analyzer.failure = "exception"
    with pytest.raises(ValueError, match="metric failed"):
        analyzer.run_parallel(3, 1)


def test_persistent_worker_crashes_fail_instead_of_looping(tmp_path: Path) -> None:
    analyzer = _transport(tmp_path, 1)
    analyzer.failure = "always"
    with pytest.raises(BrokenProcessPool, match="three consecutive pools"):
        analyzer.run_parallel(1, 1)
    assert analyzer.data.predict_set.pickles == 3


def test_small_work_avoids_idle_workers_and_empty_work(tmp_path: Path) -> None:
    analyzer = _transport(tmp_path, 0)
    assert analyzer.run_parallel(0, 32) == []
    assert analyzer.data.predict_set.pickles == 0
    analyzer.data.predict_set.generated_paths.append(tmp_path / "one.cif")
    analyzer.distribute_tasks()
    assert analyzer.data.predict_set.pickles == 0
    assert len(analyzer.run_parallel(1, 32)) == 1
    assert analyzer.data.predict_set.pickles == 1


def test_construction_leaves_parent_thread_pools_unchanged() -> None:
    before = (torch.get_num_threads(), torch.get_num_interop_threads())
    for _ in range(2):
        Analyze(name="construction", data=None)
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == before


def test_real_spawn_matches_serial_metrics_and_checks_score_provenance(
    dataset: FromGeneratedDataset,  # noqa: F811
    tmp_path: Path,
) -> None:
    dataset = copy.copy(dataset)
    dataset.generated_paths = []
    dataset.metadata_paths = []
    dataset.native_paths = []
    for index in range(4):
        feat, _ = prepare(dataset, tmp_path / f"sample_{index}", CASES["partial"])
        path = feat["path"]
        np.savez(path.with_suffix(".npz"), design_mask=feat["design_mask"].numpy())
        dataset.generated_paths.append(path)
        dataset.metadata_paths.append(path.with_suffix(".npz"))
        dataset.native_paths.append(path.with_name(f"{path.stem}_native.cif"))
    analyzer = Analyze(
        name="multiprocessing-contract",
        data=SimpleNamespace(
            predict_set=dataset, cfg=SimpleNamespace(target_id_regex=r"^(sample)_")
        ),
        design_dir=str(tmp_path / "serial"),
        allatom_fold_metrics=False,
        delta_sasa_original=True,
        compute_lddts=False,
        esmfold2_metrics=True,
    )
    scores = analyzer.design_dir / SCORE_DIR
    scores.mkdir()
    for path in dataset.generated_paths:
        request = {"design_id": path.stem, "design_sha256": file_sha256(path)}
        (scores / f"{path.stem}.input.json").write_text(json.dumps(request))
        (scores / f"{path.stem}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_revision": MODEL_REVISION,
                    "esmc_revision": ESMC_REVISION,
                    "esm_version": ESM_VERSION,
                    "input_hash": fingerprint(request),
                    "metrics": {
                        SCORE_KEY: 0.8,
                        "esmfold2_design_to_target_ipsae": 0.8,
                        "esmfold2_target_to_design_ipsae": 0.9,
                    },
                    "selected_sample": 0,
                }
            )
        )
        (scores / f"{path.stem}.cif").write_text("fixture score artifact")
        np.savez(scores / f"{path.stem}.npz", pae=np.zeros((1, 2, 2)))
    analyzer.distribute_tasks()
    serial = {}
    for path in analyzer.metrics_dir.glob("*.npz"):
        with np.load(path, allow_pickle=True) as saved:
            serial[path.name] = dict(saved)
        path.unlink()

    assert sorted(analyzer.run_parallel(4, 2)) == [f"sample_{i}" for i in range(4)]
    for name, expected in serial.items():
        with np.load(analyzer.metrics_dir / name, allow_pickle=True) as saved:
            assert set(saved.files) == set(expected)
            for key, value in expected.items():
                np.testing.assert_equal(saved[key], value)

    # Provenance errors propagate through the executor, just as in serial mode.
    request_path = scores / "sample_0.input.json"
    request = json.loads(request_path.read_text())
    request["design_sha256"] = "stale"
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="is stale"):
        analyzer.run_parallel(1, 1)
