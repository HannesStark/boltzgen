"""Public merge and generated-dataset resume contracts."""
# ruff: noqa: INP001

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_folding_export_consistency import _features

from boltzgen.cli import boltzgen as cli
from boltzgen.data import const
from boltzgen.task.esmfold2.contract import (
    ESM_VERSION,
    ESMC_REVISION,
    MODEL_REVISION,
    SCORE_DIR,
    file_sha256,
    fingerprint,
    load_result,
)
from boltzgen.task.predict import data_from_generated
from boltzgen.task.predict.writer import DesignWriter


def _pair(directory: Path, stem: str, identity: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.cif").write_text(identity)
    np.savez(directory / f"{stem}.npz", identity=identity)


def _merge(monkeypatch: pytest.MonkeyPatch, sources: list[Path], output: Path) -> None:
    monkeypatch.setattr(
        sys, "argv", ["boltzgen", "merge", *map(str, sources), "--output", str(output)]
    )
    cli.main()


@pytest.mark.parametrize("multiplicity", [1, 3, 12])
def test_merge_keeps_backbones_and_sequences_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplicity: int
) -> None:
    sources = [tmp_path / "Run A", tmp_path / "Run B"]
    for source in sources:
        backbones = source / "intermediate_designs"
        sequences = source / "intermediate_designs_inverse_folded"
        rows = []
        # One backbone has no inverse-folded outputs yet.
        for stem in ("target_with_underscores_0", "target_with_underscores_1"):
            _pair(backbones, stem, f"{source.name}:{stem}")
            (backbones / f"{stem}_native.cif").write_text(f"native:{stem}")
            (backbones / f"{stem}_native.pdb").write_text(f"native-pdb:{stem}")
        (backbones / "not_a_design.cif").mkdir()
        pd.DataFrame(
            [
                {"id": stem, "sequence": "GG"}
                for stem in ("target_with_underscores_0", "target_with_underscores_1")
            ]
        ).to_pickle(backbones / "ca_coords_sequences.pkl.gz")
        for sample in range(multiplicity):
            stem = "target_with_underscores_0"
            if multiplicity > 1:
                stem += f"_{sample:0{len(str(multiplicity - 1))}d}"
            _pair(sequences, stem, f"{source.name}:{stem}")
            for folder in (const.refold_cif_dirname, const.refold_design_cif_dirname):
                (sequences / folder).mkdir(exist_ok=True)
                (sequences / folder / f"{stem}.cif").write_text(f"{folder}:{stem}")
            rows.append({"id": stem, "file_name": f"{stem}.cif", "score": sample})
        pd.DataFrame(rows).to_csv(
            sequences / "aggregate_metrics_analyze.csv", index=False
        )
        pd.DataFrame([{"id": row["id"], "sequence": "GG"} for row in rows]).to_pickle(
            sequences / "ca_coords_sequences.pkl.gz"
        )

    output = tmp_path / "merged"
    _merge(monkeypatch, sources, output)
    for dirname in ("intermediate_designs", "intermediate_designs_inverse_folded"):
        merged = output / dirname
        for source in sources:
            tag = source.name.lower().replace(" ", "-")
            for path in (source / dirname).rglob("*"):
                if path.is_file() and path.suffix in (".cif", ".pdb", ".npz"):
                    relative = path.relative_to(source / dirname)
                    renamed = relative.with_name(f"{tag}_{relative.name}")
                    assert (merged / renamed).read_bytes() == path.read_bytes()
        assert not (merged / "run-a_not_a_design.cif").exists()
        ids = pd.read_pickle(merged / "ca_coords_sequences.pkl.gz")["id"].tolist()  # noqa: S301
        assert len(ids) == (
            4 if dirname == "intermediate_designs" else 2 * multiplicity
        )
        assert len(set(ids)) == len(ids)
    metrics = pd.read_csv(
        output / "intermediate_designs_inverse_folded" / "aggregate_metrics_analyze.csv"
    )
    assert metrics["id"].tolist() == ids
    assert metrics["file_name"].tolist() == [f"{stem}.cif" for stem in ids]

    # Running the same command again preserves names and copied content.
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.suffix != ".gz"
    }
    _merge(monkeypatch, sources, output)
    assert before == {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.suffix != ".gz"
    }


@pytest.mark.parametrize(
    "names", [("run", "run"), ("RUN!", "run"), ("run", "run", "run-2")]
)
def test_merge_disambiguates_source_tags_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> None:
    sources = [tmp_path / str(index) / name for index, name in enumerate(names)]
    for index, source in enumerate(sources):
        designs = source / "intermediate_designs_inverse_folded"
        _pair(designs, "candidate_0", str(index))
        pd.DataFrame([{"id": "candidate_0", "file_name": "candidate_0.cif"}]).to_csv(
            designs / "aggregate_metrics_analyze.csv", index=False
        )
    output = tmp_path / "merged"
    _merge(monkeypatch, sources, output)
    designs = output / "intermediate_designs_inverse_folded"
    rows = pd.read_csv(designs / "aggregate_metrics_analyze.csv")
    assert rows["id"].is_unique
    for index, row in rows.iterrows():
        assert (designs / row["file_name"]).read_text() == str(index)
        with np.load(designs / f"{row['id']}.npz") as metadata:
            assert metadata["identity"].item() == str(index)
    assert rows.iloc[0]["id"] == "run_candidate_0"
    if "run-2" in names:
        assert rows.iloc[-1]["id"] == "run-2_candidate_0"


def _module(
    inputs: Path, outputs: Path, monkeypatch: pytest.MonkeyPatch, multiplicity: int
) -> data_from_generated.FromGeneratedDataModule:
    # Canonical molecule loading is unrelated to discovery/reuse; no structures
    # are fetched when inspecting the real prediction dataset's scheduled work.
    monkeypatch.setattr(data_from_generated, "load_canonicals", lambda _: {})
    cfg = data_from_generated.DataConfig(
        num_targets=None,
        samples_per_target=100,
        moldir=str(inputs),
        tokenizer=None,
        featurizer=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        inverse_fold=True,
        multiplicity=multiplicity,
    )
    return data_from_generated.FromGeneratedDataModule(
        cfg,
        design_dir=str(inputs),
        output_dir=str(outputs),
        skip_existing=True,
        skip_existing_kind="inverse_fold",
    )


def test_merge_counts_repeated_source_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs_inverse_folded"
    _pair(designs, "candidate", "sequence")
    pd.DataFrame([{"id": "candidate", "file_name": "candidate.cif"}]).to_csv(
        designs / "aggregate_metrics_analyze.csv", index=False
    )
    output = tmp_path / "merged"
    _merge(monkeypatch, [source, source / "."], output)
    rows = pd.read_csv(output / designs.name / "aggregate_metrics_analyze.csv")
    assert rows["id"].tolist() == ["run_candidate"]


@pytest.mark.parametrize("stem", ["target_0", "target.v1", "target.v1.2"])
@pytest.mark.parametrize("with_metrics", [False, True])
def test_merge_preserves_design_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stem: str,
    with_metrics: bool,
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs_inverse_folded"
    _pair(designs, stem, "original")
    if with_metrics:
        pd.DataFrame([{"id": stem, "file_name": f"{stem}.cif"}]).to_csv(
            designs / "aggregate_metrics_analyze.csv", index=False
        )
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    merged = output / designs.name
    assert sorted(path.name for path in merged.glob("*.cif")) == [f"run_{stem}.cif"]
    module = _module(merged, tmp_path / "unused", monkeypatch, multiplicity=1)
    assert len(module.predict_set.generated_paths) == 1
    assert all(path.is_file() for path in module.predict_set.metadata_paths)


@pytest.mark.parametrize("multiplicity", [1, 3, 12])
@pytest.mark.parametrize("missing", ["cif", "npz"])
def test_inverse_fold_resume_requires_every_sequence_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplicity: int, missing: str
) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    for stem in ("target_0", "target_0_0", "target_1", "target_2"):
        _pair(inputs, stem, stem)
    for stem in ("target_0", "target_1"):
        for sample in range(multiplicity):
            output_stem = stem
            if multiplicity > 1:
                output_stem += f"_{sample:0{len(str(multiplicity - 1))}d}"
            _pair(outputs, output_stem, stem)
    (outputs / f"{output_stem}.{missing}").unlink()
    module = _module(inputs, outputs, monkeypatch, multiplicity)
    assert [path.stem for path in module.predict_set.generated_paths] == [
        "target_0_0",
        "target_1",
        "target_2",
    ]
    assert len(module.predict_dataloader()) == 3 * multiplicity
    assert [path.stem for path in module.predict_set.metadata_paths] == [
        "target_0_0",
        "target_1",
        "target_2",
    ]


def test_inverse_fold_resume_does_not_mix_padding_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    _pair(inputs, "target_0", "input")
    multiplicity = 12
    for sample in range(multiplicity):
        _pair(outputs, f"target_0_{sample:02d}", "output")
    (outputs / "target_0_00.npz").rename(outputs / "target_0_0.npz")
    assert (
        len(_module(inputs, outputs, monkeypatch, multiplicity).predict_set)
        == multiplicity
    )


@pytest.mark.parametrize("multiplicity", [1, 3, 12])
def test_inverse_fold_writer_outputs_are_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplicity: int
) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    _pair(inputs, "contract", "input")
    writer = DesignWriter(
        str(outputs),
        res_atoms_only=False,
        atom14=False,
        inverse_fold=True,
        write_native=False,
    )
    cfg = SimpleNamespace(multiplicity=multiplicity)
    trainer = SimpleNamespace(datamodule=SimpleNamespace(cfg=cfg))
    for sample in range(multiplicity):
        features = _features(padded=False, missing_atom=False, ligand=False)
        features["data_sample_idx"] = sample
        batch = data_from_generated.collate([features])
        batch["extra_mols"] = None
        prediction = {key: value for key, value in batch.items() if key != "extra_mols"}
        prediction["coords"] = batch["coords"][0]
        prediction["exception"] = False
        writer.write_on_batch_end(trainer=trainer, prediction=prediction, batch=batch)
    assert writer.failed == 0
    assert len(list(outputs.glob("*.cif"))) == multiplicity
    assert len(_module(inputs, outputs, monkeypatch, multiplicity).predict_set) == 0


def test_merge_keeps_multisequence_score_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs_inverse_folded"
    _pair(source / "intermediate_designs", "candidate", "backbone")
    _pair(designs, "candidate_0", "sequence")
    scores = designs / SCORE_DIR
    scores.mkdir()
    request = {
        "design_id": "candidate_0",
        "design_sha256": file_sha256(designs / "candidate_0.cif"),
    }
    metrics = {
        "esmfold2_ipsae_min": 0.5,
        "esmfold2_design_to_target_ipsae": 0.5,
        "esmfold2_target_to_design_ipsae": 0.6,
    }
    result = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "input_hash": fingerprint(request),
        "metrics": metrics,
    }
    (scores / "candidate_0.input.json").write_text(json.dumps(request))
    (scores / "candidate_0.json").write_text(json.dumps(result))
    _pair(scores, "candidate_0", "ESMFold2 output")
    pd.DataFrame(
        [
            {
                "id": "candidate_0",
                "file_name": "candidate_0.cif",
                "esmfold2_input_hash": fingerprint(request),
                **metrics,
            }
        ]
    ).to_csv(designs / "aggregate_metrics_analyze.csv", index=False)
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    merged = output / designs.name
    renamed = json.loads(
        (merged / SCORE_DIR / "run_candidate_0.input.json").read_text()
    )
    assert renamed == dict(request, design_id="run_candidate_0")
    row = pd.read_csv(merged / "aggregate_metrics_analyze.csv").iloc[0]
    assert row["esmfold2_input_hash"] == fingerprint(renamed)
    merged_result = load_result(
        merged / SCORE_DIR / "run_candidate_0.json", fingerprint(renamed)
    )
    assert merged_result["metrics"] == metrics
    assert merged_result["merged_from"] == {
        "design_id": "candidate_0",
        "input_hash": fingerprint(request),
    }
