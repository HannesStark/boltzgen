"""Public merge and generated-dataset resume contracts."""
# ruff: noqa: INP001

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from test_folding_export_consistency import _features

from boltzgen.cli import boltzgen as cli
from boltzgen.data import const
from boltzgen.data.parse.schema import parse_entity
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
    inputs: Path,
    outputs: Path,
    monkeypatch: pytest.MonkeyPatch,
    multiplicity: int,
    *,
    return_native: bool = False,
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
        return_native=return_native,
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


@pytest.mark.parametrize("stem", ["target_0", "target.v1", "target.v1.2", "target_gen"])
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


@pytest.mark.parametrize("stems", [("001", "1"), ("NA", "null")])
def test_merge_preserves_csv_identifier_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stems: tuple[str, str]
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs_inverse_folded"
    for stem in stems:
        _pair(designs, stem, stem)
    pd.DataFrame(
        [
            {"id": stem, "file_name": f"{stem}.cif", "esmfold2_input_hash": None}
            for stem in stems
        ]
    ).to_csv(designs / "aggregate_metrics_analyze.csv", index=False)
    pd.DataFrame([{"id": stem, "sequence": "GG"} for stem in stems]).to_pickle(
        designs / "ca_coords_sequences.pkl.gz"
    )
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    merged = output / designs.name
    rows = pd.read_csv(merged / "aggregate_metrics_analyze.csv")
    assert rows["id"].tolist() == [f"run_{stem}" for stem in stems]
    assert rows["esmfold2_input_hash"].isna().all()
    for stem in stems:
        assert (merged / f"run_{stem}.cif").read_text() == stem
        with np.load(merged / f"run_{stem}.npz") as metadata:
            assert metadata["identity"].item() == stem
    sequences = pd.read_pickle(merged / "ca_coords_sequences.pkl.gz")  # noqa: S301
    assert sequences["id"].tolist() == rows["id"].tolist()


@pytest.mark.parametrize("with_metrics", [False, True])
@pytest.mark.parametrize("stem", ["target_gen", "target_gen.cif_gen"])
def test_merge_preserves_legacy_companions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_metrics: bool, stem: str
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    _pair(designs, stem, "legacy")
    (designs / f"{stem}.npz").rename(designs / f"{stem[:-4]}_metadata.npz")
    (designs / f"{stem[:-4]}_native.cif").write_text("native")
    (designs / f"{stem[:-4]}_native.pdb").write_text("native-pdb")
    if with_metrics:
        pd.DataFrame([{"id": stem, "file_name": f"{stem}.cif"}]).to_csv(
            designs / "aggregate_metrics_analyze.csv", index=False
        )
    before = _module(designs, tmp_path / "unused", monkeypatch, multiplicity=1)
    assert before.predict_set.metadata_paths[0].is_file()
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    merged = output / designs.name
    after = _module(merged, tmp_path / "unused", monkeypatch, multiplicity=1)
    assert len(after.predict_set.generated_paths) == 1
    assert after.predict_set.metadata_paths[0].read_bytes() == (
        before.predict_set.metadata_paths[0].read_bytes()
    )
    assert (merged / f"run_{stem}_native.cif").read_text() == "native"
    assert (merged / f"run_{stem}_native.pdb").read_text() == "native-pdb"


@pytest.mark.parametrize("conflict", [False, True])
def test_merge_preserves_custom_molecules_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: bool
) -> None:
    sources = [tmp_path / "first", tmp_path / "second"]
    for index, source in enumerate(sources):
        designs = source / "intermediate_designs_inverse_folded"
        _pair(designs, "candidate", source.name)
        pd.DataFrame([{"id": "candidate", "file_name": "candidate.cif"}]).to_csv(
            designs / "aggregate_metrics_analyze.csv", index=False
        )
        molecules = designs / const.molecules_dirname
        molecules.mkdir()
        molecule = Chem.MolFromSmiles("CCN" if conflict and index else "CCO")
        (molecules / "LIG0.pkl").write_bytes(pickle.dumps(molecule))
    output = tmp_path / "merged"
    if conflict:
        with pytest.raises(ValueError, match=r"Conflicting molecule definition.*LIG0"):
            _merge(monkeypatch, sources, output)
    else:
        _merge(monkeypatch, sources, output)
        _merge(monkeypatch, sources, output)
    relative = (
        Path("intermediate_designs_inverse_folded")
        / const.molecules_dirname
        / "LIG0.pkl"
    )
    assert (output / relative).read_bytes() == (sources[0] / relative).read_bytes()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_merge_accepts_same_molecule_with_different_reference_conformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_destination: bool
) -> None:
    sources = [tmp_path / "first", tmp_path / "second"]
    extra_mols, *_ = parse_entity(
        {"ligand": {"id": "L", "smiles": "CCO"}},
        {},
        tmp_path,
        0,
        is_msa_custom=False,
        is_msa_auto=False,
    )
    molecule = extra_mols["LIG0"]
    copies = [Chem.Mol(molecule), Chem.Mol(molecule)]
    # Reference conformers vary between normal parses of the same design spec.
    conformer = copies[1].GetConformer()
    position = conformer.GetAtomPosition(0)
    conformer.SetAtomPosition(0, (position.x + 0.1, position.y, position.z))
    relative = Path("intermediate_designs") / const.molecules_dirname / "LIG0.pkl"
    for source, copy in zip(sources, copies, strict=True):
        _pair(source / "intermediate_designs", "candidate", source.name)
        path = source / relative
        path.parent.mkdir()
        path.write_bytes(pickle.dumps(copy))
    assert (sources[0] / relative).read_bytes() != (sources[1] / relative).read_bytes()
    output = tmp_path / "merged"
    if existing_destination:
        (output / relative).parent.mkdir(parents=True)
        (output / relative).write_bytes((sources[1] / relative).read_bytes())
    _merge(monkeypatch, sources, output)
    assert (output / relative).read_bytes() == (sources[0] / relative).read_bytes()
    _merge(monkeypatch, sources, output)
    assert (output / relative).read_bytes() == (sources[0] / relative).read_bytes()


def test_merge_rejects_incompatible_old_destination_molecule_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "merged"
    relative = Path("intermediate_designs") / const.molecules_dirname / "LIG0.pkl"
    for index, smiles in enumerate(("CCO", "CCN")):
        source = tmp_path / str(index) / "run"
        _pair(source / "intermediate_designs", "candidate", source.name)
        path = source / relative
        path.parent.mkdir()
        path.write_bytes(pickle.dumps(Chem.MolFromSmiles(smiles)))
        if index == 0:
            _merge(monkeypatch, [source], output)
            assert (output / relative).read_bytes() == path.read_bytes()
        else:
            before = (output / relative).read_bytes()
            with pytest.raises(ValueError, match="Conflicting molecule definition"):
                _merge(monkeypatch, [source], output)
            assert (output / relative).read_bytes() == before


@pytest.mark.parametrize("difference", ["atom_order", "atom_name", "charge", "stereo"])
def test_merge_rejects_molecule_identity_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: str
) -> None:
    original = Chem.MolFromSmiles("F[C@H](Cl)Br")
    changed = Chem.Mol(original)
    if difference == "atom_order":
        changed = Chem.RenumberAtoms(changed, [3, 2, 1, 0])
    elif difference == "atom_name":
        changed.GetAtomWithIdx(0).SetProp("name", "changed")
    elif difference == "charge":
        changed.GetAtomWithIdx(0).SetFormalCharge(-1)
    else:
        changed = Chem.MolFromSmiles("F[C@@H](Cl)Br")
    sources = [tmp_path / "first", tmp_path / "second"]
    previous_flags = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for source, molecule in zip(sources, (original, changed), strict=True):
            designs = source / "intermediate_designs"
            _pair(designs, "candidate", source.name)
            molecule_dir = designs / const.molecules_dirname
            molecule_dir.mkdir()
            (molecule_dir / "LIG0.pkl").write_bytes(pickle.dumps(molecule))
    finally:
        Chem.SetDefaultPickleProperties(previous_flags)
    with pytest.raises(ValueError, match=r"Conflicting molecule definition.*LIG0"):
        _merge(monkeypatch, sources, tmp_path / "merged")


def test_generated_reader_prefers_modern_gen_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    _pair(inputs, "target_gen", "modern")
    module = _module(inputs, tmp_path / "outputs", monkeypatch, multiplicity=1)
    assert module.predict_set.metadata_paths == [inputs / "target_gen.npz"]


@pytest.mark.parametrize("first_legacy", [False, True])
def test_merge_replaces_metadata_when_source_layout_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_legacy: bool
) -> None:
    output = tmp_path / "merged"
    for index, legacy in enumerate((first_legacy, not first_legacy)):
        source = tmp_path / str(index) / "run"
        designs = source / "intermediate_designs"
        _pair(designs, "target_gen", str(index))
        native_stem = "target" if legacy else "target_gen"
        (designs / f"{native_stem}_native.cif").write_text(f"native-{index}")
        if legacy:
            (designs / "target_gen.npz").rename(designs / "target_metadata.npz")
        _merge(monkeypatch, [source], output)
        module = _module(
            output / designs.name,
            tmp_path / "unused",
            monkeypatch,
            1,
            return_native=True,
        )
        assert module.predict_set.generated_paths[0].read_text() == str(index)
        with np.load(module.predict_set.metadata_paths[0]) as metadata:
            assert metadata["identity"].item() == str(index)
        assert module.predict_set.native_paths[0].read_text() == f"native-{index}"


def test_merge_removes_missing_optional_companions_on_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "merged"
    for index in range(2):
        source = tmp_path / str(index) / "run"
        designs = source / "intermediate_designs"
        _pair(designs, "target", str(index))
        if index == 0:
            (designs / "target_native.cif").write_text("old native")
            (designs / "target_native.pdb").write_text("old native pdb")
            for folder in (const.refold_cif_dirname, const.refold_design_cif_dirname):
                (designs / folder).mkdir()
                (designs / folder / "target.cif").write_text("old refold")
        else:
            (designs / "target.npz").unlink()
        _merge(monkeypatch, [source], output)
    merged = output / designs.name
    assert (merged / "run_target.cif").read_text() == "1"
    for filename in (
        "run_target.npz",
        "run_target_native.cif",
        "run_target_native.pdb",
    ):
        assert not (merged / filename).exists()
    for folder in (const.refold_cif_dirname, const.refold_design_cif_dirname):
        assert not (merged / folder / "run_target.cif").exists()


def test_merge_rejects_incomplete_replacement_over_legacy_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    designs.mkdir(parents=True)
    (designs / "target_gen.cif").write_text("new incomplete design")
    output = tmp_path / "merged"
    merged = output / designs.name
    merged.mkdir(parents=True)
    (merged / "run_target_gen.cif").write_text("old legacy design")
    np.savez(merged / "run_target_metadata.npz", identity="old legacy metadata")
    (merged / "run_target_native.cif").write_text("old legacy native")
    before = {path.name: path.read_bytes() for path in merged.iterdir()}
    with pytest.raises(ValueError, match=r"incomplete design.*legacy metadata"):
        _merge(monkeypatch, [source], output)
    assert before == {path.name: path.read_bytes() for path in merged.iterdir()}


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("consumer", ["merge", "reader"])
def test_legacy_fallback_does_not_borrow_another_designs_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool, consumer: str
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    _pair(designs, "target_gen", "generated")
    _pair(designs, "target_metadata", "separate-design")
    if not complete:
        (designs / "target_gen.npz").unlink()
    output = tmp_path / "merged"
    if not complete:
        if consumer == "merge":
            with pytest.raises(ValueError, match="Ambiguous legacy metadata"):
                _merge(monkeypatch, [source], output)
        else:
            with pytest.raises(ValueError, match="Ambiguous legacy metadata"):
                _module(designs, tmp_path / "unused", monkeypatch, 1)
        assert not (output / designs.name / "run_target_gen.cif").exists()
    elif consumer == "merge":
        _merge(monkeypatch, [source], output)
        with np.load(output / designs.name / "run_target_gen.npz") as metadata:
            assert metadata["identity"].item() == "generated"
    else:
        module = _module(designs, tmp_path / "unused", monkeypatch, 1)
        assert module.predict_set.metadata_paths == [
            designs / "target_gen.npz",
            designs / "target_metadata.npz",
        ]


@pytest.mark.parametrize("consumer", ["merge", "reader"])
def test_absent_legacy_metadata_does_not_block_unfinished_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    for stem in ("target_gen", "target_metadata"):
        _pair(designs, stem, "unfinished")
        (designs / f"{stem}.npz").unlink()
    if consumer == "merge":
        output = tmp_path / "merged"
        _merge(monkeypatch, [source], output)
        for stem in ("target_gen", "target_metadata"):
            assert (output / designs.name / f"run_{stem}.cif").is_file()
            assert not (output / designs.name / f"run_{stem}.npz").exists()
    else:
        module = _module(designs, tmp_path / "unused", monkeypatch, 1)
        assert [path.name for path in module.predict_set.generated_paths] == [
            "target_gen.cif",
            "target_metadata.cif",
        ]
        assert not any(path.is_file() for path in module.predict_set.metadata_paths)


def test_merge_rejects_empty_metrics_instead_of_silently_omitting_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    _pair(designs, "candidate", "unscored")
    pd.DataFrame(columns=["id", "file_name"]).to_csv(
        designs / "aggregate_metrics_analyze.csv", index=False
    )
    with pytest.raises(ValueError, match="contains no analyzed designs"):
        _merge(monkeypatch, [source], tmp_path / "merged")


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_inverse_fold_resume_requires_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplicity: int
) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    _pair(inputs, "target", "input")
    outputs.mkdir()
    for index in range(multiplicity):
        stem = f"target_{index}" if multiplicity > 1 else "target"
        for suffix in (".cif", ".npz"):
            (outputs / f"{stem}{suffix}").mkdir()
    module = _module(inputs, outputs, monkeypatch, multiplicity)
    assert len(module.predict_set) == multiplicity


@pytest.mark.parametrize(
    ("multiplicity", "old_suffix"), [(3, "00"), (12, "0"), (101, "00")]
)
def test_inverse_fold_resume_rejects_complete_padding_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multiplicity: int,
    old_suffix: str,
) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    _pair(inputs, "target_0", "input")
    _pair(outputs, f"target_0_{old_suffix}", "existing")
    before = {path.name: path.read_bytes() for path in outputs.iterdir()}
    with pytest.raises(ValueError, match="different numeric padding"):
        _module(inputs, outputs, monkeypatch, multiplicity)
    assert before == {path.name: path.read_bytes() for path in outputs.iterdir()}


def test_merged_files_do_not_share_writable_source_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    _pair(designs, "candidate", "original")
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    (output / designs.name / "run_candidate.cif").write_text("rewritten prediction")
    assert (designs / "candidate.cif").read_text() == "original"


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


def test_inverse_fold_partial_resume_preserves_completed_sibling(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    writer = DesignWriter(
        str(outputs),
        res_atoms_only=False,
        atom14=False,
        inverse_fold=True,
        write_native=False,
    )
    datamodule = SimpleNamespace(
        cfg=SimpleNamespace(multiplicity=3),
        skip_existing=True,
        skip_existing_kind="inverse_fold",
    )
    trainer = SimpleNamespace(datamodule=datamodule)

    def write_sample(sample: int, shift: float) -> None:
        features = _features(padded=False, missing_atom=False, ligand=False)
        features["data_sample_idx"] = sample
        batch = data_from_generated.collate([features])
        batch["extra_mols"] = None
        prediction = {key: value for key, value in batch.items() if key != "extra_mols"}
        prediction["coords"] = batch["coords"][0] + shift
        prediction["exception"] = False
        writer.write_on_batch_end(trainer=trainer, prediction=prediction, batch=batch)

    write_sample(0, 0.0)
    write_sample(1, 0.0)
    completed = {
        suffix: (outputs / f"contract_0{suffix}").read_bytes()
        for suffix in (".cif", ".npz")
    }
    (outputs / "contract_1.npz").unlink()
    incomplete_cif = (outputs / "contract_1.cif").read_bytes()
    for index in range(3):
        write_sample(index, 10.0)
    assert writer.failed == 0
    for suffix, content in completed.items():
        assert (outputs / f"contract_0{suffix}").read_bytes() == content
    assert (outputs / "contract_1.cif").read_bytes() != incomplete_cif
    assert (outputs / "contract_1.npz").is_file()
    assert (outputs / "contract_2.cif").is_file()
    assert (outputs / "contract_2.npz").is_file()


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
