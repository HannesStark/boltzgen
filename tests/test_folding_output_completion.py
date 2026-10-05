"""Interrupted folding exports must be retried until both outputs exist."""
# ruff: noqa: INP001

import builtins
from pathlib import Path
from typing import Any, BinaryIO, Self, TextIO

import gemmi
import numpy as np
import pytest
import torch
from test_folding_export_consistency import (
    _features,
    _forward_output,
    _InferenceBoundary,
)

from boltzgen.data.data import Structure
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.models.boltz import Boltz
from boltzgen.task.predict import data_from_generated
from boltzgen.task.predict import writer as writer_module
from boltzgen.task.predict.data_from_generated import (
    DataConfig,
    FromGeneratedDataModule,
    collate,
)
from boltzgen.task.predict.writer import AffinityWriter, FoldingWriter


def _case(
    tmp_path: Path, designfolding: bool, *, external: bool = False
) -> tuple[Path, FoldingWriter, dict[str, Any], dict[str, Any]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    features = _features(padded=True, missing_atom=False, ligand=False)
    structure, _, _ = Structure.from_feat(features)
    (inputs / "contract.cif").write_text(to_mmcif(structure))
    np.savez(inputs / "contract.npz", design_mask=np.zeros(2, dtype=bool))
    batch = collate([features])
    output = _forward_output(batch, 2)
    output["design_to_target_iptm"][1] = 0.9
    output["design_ptm"][1] = 0.8
    prediction = Boltz.predict_step(_InferenceBoundary(output, 2, mask=True), batch)
    root = tmp_path / "separate_outputs" if external else inputs
    return inputs, FoldingWriter(str(root), designfolding), batch, prediction


def _remaining(
    inputs: Path,
    writer: FoldingWriter | AffinityWriter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    external: bool = False,
) -> int:
    # Only canonical-molecule I/O is replaced: dataset discovery and the
    # public reuse filter execute normally, without fetching any structures.
    monkeypatch.setattr(data_from_generated, "load_canonicals", lambda _: {})
    config = DataConfig(
        num_targets=10,
        samples_per_target=10,
        moldir=str(inputs),
        tokenizer=Tokenizer(),
        featurizer=Featurizer(),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    if isinstance(writer, AffinityWriter):
        primary = writer.outdir
        kind = "affinity"
    else:
        primary = writer.refold_cif_dir if writer.designfolding else writer.outdir
        kind = "design_folded" if writer.designfolding else "folded"
    data = FromGeneratedDataModule(
        cfg=config,
        design_dir=str(inputs),
        skip_existing=True,
        skip_existing_kind=kind,
        output_dir=str(primary) if external else None,
    )
    return len(data.predict_set)


@pytest.mark.parametrize("designfolding", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("present", ["neither", "npz", "cif", "both"])
def test_reuse_requires_matching_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    designfolding: bool,
    external: bool,
    present: str,
) -> None:
    inputs, writer, batch, prediction = _case(
        tmp_path, designfolding, external=external
    )
    writer.write_on_batch_end(prediction=prediction, batch=batch)
    if present not in ("npz", "both"):
        (writer.outdir / "contract.npz").unlink()
    if present not in ("cif", "both"):
        (writer.refold_cif_dir / "contract.cif").unlink()
    assert _remaining(inputs, writer, monkeypatch, external=external) == int(
        present != "both"
    )


@pytest.mark.parametrize("designfolding", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_preparation_failure_preserves_only_a_previous_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    designfolding: bool,
    existing: bool,
) -> None:
    inputs, writer, batch, prediction = _case(tmp_path, designfolding)
    paths = [writer.outdir / "contract.npz", writer.refold_cif_dir / "contract.cif"]
    if existing:
        writer.write_on_batch_end(prediction=prediction, batch=batch)
    previous = [path.read_bytes() if path.exists() else None for path in paths]
    prediction["coords"] = prediction["coords"] + 25

    def fail_conversion(_structure: Structure) -> str:
        raise RuntimeError("interrupted CIF conversion")

    with monkeypatch.context() as patch:
        patch.setattr(writer_module, "to_mmcif", fail_conversion)
        with pytest.raises(RuntimeError, match="interrupted CIF conversion"):
            writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert [path.read_bytes() if path.exists() else None for path in paths] == previous
    assert _remaining(inputs, writer, monkeypatch) == int(not existing)


@pytest.mark.parametrize("designfolding", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["cif_write", "npz_write", "npz_replace"])
def test_interrupted_publication_leaves_no_completion_marker_and_retries(  # noqa: C901
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    designfolding: bool,
    existing: bool,
    stage: str,
) -> None:
    inputs, writer, batch, prediction = _case(tmp_path, designfolding)
    archive = writer.outdir / "contract.npz"
    cif = writer.refold_cif_dir / "contract.cif"
    if existing:
        writer.write_on_batch_end(prediction=prediction, batch=batch)
    prediction["coords"] = prediction["coords"] + 25
    real_open = builtins.open

    class InterruptedCif:
        def __init__(self) -> None:
            self.file = real_open(cif, "w")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            self.file.close()

        def write(self, text: str) -> None:
            self.file.write(text[:20])
            self.file.close()
            raise OSError("interrupted CIF write")

    def interrupted_open(path: str | Path, mode: str = "r") -> TextIO | InterruptedCif:
        if Path(path) == cif and mode == "w":
            return InterruptedCif()
        return real_open(path, mode)

    def interrupted_npz(path: str | Path | BinaryIO, **_arrays: object) -> None:
        if isinstance(path, (str, Path)):
            Path(path).write_bytes(b"partial zip")
        else:
            path.write(b"partial zip")
        raise OSError("interrupted NPZ write")

    def interrupted_replace(_source: str | Path, destination: str | Path) -> None:
        assert Path(destination) == archive
        raise OSError("interrupted NPZ replace")

    with monkeypatch.context() as patch:
        if stage == "cif_write":
            patch.setattr(writer_module, "open", interrupted_open, raising=False)
        elif stage == "npz_write":
            patch.setattr(writer_module.np, "savez_compressed", interrupted_npz)
        else:
            patch.setattr(writer_module.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted"):
            writer.write_on_batch_end(prediction=prediction, batch=batch)

    assert not archive.exists()
    assert not list(writer.outdir.glob("*.tmp"))
    assert _remaining(inputs, writer, monkeypatch) == 1

    writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert _remaining(inputs, writer, monkeypatch) == 0
    with np.load(archive) as saved:
        np.testing.assert_array_equal(saved["coords"], prediction["coords"].numpy())
    block = gemmi.cif.read_file(str(cif)).sole_block()
    x = np.asarray(block.find_values("_atom_site.Cartn_x"), dtype=float)
    np.testing.assert_allclose(x, prediction["coords"][1, :8, 0].numpy())


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["write", "replace"])
def test_affinity_publication_preserves_valid_archive_or_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    stage: str,
) -> None:
    inputs, _, batch, _ = _case(tmp_path, designfolding=False)
    writer = AffinityWriter(str(inputs))
    archive = writer.outdir / "contract.npz"
    prediction = {"exception": False, "affinity_pred_value": torch.tensor([1.25])}
    if existing:
        writer.write_on_batch_end(prediction=prediction, batch=batch)
    previous = archive.read_bytes() if existing else None
    prediction["affinity_pred_value"] = torch.tensor([2.75])

    def interrupted_npz(path: str | Path | BinaryIO, **_arrays: object) -> None:
        if isinstance(path, (str, Path)):
            Path(path).write_bytes(b"partial zip")
        else:
            path.write(b"partial zip")
        raise OSError("interrupted affinity write")

    def interrupted_replace(_source: str | Path, destination: str | Path) -> None:
        assert Path(destination) == archive
        raise OSError("interrupted affinity replace")

    with monkeypatch.context() as patch:
        if stage == "write":
            patch.setattr(writer_module.np, "savez_compressed", interrupted_npz)
        else:
            patch.setattr(writer_module.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted affinity"):
            writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert (archive.read_bytes() if archive.exists() else None) == previous
    assert not list(writer.outdir.glob("*.tmp"))
    assert _remaining(inputs, writer, monkeypatch) == int(not existing)
    if existing:
        with np.load(archive) as saved:
            np.testing.assert_array_equal(saved["affinity_pred_value"], [1.25])

    writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert _remaining(inputs, writer, monkeypatch) == 0
    with np.load(archive) as saved:
        np.testing.assert_array_equal(saved["affinity_pred_value"], [2.75])
