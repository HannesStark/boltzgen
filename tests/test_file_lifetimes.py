"""Exercise explicit closure through real writer and dataset entry points."""

import builtins
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest
from test_folding_export_consistency import _features

from boltzgen.model.validation import design as validation
from boltzgen.task.predict import data_from_generated as data
from boltzgen.task.predict import writer


class RetainedFile:
    """Keep the file alive so refcount cleanup cannot hide a missing close."""

    def __init__(self, stream, *, fail_write=False, fail_read=False):
        self.stream = stream
        self.fail_write = fail_write
        self.fail_read = fail_read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        if self.fail_write:
            raise OSError("injected write failure")
        return self.stream.write(text)

    def readlines(self):
        if self.fail_read:
            raise OSError("injected read failure")
        return self.stream.readlines()

    def __iter__(self):
        if self.fail_read:
            raise OSError("injected read failure")
        return iter(self.stream)


@pytest.fixture
def retain(monkeypatch):
    handles = []

    def install(module, *, failed_suffix=None, fail_read=False):
        def open_file(path, mode="r", *args, **kwargs):
            handle = RetainedFile(
                builtins.open(path, mode, *args, **kwargs),
                fail_write=failed_suffix is not None
                and str(path).endswith(failed_suffix),
                fail_read=fail_read,
            )
            handles.append(handle)
            return handle

        monkeypatch.setattr(module, "open", open_file, raising=False)
        return handles

    yield install
    for handle in handles:
        handle.close()


def _batch():
    batch = data.collate([_features(padded=False, missing_atom=False, ligand=False)])
    batch["extra_mols"] = None
    return batch


def _write_design(directory):
    batch = _batch()
    prediction = {key: value for key, value in batch.items() if key != "extra_mols"}
    prediction["coords"] = batch["coords"][0]
    prediction["exception"] = False
    prediction["coords_traj"] = [prediction["coords"], prediction["coords"] + 1]
    prediction["x0_coords_traj"] = [prediction["coords"], prediction["coords"] + 2]
    callback = writer.DesignWriter(
        str(directory),
        res_atoms_only=False,
        atom14=True,
        save_traj=True,
        save_x0_traj=True,
        write_native=True,
    )
    callback.write_on_batch_end(prediction=prediction, batch=batch)
    return callback


def _write_validation(directory):
    batch = _batch()
    features = {key: value for key, value in batch.items() if key != "extra_mols"}
    validator = validation.DesignValidator(val_names=["synthetic"], atom14=False)
    return validator.design_val_step(
        SimpleNamespace(trainer=SimpleNamespace(global_rank=0)),
        batch,
        features,
        {"sample_atom_coords": batch["coords"][0]},
        n_samples=1,
        batch_idx=0,
        generated_dir=str(directory),
    )


@pytest.mark.parametrize("kind", ["design", "validation"])
def test_success_closes_files_and_preserves_output(tmp_path, retain, kind):
    run = _write_design if kind == "design" else _write_validation
    module = writer if kind == "design" else validation
    reference = tmp_path / "reference"
    run(reference)
    handles = retain(module)
    actual = tmp_path / "actual"
    run(actual)
    assert len(handles) == (4 if kind == "design" else 2)
    assert all(handle.closed for handle in handles)
    assert sorted(p.name for p in reference.iterdir()) == sorted(
        p.name for p in actual.iterdir()
    )
    for path in reference.iterdir():
        if path.is_dir():
            continue
        other = actual / path.name
        if path.suffix == ".npz":
            with np.load(path) as expected, np.load(other) as result:
                assert set(expected.files) == set(result.files)
                for key in expected.files:
                    np.testing.assert_array_equal(result[key], expected[key])
        else:
            assert other.read_bytes() == path.read_bytes()
            structure = gemmi.read_structure(str(other))
            assert len(structure) == (2 if path.suffix == ".pdb" else 1)


@pytest.mark.parametrize(
    "suffix", ["_native.cif", "contract.cif", "_traj.pdb", "_x0_traj.pdb"]
)
@pytest.mark.parametrize("failure", ["write", "serialization"])
def test_design_closes_files_on_failure(tmp_path, monkeypatch, retain, suffix, failure):
    handles = retain(writer, failed_suffix=suffix if failure == "write" else None)
    if failure == "serialization":
        if suffix.endswith(".cif"):
            real = writer.to_mmcif
            calls = 0

            def serialize(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == (1 if suffix == "_native.cif" else 2):
                    raise ValueError("injected serialization failure")
                return real(*args, **kwargs)

            monkeypatch.setattr(writer, "to_mmcif", serialize)
        else:
            real = writer.DesignWriter.combine_pdb_models
            calls = 0

            def serialize(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == (2 if suffix == "_x0_traj.pdb" else 1):
                    raise ValueError("injected serialization failure")
                return real(self, *args, **kwargs)

            monkeypatch.setattr(writer.DesignWriter, "combine_pdb_models", serialize)
    _write_design(tmp_path)
    assert handles and str(handles[-1].name).endswith(suffix)
    assert all(handle.closed for handle in handles)
    # Optional trajectory writing follows the existing completed design pair.
    assert (tmp_path / "contract.npz").exists() == suffix.endswith(".pdb")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failure", ["write", "serialization"])
def test_validation_closes_files_on_failure(
    tmp_path, monkeypatch, retain, native, failure
):
    suffix = "contract_native.cif" if native else "contract.cif"
    handles = retain(validation, failed_suffix=suffix if failure == "write" else None)
    if failure == "serialization":
        real = validation.to_mmcif
        calls = 0

        def serialize(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == (2 if native else 1):
                raise ValueError("injected serialization failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(validation, "to_mmcif", serialize)
    assert _write_validation(tmp_path) is False
    assert handles and str(handles[-1].name).endswith(suffix)
    assert all(handle.closed for handle in handles)
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.parametrize(
    "contents,expected",
    [
        (" target_a \n", ["target_a_0.cif"]),
        ("", []),
        ("\n", ["target_a_0.cif", "target_b_0.cif"]),
    ],
)
def test_subset_reader_closes_and_preserves_selection(
    tmp_path, monkeypatch, retain, contents, expected
):
    directory = tmp_path / "designs"
    directory.mkdir()
    for name in ("target_a_0.cif", "target_b_0.cif"):
        (directory / name).touch()
    subset = tmp_path / "subset.txt"
    subset.write_text(contents)
    monkeypatch.setattr(data, "load_canonicals", lambda _: {})
    handles = retain(data)
    cfg = data.DataConfig(
        num_targets=None,
        samples_per_target=100,
        moldir=str(tmp_path),
        tokenizer=None,
        featurizer=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    module = data.FromGeneratedDataModule(
        cfg, design_dir=str(directory), subset_target_ids=str(subset)
    )
    assert [path.name for path in module.predict_set.generated_paths] == expected
    assert len(handles) == 1 and handles[0].closed


def test_subset_reader_closes_on_read_failure(tmp_path, monkeypatch, retain):
    (tmp_path / "target_0.cif").touch()
    subset = tmp_path / "subset.txt"
    subset.write_text("target")
    monkeypatch.setattr(data, "load_canonicals", lambda _: {})
    handles = retain(data, fail_read=True)
    cfg = data.DataConfig(
        num_targets=None,
        samples_per_target=100,
        moldir=str(tmp_path),
        tokenizer=None,
        featurizer=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    with pytest.raises(OSError, match="injected read failure"):
        data.FromGeneratedDataModule(
            cfg, design_dir=str(tmp_path), subset_target_ids=str(subset)
        )
    assert len(handles) == 1 and handles[0].closed
