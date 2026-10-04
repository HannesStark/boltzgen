from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boltzgen.task.predict import data_from_yaml
from boltzgen.task.predict import writer as writer_module


def _datamodule(tmp_path, monkeypatch, *, yaml_paths, output_dir, multiplicity, samples):
    monkeypatch.setattr(data_from_yaml, "load_canonicals", lambda _: {})
    cfg = data_from_yaml.DataConfig(
        moldir=str(tmp_path),
        multiplicity=multiplicity,
        yaml_path=yaml_paths,
        tokenizer=None,
        featurizer=None,
        skip_existing=True,
        output_dir=str(output_dir),
        diffusion_samples=samples,
    )
    module = data_from_yaml.FromYamlDataModule(
        cfg, batch_size=1, num_workers=0, pin_memory=False
    )
    return cfg, module


def test_reuse_detects_unsuffixed_single_writer_output(tmp_path, monkeypatch):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "candidate.cif").write_text("completed design")
    (output_dir / "candidate.npz").write_bytes(b"completed metadata")

    cfg, module = _datamodule(
        tmp_path,
        monkeypatch,
        yaml_paths=str(tmp_path / "candidate.yaml"),
        output_dir=output_dir,
        multiplicity=1,
        samples=1,
    )

    assert cfg.skip_offset == 1
    assert len(module.predict_set) == 0


def test_reuse_does_not_combine_mismatched_padding_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    (output_dir / "candidate_0.cif").touch()
    (output_dir / "candidate_00.npz").touch()

    cfg, module = _datamodule(
        tmp_path,
        monkeypatch,
        yaml_paths=str(tmp_path / "candidate.yaml"),
        output_dir=output_dir,
        multiplicity=2,
        samples=1,
    )

    assert cfg.skip_offset == 0
    assert len(module.predict_set) == 2


def test_reuse_stops_before_incomplete_diffusion_batch(tmp_path, monkeypatch):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    yaml_paths = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]

    # DesignWriter uses global_idx = sample_idx * n_samples + n. The first
    # batch is complete for both inputs; the next batch has a hole in beta even
    # though a later beta sample exists.
    for stem in ("alpha", "beta"):
        for idx in range(3):
            (output_dir / f"{stem}_{idx}.cif").touch()
            (output_dir / f"{stem}_{idx}.npz").touch()
    for idx in (3, 4, 5):
        (output_dir / f"alpha_{idx}.cif").touch()
        (output_dir / f"alpha_{idx}.npz").touch()
    for idx in (3, 5):
        (output_dir / f"beta_{idx}.cif").touch()
        (output_dir / f"beta_{idx}.npz").touch()

    cfg, module = _datamodule(
        tmp_path,
        monkeypatch,
        yaml_paths=[str(path) for path in yaml_paths],
        output_dir=output_dir,
        multiplicity=2,
        samples=3,
    )

    assert cfg.skip_offset == 1
    assert len(module.predict_set) == len(yaml_paths)


def _writer_inputs():
    tokens = 1
    atoms = 2
    token_values = {
        "design_mask": torch.ones((1, tokens)),
        "binding_type": torch.zeros((1, tokens)),
        "mol_type": torch.zeros((1, tokens)),
        "ss_type": torch.zeros((1, tokens)),
        "token_pad_mask": torch.ones((1, tokens), dtype=torch.bool),
        "token_resolved_mask": torch.ones((1, tokens), dtype=torch.bool),
        "token_to_res": torch.arange(tokens).reshape(1, tokens),
    }
    atom_values = {
        "atom_to_token": torch.ones((1, atoms, tokens)),
        "atom_pad_mask": torch.ones((1, atoms), dtype=torch.bool),
    }
    prediction = {
        **token_values,
        **atom_values,
        "coords": torch.zeros((1, atoms, 3)),
        "chain_design_mask": torch.ones((1, tokens), dtype=torch.bool),
        "exception": False,
    }
    batch = {
        **token_values,
        **atom_values,
        "coords": torch.zeros((1, 1, atoms, 3)),
        "symmetric_group": torch.zeros((1, tokens)),
        "id": ["candidate"],
        "data_sample_idx": torch.tensor([0]),
        "extra_mols": None,
    }
    return prediction, batch


def _write_one_design(writer, monkeypatch, *, multiplicity=1, sample_index=0):
    class Structure:
        def __init__(self):
            self.atoms = {}
            self.residues = []

    monkeypatch.setattr(
        writer_module,
        "BoltzMasker",
        lambda **kwargs: lambda batch: {
            "ref_element": torch.zeros((1, 1)),
            "ref_atom_name_chars": torch.zeros((1, 1)),
        },
    )
    monkeypatch.setattr(
        writer_module.Structure,
        "from_feat",
        lambda _: (Structure(), None, None),
    )
    monkeypatch.setattr(
        writer_module,
        "to_mmcif",
        lambda *args, **kwargs: "new coordinates",
    )
    prediction, batch = _writer_inputs()
    batch["data_sample_idx"] = torch.tensor([sample_index])
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            cfg=SimpleNamespace(multiplicity=multiplicity, skip_existing=True)
        )
    )
    writer.write_on_batch_end(trainer=trainer, prediction=prediction, batch=batch)


def test_reuse_writer_preserves_complete_outputs(tmp_path, monkeypatch):
    writer = writer_module.DesignWriter(
        output_dir=str(tmp_path),
        res_atoms_only=False,
        atom14=False,
        atom37=False,
        write_native=False,
    )
    cif = tmp_path / "candidate.cif"
    metadata = tmp_path / "candidate.npz"
    cif.write_bytes(b"keep cif")
    metadata.write_bytes(b"keep metadata")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (cif, metadata)
    }

    _write_one_design(writer, monkeypatch)

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (cif, metadata)
    }
    assert after == before


@pytest.mark.parametrize("missing", ["cif", "metadata"])
def test_reuse_writer_repairs_incomplete_output_pair(
    tmp_path, monkeypatch, missing
):
    writer = writer_module.DesignWriter(
        output_dir=str(tmp_path),
        res_atoms_only=False,
        atom14=False,
        atom37=False,
        write_native=False,
    )
    cif = tmp_path / "candidate.cif"
    metadata = tmp_path / "candidate.npz"
    if missing != "cif":
        cif.write_bytes(b"old cif")
    if missing != "metadata":
        metadata.write_bytes(b"old metadata")

    _write_one_design(writer, monkeypatch)

    assert cif.read_text() == "new coordinates"
    assert metadata.is_file()
    assert metadata.stat().st_size > 0


def test_reuse_requires_each_yaml_input_before_skipping(tmp_path, monkeypatch):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    yaml_paths = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    (output_dir / "alpha.cif").touch()

    cfg, module = _datamodule(
        tmp_path,
        monkeypatch,
        yaml_paths=[str(path) for path in yaml_paths],
        output_dir=output_dir,
        multiplicity=1,
        samples=1,
    )

    assert cfg.skip_offset == 0
    assert len(module.predict_set) == len(yaml_paths)


@pytest.mark.parametrize(
    "multiplicity,existing_names,expected_offset",
    [
        (2, ["candidate_0", "candidate_1"], 2),
        (12, ["candidate_0", "candidate_1"], 2),
        (2, ["candidate"], 1),
    ],
)
def test_reuse_recognizes_ids_across_padding_and_multiplicity_growth(
    tmp_path, monkeypatch, multiplicity, existing_names, expected_offset
):
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    for name in existing_names:
        (output_dir / f"{name}.cif").touch()
        (output_dir / f"{name}.npz").touch()

    cfg, module = _datamodule(
        tmp_path,
        monkeypatch,
        yaml_paths=str(tmp_path / "candidate.yaml"),
        output_dir=output_dir,
        multiplicity=multiplicity,
        samples=1,
    )

    assert cfg.skip_offset == expected_offset
    assert len(module.predict_set) == multiplicity - expected_offset


@pytest.mark.parametrize(
    "existing_name,multiplicity", [("candidate", 2), ("candidate_0", 12)]
)
def test_replayed_sample_preserves_old_writer_id_alias(
    tmp_path, monkeypatch, existing_name, multiplicity
):
    writer = writer_module.DesignWriter(
        output_dir=str(tmp_path),
        res_atoms_only=False,
        atom14=False,
        atom37=False,
        write_native=False,
    )
    cif = tmp_path / f"{existing_name}.cif"
    metadata = tmp_path / f"{existing_name}.npz"
    cif.write_bytes(b"keep cif")
    metadata.write_bytes(b"keep metadata")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (cif, metadata)
    }

    _write_one_design(writer, monkeypatch, multiplicity=multiplicity)

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (cif, metadata)
    }
    assert after == before
    assert not (tmp_path / "candidate_00.cif").exists()


@pytest.mark.parametrize("missing", ["cif", "metadata"])
def test_repair_does_not_publish_partial_metadata_marker(
    tmp_path, monkeypatch, missing
):
    writer = writer_module.DesignWriter(
        output_dir=str(tmp_path),
        res_atoms_only=False,
        atom14=False,
        atom37=False,
        write_native=False,
    )
    cif = tmp_path / "candidate.cif"
    metadata = tmp_path / "candidate.npz"
    if missing != "cif":
        cif.write_bytes(b"old cif")
    if missing != "metadata":
        metadata.write_bytes(b"old metadata")

    def fail_metadata_write(file, **kwargs):
        file.write(b"partial metadata")
        raise OSError("simulated metadata write failure")

    monkeypatch.setattr(writer_module.np, "savez_compressed", fail_metadata_write)
    _write_one_design(writer, monkeypatch)

    assert not metadata.exists()
    assert list(tmp_path.glob(".candidate.npz.*.tmp")) == []
