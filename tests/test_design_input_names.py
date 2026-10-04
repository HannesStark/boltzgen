"""Input identities remain unambiguous before fresh generation or resume."""

from types import SimpleNamespace
import pickle

import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from boltzgen.cli.boltzgen import check_design_specs
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.task.predict import data_from_yaml


@pytest.fixture
def molecules(monkeypatch, tmp_path):
    mols = {}
    for letter, name in [("A", "ALA"), ("G", "GLY")]:
        mol = Chem.MolFromSequence(letter)
        for atom in mol.GetAtoms():
            atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        mols[name] = Chem.RemoveHs(mol)
    previous = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for name, mol in mols.items():
            (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(mol))
    finally:
        Chem.SetDefaultPickleProperties(previous)
    monkeypatch.setattr(data_from_yaml, "load_canonicals", lambda _: mols.copy())
    return mols


def write_spec(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"entities": [
        {"protein": {"id": "A", "sequence": "AAA"}},
        {"protein": {"id": "B", "sequence": "3"}},
    ]}))


def make_config(paths, output, tmp_path, reuse):
    return data_from_yaml.DataConfig(
        moldir=str(tmp_path), multiplicity=1, yaml_path=list(map(str, paths)),
        tokenizer=Tokenizer(), featurizer=Featurizer(), skip_existing=reuse,
        output_dir=str(output),
    )


@pytest.mark.parametrize("same_path", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_duplicate_stems_fail_before_cli_output_and_prediction(
    tmp_path, molecules, same_path, reuse
):
    first = tmp_path / "first" / "candidate.yaml"
    second = first if same_path else tmp_path / "second" / "candidate.yaml"
    write_spec(first)
    write_spec(second)
    paths = [first, second]
    output = tmp_path / "generated"
    output.mkdir()
    if reuse:
        (output / "candidate.cif").write_bytes(b"completed coordinates")
        (output / "candidate.npz").write_bytes(b"completed metadata")
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    args = SimpleNamespace(design_spec=paths, output=output)
    with pytest.raises(ValueError, match="unique stems; repeated: candidate"):
        check_design_specs(args, tmp_path, molecules)
    config = make_config(paths, output, tmp_path, reuse)
    with pytest.raises(ValueError, match="unique stems; repeated: candidate"):
        data_from_yaml.FromYamlDataModule(config, batch_size=1, num_workers=0, pin_memory=False)
    dataset = data_from_yaml.Dataset(
        yaml_path=config.yaml_path, multiplicity=1,
        tokenizer=config.tokenizer, featurizer=config.featurizer,
    )
    with pytest.raises(ValueError, match="unique stems; repeated: candidate"):
        data_from_yaml.PredictionDataset(dataset, molecules, tmp_path)
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert after == before


@pytest.mark.parametrize("reuse", [False, True])
def test_distinct_stems_pass_actual_cli_checks_and_dataset(tmp_path, molecules, reuse):
    paths = [tmp_path / "first" / "alpha.yaml", tmp_path / "second" / "beta.yaml"]
    for path in paths:
        write_spec(path)
    output = tmp_path / "generated"
    output.mkdir()
    check_design_specs(SimpleNamespace(design_spec=paths, output=output), tmp_path, molecules)
    assert {path.name for path in output.iterdir()} == {"alpha.cif", "beta.cif"}
    config = make_config(paths, output, tmp_path, reuse)
    module = data_from_yaml.FromYamlDataModule(config, batch_size=1, num_workers=0, pin_memory=False)
    assert config.skip_offset == 0
    assert len(module.predict_set) == 2
    assert [module.predict_set[index]["id"] for index in range(2)] == ["alpha", "beta"]
