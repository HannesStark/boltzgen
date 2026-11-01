"""Independent synthetic public parser probes for the input-validation bundle."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from boltzgen.data import const
from boltzgen.data.data import Structure
from boltzgen.data.parse.schema import (
    YamlDesignParser,
    parse_range,
    parse_redesign_yaml,
    parse_residue_constraints,
)
from boltzgen.data.write.mmcif import to_mmcif


def canonicals():
    molecule = Chem.MolFromSequence("G")
    for atom in molecule.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    return {"GLY": molecule}


@pytest.mark.parametrize(
    "text,start,end,expected",
    [
        ("1", 0, 3, [0]),
        ("3", 7, 10, [9]),
        ("1..3", 7, 10, [7, 8, 9]),
        ("..2", 7, 10, [7, 8]),
        ("2..", 7, 10, [8, 9]),
        ("1,3", 7, 10, [7, 9]),
        ("1,1", 7, 10, [7, 7]),
        ("3..3", 0, 3, [2]),
    ],
)
def test_range_valid_forms_have_independent_expected_indices(
    text, start, end, expected
):
    assert parse_range(text, start, end) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1ab3",
        "1--3",
        "xx3",
        "1xx",
        "1.3",
        "1...3",
        "",
        "1,",
        "0",
        "0..2",
        "4",
        "1..4",
        "..4",
    ],
)
def test_malformed_or_out_of_bounds_ranges_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_range(text, 7, 10)


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_top_level_and_nested_yaml_preserve_real_structure(tmp_path, suffix):
    molecules = canonicals()
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "structure.cif").write_text(to_mmcif(Structure.empty_protein(2)))
    (nested / f"file{suffix}").write_text(yaml.safe_dump({"path": "structure.cif"}))
    path = root / f"design{suffix}"
    path.write_text(
        yaml.safe_dump({"entities": [{"file": {"path": f"nested/file{suffix}"}}]})
    )
    target = YamlDesignParser(tmp_path).parse_yaml(path, molecules, tmp_path)
    assert target.structure.residues["name"].tolist() == ["GLY", "GLY"]
    assert len(target.structure.chains) == 1


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_redesign_override_yaml_preserves_mask_and_checks_final_bound(tmp_path, suffix):
    tokenized = SimpleNamespace(
        tokens=np.array(
            [(0, 0, [0, 0, 0]), (1, 0, [10, 0, 0])],
            dtype=[
                ("token_idx", "i4"),
                ("asym_id", "i4"),
                ("center_coords", "f4", (3,)),
            ],
        ),
        structure=SimpleNamespace(
            chains=np.array(
                [("A", 0, 2)],
                dtype=[("name", "U1"), ("res_idx", "i4"), ("res_num", "i4")],
            )
        ),
        token_to_res=np.array([0, 1]),
    )
    path = tmp_path / f"override{suffix}"
    content = {
        "restrictions": {
            "design": [
                {
                    "chain": {
                        "binder": "A",
                        "id": "A",
                        "res_index": "1",
                        "within_proximity": 1,
                    }
                }
            ]
        }
    }
    path.write_text(yaml.safe_dump(content))
    np.testing.assert_array_equal(parse_redesign_yaml(path, tokenized), [True, False])
    content["restrictions"]["design"][0]["chain"]["res_index"] = "3"
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="higher than the length"):
        parse_redesign_yaml(path, tokenized)


@pytest.mark.parametrize("smiles", ["C1", "not-a-smiles"])
def test_invalid_ligand_smiles_fail_before_hydrogen_or_structure_work(tmp_path, smiles):
    parser = YamlDesignParser(tmp_path)
    with pytest.raises(ValueError, match="Failed to parse SMILES"):
        parser.parse_boltzgen_schema(
            "bad",
            {"entities": [{"ligand": {"id": "L", "smiles": smiles}}]},
            {},
            tmp_path,
            tmp_path,
        )
    with pytest.raises(ValueError, match="Failed to parse SMILES"):
        parser.parse_file(
            {
                "file": {
                    "path": "absent.cif",
                    "include": [{"chain": {"id": "L", "smiles": smiles}}],
                }
            },
            {},
            tmp_path,
            0,
            tmp_path,
        )


def test_valid_generic_smiles_still_reaches_real_conformer_and_structure(tmp_path):
    target = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "valid",
        {"entities": [{"ligand": {"id": "L", "smiles": "CCO"}}]},
        {},
        tmp_path,
        tmp_path,
    )
    assert len(target.extra_mols) == 1
    molecule = next(iter(target.extra_mols.values()))
    assert molecule.GetNumAtoms() == 3
    assert molecule.GetNumConformers() == 1
    assert len(target.structure.residues) == 1
    assert len(target.structure.atoms) == 3


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_empty_yaml_is_rejected_without_silent_empty_target(tmp_path, suffix):
    path = tmp_path / f"empty{suffix}"
    path.write_text("")
    # Existing .yaml behavior is TypeError for an empty document; .yml matches it.
    with pytest.raises(TypeError):
        YamlDesignParser(tmp_path).parse_yaml(path, {}, tmp_path)


def _gly_mols():
    mol = Chem.MolFromSequence("G")
    assert mol is not None
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    return {"GLY": mol}


def _write_one_gly_pdb(path: Path) -> None:
    """Write one real, fixed-column GLY PDB record set for parser integration."""
    lines = [
        "HEADER    LOW-RISK INPUT REVIEW",
        "COMPND    MOL_ID: 1;",
        "COMPND   2 MOLECULE: GLYCINE;",
        "COMPND   3 CHAIN: A;",
        "SEQRES   1 A    1  GLY",
    ]
    for serial, atom_name in enumerate(const.ref_atoms["GLY"], start=1):
        x = serial * 1.2
        atom_field = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name
        element = atom_name[0]
        lines.append(
            f"ATOM  {serial:5d} {atom_field} GLY A{1:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
            f"          {element:>2}"
        )
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n")


def test_valid_smiles_in_file_include_still_parses(tmp_path):
    _write_one_gly_pdb(tmp_path / "reference.pdb")
    path = tmp_path / "include.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "entities": [
                    {
                        "file": {
                            "path": "reference.pdb",
                            "include": [{"chain": {"id": "A", "smiles": "CCO"}}],
                        }
                    }
                ]
            }
        )
    )
    target = YamlDesignParser(tmp_path).parse_yaml(path, _gly_mols(), tmp_path)
    assert len(target.structure.chains) == 1
    assert target.structure.chains["name"].tolist() == ["A"]
    assert len(target.structure.residues) == 1


@pytest.mark.parametrize("res_index", ["1", "2"])
def test_file_include_single_residue_index_bounds(res_index, tmp_path):
    _write_one_gly_pdb(tmp_path / "reference.pdb")
    path = tmp_path / f"include_{res_index}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "entities": [
                    {
                        "file": {
                            "path": "reference.pdb",
                            "include": [{"chain": {"id": "A", "res_index": res_index}}],
                        }
                    }
                ]
            }
        )
    )
    if res_index == "2":
        with pytest.raises(ValueError, match="higher than the length"):
            YamlDesignParser(tmp_path).parse_yaml(path, _gly_mols(), tmp_path)
    else:
        target = YamlDesignParser(tmp_path).parse_yaml(path, _gly_mols(), tmp_path)
        assert len(target.structure.residues) == 1


def test_open_ended_constraint_must_not_silently_ignore_start_beyond_chain():
    with pytest.raises(ValueError):
        parse_residue_constraints(
            [{"position": "4..", "allowed": "A"}],
            3,
            const.canonical_tokens,
            const.prot_letter_to_token,
        )


@pytest.mark.parametrize("selection", ["1", "1.."])
def test_nonempty_selection_rejects_empty_chain(selection):
    with pytest.raises(ValueError, match="out of bounds"):
        parse_range(selection, c_start=10, c_end=10)


@pytest.mark.parametrize("selection", ["..0", "1..0"])
def test_residue_constraints_reject_zero_range_endpoint(selection):
    with pytest.raises(ValueError, match="1 indexed"):
        parse_residue_constraints(
            [{"position": selection, "allowed": "A"}],
            3,
            const.canonical_tokens,
            const.prot_letter_to_token,
        )
