"""Insertion and fusion preserve structure bookkeeping across NumPy versions."""

# Numeric expectations describe the fixtures, including their absolute offsets.
# ruff: noqa: CPY001, INP001, PLR2004

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from boltzgen.data import const
from boltzgen.data.data import (
    Atom,
    Bond,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    Structure,
    Target,
)
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.write.mmcif import to_mmcif

pytestmark = pytest.mark.filterwarnings(
    "error:Conversion of an array with ndim > 0 to a scalar is deprecated"
    ":DeprecationWarning"
)


def make_structure(chain_count: int = 3) -> Structure:
    """Build distinct coordinates and explicit intra/interchain bonds."""
    residue_count = chain_count * 3
    atom_count = residue_count * 4
    atoms = np.zeros(atom_count, dtype=Atom)
    atoms["name"] = np.tile(const.ref_atoms["GLY"], residue_count)
    atoms["coords"] = np.arange(atom_count * 3).reshape(-1, 3) + 1
    atoms["is_present"] = True
    atoms["bfactor"] = 20
    coords = np.zeros(atom_count, dtype=Coords)
    coords["coords"] = atoms["coords"]
    residues = np.zeros(residue_count, dtype=Residue)
    residues["name"] = "GLY"
    residues["res_type"] = const.token_ids["GLY"]
    residues["res_idx"] = np.tile(np.arange(3), chain_count)
    residues["atom_idx"] = np.arange(residue_count) * 4
    residues["atom_num"] = 4
    residues["atom_center"] = residues["atom_idx"] + 1
    residues["atom_disto"] = residues["atom_center"]
    residues["is_standard"] = True
    residues["is_present"] = True
    chains = np.zeros(chain_count, dtype=Chain)
    chains["name"] = [chr(65 + i) for i in range(chain_count)]
    chains["entity_id"] = np.arange(chain_count) % 2
    chains["asym_id"] = np.arange(chain_count)
    chains["atom_idx"] = np.arange(chain_count) * 12
    chains["atom_num"] = 12
    chains["res_idx"] = np.arange(chain_count) * 3
    chains["res_num"] = 3
    chains["cyclic_period"] = -1
    bonds = np.array(
        [(0, 0, 0, 2, 1, 9, 1), (0, 1, 2, 3, 9, 13, 1), (2, 2, 7, 8, 29, 33, 1)]
        if chain_count == 3
        else [],
        dtype=Bond,
    )
    return Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=np.array([(0, 2)] if chain_count == 3 else [], dtype=Interface),
        mask=np.ones(chain_count, dtype=bool),
        coords=coords,
        ensemble=np.array([(0, atom_count)], dtype=Ensemble),
    )


def assert_consistent(structure: Structure) -> None:
    """Check absolute offsets, representative atoms and bond endpoints."""
    residue_offset = atom_offset = 0
    for chain in structure.chains:
        assert chain["res_idx"] == residue_offset
        assert chain["atom_idx"] == atom_offset
        residues = structure.residues[
            residue_offset : residue_offset + chain["res_num"]
        ]
        assert residues["atom_num"].sum() == chain["atom_num"]
        for residue in residues:
            assert residue["atom_idx"] == atom_offset
            assert residue["atom_center"] == atom_offset + 1
            assert residue["atom_disto"] == atom_offset + 1
            atom_offset += residue["atom_num"]
        residue_offset += chain["res_num"]
    assert residue_offset == len(structure.residues)
    assert atom_offset == len(structure.atoms)
    np.testing.assert_array_equal(
        structure.atoms["coords"], structure.coords["coords"][:atom_offset]
    )
    assert len(structure.coords) == atom_offset * max(1, len(structure.ensemble))
    for model, ensemble in enumerate(structure.ensemble):
        assert ensemble["atom_coord_idx"] == model * atom_offset
        assert ensemble["atom_num"] == atom_offset
    for bond in structure.bonds:
        for end in [1, 2]:
            residue = structure.residues[bond[f"res_{end}"]]
            chain = structure.chains[bond[f"chain_{end}"]]
            assert (
                residue["atom_idx"]
                <= bond[f"atom_{end}"]
                < residue["atom_idx"] + residue["atom_num"]
            )
            assert (
                chain["res_idx"]
                <= bond[f"res_{end}"]
                < chain["res_idx"] + chain["res_num"]
            )


def assert_unmodified(original: Structure, snapshot: Structure) -> None:
    """Public operations must leave every input table unchanged."""
    for field in fields(Structure):
        np.testing.assert_array_equal(
            getattr(original, field.name), getattr(snapshot, field.name)
        )


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
@pytest.mark.parametrize("position", [0, 1, 3])
@pytest.mark.parametrize("count", [0, 2])
def test_insert_preserves_all_tables(chain_idx: int, position: int, count: int) -> None:
    structure = make_structure()
    snapshot = deepcopy(structure)
    result = Structure.insert(structure, chr(65 + chain_idx), position, count)
    assert_consistent(result)
    assert_unmodified(structure, snapshot)
    absolute_residue = chain_idx * 3 + position
    absolute_atom = absolute_residue * 4
    np.testing.assert_array_equal(
        result.coords["coords"],
        np.concatenate(
            [
                structure.coords["coords"][:absolute_atom],
                np.zeros((count * 4, 3)),
                structure.coords["coords"][absolute_atom:],
            ]
        ),
    )
    for idx, chain in enumerate(result.chains):
        count_expected = 3 + (count if idx == chain_idx else 0)
        residues = result.residues[
            chain["res_idx"] : chain["res_idx"] + chain["res_num"]
        ]
        np.testing.assert_array_equal(residues["res_idx"], np.arange(count_expected))
    expected_bonds = structure.bonds.copy()
    for end in [1, 2]:
        expected_bonds[f"res_{end}"] += count * (
            structure.bonds[f"res_{end}"] >= absolute_residue
        )
        expected_bonds[f"atom_{end}"] += (
            count * 4 * (structure.bonds[f"atom_{end}"] >= absolute_atom)
        )
    np.testing.assert_array_equal(result.bonds, expected_bonds)
    for name in ["name", "entity_id", "asym_id", "cyclic_period"]:
        np.testing.assert_array_equal(result.chains[name], structure.chains[name])
    np.testing.assert_array_equal(result.mask, structure.mask)
    np.testing.assert_array_equal(result.interfaces, structure.interfaces)


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
@pytest.mark.parametrize("reindex", [False, True])
def test_fuse_preserves_indices_coordinates_and_bonds(
    chain_idx: int, *, reindex: bool
) -> None:
    structure = make_structure()
    donor = make_structure(chain_count=1)
    donor.residues["res_idx"] = [5, 7, 8]
    snapshots = deepcopy((structure, donor))
    result = Structure.fuse(structure, donor, chr(65 + chain_idx), res_reindex=reindex)
    assert_consistent(result)
    assert_unmodified(structure, snapshots[0])
    assert_unmodified(donor, snapshots[1])
    absolute_residue = (chain_idx + 1) * 3
    absolute_atom = absolute_residue * 4
    np.testing.assert_array_equal(
        result.coords["coords"],
        np.concatenate(
            [
                structure.coords["coords"][:absolute_atom],
                donor.coords["coords"],
                structure.coords["coords"][absolute_atom:],
            ]
        ),
    )
    np.testing.assert_array_equal(
        result.residues["res_idx"][absolute_residue : absolute_residue + 3],
        [3, 5, 6] if reindex else [5, 7, 8],
    )
    expected_entities = [[0, 2, 1], [0, 1, 0], [1, 2, 0]][chain_idx]
    np.testing.assert_array_equal(result.chains["entity_id"], expected_entities)
    expected_bonds = structure.bonds.copy()
    for end in [1, 2]:
        expected_bonds[f"res_{end}"] += 3 * (
            structure.bonds[f"res_{end}"] >= absolute_residue
        )
        expected_bonds[f"atom_{end}"] += 12 * (
            structure.bonds[f"atom_{end}"] >= absolute_atom
        )
    np.testing.assert_array_equal(result.bonds, expected_bonds)
    np.testing.assert_array_equal(result.mask, structure.mask)
    np.testing.assert_array_equal(result.interfaces, structure.interfaces)


@pytest.mark.parametrize("operation", ["insert", "fuse"])
@pytest.mark.parametrize("names", [["A", "B", "C"], ["B", "B", "C"]])
def test_chain_name_must_match_exactly_once(operation: str, names: list[str]) -> None:
    structure = make_structure()
    structure.chains["name"] = names
    chain_name = "missing" if names[0] == "A" else "B"
    if operation == "insert":
        with pytest.raises(ValueError, match="expected exactly one"):
            Structure.insert(structure, chain_name, 1, 1)
    else:
        with pytest.raises(ValueError, match="expected exactly one"):
            Structure.fuse(structure, make_structure(1), chain_name)


@pytest.mark.parametrize(
    "operation", ["control", "sequence_range", "insert", "fuse_protein", "fuse_file"]
)
def test_real_yaml_parser(operation: str, tmp_path: Path) -> None:
    mol = Chem.MolFromSequence("G")
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    source = tmp_path / "source.cif"
    source.write_text(to_mmcif(make_structure(1)))
    file = {"path": "source.cif"}
    entities = [{"file": file}]
    if operation == "insert":
        file["design_insertions"] = [
            {"insertion": {"id": "A", "res_index": 2, "num_residues": "2..2"}}
        ]
    elif operation in {"sequence_range", "fuse_protein"}:
        protein = {"id": "B", "sequence": "2..2"}
        if operation == "fuse_protein":
            protein["fuse"] = "A"
        entities.append({"protein": protein})
    elif operation == "fuse_file":
        entities.append({"file": {"path": "source.cif", "fuse": "A"}})
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"entities": entities}))
    parsed = YamlDesignParser(tmp_path).parse_yaml(path, {"GLY": mol}, tmp_path)
    expected = {
        "control": 3,
        "sequence_range": 5,
        "insert": 5,
        "fuse_protein": 5,
        "fuse_file": 6,
    }[operation]
    assert len(parsed.structure.residues) == expected
    assert_consistent(parsed.structure)
    assert len(parsed.structure.chains) == (2 if operation == "sequence_range" else 1)
    np.testing.assert_array_equal(
        parsed.structure.chains["res_num"],
        [3, 2] if operation == "sequence_range" else [expected],
    )


def with_ensembles(structure: Structure, count: int) -> Structure:
    """Give each conformer distinguishable coordinates, keeping model zero."""
    atom_count = len(structure.atoms)
    coords = np.concatenate([structure.coords] * max(1, count))
    for model in range(count):
        coords["coords"][model * atom_count : (model + 1) * atom_count] += model * 1000
    return replace(
        structure,
        coords=coords,
        ensemble=np.array(
            [(i * atom_count, atom_count) for i in range(count)], dtype=Ensemble
        ),
    )


@pytest.mark.parametrize("ensembles", [0, 1, 2])
def test_insert_updates_every_coordinate_ensemble(ensembles: int) -> None:
    structure = with_ensembles(make_structure(), ensembles)
    result = Structure.insert(structure, "B", 1, 2)
    assert_consistent(result)
    for model in range(max(1, ensembles)):
        expected = np.concatenate(
            [
                structure.coords[model * 36 : model * 36 + 16],
                np.zeros(8, dtype=Coords),
                structure.coords[model * 36 + 16 : (model + 1) * 36],
            ]
        )
        np.testing.assert_array_equal(
            result.coords[model * 44 : (model + 1) * 44], expected
        )


@pytest.mark.parametrize(
    ("target_models", "donor_models"), [(0, 0), (1, 1), (2, 1), (2, 2)]
)
def test_fuse_coordinates_and_donor_bonds(
    target_models: int, donor_models: int
) -> None:
    target = with_ensembles(make_structure(), target_models)
    donor = with_ensembles(make_structure(1), donor_models)
    donor = replace(donor, bonds=np.array([(0, 0, 0, 2, 1, 9, 1)], dtype=Bond))
    donor_snapshot = deepcopy(donor)
    result = Structure.fuse(target, donor, "B")
    assert_consistent(result)
    assert_unmodified(donor, donor_snapshot)
    np.testing.assert_array_equal(
        result.bonds[-1:], np.array([(1, 1, 6, 8, 25, 33, 1)], dtype=Bond)
    )
    for model in range(max(1, target_models)):
        donor_model = model if donor_models > 1 else 0
        expected = np.concatenate(
            [
                target.coords[model * 36 : model * 36 + 24],
                donor.coords[donor_model * 12 : (donor_model + 1) * 12],
                target.coords[model * 36 + 24 : (model + 1) * 36],
            ]
        )
        np.testing.assert_array_equal(
            result.coords[model * 48 : (model + 1) * 48], expected
        )


def test_fuse_rejects_unmatched_coordinate_ensembles() -> None:
    target = make_structure()
    donor = with_ensembles(make_structure(1), 2)
    with pytest.raises(ValueError, match="one conformer or the same number"):
        Structure.fuse(target, donor, "A")


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
def test_fuse_moves_cyclic_bond_to_new_terminal_atom(chain_idx: int) -> None:
    target = make_structure()
    target.chains["cyclic_period"][chain_idx] = 3
    start = chain_idx * 3
    target = replace(
        target,
        bonds=np.array(
            [
                (
                    chain_idx,
                    chain_idx,
                    start,
                    start + 2,
                    start * 4,
                    (start + 2) * 4 + 2,
                    1,
                )
            ],
            dtype=Bond,
        ),
    )
    result = Structure.fuse(target, make_structure(1), chr(65 + chain_idx))
    assert_consistent(result)
    assert result.chains["cyclic_period"][chain_idx] == 6
    bond = result.bonds[0]
    assert bond["res_2"] == start + 5
    assert bond["atom_2"] == (start + 5) * 4 + 2
    assert result.atoms["name"][bond["atom_2"]] == "C"


@pytest.mark.parametrize("targets", [("A",), ("B",), ("C",), ("A", "A"), ("A", "B")])
@pytest.mark.parametrize("donor_kind", ["protein", "file"])
def test_yaml_fusion_keeps_metadata_with_its_residues(
    targets: tuple[str, ...], donor_kind: str, tmp_path: Path
) -> None:
    """Fusing before another chain must move all conditioning with the donor."""
    mol = Chem.MolFromSequence("G")
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    mols = {"GLY": mol}
    parser = YamlDesignParser(tmp_path)
    (tmp_path / "base.cif").write_text(to_mmcif(make_structure()))
    entities = [{"file": {"path": "base.cif"}}]

    def parse(items: list[dict]) -> Target:
        return parser.parse_boltzgen_schema(
            "metadata", {"entities": items}, mols, tmp_path, tmp_path
        )

    current = parse(entities)
    names = (
        "res_design_mask",
        "res_structure_groups",
        "res_binding_type",
        "res_ss_types",
        "res_aa_constraint_mask",
    )
    for index, target in enumerate(targets):
        donor_id = chr(ord("D") + index)
        if donor_kind == "protein":
            donor = {
                "id": donor_id,
                "sequence": "G2..2",
                "binding_types": "buu",
                "secondary_structure": "uhs",
                "residue_constraints": [{"position": 2, "allowed": "G"}],
            }
        else:
            structure = make_structure(1)
            structure.chains["name"] = donor_id
            path = tmp_path / f"{donor_id}.cif"
            path.write_text(to_mmcif(structure))
            donor = {
                "path": str(path),
                "design": [{"chain": {"id": donor_id, "res_index": "2..3"}}],
                "structure_groups": [{"group": {"id": donor_id, "visibility": 2}}],
                "binding_types": [{"chain": {"id": donor_id, "binding": "1"}}],
                "secondary_structure": [
                    {"chain": {"id": donor_id, "helix": "2", "sheet": "3"}}
                ],
            }
        standalone = parse([{donor_kind: donor}])
        chain = current.structure.chains[current.structure.chains["name"] == target][0]
        boundary = int(chain["res_idx"] + chain["res_num"])
        expected = {
            name: np.concatenate(
                [
                    getattr(current.design_info, name)[:boundary],
                    getattr(standalone.design_info, name),
                    getattr(current.design_info, name)[boundary:],
                ]
            )
            for name in names
        }
        entities.append({donor_kind: {**donor, "fuse": target}})
        current = parse(entities)
        for name in names:
            np.testing.assert_array_equal(
                getattr(current.design_info, name), expected[name], err_msg=name
            )
