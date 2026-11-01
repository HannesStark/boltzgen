"""Parser-backed regression tests for consecutive backbone-CA filtering."""

from dataclasses import replace

import gemmi
import numpy as np
import pytest
from rdkit import Chem

from boltzgen.data import const
from boltzgen.data.data import Structure
from boltzgen.data.filter.static.polymer import ConsecutiveCA
from boltzgen.data.parse.mmcif import mmcif_from_block


def canonical(letter):
    mol = Chem.MolFromSequence(letter)
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    return mol


def phosphoserine(*, ca_first=False):
    # Chemically explicit generic phosphoserine reference; no CCD download.
    mol = Chem.MolFromSmiles(
        "[NH2:1][C@@H:2]([CH2:3][O:4][P:5](=[O:6])([OH:7])[OH:8])[C:9](=[O:10])[OH:11]"
    )
    names = {
        1: "N",
        2: "CA",
        3: "CB",
        4: "OG",
        5: "P",
        6: "O1P",
        7: "O2P",
        8: "O3P",
        9: "C",
        10: "O",
        11: "OXT",
    }
    for atom in mol.GetAtoms():
        atom.SetProp("name", names[atom.GetAtomMapNum()])
        atom.SetAtomMapNum(0)
    if ca_first:
        index = next(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp("name") == "CA"
        )
        mol = Chem.RenumberAtoms(
            mol, [index] + [i for i in range(mol.GetNumAtoms()) if i != index]
        )
    return mol


def parse_chain(
    names,
    ca_positions,
    *,
    ca_first=False,
    missing_atoms=(),
    absent_residues=(),
    atom_offsets=None,
):
    mols = {
        "ALA": canonical("A"),
        "GLY": canonical("G"),
        "SER": canonical("S"),
        "MET": canonical("M"),
        "SEP": phosphoserine(ca_first=ca_first),
    }
    molecule = Chem.MolFromSmiles("CC")
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("name", f"C{index + 1}")
    mols["XXX"] = molecule
    calcium = Chem.MolFromSmiles("[Ca+2]")
    calcium.GetAtomWithIdx(0).SetProp("name", "CA")
    mols["CA"] = calcium
    st = gemmi.Structure()
    st.name = "synthetic"
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for index, (name, ca) in enumerate(zip(names, ca_positions)):
        if index in absent_residues:
            continue
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(index + 1, " ")
        residue.label_seq = index + 1
        residue.subchain = "A"
        residue.entity_id = "1"
        residue.entity_type = gemmi.EntityType.Polymer
        residue.het_flag = "A"
        for ref in mols["MET" if name == "MSE" else name].GetAtoms():
            atom_name = ref.GetProp("name")
            if (index, atom_name) in missing_atoms:
                continue
            atom = gemmi.Atom()
            atom.name = "SE" if name == "MSE" and atom_name == "SD" else atom_name
            atom.element = gemmi.Element("Se" if atom.name == "SE" else ref.GetSymbol())
            offset = {"N": -1.4, "CA": 0.0, "C": 1.5, "O": 2.2}.get(atom_name, 1.0)
            if atom_offsets is not None:
                offset = atom_offsets.get((index, atom_name), offset)
            atom.pos = gemmi.Position(float(ca) + offset, 0, 0)
            atom.occ = 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.full_sequence = names
    entity.subchains = ["A"]
    st.entities.append(entity)
    block = st.make_mmcif_block()
    # Deliberately serialize and parse the CIF before exercising its true parser.
    block = gemmi.cif.read_string(block.as_string()).sole_block()
    return mmcif_from_block(block, mols=mols, moldir=None, use_assembly=False).data


@pytest.mark.parametrize("ca_first", [False, True])
@pytest.mark.parametrize("names", [["ALA", "SEP"], ["SEP", "ALA"], ["SEP", "SEP"]])
@pytest.mark.parametrize(("distance", "keep"), [(3.8, True), (20.0, False)])
def test_modified_backbone_uses_actual_ca(names, distance, keep, ca_first):
    structure = parse_chain(names, [0, distance], ca_first=ca_first)
    assert not structure.residues["is_standard"].all()
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [keep])


def test_placeholder_center_does_not_create_false_break():
    structure = parse_chain(["ALA", "SEP"], [0, 9.5], atom_offsets={(1, "N"): 1.4})
    modified = structure.residues[1]
    assert structure.atoms[modified["atom_center"]]["name"] == "N"
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])


@pytest.mark.parametrize("names", [["ALA", "SER", "GLY"], ["ALA", "MSE", "ALA"]])
@pytest.mark.parametrize(
    ("positions", "keep"),
    [([0, 3.8, 7.6], True), ([0, 20, 23.8], False), ([0, 10, 20], True)],
)
def test_standard_and_mse_backbones(names, positions, keep):
    structure = parse_chain(names, positions)
    assert structure.residues["is_standard"].all()
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [keep])


@pytest.mark.parametrize("hetero", ["XXX", "CA"])
def test_hetero_groups_do_not_supply_a_ca_or_bridge_neighbors(hetero):
    structure = parse_chain(["ALA", hetero, "ALA"], [0, 100, 20])
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])


@pytest.mark.parametrize("residue", ["ALA", "SEP"])
@pytest.mark.parametrize("missing_residue", [False, True])
def test_missing_ca_or_residue_interrupts_adjacency(residue, missing_residue):
    structure = parse_chain(
        ["ALA", residue, "ALA"],
        [0, 20, 40],
        absent_residues=[1] if missing_residue else [],
        missing_atoms=[] if missing_residue else [(1, "CA")],
    )
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])


@pytest.mark.parametrize("missing", ["N", "C", "O"])
def test_reference_bonds_identify_ca_without_other_backbone_coordinates(missing):
    structure = parse_chain(["ALA", "SEP"], [0, 20], missing_atoms=[(1, missing)])
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [False])


@pytest.mark.parametrize("bond_type", ["SINGLE", "DOUBLE"])
def test_named_atoms_without_backbone_connectivity_do_not_qualify(bond_type):
    structure = parse_chain(["ALA", "SEP", "ALA"], [0, 100, 20])
    modified = structure.residues[1]
    atoms = structure.atoms[
        modified["atom_idx"] : modified["atom_idx"] + modified["atom_num"]
    ]
    atom_ids = {a["name"]: modified["atom_idx"] + i for i, a in enumerate(atoms)}
    # Either remove N-CA or turn the carbonyl into a single bond.
    names = ("N", "CA") if bond_type == "SINGLE" else ("C", "O")
    edge = {atom_ids[n] for n in names}
    matches = np.array([{b["atom_1"], b["atom_2"]} == edge for b in structure.bonds])
    assert matches.sum() == 1
    if bond_type == "SINGLE":
        structure = replace(structure, bonds=structure.bonds[~matches])
    else:
        structure.bonds["type"][matches] = const.bond_type_ids["SINGLE"]
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])


def test_serialization_and_concatenation_preserve_modified_backbone_checks(tmp_path):
    structure = parse_chain(["ALA", "SEP"], [0, 20])
    path = tmp_path / "structure.npz"
    structure.dump(path)
    restored = Structure.load(path)
    good = parse_chain(["ALA", "SEP"], [0, 3.8])
    combined = Structure.concatenate(good, restored)
    np.testing.assert_array_equal(ConsecutiveCA().filter(combined), [True, False])


def test_cropping_does_not_create_sequence_adjacency():
    structure = parse_chain(["ALA", "SEP", "ALA"], [0, 7, 14])
    cropped = Structure.extract_residues(structure, np.array([True, False, True]))
    assert cropped.residues["res_idx"].tolist() == [0, 2]
    np.testing.assert_array_equal(ConsecutiveCA().filter(cropped), [True])


def test_removed_ca_interrupts_trace_after_atom_extraction():
    structure = parse_chain(["ALA", "SER", "ALA"], [0, 20, 40])
    middle = structure.residues[1]
    atom_indices = np.arange(len(structure.atoms))
    cropped = Structure.extract_atoms(
        structure, atom_indices[atom_indices != middle["atom_center"]]
    )
    assert cropped.residues[1]["atom_center"] == -1
    np.testing.assert_array_equal(ConsecutiveCA().filter(cropped), [True])


@pytest.mark.parametrize("mol_type", ["RNA", "DNA", "NONPOLYMER"])
def test_nonprotein_chains_are_unchanged(mol_type):
    structure = parse_chain(["ALA", "SEP"], [0, 20])
    structure.chains["mol_type"] = const.chain_type_ids[mol_type]
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])


def test_exact_custom_threshold_and_next_float():
    structure = parse_chain(["ALA", "SEP"], [0, 4])
    np.testing.assert_array_equal(ConsecutiveCA(max_dist=4).filter(structure), [True])
    modified = structure.residues[1]
    atom_slice = slice(
        modified["atom_idx"], modified["atom_idx"] + modified["atom_num"]
    )
    atoms = structure.atoms[atom_slice]
    atoms["coords"][atoms["name"] == "CA", 0] = np.nextafter(
        np.float32(4), np.float32(5)
    )
    np.testing.assert_array_equal(ConsecutiveCA(max_dist=4).filter(structure), [False])


def test_empty_chain_and_single_residue_have_no_pair():
    structure = parse_chain(["ALA"], [0])
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])
    structure.chains["res_num"] = 0
    structure.chains["atom_num"] = 0
    empty = replace(
        structure,
        atoms=structure.atoms[:0],
        residues=structure.residues[:0],
        bonds=structure.bonds[:0],
    )
    np.testing.assert_array_equal(ConsecutiveCA().filter(empty), [True])


def test_ca_lookup_ignores_placeholder_center_and_ambiguous_names():
    structure = parse_chain(["ALA", "SEP"], [0, 20])
    structure.residues["atom_center"] = -1
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [False])
    modified = structure.residues[1]
    atoms = structure.atoms[
        modified["atom_idx"] : modified["atom_idx"] + modified["atom_num"]
    ]
    atoms["name"][atoms["name"] == "CB"] = "CA"
    np.testing.assert_array_equal(ConsecutiveCA().filter(structure), [True])
