"""Real file and crop boundaries for insertion and fusion bookkeeping."""

# Fixture indices are explicit independent expectations.
# ruff: noqa: CPY001, INP001, PLR2004

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from test_structure_insert_fuse import (
    assert_consistent,
    assert_unmodified,
    make_structure,
)

from boltzgen.data import const
from boltzgen.data.crop.multimer import MultimerCropper
from boltzgen.data.data import Bond, Input, Structure, Target
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif

pytestmark = pytest.mark.filterwarnings(
    "error:Conversion of an array with ndim > 0 to a scalar is deprecated"
    ":DeprecationWarning"
)


def molecule(letter: str = "G") -> Chem.Mol:
    """Create the named atoms and conformer needed by the real parser."""
    mol = Chem.MolFromSequence(letter)
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return mol


def parse_entities(
    tmp_path: Path, entities: list[dict], constraints: list[dict] | None = None
) -> Target:
    """Use the actual parser with a deterministic minimal molecule dictionary."""
    schema = {"entities": entities}
    if constraints is not None:
        schema["constraints"] = constraints
    return YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "boundaries",
        schema,
        {"GLY": molecule(), "CYS": molecule("C")},
        tmp_path,
        tmp_path,
    )


@pytest.mark.parametrize(
    "indices", [(0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (3, 4, 5), (0, 2, 3, 5)]
)
def test_explicit_crop_preserves_residue_mapping(
    indices: tuple[int, ...], tmp_path: Path
) -> None:
    source = parse_entities(
        tmp_path,
        [{"protein": {"id": ["A", "B"], "sequence": "GGG", "cyclic": True}}],
    ).structure
    original = deepcopy(source)
    tokenized = Tokenizer().tokenize(source)
    original_mapping = tokenized.token_to_res.copy()
    cropped = MultimerCropper([3]).crop_indices(tokenized, list(indices))
    np.testing.assert_array_equal(cropped.token_to_res, indices)
    np.testing.assert_array_equal(tokenized.token_to_res, original_mapping)
    features = Featurizer().process(
        Input(cropped.tokens, cropped.bonds, cropped.token_to_res, source, {}, {}),
        random=np.random.default_rng(0),
        molecules={"GLY": Chem.RemoveHs(molecule())},
        training=False,
        max_seqs=1,
    )
    features["id"] = "explicit_crop"
    result, _, _ = Structure.from_feat(features)
    retained = [
        bond
        for bond in source.bonds
        if bond["res_1"] in indices and bond["res_2"] in indices
    ]
    assert len(result.bonds) == len(retained)
    for actual, before in zip(result.bonds, retained, strict=True):
        for end, atom_name in ((1, "N"), (2, "C")):
            expected_residue = indices.index(int(before[f"res_{end}"]))
            assert actual[f"res_{end}"] == expected_residue
            residue = result.residues[expected_residue]
            assert residue["atom_idx"] <= actual[f"atom_{end}"]
            assert actual[f"atom_{end}"] < residue["atom_idx"] + residue["atom_num"]
            assert result.atoms[actual[f"atom_{end}"]]["name"] == atom_name
    to_mmcif(result)
    assert_unmodified(source, original)


@pytest.mark.parametrize("indices", [(), (2, 0, 2)])
def test_explicit_crop_preserves_optional_mapping(indices: tuple[int, ...]) -> None:
    tokenized = replace(Tokenizer().tokenize(make_structure(1)), token_to_res=None)
    result = MultimerCropper([3]).crop_indices(tokenized, list(indices))
    assert result.token_to_res is None
    np.testing.assert_array_equal(result.tokens["token_idx"], sorted(set(indices)))


@pytest.mark.parametrize("indices", [(), (1,)])
def test_explicit_crop_maps_automatically_retained_ligands(
    indices: tuple[int, ...], tmp_path: Path
) -> None:
    source = parse_entities(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "GGG"}},
            {"ligand": {"id": "L", "smiles": "CC"}},
        ],
    ).structure
    tokenized = Tokenizer().tokenize(source)
    selected = sorted(set(indices) | {3, 4})
    result = MultimerCropper([3]).crop_indices(tokenized, list(indices))
    np.testing.assert_array_equal(result.tokens["token_idx"], selected)
    np.testing.assert_array_equal(result.token_to_res, tokenized.token_to_res[selected])


@pytest.mark.parametrize(
    ("first_method", "has_mapping"),
    [("explicit", True), ("normal", True), ("explicit", False)],
)
def test_explicit_crop_accepts_retained_ligand_token_ids(
    first_method: str, has_mapping: bool, tmp_path: Path
) -> None:
    source = parse_entities(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "GGG"}},
            {"ligand": {"id": "L", "smiles": "CC"}},
        ],
    ).structure
    tokenized = Tokenizer().tokenize(source)
    cropper = MultimerCropper([3])
    if first_method == "normal":
        first = cropper.crop(
            tokenized,
            max_tokens=4,
            random=np.random.default_rng(0),
            initial_crop=[1, 2, 3, 4],
        )
    else:
        first = cropper.crop_indices(tokenized, [1, 2])
    if not has_mapping:
        first = replace(first, token_to_res=None)
    original = deepcopy(first)

    second = cropper.crop_indices(first, [0])

    np.testing.assert_array_equal(second.tokens, original.tokens[[0, 2, 3]])
    if has_mapping:
        np.testing.assert_array_equal(second.token_to_res, [1, 3, 3])
        np.testing.assert_array_equal(first.token_to_res, original.token_to_res)
    else:
        assert second.token_to_res is None
        assert first.token_to_res is None
    np.testing.assert_array_equal(first.tokens, original.tokens)
    np.testing.assert_array_equal(first.bonds, original.bonds)
    assert_unmodified(first.structure, original.structure)


@pytest.mark.parametrize("chain_name", ["A", "B", "C"])
@pytest.mark.parametrize("position", [-1, 4])
def test_insert_rejects_positions_outside_target_chain(
    chain_name: str, position: int
) -> None:
    source = make_structure()
    original = deepcopy(source)
    with pytest.raises(ValueError, match=r"Insertion position .* outside"):
        Structure.insert(source, chain_name, position, 1)
    assert_unmodified(source, original)


@pytest.mark.parametrize("position", [0, 5])
def test_yaml_rejects_positions_outside_target_chain(
    position: int, tmp_path: Path
) -> None:
    path = tmp_path / "input.cif"
    path.write_text(to_mmcif(make_structure()))
    original = path.read_bytes()
    with pytest.raises(ValueError, match=r"Insertion position .* outside"):
        parse_entities(
            tmp_path,
            [
                {
                    "file": {
                        "path": str(path),
                        "design_insertions": [
                            {
                                "insertion": {
                                    "id": "A",
                                    "res_index": position,
                                    "num_residues": "1..1",
                                }
                            }
                        ],
                    }
                }
            ],
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_insert_retains_gapped_residue_identities(
    chain_idx: int, position: int
) -> None:
    source = make_structure()
    start = chain_idx * 3
    source.residues["res_idx"][start : start + 3] = [2, 5, 7]
    original = deepcopy(source)
    result = Structure.insert(source, chr(65 + chain_idx), position, 2)
    expected = [
        [2, 3, 4, 7, 9],
        [2, 5, 6, 7, 9],
        [2, 5, 7, 8, 9],
        [2, 5, 7, 8, 9],
    ][position]
    np.testing.assert_array_equal(
        result.residues["res_idx"][start : start + 5], expected
    )
    assert_consistent(result)
    assert_unmodified(source, original)


@pytest.mark.parametrize("position", [1, 2, 3, 4])
def test_yaml_gapped_insertion_roundtrip_and_source_context(
    position: int, tmp_path: Path
) -> None:
    source = make_structure(1)
    source.residues["res_idx"] = [0, 3, 5]
    path = tmp_path / "gap.cif"
    path.write_text(to_mmcif(source))
    parsed = parse_entities(
        tmp_path,
        [
            {
                "file": {
                    "path": str(path),
                    "full_sequences": [
                        {
                            "chain": {
                                "id": "A",
                                "sequence": "GGGGGG",
                                "source_res_indices": [1, 4, 6],
                            }
                        }
                    ],
                    "design_insertions": [
                        {
                            "insertion": {
                                "id": "A",
                                "res_index": position,
                                "num_residues": "2..2",
                            }
                        }
                    ],
                }
            }
        ],
    )
    expected = [
        [0, 1, 2, 5, 7],
        [0, 3, 4, 5, 7],
        [0, 3, 5, 6, 7],
        [0, 3, 5, 6, 7],
    ][position - 1]
    np.testing.assert_array_equal(parsed.structure.residues["res_idx"], expected)
    assert parsed.source_context["chains"][0]["indices"] == expected
    saved = tmp_path / "inserted.cif"
    saved.write_text(to_mmcif(parsed.structure))
    reparsed = parse_entities(tmp_path, [{"file": {"path": str(saved)}}])
    np.testing.assert_array_equal(reparsed.structure.residues["res_idx"], expected)


@pytest.mark.parametrize("initial_len", [0, 1])
@pytest.mark.parametrize("placement", ["initial", "preceded", "intervening", "both"])
def test_fuse_into_empty_target_uses_its_own_indices(
    initial_len: int, placement: str, tmp_path: Path
) -> None:
    source = make_structure(1)
    source.residues["res_idx"] = [3, 5, 8]
    path = tmp_path / "donor.cif"
    path.write_text(to_mmcif(source))
    entities = []
    if placement in {"preceded", "both"}:
        entities.append({"protein": {"id": "P", "sequence": "GGG"}})
    entities.append(
        {"protein": {"id": "L", "sequence": f"{initial_len}..{initial_len}"}}
    )
    if placement in {"intervening", "both"}:
        entities.append({"protein": {"id": "Q", "sequence": "GG"}})
    entities.extend(
        [
            {"file": {"path": str(path), "fuse": "L"}},
            {"protein": {"id": "B", "sequence": "3"}},
        ]
    )
    parsed = parse_entities(tmp_path, entities)
    chain = parsed.structure.chains[parsed.structure.chains["name"] == "L"][0]
    start = int(chain["res_idx"]) + initial_len
    np.testing.assert_array_equal(
        parsed.structure.residues["res_idx"][start : start + 3],
        np.array([0, 2, 5]) + initial_len,
    )
    assert_consistent(parsed.structure)
    assert [
        chain["source_chain"] for chain in parsed.source_context["chains"]
    ] == parsed.structure.chains["name"].tolist()


@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
@pytest.mark.parametrize("return_renaming", [False, True])
def test_concatenate_chainless_identity(
    empty_side: str, *, return_renaming: bool
) -> None:
    empty = Structure.empty_protein(0)
    neutral = replace(empty, chains=empty.chains[:0], mask=empty.mask[:0])
    source = make_structure(1)
    left = neutral if empty_side in {"left", "both"} else source
    right = neutral if empty_side in {"right", "both"} else source
    result = Structure.concatenate(left, right, return_renaming=return_renaming)
    if return_renaming:
        result, names = result
        assert names == {}
    assert isinstance(result, Structure)
    assert_unmodified(result, neutral if empty_side == "both" else source)


def test_concatenate_preserves_explicit_empty_chain() -> None:
    empty = Structure.empty_protein(0)
    result, renaming = Structure.concatenate(
        make_structure(1), empty, return_renaming=True
    )
    np.testing.assert_array_equal(result.chains["res_num"], [3, 0])
    np.testing.assert_array_equal(result.chains["name"], ["A", "B"])
    assert renaming == {"A": "B"}
    assert_consistent(result)


@pytest.mark.parametrize("models", [1, 2])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("declared", [False, True])
def test_empty_concatenation_preserves_parsed_coordinate_models(
    models: int, side: str, *, declared: bool, tmp_path: Path
) -> None:
    source_path = tmp_path / "source.cif"
    source_path.write_text(to_mmcif(make_structure(1)))
    raw = gemmi.read_structure(str(source_path))
    if models == 2:
        second = raw[0].clone()
        second.num = 2
        for chain in second:
            for residue in chain:
                for atom in residue:
                    atom.pos += gemmi.Position(100, 200, 300)
        raw.add_model(second)
    raw.make_mmcif_document().write_file(str(source_path))
    source = parse_mmcif(
        str(source_path), {"GLY": molecule()}, str(tmp_path), use_assembly=False
    ).data
    empty = Structure.empty_protein(0)
    empty.chains["name"] = "E"
    if not declared:
        empty = replace(empty, chains=empty.chains[:0], mask=empty.mask[:0])
    originals = deepcopy((empty, source))
    operands = (empty, source) if side == "left" else (source, empty)
    result, _ = Structure.concatenate(*operands, return_renaming=True)
    np.testing.assert_array_equal(result.ensemble, source.ensemble)
    np.testing.assert_array_equal(result.coords, source.coords)
    edited = Structure.insert(result, "A", 1, 1)
    assert len(edited.ensemble) == models
    for model in range(models):
        expected = np.concatenate(
            [
                source.coords["coords"][model * 12 : model * 12 + 4],
                np.zeros((4, 3)),
                source.coords["coords"][model * 12 + 4 : (model + 1) * 12],
            ]
        )
        np.testing.assert_array_equal(
            edited.coords["coords"][model * 16 : (model + 1) * 16], expected
        )
    assert_unmodified(empty, originals[0])
    assert_unmodified(source, originals[1])


@pytest.mark.parametrize("copied", [False, True])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("fusion_target", [None, "A", "B"])
def test_inline_constraints_and_copied_cycles_follow_final_atoms(
    *,
    copied: bool,
    cyclic: bool,
    prefix: bool,
    fusion_target: str | None,
    tmp_path: Path,
) -> None:
    entities = [{"protein": {"id": "P", "sequence": "GG"}}] if prefix else []
    ids = [["A", "B"]] if copied else ["A", "B"]
    entities.extend(
        {"protein": {"id": value, "sequence": "CGC", "cyclic": cyclic}} for value in ids
    )
    if fusion_target is not None:
        entities.append(
            {"protein": {"id": "D", "sequence": "2", "fuse": fusion_target}}
        )
    result = parse_entities(
        tmp_path,
        entities,
        [
            {"bond": {"atom1": ["B", 1, "SG"], "atom2": ["B", 3, "SG"]}},
            {"bond": {"atom1": ["A", 1, "SG"], "atom2": ["B", 3, "SG"]}},
        ],
    ).structure
    for bond, names, positions in zip(
        result.bonds[-2:], [("B", "B"), ("A", "B")], [(0, 2), (0, 2)], strict=True
    ):
        for end, name, position in zip((1, 2), names, positions, strict=True):
            chain_idx = result.chains["name"].tolist().index(name)
            residue_idx = int(result.chains[chain_idx]["res_idx"]) + position
            residue = result.residues[residue_idx]
            atoms = result.atoms[
                residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
            ]
            atom_idx = (
                residue["atom_idx"] + np.flatnonzero(atoms["name"] == "SG").item()
            )
            assert bond[f"chain_{end}"] == chain_idx
            assert bond[f"res_{end}"] == residue_idx
            assert bond[f"atom_{end}"] == atom_idx
    if cyclic:
        for bond, name in zip(result.bonds[:2], ("A", "B"), strict=True):
            chain_idx = result.chains["name"].tolist().index(name)
            chain = result.chains[chain_idx]
            assert bond["res_1"] == chain["res_idx"]
            assert bond["res_2"] == chain["res_idx"] + chain["res_num"] - 1
            assert bond["chain_1"] == bond["chain_2"] == chain_idx
            assert result.atoms["name"][bond["atom_1"]] == "N"
            assert result.atoms["name"][bond["atom_2"]] == "C"
    to_mmcif(result)


@pytest.mark.parametrize("target", ["A", "B"])
def test_constraint_donor_alias_survives_repeated_fusion(
    target: str, tmp_path: Path
) -> None:
    result = parse_entities(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "GG"}},
            {"protein": {"id": "B", "sequence": "GGG"}},
            {"protein": {"id": "D", "sequence": "CGC", "fuse": target}},
            {"protein": {"id": "E", "sequence": "2", "fuse": target}},
        ],
        [{"bond": {"atom1": ["D", 1, "SG"], "atom2": ["D", 3, "SG"]}}],
    ).structure
    chain_idx = result.chains["name"].tolist().index(target)
    start = int(result.chains[chain_idx]["res_idx"]) + (2 if target == "A" else 3)
    bond = result.bonds[0]
    for end, position in ((1, start), (2, start + 2)):
        residue = result.residues[position]
        assert bond[f"chain_{end}"] == chain_idx
        assert bond[f"res_{end}"] == position
        assert (
            residue["atom_idx"]
            <= bond[f"atom_{end}"]
            < residue["atom_idx"] + residue["atom_num"]
        )
        assert result.atoms["name"][bond[f"atom_{end}"]] == "SG"
    to_mmcif(result)


@pytest.mark.parametrize("selected_chain", [0, 1, 2])
def test_fused_donor_bonds_use_chain_table_indices(selected_chain: int) -> None:
    source = make_structure()
    source = replace(source, bonds=source.bonds[:0])
    mol = Chem.MolFromSequence("G")
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    mols = {"GLY": Chem.RemoveHs(mol)}
    tokenized = Tokenizer().tokenize(source)
    cropped = MultimerCropper([3]).crop(
        tokenized,
        max_tokens=3,
        random=np.random.default_rng(0),
        initial_crop=list(range(selected_chain * 3, selected_chain * 3 + 3)),
    )
    features = Featurizer().process(
        Input(cropped.tokens, cropped.bonds, cropped.token_to_res, source, {}, {}),
        random=np.random.default_rng(0),
        molecules=mols,
        training=False,
        max_seqs=1,
    )
    features["id"] = "crop"
    target, _, _ = Structure.from_feat(features)
    assert len(target.chains) == 1
    assert target.chains[0]["asym_id"] == selected_chain
    donor = replace(
        make_structure(1),
        bonds=np.array(
            [(0, 0, 0, 2, 1, 9, const.bond_type_ids["COVALENT"])], dtype=Bond
        ),
    )
    result = Structure.fuse(
        target, donor, str(target.chains[0]["name"]), res_reindex=True
    )
    np.testing.assert_array_equal(result.bonds["chain_1"], [0])
    np.testing.assert_array_equal(result.bonds["chain_2"], [0])
    to_mmcif(result)
    assert_consistent(result)


@pytest.mark.parametrize("position", [0, 1, 3])
@pytest.mark.parametrize("period", [-1, 0, 3])
def test_insert_keeps_non_backbone_cyclic_connections(
    position: int, period: int, tmp_path: Path
) -> None:
    source = replace(
        make_structure(1),
        bonds=np.array(
            [
                (0, 0, 0, 2, 1, 9, const.bond_type_ids["COVALENT"]),
            ],
            dtype=Bond,
        ),
    )
    path = tmp_path / "other_connection.cif"
    path.write_text(to_mmcif(source))
    source = parse_entities(tmp_path, [{"file": {"path": str(path)}}]).structure
    assert source.chains[0]["cyclic_period"] == 3
    source.chains[0]["cyclic_period"] = period
    result = Structure.insert(source, "A", position, 2)
    bond = result.bonds[0]
    assert result.atoms["name"][bond["atom_1"]] == "CA"
    assert result.atoms["name"][bond["atom_2"]] == "CA"
    assert bond["res_1"] == (2 if position == 0 else 0)
    assert bond["res_2"] == (4 if position <= 2 else 2)
    assert result.chains[0]["cyclic_period"] == source.chains[0]["cyclic_period"]
    assert_consistent(result)
    to_mmcif(result)


@pytest.mark.parametrize("position", [0, 1, 3, None])
def test_cropped_cycle_uses_real_bond_when_period_is_missing(
    position: int | None, tmp_path: Path
) -> None:
    source = parse_entities(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "GGG"}},
            {"protein": {"id": "B", "sequence": "GGG", "cyclic": True}},
        ],
    ).structure
    tokenized = Tokenizer().tokenize(source)
    cropped = MultimerCropper([3]).crop(
        tokenized, max_tokens=3, random=np.random.default_rng(0), initial_crop=[3, 4, 5]
    )
    features = Featurizer().process(
        Input(cropped.tokens, cropped.bonds, cropped.token_to_res, source, {}, {}),
        random=np.random.default_rng(0),
        molecules={"GLY": Chem.RemoveHs(molecule())},
        training=False,
        max_seqs=1,
    )
    features["id"] = "cyclic_crop"
    target, _, _ = Structure.from_feat(features)
    assert target.chains[0]["cyclic_period"] == 0
    assert target.chains[0]["asym_id"] == 1
    snapshot = deepcopy(target)
    name = str(target.chains[0]["name"])
    if position is None:
        result = Structure.fuse(target, make_structure(1), name, res_reindex=True)
        count = 6
    else:
        result = Structure.insert(target, name, position, 2)
        count = 5
    assert result.chains[0]["cyclic_period"] == count
    bond = result.bonds[0]
    assert bond["res_1"] == 0
    assert bond["res_2"] == count - 1
    assert result.atoms["name"][bond["atom_1"]] == "N"
    assert result.atoms["name"][bond["atom_2"]] == "C"
    assert_consistent(result)
    assert_unmodified(target, snapshot)


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
def test_fuse_retains_target_and_donor_gaps(chain_idx: int) -> None:
    source = make_structure()
    start = chain_idx * 3
    source.residues["res_idx"][start : start + 3] = [2, 5, 7]
    donor = make_structure(1)
    donor.residues["res_idx"] = [4, 6, 9]
    result = Structure.fuse(source, donor, chr(65 + chain_idx), res_reindex=True)
    np.testing.assert_array_equal(
        result.residues["res_idx"][start : start + 6], [2, 5, 7, 8, 10, 13]
    )
    assert_consistent(result)


@pytest.mark.parametrize("chain_idx", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("position", [0, 1, 3, None])
def test_cyclic_edits_keep_terminal_closure(
    chain_idx: int, *, reverse: bool, position: int | None
) -> None:
    source = make_structure()
    start = chain_idx * 3
    source.chains["cyclic_period"][chain_idx] = 3
    closure = np.array(
        [(chain_idx, chain_idx, start, start + 2, start * 4, (start + 2) * 4 + 2, 1)],
        dtype=Bond,
    )
    if reverse:
        for field in ("res", "atom"):
            closure[f"{field}_1"], closure[f"{field}_2"] = (
                closure[f"{field}_2"].copy(),
                closure[f"{field}_1"].copy(),
            )
    source = replace(source, bonds=np.concatenate([source.bonds, closure]))
    original = deepcopy(source)
    chain_name = chr(65 + chain_idx)
    if position is None:
        result = Structure.fuse(source, make_structure(1), chain_name, res_reindex=True)
        length = 6
    else:
        result = Structure.insert(source, chain_name, position, 2)
        length = 5
    assert result.chains["cyclic_period"][chain_idx] == length
    bond = result.bonds[-1]
    n_end, c_end = (2, 1) if reverse else (1, 2)
    assert bond[f"res_{n_end}"] == start
    assert bond[f"res_{c_end}"] == start + length - 1
    assert result.atoms["name"][bond[f"atom_{n_end}"]] == "N"
    assert result.atoms["name"][bond[f"atom_{c_end}"]] == "C"
    assert_consistent(result)
    assert_unmodified(source, original)
    tokenized = Tokenizer().tokenize(result)
    np.testing.assert_array_equal(
        tokenized.tokens["cyclic_period"][start : start + length], length
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("position", [1, 2, 4])
def test_yaml_cyclic_insertion_then_fusion_roundtrips(
    *, reverse: bool, position: int, tmp_path: Path
) -> None:
    source = parse_entities(
        tmp_path, [{"protein": {"id": "A", "sequence": "GGG", "cyclic": True}}]
    ).structure
    source.atoms["coords"] = np.arange(len(source.atoms) * 3).reshape(-1, 3) + 1
    source.coords["coords"] = source.atoms["coords"]
    if reverse:
        for field in ("res", "atom"):
            source.bonds[f"{field}_1"], source.bonds[f"{field}_2"] = (
                source.bonds[f"{field}_2"].copy(),
                source.bonds[f"{field}_1"].copy(),
            )
    path = tmp_path / "cycle.cif"
    path.write_text(to_mmcif(source))
    control = parse_entities(tmp_path, [{"file": {"path": str(path)}}])
    assert control.structure.chains[0]["cyclic_period"] == 3
    parsed = parse_entities(
        tmp_path,
        [
            {
                "file": {
                    "path": str(path),
                    "design_insertions": [
                        {
                            "insertion": {
                                "id": "A",
                                "res_index": position,
                                "num_residues": "2..2",
                            }
                        }
                    ],
                }
            },
            {"protein": {"id": "D", "sequence": "2", "fuse": "A"}},
        ],
    )
    assert parsed.structure.chains[0]["cyclic_period"] == 7
    bond = parsed.structure.bonds[0]
    n_end, c_end = (2, 1) if reverse else (1, 2)
    assert bond[f"res_{n_end}"] == 0
    assert bond[f"res_{c_end}"] == 6
    assert_consistent(parsed.structure)
    saved = tmp_path / "edited.cif"
    saved.write_text(to_mmcif(parsed.structure))
    reparsed = parse_entities(tmp_path, [{"file": {"path": str(saved)}}])
    assert reparsed.structure.chains[0]["cyclic_period"] == 7
    np.testing.assert_array_equal(reparsed.structure.bonds, parsed.structure.bonds)
