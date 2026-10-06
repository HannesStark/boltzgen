"""Real file and crop boundaries for insertion and fusion bookkeeping."""

# Fixture indices are explicit independent expectations.
# ruff: noqa: CPY001, INP001, PLR2004

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

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
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif

pytestmark = pytest.mark.filterwarnings(
    "error:Conversion of an array with ndim > 0 to a scalar is deprecated"
    ":DeprecationWarning"
)


def parse_entities(tmp_path: Path, entities: list[dict]) -> Target:
    """Use the actual parser with a deterministic minimal molecule dictionary."""
    mol = Chem.MolFromSequence("G")
    for atom in mol.GetAtoms():
        atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "boundaries", {"entities": entities}, {"GLY": mol}, tmp_path, tmp_path
    )


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
def test_insert_keeps_non_backbone_cyclic_connections(
    position: int, tmp_path: Path
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
    result = Structure.insert(source, "A", position, 2)
    bond = result.bonds[0]
    assert result.atoms["name"][bond["atom_1"]] == "CA"
    assert result.atoms["name"][bond["atom_2"]] == "CA"
    assert bond["res_1"] == (2 if position == 0 else 0)
    assert bond["res_2"] == (4 if position <= 2 else 2)
    assert result.chains[0]["cyclic_period"] == source.chains[0]["cyclic_period"]
    assert_consistent(result)
    to_mmcif(result)


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
