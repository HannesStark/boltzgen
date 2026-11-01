"""File-loaded atoms support bonds and explicit leaving-atom selections."""

# Pytest injects the molecule fixture imported from the parser tests.
# ruff: noqa: F811

from pathlib import Path

import pytest
from test_pdb_parser_integrity import PROTEIN, mols, write_pdb  # noqa: F401
from test_structure_transform_boundaries import molecule

from boltzgen.data.parse.schema import YamlDesignParser


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("use_bond", [False, True])
def test_file_chain_leaving_atom_preserves_source_position(
    tmp_path: Path, mols, prefix: bool, crop: bool, use_bond: bool
) -> None:
    mols = {**mols, "GLY": molecule()}
    path = write_pdb(tmp_path / "fixture.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})
    selection = {"id": "R"}
    if crop:
        selection["res_index"] = "2..5"
    entities = [{"protein": {"id": "P", "sequence": "GG"}}] if prefix else []
    entities.append({"file": {"path": str(path), "include": [{"chain": selection}]}})
    schema = {"entities": entities, "leaving_atoms": [{"atom": ["R", 2, "O"]}]}
    if use_bond:
        schema["constraints"] = [
            {"bond": {"atom1": ["R", 2, "N"], "atom2": ["R", 3, "C"]}}
        ]
    target = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "test", schema, mols, tmp_path, tmp_path
    )
    chain_index = int(prefix)
    chain = target.structure.chains[chain_index]
    residues = target.structure.residues[
        chain["res_idx"] : chain["res_idx"] + chain["res_num"]
    ]
    residue = residues[residues["res_idx"] == 1][0]
    atoms = target.structure.atoms[
        residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
    ]
    assert not atoms[atoms["name"] == "O"]["is_present"].item()
    assert atoms[atoms["name"] == "CA"]["is_present"].item()
    assert target.source_context["chains"][chain_index]["omitted_atoms"] == [[1, "O"]]


@pytest.mark.parametrize("file_chain", [False, True])
@pytest.mark.parametrize("missing_end", [1, 2])
def test_bond_unknown_chain_reports_the_constraint(
    tmp_path: Path, mols, file_chain: bool, missing_end: int
) -> None:
    mols = {**mols, "GLY": molecule()}
    if file_chain:
        path = write_pdb(tmp_path / "fixture.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})
        entities = [{"file": {"path": str(path)}}]
    else:
        entities = [{"protein": {"id": "R", "sequence": "GG"}}]
    bond = {"atom1": ["R", 1, "N"], "atom2": ["R", 2, "C"]}
    bond[f"atom{missing_end}"][0] = "MISSING"
    with pytest.raises(ValueError, match="Chain MISSING.*does not exist"):
        YamlDesignParser(tmp_path).parse_boltzgen_schema(
            "test",
            {"entities": entities, "constraints": [{"bond": bond}]},
            mols,
            tmp_path,
            tmp_path,
        )
