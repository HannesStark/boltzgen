"""Public-parser regression expectations for atom selections."""

# Pytest injects the imported molecule fixture.
# ruff: noqa: F811

from pathlib import Path

import numpy as np
import pytest
from test_pdb_parser_integrity import PROTEIN, mols, write_pdb  # noqa: F401
from test_structure_transform_boundaries import molecule

from boltzgen.data.parse.schema import YamlDesignParser


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


CASES = [
    ("inline_file_alias", "R", 4, "A", 4),
    ("inline_file_final", "A", 4, "A", 4),
    ("sampled_inline_file_alias", "R", 3, "A", 3),
    ("sampled_inline_file_final", "A", 3, "A", 3),
    ("file_sampled_inline_alias", "R", 3, "A", 5),
    ("file_sampled_inline_final", "A", 3, "A", 5),
    ("inline_sampled_inline_alias", "R", 3, "A", 5),
    ("inline_sampled_inline_final", "A", 3, "A", 5),
    ("fused_donor", "D", 3, "R", 7),
    ("fused_donor_with_trailing", "D", 3, "R", 7),
    ("fused_after_collision", "D", 3, "A", 11),
    ("repeat_file_alias", "R", 4, "B", 4),
    ("repeat_file_final", "A", 4, "A", 4),
    ("renaming_chain_alias", "R", 3, "A", 3),
    ("renaming_chain_final", "A", 3, "B", 5),
    ("renaming_chain_latest", "B", 3, "B", 5),
    ("crop_alias", "R", 3, "A", 3),
    ("crop_final", "A", 3, "A", 3),
    ("middle_fuse_shifted_sampled", "T", 3, "T", 5),
    ("later_fuse_shifts_donor", "D", 3, "T", 7),
    ("later_file_fuse_shifts_inline", "T", 3, "T", 5),
    ("empty_fusion_target", "D", 3, "R", 5),
    ("inline_copies_collision_alias", "R", 3, "A", 5),
    ("inline_copies_collision_final", "A", 3, "A", 5),
    ("inline_copies_collision_other", "S", 3, "S", 5),
    ("repeat_inline_earlier_final", "A", 3, "A", 5),
    ("repeat_inline_latest_alias", "R", 3, "B", 3),
    ("repeat_inline_latest_final", "B", 3, "B", 3),
    ("gapped_crop_alias", "R", 4, "A", 4),
    ("gapped_crop_final", "A", 4, "A", 4),
]


@pytest.mark.parametrize("action", ["bond", "leaving"])
@pytest.mark.parametrize("case,chain,residue,expected_chain,expected_residue", CASES)
def test_chain_atom_identity(
    tmp_path: Path, mols, action, case, chain, residue, expected_chain, expected_residue
):
    references = {**mols, "GLY": molecule()}
    path = write_pdb(tmp_path / "source.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})

    def inline(name, sequence, **kw):
        return {"protein": {"id": name, "sequence": sequence, **kw}}

    def file(crop=False, **kw):
        selection = {"id": "R"}
        if crop:
            selection["res_index"] = "2,4..5" if crop == "gapped" else "2..5"
        return {"file": {"path": str(path), "include": [{"chain": selection}], **kw}}

    entities = {
        "inline_file": [inline("R", "GG"), file()],
        "sampled_inline_file": [inline("R", "G1..3G"), file()],
        "file_sampled_inline": [file(), inline("R", "G1..3G")],
        "inline_sampled_inline": [inline("R", "GG"), inline("R", "G1..3G")],
        "fused_donor": [inline("R", "GG"), inline("D", "G1..3G", fuse="R")],
        "fused_donor_with_trailing": [
            inline("R", "GG"),
            inline("T", "GGG"),
            inline("D", "G1..3G", fuse="R"),
        ],
        "fused_after_collision": [
            inline("R", "GG"),
            file(),
            inline("D", "G1..3G", fuse="A"),
        ],
        "repeat_file": [inline("R", "GG"), file(), file()],
        "renaming_chain": [inline("R", "GG"), file(), inline("A", "G1..3G")],
        "crop": [inline("R", "G1..3G"), file(crop=True)],
        "middle_fuse_shifted_sampled": [
            inline("R", "GG"),
            inline("T", "G1..3G"),
            inline("D", "GG", fuse="R"),
        ],
        "later_fuse_shifts_donor": [
            inline("R", "GG"),
            inline("T", "GG"),
            inline("D", "G1..3G", fuse="T"),
            inline("E", "GG", fuse="R"),
        ],
        "later_file_fuse_shifts_inline": [
            file(),
            inline("T", "G1..3G"),
            file(fuse="R"),
        ],
        "empty_fusion_target": [inline("R", ""), inline("D", "G1..3G", fuse="R")],
        "inline_copies_collision": [inline("R", "GG"), inline(["R", "S"], "G1..3G")],
        "repeat_inline_earlier": [
            inline("R", "GG"),
            inline("R", "G1..3G"),
            inline("R", "GGG"),
        ],
        "repeat_inline_latest": [
            inline("R", "GG"),
            inline("R", "G1..3G"),
            inline("R", "GGG"),
        ],
        "gapped_crop": [inline("R", "GG"), file(crop="gapped")],
    }
    key = case
    for suffix in ("_alias", "_final", "_other"):
        key = key.removesuffix(suffix)
    if key == "renaming_chain_latest":
        key = "renaming_chain"
    schema = {"entities": entities[key]}
    if action == "bond":
        schema["constraints"] = [
            {"bond": {"atom1": [chain, residue, "O"], "atom2": [chain, residue, "N"]}}
        ]
    else:
        schema["leaving_atoms"] = [{"atom": [chain, residue, "O"]}]
    # The range samples three slots: spec G1..3G becomes five residues.
    np.random.seed(3)
    result = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "test", schema, references, tmp_path, tmp_path
    )
    structure = result.structure
    chain_index = np.flatnonzero(structure.chains["name"] == expected_chain).item()
    selected_chain = structure.chains[chain_index]
    residues = structure.residues[
        selected_chain["res_idx"] : selected_chain["res_idx"]
        + selected_chain["res_num"]
    ]
    residue_offset = np.flatnonzero(residues["res_idx"] == expected_residue - 1).item()
    selected_residue = residues[residue_offset]
    atoms = structure.atoms[
        selected_residue["atom_idx"] : selected_residue["atom_idx"]
        + selected_residue["atom_num"]
    ]
    expected_atom = (
        int(selected_residue["atom_idx"]) + np.flatnonzero(atoms["name"] == "O").item()
    )
    if action == "bond":
        actual = structure.bonds[-1]
        assert actual["chain_1"] == chain_index
        assert actual["res_1"] == selected_chain["res_idx"] + residue_offset
        assert actual["atom_1"] == expected_atom
    else:
        absent_oxygen = np.flatnonzero(
            (structure.atoms["name"] == "O") & ~structure.atoms["is_present"]
        )
        np.testing.assert_array_equal(absent_oxygen, [expected_atom])
        context = result.source_context["chains"][chain_index]
        assert context["omitted_atoms"] == [[context["indices"][residue_offset], "O"]]
        assert all(
            not c.get("omitted_atoms")
            for i, c in enumerate(result.source_context["chains"])
            if i != chain_index
        )


@pytest.mark.parametrize(
    "case", ["target_name", "unrelated_chain", "earlier_donor", "later_chain"]
)
def test_fused_donor_alias_collision_rejected(tmp_path: Path, mols, case):
    def inline(name, sequence="GG", **kw):
        return {"protein": {"id": name, "sequence": sequence, **kw}}

    entities = {
        "target_name": [inline("R"), inline("R", fuse="R")],
        "unrelated_chain": [inline("R"), inline("D"), inline("D", fuse="R")],
        "earlier_donor": [inline("R"), inline("D", fuse="R"), inline("D", fuse="R")],
        "later_chain": [inline("R"), inline("D", fuse="R"), inline("D")],
    }[case]
    with pytest.raises(
        ValueError, match="(?i)(alias|ambiguous|collision|conflict|duplicate)"
    ):
        YamlDesignParser(tmp_path).parse_boltzgen_schema(
            "test",
            {"entities": entities},
            {**mols, "GLY": molecule()},
            tmp_path,
            tmp_path,
        )


@pytest.mark.parametrize("action", ["bond", "leaving"])
@pytest.mark.parametrize("kind", ["inline", "file"])
@pytest.mark.parametrize(
    "selection",
    [
        ["R", 0, "O"],
        ["R", -1, "O"],
        ["R", 99, "O"],
        ["R", 1, "MISSING"],
        ["MISSING", 1, "O"],
    ],
)
def test_invalid_atom_selection_has_context(tmp_path, mols, action, kind, selection):
    references = {**mols, "GLY": molecule()}
    if kind == "file":
        path = write_pdb(tmp_path / "source.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})
        entities = [{"file": {"path": str(path)}}]
    else:
        entities = [{"protein": {"id": "R", "sequence": "GG"}}]
    schema = {"entities": entities}
    if action == "bond":
        schema["constraints"] = [{"bond": {"atom1": selection, "atom2": ["R", 1, "N"]}}]
    else:
        schema["leaving_atoms"] = [{"atom": selection}]
    with pytest.raises(ValueError, match="selection"):
        YamlDesignParser(tmp_path).parse_boltzgen_schema(
            "test", schema, references, tmp_path, tmp_path
        )


@pytest.mark.parametrize("action", ["bond", "leaving"])
def test_excluded_file_residue_is_not_reindexed_to_neighbor(tmp_path, mols, action):
    path = write_pdb(tmp_path / "source.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})
    schema = {
        "entities": [
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": "R", "res_index": "2,4..5"}}],
                }
            }
        ]
    }
    if action == "bond":
        schema["constraints"] = [
            {"bond": {"atom1": ["R", 3, "O"], "atom2": ["R", 4, "N"]}}
        ]
    else:
        schema["leaving_atoms"] = [{"atom": ["R", 3, "O"]}]
    with pytest.raises(ValueError, match="Residue.*does not exist"):
        YamlDesignParser(tmp_path).parse_boltzgen_schema(
            "test", schema, mols, tmp_path, tmp_path
        )


@pytest.mark.parametrize("action", ["bond", "leaving"])
@pytest.mark.parametrize("context", ["inline", "renamed", "fused"])
@pytest.mark.parametrize(
    "sequence,position",
    [
        ("G1..3G", 2),
        ("1..3GG", 1),
        ("GG1..3", 3),
        ("1..3", 1),
        ("G0..3G", 1),
        ("2..3", 2),
    ],
)
def test_expanded_ranges_keep_first_minimum_specification_match(
    tmp_path, mols, action, context, sequence, position
):
    references = {**mols, "GLY": molecule()}
    inline = {"protein": {"id": "R", "sequence": sequence}}
    if context == "renamed":
        path = write_pdb(tmp_path / "source.pdb", PROTEIN, {"R": [1, 2, 3, 4, 5, 6]})
        entities = [{"file": {"path": str(path)}}, inline]
        final_chain, offset = "A", 0
    elif context == "fused":
        inline["protein"]["fuse"] = "T"
        entities = [{"protein": {"id": "T", "sequence": "GG"}}, inline]
        final_chain, offset = "T", 2
    else:
        entities = [inline]
        final_chain, offset = "R", 0
    schema = {"entities": entities}
    if action == "bond":
        schema["constraints"] = [
            {"bond": {"atom1": ["R", position, "O"], "atom2": ["R", 1, "N"]}}
        ]
    else:
        schema["leaving_atoms"] = [{"atom": ["R", position, "O"]}]
    np.random.seed(1 if sequence == "2..3" else 3)
    result = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "test", schema, references, tmp_path, tmp_path
    )
    structure = result.structure
    chain_index = np.flatnonzero(structure.chains["name"] == final_chain).item()
    chain = structure.chains[chain_index]
    residue_index = int(chain["res_idx"]) + offset + position - 1
    residue = structure.residues[residue_index]
    atoms = structure.atoms[
        residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
    ]
    atom_index = int(residue["atom_idx"]) + np.flatnonzero(atoms["name"] == "O").item()
    if action == "bond":
        bond = structure.bonds[-1]
        assert (bond["chain_1"], bond["res_1"], bond["atom_1"]) == (
            chain_index,
            residue_index,
            atom_index,
        )
    else:
        np.testing.assert_array_equal(
            np.flatnonzero(
                (structure.atoms["name"] == "O") & ~structure.atoms["is_present"]
            ),
            [atom_index],
        )
        entry = result.source_context["chains"][chain_index]
        assert entry["omitted_atoms"] == [
            [entry["indices"][offset + position - 1], "O"]
        ]
