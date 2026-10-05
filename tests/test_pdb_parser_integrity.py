"""PDB labels must preserve each chain's coverage throughout the input pipeline."""

# Numerical expectations describe the small fixtures directly.
# ruff: noqa: INP001, PLR2004

from pathlib import Path

import gemmi
import numpy as np
import pytest
from rdkit import Chem

from boltzgen.data import const
from boltzgen.data.parse.mmcif import ParsedStructure, mmcif_from_block
from boltzgen.data.parse.pdb_parser import parse_pdb
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif

PROTEIN = ["ALA", "GLY", "SER", "THR", "VAL", "LEU"]


@pytest.fixture(scope="module")
def mols() -> dict[str, Chem.Mol]:
    """Use real reference monomers without a downloaded CCD or model weights."""
    result = {}
    for letters, flavor in [("AGSTVL", 0), ("ACGU", 3), ("ACGT", 7)]:
        for letter in letters:
            mol = Chem.MolFromSequence(letter, flavor=flavor)
            assert mol is not None
            name = mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()
            for atom in mol.GetAtoms():
                atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
            result[name] = mol
    return result


def write_pdb(  # noqa: C901, PLR0912
    path: Path,
    sequence: list[str],
    coverage: dict[str, list[int]],
    *,
    seqres: bool = True,
    models: int = 1,
    alternate_chain: str | None = None,
    alternate_identity: bool = True,
    assembly: bool = False,
    water: bool = False,
) -> Path:
    """Write fixed-column records with known chain, residue, and model coordinates."""
    # Keep gaps spatially separated as well as numbered: Gemmi's alignment
    # scoring uses backbone connectivity, including nucleotide O3'-P distances.
    spacing = 3.8 if sequence == PROTEIN else 6.0
    lines = [
        "HEADER    PARSER REGRESSION",
        "COMPND    MOL_ID: 1;",
        "COMPND   2 MOLECULE: TEST POLYMER;",
        f"COMPND   3 CHAIN: {', '.join(coverage)};",
    ]
    if seqres:
        for chain in coverage:
            names = " ".join(f"{name:>3}" for name in sequence)
            lines.append(f"SEQRES   1 {chain} {len(sequence):4d}  {names}")
    if assembly:
        lines.extend(
            [
                "REMARK 350 BIOMOLECULE: 1",
                f"REMARK 350 APPLY THE FOLLOWING TO CHAINS: {', '.join(coverage)}",
            ]
        )
        for operation, translation in [(1, 0.0), (2, 40.0)]:
            for row in range(3):
                matrix = [float(row == col) for col in range(3)]
                offset = translation if row == 0 else 0.0
                lines.append(
                    f"REMARK 350   BIOMT{row + 1} {operation:3d} "
                    f"{matrix[0]:9.6f} {matrix[1]:9.6f} {matrix[2]:9.6f} {offset:14.5f}"
                )
    for model in range(models):
        if models > 1:
            lines.append(f"MODEL     {model + 1:4d}")
        serial = 1
        for chain_idx, (chain, positions) in enumerate(coverage.items()):
            for position in positions:
                name = sequence[position - 1]
                conformers = [(name, " ", 1.0, 0.0)]
                if chain == alternate_chain and position == 3:
                    other_name = "THR" if alternate_identity else name
                    conformers = [(name, "A", 0.6, 0.0), (other_name, "B", 0.4, 9.0)]
                for res_name, altloc, occupancy, alt_offset in conformers:
                    for atom_idx, atom_name in enumerate(const.ref_atoms[res_name]):
                        x = (position - 1) * spacing + atom_idx * 0.7 + alt_offset
                        y = chain_idx * 15.0 + (atom_idx % 3) * 0.6
                        z = model * 0.2
                        atom_field = (
                            f" {atom_name:<3}" if len(atom_name) < 4 else atom_name
                        )
                        lines.append(
                            f"ATOM  {serial:5d} {atom_field}{altloc}{res_name:>3} "
                            f"{chain}{position:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                            f"{occupancy:6.2f}{20.0:6.2f}          {atom_name[0]:>2}"
                        )
                        serial += 1
            if water:
                lines.append(
                    f"HETATM{serial:5d}  O   HOH {chain} 100    "
                    f"{50.0:8.3f}{50.0:8.3f}{50.0:8.3f}"
                    f"{1.0:6.2f}{20.0:6.2f}           O"
                )
                serial += 1
            lines.append("TER")
        if models > 1:
            lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def assert_coverage(
    parsed: ParsedStructure,
    sequence: list[str],
    coverage: dict[str, list[int]],
    *,
    seqres: bool = True,
) -> None:
    spacing = 3.8 if sequence == PROTEIN else 6.0
    assert parsed.data.chains["name"].tolist() == [
        c for c, pos in coverage.items() if pos
    ]
    for chain_idx, (name, positions) in enumerate(coverage.items()):
        if not positions:
            continue
        chain = parsed.data.chains[parsed.data.chains["name"] == name][0]
        residues = parsed.data.residues[
            chain["res_idx"] : chain["res_idx"] + chain["res_num"]
        ]
        expected_positions = list(range(1, len(sequence) + 1)) if seqres else positions
        assert residues["name"].tolist() == [
            sequence[p - 1] for p in expected_positions
        ]
        assert residues["res_idx"].tolist() == list(range(len(expected_positions)))
        assert residues["is_present"].tolist() == [
            p in positions for p in expected_positions
        ]
        for residue, position in zip(residues, expected_positions):
            if position not in positions:
                continue
            atoms = parsed.data.atoms[
                residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
            ]
            coords = parsed.data.coords["coords"][
                residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
            ]
            assert atoms["is_present"].all()
            np.testing.assert_allclose(
                coords[:, 0],
                (position - 1) * spacing + np.arange(len(atoms)) * 0.7,
                atol=0.001,
            )
            np.testing.assert_allclose(
                coords[:, 1],
                chain_idx * 15.0 + (np.arange(len(atoms)) % 3) * 0.6,
                atol=0.001,
            )


def round_trip(parsed: ParsedStructure, mols: dict[str, Chem.Mol]) -> ParsedStructure:
    return mmcif_from_block(
        gemmi.cif.read_string(to_mmcif(parsed.data)).sole_block(),
        mols,
        use_assembly=False,
    )


@pytest.mark.parametrize(
    "sequence",
    [PROTEIN, ["A", "C", "G", "U"], ["DA", "DC", "DG", "DT"]],
    ids=["protein", "rna", "dna"],
)
@pytest.mark.parametrize(
    "gaps", ["complete", "opposite_ends", "internal", "first_absent"]
)
def test_each_subchain_has_its_own_alignment(
    tmp_path: Path, mols: dict[str, Chem.Mol], sequence: list[str], gaps: str
) -> None:
    full = list(range(1, len(sequence) + 1))
    coverage = {
        "complete": {"A": full, "B": full},
        "opposite_ends": {"A": full[1:], "B": full[:-1]},
        "internal": {"A": full[:1] + full[2:], "B": full[:-2] + full[-1:]},
        "first_absent": {"A": [], "B": full},
    }[gaps]
    path = write_pdb(tmp_path / "input.pdb", sequence, coverage)
    parsed = parse_pdb(path, mols=mols, use_assembly=False)
    assert_coverage(parsed, sequence, coverage)
    assert_coverage(round_trip(parsed, mols), sequence, coverage)


@pytest.mark.parametrize("first_shorter", [False, True])
def test_generated_pdb_without_seqres(
    tmp_path: Path, mols: dict[str, Chem.Mol], first_shorter: bool
) -> None:
    full = list(range(1, len(PROTEIN) + 1))
    coverage = {
        "A": full[1:] if first_shorter else full,
        "B": full if first_shorter else full[2:],
    }
    path = write_pdb(tmp_path / "input.pdb", PROTEIN, coverage, seqres=False)
    parsed = parse_pdb(path, mols=mols, use_assembly=False)
    assert_coverage(parsed, PROTEIN, coverage, seqres=False)
    assert_coverage(round_trip(parsed, mols), PROTEIN, coverage, seqres=False)


@pytest.mark.parametrize("alternate_chain", ["A", "B"])
@pytest.mark.parametrize(
    ("seqres", "alternate_identity"), [(True, True), (False, True), (True, False)]
)
def test_conformer_selection_precedes_alignment(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    alternate_chain: str,
    seqres: bool,
    alternate_identity: bool,
) -> None:
    coverage = {"A": list(range(1, 7)), "B": list(range(1, 7))}
    path = write_pdb(
        tmp_path / "input.pdb",
        PROTEIN,
        coverage,
        seqres=seqres,
        alternate_chain=alternate_chain,
        alternate_identity=alternate_identity,
    )
    parsed = parse_pdb(path, mols=mols, use_assembly=False)
    assert_coverage(parsed, PROTEIN, coverage)
    assert_coverage(round_trip(parsed, mols), PROTEIN, coverage)


@pytest.mark.parametrize("seqres", [True, False])
def test_all_models_receive_labels(
    tmp_path: Path, mols: dict[str, Chem.Mol], seqres: bool
) -> None:
    coverage = {"A": [2, 3, 4, 6], "B": [1, 2, 3, 5]}
    path = write_pdb(
        tmp_path / "input.pdb",
        PROTEIN,
        coverage,
        seqres=seqres,
        models=2,
        alternate_chain="B",
    )
    parsed = parse_pdb(path, mols=mols, use_assembly=False)
    assert_coverage(parsed, PROTEIN, coverage, seqres=seqres)
    assert len(parsed.data.ensemble) == 2
    present = parsed.data.atoms["is_present"]
    for model, ensemble in enumerate(parsed.data.ensemble):
        coords = parsed.data.coords["coords"][
            ensemble["atom_coord_idx"] : ensemble["atom_coord_idx"]
            + ensemble["atom_num"]
        ]
        np.testing.assert_allclose(coords[present, 2], model * 0.2)
        np.testing.assert_array_equal(coords[~present], 0)
    # The writer exports the reference model; parsing must retain both models above.
    assert_coverage(round_trip(parsed, mols), PROTEIN, coverage, seqres=seqres)


def test_models_with_different_atom_presence_still_fail(
    tmp_path: Path, mols: dict[str, Chem.Mol]
) -> None:
    path = write_pdb(
        tmp_path / "input.pdb", PROTEIN, {"A": list(range(1, 7))}, models=2
    )
    lines = path.read_text().splitlines()
    # An ensemble has one shared atom-presence mask. A model missing the final
    # atom cannot be represented and must not silently inherit model 1's mask.
    last_atom = max(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    del lines[last_atom]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError):
        parse_pdb(path, mols=mols, use_assembly=False)


@pytest.mark.parametrize("use_original_res_idx", [False, True])
def test_missing_residue_indices_are_consistently_zero_based(
    tmp_path: Path, mols: dict[str, Chem.Mol], use_original_res_idx: bool
) -> None:
    coverage = {"A": [2, 4, 5], "B": [1, 2, 4, 6]}
    path = write_pdb(tmp_path / "input.pdb", PROTEIN, coverage)
    parsed = parse_pdb(
        path, mols=mols, use_assembly=False, use_original_res_idx=use_original_res_idx
    )
    assert_coverage(parsed, PROTEIN, coverage)
    assert_coverage(round_trip(parsed, mols), PROTEIN, coverage)


@pytest.mark.parametrize("input_format", ["pdb", "mmcif"])
@pytest.mark.parametrize("use_assembly", [False, True])
def test_assembly_expansion_keeps_water_references_until_expanded(
    tmp_path: Path, mols: dict[str, Chem.Mol], input_format: str, use_assembly: bool
) -> None:
    coverage = {"A": list(range(1, 7))}
    path = write_pdb(
        tmp_path / "input.pdb", PROTEIN, coverage, assembly=True, water=True
    )
    if input_format == "pdb":
        parsed = parse_pdb(path, mols=mols, use_assembly=use_assembly)
    else:
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()
        structure.assign_label_seq_id()
        parsed = mmcif_from_block(
            structure.make_mmcif_block(), mols, use_assembly=use_assembly
        )
    copies = 2 if use_assembly else 1
    assert len(parsed.data.chains) == copies
    assert (
        parsed.data.chains["mol_type"].tolist()
        == [const.chain_type_ids["PROTEIN"]] * copies
    )
    assert parsed.data.residues["is_present"].all()
    assert "HOH" not in parsed.data.residues["name"]
    coords = parsed.data.coords["coords"].reshape(copies, -1, 3)
    if use_assembly:
        np.testing.assert_allclose(
            coords[1] - coords[0],
            np.tile([40.0, 0.0, 0.0], (coords.shape[1], 1)),
            atol=0.001,
        )
    reloaded = round_trip(parsed, mols)
    np.testing.assert_array_equal(reloaded.data.residues, parsed.data.residues)
    np.testing.assert_allclose(
        reloaded.data.coords["coords"], parsed.data.coords["coords"], atol=0.001
    )


def test_yaml_design_mask_survives_mmcif_export(
    tmp_path: Path, mols: dict[str, Chem.Mol]
) -> None:
    path = write_pdb(
        tmp_path / "input.pdb", PROTEIN, {"A": [2, 3, 4, 5, 6], "B": [1, 2, 3, 4, 5]}
    )
    definition = {
        "entities": [
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": "A"}}, {"chain": {"id": "B"}}],
                    "design": [{"chain": {"id": "B"}}],
                }
            }
        ]
    }
    target = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "test", definition, mols, tmp_path, base_file_path=tmp_path
    )
    tokens = Tokenizer().tokenize(target.structure)
    design_mask = target.design_info.res_design_mask[tokens.token_to_res].astype(bool)
    reloaded = mmcif_from_block(
        gemmi.cif.read_string(to_mmcif(target.structure)).sole_block(),
        mols,
        use_assembly=False,
    )
    reloaded_tokens = Tokenizer().tokenize(reloaded.data)
    assert len(tokens.tokens) == len(reloaded_tokens.tokens) == 10
    assert design_mask.sum() == 5
    np.testing.assert_array_equal(reloaded_tokens.tokens["asym_id"][design_mask], 1)


@pytest.mark.parametrize("input_format", ["pdb", "mmcif"])
@pytest.mark.parametrize("model_counts", [(1, 1), (2, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("crop", [False, True])
def test_yaml_uses_reference_model_before_combining_files(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    input_format: str,
    model_counts: tuple[int, int],
    crop: bool,
) -> None:
    entities = []
    for chain, count, shift in zip(("A", "B"), model_counts, (0.0, 100.0)):
        path = write_pdb(
            tmp_path / f"{chain}.pdb",
            PROTEIN,
            {chain: list(range(1, 7))},
            models=count,
        )
        lines = [
            line[:30] + f"{float(line[30:38]) + shift:8.3f}" + line[38:]
            if line.startswith("ATOM")
            else line
            for line in path.read_text().splitlines()
        ]
        path.write_text("\n".join(lines) + "\n")
        selected_chain = chain
        if input_format == "mmcif":
            raw = gemmi.read_structure(str(path))
            raw.setup_entities()
            raw.assign_label_seq_id()
            selected_chain = raw[0][0].get_polymer().subchain_id()
            path = path.with_suffix(".cif")
            raw.make_mmcif_document().write_file(str(path))
        selection = {"id": selected_chain}
        if crop:
            selection["res_index"] = "2..5"
        entities.append(
            {"file": {"path": str(path), "include": [{"chain": selection}]}}
        )

    parser = YamlDesignParser(tmp_path)
    # Repeating the public operation also exercises cached parsed ensembles.
    for _ in range(2):
        target = parser.parse_boltzgen_schema(
            "reference", {"entities": entities}, mols, tmp_path, base_file_path=tmp_path
        )
        structure = target.structure
        assert structure.ensemble.tolist() == [(0, len(structure.atoms))]
        contexts = target.source_context["chains"]
        assert [entry["source_chain"] for entry in contexts] == (
            structure.chains["name"].tolist()
        )
        for entry in contexts:
            assert entry["complete"]
            assert entry["residue_names"] == PROTEIN
            assert entry["indices"] == list(range(1, 5) if crop else range(6))
        assert len(structure.coords) == len(structure.atoms)
        np.testing.assert_allclose(
            structure.coords["coords"], structure.atoms["coords"]
        )
        tokens = Tokenizer().tokenize(structure).tokens
        first_b_center = tokens["center_coords"][tokens["asym_id"] == 1][0]
        np.testing.assert_allclose(
            first_b_center, [100.7 + (3.8 if crop else 0.0), 0.6, 0.0], atol=0.001
        )
        reloaded = mmcif_from_block(
            gemmi.cif.read_string(to_mmcif(structure)).sole_block(),
            mols,
            use_assembly=False,
        )
        np.testing.assert_allclose(
            reloaded.data.coords["coords"], structure.coords["coords"], atol=0.001
        )


@pytest.mark.parametrize("chain", ["A", "B"])
@pytest.mark.parametrize(
    ("coordinates", "seqres"),
    [
        (PROTEIN, ["VAL", *PROTEIN[1:]]),
        (PROTEIN, [*PROTEIN[:2], "VAL", *PROTEIN[3:]]),
        (PROTEIN, [*PROTEIN[:-1], "VAL"]),
        (
            ["ALA", "GLY", "GLY", "GLY", "GLY", "GLY"],
            ["ALA", "SER", "GLY", "GLY", "GLY", "GLY"],
        ),
        (PROTEIN, PROTEIN[1:]),
        (PROTEIN, [*PROTEIN[:2], *PROTEIN[3:]]),
        (PROTEIN, PROTEIN[:-1]),
    ],
    ids=[
        "first_mismatch",
        "middle_mismatch",
        "last_mismatch",
        "silent_shift",
        "first_insertion",
        "middle_insertion",
        "last_insertion",
    ],
)
def test_inconsistent_seqres_fails_before_assigning_wrong_labels(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    chain: str,
    coordinates: list[str],
    seqres: list[str],
) -> None:
    """Invalid SEQRES must not silently shift or drop coordinate residues."""
    path = write_pdb(
        tmp_path / "inconsistent.pdb",
        coordinates,
        {"A": list(range(1, 7)), "B": list(range(1, 7))},
    )
    lines = path.read_text().splitlines()
    names = " ".join(f"{name:>3}" for name in seqres)
    path.write_text(
        "\n".join(
            f"SEQRES   1 {chain} {len(seqres):4d}  {names}"
            if line.startswith("SEQRES") and line[11] == chain
            else line
            for line in lines
        )
        + "\n"
    )
    with pytest.raises(ValueError, match=rf"subchain {chain}.*SEQRES"):
        parse_pdb(path, mols=mols, use_assembly=False)


@pytest.mark.parametrize("input_format", ["pdb", "mmcif"])
@pytest.mark.parametrize(
    ("positions", "selection", "expected_a"),
    [
        ([3, 4], "1..2", []),
        ([3, 4], "5..6", []),
        ([3, 4], "1..4", [3, 4]),
        ([1, 6], "1..6", [1, 2, 3, 4, 5, 6]),
    ],
    ids=["unresolved_leading", "unresolved_trailing", "partial", "internal_gap"],
)
def test_yaml_removes_only_unresolved_chain_ends(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    input_format: str,
    positions: list[int],
    selection: str,
    expected_a: list[int],
) -> None:
    """An entirely unresolved selection is empty; internal gaps stay in place."""
    path = write_pdb(
        tmp_path / "missing.pdb", PROTEIN, {"A": positions, "B": list(range(1, 7))}
    )
    chain_ids = ["A", "B"]
    if input_format == "mmcif":
        raw = gemmi.read_structure(str(path))
        raw.setup_entities()
        raw.assign_label_seq_id()
        chain_ids = [chain.get_polymer().subchain_id() for chain in raw[0]]
        path = path.with_suffix(".cif")
        raw.make_mmcif_document().write_file(str(path))
    definition = {
        "entities": [
            {
                "file": {
                    "path": str(path),
                    "include": [
                        {"chain": {"id": chain_ids[0], "res_index": selection}},
                        {"chain": {"id": chain_ids[1]}},
                    ],
                    "design": [{"chain": {"id": chain_ids[0]}}],
                }
            }
        ]
    }
    parser = YamlDesignParser(tmp_path)
    for _ in range(2):
        target = parser.parse_boltzgen_schema(
            "missing", definition, mols, tmp_path, base_file_path=tmp_path
        )
        structure = target.structure
        assert structure.chains["name"].tolist() == (
            chain_ids if expected_a else chain_ids[1:]
        )
        contexts = target.source_context["chains"]
        assert [entry["source_chain"] for entry in contexts] == (
            structure.chains["name"].tolist()
        )
        assert all(entry["residue_names"] == PROTEIN for entry in contexts)
        assert contexts[0]["indices"] == (
            [position - 1 for position in expected_a] if expected_a else list(range(6))
        )
        assert len(structure.residues) == len(expected_a) + 6
        assert target.design_info.res_design_mask.sum() == len(expected_a)
        assert structure.residues["res_idx"][: len(expected_a)].tolist() == [
            position - 1 for position in expected_a
        ]
        assert structure.residues["is_present"][: len(expected_a)].tolist() == [
            position in positions for position in expected_a
        ]
        tokens = Tokenizer().tokenize(structure).tokens
        assert len(tokens) == len(expected_a) + 6
        np.testing.assert_allclose(tokens["center_coords"][-6:, 1], 15.6, atol=0.001)


@pytest.mark.parametrize("selection", ["1..2", "5..6"])
def test_yaml_reports_empty_selection(
    tmp_path: Path, mols: dict[str, Chem.Mol], selection: str
) -> None:
    """Removing every unresolved selection gives a useful input error."""
    path = write_pdb(tmp_path / "empty.pdb", PROTEIN, {"A": [3, 4]})
    definition = {
        "entities": [
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": "A", "res_index": selection}}],
                }
            }
        ]
    }
    with pytest.raises(ValueError, match=r"No residues remain.*empty\.pdb"):
        YamlDesignParser(tmp_path).parse_boltzgen_schema(
            "empty", definition, mols, tmp_path, base_file_path=tmp_path
        )


@pytest.mark.parametrize("second_selection", ["1..4", "3..6", "5..8", "1,3,5,7"])
def test_equal_sequences_with_different_indices_survive_export(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    second_selection: str,
) -> None:
    """Identical cropped sequences may still need different entity label indices."""
    entities = []
    for chain, selection in [("A", "1..4"), ("B", second_selection)]:
        path = write_pdb(
            tmp_path / f"{chain}.pdb", ["GLY"] * 8, {chain: list(range(1, 9))}
        )
        entities.append(
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": chain, "res_index": selection}}],
                }
            }
        )
    target = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "indices", {"entities": entities}, mols, tmp_path, base_file_path=tmp_path
    )
    original = target.structure
    block = gemmi.cif.read_string(to_mmcif(original)).sole_block()
    reloaded = mmcif_from_block(block, mols, use_assembly=False).data
    assert reloaded.residues["is_present"].all()
    np.testing.assert_array_equal(
        reloaded.residues["res_idx"], original.residues["res_idx"]
    )
    np.testing.assert_array_equal(reloaded.residues["name"], original.residues["name"])
    np.testing.assert_allclose(
        reloaded.coords["coords"], original.coords["coords"], atol=0.001
    )
    entity_ids = set(block.find_values("_entity_poly_seq.entity_id"))
    assert len(entity_ids) == (1 if second_selection == "1..4" else 2)


@pytest.mark.parametrize("chain_names", [("A", "B"), ("A1", "A2")])
@pytest.mark.parametrize("design_coloring", [False, True])
def test_export_annotations_use_actual_renamed_chain_ids(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    chain_names: tuple[str, str],
    design_coloring: bool,
) -> None:
    """Sequence and confidence annotations must reference the emitted atom chains."""
    entities = []
    for chain, selection in zip(chain_names, ["1..4", "3..6"]):
        path = write_pdb(
            tmp_path / f"{chain}.pdb", ["GLY"] * 6, {"A": list(range(1, 7))}
        )
        raw = gemmi.read_structure(str(path))
        raw.setup_entities()
        raw.assign_label_seq_id()
        raw[0][0].name = chain
        for residue in raw[0][0]:
            residue.subchain = chain
        raw.entities[0].subchains = [chain]
        path = path.with_suffix(".cif")
        raw.make_mmcif_document().write_file(str(path))
        entities.append(
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": chain, "res_index": selection}}],
                }
            }
        )
    target = YamlDesignParser(tmp_path).parse_boltzgen_schema(
        "renamed", {"entities": entities}, mols, tmp_path, base_file_path=tmp_path
    )
    structure = target.structure
    assert structure.chains["name"].tolist() == list(chain_names)
    block = gemmi.cif.read_string(
        to_mmcif(
            structure,
            plddt_cols=not design_coloring,
            design_coloring=design_coloring,
            color_features=np.linspace(0.1, 0.8, len(structure.residues)),
        )
    ).sole_block()
    atom_keys = set(
        zip(
            block.find_values("_atom_site.label_asym_id"),
            block.find_values("_atom_site.label_seq_id"),
        )
    )
    scheme_keys = list(
        zip(
            block.find_values("_pdbx_poly_seq_scheme.asym_id"),
            block.find_values("_pdbx_poly_seq_scheme.seq_id"),
        )
    )
    qa_keys = list(
        zip(
            block.find_values("_ma_qa_metric_local.label_asym_id"),
            block.find_values("_ma_qa_metric_local.label_seq_id"),
        )
    )
    assert len(scheme_keys) == len(set(scheme_keys)) == len(atom_keys) == 8
    assert set(scheme_keys) == atom_keys
    assert len(qa_keys) == len(set(qa_keys)) == 8
    assert set(qa_keys) == atom_keys
    chain_entities = dict(
        zip(
            block.find_values("_struct_asym.id"),
            block.find_values("_struct_asym.entity_id"),
        )
    )
    for row in block.find(
        "_pdbx_poly_seq_scheme.", ["asym_id", "entity_id", "pdb_strand_id"]
    ):
        assert row[1] == chain_entities[row[0]]
        assert row[2] == row[0]
    reloaded = mmcif_from_block(block, mols, use_assembly=False).data
    assert reloaded.residues["is_present"].all()
    np.testing.assert_allclose(
        reloaded.coords["coords"], structure.coords["coords"], atol=0.001
    )


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("reset_indices", [False, True])
@pytest.mark.parametrize("explicit_second_msa", [False, True])
def test_removed_chain_does_not_leave_metadata_for_later_files(
    tmp_path: Path,
    mols: dict[str, Chem.Mol],
    metadata: bool,
    reset_indices: bool,
    explicit_second_msa: bool,
) -> None:
    """Dropped chains must not reserve MSA names or break index-reset requests."""
    first = write_pdb(
        tmp_path / "first.pdb", PROTEIN, {"A": [3, 4], "B": list(range(1, 7))}
    )
    second = write_pdb(tmp_path / "second.pdb", PROTEIN, {"A": list(range(1, 7))})
    selection = {"id": "A", "res_index": "1..2"}
    if metadata:
        selection.update({"msa": 0, "symmetric_group": 2})
    first_spec = {
        "path": str(first),
        "include": [{"chain": selection}, {"chain": {"id": "B"}}],
    }
    if reset_indices:
        first_spec["reset_res_index"] = [{"chain": {"id": "A"}}, {"chain": {"id": "B"}}]
    second_selection = {"id": "A"}
    if explicit_second_msa:
        second_selection["msa"] = 0
    definition = {
        "entities": [
            {"file": first_spec},
            {
                "file": {
                    "path": str(second),
                    "include": [{"chain": second_selection}],
                    "msa": -1,
                }
            },
        ]
    }
    parser = YamlDesignParser(tmp_path)
    for _ in range(2):
        target = parser.parse_boltzgen_schema(
            "metadata", definition, mols, tmp_path, base_file_path=tmp_path
        )
        assert target.structure.chains["name"].tolist() == ["B", "A"]
        contexts = target.source_context["chains"]
        assert [entry["source_chain"] for entry in contexts] == ["B", "A"]
        assert all(entry["residue_names"] == PROTEIN for entry in contexts)
        assert all(entry["indices"] == list(range(6)) for entry in contexts)
        assert target.structure.residues["is_present"].all()
        assert target.structure.chains["symmetric_group"].tolist() == [0, 0]
        msa_ids = {chain.chain_name: chain.msa_id for chain in target.record.chains}
        assert msa_ids["A"] == (0 if explicit_second_msa else -1)
        np.testing.assert_array_equal(
            target.structure.residues["res_idx"], list(range(6)) * 2
        )
