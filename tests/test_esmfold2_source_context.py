"""Full-source provenance through actual Boltz parsing, selection, and output."""

from copy import deepcopy
import json

import gemmi
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from boltzgen.data import source_context
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.esmfold2.score import validate_context


@pytest.fixture
def parser_and_mols(tmp_path):
    mols = {}
    for letter, name in [
        ("A", "ALA"),
        ("G", "GLY"),
        ("C", "CYS"),
        ("E", "GLU"),
        ("M", "MET"),
    ]:
        mol = Chem.MolFromSequence(letter)
        for atom in mol.GetAtoms():
            atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        mols[name] = mol
    return YamlDesignParser(tmp_path), mols


def test_full_mse_source_matches_parser_normalization(parser_and_mols, tmp_path):
    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "source",
        {"entities": [{"protein": {"id": "A", "sequence": "AMG"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    parsed.structure.atoms["coords"] = (
        np.arange(len(parsed.structure.atoms) * 3).reshape(-1, 3) + 1
    )
    parsed.structure.coords["coords"] = parsed.structure.atoms["coords"]
    path = tmp_path / "mse.cif"
    path.write_text(
        to_mmcif(parsed.structure).replace("MET", "MSE").replace(" SD ", " SE ")
    )
    result = parser.parse_boltzgen_schema(
        "mse",
        {"entities": [{"file": {"path": str(path)}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    validate_context(result.source_context)
    assert result.source_context["chains"][0]["residue_names"] == ["ALA", "MET", "GLY"]


def test_custom_smiles_request_preserves_atom_identities(parser_and_mols, tmp_path):
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "ligand",
        {
            "entities": [
                {"protein": {"id": "A", "sequence": "ACG"}},
                {"protein": {"id": "B", "sequence": "3"}},
                {"ligand": {"id": "L", "smiles": "OC(C)C"}},
            ],
            "constraints": [
                {"bond": {"atom1": ["A", 2, "SG"], "atom2": ["L", 1, "C1"]}}
            ],
        },
        mols,
        tmp_path,
        tmp_path,
    )
    path = tmp_path / "ligand.cif"
    path.write_text(to_mmcif(parsed.structure))
    tokenized = Tokenizer().tokenize(parsed.structure)
    request = make_request(
        {
            "str_gen": parsed.structure,
            "tokenized": tokenized,
            "source_context": json.dumps(parsed.source_context),
            "chain_design_mask": tokenized.tokens["asym_id"] == 1,
            "design_mask": tokenized.tokens["asym_id"] == 1,
            "extra_mols": parsed.extra_mols,
            "id": "ligand",
            "path": str(path),
        },
        {"seed": 0},
    )
    ligand = request["chains"][-1]
    serialized = Chem.MolFromSmiles(ligand["smiles"])
    center = ligand["smiles_atom_names"].index("C1")
    assert serialized.GetAtomWithIdx(center).GetDegree() == 3
    assert request["bonds"] == [["A", 1, "SG", "L", 0, "C1"]]


def test_ccd_lig_is_not_treated_as_custom_smiles(parser_and_mols, tmp_path):
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    # A small test CCD molecule exercises the CCD path with the legitimate code
    # LIG. The molecule is supplied by CCD, rather than custom SMILES metadata.
    mols["LIG"] = Chem.Mol(mols["ALA"])
    parsed = parser.parse_boltzgen_schema(
        "ccd",
        {
            "entities": [
                {"protein": {"id": "A", "sequence": "AGC"}},
                {"protein": {"id": "B", "sequence": "3"}},
                {"ligand": {"id": "L", "ccd": "LIG"}},
            ]
        },
        mols,
        tmp_path,
        tmp_path,
    )
    path = tmp_path / "ccd.cif"
    path.write_text(to_mmcif(parsed.structure))
    tokenized = Tokenizer().tokenize(parsed.structure)
    request = make_request(
        {
            "str_gen": parsed.structure,
            "tokenized": tokenized,
            "source_context": json.dumps(parsed.source_context),
            "chain_design_mask": tokenized.tokens["asym_id"] == 1,
            "design_mask": tokenized.tokens["asym_id"] == 1,
            "extra_mols": parsed.extra_mols,
            "id": "ccd",
            "path": str(path),
        },
        {"seed": 0},
    )
    assert request["chains"][-1]["residue_names"] == ["LIG"]
    assert "smiles" not in request["chains"][-1]


def test_crop_keeps_full_context_after_renumbering_and_sequence_changes(
    parser_and_mols, tmp_path
):
    parser, mols = parser_and_mols
    target = parser.parse_boltzgen_schema(
        "full",
        {"entities": [{"protein": {"id": "A", "sequence": "AGCAG"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    target.structure.atoms["coords"] = (
        np.arange(len(target.structure.atoms) * 3).reshape(-1, 3) + 1
    )
    target.structure.coords["coords"] = target.structure.atoms["coords"]
    path = tmp_path / "full.cif"
    path.write_text(to_mmcif(target.structure))
    schema = {
        "entities": [
            {
                "file": {
                    "path": str(path),
                    "include": [{"chain": {"id": "A", "res_index": "1,3,5"}}],
                    "reset_res_index": [{"chain": {"id": "A"}}],
                }
            },
            {"protein": {"id": "B", "sequence": "3"}},
        ]
    }
    parsed = parser.parse_boltzgen_schema("crop", schema, mols, tmp_path, tmp_path)
    context = parsed.source_context
    validate_context(context)
    assert context["chains"][0]["residue_names"] == ["ALA", "GLY", "CYS", "ALA", "GLY"]
    assert context["chains"][0]["indices"] == [0, 2, 4]
    assert parsed.structure.residues["res_idx"][:3].tolist() == [0, 1, 2]
    parsed.structure.residues["name"][3:] = "ALA"
    updated = source_context.update_designed(context, parsed.structure)
    assert updated["chains"][1]["residue_names"] == ["ALA"] * 3
    assert updated["chains"][0]["indices"] == [0, 2, 4]
    np.savez(tmp_path / "design.npz", source_context=np.asarray(json.dumps(updated)))
    with np.load(tmp_path / "design.npz", allow_pickle=False) as metadata:
        loaded = json.loads(metadata["source_context"].item())
    assert loaded == updated
    # Another sequence-design round changes only sampled positions, never the
    # omitted portions of the target source chain.
    assert source_context.update_designed(loaded, parsed.structure) == updated


def test_missing_seqres_requires_explicit_full_sequence_and_mapping(
    parser_and_mols, tmp_path
):
    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "source",
        {"entities": [{"protein": {"id": "A", "sequence": "AGC"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    path = tmp_path / "unknown.pdb"
    # An ATOM-only file provides coordinates, not evidence of a complete chain.
    path.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\nEND\n"
    )
    context = source_context.from_file(parsed.structure, path, None)
    with pytest.raises(ValueError, match="Full sequence is unavailable"):
        validate_context(context)
    context = source_context.from_file(
        parsed.structure,
        path,
        [{"chain": {"id": "A", "sequence": "AAGGC", "source_res_indices": [2, 4, 5]}}],
    )
    validate_context(context)
    assert context["chains"][0]["indices"] == [1, 3, 4]
    with pytest.raises(ValueError, match="disagrees"):
        source_context.from_file(
            parsed.structure,
            path,
            [
                {
                    "chain": {
                        "id": "A",
                        "sequence": "AAAAA",
                        "source_res_indices": [2, 4, 5],
                    }
                }
            ],
        )
    for invalid in [1.9, True]:
        with pytest.raises(ValueError, match="integer positions"):
            source_context.from_file(
                parsed.structure,
                path,
                [
                    {
                        "chain": {
                            "id": "A",
                            "sequence": "AGC",
                            "source_res_indices": [invalid, 2, 3],
                        }
                    }
                ],
            )


def test_insertions_are_added_to_the_full_sequence(parser_and_mols, tmp_path):
    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "source",
        {"entities": [{"protein": {"id": "A", "sequence": "AGC"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    context = deepcopy(parsed.source_context)
    source_context.insert(context, "A", 1, 2)
    assert context["chains"][0]["residue_names"] == ["ALA", "GLY", "GLY", "GLY", "CYS"]
    assert context["chains"][0]["indices"] == [0, 1, 2, 3, 4]


def test_explicit_leaving_atoms_keep_source_positions(parser_and_mols, tmp_path):
    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "cyclized",
        {
            "entities": [{"protein": {"id": "C", "sequence": "2E"}}],
            "constraints": [
                {"bond": {"atom1": ["C", 1, "N"], "atom2": ["C", 3, "CD"]}}
            ],
            "leaving_atoms": [{"atom": ["C", 3, "OE2"]}],
        },
        mols,
        tmp_path,
        tmp_path,
    )
    assert parsed.source_context["chains"][0]["omitted_atoms"] == [[2, "OE2"]]


@pytest.mark.parametrize("inverse_fold", [False, True])
@pytest.mark.parametrize(
    "scaffold", [None, "spatial_design", "spatial_insertion", "replacement"]
)
def test_real_design_writer_and_generated_reader(
    parser_and_mols, tmp_path, inverse_fold, scaffold
):

    from types import SimpleNamespace
    import pickle
    import torch
    import yaml
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.task.predict.data_from_yaml import PredictionDataset, collate
    from boltzgen.task.predict.data_from_yaml import FromYamlDataModule
    from boltzgen.task.predict.data_from_generated import (
        FromGeneratedDataset,
        FromGeneratedDataModule,
        collate as collate_generated,
    )
    from boltzgen.task.predict.writer import DesignWriter
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    mols = {name: Chem.RemoveHs(mol) for name, mol in mols.items()}
    previous = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for name, mol in mols.items():
            (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(mol))
    finally:
        Chem.SetDefaultPickleProperties(previous)
    spec = tmp_path / "example.yaml"
    schema = {
        "entities": [
            {"protein": {"id": "A", "sequence": "AGCAG"}},
            {"protein": {"id": "B", "sequence": "3"}},
        ]
    }
    if scaffold:
        source = parser.parse_boltzgen_schema(
            "source",
            {
                "entities": [
                    {"protein": {"id": "A", "sequence": "AGCAG"}},
                    {"protein": {"id": "B", "sequence": "AGCAE"}},
                ]
            },
            mols,
            tmp_path,
            tmp_path,
        )
        source.structure.atoms["coords"] = (
            np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
        )
        source.structure.coords["coords"] = source.structure.atoms["coords"]
        path = tmp_path / "source.cif"
        path.write_text(to_mmcif(source.structure))
        schema = {
            "entities": [
                {
                    "file": {
                        "path": str(path),
                        "include": [
                            {"chain": {"id": "A", "res_index": "1,3,5"}},
                            {"chain": {"id": "B"}},
                        ],
                        "exclude": [{"chain": {"id": "B", "res_index": "2..3"}}],
                        "design": [{"chain": {"id": "B", "res_index": "2..3"}}],
                        "design_insertions": [
                            {
                                "insertion": {
                                    "id": "B",
                                    "res_index": 2,
                                    "num_residues": 1,
                                }
                            }
                        ],
                        "reset_res_index": [
                            {"chain": {"id": "A"}},
                            {"chain": {"id": "B"}},
                        ],
                    }
                }
            ],
        }
        if scaffold == "spatial_design":
            file = schema["entities"][0]["file"]
            del file["exclude"], file["design_insertions"]
            file["include"][1]["chain"]["res_index"] = "1,3,5"
            file["design"] = [{"chain": {"id": "B", "res_index": "3"}}]
        elif scaffold == "spatial_insertion":
            # An insertion on this chain must not delete a non-designable crop.
            schema["entities"][0]["file"]["design"] = [
                {"chain": {"id": "B", "res_index": "4"}}
            ]
        else:
            # The anchor is inside the exclusion, rather than at its first residue.
            schema["entities"][0]["file"]["design_insertions"][0]["insertion"][
                "res_index"
            ] = 3
    spec.write_text(yaml.safe_dump(schema))
    config = SimpleNamespace(
        yaml_path=str(spec),
        tokenizer=Tokenizer(),
        featurizer=Featurizer(),
        multiplicity=1,
    )
    dataset = PredictionDataset(config, mols, str(tmp_path), atom14=False)
    features = dataset[0]
    batch = collate([features])
    context_json = batch["source_context"]
    batch = FromYamlDataModule.transfer_batch_to_device(
        None, batch, torch.device("cpu")
    )
    assert batch["source_context"] is context_json
    prediction = dict(batch)
    prediction["coords"] = (
        torch.arange(batch["coords"][:, 0].numel(), dtype=torch.float32).reshape_as(
            batch["coords"][:, 0]
        )
        + 1
    )
    prediction["exception"] = False
    writer = DesignWriter(
        str(tmp_path / "generated"),
        res_atoms_only=False,
        atom14=False,
        inverse_fold=inverse_fold,
    )
    writer.write_on_batch_end(prediction=prediction, batch=batch, sample_id="example")
    assert writer.failed == 0
    output = tmp_path / "generated" / "example_0.cif"
    metadata_path = output.with_suffix(".npz")
    readback = FromGeneratedDataset(
        [output],
        [metadata_path],
        [output],
        tmp_path,
        mols,
        Tokenizer(),
        Featurizer(),
        extra_mol_dir=tmp_path / "generated" / "molecules",
        extra_features=["tokenized"],
    )
    feat = readback[0]
    refold_batch = collate_generated([feat])
    context_json = refold_batch["source_context"]
    transferred = FromGeneratedDataModule.transfer_batch_to_device(
        None, refold_batch, torch.device("cpu")
    )
    assert transferred["source_context"] is context_json
    request = make_request(feat, {})
    assert request["design_chains"] == ["B"]
    assert request["target_chains"] == ["A"]
    assert request["chains"][0]["residue_names"] == ["ALA", "GLY", "CYS", "ALA", "GLY"]
    if scaffold:
        assert request["chains"][0]["indices"] == [0, 2, 4]
        binder = request["chains"][1]
        if scaffold == "replacement":
            assert binder["residue_names"] == ["ALA", "GLY", "ALA", "GLU"]
            assert binder["indices"] == [0, 1, 2, 3]
            assert binder["replacement_sources"][0]["residue_names"] == [
                "ALA",
                "GLY",
                "GLY",
                "CYS",
                "ALA",
                "GLU",
            ]
            assert binder["replacement_sources"][0]["indices"] == [0, 2, 4, 5]
        elif scaffold == "spatial_insertion":
            # Excluded GLY/CYS stay in ESMC's source sequence alongside the
            # inserted residue; only the folding features use the selection.
            assert binder["residue_names"] == ["ALA", "GLY", "GLY", "CYS", "ALA", "GLU"]
            assert binder["indices"] == [0, 1, 4, 5]
        else:
            assert binder["residue_names"] == ["ALA", "GLY", "CYS", "ALA", "GLU"]
            assert binder["indices"] == [0, 2, 4]
    else:
        assert request["chains"][1]["residue_names"] == ["GLY"] * 3


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize(
    "anchors,not_design,expected_sequence,expected_indices",
    [
        ([2], None, "AGMEG", [0, 1, 2, 4]),
        ([3], None, "AGMEG", [0, 1, 2, 4]),
        ([4], None, "AGMEG", [0, 1, 2, 4]),
        ([2, 4], None, "AGGMEG", [0, 1, 2, 3, 5]),
        ([4, 2], None, "AGGMEG", [0, 1, 2, 3, 5]),
        ([3, 3], None, "AGGMEG", [0, 1, 2, 3, 5]),
        ([1], None, "GAGCAMEG", [0, 1, 5, 7]),
        ([5], None, "AGCAGMEG", [0, 4, 5, 7]),
        ([6, 1], None, "GAGCAMGEG", [0, 1, 5, 6, 8]),
        ([], None, "AGCAMEG", [0, 4, 6]),
        ([3], "3", "AGGCAMEG", [0, 2, 5, 7]),
    ],
)
def test_replacement_interval_and_spatial_crop_on_same_chain(
    parser_and_mols,
    tmp_path,
    reset,
    anchors,
    not_design,
    expected_sequence,
    expected_indices,
):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source",
        {"entities": [{"protein": {"id": "B", "sequence": "AGCAMEG"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    source.structure.atoms["coords"] = (
        np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    )
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    file = {
        "path": str(path),
        "exclude": [{"chain": {"id": "B", "res_index": "2..4,6"}}],
        "design": [{"chain": {"id": "B", "res_index": "2..5"}}],
        "design_insertions": [
            {"insertion": {"id": "B", "res_index": anchor, "num_residues": 1}}
            for anchor in anchors
        ],
    }
    if reset:
        file["reset_res_index"] = [{"chain": {"id": "B"}}]
    if not_design:
        file["not_design"] = [{"chain": {"id": "B", "res_index": not_design}}]
    parsed = parser.parse_boltzgen_schema(
        "test",
        {"entities": [{"file": file}]},
        mols,
        tmp_path,
        tmp_path,
    )
    context = parsed.source_context
    validate_context(context)
    chain = context["chains"][0]
    names = dict(A="ALA", G="GLY", C="CYS", M="MET", E="GLU")
    assert chain["residue_names"] == [names[aa] for aa in expected_sequence]
    assert chain["indices"] == expected_indices
    assert [
        chain["residue_names"][i] for i in chain["indices"]
    ] == parsed.structure.residues["name"].tolist()
    # A generated substitution updates its edited full-sequence position, while
    # the separate cropped GLU remains present and the mapping stays unchanged.
    parsed.structure.residues["name"][1] = "CYS"
    updated = source_context.update_designed(context, parsed.structure)["chains"][0]
    assert updated["residue_names"][expected_indices[1]] == "CYS"
    assert updated["residue_names"][-2] == "GLU"
    assert chain["residue_names"] == [names[aa] for aa in expected_sequence]


@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("mode", ["spatial", "replacement", "whole_chain"])
@pytest.mark.parametrize("anchors", [[3], [3, 2]])
def test_variable_length_insertions_preserve_context_and_mapping(
    parser_and_mols,
    tmp_path,
    count,
    mode,
    anchors,
):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source",
        {
            "entities": [
                {"protein": {"id": "A", "sequence": "AGC"}},
                {"protein": {"id": "B", "sequence": "AGCAG"}},
            ]
        },
        mols,
        tmp_path,
        tmp_path,
    )
    source.structure.atoms["coords"] = (
        np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    )
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    exclusion = {"id": "B"}
    if mode != "whole_chain":
        exclusion["res_index"] = "2..3"
    file = {
        "path": str(path),
        "exclude": [{"chain": exclusion}],
        "design": [{"chain": {"id": "B", "res_index": "4"}}]
        if mode == "spatial"
        else [{"chain": {"id": "B"}}],
        "design_insertions": [
            {"insertion": {"id": "B", "res_index": anchor, "num_residues": count}}
            for anchor in anchors
        ],
    }
    parsed = parser.parse_boltzgen_schema(
        "test",
        {"entities": [{"file": file}]},
        mols,
        tmp_path,
        tmp_path,
    )
    validate_context(parsed.source_context)
    entries = parsed.source_context["chains"]
    assert entries[0]["residue_names"] == ["ALA", "GLY", "CYS"]
    assert entries[0]["indices"] == [0, 1, 2]
    binder = entries[1]
    inserted = count * len(anchors)
    if mode == "whole_chain":
        assert binder["residue_names"] == ["GLY"] * inserted
        assert binder["indices"] == list(range(inserted))
    elif mode == "replacement":
        assert binder["residue_names"] == ["ALA", *["GLY"] * inserted, "ALA", "GLY"]
        assert binder["indices"] == list(range(inserted + 3))
    else:
        # One insertion before CYS; a second is before the original GLY.
        before_gly = count if len(anchors) == 2 else 0
        assert binder["residue_names"] == [
            "ALA",
            *["GLY"] * before_gly,
            "GLY",
            *["GLY"] * count,
            "CYS",
            "ALA",
            "GLY",
        ]
        assert binder["indices"] == [
            0,
            *range(1, 1 + before_gly),
            *range(2 + before_gly, 2 + before_gly + count),
            3 + inserted,
            4 + inserted,
        ]
        assert "replacement_sources" not in binder
    chain = parsed.structure.chains[1]
    start, end = chain["res_idx"], chain["res_idx"] + chain["res_num"]
    assert [binder["residue_names"][i] for i in binder["indices"]] == (
        parsed.structure.residues["name"][start:end].tolist()
    )


@pytest.mark.parametrize("fmt", ["pdb", "cif"])
@pytest.mark.parametrize("assembly", [False, True])
def test_original_entity_ids_map_to_declared_sequences_before_and_after_assembly(
    parser_and_mols, tmp_path, fmt, assembly
):
    path = tmp_path / f"assembly_source.{fmt}"
    # Preserve one undeclared chain beside the declared matching chain. When
    # converted to CIF, supply generated sequence B only to the real parser;
    # track the raw source declarations without synthesizing them.
    text = "SEQRES   1 A    3  ALA GLY CYS\n"
    serial = 0
    for chain in ["A", "B"]:
        for i, name in enumerate(["ALA", "GLY", "CYS"], 1):
            serial += 1
            text += f"ATOM  {serial:5d}  CA  {name:3} {chain:1}{i:4d}    {float(i):8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 20.00           C\n"
        text += "TER\n"
    seed = tmp_path / "seed.pdb"
    seed.write_text(text + "END\n")
    st = gemmi.read_structure(str(seed))
    st.setup_entities()
    if fmt == "cif":
        st.entities[1].full_sequence = ["ALA", "GLY", "CYS"]
    ass = gemmi.Assembly("1")
    gen = gemmi.Assembly.Gen()
    gen.chains = ["A", "B"]
    for name in ["1", "2"]:
        op = gemmi.Assembly.Operator()
        op.name = name
        if name == "2":
            op.transform.vec.x = 100
        gen.operators.append(op)
    ass.generators.append(gen)
    st.assemblies.append(ass)
    if fmt == "pdb":
        st.write_pdb(str(path))
    else:
        # PDB-to-CIF conversion normally assigns label_seq during Boltz parsing.
        for chain in st[0]:
            for i, residue in enumerate(chain, 1):
                residue.label_seq = i
        st.make_mmcif_document().write_file(str(path))
    raw = gemmi.read_structure(str(path))
    if fmt == "pdb":
        raw.setup_entities()
    raw_sequences = [list(e.full_sequence) for e in raw.entities]
    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "assembly",
        {"entities": [{"file": {"path": str(path), "use_assembly": assembly}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    assignments = [
        (str(c["name"]), int(c["entity_id"]), bool(raw_sequences[int(c["entity_id"])]))
        for c in parsed.structure.chains
    ]
    assert len(assignments) == (4 if assembly else 2)
    assert [a[2] for a in assignments] == (
        [True, True] if fmt == "cif" else [True, False]
    ) * (2 if assembly else 1)

    assert [c["complete"] for c in parsed.source_context["chains"]] == [
        a[2] for a in assignments
    ]


@pytest.mark.parametrize("chain_count", [1, 2, 3])
def test_redesign_request_accepts_all_designed_chains(
    parser_and_mols, tmp_path, chain_count
):
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "redesign",
        {
            "entities": [
                {"protein": {"id": chain, "sequence": "AGC"}}
                for chain in "ABC"[:chain_count]
            ]
        },
        mols,
        tmp_path,
        tmp_path,
    )
    path = tmp_path / "redesign.cif"
    path.write_text(to_mmcif(parsed.structure))
    tokenized = Tokenizer().tokenize(parsed.structure)
    feat = dict(
        str_gen=parsed.structure,
        tokenized=tokenized,
        source_context=json.dumps(parsed.source_context),
        chain_design_mask=np.ones(len(tokenized.tokens), dtype=bool),
        design_mask=np.ones(len(tokenized.tokens), dtype=bool),
        id="redesign",
        path=str(path),
    )
    with pytest.raises(ValueError, match="separate target"):
        make_request(feat, {})
    request = make_request(feat, {}, scoring_mode="redesign")
    assert request["scoring_mode"] == "redesign"
    assert request["design_chains"] == list("ABC"[:chain_count])
    assert request["target_chains"] == []
    with pytest.raises(ValueError, match="every polymer"):
        make_request(feat, {}, target_chains=["A"], scoring_mode="redesign")


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_symmetric_replacements_share_lengths_in_original_coordinates(
    parser_and_mols, tmp_path, seed
):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source",
        {"entities": [{"protein": {"id": ["B", "C"], "sequence": "AGCAG"}}]},
        mols,
        tmp_path,
        tmp_path,
    )
    source.structure.atoms["coords"] = (
        np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    )
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    # The same original sites are visited in opposite orders on the two chains.
    file = {
        "path": str(path),
        "include": [{"chain": {"id": c, "symmetric_group": 1}} for c in "BC"],
        "exclude": [{"chain": {"id": c, "res_index": "2..3"}} for c in "BC"],
        "design": [{"chain": {"id": c}} for c in "BC"],
        "design_insertions": [
            {
                "insertion": {
                    "id": c,
                    "res_index": pos,
                    "num_residues": "2..5",
                    "secondary_structure": "HELIX" if pos == 2 else "SHEET",
                }
            }
            for c, pos in [("B", 3), ("C", 2), ("B", 2), ("C", 3)]
        ],
    }
    np.random.seed(seed)
    parsed = parser.parse_boltzgen_schema(
        "test", {"entities": [{"file": file}]}, mols, tmp_path, tmp_path
    )
    left, right = parsed.source_context["chains"]
    assert left["residue_names"] == right["residue_names"]
    assert (
        left["indices"] == right["indices"] == list(range(len(left["residue_names"])))
    )
    count = parsed.structure.chains["res_num"][0]
    assert np.array_equal(
        parsed.design_info.res_ss_types[:count], parsed.design_info.res_ss_types[count:]
    )


@pytest.mark.parametrize("selection", ["1..3", "1..2,5..6"])
def test_explicit_fusion_uses_assembled_sequence(parser_and_mols, tmp_path, selection):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source", {"entities": [{"protein": {"id": "A", "sequence": "AGCMEAGCME"}}]},
        mols, tmp_path, tmp_path,
    )
    source.structure.atoms["coords"] = np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    fragment = {"file": {"path": str(path), "include": [{"chain": {"id": "A", "res_index": selection}}]}}
    cropped = parser.parse_boltzgen_schema("crop", {"entities": [fragment]}, mols, tmp_path, tmp_path)
    full_names = source.source_context["chains"][0]["residue_names"]
    assert cropped.source_context["chains"][0]["residue_names"] == full_names
    fused = parser.parse_boltzgen_schema(
        "fusion", {"entities": [fragment, {"protein": {"id": "L", "fuse": "A", "sequence": "GG"}}]},
        mols, tmp_path, tmp_path,
    )
    validate_context(fused.source_context)
    context = fused.source_context["chains"][0]
    expected = cropped.structure.residues["name"].tolist() + ["GLY", "GLY"]
    assert context["residue_names"] == expected == fused.structure.residues["name"].tolist()
    assert context["indices"] == list(range(len(expected)))
    assert context["context_mode"] == "fused_construct"
    # A declared fusion does not make an undeclared/incomplete source trustworthy.
    incomplete = deepcopy(cropped.source_context)
    incomplete["chains"][0]["complete"] = False
    joined = source_context.merge(incomplete, source_context.from_structure(source.structure), cropped.structure, "A")
    with pytest.raises(ValueError, match="Full sequence is unavailable"):
        validate_context(joined)


@pytest.mark.parametrize("selection", ["1..3", "3..5", "1,3,5"])
@pytest.mark.parametrize("atom14", [False, True])
@pytest.mark.parametrize("inverse_fold", [False, True])
def test_cropped_fusion_survives_writer_and_generated_reader(
    parser_and_mols, tmp_path, selection, atom14, inverse_fold
):
    from types import SimpleNamespace
    import pickle
    import torch
    import yaml
    from boltzgen.data import const
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.predict.data_from_yaml import PredictionDataset, collate
    from boltzgen.task.predict.data_from_generated import FromGeneratedDataset
    from boltzgen.task.predict.writer import DesignWriter
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    mols = {name: Chem.RemoveHs(mol) for name, mol in mols.items()}
    previous = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for name, mol in mols.items():
            (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(mol))
    finally:
        Chem.SetDefaultPickleProperties(previous)
    source = parser.parse_boltzgen_schema(
        "source", {"entities": [{"protein": {"id": "A", "sequence": "AGCAG"}}]},
        mols, tmp_path, tmp_path,
    )
    source.structure.atoms["coords"] = np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    fragment = {"file": {"path": str(path), "include": [{"chain": {"id": "A", "res_index": selection}}]}}
    cropped = parser.parse_boltzgen_schema("crop", {"entities": [fragment]}, mols, tmp_path, tmp_path)
    # Ordinary crops keep their original indices and complete source sequence.
    assert cropped.source_context["chains"][0]["residue_names"] == source.structure.residues["name"].tolist()
    original_indices = cropped.structure.residues["res_idx"].tolist()
    expected = cropped.structure.residues["name"].tolist() + ["GLY", "GLY", "ALA", "CYS"]
    spec = tmp_path / "fusion.yaml"
    spec.write_text(yaml.safe_dump({"entities": [
        fragment,
        {"protein": {"id": "L", "fuse": "A", "sequence": "GG"}},
        {"file": {"path": str(path), "fuse": "A", "include": [{"chain": {"id": "A", "res_index": "1,3"}}]}},
        {"protein": {"id": "B", "sequence": "3"}},
    ]}))
    config = SimpleNamespace(yaml_path=str(spec), tokenizer=Tokenizer(), featurizer=Featurizer(), multiplicity=1)
    features = PredictionDataset(config, mols, str(tmp_path), atom14=atom14)[0]
    indices = features["residue_index"][features["asym_id"] == 0].tolist()
    assert indices[:len(original_indices)] == original_indices
    # Repeated fusion must append after the last actual index, including gaps.
    last = original_indices[-1]
    assert indices == original_indices + [last + 1, last + 2, last + 3, last + 5]
    assert len(set(indices)) == len(expected)
    batch = collate([features])
    prediction = dict(batch)
    prediction["coords"] = torch.arange(batch["coords"][:, 0].numel(), dtype=torch.float32).reshape_as(batch["coords"][:, 0]) + 1
    if atom14:
        # Encode GLY with all ten unused atom slots placed on its backbone O.
        atom_design_mask = batch["design_mask"][0].bool()[batch["atom_to_token"][0].int().argmax(-1)]
        atom_design_mask &= batch["atom_pad_mask"][0].bool()
        coords = prediction["coords"][0, atom_design_mask].reshape(-1, 14, 3)
        coords[:, 4:] = coords[:, 3:4]
        prediction["coords"][0, atom_design_mask] = coords.reshape(-1, 3)
    prediction["exception"] = False
    generated = tmp_path / "generated"
    writer = DesignWriter(str(generated), res_atoms_only=False, atom14=atom14, inverse_fold=inverse_fold)
    writer.write_on_batch_end(prediction=prediction, batch=batch, sample_id="fusion")
    output = generated / "fusion_0.cif"
    assert writer.failed == 0
    assert output.with_suffix(".npz").is_file()
    feat = FromGeneratedDataset(
        [output], [output.with_suffix(".npz")], [output], tmp_path, mols,
        Tokenizer(), Featurizer(), extra_mol_dir=generated / const.molecules_dirname,
        extra_features=["tokenized"],
    )[0]
    request = make_request(feat, {})
    target, binder = request["chains"]
    assert target["residue_names"] == expected
    assert target["indices"] == list(range(len(expected)))
    assert target["context_mode"] == "fused_construct"
    assert binder["residue_names"] == ["GLY"] * 3
    assert request["design_chains"] == ["B"]
    assert request["target_chains"] == ["A"]


def test_fusion_reassembles_source_around_linker(parser_and_mols, tmp_path):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source", {"entities": [{"protein": {"id": "A", "sequence": "AGCMEAGCME"}}]},
        mols, tmp_path, tmp_path,
    )
    source.structure.atoms["coords"] = np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    entities = [
        {"file": {"path": str(path), "include": [{"chain": {"id": "A", "res_index": "1..3"}}]}},
        {"protein": {"id": "L", "fuse": "A", "sequence": "GG"}},
        {"file": {"path": str(path), "fuse": "A", "include": [{"chain": {"id": "A", "res_index": "4.."}}]}},
    ]
    fused = parser.parse_boltzgen_schema("fusion", {"entities": entities}, mols, tmp_path, tmp_path)
    validate_context(fused.source_context)
    context = fused.source_context["chains"][0]
    full = source.source_context["chains"][0]["residue_names"]
    assert context["residue_names"] == full[:3] + ["GLY", "GLY"] + full[3:]
    assert context["indices"] == list(range(12))
    assert context["context_mode"] == "fused_construct"


@pytest.mark.parametrize("reverse", [False, True])
def test_cropped_file_bonds_survive_mmcif_roundtrip(parser_and_mols, tmp_path, reverse):
    parser, mols = parser_and_mols
    source = parser.parse_boltzgen_schema(
        "source", {"entities": [{"protein": {"id": ["A", "B"], "sequence": "AGCGAC"}}]},
        mols, tmp_path, tmp_path,
    )
    source.structure.atoms["coords"] = np.arange(len(source.structure.atoms) * 3).reshape(-1, 3) + 1
    source.structure.coords["coords"] = source.structure.atoms["coords"]
    path = tmp_path / "source.cif"
    path.write_text(to_mmcif(source.structure))
    endpoints = [["A", 3, "SG"], ["B", 6, "SG"]]
    if reverse:
        endpoints.reverse()
    spec = {"entities": [{"file": {"path": str(path), "include": [
        {"chain": {"id": "A", "res_index": "3..6"}},
        {"chain": {"id": "B", "res_index": "5..6"}},
    ]}}], "constraints": [{"bond": dict(zip(("atom1", "atom2"), endpoints))}]}
    parsed = parser.parse_boltzgen_schema("crop", spec, mols, tmp_path, tmp_path)
    assert len(parsed.structure.bonds) == 1
    bond = parsed.structure.bonds[0]
    assert {int(bond["res_1"]), int(bond["res_2"])} == {0, 5}
    for endpoint in (1, 2):
        residue = parsed.structure.residues[bond[f"res_{endpoint}"]]
        assert residue["name"] == "CYS"
        assert residue["atom_idx"] <= bond[f"atom_{endpoint}"] < residue["atom_idx"] + residue["atom_num"]
    saved = tmp_path / "saved.cif"
    saved.write_text(to_mmcif(parsed.structure))
    reloaded = parser.parse_boltzgen_schema(
        "reloaded", {"entities": [{"file": {"path": str(saved)}}]}, mols, tmp_path, tmp_path,
    )
    assert len(reloaded.structure.bonds) == 1
    restored = reloaded.structure.bonds[0]
    for endpoint in (1, 2):
        assert reloaded.structure.residues[restored[f"res_{endpoint}"]]["name"] == "CYS"
        assert reloaded.structure.atoms[restored[f"atom_{endpoint}"]]["name"] == "SG"


def test_shipped_4g37_preserves_complete_deposited_sequence():
    from hashlib import sha256
    from pathlib import Path

    path = Path(__file__).parents[1] / "example/small_molecule_from_file_and_smiles/4g37.pdb"
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    sequence = list(structure.entities[0].full_sequence)
    # PDB SEQRES is fixed-width: stripping its padding corrupts the final row
    # in Gemmi 0.6.5. This is the deposited 4G37 sequence, including missing residues.
    assert len(sequence) == 555
    assert sha256(" ".join(sequence).encode()).hexdigest() == (
        "bb1960c8759d16b2122962e79992ac1178f90309e62f5d8eed4544c9d99db06b"
    )


def test_covalently_linked_target_remains_a_scoring_partner(parser_and_mols, tmp_path):
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.task.predict.data_from_generated import FromGeneratedDataset
    from boltzgen.task.esmfold2.score import make_request

    parser, mols = parser_and_mols
    parsed = parser.parse_boltzgen_schema(
        "linked", {"entities": [
            {"protein": {"id": "A", "sequence": "ACG"}},
            {"protein": {"id": "B", "sequence": "1C1"}},
        ], "constraints": [{"bond": {"atom1": ["A", 2, "SG"], "atom2": ["B", 2, "SG"]}}]},
        mols, tmp_path, tmp_path,
    )
    parsed.structure.atoms["coords"] = np.arange(len(parsed.structure.atoms) * 3).reshape(-1, 3) + 1
    parsed.structure.coords["coords"] = parsed.structure.atoms["coords"]
    path = tmp_path / "linked.cif"
    path.write_text(to_mmcif(parsed.structure))
    tokenized = Tokenizer().tokenize(parsed.structure)
    design_mask = parsed.design_info.res_design_mask[tokenized.token_to_res]
    import pickle

    previous = Chem.GetDefaultPickleProperties()
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for name, molecule in mols.items():
            (tmp_path / f"{name}.pkl").write_bytes(pickle.dumps(Chem.RemoveHs(molecule)))
    finally:
        Chem.SetDefaultPickleProperties(previous)
    extra_mols = tmp_path / "extra"
    extra_mols.mkdir()
    np.savez(path.with_suffix(".npz"), design_mask=design_mask, source_context=json.dumps(parsed.source_context))
    dataset = FromGeneratedDataset(
        [path], [path.with_suffix(".npz")], [path], tmp_path, mols,
        Tokenizer(), Featurizer(), extra_mol_dir=extra_mols, extra_features=["tokenized"],
    )
    feat = dataset.getitem_from_paths(path.with_suffix(".npz"), path, path)
    # Refolding includes the whole linked construct, but only B is designed.
    assert feat["chain_design_mask"].all()
    assert not feat["design_mask"][:3].any()
    request = make_request(feat, {})
    assert request["design_chains"] == ["B"]
    assert request["target_chains"] == ["A"]
    assert request["bonds"] == [["A", 1, "SG", "B", 1, "SG"]]
