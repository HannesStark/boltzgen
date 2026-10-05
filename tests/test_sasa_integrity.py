"""SASA integration through parsing, featurization, refold export and analysis."""

# ruff: noqa: INP001

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import biotite.structure as bst
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor

from boltzgen.data import const
from boltzgen.data.data import (
    Atom,
    Bond,
    Chain,
    Coords,
    Ensemble,
    Input,
    Interface,
    Residue,
    Structure,
    biotite_array_from_feat,
)
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.analyze.analyze_utils import _load_stack, _radius, get_delta_sasa
from boltzgen.task.predict.data_from_generated import FromGeneratedDataset

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ResidueSpec:
    """One synthetic residue in a small interface fixture."""

    chain: str
    name: str
    origin: tuple[float, float, float]
    designed: bool = False
    absent: tuple[str, ...] = ()


def structure_from_specs(
    specs: list[ResidueSpec], links: tuple[tuple[int, int], ...] = ()
) -> Structure:
    """Create a tiny valid Boltz structure in canonical atom order."""
    atoms, residues, chains, coords, bonds = [], [], [], [], []
    names = list(dict.fromkeys(spec.chain for spec in specs))
    offsets = [(0, 0, 0), (1.45, 0, 0), (2, 1.4, 0), (1.4, 2.4, 0), (2, -0.8, 1.2)]
    for chain_idx, name in enumerate(names):
        first_atom, first_res = len(atoms), len(residues)
        subset = [spec for spec in specs if spec.chain == name]
        mol_type = const.chain_type_ids[
            "NONPOLYMER" if subset[0].name == "LIG" else "PROTEIN"
        ]
        for res_idx, spec in enumerate(subset):
            atom_start = len(atoms)
            atom_names = (
                ["C1", "C2"] if spec.name == "LIG" else const.ref_atoms[spec.name]
            )
            for atom_name, offset in zip(atom_names, offsets):
                coord = tuple(np.asarray(spec.origin) + offset)
                atoms.append(
                    (atom_name, coord, atom_name not in spec.absent, 50.0, 1.0)
                )
                coords.append((coord,))
            nonpolymer = spec.name == "LIG"
            token_name = "UNK" if nonpolymer else spec.name
            center = 0 if nonpolymer else const.res_to_center_atom_id[token_name]
            disto = 0 if nonpolymer else const.res_to_disto_atom_id[token_name]
            residues.append(
                (
                    spec.name,
                    const.token_ids[token_name],
                    res_idx,
                    atom_start,
                    len(atom_names),
                    atom_start + center,
                    atom_start + disto,
                    not nonpolymer,
                    True,
                )
            )
        chains.append(
            (
                name,
                mol_type,
                chain_idx,
                0,
                chain_idx,
                first_atom,
                len(atoms) - first_atom,
                first_res,
                len(subset),
                0,
                0,
            )
        )
    for left, right in links:
        left_chain = names.index(specs[left].chain)
        right_chain = names.index(specs[right].chain)
        bonds.append(
            (
                left_chain,
                right_chain,
                left,
                right,
                residues[left][3],
                residues[right][3],
                const.bond_type_ids["COVALENT"],
            )
        )
    return Structure(
        atoms=np.array(atoms, dtype=Atom),
        bonds=np.array(bonds, dtype=Bond),
        residues=np.array(residues, dtype=Residue),
        chains=np.array(chains, dtype=Chain),
        interfaces=np.empty(0, dtype=Interface),
        mask=np.ones(len(chains), dtype=bool),
        coords=np.array(coords, dtype=Coords),
        ensemble=np.array([(0, len(atoms))], dtype=Ensemble),
    )


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> FromGeneratedDataset:
    directory = tmp_path_factory.mktemp("molecules")
    molecules_dir = directory / "molecules"
    extras_dir = directory / "extras"
    molecules_dir.mkdir(parents=True, exist_ok=True)
    extras_dir.mkdir(parents=True, exist_ok=True)
    canonicals = {}
    for code, letter in [("GLY", "G"), ("ALA", "A")]:
        molecule = Chem.MolFromSequence(letter)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetProp("name", atom.GetPDBResidueInfo().GetName().strip())
        rdDepictor.Compute2DCoords(molecule)
        canonicals[code] = molecule
    ligand = Chem.MolFromSmiles("CC")
    for index, atom in enumerate(ligand.GetAtoms(), start=1):
        atom.SetProp("name", f"C{index}")
    rdDepictor.Compute2DCoords(ligand)
    for code, molecule in {**canonicals, "LIG": ligand}.items():
        with (molecules_dir / f"{code}.pkl").open("wb") as stream:
            previous = Chem.GetDefaultPickleProperties()
            try:
                Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
                pickle.dump(molecule, stream)
            finally:
                Chem.SetDefaultPickleProperties(previous)
    return FromGeneratedDataset(
        [],
        [],
        [],
        molecules_dir,
        canonicals,
        Tokenizer(),
        Featurizer(),
        extra_mol_dir=extras_dir,
    )


def prepare(
    dataset: FromGeneratedDataset,
    directory: Path,
    specs: list[ResidueSpec],
    links: tuple[tuple[int, int], ...] = (),
) -> tuple[dict, Path]:
    name = directory.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.cif"
    structure = structure_from_specs(specs, links)
    path.write_text(to_mmcif(structure))
    design_mask = np.array(
        [
            spec.designed
            for spec in specs
            for _ in range(2 if spec.name == "LIG" else 1)
        ],
        dtype=bool,
    )
    feat = dataset.get_feat(path, design_mask)
    assert not feat["exception"]
    atom_array = _load_stack(path)[0]
    assert len(atom_array) == int(feat["atom_resolved_mask"].sum())
    parsed = feat["str_gen"]
    np.testing.assert_allclose(
        atom_array.coord,
        parsed.coords["coords"][parsed.atoms["is_present"]],
        atol=0.001,
    )
    return feat, directory


def expected_sasa(
    path: Path, design_chains: set[str], target_chains: set[str]
) -> tuple[float, float, float]:
    """Independent chain-name oracle; does not use the pipeline design mask."""
    atoms = _load_stack(path)[0]
    if not target_chains:
        return 0.0, 0.0, 0.0
    design = np.isin(atoms.chain_id, list(design_chains))
    target = np.isin(atoms.chain_id, list(target_chains))
    radii = np.array(
        [
            _radius(r, a, e)
            for r, a, e in zip(atoms.res_name, atoms.atom_name, atoms.element)
        ]
    )
    unbound = bst.sasa(
        atoms[target], probe_radius=1.4, point_number=960, vdw_radii=radii[target]
    ).sum()
    bound_mask = target | design
    bound = bst.sasa(
        atoms[bound_mask],
        probe_radius=1.4,
        point_number=960,
        vdw_radii=radii[bound_mask],
    )[target[bound_mask]].sum()
    return float(unbound - bound), float(unbound), float(bound)


@pytest.fixture(scope="session")
def analysis_template() -> Analyze:
    # Other analysis tests may already have configured the process-wide pool.
    # Its size is unrelated to the SASA behavior exercised here.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "set_num_interop_threads", lambda _: None)
        return Analyze(
            name="sasa",
            data=SimpleNamespace(),
            backbone_fold_metrics=False,
            allatom_fold_metrics=False,
            delta_sasa_original=True,
            delta_sasa_refolded=True,
            compute_lddts=False,
        )


def run_analysis(
    analysis_template: Analyze, feat: dict, directory: Path, symmetric: bool = False
) -> tuple[dict[str, float], Path]:
    analysis = copy.copy(analysis_template)
    analysis.data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"^(.*)$"),
        predict_set=SimpleNamespace(get_sample=lambda **_kwargs: feat),
    )
    analysis.init_datasets(directory / "analysis")
    analysis.use_design_mask_for_target = symmetric
    # Exercise both SASA paths without running unrelated RMSD calculations.
    analysis.fold_metrics = True
    folded_dir = analysis.design_dir / const.folding_dirname
    folded_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        folded_dir / f"{feat['id']}.npz",
        res_type=feat["res_type"].numpy()[None],
        coords=feat["coords"].numpy(),
        design_to_target_iptm=np.array([0.8]),
        design_ptm=np.array([0.8]),
    )
    prediction = copy.deepcopy(feat)
    prediction["coords"] = prediction["coords"][0]
    # Move the target in refolded coordinates to detect use of the wrong file.
    last_chain = feat["asym_id"][-1]
    last_chain_atoms = (
        feat["atom_to_token"].float() @ (feat["asym_id"] == last_chain).float()
    ).bool()
    prediction["coords"][last_chain_atoms, 1] += 0.8
    refold, _, _ = Structure.from_feat(prediction)
    refold_path = analysis.refold_cif_dir / f"{feat['id']}.cif"
    refold_path.write_text(to_mmcif(refold))
    assert len(_load_stack(refold_path)[0]) == int(feat["atom_resolved_mask"].sum())
    result = analysis.compute_metrics(sample_id=feat["id"])
    assert result == feat["id"]
    with np.load(analysis.metrics_dir / f"metrics_{feat['id']}.npz") as saved:
        metrics = {key: saved[key].item() for key in saved.files if "sasa" in key}
    return metrics, refold_path


CASES = {
    "partial": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "GLY", (11, 1, 0)),
    ],
    "full": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0), designed=True),
        ResidueSpec("B", "GLY", (11, 1, 0)),
    ],
    "multiple": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "GLY", (0, 7, 0), designed=True),
        ResidueSpec("B", "ALA", (8, 6, 0)),
        ResidueSpec("C", "GLY", (11, 3, 0)),
    ],
    "multiple_targets": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "GLY", (11, 1, 0)),
        ResidueSpec("C", "GLY", (8, -3, 0)),
    ],
    "missing_atom": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0), absent=("O",)),
        ResidueSpec("B", "GLY", (11, 1, 0), absent=("O",)),
    ],
    "missing_scaffold_ca": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0), absent=("CA",)),
        ResidueSpec("B", "GLY", (11, 1, 0)),
    ],
    "missing_target_ca": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "GLY", (11, 1, 0), absent=("CA",)),
    ],
    "mixed_target_resolution": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "GLY", (11, 1, 0), absent=("CA",)),
        ResidueSpec("B", "ALA", (8, -3, 0)),
    ],
    "absent_scaffold": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0), absent=tuple(const.ref_atoms["ALA"])),
        ResidueSpec("B", "GLY", (11, 1, 0)),
    ],
    "empty_target": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0), designed=True),
    ],
    "ligand_target": [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (8, 0, 0)),
        ResidueSpec("B", "LIG", (11, 1, 0)),
    ],
}


@pytest.mark.parametrize("case", CASES)
def test_analysis_matches_complete_chain_sasa(
    case: str, dataset: FromGeneratedDataset, tmp_path: Path, analysis_template: Analyze
) -> None:
    specs = CASES[case]
    feat, directory = prepare(dataset, tmp_path / case, specs)
    design_chains = {spec.chain for spec in specs if spec.designed}
    target_chains = {spec.chain for spec in specs} - design_chains
    metrics, refold = run_analysis(analysis_template, feat, directory)
    for suffix, path in [("original", feat["path"]), ("refolded", refold)]:
        expected = expected_sasa(path, design_chains, target_chains)
        # The legacy design_sasa column names contain target SASA.
        assert metrics[f"delta_sasa_{suffix}"] == pytest.approx(expected[0], abs=1e-4)
        assert metrics[f"design_sasa_unbound_{suffix}"] == pytest.approx(
            expected[1], abs=1e-4
        )
        assert metrics[f"design_sasa_bound_{suffix}"] == pytest.approx(
            expected[2], abs=1e-4
        )
    if case == "partial":
        assert metrics["delta_sasa_original"] > 0
        assert metrics["delta_sasa_original"] != metrics["delta_sasa_refolded"]


@pytest.mark.parametrize("absent", [(), ("CA",)])
@pytest.mark.parametrize("distant_design", [False, True])
def test_symmetric_target_override(
    dataset: FromGeneratedDataset,
    tmp_path: Path,
    analysis_template: Analyze,
    absent: tuple[str, ...],
    distant_design: bool,
) -> None:
    design_x = 100 if distant_design else 0
    specs = [
        ResidueSpec("A", "GLY", (design_x, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (3, 0, 0), absent=absent),
        ResidueSpec("B", "GLY", (design_x, 5, 0), designed=True),
        ResidueSpec("B", "ALA", (3, 5, 0)),
    ]
    feat, directory = prepare(dataset, tmp_path / "symmetric", specs)
    metrics, refold = run_analysis(analysis_template, feat, directory, symmetric=True)
    for suffix, path in [("original", feat["path"]), ("refolded", refold)]:
        atoms = _load_stack(path)[0]
        target = atoms.res_name == "ALA"
        # A union of the redesigned GLY and fixed ALA atoms contains every atom
        # once, even when the complete-chain design mask overlaps the target.
        expected = get_delta_sasa(path, target, ~target)
        if distant_design:
            assert expected[0] == 0
        else:
            assert expected[0] > 0
        assert metrics[f"delta_sasa_{suffix}"] == pytest.approx(expected[0])


def test_covalently_attached_ligand_is_on_design_side(
    dataset: FromGeneratedDataset, tmp_path: Path, analysis_template: Analyze
) -> None:
    specs = [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("A", "ALA", (6, 0, 0)),
        ResidueSpec("B", "LIG", (8, 0, 0)),
        ResidueSpec("C", "GLY", (10, 0, 0)),
    ]
    feat, directory = prepare(dataset, tmp_path / "covalent", specs, links=((1, 2),))
    assert feat["chain_design_mask"].tolist() == [True, True, True, True, False]
    metrics, refold = run_analysis(analysis_template, feat, directory)
    for suffix, path in [("original", feat["path"]), ("refolded", refold)]:
        expected = expected_sasa(path, {"A", "B"}, {"C"})
        assert metrics[f"delta_sasa_{suffix}"] == pytest.approx(expected[0])


@pytest.mark.parametrize("kind", ["target_only", "design_only", "neither"])
def test_empty_interface(
    dataset: FromGeneratedDataset, tmp_path: Path, kind: str
) -> None:
    feat, _ = prepare(dataset, tmp_path / kind, CASES["full"])
    count = int(feat["atom_resolved_mask"].sum())
    target = np.full(count, kind == "target_only", dtype=bool)
    design = np.full(count, kind == "design_only", dtype=bool)
    delta, unbound, bound = get_delta_sasa(feat["path"], target, design)
    assert delta == 0.0
    assert unbound == bound
    assert (unbound > 0.0) == (kind == "target_only")


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize(
    "case", ["missing_atom", "missing_scaffold_ca", "absent_scaffold"]
)
def test_partial_residue_roundtrip_preserves_atoms_and_token_resolution(
    dataset: FromGeneratedDataset, tmp_path: Path, case: str, padded: bool
) -> None:
    feat, directory = prepare(dataset, tmp_path / case, CASES[case])
    if padded:
        tokenized = dataset.tokenizer.tokenize(feat["str_gen"])
        features = dataset.featurizer.process(
            Input(
                tokens=tokenized.tokens,
                bonds=tokenized.bonds,
                token_to_res=tokenized.token_to_res,
                structure=feat["str_gen"],
                msa={},
                templates=None,
            ),
            random=np.random.default_rng(0),
            molecules=dataset.canonicals,
            training=False,
            max_seqs=1,
            max_tokens=8,
            max_atoms=64,
        )
        feat.update(features)
        feat["chain_design_mask"] = torch.nn.functional.pad(
            feat["chain_design_mask"], (0, 8 - len(feat["chain_design_mask"]))
        )
        assert feat["token_pad_mask"].sum() < len(feat["token_pad_mask"])
    assert feat["atom_pad_mask"].sum() < len(feat["atom_pad_mask"])
    token_resolution = feat["token_resolved_mask"].clone()
    converted, _, _ = Structure.from_feat(feat)
    expected_presence = feat["str_gen"].residues["is_present"]
    np.testing.assert_array_equal(converted.residues["is_present"], expected_presence)
    np.testing.assert_array_equal(
        converted.atoms["is_present"], feat["str_gen"].atoms["is_present"]
    )
    assert torch.equal(feat["token_resolved_mask"], token_resolution)
    # Re-tokenizing still rejects a residue whose representative CA is absent.
    retokenized = Tokenizer().tokenize(converted)
    np.testing.assert_array_equal(
        retokenized.tokens["resolved_mask"],
        token_resolution[feat["token_pad_mask"].bool()],
    )
    path = directory / "roundtrip.cif"
    path.write_text(to_mmcif(converted))
    original, written = _load_stack(feat["path"])[0], _load_stack(path)[0]
    biotite_atoms = biotite_array_from_feat(feat)
    for atoms in (written, biotite_atoms):
        for field in ("atom_name", "res_name", "chain_id"):
            np.testing.assert_array_equal(
                getattr(atoms, field), getattr(original, field)
            )
        np.testing.assert_allclose(
            atoms.coord, feat["coords"][0, feat["atom_resolved_mask"]], atol=0.001
        )
    assert len(written) == int(feat["atom_resolved_mask"].sum())


@pytest.mark.parametrize("missing", [("C1",), ("C2",)])
def test_partially_resolved_atomized_residue(
    dataset: FromGeneratedDataset, tmp_path: Path, missing: tuple[str, ...]
) -> None:
    specs = [
        ResidueSpec("A", "GLY", (0, 0, 0), designed=True),
        ResidueSpec("B", "LIG", (8, 0, 0), absent=missing),
    ]
    feat, directory = prepare(dataset, tmp_path / "partial_ligand", specs)
    converted, _, _ = Structure.from_feat(feat)
    assert converted.residues["is_present"].tolist() == [True, True]
    np.testing.assert_array_equal(
        converted.atoms["is_present"], feat["str_gen"].atoms["is_present"]
    )
    assert feat["token_resolved_mask"].tolist() == [
        True,
        missing != ("C1",),
        missing != ("C2",),
    ]
    path = directory / "roundtrip.cif"
    path.write_text(to_mmcif(converted))
    original, written = _load_stack(feat["path"])[0], _load_stack(path)[0]
    np.testing.assert_array_equal(written.atom_name, original.atom_name)
    assert len(written) == int(feat["atom_resolved_mask"].sum())


@pytest.mark.parametrize("target", [False, True])
def test_sasa_rejects_misaligned_masks_even_for_empty_target(
    dataset: FromGeneratedDataset, tmp_path: Path, target: bool
) -> None:
    feat, _ = prepare(dataset, tmp_path / "invalid_masks", CASES["full"])
    wrong_count = int(feat["atom_resolved_mask"].sum()) - 1
    with pytest.raises(IndexError, match="boolean index"):
        get_delta_sasa(
            feat["path"],
            np.full(wrong_count, target),
            np.zeros(wrong_count, dtype=bool),
        )
