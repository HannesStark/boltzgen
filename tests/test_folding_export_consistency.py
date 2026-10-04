"""Exercise folding export through prediction assembly and real mmCIF IO."""
# ruff: noqa: INP001

from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest
import torch
from torch.nn.functional import one_hot

from boltzgen.data import const
from boltzgen.data.data import Bond, convert_atom_name, convert_ccd
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.task.analyze.analyze_utils import get_best_folding_sample
from boltzgen.task.predict.data_from_generated import collate
from boltzgen.task.predict.writer import FoldingWriter


def _features(*, padded: bool, missing_atom: bool, ligand: bool) -> dict[str, Any]:
    """Build two GLY residues and optionally an atomized three-atom ligand."""
    real_tokens, real_atoms = (5, 11) if ligand else (2, 8)
    n_tokens = real_tokens + int(padded)
    n_atoms = real_atoms + 4 * int(padded)
    names = ["N", "CA", "C", "O"] * 2
    names += ["C1", "C2", "O1"] if ligand else []
    names += [""] * (n_atoms - real_atoms)
    elements = [7, 6, 6, 8] * 2 + ([6, 6, 8] if ligand else [])
    elements += [0] * (n_atoms - real_atoms)
    atom_to_token = torch.zeros(n_atoms, n_tokens)
    atom_to_token[:4, 0] = 1
    atom_to_token[4:8, 1] = 1
    if ligand:
        atom_to_token[8:11, 2:5] = torch.eye(3)
    atom_pad = torch.arange(n_atoms) < real_atoms
    resolved = atom_pad.clone()
    if missing_atom:
        resolved[3] = False
    res_types = [const.token_ids["GLY"]] * 2
    res_types += [const.token_ids["UNK"]] * 3 if ligand else []
    res_types += [const.token_ids["<pad>"]] * (n_tokens - real_tokens)
    coords = torch.zeros(n_atoms, 3)
    coords[:8] = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 2, 0],
            [4, 0, 0],
            [5, 0, 0],
            [5, 1, 0],
            [5, 2, 0],
        ],
    )
    if ligand:
        coords[8:11] = torch.tensor([[1, 4, 0], [2, 4, 0], [2, 5, 0]])
    representative = torch.zeros(n_tokens, n_atoms)
    representative[0, 1] = representative[1, 5] = 1
    if ligand:
        representative[2:5, 8:11] = torch.eye(3)
    asym = torch.zeros(n_tokens, dtype=torch.long)
    mol_type = torch.zeros(n_tokens, dtype=torch.long)
    residue_index = torch.arange(n_tokens)
    token_to_res = torch.arange(n_tokens)
    ccd = torch.tensor([convert_ccd("GLY")] * n_tokens)
    standard = torch.ones(n_tokens, dtype=torch.bool)
    if ligand:
        asym[2:5] = 1
        mol_type[2:5] = const.chain_type_ids["NONPOLYMER"]
        residue_index[2:5] = 0
        token_to_res[2:5] = 2
        ccd[2:5] = torch.tensor(convert_ccd("LIG"))
        standard[2:5] = False
    features = {
        "id": "contract",
        "exception": False,
        "skip": False,
        "structure_bonds": np.array([], dtype=Bond),
        "entity_id": asym.clone(),
        "asym_id": asym,
        "sym_id": torch.zeros(n_tokens, dtype=torch.long),
        "mol_type": mol_type,
        "res_type": one_hot(torch.tensor(res_types), len(const.tokens)),
        "coords": coords.unsqueeze(0),
        "type_bonds": torch.zeros(n_tokens, n_tokens, dtype=torch.long),
        "new_to_old_atomidx": torch.arange(n_atoms),
        "ref_element": one_hot(torch.tensor(elements), const.num_elements),
        "ref_charge": torch.zeros(n_atoms),
        "ref_atom_name_chars": one_hot(
            torch.tensor([convert_atom_name(name) for name in names]),
            64,
        ),
        "atom_to_token": atom_to_token,
        "residue_index": residue_index,
        "token_index": torch.arange(n_tokens),
        "atom_resolved_mask": resolved,
        "token_resolved_mask": torch.arange(n_tokens) < real_tokens,
        "design_mask": torch.zeros(n_tokens, dtype=torch.bool),
        "chain_design_mask": torch.zeros(n_tokens, dtype=torch.bool),
        "atom_pad_mask": atom_pad,
        "token_pad_mask": (torch.arange(n_tokens) < real_tokens).float(),
        "is_standard": standard,
        "ccd": ccd,
        "token_to_res": token_to_res,
        "backbone_mask": atom_pad.clone(),
        "token_to_rep_atom": representative,
        "bfactor": torch.zeros(n_atoms),
        "plddt": torch.zeros(n_atoms),
    }
    # Supply the normal feature families consumed by the real mask=True path.
    for key in (
        "contact_threshold",
        "contact_conditioning",
        "token_disto_mask",
        "binding_type",
        "structure_group",
        "cyclic",
        "modified",
        "token_distance_mask",
        "method_feature",
        "temp_feature",
        "ph_feature",
        "design_ss_mask",
        "ss_type",
        "feature_residue_index",
        "feature_asym_id",
        "symmetric_group",
        "target_msa_mask",
        "deletion_mean",
    ):
        features[key] = torch.zeros(n_tokens)
    for key in ("token_pair_mask", "token_bonds"):
        features[key] = torch.zeros(n_tokens, n_tokens)
    for key in ("ref_space_uid", "fake_atom_mask", "ref_chirality"):
        features[key] = torch.zeros(n_atoms)
    for key in ("msa", "msa_mask", "msa_paired", "deletion_value", "has_deletion"):
        features[key] = torch.zeros(1, n_tokens)
    features.update(
        {
            "center_coords": torch.zeros(n_tokens, 3),
            "res_type_clone": features["res_type"].clone(),
            "profile": torch.zeros(n_tokens, len(const.tokens)),
            "ref_pos": coords.clone(),
            "masked_ref_atom_name_chars": features["ref_atom_name_chars"].clone(),
            "r_set_to_rep_atom": representative.clone(),
            "token_to_bb4_atoms": torch.zeros(n_tokens, 4, n_atoms),
        }
    )
    return features


def _forward_output(batch: dict[str, Any], samples: int) -> dict[str, Any]:
    """Give each sample distinct coordinates and token confidence."""
    n_tokens = batch["token_index"].shape[1]
    coords = torch.cat([batch["coords"][0] + 10 * i for i in range(samples)])
    plddt = torch.tensor(
        [
            [0.15 + 0.1 * i + 0.03 * token for token in range(n_tokens)]
            for i in range(samples)
        ]
    )
    global_score = torch.full((samples,), 0.1)
    global_score[0] = 0.9
    output = {key: torch.zeros(samples) for key in const.eval_keys_confidence}
    output.update(
        {
            "sample_atom_coords": coords,
            "plddt": plddt,
            "ptm": global_score,
            "iptm": global_score.clone(),
            "complex_plddt": plddt[:, :2].mean(-1),
            "pde": torch.zeros(samples, n_tokens, n_tokens),
            "pae": torch.zeros(samples, n_tokens, n_tokens),
            "pair_chains_iptm": {0: {0: global_score.clone()}},
            "coords_traj": [coords - 1, coords],
            "x0_coords_traj": [coords],
        }
    )
    return output


class _InferenceBoundary:
    """Replace only the trained forward pass; keep prediction assembly real."""

    def __init__(self, output: dict[str, Any], samples: int, *, mask: bool) -> None:
        self.output = output
        self.checkpoints = None
        self.step_scale_schedule = None
        self.noise_scale_schedule = None
        self.masker = BoltzMasker(mask=mask)
        self.predict_args = {
            "recycling_steps": 0,
            "sampling_steps": 2,
            "diffusion_samples": samples,
            "keys_dict_out": list(dict.fromkeys(const.eval_keys_confidence)),
        }
        self.inverse_fold = False
        self.confidence_prediction = True
        self.alpha_pae = 1
        self.affinity_prediction = False
        self.inference_counter = 0

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
        return self.output


@pytest.mark.parametrize(("samples", "winner"), [(1, 0), (2, 1), (5, 4)])
@pytest.mark.parametrize(
    ("padded", "missing_atom", "ligand"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("mask", [False, True])
def test_export_matches_analysis_and_selected_confidence(
    samples: int,
    winner: int,
    padded: bool,
    missing_atom: bool,
    ligand: bool,
    mask: bool,
    tmp_path: Path,
) -> None:
    batch = collate(
        [_features(padded=padded, missing_atom=missing_atom, ligand=ligand)]
    )
    output = _forward_output(batch, samples)
    output["design_to_target_iptm"][winner] = 0.9
    output["design_ptm"][winner] = 0.8
    prediction = Boltz.predict_step(
        _InferenceBoundary(output, samples, mask=mask), batch
    )
    assert prediction["exception"] == (False if mask else [False])
    before = {
        key: value.clone()
        for key, value in prediction.items()
        if isinstance(value, torch.Tensor)
    }

    writer = FoldingWriter(str(tmp_path))
    writer.write_on_batch_end(prediction=prediction, batch=batch)

    assert writer.failed == 0
    with np.load(writer.outdir / "contract.npz") as archive:
        assert set(archive.files) == set(const.eval_keys) & set(prediction)
        for key in archive.files:
            np.testing.assert_array_equal(archive[key], before[key].numpy())
        analyzed = get_best_folding_sample(archive)
    np.testing.assert_array_equal(
        analyzed["coords"], output["sample_atom_coords"][winner]
    )
    for key, value in before.items():
        torch.testing.assert_close(prediction[key], value)

    block = gemmi.cif.read_file(
        str(writer.refold_cif_dir / "contract.cif")
    ).sole_block()
    coordinates = np.column_stack(
        [
            np.array(block.find_values(f"_atom_site.Cartn_{axis}"), dtype=float)
            for axis in "xyz"
        ]
    )
    bfactor = np.array(block.find_values("_atom_site.B_iso_or_equiv"), dtype=float)
    emitted = batch["atom_resolved_mask"][0] & batch["atom_pad_mask"][0]
    np.testing.assert_allclose(coordinates, analyzed["coords"][emitted], atol=1e-4)
    expected_bfactor = batch["atom_to_token"][0] @ output["plddt"][winner]
    np.testing.assert_allclose(bfactor, expected_bfactor[emitted], atol=1e-5)
    qa = np.array(block.find_values("_ma_qa_metric_local.metric_value"), dtype=float)
    np.testing.assert_allclose(qa, output["plddt"][winner, :2] * 100, atol=1e-3)
    residues = list(block.find_values("_atom_site.label_comp_id"))
    assert residues == ["GLY"] * (8 - int(missing_atom)) + (
        ["LIG"] * 3 if ligand else []
    )


@pytest.mark.parametrize(
    "ranking", ["agree_nonzero", "weighted", "tie", "zero_interface"]
)
def test_ranking_boundaries(ranking: str, tmp_path: Path) -> None:
    batch = collate([_features(padded=True, missing_atom=False, ligand=False)])
    output = _forward_output(batch, 2)
    if ranking == "agree_nonzero":
        output["iptm"] = output["ptm"] = torch.tensor([0.1, 0.9])
        output["design_to_target_iptm"] = torch.tensor([0.1, 0.9])
        output["design_ptm"] = torch.tensor([0.1, 0.9])
        winner = 1
    elif ranking == "weighted":
        output["design_to_target_iptm"] = torch.tensor([0.9, 0.7])
        output["design_ptm"] = torch.tensor([0.0, 1.0])
        winner = 1
    elif ranking == "tie":
        output["design_to_target_iptm"] = torch.tensor([0.5, 0.5])
        output["design_ptm"] = torch.tensor([0.5, 0.5])
        winner = 0
    else:
        output["design_to_target_iptm"] = torch.zeros(2)
        output["design_ptm"] = torch.tensor([0.1, 0.9])
        winner = 1
    prediction = Boltz.predict_step(_InferenceBoundary(output, 2, mask=False), batch)
    writer = FoldingWriter(str(tmp_path), designfolding=True)
    writer.write_on_batch_end(prediction=prediction, batch=batch)

    block = gemmi.cif.read_file(
        str(writer.refold_cif_dir / "contract.cif")
    ).sole_block()
    x = np.array(block.find_values("_atom_site.Cartn_x"), dtype=float)
    bfactor = np.array(block.find_values("_atom_site.B_iso_or_equiv"), dtype=float)
    with np.load(writer.outdir / "contract.npz") as archive:
        selected = get_best_folding_sample(archive)
    np.testing.assert_allclose(x, output["sample_atom_coords"][winner, :8, 0])
    np.testing.assert_allclose(x, selected["coords"][:8, 0])
    expected_bfactor = batch["atom_to_token"][0] @ output["plddt"][winner]
    np.testing.assert_allclose(bfactor, expected_bfactor[:8], atol=1e-5)


@pytest.mark.parametrize("list_status", [False, True])
@pytest.mark.parametrize("flag", ["exception", "skip"])
def test_status_only_predictions_do_not_write_files(
    flag: str,
    list_status: bool,
    tmp_path: Path,
) -> None:
    batch = {"id": ["contract"], flag: [True]}
    prediction = Boltz.predict_step(_InferenceBoundary({}, 1, mask=False), batch)
    assert prediction == {flag: True}
    if list_status:
        prediction[flag] = [True]
    writer = FoldingWriter(str(tmp_path))
    writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert writer.failed == int(flag == "exception")
    assert list(writer.outdir.iterdir()) == []
    assert list(writer.refold_cif_dir.iterdir()) == []
