"""Exercise folding export through prediction assembly and real mmCIF IO."""
# ruff: noqa: INP001

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

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
from boltzgen.task.predict.predict import Predict
from boltzgen.task.predict.writer import AffinityWriter, FoldingWriter


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
        self.token_level_confidence = True
        self.alpha_pae = 1
        self.affinity_prediction = False
        self.inference_counter = 0

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
        return self.output


@pytest.mark.parametrize("keys_dict_out", [None, [], ["plddt"], ["ptm", "iptm"]])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("alpha_pae", [0.0, 1.0])
def test_default_and_reduced_prediction_keys_preserve_ranking(
    keys_dict_out: list[str] | None, mask: bool, alpha_pae: float, tmp_path: Path
) -> None:
    batch = collate([_features(padded=True, missing_atom=False, ligand=False)])
    output = _forward_output(batch, 2)
    output["design_to_target_iptm"][1] = 0.9
    output["design_ptm"][1] = 0.8
    writer = FoldingWriter(str(tmp_path))
    task = Predict(
        data=None,
        writer=writer,
        checkpoint="unused.ckpt",
        output=str(tmp_path),
        name="folding",
        recycling_steps=0,
        sampling_steps=2,
        diffusion_samples=2,
        keys_dict_out=keys_dict_out,
    )
    boundary = _InferenceBoundary(output, 2, mask=mask)
    boundary.alpha_pae = alpha_pae
    boundary.predict_args = task.predict_args
    prediction = Boltz.predict_step(boundary, batch)
    writer.write_on_batch_end(prediction=prediction, batch=batch)

    with np.load(writer.outdir / "contract.npz") as archive:
        selected = get_best_folding_sample(archive)
    np.testing.assert_array_equal(selected["coords"], output["sample_atom_coords"][1])
    block = gemmi.cif.read_file(
        str(writer.refold_cif_dir / "contract.cif")
    ).sole_block()
    x = np.array(block.find_values("_atom_site.Cartn_x"), dtype=float)
    np.testing.assert_allclose(x, output["sample_atom_coords"][1, :8, 0])


@pytest.mark.parametrize(
    ("status", "affinity_status", "mask"),
    [
        ("exception", None, False),
        ("skip", None, False),
        ("success", None, False),
        ("success", "exception", True),
        ("success", "skip", True),
        ("success", "success", True),
    ],
)
def test_refolding_validation_stops_after_failed_or_skipped_prediction(
    status: str, affinity_status: str | None, mask: bool, tmp_path: Path
) -> None:
    pytest.importorskip(
        "wandb", reason="Refolding validation uses optional dev dependencies"
    )
    from boltzgen.model.validation.refolding import RefoldingValidator  # noqa: PLC0415

    features = _features(padded=False, missing_atom=False, ligand=False)
    if status != "success":
        features[status] = True
    folded_output = _forward_output(collate([features]), 1)
    boundary = _InferenceBoundary(folded_output, 1, mask=mask)
    validator = object.__new__(RefoldingValidator)
    validator.dataset_to_logname = {0: "val_monomer"}
    validator.inverse_fold = True
    validator.writer = FoldingWriter(design_dir=None)
    validator.aff_writer = AffinityWriter(design_dir=None)
    validator.design_val_step = Mock(return_value=True)
    validator.init_folding_model = Mock()
    validator.init_affinity_model = Mock()
    validator.affinity_model = None
    if affinity_status is not None:
        affinity_prediction = (
            {"exception": False, "affinity_pred_value": torch.tensor([1.25])}
            if affinity_status == "success"
            else {affinity_status: True}
        )
        validator.affinity_model = SimpleNamespace(
            predict_step=Mock(return_value=affinity_prediction)
        )
    validator.folding_model = SimpleNamespace(
        predict_step=lambda batch, batch_idx: Boltz.predict_step(
            boundary, batch, batch_idx
        )
    )

    def set_design_dir(design_dir: str) -> None:
        validator.design_dir = design_dir
        validator.writer.init_outdir(design_dir)
        validator.aff_writer.init_outdir(design_dir)

    validator.set_design_dir = set_design_dir
    compute_metrics = Mock(side_effect=StopIteration("analysis reached"))
    validator.analyze_task = SimpleNamespace(
        data=SimpleNamespace(
            predict_set=SimpleNamespace(get_feat=Mock(return_value=features)),
            transfer_batch_to_device=Mock(),
        ),
        compute_metrics=compute_metrics,
    )
    model = SimpleNamespace(
        device=torch.device("cpu"),
        trainer=SimpleNamespace(default_root_dir=tmp_path, global_rank=0),
        current_epoch=0,
        global_step=0,
        log=Mock(),
    )
    args = {
        "model": model,
        "batch": {"id": ["contract"]},
        "out": {"feat_masked": {"design_mask": features["design_mask"].unsqueeze(0)}},
        "idx_dataset": 0,
        "dataloader_idx": 0,
        "n_samples": 1,
        "batch_idx": 0,
    }
    if status == "success":
        with pytest.raises(StopIteration, match="analysis reached"):
            validator.process(**args)
        compute_metrics.assert_called_once()
        validator.init_affinity_model.assert_called_once()
        assert (validator.writer.refold_cif_dir / "contract.cif").exists()
    else:
        validator.process(**args)
        compute_metrics.assert_not_called()
        validator.init_affinity_model.assert_not_called()
        assert list(validator.writer.outdir.iterdir()) == []
        assert list(validator.writer.refold_cif_dir.iterdir()) == []
    assert validator.writer.failed == int(status == "exception")
    assert validator.aff_writer.failed == int(affinity_status == "exception")
    affinity_path = validator.aff_writer.outdir / "contract.npz"
    assert affinity_path.exists() == (affinity_status == "success")
    if affinity_status == "success":
        with np.load(validator.writer.outdir / "contract.npz") as archive:
            np.testing.assert_array_equal(archive["affinity_pred_value"], [1.25])


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
    features = _features(padded=padded, missing_atom=missing_atom, ligand=ligand)
    features["design_mask"][0] = True
    features["chain_design_mask"][:2] = True
    if ligand:
        features["design_mask"][2:5] = True
        features["backbone_mask"][8:11] = False
        features["masked_ref_atom_name_chars"][8:11] = one_hot(
            torch.tensor([convert_atom_name("X")] * 3), 64
        )
    batch = collate([features])
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
            expected = batch[key] if key == "res_type" else before[key]
            np.testing.assert_array_equal(archive[key], expected.numpy())
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
    atom_names = np.array(
        ["N", "CA", "C", "O"] * 2 + (["C1", "C2", "O1"] if ligand else [])
    )
    np.testing.assert_array_equal(
        list(block.find_values("_atom_site.label_atom_id")),
        atom_names[emitted[: len(atom_names)]],
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
@pytest.mark.parametrize("include_id", [False, True])
@pytest.mark.parametrize("writer_class", [FoldingWriter, AffinityWriter])
def test_status_only_predictions_do_not_write_files(
    flag: str,
    list_status: bool,
    include_id: bool,
    writer_class: type,
    tmp_path: Path,
) -> None:
    batch = {flag: [True]}
    if include_id:
        batch["id"] = ["contract"]
    prediction = Boltz.predict_step(_InferenceBoundary({}, 1, mask=False), batch)
    assert prediction == {flag: True}
    if list_status:
        prediction[flag] = [True]
    writer = writer_class(str(tmp_path))
    writer.write_on_batch_end(prediction=prediction, batch=batch)
    assert writer.failed == int(flag == "exception")
    assert list(writer.outdir.iterdir()) == []
    assert list(tmp_path.rglob("*.cif")) == []
