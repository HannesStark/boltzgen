"""Real ESMFold2 input construction tests; run in the separate ESM environment.

Set ESMCFOLD_CCD_PATH to the pinned CCD pickle (no GPU or model weights needed).
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("esm")
if not os.environ.get("ESMCFOLD_CCD_PATH"):
    pytest.skip(
        "Set ESMCFOLD_CCD_PATH to run real ESM input tests", allow_module_level=True
    )

from esm.models.esmfold2 import ESMFold2InputBuilder

from boltzgen.task.esmfold2.crop import crop_features, polymer_representatives
from boltzgen.task.esmfold2.worker import prepare_request, run_request


def test_installed_worker_ignores_parent_site_packages(tmp_path):
    from boltzgen.task.esmfold2 import runtime, worker

    parent = tmp_path / "parent_site_packages"
    package = parent / "boltzgen"
    shutil.copytree(Path(worker.__file__).resolve().parents[2], package)
    for name in ("torch", "numpy", "esm"):
        (parent / f"{name}.py").write_text(
            "raise RuntimeError('parent environment leaked')\n"
        )
    # worker_command uses its own installed location; substitute the equivalent
    # wheel location so the parent directory also contains incompatible packages.
    command = runtime.worker_command(
        sys.executable, tmp_path / "manifest.json", "cuda:0"
    )
    command[2] = str(package / "task/esmfold2/worker.py")
    result = subprocess.run(
        [*command, "--help"],
        env=dict(
            os.environ,
            PYTHONPATH=str(parent),
            PYTHONHOME=str(tmp_path / "invalid_home"),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "manifest" in result.stdout


@pytest.fixture(scope="module")
def builder():
    return ESMFold2InputBuilder()


def request_for(target_names=None, target_kind=0):
    return {
        "design_id": "example.design_0",
        "design_sha256": "test",
        "chains": [
            {
                "id": "A",
                "mol_type": target_kind,
                "residue_names": target_names or ["ALA", "CYS", "GLY", "ALA", "GLY"],
                "indices": [0, 2, 4],
            },
            {
                "id": "B",
                "mol_type": 0,
                "residue_names": ["ALA", "GLY", "CYS"],
                "indices": [0, 1, 2],
            },
        ],
        "bonds": [],
        "design_chains": ["B"],
        "target_chains": ["A"],
        "nucleic_acid": target_kind in (1, 2),
        "options": {
            "seed": 0,
            "lm_dropout": 0.3,
            "num_loops": 20,
            "sampling_steps": 200,
            "diffusion_samples": 5,
        },
    }


@pytest.mark.parametrize(
    "kind,names",
    [
        (0, ["ALA", "GLY", "HYP", "ALA", "GLY"]),
        (1, ["DA", "DG", "DC", "DT", "DA"]),
        (2, ["A", "G", "C", "U", "A"]),
    ],
)
def test_real_noncontiguous_crop_modified_polymers_and_nucleic_acids(
    builder, kind, names
):
    request = request_for(names, kind)
    # A cofactor remains in folding context without being counted as a residue.
    request["chains"].append(
        {"id": "Z", "mol_type": 3, "residue_names": ["ZN"], "indices": [0]}
    )
    full, full_infos = prepare_request(request, builder)
    cropped, infos, indices, audit = crop_features(full, full_infos, request["chains"])
    reps = polymer_representatives(cropped, infos)
    assert set(reps) == {"A", "B"}
    assert len(reps["A"]) == 3
    assert cropped["residue_index"][0, reps["A"]].tolist() == [0, 2, 4]
    for key in ("asym_id", "entity_id", "sym_id", "residue_index", "input_ids"):
        assert torch.equal(cropped[key], full[key][:, indices])
    assert torch.equal(
        cropped["token_bonds"], full["token_bonds"][:, indices][:, :, indices]
    )
    assert audit["msa_depth"] == 1


def test_identity_crop_and_cyclic_bond(builder):
    request = request_for()
    request["chains"][0]["indices"] = [0, 1, 2, 3, 4]
    request["bonds"] = [["B", 0, "N", "B", 2, "C"]]
    full, infos = prepare_request(request, builder)
    cropped, _, _, _ = crop_features(full, infos, request["chains"])
    for key in full:
        assert torch.equal(cropped[key], full[key]), key
    assert cropped["token_bonds"][0, 5, 7, 0] == 1


def test_sidechain_cyclization_preserves_explicit_leaving_atom(builder):
    request = request_for()
    request["chains"][1]["residue_names"] = ["GLY", "ALA", "GLU"]
    request["chains"][1]["omitted_atoms"] = [[2, "OE2"]]
    request["bonds"] = [["B", 0, "N", "B", 2, "CD"]]
    full, full_infos = prepare_request(request, builder)
    cropped, infos, _, audit = crop_features(full, full_infos, request["chains"])
    last = infos[1].tokens[-1]
    names = cropped["ref_atom_name_chars"][0].tolist()
    atom_names = {
        "".join(chr(n + 32) for n in names[i] if n)
        for i in range(last.atom_start, last.atom_start + last.atom_count)
    }
    assert "OE1" in atom_names and "OE2" not in atom_names
    assert cropped["token_bonds"][0, 3, 5, 0] == 1
    assert audit["explicitly_omitted_atoms"] == 1


def test_covalent_smiles_keeps_source_atom_names(builder):
    request = request_for()
    request["chains"].append(
        {
            "id": "L",
            "mol_type": 3,
            "residue_names": ["LIG0"],
            "indices": [0],
            "smiles": "CCO",
            "smiles_atom_names": ["C1", "C2", "O1"],
        }
    )
    request["bonds"] = [["B", 2, "SG", "L", 0, "C2"]]
    full, full_infos = prepare_request(request, builder)
    cropped, infos, _, _ = crop_features(full, full_infos, request["chains"])
    ligand = infos[-1]
    names = [
        "".join(
            chr(n + 32)
            for n in cropped["ref_atom_name_chars"][0, t.atom_start].tolist()
            if n
        )
        for t in ligand.tokens
    ]
    assert names == ["C1", "C2", "O1"]
    assert ligand.ligand_bonds == [("C1", "C2"), ("C2", "O1")]
    assert cropped["token_bonds"][0, 5, ligand.tokens[1].token_index, 0] == 1
    # The original omission name resolves too; ESM requires all three atoms of
    # this tiny ligand's frame, so deletion reaches the deliberate frame guard.
    request["chains"][-1]["omitted_atoms"] = [[0, "O1"]]
    with pytest.raises(ValueError, match="required by frames_idx"):
        crop_features(full, full_infos, request["chains"])


def test_native_ccd_leaving_atom_removal_is_idempotent(builder):
    request = request_for()
    request["chains"].append(
        {
            "id": "L",
            "mol_type": 3,
            "residue_names": ["NAG"],
            "indices": [0],
            "omitted_atoms": [[0, "O1"]],
        }
    )
    request["bonds"] = [["B", 2, "SG", "L", 0, "C1"]]
    full, infos = prepare_request(request, builder)
    cropped, _, _, _ = crop_features(full, infos, request["chains"])
    assert cropped["token_bonds"].count_nonzero()
    request["chains"][-1]["omitted_atoms"] = [[0, "TYPO"]]
    with pytest.raises(ValueError, match="Cannot resolve omitted atom"):
        crop_features(full, infos, request["chains"])


def test_smiles_halogen_names_fit_model_vocabulary(builder):
    from esm.models.esmfold2.layers import CHAR_VOCAB_SIZE

    request = request_for()
    request["chains"].append(
        {
            "id": "L",
            "mol_type": 3,
            "residue_names": ["LIG0"],
            "indices": [0],
            "smiles": "ClCBr",
            "smiles_atom_names": ["Cl1", "C1", "Br1"],
        }
    )
    request["bonds"] = [["B", 2, "SG", "L", 0, "Cl1"]]
    full, infos = prepare_request(request, builder)
    cropped, infos, _, _ = crop_features(full, infos, request["chains"])
    # This is the actual model's atom-name encoding, which rejected lowercase.
    torch.nn.functional.one_hot(cropped["ref_atom_name_chars"].long(), CHAR_VOCAB_SIZE)
    ligand = infos[-1]
    assert cropped["token_bonds"][0, 5, ligand.tokens[0].token_index, 0] == 1
    names = [
        "".join(
            chr(n + 32)
            for n in cropped["ref_atom_name_chars"][0, t.atom_start].tolist()
            if n
        )
        for t in ligand.tokens
    ]
    assert names == ["CL1", "C1", "BR1"]


def test_single_atom_polymer_cap_cannot_be_scored_as_an_amino_acid(builder):
    request = request_for(["ALA", "GLY", "NH2"])
    request["chains"][0]["indices"] = [0, 1, 2]
    full, infos = prepare_request(request, builder)
    cropped, infos, _, _ = crop_features(full, infos, request["chains"])
    with pytest.raises(ValueError, match="lacks a unique CA"):
        polymer_representatives(cropped, infos)


@pytest.mark.parametrize("chain_ids", [("A", "B"), ("ABCDE", "ABCDF")])
def test_exported_structure_is_readable_by_boltz_gemmi(builder, tmp_path, chain_ids):
    gemmi = pytest.importorskip("gemmi")
    from boltzgen.task.esmfold2.worker import write_structure

    request = request_for()
    for chain, chain_id in zip(request["chains"], chain_ids, strict=True):
        chain["id"] = chain_id
    request["target_chains"], request["design_chains"] = [chain_ids[0]], [chain_ids[1]]
    full, infos = prepare_request(request, builder)
    features, infos, _, _ = crop_features(full, infos, request["chains"])
    path = tmp_path / "prediction.cif"
    write_structure(
        path,
        torch.zeros(features["atom_attention_mask"].shape[1], 3),
        torch.ones(features["input_ids"].shape[1]),
        features,
        infos,
    )
    structure = gemmi.read_structure(str(path))
    assert len(structure) == 1
    assert sum(1 for c in structure[0] for r in c for _ in r) == int(
        features["atom_attention_mask"].sum()
    )
    assert [(c.name, r.seqid.num) for c in structure[0] for r in c] == [
        (c["id"], i + 1) for c in request["chains"] for i in c["indices"]
    ]


@pytest.mark.parametrize("lm_dropout", [0.0, 0.3])
def test_full_esmc_then_cropped_forward_and_best_ipsae_sample(
    builder, tmp_path, monkeypatch, lm_dropout
):
    request = request_for()
    request["options"]["lm_dropout"] = lm_dropout
    # Allow a CUDA autocast context on this CPU test to detect unintended outer
    # autocast independently of hardware. No CUDA tensors are allocated here.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class ModelBoundary:
        config = SimpleNamespace(
            type="standard",
            lm_encoder=SimpleNamespace(lm_dropout=0.25, per_loop_lm_dropout=False),
        )

        def _compute_lm_hidden_states(
            self, input_ids, asym_id, residue_index, mol_type, mask, **kwargs
        ):
            assert input_ids.shape == (1, 8)
            assert residue_index.tolist() == [[0, 1, 2, 3, 4, 0, 1, 2]]
            assert kwargs["lm_mask_pct"] == 0
            return torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)

        def __call__(self, **kwargs):
            assert not torch.is_autocast_enabled("cuda")
            assert self.config.lm_encoder.lm_dropout == lm_dropout
            assert self.config.lm_encoder.per_loop_lm_dropout
            assert kwargs["input_ids"].shape == (1, 6)
            assert kwargs["lm_hidden_states"].flatten().tolist() == [0, 2, 4, 5, 6, 7]
            assert kwargs["msa"].shape == (1, 1, 6)
            assert kwargs["num_diffusion_samples"] == 5
            pae = torch.full((5, 6, 6), 20.0)
            pae[2, :3, 3:] = 0.5
            pae[2, 3:, :3] = 0.5
            atoms = kwargs["atom_attention_mask"].shape[1]
            return {
                "pae": pae,
                "sample_atom_coords": torch.zeros(5, atoms, 3),
                "plddt": torch.ones(5, 6) * 0.8,
                "iptm": torch.tensor([0.99, 0.9, 0.01, 0.9, 0.8]),
            }

    model = ModelBoundary()
    run_request(model, builder, request, tmp_path, "cpu")
    assert model.config.lm_encoder.lm_dropout == 0.25
    assert not model.config.lm_encoder.per_loop_lm_dropout
    result = json.loads((tmp_path / "example.design_0.json").read_text())
    assert result["selected_sample"] == 2
    assert result["metrics"]["esmfold2_ipsae_min"] == pytest.approx(0.8)
    assert result["input_audit"]["full_lm_shape"] == [1, 8, 1, 1]
    assert result["input_audit"]["crop_lm_shape"] == [1, 6, 1, 1]
    assert (tmp_path / "example.design_0.cif").exists()
    assert np.load(tmp_path / "example.design_0.npz")["selected_sample"] == 2

    from boltzgen.task.esmfold2 import worker

    def interrupted_write(*args):
        raise OSError("Interrupted output write")

    monkeypatch.setattr(worker, "write_structure", interrupted_write)
    with pytest.raises(OSError, match="Interrupted"):
        run_request(ModelBoundary(), builder, request, tmp_path, "cpu")
    assert not (tmp_path / "example.design_0.json").exists()

    def failed_forward(self, **kwargs):
        assert self.config.lm_encoder.lm_dropout == lm_dropout
        raise RuntimeError("Forward failed")

    monkeypatch.setattr(ModelBoundary, "__call__", failed_forward)
    with pytest.raises(RuntimeError, match="Forward failed"):
        run_request(model, builder, request, tmp_path, "cpu")
    assert model.config.lm_encoder.lm_dropout == 0.25
    assert not model.config.lm_encoder.per_loop_lm_dropout


@pytest.mark.parametrize(
    "polymer_count,ligand", [(1, False), (1, True), (2, False), (3, False)]
)
def test_redesign_selects_native_ptm_or_weakest_chain_interface(
    builder, tmp_path, polymer_count, ligand
):
    from boltzgen.task.esmfold2.contract import load_result, fingerprint

    request = request_for()
    request["scoring_mode"] = "redesign"
    if polymer_count == 1:
        request["chains"] = request["chains"][:1]
    elif polymer_count == 3:
        request["chains"].append(
            {
                "id": "C",
                "mol_type": 0,
                "residue_names": ["GLU", "ALA", "GLY", "GLY"],
                "indices": [0, 2, 3],
            }
        )
    request["design_chains"] = [chain["id"] for chain in request["chains"]]
    request["target_chains"] = []
    if ligand:
        request["chains"].append(
            {
                "id": "L",
                "mol_type": 3,
                "residue_names": ["MG"],
                "indices": [0],
            }
        )
    expected_full = sum(len(chain["residue_names"]) for chain in request["chains"])
    expected_selected = []
    offset = 0
    for chain in request["chains"]:
        expected_selected.extend(offset + i for i in chain["indices"])
        offset += len(chain["residue_names"])

    class ModelBoundary:
        config = SimpleNamespace(
            lm_encoder=SimpleNamespace(lm_dropout=0.25, per_loop_lm_dropout=False)
        )

        def _compute_lm_hidden_states(self, input_ids, *args, **kwargs):
            assert input_ids.shape == (1, expected_full)
            return torch.arange(expected_full, dtype=torch.float32).reshape(
                1, expected_full, 1, 1
            )

        def __call__(self, **kwargs):
            assert kwargs["lm_hidden_states"].flatten().tolist() == expected_selected
            n = len(expected_selected)
            assert kwargs["msa"].shape == (1, 1, n)
            pae = torch.full((5, n, n), 20.0)
            for sample, error in [(1, 2.0), (2, 1.0), (3, 3.0)]:
                pae[sample] = error
            if polymer_count > 1:
                # A very strong A:B pair must not hide a disconnected chain C.
                pae[0, :3, 3:6] = 0.1
                pae[0, 3:6, :3] = 0.1
            atoms = kwargs["atom_attention_mask"].shape[1]
            return {
                "pae": pae,
                "ptm": torch.tensor([0.1, 0.4, 0.3, 0.9, 0.5]),
                "iptm": torch.tensor([0.01, 0.01, 0.01, 0.01, 0.99]),
                "sample_atom_coords": torch.zeros(5, atoms, 3),
                "plddt": torch.full((5, n), 0.8),
            }

    run_request(ModelBoundary(), builder, request, tmp_path, "cpu")
    result = load_result(tmp_path / "example.design_0.json", fingerprint(request))
    if polymer_count == 1:
        assert result["selected_sample"] == 3
        assert result["score_metric"] == "esmfold2_ptm"
        assert result["metrics"]["esmfold2_ptm"] == pytest.approx(0.9)
        assert "esmfold2_ipsae_min" not in result["metrics"]
        assert result["chain_vs_rest_samples"] == []
    else:
        # With just two chains, the strong A:B sample0 is the correct winner.
        assert result["selected_sample"] == (0 if polymer_count == 2 else 2)
        expected = 1 / 1.01 if polymer_count == 2 else 0.5
        assert result["metrics"]["esmfold2_ipsae_min"] == pytest.approx(expected)
        assert result["score_metric"] == "esmfold2_ipsae_min"
        assert len(result["chain_vs_rest_samples"]) == 5
        if polymer_count == 3:
            assert result["samples"][0]["esmfold2_score"] == 0.0
    assert (
        result["metrics"]["esmfold2_score"] == result["metrics"][result["score_metric"]]
    )
