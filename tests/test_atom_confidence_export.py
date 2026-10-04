"""Exercise token and atom confidence through real feature generation and CIF IO."""
# ruff: noqa: INP001, PLR2004

from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from rdkit import Chem
from test_folding_export_consistency import (
    _features,
    _forward_output,
    _InferenceBoundary,
)

from boltzgen.data.data import Input, Structure
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.modules.confidence import ConfidenceModule
from boltzgen.task.analyze.analyze_utils import get_best_folding_sample
from boltzgen.task.predict.data_from_generated import collate
from boltzgen.task.predict.writer import FoldingWriter


def _real_confidence_features(max_tokens: int | None) -> dict:
    structure, _, _ = Structure.from_feat(
        _features(padded=False, missing_atom=False, ligand=False)
    )
    tokenized = Tokenizer().tokenize(structure)
    gly = Chem.MolFromSmiles("NCC(=O)O")
    for atom, name in zip(gly.GetAtoms(), ["N", "CA", "C", "O", "OXT"]):
        atom.SetProp("name", name)
    conformer = Chem.Conformer(5)
    for index, xyz in enumerate(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 2.0, 0.0),
            (2.0, 1.0, 0.0),
        ]
    ):
        conformer.SetAtomPosition(index, xyz)
    gly.AddConformer(conformer)
    input_data = Input(
        tokens=tokenized.tokens,
        bonds=tokenized.bonds,
        token_to_res=tokenized.token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )
    features = Featurizer().process(
        input_data,
        random=np.random.default_rng(2718),
        molecules={"GLY": gly},
        training=False,
        max_seqs=1,
        max_tokens=max_tokens,
        design=True,
    )
    features["id"] = "contract"
    features["exception"] = False
    features["chain_design_mask"] = features["design_mask"].clone()
    return features


@pytest.mark.parametrize("designfolding", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 32])
@pytest.mark.parametrize("token_level", [False, True])
def test_actual_confidence_granularity_survives_cif_export(  # noqa: PLR0915
    tmp_path: Path,
    designfolding: bool,
    max_tokens: int | None,
    token_level: bool,
) -> None:
    batch = collate([_real_confidence_features(max_tokens)])
    n_atoms = batch["atom_pad_mask"].shape[-1]
    n_tokens = batch["token_pad_mask"].shape[-1]
    assert n_atoms == 32
    assert n_tokens == (2 if max_tokens is None else 32)
    output = _forward_output(batch, 2)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(31415)
            module = ConfidenceModule(
                token_s=8,
                token_z=8,
                pairformer_args={"num_blocks": 0},
                token_level_confidence=token_level,
                confidence_args={},
            ).eval()
            confidence = module(
                s_inputs=torch.randn(1, n_tokens, 8),
                s=torch.randn(1, n_tokens, 8),
                z=torch.randn(1, n_tokens, n_tokens, 8),
                x_pred=output["sample_atom_coords"],
                feats=batch,
                pred_distogram_logits=torch.randn(1, n_tokens, n_tokens, 64),
                multiplicity=2,
                run_sequentially=True,
            )
    finally:
        torch.set_num_threads(previous_threads)
    assert confidence["plddt"].shape == (2, n_tokens if token_level else n_atoms)
    if not token_level:
        assert torch.std(confidence["plddt"][1, :4]).item() > 0.001
    output.update(confidence)
    output["design_to_target_iptm"] = torch.tensor([0.1, 0.9])
    output["design_ptm"] = torch.tensor([0.1, 0.9])
    boundary = _InferenceBoundary(output, 2, mask=False)
    boundary.token_level_confidence = token_level
    boundary.alpha_pae = 0.0
    boundary.predict_args["keys_dict_out"] = []
    prediction = Boltz.predict_step(boundary, batch)
    assert prediction["token_level_confidence"] == token_level

    writer = FoldingWriter(str(tmp_path), designfolding=designfolding)
    writer.write_on_batch_end(prediction=prediction, batch=batch)
    block = gemmi.cif.read_file(
        str(writer.refold_cif_dir / "contract.cif")
    ).sole_block()
    xyz = np.array(
        [list(block.find_values(f"_atom_site.Cartn_{axis}")) for axis in "xyz"],
        dtype=float,
    ).T
    actual_bfactor = np.array(
        block.find_values("_atom_site.B_iso_or_equiv"), dtype=float
    )
    real = batch["atom_pad_mask"][0].bool()
    selected = confidence["plddt"][1]
    atom_values = (
        batch["atom_to_token"][0].float() @ selected if token_level else selected
    )
    np.testing.assert_allclose(xyz, output["sample_atom_coords"][1, real], atol=1e-4)
    np.testing.assert_allclose(actual_bfactor, atom_values[real].numpy(), atol=1e-5)
    qa = np.array(block.find_values("_ma_qa_metric_local.metric_value"), dtype=float)
    np.testing.assert_allclose(qa, atom_values[[0, 4]].numpy() * 100, atol=1e-3)

    with np.load(writer.outdir / "contract.npz") as archive:
        assert "token_level_confidence" not in archive
        assert "plddt" not in archive
        np.testing.assert_array_equal(archive["coords"], prediction["coords"])
        selected_archive = get_best_folding_sample(archive)
        np.testing.assert_array_equal(
            selected_archive["coords"], prediction["coords"][1]
        )

    if token_level:
        legacy_prediction = {
            key: value
            for key, value in prediction.items()
            if key != "token_level_confidence"
        }
        legacy_writer = FoldingWriter(
            str(tmp_path / "legacy"), designfolding=designfolding
        )
        legacy_writer.write_on_batch_end(prediction=legacy_prediction, batch=batch)
        legacy = gemmi.cif.read_file(
            str(legacy_writer.refold_cif_dir / "contract.cif")
        ).sole_block()
        legacy_bfactor = np.array(
            legacy.find_values("_atom_site.B_iso_or_equiv"), dtype=float
        )
        np.testing.assert_allclose(legacy_bfactor, actual_bfactor, atol=1e-5)
