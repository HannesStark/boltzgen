import os
import pickle
import json
from pathlib import Path
import tempfile
from typing import Dict, List

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from tqdm import tqdm

from boltzgen.data import const
from boltzgen.data.source_context import update_designed
from boltzgen.data.data import (
    Structure,
    convert_ccd,
)
from boltzgen.data.feature.featurizer import (
    res_from_atom14,
    res_from_atom37,
    res_all_gly,
)
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.data.write.pdb import to_pdb
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.model.modules.masker import BoltzMasker


def _existing_output_alias(
    output_dir: Path, source_stem: str, sample_index: int, file_suffix: str
) -> tuple[str | None, bool]:
    """Find an output using this sample index despite changed zero padding."""
    output_stems = {
        path.stem
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".cif", ".npz"}
    }
    matches = []
    for output_stem in output_stems:
        normalized_stem = output_stem
        if file_suffix and normalized_stem.endswith(file_suffix):
            normalized_stem = normalized_stem[: -len(file_suffix)]
        if normalized_stem == source_stem:
            output_index = 0
        else:
            prefix = f"{source_stem}_"
            sample = (
                normalized_stem[len(prefix) :]
                if normalized_stem.startswith(prefix)
                else ""
            )
            if not sample.isdigit():
                continue
            output_index = int(sample)
        if output_index == sample_index:
            matches.append(output_stem)

    for output_stem in sorted(matches):
        if (
            (output_dir / f"{output_stem}.cif").is_file()
            and (output_dir / f"{output_stem}.npz").is_file()
        ):
            return output_stem, True
    return (sorted(matches)[0], False) if matches else (None, False)


def _write_npz_atomically(path: str | Path, **arrays) -> None:
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class FoldingWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(self, design_dir: str, designfolding: bool = False) -> None:
        super().__init__(write_interval="batch")
        self.designfolding = designfolding
        if design_dir is not None:
            self.init_outdir(design_dir)

    def init_outdir(self, design_dir):
        self.outdir = Path(design_dir) / (
            const.folding_design_dirname
            if self.designfolding
            else const.folding_dirname
        )
        self.refold_cif_dir = Path(design_dir) / (
            const.refold_design_cif_dirname
            if self.designfolding
            else const.refold_cif_dirname
        )
        self.refold_cif_dir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.failed = 0

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        """Write the predictions to disk."""
        exception = prediction.get("exception", False)
        if isinstance(exception, list):
            exception = exception[0]
        if exception:
            self.failed += 1
            return
        skip = prediction.get("skip", False)
        if isinstance(skip, list):
            skip = skip[0]
        if skip:
            return

        # Folding preserves the input molecule identities even when the model
        # masks them internally. Keep those identities in both NPZ and CIF.
        prediction = prediction.copy()
        for key in (
            "res_type",
            "ccd",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
        ):
            if key in batch:
                prediction[key] = batch[key]

        pred_dict = {}
        for key, value in prediction.items():
            # check object is tensor
            if key in const.eval_keys:
                pred_dict[key] = value.cpu().numpy()
        # Match the sample selected by analyze_utils.get_best_folding_sample.
        confidence = (
            0.8 * pred_dict["design_to_target_iptm"] + 0.2 * pred_dict["design_ptm"]
        )
        best_idx = np.argmax(confidence)

        prediction_out = {}
        for k in prediction:
            # Coordinates and pLDDT have a sample axis; structure features have
            # a batch axis and are shared by all samples of this input.
            if k in ("coords", "plddt"):
                prediction_out[k] = prediction[k][best_idx]
            elif k in ("exception", "skip", "token_level_confidence"):
                continue
            else:
                prediction_out[k] = prediction[k][0]

        # Write structure
        structure, _, _ = Structure.from_feat(prediction_out)
        if prediction.get("token_level_confidence", True):
            plddt_atom = (
                prediction_out["atom_to_token"].float()
                @ prediction_out["plddt"].float()
            )
        else:
            plddt_atom = prediction_out["plddt"].float()
        structure.atoms["bfactor"] = (
            plddt_atom[prediction_out["atom_pad_mask"].bool()].float().cpu().numpy()
        )
        cif_text = to_mmcif(structure)
        # Publish the complete archive last. An interrupted overwrite must not
        # leave an old completion marker paired with a new or partial CIF.
        archive_path = self.outdir / f"{batch['id'][0]}.npz"
        archive_path.unlink(missing_ok=True)
        with open(self.refold_cif_dir / f"{batch['id'][0]}.cif", "w") as cif_file:
            cif_file.write(cif_text)
        _write_npz_atomically(archive_path, **pred_dict)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        print(f"Number of failed structure predictions: {self.failed}")  # noqa: T201


class AffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        design_dir: str,
    ) -> None:
        super().__init__(write_interval="batch")
        if design_dir is not None:
            self.init_outdir(design_dir)

    def init_outdir(self, design_dir):
        self.outdir = Path(design_dir) / const.affinity_dirname
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.failed = 0

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        """Write the predictions to disk."""
        exception = prediction.get("exception", False)
        if isinstance(exception, list):
            exception = exception[0]
        if exception:
            self.failed += 1
            return
        skip = prediction.get("skip", False)
        if isinstance(skip, list):
            skip = skip[0]
        if skip:
            return

        pred_dict = {}
        for key, value in prediction.items():
            # check object is tensor
            if key in const.eval_keys:
                pred_dict[key] = value.cpu().numpy()
        _write_npz_atomically(self.outdir / f"{batch['id'][0]}.npz", **pred_dict)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        print(f"Number of failed affinity predictions: {self.failed}")  # noqa: T201


class DesignWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        output_dir: str,
        res_atoms_only: bool,
        save_traj: bool = False,
        save_x0_traj: bool = False,
        atom14: bool = True,
        atom37: bool = False,
        backbone_only: bool = False,
        inverse_fold: bool = False,
        file_suffix: str = "",
        write_native: bool = True,
        design: bool = True,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.mol_dir = Path(output_dir) / const.molecules_dirname
        self.mol_dir.mkdir(parents=True, exist_ok=True)
        self.save_traj = save_traj
        self.save_x0_traj = save_x0_traj
        self.res_atoms_only = res_atoms_only
        self.file_suffix = file_suffix
        self.failed = 0
        self.write_native = write_native
        self.design = design

        # Create the output directories
        self.atom14 = atom14
        self.atom37 = atom37
        self.inverse_fold = inverse_fold
        self.backbone_only = backbone_only
        self.used_stems = set()
        self.init_outdir(output_dir)

    def init_outdir(self, outdir):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(  # noqa: PLR0915
        self,
        trainer: Trainer = None,  # noqa: ARG002
        pl_module: LightningModule = None,  # noqa: ARG002
        prediction: Dict[str, Tensor] = None,
        batch_indices: List[int] = None,  # noqa: ARG002
        batch: Dict[str, Tensor] = None,
        batch_idx: int = None,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
        sample_id: str = None,
    ) -> None:
        if prediction["exception"]:
            self.failed += 1
            return
        n_samples, _, _ = prediction["coords"].shape

        # TODO: remove this which is only here for temporary backward compatibility
        masker = BoltzMasker(mask=True, mask_backbone=False)
        feat_masked = masker(batch)
        prediction["ref_element"] = feat_masked["ref_element"]
        prediction["ref_atom_name_chars"] = feat_masked["ref_atom_name_chars"]
        """Write the predictions to disk."""
        # Check for extra molecules
        if batch["extra_mols"] is not None:
            extra_mols = batch["extra_mols"][0]
            for k, v in extra_mols.items():
                with open(self.mol_dir / f"{k}.pkl", "wb") as f:
                    pickle.dump(v, f)

        # write samples to disk
        datamodule = getattr(trainer, "datamodule", None)
        cfg = getattr(datamodule, "cfg", None)
        resume_inverse_fold = (
            self.inverse_fold
            and getattr(datamodule, "skip_existing", False)
            and getattr(datamodule, "skip_existing_kind", None) == "inverse_fold"
        )
        for n in range(n_samples):
            if sample_id is not None:
                file_name = f"{sample_id}_{n}{self.file_suffix}"
            else:
                stem = str(batch["id"][0])
                multiplicity = getattr(cfg, "multiplicity", 1)
                total_files = multiplicity * n_samples
                sample_idx = (
                    int(batch["data_sample_idx"][0])
                    if "data_sample_idx" in batch
                    else 0
                )
                global_idx = sample_idx * n_samples + n

                if total_files > 1:
                    num_digits = len(str(total_files - 1))
                    file_name = (
                        f"{stem}_{global_idx:0{num_digits}d}{self.file_suffix}"
                    )
                else:
                    file_name = f"{stem}{self.file_suffix}"

            native_path = f"{self.outdir}/{file_name}_native.cif"
            gen_path = f"{self.outdir}/{file_name}.cif"
            metadata_path = f"{self.outdir}/{file_name}.npz"
            if resume_inverse_fold:
                # Completed siblings must keep their sequence and cached refolds
                # when another item makes the whole backbone run again.
                if Path(gen_path).is_file() and Path(metadata_path).is_file():
                    continue
            elif getattr(cfg, "skip_existing", False):
                output_dir = Path(self.outdir)
                if sample_id is None:
                    alias, complete = _existing_output_alias(
                        output_dir, stem, global_idx, self.file_suffix
                    )
                    if complete:
                        continue
                    if alias is not None:
                        file_name = alias
                        native_path = f"{self.outdir}/{file_name}_native.cif"
                        gen_path = f"{self.outdir}/{file_name}.cif"
                        metadata_path = f"{self.outdir}/{file_name}.npz"
                elif Path(gen_path).is_file() and Path(metadata_path).is_file():
                    continue
            # NPZ is the completion marker for the CIF/metadata pair.
            Path(metadata_path).unlink(missing_ok=True)

            # get structure for all generated coords
            sample, native = {}, {}

            for k in set(prediction.keys()) & set(batch.keys()):
                if k == "coords":
                    native[k] = batch[k][0][0].unsqueeze(0)
                    sample[k] = prediction[k][n]
                    
                if k in const.token_features:
                    sample[k] = prediction[k][0]
                    native[k] = batch[k][0]
                elif k in const.atom_features:
                    if k == "coords":
                        native[k] = batch[k][0][0].unsqueeze(0)
                        sample[k] = prediction[k][n]
                    else:
                        native[k] = batch[k][0]
                        sample[k] = prediction[k][0]
                elif k == "exception":
                    sample[k] = prediction[k]
                    native[k] = batch[k]
                else:
                    native[k] = batch[k][0]
                    sample[k] = prediction[k][0]
                    native[k] = batch[k][0]

            if self.atom14:
                sample = res_from_atom14(sample)
            elif self.atom37:
                sample = res_from_atom37(sample)
            elif self.backbone_only:
                sample = res_all_gly(sample)


            design_mask = batch["design_mask"][0].bool()
            assert design_mask.sum() == sample["design_mask"].sum()

            if self.inverse_fold:
                token_ids = torch.argmax(sample["res_type"], dim=-1)
                tokens = [const.tokens[i] for i in token_ids]
                ccds = [convert_ccd(token) for token in tokens]

                ccds = torch.tensor(ccds).to(sample["res_type"])
                sample["ccd"][design_mask] = ccds[design_mask]

            try:
                structure, _, _ = Structure.from_feat(sample)
                str_native, _, _ = Structure.from_feat(native)

                # design mask bfactor
                design_mask = batch["design_mask"][0].float()
                atom_design_mask = (
                    sample["atom_to_token"].float() @ design_mask.unsqueeze(-1).float()
                )
                design_mask = native["design_mask"].float()

                atom_design_mask = atom_design_mask.squeeze().bool()
                bfactor = atom_design_mask * 100

                # binding type bfactor
                binding_type = batch["binding_type"][0].float()
                atom_binding_type = (
                    sample["atom_to_token"].float() @ binding_type.unsqueeze(-1).float()
                )

                atom_binding_type = atom_binding_type.squeeze().int()
                bfactor[atom_binding_type == const.binding_type_ids["BINDING"]] = 60

                bfactor = atom_design_mask[sample["atom_pad_mask"].bool()].float()
                str_native.atoms["bfactor"] = bfactor.cpu().numpy()
                structure.atoms["bfactor"] = bfactor.cpu().numpy()

                # Add dummy (0-coord) design side chains if inverse fold
                if self.inverse_fold:
                    atom_design_mask_no_pad = atom_design_mask[
                        native["atom_pad_mask"].bool()
                    ]
                    res_design_mask = np.array(
                        [
                            all(
                                atom_design_mask_no_pad[
                                    res["atom_idx"] : res["atom_idx"] + res["atom_num"]
                                ]
                            )
                            for res in structure.residues
                        ]
                    )
                    structure = Structure.add_side_chains(
                        structure, residue_mask=res_design_mask
                    )

                if self.write_native:
                    open(native_path, "w").write(to_mmcif(str_native))

                pred_binding_mask = prediction["binding_type"][0].cpu().bool().numpy()
                if self.design:
                    chain_design_mask = (
                        prediction["chain_design_mask"][0].cpu().bool().numpy()
                    )
                pred_design_mask = prediction["design_mask"][0].cpu().bool().numpy()
                design_color_features = np.ones_like(pred_binding_mask) * 0.8
                design_color_features[pred_binding_mask] = 1.0
                if self.design:
                    design_color_features[chain_design_mask] = 0.0
                design_color_features[pred_design_mask] = 0.6

                # Create a mask to identify unique token-to-res mappings.
                # This is for small molecules where multiple tokens can be mapped to the same residue.
                token_to_res = prediction["token_to_res"][0].cpu().numpy()
                unique_mask = np.ones_like(token_to_res, dtype=bool)
                unique_mask[1:] = token_to_res[1:] != token_to_res[:-1]
                design_color_features = design_color_features[unique_mask]
                open(gen_path, "w").write(
                    to_mmcif(
                        structure,
                        design_coloring=True,
                        color_features=design_color_features,
                    )
                )

                # Write metadata
                token_mask = sample["token_pad_mask"].bool()

                context = json.loads(batch.get("source_context", ["null"])[0])
                if context is not None:
                    context = update_designed(context, structure)

                # Build metadata dict with required fields
                metadata_dict = {
                    "source_context": np.asarray(json.dumps(context)),
                    "design_mask": design_mask[token_mask].cpu().numpy(),
                    "mol_type": sample["mol_type"][token_mask].cpu().numpy(),
                    "ss_type": sample["ss_type"][token_mask].cpu().numpy(),
                    "symmetric_group": batch["symmetric_group"][0][token_mask].cpu().numpy(),
                    "token_resolved_mask": sample["token_resolved_mask"][token_mask].cpu().numpy(),
                    "binding_type": binding_type[token_mask].cpu().numpy(),
                }

                # Add optional fields only if they have valid values (avoid None -> object array)
                if "inverse_fold_design_mask" in sample:
                    metadata_dict["inverse_fold_design_mask"] = (
                        sample["inverse_fold_design_mask"][token_mask].cpu().numpy()
                    )

                # Per-residue amino acid constraints (for inverse folding step)
                # Only save if constraints exist AND have non-zero values
                if "aa_constraint_mask" in batch:
                    aa_mask = batch["aa_constraint_mask"][0]
                    if aa_mask.any():  # Only save if there are actual constraints
                        metadata_dict["aa_constraint_mask"] = aa_mask[token_mask].cpu().numpy()

                _write_npz_atomically(metadata_path, **metadata_dict)

                # Write trajectories
                if self.save_traj:
                    trajs = torch.stack(prediction["coords_traj"], dim=1)
                    traj = trajs[n]
                    aligned = [traj[0]]
                    for frame in traj[1:]:
                        with torch.autocast("cuda", enabled=False):
                            aligned.append(
                                weighted_rigid_align(
                                    frame.float().unsqueeze(0),
                                    aligned[-1].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                )
                                .to(frame)
                                .squeeze()
                            )

                    pdbs = []
                    all_coords = []
                    ensemble = []
                    atom_idx = 0
                    for idx, frame in tqdm(
                        enumerate(aligned), desc="Writing traj.", total=len(aligned)
                    ):
                        sample["coords"] = frame
                        if self.atom14:
                            sample = res_from_atom14(sample)
                        elif self.atom37:
                            sample = res_from_atom37(sample)
                        else:
                            raise ValueError("Either atom14 or atom37 must be true")

                        str_frame, _, _ = Structure.from_feat(sample)
                        pdbs.append(to_pdb(str_frame))
                        all_coords.append(str_frame.coords)
                        ensemble.append(
                            (
                                atom_idx,
                                len(str_frame.coords),
                            )
                        )
                        atom_idx += len(str_frame.coords)

                    open(self.outdir / f"{file_name}_traj.pdb", "w").write(
                        self.combine_pdb_models(pdbs)
                    )

                # Write x0 trajectories
                if self.save_x0_traj:
                    trajs = torch.stack(prediction["x0_coords_traj"], dim=1)
                    traj = trajs[n]
                    aligned = [traj[0]]
                    for frame in traj[1:]:
                        with torch.autocast("cuda", enabled=False):
                            aligned.append(
                                weighted_rigid_align(
                                    frame.float().unsqueeze(0),
                                    aligned[-1].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                    sample["atom_pad_mask"].float().unsqueeze(0),
                                )
                                .to(frame)
                                .squeeze()
                            )

                    pdbs = []
                    all_coords = []
                    ensemble = []
                    atom_idx = 0
                    for idx, frame in tqdm(
                        enumerate(aligned), desc="Writing x0 traj.", total=len(aligned)
                    ):
                        sample["coords"] = frame
                        if self.atom14:
                            sample = res_from_atom14(sample)
                        elif self.atom37:
                            sample = res_from_atom37(sample)
                        else:
                            raise ValueError("Either atom14 or atom37 must be true")

                        str_frame, _, _ = Structure.from_feat(sample)
                        pdbs.append(to_pdb(str_frame))
                        all_coords.append(str_frame.coords)
                        ensemble.append(
                            (
                                atom_idx,
                                len(str_frame.coords),
                            )
                        )
                        atom_idx += len(str_frame.coords)

                    open(self.outdir / f"{file_name}_x0_traj.pdb", "w").write(
                        self.combine_pdb_models(pdbs)
                    )

            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()  # noqa: T201
                msg = f"predict/writer.py: Validation structure writing failed on {batch['id'][0]} with error {e}. Skipping."
                print(msg)

    def combine_pdb_models(self, pdb_strings):
        combined_pdb = ""
        model_number = 1

        for pdb in pdb_strings:
            # Add a model number at the start of each model
            combined_pdb += f"MODEL     {model_number}\n"
            combined_pdb += pdb.split("\nEND")[0]
            combined_pdb += "\nENDMDL\n"  # End of model marker
            model_number += 1

        return combined_pdb

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
