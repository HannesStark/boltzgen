import os
from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
import pickle
import pydssp
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from boltzgen.data import const
from boltzgen.data.data import Structure, convert_ccd

from boltzgen.data.feature.featurizer import (
    repopulate_res_type,
    res_from_atom14,
    res_from_atom37,
    res_all_gly,
)
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.validation.validator import Validator

import random
import string


class DesignValidator(Validator):
    """Validation step implementation for Design."""

    def __init__(
        self,
        val_names: List[str],
        confidence_prediction: bool = False,
        atom14: bool = True,
        atom37: bool = False,
        backbone_only: bool = False,
        inverse_fold: bool = False,
    ) -> None:
        super().__init__(
            val_names=val_names, confidence_prediction=confidence_prediction
        )
        self.backbone_only = backbone_only
        self.inverse_fold = inverse_fold
        # Design Metrics
        self.seq_metric = nn.ModuleDict()
        for t in const.fake_atom_placements.keys():
            self.seq_metric[f"design_{t}"] = MeanMetric()
            self.seq_metric[f"data_{t}"] = MeanMetric()
        self.seq_metric["design_seq_recovery"] = MeanMetric()

        self.ss_metric = nn.ModuleDict()
        self.ss_metric["loop"] = MeanMetric()
        self.ss_metric["helix"] = MeanMetric()
        self.ss_metric["sheet"] = MeanMetric()

        self.ss_metric["loop_native"] = MeanMetric()
        self.ss_metric["helix_native"] = MeanMetric()
        self.ss_metric["sheet_native"] = MeanMetric()

        self.atom14 = atom14
        self.atom37 = atom37
        self.used_stems = set()

    def process(
        self,
        model: LightningModule,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx_dataset: int,
        dataloader_idx: int,
        n_samples: int,
        batch_idx,
    ) -> None:
        # Take common step for folding and confidence

        if not self.inverse_fold:
            self.common_val_step(model, batch, out, idx_dataset)

        feat_masked = out["feat_masked"]

        generated_dir = f"{model.trainer.default_root_dir}/generated/epoch{model.current_epoch}_step{model.global_step}"

        self.design_val_step(
            model, batch, feat_masked, out, n_samples, batch_idx, generated_dir
        )

    def on_epoch_end(self, model):
        # Take the common epoch end for folding and affinity
        self.common_on_epoch_end(model)

        # Take the affinity specific epoch end
        self.on_epoch_end_design(model)

    def design_val_step(
        self,
        model: LightningModule,
        batch: Dict[str, torch.Tensor],
        feat_masked: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        n_samples: int,
        batch_idx: int,
        generated_dir: str = "generated",
        invalid_token: str = "UNK",
        logname: str = "val",
    ) -> None:
        """Run a validation step.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : Dict[str, torch.Tensor]
            The batch input.
        out : Dict[str, torch.Tensor]
            The output of the model.

        """

        for n in range(n_samples):
            # get structure for all generated coords
            sample, native = {}, {}
            for k in feat_masked.keys():
                if k == "coords":
                    sample[k] = out["sample_atom_coords"][n]
                    native[k] = batch[k][0]
                else:
                    sample[k] = feat_masked[k][0]
                    native[k] = batch[k][0]

            # Design metrics and sample writing
            try:
                if self.atom14:
                    sample = res_from_atom14(sample, invalid_token=invalid_token)
                elif self.atom37:
                    sample = res_from_atom37(sample, invalid_token=invalid_token)
                elif self.backbone_only:
                    sample = res_all_gly(sample)


                if self.backbone_only and not self.inverse_fold:
                    sample = repopulate_res_type(sample)
                    native = repopulate_res_type(native)

                design_mask = batch["design_mask"][0].bool()
                assert design_mask.sum() == sample["design_mask"].sum()

                if self.inverse_fold:
                    token_ids = torch.argmax(sample["res_type"], dim=-1)
                    tokens = [const.tokens[i] for i in token_ids]
                    ccds = [convert_ccd(token) for token in tokens]

                    ccds = torch.tensor(ccds).to(sample["res_type"])
                    sample["ccd"][design_mask] = ccds[design_mask]

                structure, _, _ = Structure.from_feat(sample)
                str_native, _, _ = Structure.from_feat(native)

                # Write structure to cif
                os.makedirs(generated_dir, exist_ok=True)
                basename = f"{generated_dir}/sample{n}_batch{batch_idx}_rank{model.trainer.global_rank}_{batch['id'][0]}"
                gen_path = f"{basename}.cif"
                native_path = f"{basename}_native.cif"

                atom_design_mask = (
                    sample["atom_to_token"].float()
                    @ sample["design_mask"].unsqueeze(-1).float()
                )
                atom_design_mask = atom_design_mask.squeeze().bool()
                bfactor = atom_design_mask[sample["atom_pad_mask"].bool()].float()
                structure.atoms["bfactor"] = bfactor.cpu().numpy()
                str_native.atoms["bfactor"] = bfactor.cpu().numpy()

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
                with open(gen_path, "w") as f:
                    f.write(to_mmcif(structure))
                with open(native_path, "w") as f:
                    f.write(to_mmcif(str_native))

                # Write metadata
                metadata_path = f"{basename}.npz"
                np.savez_compressed(
                    metadata_path,
                    design_mask=design_mask[sample["token_pad_mask"].bool()]
                    .cpu()
                    .numpy(),
                    mol_type=sample["mol_type"][sample["token_pad_mask"].bool()]
                    .cpu()
                    .numpy(),
                    native_atom_design_mask=atom_design_mask[
                        ~native["fake_atom_mask"].bool()
                    ]
                    .cpu()
                    .numpy(),
                    design_atom_design_mask=atom_design_mask[
                        ~sample["fake_atom_mask"].bool()
                    ]
                    .cpu()
                    .numpy(),
                )

                # Compute metrics
                design_mask = sample["design_mask"].bool()
                if design_mask.sum() > 5:
                    # Compute res type distribution
                    design_seq = torch.argmax(sample["res_type"], dim=-1)[design_mask]
                    true_seq = torch.argmax(native["res_type"], dim=-1)[design_mask]
                    self.seq_metric["design_seq_recovery"].update(
                        (design_seq == true_seq).float().mean()
                    )
                    for t in const.fake_atom_placements.keys():
                        self.seq_metric[f"design_{t}"].update(
                            (design_seq == const.token_ids[t]).float().mean()
                        )
                        self.seq_metric[f"data_{t}"].update(
                            (true_seq == const.token_ids[t]).float().mean()
                        )

                    # Compute secondary structure distribution. First get backbone then use pydssp to compute.
                    bb_design_mask = (
                        sample["atom_pad_mask"].bool()
                        & atom_design_mask
                        & sample["backbone_mask"].bool()
                    )
                    bb = sample["coords"][bb_design_mask].reshape(-1, 4, 3)
                    bb_native = native["coords"][0][bb_design_mask].reshape(-1, 4, 3)

                    # Run DSSP only if at least two backbone residues are present

                    if bb.shape[0] >= 2:
                        # 0: loop,  1: alpha-helix,  2: beta-strand
                        dssp = pydssp.assign(bb, out_type="index")
                        self.ss_metric["loop"].update((dssp == 0).float().mean())
                        self.ss_metric["helix"].update((dssp == 1).float().mean())
                        self.ss_metric["sheet"].update((dssp == 2).float().mean())

                        dssp_native = pydssp.assign(bb_native, out_type="index")
                        self.ss_metric["loop_native"].update(
                            (dssp_native == 0).float().mean()
                        )
                        self.ss_metric["helix_native"].update(
                            (dssp_native == 1).float().mean()
                        )
                        self.ss_metric["sheet_native"].update(
                            (dssp_native == 2).float().mean()
                        )

                return True
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()  # noqa: T201
                print(
                    f"Validation structure writing failed on {batch['id'][0]} with error {e}. Skipping."
                )  # noqa: T201
                return False

    def on_epoch_end_design(self, model, logname: str = "val"):
        # Design Metrics
        # compute residue distribution metrics
        design_freqs = []
        data_freqs = []
        for t in const.fake_atom_placements.keys():
            design_freqs.append(self.seq_metric[f"design_{t}"].compute().cpu())
            data_freqs.append(self.seq_metric[f"data_{t}"].compute().cpu())
            model.log(
                f"{logname}/design_seq_recovery",
                self.seq_metric["design_seq_recovery"].compute(),
                prog_bar=False,
            )
        for v in self.seq_metric.values():
            v.reset()

        # Make residue distribution plot
        x = np.arange(len(const.fake_atom_placements.keys()))
        width = 0.15
        _, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, design_freqs, width, label="Design frequency")
        ax.bar(x + width / 2, data_freqs, width, label="Data frequency")
        ax.set_xlabel("Res Type")
        ax.set_ylabel("Probability")
        ax.set_title("Res Type distributions")
        ax.set_xticks(x)
        ax.set_xticklabels(const.fake_atom_placements.keys())
        ax.legend()
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        img_dir = Path(f"{model.trainer.default_root_dir}/images")
        img_dir.mkdir(exist_ok=True)
        plt.savefig(img_dir / f"res_dist{model.current_epoch}.png")
        plt.close()
        model.log_image(
            f"{logname}/res_dist", img_dir / f"res_dist{model.current_epoch}.png"
        )

        # Compute secondary structure distribution and log
        ss_dist = []
        ss_dist_native = []
        for k, v in self.ss_metric.items():
            metric = v.compute().cpu()
            model.log(f"{logname}/{k}", metric, prog_bar=False)
            if "_native" in k:
                ss_dist_native.append(metric)
            else:
                ss_dist.append(metric)
        for v in self.ss_metric.values():
            v.reset()

        # Make secondary structure distribution plot
        x = np.arange(3)
        width = 0.15
        _, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, ss_dist, width, label="Designed")
        ax.bar(x + width / 2, ss_dist_native, width, label="Native data")
        ax.set_xlabel("Secondary Structure type")
        ax.set_ylabel("Frequency")
        ax.set_title("Secondary Structure distributions")
        ax.set_xticks(x)
        ax.set_xticklabels(["loop", "helix", "sheet"])
        ax.legend()
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        img_dir = Path(f"{model.trainer.default_root_dir}/images")
        img_dir.mkdir(exist_ok=True)
        plt.savefig(img_dir / f"ss_dist{model.current_epoch}.png")
        plt.close()
        self.log_image(
            f"{logname}/ss_dist", img_dir / f"ss_dist{model.current_epoch}.png", model
        )

    def log_image(self, name, path, model):
        if model.logger is not None:
            try:
                model.logger.log_image(name, images=[str(path)])
            except:
                import traceback

                traceback.print_exc()  # noqa: T201
                print(f"Image logging failed for {name} {str(path)}.")  # noqa: T201
