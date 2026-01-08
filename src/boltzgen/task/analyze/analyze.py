from boltzgen.utils.quiet import quiet_startup


quiet_startup()
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import copy
import multiprocessing
import numbers
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List
import subprocess
import re
import json

from boltzgen.task.analyze.analyze_utils import (
    TARGET_ID_RE,
    calc_hydrophobicity,
    compute_liability_metrics,
    compute_novelty_foldseek,
    compute_rmsd,
    compute_ss_metrics,
    get_best_folding_sample,
    get_delta_sasa,
    get_fold_metrics,
    get_motif_set,
    count_noncovalents,
    largest_hydrophobic_patch_area,
    make_histogram,
    save_design_only_structure_to_cif,
    save_design_only_structure_to_pdb,
    vendi_scores,
    vendi_sequences,
)
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.max_open_warning"] = 100
import pydssp
import rdkit
from boltzgen.task.predict.data_from_generated import FromGeneratedDataModule

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from boltzgen.data import const
from boltzgen.task.task import Task
from boltzgen.data.data import Structure
from boltzgen.data.write.mmcif import to_mmcif


class Analyze(Task):
    """
    The Analyze step of the BoltzGen pipeline.
    It computes various metrics on CPU.
    It aggregates metrics from the Folding, Designfolding, and Affinity predictions step.
    It writes metrics into a single csv file which is then used in the filter task.
    It also creates a pickle file of sequences and structures for similarity assessments during filtering.
    (needed for quality-diversity optimization algorithm)
    """

    def __init__(
        self,
        name: str,
        data: FromGeneratedDataModule,
        design_dir: str = None,
        backbone_fold_metrics: bool = False,
        allatom_fold_metrics: bool = True,
        affinity_metrics: bool = False,
        noncovalents_original: bool = False,
        noncovalents_refolded: bool = False,
        diversity_original: bool = False,
        diversity_refolded: bool = False,
        diversity_per_target_original: bool = False,
        diversity_per_target_refolded: bool = False,
        novelty_original: bool = False,
        novelty_refolded: bool = False,
        novelty_per_target_original: bool = False,
        novelty_per_target_refolded: bool = False,
        delta_sasa_original: bool = False,
        delta_sasa_refolded: bool = False,
        largest_hydrophobic: bool = False,
        largest_hydrophobic_refolded: bool = False,
        disulfide_quality: bool = False,
        free_cys: bool = False,
        compute_lddts: bool = True,  # computing LDDTs takes ~5-15 sec so it is optional
        run_clustering: bool = False,
        native: bool = False,
        sequence_recovery: bool = False,
        ss_conditioning_metrics: bool = False,
        liability_analysis: bool = False,
        liability_modality: str = "antibody",
        liability_peptide_type: str = "linear",
        debug: bool = False,
        wandb: Optional[Dict[str, Any]] = None,
        slurm: bool = False,
        diversity_subset: int = None,
        num_processes: int = 1,
        foldseek_db: str = "/data/rbg/users/hstark/proteinblobs/data/foldseek_pdb/pdb",
        foldseek_binary: str = "/data/rbg/users/hstark/foldseek/bin/foldseek",
        skip_specific_ids: List[str] = None,
        designfolding_metrics: bool = False,
    ) -> None:
        """Initialize the task.

        Parameters
        ----------
        fold_metrics : bool,
            Compute folding metrics and assume that the folding directory exists.
        """
        super().__init__()
        self.name = name
        self.num_processes = num_processes
        self.foldseek_db = Path(foldseek_db)
        self.foldseek_binary = foldseek_binary
        self.skip_specific_ids = set(skip_specific_ids or [])
        self.data = data
        self.noncovalents_original = noncovalents_original
        self.noncovalents_refolded = noncovalents_refolded
        self.diversity_original = diversity_original
        self.diversity_refolded = diversity_refolded
        self.diversity_per_target_original = diversity_per_target_original
        self.diversity_per_target_refolded = diversity_per_target_refolded
        self.novelty_original = novelty_original
        self.novelty_refolded = novelty_refolded
        self.novelty_per_target_original = novelty_per_target_original
        self.novelty_per_target_refolded = novelty_per_target_refolded
        self.delta_sasa_original = delta_sasa_original
        self.delta_sasa_refolded = delta_sasa_refolded
        self.largest_hydrophobic = largest_hydrophobic
        self.largest_hydrophobic_refolded = largest_hydrophobic_refolded
        self.compute_lddts = compute_lddts
        self.run_clustering = run_clustering
        self.affinity_metrics = affinity_metrics
        self.fold_metrics = backbone_fold_metrics or allatom_fold_metrics
        self.backbone_fold_metrics = backbone_fold_metrics
        self.allatom_fold_metrics = allatom_fold_metrics
        self.designfolding_metrics = designfolding_metrics
        self.disulfide_quality = disulfide_quality
        self.free_cys = free_cys
        self.native = native
        self.sequence_recovery = sequence_recovery
        self.ss_conditioning_metrics = ss_conditioning_metrics
        self.liability_analysis = liability_analysis
        self.liability_modality = liability_modality
        self.liability_peptide_type = liability_peptide_type
        self.debug = debug
        self.wandb = wandb
        self.slurm = slurm
        self.diversity_subset = diversity_subset

        # Prevent each worker process from spawning its own multithreaded pools
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        if design_dir is not None:
            self.init_datasets(design_dir, load_dataset=False)

        # Check that native structure is available if native metrics are desired
        if self.native and not self.data.return_native:
            msg = "native=True requires return_native=True in data config."
            raise ValueError(msg)
        if self.sequence_recovery and not self.native:
            msg = "sequence_recovery=True requires native structure (native=True)."
            raise ValueError(msg)

        self.bindsite_adherence_thresholds = [3, 4, 5, 6, 7, 8, 9]

    def init_datasets(self, design_dir: str, load_dataset: bool = False):
        self.design_dir = Path(design_dir)

        if load_dataset:
            self.data.init_dataset(design_dir, skip_specific_ids=self.skip_specific_ids)

        self.des_pdb_dir = self.design_dir / "des_pdbs"
        self.des_pdb_dir.mkdir(parents=True, exist_ok=True)
        self.des_refold_pdb_dir = self.design_dir / "des_refold_pdbs"
        self.des_refold_pdb_dir.mkdir(parents=True, exist_ok=True)
        self.refold_cif_dir = self.design_dir / const.refold_cif_dirname
        self.refold_cif_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.design_dir / const.metrics_dirname
        self.metrics_dir.mkdir(exist_ok=True, parents=True)

    def run_parallel(self, num, num_processes):
        """
        Run tasks  in parallel. If a worker crashes and the pool breaks,
        restart a fresh pool and only rerun tasks that truly didn't finish.
        """
        ctx = multiprocessing.get_context("spawn")

        all_task_ids = set(range(num))
        completed_task_ids = set()
        sample_ids = []

        pbar = tqdm(total=num, desc="Processing samples")

        while completed_task_ids != all_task_ids:
            remaining = sorted(all_task_ids - completed_task_ids)

            try:
                with ProcessPoolExecutor(
                    max_workers=num_processes, mp_context=ctx
                ) as ex:
                    fut2idx = {ex.submit(self.compute_metrics, i): i for i in remaining}

                    # Iterate over futures that actually *completed* (finished or raised)
                    for f in as_completed(fut2idx):
                        idx = fut2idx[f]
                        try:
                            sid = f.result()
                            if sid is not None:
                                sample_ids.append(sid)
                            # Count successful completion
                            completed_task_ids.add(idx)
                            pbar.update(1)

                        except BrokenProcessPool:
                            # Pool is dead, mark this idx completed.
                            # Let the outer except restart a fresh pool for all unfinished.
                            completed_task_ids.add(idx)
                            pbar.update(1)
                            raise

            except BrokenProcessPool:
                print("\nPOOL BROKEN: A worker died. Restarting with remaining tasks…")
                # Nothing else to do: the while-loop will retry only the unfinished tasks.
                continue

        pbar.close()
        return sample_ids

    def run(self, config=None, run_prediction=False):
        self.distribute_tasks()
        self.aggregate_metrics()

    def distribute_tasks(self):
        # The rdkit thing is necessary to make multiprocessing with the rdkit molecules work.
        rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)

        # Compute metrics and write them to disk
        sample_ids = []
        num = len(self.data.predict_set)
        if num == 0:
            msg = "There were 0 samples to compute metrics for. Skipping the distribute_tasks step that calls compute_metrics"
            print(msg)
            return
        num_processes = min(self.num_processes, multiprocessing.cpu_count())
        if num_processes == 1:
            for idx in tqdm(range(num)):
                sample_id = self.compute_metrics(idx)
                if sample_id is not None:
                    sample_ids.append(sample_id)
        else:
            sample_ids = self.run_parallel(num, num_processes)
        print(f"Computed metrics successfully for {len(sample_ids)} out of {num}.")

    def aggregate_metrics(self):
        # Load and aggregate saved metrics from disk

        # Collect sample IDs for data_*.npz and metrics_*.npz
        data_ids = {
            f.stem.replace("data_", "") for f in self.metrics_dir.glob("data_*.npz")
        }
        metrics_ids = {
            f.stem.replace("metrics_", "")
            for f in self.metrics_dir.glob("metrics_*.npz")
        }
        sample_ids = sorted(data_ids & metrics_ids)
        assert len(sample_ids) > 0

        all_metrics, all_data = [], []
        for sample_id in tqdm(
            sample_ids, desc=f"Loading saved metrics from disk. 1% of total"
        ):
            data = np.load(
                self.metrics_dir / f"data_{sample_id}.npz", allow_pickle=True
            )
            metrics = np.load(
                self.metrics_dir / f"metrics_{sample_id}.npz", allow_pickle=True
            )
            data = {
                k: v.item() if v.shape == () else torch.tensor(v)
                for k, v in data.items()
            }
            metrics = {
                k: v.item() if v.shape == () else torch.tensor(v)
                for k, v in metrics.items()
            }
            all_metrics.append(metrics)
            all_data.append(data)
        df = pd.DataFrame(all_metrics)

        # Cast per-motif integer fields and reconstruct a consolidated details column
        try:
            motif_keys = list(
                get_motif_set(
                    modality=self.liability_modality,
                    peptide_type=self.liability_peptide_type,
                ).keys()
            )
        except Exception:
            motif_keys = []

        # Build a consolidated details string if components exist
        details_cols = []
        for motif in motif_keys:
            pos_col = f"liability_{motif}_position"
            len_col = f"liability_{motif}_length"
            sev_col = f"liability_{motif}_severity"
            det_col = f"liability_{motif}_details"
            cnt_col = f"liability_{motif}_count"
            numpos_col = f"liability_{motif}_num_positions"
            # Standardize dtypes: fill NA for ints, then cast
            for col in [pos_col, len_col, sev_col, cnt_col, numpos_col]:
                if col in df.columns:
                    if df[col].dtype.kind in ("f", "O"):
                        df[col] = df[col].fillna(-1).astype(int)
            # Keep details as string
            if det_col in df.columns:
                df[det_col] = df[det_col].fillna("").astype(str)
            # Track columns for a consolidated details view
            if all(
                c in df.columns for c in [pos_col, len_col, sev_col, det_col, cnt_col]
            ):
                details_cols.append(
                    (motif, pos_col, len_col, sev_col, det_col, cnt_col)
                )

        # Optional: single consolidated details column combining motifs
        if details_cols:

            def _compose_details(row):
                items = []
                for motif, pos_col, len_col, sev_col, det_col, cnt_col in details_cols:
                    cnt = row[cnt_col]
                    if cnt and cnt > 0:
                        pos = row[pos_col]
                        length = row[len_col]
                        sev = row[sev_col]
                        det = row[det_col]
                        base = f"{motif}x{cnt}"
                        if pos >= 0:
                            base += f"(pos{pos},len{length},sev{sev})"
                        else:
                            base += f"(sev{sev})"
                        if det:
                            base += f"[{det}]"
                        items.append(base)
                return "; ".join(items)

            df["liability_details"] = df.apply(_compose_details, axis=1)

        # Run clustering
        if self.run_clustering:
            df = self.run_foldseek_clustering(df)
        # Write individual metrics to disk
        csv_path = Path(self.design_dir) / f"aggregate_metrics_{self.name}.csv"
        df.to_csv(csv_path, float_format="%.5f", index=False)

        # Store ca coords and seq in a pickle file for later usage in e.g. diversity aware filtering
        data_rows = []
        for data in all_data:
            data_rows.append(
                {
                    "id": data["sample_id"],
                    "target_id": data["target_id"],
                    "sequence": "".join(
                        [
                            const.prot_token_to_letter[const.tokens[t]]
                            for t in data["design_seq"]
                        ]
                    ),
                    "ca_coords": json.dumps(data["ca_coords"].numpy().tolist()),
                }
            )
        ca_seq_df = pd.DataFrame(data_rows)
        ca_seq_df.to_pickle(
            Path(self.design_dir) / "ca_coords_sequences.pkl.gz", compression="gzip"
        )

        # Compute per target metrics
        df["target_id"] = df["id"].apply(lambda s: TARGET_ID_RE.match(s).group(1))
        per_target_df = df.groupby("target_id").mean(numeric_only=True).reset_index()
        csv_path = Path(self.design_dir) / f"per_target_metrics_{self.name}.csv"
        per_target_df.to_csv(csv_path, float_format="%.5f", index=False)

        avg_metrics = df.mean(numeric_only=True).round(5).to_dict()
        avg_metrics["num_targets"] = len(all_metrics)
        if self.run_clustering:
            if "cluster_07_seqidentity" in df.columns:
                avg_metrics["num_cluster_07_seqidentity"] = len(
                    np.unique(df["cluster_07_seqidentity"].to_numpy())
                )
            if "num_clusters_05_tmscore" in df.columns:
                avg_metrics["num_clusters_05_tmscore"] = len(
                    np.unique(df["clusters_05_tmscore"].to_numpy())
                )

        diversity_metrics, diversity_data = self.compute_diversity(
            all_data, all_metrics
        )
        for k in diversity_metrics:
            avg_metrics[k] = diversity_metrics[k]

        novelty_metrics, novelty_data = self.compute_novelty()
        for k in novelty_metrics:
            avg_metrics[k] = novelty_metrics[k]

        _, histograms = self.make_histograms(all_metrics)

        # Log to Wandb
        if self.wandb is not None and not self.debug:
            import wandb

            print("\nOverall average metrics:", avg_metrics)

            # Make residue distribution plot
            native_stats = np.load("data/native_statistics.npz")
            design_freqs = np.array(
                [
                    avg_metrics[f"{k}_fraction"]
                    for k in const.fake_atom_placements.keys()
                ]
            )
            x = np.arange(len(const.fake_atom_placements.keys()))
            width = 0.15
            fig_res, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width / 2, design_freqs, width, label="Design frequency")
            ax.bar(
                x + width / 2, native_stats["res_dist"], width, label="Data frequency"
            )
            ax.set_xlabel("Res Type")
            ax.set_ylabel("Probability")
            ax.set_title("Res Type distributions")
            ax.set_xticks(x)
            ax.set_xticklabels(const.fake_atom_placements.keys())
            ax.legend()
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            fig_res.savefig(Path(self.design_dir) / "res_type_distribution.png")

            # Make secondary structure distribution plot
            ss_dist = np.array(
                [avg_metrics["loop"], avg_metrics["helix"], avg_metrics["sheet"]]
            )
            x = np.arange(3)
            width = 0.15
            fig_ss, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width / 2, ss_dist, width, label="Designed")
            ax.bar(x + width / 2, native_stats["ss_dist"], width, label="Native data")
            ax.set_xlabel("Secondary Structure type")
            ax.set_ylabel("Frequency")
            ax.set_title("Secondary Structure distributions")
            ax.set_xticks(x)
            ax.set_xticklabels(["loop", "helix", "sheet"])
            ax.legend()
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            fig_ss.savefig(
                Path(self.design_dir) / "secondary_structure_distribution.png"
            )

            wandb.init(name=self.name, **self.wandb)
            wandb.log(avg_metrics)
            wandb.log({"res_dist": wandb.Image(fig_res)})
            wandb.log({"ss_dist": wandb.Image(fig_ss)})

            # Log histograms
            for name, fig in histograms.items():
                wandb.log({f"{name}_hist": wandb.Image(fig)})

            # plot per target vendiscore histogram
            if self.diversity_per_target_original:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    diversity_data["vendi_tm_fixed"], bins=50, color="blue", alpha=0.7
                )
                ax.set_title("Vendi Score Per Target Distribution")
                ax.set_xlabel("Vendi Score")
                ax.set_ylabel("Count")
                plt.tight_layout()
                wandb.log({"vendi_per_target": wandb.Image(fig)})
                plt.close(fig)

            if self.novelty_per_target_original:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    novelty_data["nov_df"]["novelty"],
                    bins=50,
                    color="blue",
                    alpha=0.7,
                )
                ax.set_title("Novelty Per Target Original Distribution")
                ax.set_xlabel("Novelty")
                ax.set_ylabel("Count")
                plt.tight_layout()
                wandb.log({"novelty_per_target_original_hist": wandb.Image(fig)})
                plt.close(fig)

            if self.novelty_per_target_refolded and (self.fold_metrics):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    novelty_data["nov_df_refold"]["novelty"],
                    bins=50,
                    color="blue",
                    alpha=0.7,
                )
                ax.set_title("Novelty Per Target Refolded Distribution")
                ax.set_xlabel("Novelty")
                ax.set_ylabel("Count")
                plt.tight_layout()
                wandb.log({"novelty_per_target_refolded_hist": wandb.Image(fig)})
                plt.close(fig)

    def compute_metrics(self, idx=None, sample_id=None, suffix=None, design_dir=None):
        if sample_id is None:
            sample_id = self.data.predict_set.generated_paths[idx].stem
            feat = self.data.predict_set[idx]
        else:
            if design_dir is None:
                design_dir = self.design_dir

            feat = self.data.predict_set.get_sample(
                design_dir=design_dir, sample_id=sample_id
            )
        path = feat["path"]
        if feat["exception"]:
            msg = f"Failed obtaining valid features for {path} due to feat['exception']. Skipping."
            print(msg)
            return None

        # Get designed sequence
        res_type_argmax = torch.argmax(feat["res_type"], dim=-1)
        design_seq_tensor = res_type_argmax[
            feat["design_mask"].bool() & feat["token_pad_mask"].bool()
        ]
        design_chain_id = feat["asym_id"][
            torch.where(feat["design_mask"].bool() & feat["token_pad_mask"].bool())[0][
                0
            ]
        ].item()
        design_chain_seq = res_type_argmax[design_chain_id == feat["asym_id"]]
        design_seq = "".join(
            [
                const.prot_token_to_letter.get(const.tokens[t], "X")
                for t in design_seq_tensor
            ]
        )
        design_chain_seq = "".join(
            [
                const.prot_token_to_letter.get(const.tokens[t], "X")
                for t in design_chain_seq
            ]
        )

        # initialize metrics
        metrics = {
            "id": sample_id,
            "file_name": path.name,
            "designed_sequence": design_seq,
            "designed_chain_sequence": design_chain_seq,
        }

        # Add per-chain sequences to csv when designing multiple chains
        design_token_indices = torch.where(feat["design_mask"].bool() & feat["token_pad_mask"].bool())[0]
        designed_chain_ids = feat["asym_id"][design_token_indices].unique().tolist()
        if len(designed_chain_ids) > 1:
            for chain_id in designed_chain_ids:
                chain_mask = feat["asym_id"] == chain_id

                # Full chain sequence
                chain_res_types = res_type_argmax[chain_mask]
                full_chain_seq = "".join(
                    [
                        const.prot_token_to_letter.get(const.tokens[t], "X")
                        for t in chain_res_types
                    ]
                )

                # Designed residues only from this chain
                design_chain_mask = feat["design_mask"].bool() & feat["token_pad_mask"].bool() & chain_mask
                design_res_types = res_type_argmax[design_chain_mask]
                design_seq = "".join(
                    [
                        const.prot_token_to_letter.get(const.tokens[t], "X")
                        for t in design_res_types
                    ]
                )

                metrics[f"designed_sequence_{chain_id}"] = design_seq
                metrics[f"full_sequence_{chain_id}"] = full_chain_seq


        target_id = re.search(rf"{self.data.cfg.target_id_regex}", sample_id).group(1)

        # Get masks
        design_mask = feat["design_mask"].bool()
        chain_design_mask   = feat["chain_design_mask"].bool() 

        design_resolved_mask = design_mask & feat["token_resolved_mask"].bool()

        target_resolved_mask = (~chain_design_mask) & feat["token_resolved_mask"].bool()
        atom_design_resolved_mask = (
            (feat["atom_to_token"].float() @ design_resolved_mask.unsqueeze(-1).float())
            .bool()
            .squeeze()
        )
        atom_target_resolved_mask = (
            (feat["atom_to_token"].float() @ target_resolved_mask.unsqueeze(-1).float())
            .bool()
            .squeeze()
        )
        atom_resolved_mask = feat["atom_resolved_mask"]
        resolved_atoms_design_mask = atom_design_resolved_mask[atom_resolved_mask]
        resolved_atoms_target_mask = atom_target_resolved_mask[atom_resolved_mask]
        atom_chain_mask = (
            (
                feat["atom_to_token"].float()
                @ chain_design_mask.unsqueeze(-1).float()
            )
            .bool()
            .squeeze()
        )

        # Get masks for native structure
        if self.native:
            native_design_mask = feat["native_design_mask"].bool()
            native_target_resolved_mask = (
                ~native_design_mask & feat["native_token_resolved_mask"].bool()
            )
            native_atom_target_resolved_mask = (
                (
                    feat["native_atom_to_token"].float()
                    @ native_target_resolved_mask.unsqueeze(-1).float()
                )
                .bool()
                .squeeze()
            )

        # add to design_only directory for novelty computation
        des_cif_path = None
        if not suffix is None:
            des_pdb_dir = self.des_pdb_dir / suffix
            des_pdb_dir.mkdir(exist_ok=True, parents=True)
        else:
            des_pdb_dir = self.des_pdb_dir
        des_pdb_path = des_pdb_dir / f"{feat['id']}_des.pdb"
        des_cif_path = des_pdb_path.with_suffix(".cif")
        if (
            self.novelty_original
            or self.novelty_refolded
            or self.novelty_per_target_original
            or self.novelty_per_target_refolded
            or self.run_clustering
        ):
            try:
                save_design_only_structure_to_pdb(
                    atom_design_mask=atom_chain_mask,
                    structure=feat["str_gen"],
                    output_path=des_pdb_path,
                )
                des_cif_path = des_pdb_path.with_suffix(".cif")
            except Exception as e:
                print(
                    f"[Warning] Could not save design-only structure for {feat['id']}: {e}. Skipping this file."
                )
                traceback.print_exc()
                return None

        # largest hydrophobic patch area original
        if self.largest_hydrophobic:
            if not des_cif_path.exists():
                save_design_only_structure_to_cif(
                    atom_design_mask=atom_chain_mask,
                    structure=feat["str_gen"],
                    output_path=des_cif_path,
                )
            area = largest_hydrophobic_patch_area(des_cif_path)
            metrics["design_largest_hydrophobic_patch"] = area
        if des_cif_path is not None:
            des_cif_path.unlink(missing_ok=True)

        # Count logging
        metrics["num_prot_tokens"] = (
            (feat["mol_type"] == const.chain_type_ids["PROTEIN"]).sum().item()
        )
        metrics["num_lig_atoms"] = (
            (feat["mol_type"] == const.chain_type_ids["NONPOLYMER"]).sum().item()
        )
        metrics["num_resolved_tokens"] = feat["token_resolved_mask"].sum().item()
        metrics["num_tokens"] = feat["token_pad_mask"].sum().item()
        metrics["num_design"] = feat["design_mask"].sum().item()

        # delta sasa for original
        if self.delta_sasa_original:
            (
                delta_sasa_orig,
                design_sasa_unbound,
                design_sasa_bound,
            ) = get_delta_sasa(
                path,
                atom_target_mask=resolved_atoms_target_mask,
                atom_design_mask=resolved_atoms_design_mask,
            )
            metrics["delta_sasa_original"] = delta_sasa_orig
            metrics["design_sasa_unbound_original"] = design_sasa_unbound
            metrics["design_sasa_bound_original"] = design_sasa_bound

        # Noncovalents metrics for original
        try:
            if self.noncovalents_original:
                metrics.update(count_noncovalents(feat))
        except Exception as e:
            print(
                f"[Error] computing noncovalents for {path}: {e}. Skipping this file."
            )
            traceback.print_exc()
            return None

        # Sequence metrics
        if self.sequence_recovery:
            native_seq = torch.argmax(feat["native_res_type"], dim=-1)[
                native_design_mask
            ]
            metrics["seq_recovery"] = (
                (design_seq_tensor == native_seq).float().mean().item()
            )
        for t in const.fake_atom_placements.keys():
            metrics[f"{t}_fraction"] = (
                (design_seq_tensor == const.token_ids[t]).float().mean().item()
            )

        # Secondary structure metrics
        # Compute secondary structure distribution. First get backbone then use pydssp to compute.
        bb_design_mask = (
            feat["atom_pad_mask"].bool()
            & atom_design_resolved_mask
            & feat["backbone_mask"].bool()
        )
        bb_coords = feat["coords"][0][bb_design_mask]
        num_atoms = bb_coords.shape[0]
        if num_atoms % 4 != 0:
            msg = f"BB atoms {num_atoms} is not divisible by 4 for {path}"
            print(msg)
            traceback.print_exc()
            return None
        bb = bb_coords.reshape(-1, 4, 3)
        ca_coords = bb[:, 1, :]
        if len(bb) > 5:
            try:
                dssp = (
                    torch.zeros(bb.shape[0], dtype=torch.long)
                    if torch.sum(bb_design_mask).item() == 0
                    else pydssp.assign(bb, out_type="index")
                )
                # Secondary structure conditioning metric
                if self.ss_conditioning_metrics:
                    ss_conditioning_metricsed = feat["ss_type"][design_mask]
                    dssp_adjusted = dssp + 1
                    ss_metrics = compute_ss_metrics(
                        dssp_adjusted, ss_conditioning_metricsed
                    )
                    metrics.update(ss_metrics)
                metrics["loop"] = (dssp == 0).float().mean().item()
                metrics["helix"] = (dssp == 1).float().mean().item()
                metrics["sheet"] = (dssp == 2).float().mean().item()
            except:
                traceback.print_exc()
                print(f"DSSP failed for {path}.")
                return None
        else:
            metrics["loop"] = float("nan")
            metrics["helix"] = float("nan")
            metrics["sheet"] = float("nan")

        # Liability analysis
        if self.liability_analysis:
            try:
                liability_metrics = compute_liability_metrics(
                    design_chain_seq,
                    self.liability_modality,
                    self.liability_peptide_type,
                )
                metrics.update(liability_metrics)
            except Exception as e:
                traceback.print_exc()
                print(f"Liability analysis failed for {sample_id}: {e}")
                return None

        # Compute RMSD between native (input) and generated conditioning structures.
        # conditioning structure does not have fake atoms, just parse the coordinates and compute rmsd.
        metrics["native_rmsd"] = 0.0
        metrics["native_rmsd_bb"] = 0.0
        if self.native:
            target_coords = feat["coords"][:, atom_target_resolved_mask]
            native_target_coords = feat["native_coords"][
                :, native_atom_target_resolved_mask
            ]
            target_rmsd = compute_rmsd(native_target_coords, target_coords)

            bb_target_coords = feat["coords"][
                :, atom_target_resolved_mask & feat["backbone_mask"].bool()
            ]
            bb_native_target_coords = feat["native_coords"][
                :,
                native_atom_target_resolved_mask & feat["native_backbone_mask"].bool(),
            ]
            bb_target_rmsd = compute_rmsd(bb_native_target_coords, bb_target_coords)

            bb_coords = feat["coords"][:, feat["backbone_mask"].bool()]
            bb_native_coords = feat["native_coords"][
                :, feat["native_backbone_mask"].bool()
            ]
            bb_rmsd = compute_rmsd(bb_native_coords, bb_coords)
            metrics["native_rmsd_all_bb"] = bb_rmsd.item()

            metrics["native_rmsd"] = target_rmsd.item()
            metrics["native_rmsd_bb"] = bb_target_rmsd.item()

        # Check binding site adherence. For each binding site token, find closest design token
        binding_site_mask = feat["binding_type"] == 1
        if binding_site_mask.sum() > 1:
            token_distances = torch.cdist(feat["center_coords"], feat["center_coords"])
            bindsite_design_distances = token_distances[binding_site_mask][
                :, feat["design_mask"]
            ]
            min_bindsite_design_distances = bindsite_design_distances.min(axis=1).values
            for threshold in self.bindsite_adherence_thresholds:
                metrics[f"bindsite_under_{threshold}rmsd"] = (
                    (min_bindsite_design_distances < threshold).float().mean().item()
                )

        # Count free Cysteines
        if self.free_cys:
            cysteine_mask = (
                torch.argmax(feat["res_type"], dim=-1)
                == const.token_ids["CYS"] & feat["design_mask"]
            )
            cysteine_sulfur_indices = (
                torch.argmax(feat["token_to_rep_atom"].int(), dim=-1)[cysteine_mask]
                + const.ref_atoms["CYS"].index("SG")
                - const.ref_atoms["CYS"].index("CA")
            )
            cysteine_coords = feat["coords"][0][cysteine_sulfur_indices]
            free_cysteines = 0
            dist = torch.cdist(cysteine_coords, cysteine_coords)
            for i in range(dist.shape[0]):
                if dist[i, torch.argsort(dist[i])[1]] > 4.0:
                    free_cysteines += 1
            metrics["free_cysteines"] = free_cysteines

        # Quality of Cysteine-Cysteine bonds
        if self.disulfide_quality:
            bonds = feat["structure_bonds"]
            sulfur_mask = (
                torch.argmax(feat["ref_element"], dim=-1)
                == const.element_to_atomic_num["S"]
            )
            dist = []
            disulfide_bonds = []
            for bond in bonds:
                if (
                    bond[6] == const.bond_type_ids["COVALENT"]
                    and sulfur_mask[bond[4]]
                    and sulfur_mask[bond[5]]
                ):
                    disulfide_bonds.append(bond)
            for ds_bond in disulfide_bonds:
                dist.append(
                    torch.cdist(
                        feat["coords"][0][ds_bond[4]].unsqueeze(0),
                        feat["coords"][0][ds_bond[5]].unsqueeze(0),
                    )
                )
            if len(dist) > 0:
                min_dist = torch.cat(dist).min()
                max_dist = torch.cat(dist).max()
                metrics["disulfide_bond_len_qual"] = [max_dist.item(), min_dist.item()]

        # Folding metrics
        ca_coords_refolded = None
        metrics["native_rmsd_refolded"] = 0.0
        metrics["native_rmsd_bb_refolded"] = 0.0
        if self.fold_metrics:
            # Compute refolding metrics when refolding the design only in absence of the target (the whole design chain and anything covalently attached is refolded).
            if self.designfolding_metrics:
                folded_path = (
                    self.design_dir / const.folding_design_dirname / f"{feat['id']}.npz"
                )
                if not folded_path.exists():
                    print(f"Folded path does not exist. Skipping: {folded_path}")
                    return None

                folded = np.load(
                    self.design_dir / const.folding_design_dirname / f"{feat['id']}.npz"
                )
                feat_design = {
                    k: torch.from_numpy(folded[k]).squeeze(0)
                    for k in [
                        "input_coords",
                        "res_type",
                        "token_index",
                        "atom_resolved_mask",
                        "atom_to_token",
                        "mol_type",
                        "backbone_mask",
                    ]
                }
                feat_design["design_mask"] = torch.ones_like(
                    feat_design["token_index"]
                ).bool()
                feat_design["chain_design_mask"] = torch.ones_like(
                    feat_design["token_index"]
                ).bool()
                # Use the same features as refolded, just need to change the coordinates back to the original designed coordinates
                feat_design["coords"] = feat_design["input_coords"]
                if (
                    not len(folded["res_type"].squeeze())
                    == len(feat_design["res_type"])
                    or not (
                        folded["res_type"].squeeze() == feat_design["res_type"]
                    ).all()
                ):
                    msg = f"Skipping {path}. The sequences for which the refolding was run are not the same as the sequences in the design_dir. Maybe the designs in the design_dir were overwritten. Or maybe two processes are operating on the same design_dir."
                    print(msg)
                    return None

                # Compute all-atom folding metrics when refolding only the designed part (this does not make sense when using inverse folded structures)
                if self.allatom_fold_metrics:
                    fold_metrics = get_fold_metrics(
                        feat_design,
                        folded,
                        compute_lddts=self.compute_lddts,
                    )
                    fold_metrics = {
                        f"designfolding-{k}": v for k, v in fold_metrics.items()
                    }
                    metrics.update(fold_metrics)

                # Compute backbone folding metrics when refolding only the designed part
                if self.backbone_fold_metrics:
                    feat_bb = copy.deepcopy(feat_design)
                    feat_bb["atom_resolved_mask"] = feat_bb["atom_resolved_mask"].to(
                        bool
                    ) & feat_bb["backbone_mask"].to(bool)
                    fold_metrics_bb = get_fold_metrics(
                        feat_bb,
                        folded,
                        compute_lddts=self.compute_lddts,
                        prefix="bb_",
                    )
                    fold_metrics_bb = {
                        f"designfolding-{k}": v for k, v in fold_metrics_bb.items()
                    }
                    metrics.update(fold_metrics_bb)

            folded_path = self.design_dir / const.folding_dirname / f"{feat['id']}.npz"
            if not folded_path.exists():
                print(f"Folded path does not exist. Skipping: {folded_path}")
                return None
            folded = np.load(
                self.design_dir / const.folding_dirname / f"{feat['id']}.npz"
            )
            if (
                not len(folded["res_type"].squeeze()) == len(feat["res_type"])
                or not (folded["res_type"].squeeze() == feat["res_type"].numpy()).all()
            ):
                msg = f"Skipping {path}. The sequences for which the refolding was run are not the same as the sequences in the design_dir. Maybe the designs in the design_dir were overwritten. Or maybe two processes are operating on the same design_dir."
                print(msg)
                return None

            # Compute allatom folding metrics when refolding the whole complex (this does not make sense when the design_dir contains inverse folded structures).
            if self.allatom_fold_metrics:
                fold_metrics = get_fold_metrics(
                    feat,
                    folded,
                    compute_lddts=self.compute_lddts,
                )
                metrics.update(fold_metrics)

            # Compute backbone folding metrics when refolding the whole complex
            if self.backbone_fold_metrics:
                feat_bb = copy.deepcopy(feat)
                feat_bb["atom_resolved_mask"] = feat_bb["atom_resolved_mask"].to(
                    bool
                ) & feat_bb["backbone_mask"].to(bool)
                fold_metrics_bb = get_fold_metrics(
                    feat_bb, folded, compute_lddts=self.compute_lddts, prefix="bb_"
                )
                fold_metrics_bb = {f"{k}": v for k, v in fold_metrics_bb.items()}
                metrics.update(fold_metrics_bb)

            # Construct features for refolded complex
            feat_out = {}
            for k in feat.keys():
                if k == "coords":
                    best_sample = get_best_folding_sample(folded)
                    feat_out[k] = torch.from_numpy(best_sample["coords"])
                else:
                    feat_out[k] = feat[k]
            refold_atom_target_resolved_mask = (
                (
                    feat_out["atom_to_token"].float()
                    @ target_resolved_mask.unsqueeze(-1).float()
                )
                .bool()
                .squeeze()
            )
            refold_target_coords = feat_out["coords"][
                refold_atom_target_resolved_mask, :
            ][None, ...]

            # Compute reconstruction RMDS compared to a native binder structure (if a native binder exists).
            if self.native:
                refold_target_rmsd = compute_rmsd(
                    native_target_coords,
                    refold_target_coords,
                )
                bb_refold_target_coords = feat_out["coords"][
                    refold_atom_target_resolved_mask & feat_out["backbone_mask"].bool()
                ][None, ...]
                bb_refold_target_rmsd = compute_rmsd(
                    bb_native_target_coords,
                    bb_refold_target_coords,
                )
                metrics["native_rmsd_refolded"] = refold_target_rmsd.item()
                metrics["native_rmsd_bb_refolded"] = bb_refold_target_rmsd.item()

            # Save the refolded structure of the design to a pdb file if novelty computation needs to be run on it later.
            des_refold_pdb_path = (
                self.des_refold_pdb_dir / f"{feat['id']}_des_refold.pdb"
            )
            des_refold_cif_path = des_refold_pdb_path.with_suffix(".cif")
            if self.novelty_refolded or self.novelty_per_target_refolded:
                structure, _, _ = Structure.from_feat(feat_out)
                try:
                    save_design_only_structure_to_pdb(
                        atom_design_mask=atom_chain_mask,
                        structure=structure,
                        output_path=des_refold_pdb_path,
                    )
                except Exception as e:
                    print(
                        f"[Warning] Could not save design-only structure for {feat['id']}: {e}. Skipping this file."
                    )
                    traceback.print_exc()
                    return None

            # largest hydrophobic patch area refolded
            if self.largest_hydrophobic_refolded:
                if not des_refold_cif_path.exists():
                    structure, _, _ = Structure.from_feat(feat_out)

                    save_design_only_structure_to_cif(
                        atom_design_mask=atom_chain_mask,
                        structure=structure,
                        output_path=des_refold_cif_path,
                    )
                area_refold = largest_hydrophobic_patch_area(des_refold_cif_path)
                metrics["design_largest_hydrophobic_patch_refolded"] = area_refold
            if des_refold_cif_path is not None:
                des_refold_cif_path.unlink(missing_ok=True)

            # Compute sequence based hydrophobicity
            metrics["design_chain_hydrophobicity"] = calc_hydrophobicity(
                design_chain_seq
            )
            metrics["design_hydrophobicity"] = calc_hydrophobicity(design_seq)

            # delta sasa for refolded
            if self.delta_sasa_refolded:
                cif_path_refolded = self.refold_cif_dir / f"{feat['id']}.cif"

                if not cif_path_refolded.exists():
                    msg = f"Refolded cif path does not exist. This can happen if a process was interrupted between writing the refold .npz file and the refold .cif file. Missing path: {cif_path_refolded}"
                    print(msg)
                    return None

                # Compute delta sasa
                (
                    delta_sasa_refolded,
                    design_sasa_unbound,
                    design_sasa_bound,
                ) = get_delta_sasa(
                    cif_path_refolded,
                    atom_target_mask=resolved_atoms_target_mask,
                    atom_design_mask=resolved_atoms_design_mask,
                )

                metrics["delta_sasa_refolded"] = delta_sasa_refolded
                metrics["design_sasa_unbound_refolded"] = design_sasa_unbound
                metrics["design_sasa_bound_refolded"] = design_sasa_bound

            # noncovalents metrics for refolded structure
            if self.noncovalents_refolded:
                try:
                    _metrics = count_noncovalents(feat_out)
                    _metrics = {f"{k}_refolded": v for k, v in _metrics.items()}
                    metrics.update(_metrics)
                except Exception as e:
                    print(
                        f"[Error] computing noncovalents refolded for {path}: {e}. Skipping this file."
                    )
                    traceback.print_exc()
                    return None

            bb_out = feat_out["coords"][bb_design_mask].reshape(-1, 4, 3)
            ca_coords_refolded = bb_out[:, 1, :].cpu()

        # Affinity metrics
        if self.affinity_metrics:
            affinity_path = (
                self.design_dir / const.affinity_dirname / f"{feat['id']}.npz"
            )
            if not affinity_path.exists():
                print(f"Affinity path does not exist. Skipping: {affinity_path}")
                return None

            affinity = np.load(
                self.design_dir / const.affinity_dirname / f"{feat['id']}.npz"
            )

            for key in const.eval_keys_affinity:
                if key in affinity:
                    metrics[key] = affinity[key].item()

            if "affinity_probability_binary1" in metrics:
                metrics["affinity_probability_binary1>50"] = (
                    metrics["affinity_probability_binary1"] > 0.5
                )
                metrics["affinity_probability_binary1>75"] = (
                    metrics["affinity_probability_binary1"] > 0.75
                )

        # Write outputs to files and return sample_id for conformation of successful processing
        data = {
            "target_id": target_id,
            "sample_id": sample_id,
            "design_seq": design_seq_tensor.cpu(),
            "ca_coords": ca_coords.cpu(),
            "ca_coords_refolded": ca_coords_refolded,
        }
        data_path = self.metrics_dir / f"data_{sample_id}.npz"
        metrics_path = self.metrics_dir / f"metrics_{sample_id}.npz"
        np.savez_compressed(metrics_path, **metrics)
        np.savez_compressed(data_path, **data)
        return sample_id

    def compute_diversity(self, all_data, all_metrics):
        avg_metrics = {}
        metrics_data = {}
        fold_metrics = self.fold_metrics

        # Aggregate alpha carbon positions for diversity eval
        ca_gen = defaultdict(list)
        input_metrics = defaultdict(list)
        ca_refold = defaultdict(list)
        sequences = defaultdict(list)
        for i, data in enumerate(all_data):
            ca_gen[data["target_id"]].append(data["ca_coords"])
            ca_refold[data["target_id"]].append(data["ca_coords_refolded"])
            input_metrics[data["target_id"]].append(all_metrics[i])

            seq = data["design_seq"]
            try:
                seq = "".join(
                    [const.prot_token_to_letter[const.tokens[t]] for t in seq]
                )
                sequences[data["target_id"]].append(seq)
            except KeyError as e:
                print(
                    f"[Error] KeyError '{e.args[0]}' for target_id: {data['target_id']}, sample_id: {data['sample_id']}"
                )
        print(
            f"Number of targets: {len(ca_gen)}. Number of designs: {len(all_metrics)}."
        )

        if self.diversity_original:
            print("Computing diveristy original.")
            ca_filtered = [ca[0] for ca in ca_gen.values() if len(ca[0]) >= 3]
            metrics_filtered = [
                m[0]
                for ca, m in zip(ca_gen.values(), input_metrics.values())
                if len(ca[0]) >= 3
            ]
            scores = vendi_scores(
                all_ca_coords=ca_filtered,
                all_metrics=metrics_filtered,
                fold_metrics=fold_metrics,
                diversity_subset=self.diversity_subset,
                compute_lddts=self.compute_lddts,
                backbone_fold_metrics=self.backbone_fold_metrics,
                allatom_fold_metrics=self.allatom_fold_metrics,
            )
            for k, v in scores.items():
                avg_metrics[k + "_original"] = round(float(v), 5)

            # Sequence diversity:
            seqs_filtered = [seq[0] for seq in sequences.values()]
            seq_scores = vendi_sequences(seqs_filtered, self.diversity_subset)
            for k, v in seq_scores.items():
                avg_metrics[k] = round(float(v), 5)

        if self.diversity_per_target_original:
            print("Computing diveristy original per target.")
            vendi_per_target = []
            for target_id, ca_list in ca_gen.items():
                seq_list = sequences[target_id]
                ca_filtered = [e for e in ca_list if len(e) >= 3]
                metrics_filtered = [
                    m
                    for ca, m in zip(ca_list, input_metrics[target_id])
                    if len(ca[0]) >= 3
                ]
                count = len(ca_filtered)
                scores = vendi_scores(
                    ca_filtered,
                    all_metrics=metrics_filtered,
                    fold_metrics=fold_metrics,
                    diversity_subset=self.diversity_subset,
                    compute_lddts=self.compute_lddts,
                    backbone_fold_metrics=self.backbone_fold_metrics,
                    allatom_fold_metrics=self.allatom_fold_metrics,
                )
                seq_scores = vendi_sequences(seq_list, self.diversity_subset)
                scores.update(seq_scores)
                scores.update(
                    {
                        "target_id": target_id,
                        "num_filtered_ca": count,
                    }
                )
                vendi_per_target.append(scores)
            df_vendi = pd.DataFrame(vendi_per_target)
            vendi_csv_path = Path(self.design_dir) / f"vendi_per_target_{self.name}.csv"
            df_vendi.to_csv(vendi_csv_path, index=False, float_format="%.5f")

            for k in vendi_per_target[0].keys():
                if isinstance(vendi_per_target[0][k], numbers.Number):
                    vendis = [e[k] for e in vendi_per_target if not np.isnan(e[k])]
                    avg_metrics[k + "_mean_per_target"] = float(np.mean(vendis))
                    avg_metrics[k + "_median_per_target"] = float(np.median(vendis))

                    metrics_data[k] = vendis

        if self.diversity_refolded and (self.fold_metrics):
            print("Computing diveristy refolded.")
            sample0 = [ca[0] for ca in ca_refold.values() if ca[0].shape[0] >= 3]
            metrics_filtered = [
                m[0]
                for ca, m in zip(ca_refold.values(), input_metrics.values())
                if len(ca[0]) >= 3
            ]
            scores = vendi_scores(
                sample0,
                metrics_filtered,
                fold_metrics,
                self.diversity_subset,
                self.compute_lddts,
                backbone_fold_metrics=self.backbone_fold_metrics,
                allatom_fold_metrics=self.allatom_fold_metrics,
            )

            for k, v in scores.items():
                avg_metrics[k + "_refolded"] = round(float(v), 5)

        return avg_metrics, metrics_data

    def compute_novelty(self, suffix=None):
        """
        Novelty computation using foldseek.
        This function can be optionally run for computing novelty compared to a reference database which needs to be provided
        The compute_metrics function writes pdb files which are then used by this function.
        """
        avg_metrics = {}
        metrics_data = {}

        des_pdb_dir = Path(self.des_pdb_dir)
        design_dir = Path(self.design_dir)
        des_refold_pdb_dir = Path(self.des_refold_pdb_dir)
        if not suffix is None:
            des_pdb_dir = des_pdb_dir / suffix
            design_dir = design_dir / suffix
            des_refold_pdb_dir = des_refold_pdb_dir / suffix
            design_dir.mkdir(exist_ok=True, parents=True)

        # novelty original
        if self.novelty_original or self.novelty_per_target_original:
            print("Computing novelty original.")
            novelty_original_df = compute_novelty_foldseek(
                indir=des_pdb_dir,
                outdir=design_dir,
                reference_db=self.foldseek_db,
                files=[str(p) for p in des_pdb_dir.glob("*.pdb")],
                foldseek_binary=self.foldseek_binary,
            )

        if self.novelty_original:
            avg_metrics["novelty_original"] = round(
                float(novelty_original_df["novelty"].mean()), 5
            )

        if self.novelty_per_target_original:
            novelty_original_df["target_id"] = novelty_original_df["query"].apply(
                lambda s: TARGET_ID_RE.match(s).group(1)
            )
            nov_df = (
                novelty_original_df.groupby("target_id")["novelty"].mean().reset_index()
            )
            nov_csv = Path(design_dir) / f"novelty_per_target_original_{self.name}.csv"
            nov_df.to_csv(nov_csv, index=False, float_format="%.5f")
            avg_metrics["mean_novelty_per_target_original"] = (
                nov_df["novelty"].mean().round(5)
            )
            avg_metrics["median_novelty_per_target_original"] = (
                nov_df["novelty"].median().round(5)
            )
            metrics_data["nov_df"] = nov_df

        # Novelty refolded
        if (self.novelty_refolded or self.novelty_per_target_refolded) and (
            self.fold_metrics
        ):
            print("Computing novelty refolded.")
            novelty_refolded_df = compute_novelty_foldseek(
                indir=des_refold_pdb_dir,
                outdir=Path(design_dir),
                reference_db=self.foldseek_db,
                files=[str(p) for p in des_refold_pdb_dir.glob("*.pdb")],
                foldseek_binary=self.foldseek_binary,
            )

        if self.novelty_refolded and (self.fold_metrics):
            avg_metrics["novelty_refolded"] = round(
                float(novelty_refolded_df["novelty"].mean()), 5
            )

        if self.novelty_per_target_refolded and (self.fold_metrics):
            novelty_refolded_df["target_id"] = novelty_refolded_df["query"].apply(
                lambda s: TARGET_ID_RE.match(s).group(1)
            )
            nov_df_refold = (
                novelty_refolded_df.groupby("target_id")["novelty"].mean().reset_index()
            )
            nov_csv = Path(design_dir) / f"novelty_per_target_refolded_{self.name}.csv"
            nov_df_refold.to_csv(nov_csv, index=False, float_format="%.5f")
            avg_metrics["mean_novelty_per_target_refolded"] = (
                nov_df_refold["novelty"].mean().round(5)
            )
            avg_metrics["median_novelty_per_target_refolded"] = round(
                nov_df_refold["novelty"].median(), 5
            )
            metrics_data["nov_df_refold"] = nov_df_refold
        return avg_metrics, metrics_data

    def run_foldseek_clustering(self, df: pd.DataFrame, suffix=None) -> pd.DataFrame:
        """
        Annotates each design with a cluster based on foldseek clustering.
        This function can be optionally run.
        The compute_metrics function writes pdb files which are then used by this function.
        """
        des_pdb_dir = self.des_pdb_dir / suffix if suffix else self.des_pdb_dir
        design_dir = Path(self.design_dir) / suffix if suffix else Path(self.design_dir)

        cluster_output_dir = design_dir / "foldseek_cluster"
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        cluster_prefix = cluster_output_dir / "cluster"
        tmp_dir = cluster_output_dir / "tmp"

        min_num_design = int(df["num_design"].min())
        cmd = [
            self.foldseek_binary,
            "easy-cluster",
            str(des_pdb_dir),
            str(cluster_prefix),
            str(tmp_dir),
            "--alignment-type",
            "1",
            "--cov-mode",
            "0",
            "--min-seq-id",
            "0",
            "--tmscore-threshold",
            "0.5",
        ]
        if min_num_design < 20:
            msg = f"[FoldSeek] Using --kmer-per-seq {2} due to short designs."
            print(msg)
            cmd += ["--kmer-per-seq", str(2)]

        try:
            subprocess.run(cmd, check=True)

            df_cluster = pd.read_csv(
                str(cluster_prefix) + "_cluster.tsv",
                sep="\t",
                header=None,
                names=["file", "clusters_05_tmscore"],
            )
            df_cluster["file"] = df_cluster["file"].apply(lambda x: Path(x).stem)

            df = df.merge(df_cluster, left_on="id", right_on="file", how="left").drop(
                columns=["file"]
            )
        except Exception as e:
            msg = f"Structure clustering was unsuccessful. No cluster labels are added to the dataframe / csv file output."
            print(msg)
        return df

    def make_histograms(self, all_metrics):
        df = pd.DataFrame(all_metrics)

        # Make aggregate histograms
        histograms = {}
        cols = [
            "delta_sasa_refolded",
            "rmsd",
            "iptm",
            "ptm",
            "rmsd",
            "design_ptm",
            "min_design_to_target_pae",
            "helix",
            "sheet",
            "loop",
            "plip_saltbridge",
            "plip_hbonds",
            "design_sasa_bound_original",
            "design_sasa_unbound_original",
            "delta_sasa_original",
            "num_design",
            "precision_loop",
            "recall_loop",
            "precision_helix",
            "recall_helix",
            "precision_sheet",
            "recall_sheet",
            "accuracy_overall",
            "liability_score",
            "liability_num_violations",
        ]
        for col in cols:
            if col in df.columns:
                histograms["hist" + col] = make_histogram(df, col)

        # make per target histograms
        df["target_id"] = df["id"].apply(lambda s: TARGET_ID_RE.match(s).group(1))
        per_target_df = df.groupby("target_id").mean(numeric_only=True).reset_index()
        cols += ["rmsd<2.5", "designability_rmsd_2"]
        if self.compute_lddts:
            cols += [
                "designability_lddt_90",
                "designability_lddt_85",
                "designability_lddt_80",
                "designability_lddt_75",
                "designability_lddt_70",
                "designability_lddt_65",
                "designability_lddt_60",
            ]
        for col in cols:
            if col in df.columns:
                # Per target histograms
                histograms["per_target" + col] = make_histogram(per_target_df, col)
        return df, histograms
