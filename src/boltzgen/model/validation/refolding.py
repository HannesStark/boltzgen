from collections import defaultdict
import gc
import itertools
from pathlib import Path
import time
from typing import Dict, List

import wandb

import torch
import numpy as np
import torch._dynamo
from pytorch_lightning import LightningModule



from boltzgen.data import const
from boltzgen.model.validation import design
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.predict.data_from_generated import collate
from boltzgen.task.predict.writer import AffinityWriter, FoldingWriter


class RefoldingValidator(design.DesignValidator):
    """Validation step implementation for Affinity."""

    def __init__(
        self,
        val_names: List[str],
        analyze_task: Analyze,
        folding_model_args: Dict,
        folding_args: Dict,
        folding_checkpoint: str,
        step_scale: float = 2,
        noise_scale: float = 0.75,
        ligand_plip: bool = False,
        atom14: bool = True,
        atom37: bool = False,
        backbone_only: bool = False,
        val_monomer: str = None,
        val_ligand: str = None,
        inverse_fold: bool = False,
        affinity_model_args: Dict = None,
        affinity_args: Dict = None,
        affinity_checkpoint: str = None,
    ) -> None:
        super().__init__(
            val_names=val_names,
            atom14=atom14,
            atom37=atom37,
            backbone_only=backbone_only,
            inverse_fold=inverse_fold,
        )


        self.analyze_task = analyze_task
        self.writer = None
        self.folding_model = None
        self.design_dir = None
        self.folding_model_args = folding_model_args
        self.folding_checkpoint = folding_checkpoint
        self.step_scale = step_scale
        self.noise_scale = noise_scale
        self.ligand_plip = ligand_plip
        self.writer = FoldingWriter(design_dir=None)

        # Affinity model / writer
        self.affinity_model_args = affinity_model_args or {}
        self.affinity_args = affinity_args or {}
        self.affinity_checkpoint = affinity_checkpoint
        self.affinity_model = None
        self.aff_writer = AffinityWriter(design_dir=None)

        self.folding_args = folding_args
        self.folding_args["keys_dict_out"] = const.eval_keys_confidence

        # This is the hardcoded order in the training dataset (FoldEverythingDataModule)
        idx = 1
        dataset_to_logname = {}
        if val_monomer:
            dataset_to_logname[idx] = "val_monomer"
            idx += 1
        if val_ligand:
            dataset_to_logname[idx] = "val_ligand"

        self.dataset_to_logname = dataset_to_logname
        self.all_refolding_data = defaultdict(list)
        self.all_refold_metrics = defaultdict(list)

    def run_model(
        self, model: LightningModule, batch: Dict[str, torch.Tensor], idx_dataset: int
    ) -> Dict[str, torch.Tensor]:
        """Compute the forward pass."""
        out = model(
            batch,
            recycling_steps=model.validation_args.recycling_steps,
            num_sampling_steps=model.validation_args.sampling_steps,
            diffusion_samples=model.validation_args.diffusion_samples,
            step_scale=self.step_scale,
            noise_scale=self.noise_scale,
        )

        return out

    def init_folding_model(self, model, logname):
        if self.folding_model is None:
            self.timestamp = time.time()
            print("Loading folding model for refolding validation.")
            model_module: LightningModule = type(model).load_from_checkpoint(
                checkpoint_path=self.folding_checkpoint,
                strict=False,
                map_location=model.device,
                predict_args=self.folding_args,
                **self.folding_model_args,
            )
            model_module.eval()
            self.folding_model = model_module
            model.log(
                f"{logname}/init_folding_model_dur",
                time.time() - self.timestamp,
                sync_dist=True,
            )

    def init_affinity_model(self, model, logname):
        if (self.affinity_checkpoint is None) or (self.affinity_args is None):
            return

        if self.affinity_model is None:
            self.timestamp = time.time()
            print("Loading affinity model for validation.")
            model_module: LightningModule = type(model).load_from_checkpoint(
                checkpoint_path=self.affinity_checkpoint,
                strict=False,
                predict_args=self.affinity_args,
                map_location=model.device,
                **self.affinity_model_args,
            )
            model_module.eval()
            self.affinity_model = model_module
            model.log(
                f"{logname}/init_affinity_model_dur",
                time.time() - self.timestamp,
                sync_dist=True,
            )

    def set_design_dir(self, design_dir):
        if design_dir != self.design_dir:
            # Set directories for the writer
            self.design_dir = design_dir
            self.writer.init_outdir(design_dir)
            self.aff_writer.init_outdir(design_dir)
            self.analyze_task.init_datasets(design_dir, load_dataset=True)

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
        logname = self.dataset_to_logname[dataloader_idx]
        # Take common step for folding and confidence
        start = time.time()
        if not self.inverse_fold:
            self.common_val_step(
                model, batch, out, idx_dataset=0, symmetry_correction=False
            )
            model.log(
                f"{logname}/common_val_step_dur", time.time() - start, sync_dist=True
            )
        feat_masked = out["feat_masked"]

        design_dir = f"{model.trainer.default_root_dir}/{logname}/epoch{model.current_epoch}_step{model.global_step}"

        start = time.time()
        success = self.design_val_step(
            model,
            batch,
            feat_masked,
            out,
            n_samples,
            batch_idx,
            generated_dir=design_dir,
            invalid_token="ALA",  # Use invalid token ALA here because sometimes all decodings are invalid and then we produce a sequence of only UNK and then the mmcif parser throws the error  ValueError: Alignment mismatch!
            logname=logname,
        )
        if not success:
            return
        model.log(f"{logname}/design_val_step_dur", time.time() - start, sync_dist=True)

        # Initialize folding model
        self.init_folding_model(model, logname)

        # Set design directory
        self.set_design_dir(design_dir)

        # Perform refolding
        start = time.time()
        samples = []
        for n in range(n_samples):
            cif_path = (
                Path(self.design_dir)
                / f"sample{n}_batch{batch_idx}_rank{model.trainer.global_rank}_{batch['id'][n]}.cif"
            )
            samples.append(
                self.analyze_task.data.predict_set.get_feat(
                    cif_path,
                    design_mask=feat_masked["design_mask"][n]
                    .cpu()
                    .numpy(),
                )
            )

        batch_gen = collate(samples)
        self.analyze_task.data.transfer_batch_to_device(
            batch_gen, device=model.device, dataloader_idx=None
        )
        model.log(
            f"{logname}/loading_data_for_folding_dur",
            time.time() - start,
            sync_dist=True,
        )

        start = time.time()
        refolded = self.folding_model.predict_step(batch_gen, batch_idx=batch_idx)

        # Affinity prediction
        self.init_affinity_model(model, logname)
        if self.affinity_model is not None:
            start_aff = time.time()
            aff_pred = self.affinity_model.predict_step(batch_gen, batch_idx=batch_idx)

            for k, v in aff_pred.items():
                if k not in refolded:
                    refolded[k] = v

            self.aff_writer.write_on_batch_end(
                trainer=None,
                pl_module=None,
                prediction=aff_pred,
                batch=batch_gen,
                batch_indices=None,
                batch_idx=batch_idx,
                dataloader_idx=None,
            )
            model.log(
                f"{logname}/affinity_dur", time.time() - start_aff, sync_dist=True
            )

        self.writer.write_on_batch_end(
            trainer=None,
            pl_module=None,
            prediction=refolded,
            batch=batch_gen,
            batch_indices=None,
            batch_idx=batch_idx,
            dataloader_idx=None,
        )

        model.log(f"{logname}/folding_dur", time.time() - start, sync_dist=True)

        basename = f"{design_dir}/sample{n}_batch{batch_idx}_rank{model.trainer.global_rank}_{batch['id'][0]}"
        sample_id = basename.split("/")[-1]

        if logname == "val_ligand" and self.ligand_plip:
            self.analyze_task.noncovalents_original = True

        self.analyze_task.compute_metrics(
            sample_id=sample_id,
            suffix=Path(f"rank{model.trainer.global_rank}"),
            design_dir=Path(design_dir),
        )

        data = np.load(self.analyze_task.metrics_dir / f"data_{sample_id}.npz")
        metrics = np.load(self.analyze_task.metrics_dir / f"metrics_{sample_id}.npz")
        data = {
            k: v.item() if v.shape == () else torch.tensor(v) for k, v in data.items()
        }
        metrics = {
            k: v.item() if v.shape == () else torch.tensor(v)
            for k, v in metrics.items()
        }

        (Path(design_dir) / const.folding_dirname / f"{sample_id}.npz").unlink(
            missing_ok=True
        )

        if metrics is not None:
            self.all_refold_metrics[logname].append(metrics)
            self.all_refolding_data[logname].append(data)

    def on_epoch_end(self, model):
        # Cleanup
        from boltzgen.utils.device import empty_cache, get_device_type

        del self.folding_model
        self.folding_model = None
        del self.affinity_model
        self.affinity_model = None

        # CUDA-specific cleanup
        if get_device_type() == "cuda":
            torch._C._cuda_clearCublasWorkspaces()

        torch._dynamo.reset()
        gc.collect()
        empty_cache()

        # Compute standard metrics
        self.common_on_epoch_end(model, logname="val_monomer_ligand")
        self.on_epoch_end_design(model, logname="val_monomer_ligand")
        
        # Compute the refolding metrics
        for logname in self.dataset_to_logname.values():
            self.on_epoch_end_refolding(model, logname=logname)

        model.log(
            f"val_monomer_ligand/dur", time.time() - self.timestamp, sync_dist=True
        )

    def gather_lists(self, model: LightningModule, object):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            object_list = [None] * model.trainer.world_size
            torch.distributed.all_gather_object(object_list, object)
            return list(itertools.chain(*object_list))
        else:
            return object

    def on_epoch_end_refolding(self, model: LightningModule, logname: str):
        # Run evaluations
        start = time.time()
        design_dir = Path(
            f"{model.trainer.default_root_dir}/{logname}/epoch{model.current_epoch}_step{model.global_step}"
        )
        design_dir.mkdir(exist_ok=True, parents=True)
        design_dir = str(design_dir)
        self.analyze_task.init_datasets(design_dir)

        all_metrics = self.gather_lists(model, self.all_refold_metrics[logname])
        df, histograms = self.analyze_task.make_histograms(all_metrics)
        avg_metrics = df.mean(numeric_only=True).round(5).to_dict()
        avg_metrics["num_targets"] = len(all_metrics)

        print(f"Computing diversity for {logname}.")
        refolding_data = self.gather_lists(model, self.all_refolding_data[logname])
        diversity_metrics, _ = self.analyze_task.compute_diversity(
            refolding_data, all_metrics
        )
        for k in diversity_metrics:
            avg_metrics[k] = diversity_metrics[k]

        print(f"computing novelty {logname}.")
        novelty_metrics, _ = self.analyze_task.compute_novelty(
            suffix=Path(f"rank{model.trainer.global_rank}"),
        )
        for k in novelty_metrics:
            avg_metrics[k] = novelty_metrics[k]

        # Log results
        for name, fig in histograms.items():
            if model.logger is not None:
                model.logger.log_image(f"{logname}/{name}", [wandb.Image(fig)])

        for k, v in avg_metrics.items():
            model.log(f"{logname}/{k}", v, prog_bar=False, sync_dist=True)

        self.all_refold_metrics[logname] = []
        self.all_refolding_data[logname] = []
        model.log(
            f"{logname}/one_epoch_end_refolding_dur",
            time.time() - start,
            sync_dist=True,
        )
