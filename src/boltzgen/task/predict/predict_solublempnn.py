"""Run SolubleMPNN with BoltzGen's existing data modules and design writer."""

import logging
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from boltzgen._vendor.ligandmpnn import ProteinMPNN
from boltzgen.data import const
from boltzgen.data.data import convert_atom_name
from boltzgen.model.modules.inverse_fold import build_constraint_logit_mask
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.task.predict.writer import DesignWriter
from boltzgen.task.task import Task
from boltzgen.utils.pipeline_progress_bar import PipelineProgressBar
from boltzgen.utils.quiet import quiet_startup

logger = logging.getLogger(__name__)
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class SolubleMPNN(LightningModule):
    """Sample protein sequences while retaining BoltzGen's structure metadata."""

    def __init__(
        self,
        checkpoint: str,
        sampling_temperature: float = 0.1,
        inverse_fold_restriction: list[str] | None = None,
        tie_symmetric_sequences: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(sampling_temperature) or sampling_temperature <= 0:
            raise ValueError(
                "SolubleMPNN sampling_temperature must be finite and positive"
            )
        self.sampling_temperature = sampling_temperature
        restrictions = set(inverse_fold_restriction or [])
        self.inverse_fold_restriction = list(restrictions)
        self.tie_symmetric_sequences = tie_symmetric_sequences
        if restrictions - set(const.canonical_tokens):
            raise ValueError(
                "inverse_fold_restriction must contain canonical residue names"
            )
        if len(restrictions) == len(const.canonical_tokens):
            raise ValueError("Cannot exclude every amino acid from inverse folding")

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=int(checkpoint_data["num_edges"]),
            model_type="soluble_mpnn",
        )
        self.model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.masker = BoltzMasker(mask=True, mask_backbone=False)

        boltz_to_mpnn = torch.full((const.num_tokens,), 20, dtype=torch.long)
        mpnn_to_boltz = torch.full((21,), const.token_ids["UNK"], dtype=torch.long)
        bias = torch.zeros(21)
        bias[-1] = -torch.inf  # Never sample the unknown residue.
        for residue in const.canonical_tokens:
            boltz_id = const.token_ids[residue]
            mpnn_id = MPNN_ALPHABET.index(const.prot_token_to_letter[residue])
            boltz_to_mpnn[boltz_id] = mpnn_id
            mpnn_to_boltz[mpnn_id] = boltz_id
            if residue in restrictions:
                bias[mpnn_id] = -torch.inf
        self.register_buffer("boltz_to_mpnn", boltz_to_mpnn, persistent=False)
        self.register_buffer("mpnn_to_boltz", mpnn_to_boltz, persistent=False)
        self.register_buffer("residue_bias", bias, persistent=False)

    def _sampling_constraints(
        self, batch: dict, keep: torch.Tensor, design: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """Translate position constraints and homomer ties to the MPNN alphabet."""
        length = int(keep.sum())
        constraints = batch.get("aa_constraint_mask")
        blocked = (
            build_constraint_logit_mask(
                num_nodes=length,
                aa_constraint_mask=None
                if constraints is None
                else constraints[0, keep],
                inverse_fold_restriction=self.inverse_fold_restriction,
                canonical_tokens=const.canonical_tokens,
                inf=1.0,
                device=keep.device,
            )
            < 0
        )
        bias = self.residue_bias[None, None].expand(1, length, -1).clone()
        canonical_to_mpnn = [
            MPNN_ALPHABET.index(const.prot_token_to_letter[aa])
            for aa in const.canonical_tokens
        ]
        bias[0, :, canonical_to_mpnn] = torch.zeros_like(
            blocked, dtype=bias.dtype
        ).masked_fill(blocked, -torch.inf)

        groups = defaultdict(list)
        if self.tie_symmetric_sequences and "symmetric_group" in batch:
            group_ids = batch["symmetric_group"][0, keep].tolist()
            residue_ids = batch["feature_residue_index"][0, keep].tolist()
            for position, (group, residue, redesign) in enumerate(
                zip(group_ids, residue_ids, design[0, keep].tolist())
            ):
                if group > 0 and redesign:
                    groups[group, residue].append(position)
        tied = [positions for positions in groups.values() if len(positions) > 1]
        for positions in tied:
            # Upstream applies the last member's bias to the tied group. Give
            # every member the intersection of allowed identities explicitly.
            blocked_group = torch.isneginf(bias[0, positions]).any(0)
            if blocked_group[:20].all():
                raise ValueError(
                    "Symmetric inverse-folding positions have no common allowed "
                    "amino acid; use compatible residue constraints for tied chains"
                )
            bias[0, positions] = torch.zeros_like(bias[0, positions]).masked_fill(
                blocked_group, -torch.inf
            )
        return bias, tied

    @torch.no_grad()
    def predict_step(
        self, batch: dict, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> dict:
        """Redesign only requested residues, preserving fixed and nonprotein tokens."""
        try:
            return self._predict(batch)
        except RuntimeError as error:
            if "out of memory" not in str(error):
                raise
            logger.warning("Ran out of memory, skipping inverse-folding batch")
            torch.cuda.empty_cache()
            return {"exception": True}

    def _predict(self, batch: dict) -> dict:
        for flag in ("exception", "skip"):
            if flag in batch and torch.as_tensor(batch[flag]).any():
                # DesignWriter on this branch expects an exception key even for skips.
                return {"exception": True}

        coords = batch["coords"]
        if coords.ndim == 4:
            assert coords.shape[1] == 1, "Expected one input conformer"
            coords = coords[:, 0]
        assert coords.shape[0] == 1, "SolubleMPNN requires batch_size=1"
        original = batch.get("res_type_clone", batch["res_type"])
        sequence = self.boltz_to_mpnn[original.argmax(-1)]
        design = batch.get("inverse_fold_design_mask", batch["design_mask"]).bool()
        design = design & batch["token_pad_mask"].bool()
        # Generated structures can contain UNK with a complete backbone. The
        # generic frame map collapses UNK to one atom, so identify atoms by name.
        names = batch["ref_atom_name_chars"].argmax(-1)
        backbone_names = names.new_tensor(
            [convert_atom_name(name) for name in ("N", "CA", "C", "O")]
        )
        matches = (names[:, :, None, :] == backbone_names[None, None]).all(-1)
        bb_map = (
            batch["atom_to_token"].transpose(1, 2)[:, :, None, :].bool()
            & matches.transpose(1, 2)[:, None]
            & batch["atom_pad_mask"][:, None, None, :].bool()
        ).to(coords.dtype)
        backbone = torch.einsum("blam,bmd->blad", bb_map, coords)
        resolved = (
            torch.einsum(
                "blam,bm->bla", bb_map, batch["atom_resolved_mask"].to(coords.dtype)
            )
            .bool()
            .all(-1)
        )
        # Missing/atomized backbones can map several slots to the same atom.
        complete = (bb_map.sum(-1) == 1).all(-1) & (bb_map.sum(-2) <= 1).all(-1)
        protein = batch["mol_type"] == const.chain_type_ids["PROTEIN"]
        valid = (
            protein
            & batch["token_pad_mask"].bool()
            & ((sequence != 20) | design)
            & resolved
            & complete
            & torch.isfinite(backbone).all(dim=(-1, -2))
        )
        if (design & ~batch["design_mask"].bool()).any():
            raise ValueError(
                "SolubleMPNN inverse_fold_design_mask must be within design_mask "
                "so the design writer can update residue identities and side chains"
            )
        if (design & ~valid).any():
            logger.warning(
                "Skipping %s: SolubleMPNN requires a complete, resolved N/CA/C/O "
                "backbone at every redesigned protein position",
                batch.get("id"),
            )
            return {"exception": True}

        res_type = original.clone()
        if design.any():
            # Remove nonprotein and missing tokens entirely: they are not MPNN context.
            keep = valid[0]
            length = int(keep.sum())
            bias, tied = self._sampling_constraints(batch, keep, design)
            features = {
                "X": backbone[:, keep],
                "S": sequence[:, keep],
                "mask": torch.ones(1, length, device=coords.device),
                "chain_mask": design[:, keep].float(),
                "chain_labels": batch["asym_id"][:, keep].long(),
                "R_idx": batch["residue_index"][:, keep].long(),
                "batch_size": 1,
                "temperature": self.sampling_temperature,
                "bias": bias,
                "randn": torch.randn(1, length, device=coords.device),
                "symmetry_residues": tied or [[]],
                "symmetry_weights": [
                    [1.0 / len(positions)] * len(positions) for positions in tied
                ]
                or [[]],
            }
            sampled = self.model.sample(features)["S"]
            sampled_tokens = self.mpnn_to_boltz[sampled]
            proposed = F.one_hot(sampled_tokens, num_classes=const.num_tokens).to(
                res_type
            )
            res_type[:, keep] = torch.where(
                design[:, keep, None], proposed, original[:, keep]
            )

        prediction = self.masker(batch)
        prediction.update(
            exception=False,
            masks=batch["atom_pad_mask"],
            token_masks=batch["token_pad_mask"],
            input_coords=batch["coords"],
            coords=coords,
            res_type=res_type,
        )
        return prediction


class PredictSolubleMPNN(Task):
    """Execute SolubleMPNN through the same Lightning pipeline as BoltzIF."""

    def __init__(
        self,
        data: LightningDataModule,
        writer: DesignWriter,
        checkpoint: str,
        output: str,
        name: str,
        sampling_temperature: float = 0.1,
        inverse_fold_restriction: list[str] | None = None,
        tie_symmetric_sequences: bool = True,
        trainer: dict | None = None,
        matmul_precision: str | None = None,
    ) -> None:
        self.data = data
        self.writer = writer
        self.checkpoint = checkpoint
        self.output = output
        self.name = name
        self.sampling_temperature = sampling_temperature
        self.inverse_fold_restriction = inverse_fold_restriction
        self.tie_symmetric_sequences = tie_symmetric_sequences
        self.trainer = dict(trainer or {})
        self.matmul_precision = matmul_precision

    def run(self, config: OmegaConf | None = None) -> None:
        """Load the weights and write sequences using the configured data module."""
        quiet_startup()
        if len(self.data.predict_set) == 0:
            logger.info("No predictions required")
            return
        if self.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.matmul_precision)
        model = SolubleMPNN(
            self.checkpoint,
            self.sampling_temperature,
            self.inverse_fold_restriction,
            self.tie_symmetric_sequences,
        ).eval()
        callbacks = [self.writer]
        if os.environ.get("BOLTZGEN_PIPELINE_STEP"):
            callbacks.append(PipelineProgressBar())
        trainer_args = dict(self.trainer)
        devices = trainer_args.get("devices", 1)
        num_samples = len(self.data.predict_set)
        if isinstance(devices, (list, tuple, ListConfig)):
            trainer_args["devices"] = list(devices)[:num_samples]
        elif isinstance(devices, int) and devices > num_samples:
            trainer_args["devices"] = num_samples
        trainer = Trainer(
            default_root_dir=self.output, callbacks=callbacks, **trainer_args
        )
        trainer.predict(model, datamodule=self.data, return_predictions=False)
