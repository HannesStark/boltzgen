"""
BBB (Blood-Brain Barrier) Permeability Prediction Module.

This module extends BoltzGen with a property prediction head for BBB permeability,
enabling the model to predict whether peptides can cross the blood-brain barrier.
This is a multi-head extension where the shared backbone learns both structural
representations (energy/potential) and pharmacokinetic properties (BBB permeability).
"""

import torch
from torch import nn
from typing import Optional

import boltzgen.model.layers.initialize as init
from boltzgen.model.layers.pairformer import PairformerNoSeqModule
from boltzgen.model.modules.encoders import PairwiseConditioning
from boltzgen.model.modules.utils import LinearNoBias


class BBBModule(nn.Module):
    """
    BBB Permeability Prediction Module.
    
    Predicts BBB permeability for peptides using shared representations from
    the BoltzGen backbone. This module operates similarly to AffinityModule
    but is designed for single-molecule property prediction rather than
    protein-ligand interaction.
    
    Parameters
    ----------
    token_s : int
        Token sequence embedding dimension
    token_z : int
        Token pairwise embedding dimension
    pairformer_args : dict
        Arguments for the pairformer module
    transformer_args : dict
        Arguments for the transformer (token_s, num_blocks, num_heads, etc.)
    num_dist_bins : int, default=64
        Number of distance bins for pairwise embeddings
    max_dist : float, default=22
        Maximum distance for distance binning
    use_sequence_features : bool, default=True
        Whether to use sequence-level features in addition to structure
    task_type : str, default='binary'
        Type of prediction task: 'binary' (permeable/not permeable) or 
        'regression' (continuous permeability score)
    use_tau_conditioning : bool, default=False
        Whether to use Tau conditioning for Tau-specific BBB prediction
    tau_conditioning_args : dict, optional
        Arguments for Tau conditioning module
    """

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        use_sequence_features: bool = True,
        task_type: str = "binary",
        use_tau_conditioning: bool = False,
        tau_conditioning_args: Optional[dict] = None,
    ):
        super().__init__()
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)
        
        self.use_sequence_features = use_sequence_features
        self.task_type = task_type
        self.use_tau_conditioning = use_tau_conditioning
        
        if use_tau_conditioning:
            from boltzgen.model.modules.tau_conditioning import TauConditioning
            tau_args = tau_conditioning_args or {}
            self.tau_conditioning_module = TauConditioning(
                token_s=token_s,
                token_z=token_z,
                **tau_args,
            )
        
        self.bbb_heads = BBBHeadsTransformer(
            token_z,
            transformer_args["token_s"],
            transformer_args["num_blocks"],
            transformer_args["num_heads"],
            transformer_args["activation_checkpointing"],
            use_sequence_features=use_sequence_features,
            task_type=task_type,
        )

    def forward(
        self,
        s_inputs,
        z,
        x_pred,
        feats,
        multiplicity=1,
        use_kernels: bool = False,
    ):
        """
        Forward pass for BBB permeability prediction.
        
        Parameters
        ----------
        s_inputs : Tensor
            Sequence embeddings from the backbone (B, L, token_s)
        z : Tensor
            Pairwise embeddings from the backbone (B, L, L, token_z)
        x_pred : Tensor
            Predicted atom coordinates (B*mult, N, 3) or (B, mult, N, 3)
        feats : dict
            Feature dictionary containing masks and other metadata
        multiplicity : int, default=1
            Number of conformations per sample
        use_kernels : bool, default=False
            Whether to use optimized kernels
            
        Returns
        -------
        dict
            Dictionary containing BBB predictions:
            - 'bbb_pred_value': continuous permeability score (if regression)
            - 'bbb_logits_binary': binary logits (if binary classification)
            - 'bbb_probability': probability of BBB permeability (0-1)
        """
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        # Incorporate sequence information into pairwise embeddings
        if s_inputs is not None:
            z = (
                z
                + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        # Convert atom coordinates to token-level representations
        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        
        # Compute pairwise distances and create distance embeddings
        d = torch.cdist(x_pred_repr, x_pred_repr)
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        # Add distance conditioning
        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        # Create mask for peptide tokens (assuming peptide is the designed molecule)
        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        # For BBB prediction, we focus on the ligand/peptide tokens
        # Assume ligand has mol_type != 0, or use a specific mask if available
        if "bbb_token_mask" in feats:
            peptide_mask = (
                feats["bbb_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
            )
        else:
            # Fallback: assume ligand is non-protein (mol_type != 0)
            peptide_mask = (
                (feats["mol_type"] != 0).repeat_interleave(multiplicity, 0)
                * pad_token_mask
            )
        
        # Create pairwise mask for peptide-peptide interactions
        pair_mask = (
            peptide_mask[:, :, None] * peptide_mask[:, None, :]
        ) * pad_token_mask[:, :, None] * pad_token_mask[:, None, :]

        # Apply pairformer
        z = self.pairformer_stack(
            z,
            pair_mask=pair_mask,
            use_kernels=use_kernels,
        )
        
        # Apply Tau conditioning if enabled
        if self.use_tau_conditioning and "tau_embeddings" in feats:
            tau_embeddings = feats["tau_embeddings"]
            tau_mask = feats.get("tau_mask", None)
            
            # Condition peptide embeddings on Tau
            s_inputs_conditioned = s_inputs if s_inputs is not None else None
            z_conditioned, z_pair_conditioned = self.tau_conditioning_module(
                peptide_seq=s_inputs_conditioned,
                peptide_pair=z,
                tau_embeddings=tau_embeddings,
                tau_mask=tau_mask,
                peptide_mask=pad_token_mask,
            )
            # Use conditioned pairwise embeddings
            z = z_pair_conditioned
            # Update s_inputs if sequence features are used
            if s_inputs_conditioned is not None and self.use_sequence_features:
                s_inputs = s_inputs_conditioned

        # Get BBB predictions
        out_dict = self.bbb_heads(
            z=z,
            s_inputs=s_inputs if self.use_sequence_features else None,
            feats=feats,
            peptide_mask=peptide_mask,
            multiplicity=multiplicity,
        )

        return out_dict


class BBBHeadsTransformer(nn.Module):
    """
    Transformer heads for BBB permeability prediction.
    
    Converts pairwise embeddings into BBB permeability predictions.
    Supports both binary classification and regression tasks.
    """

    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks,
        num_heads,
        activation_checkpointing,
        use_sequence_features: bool = True,
        task_type: str = "binary",
    ):
        super().__init__()
        self.use_sequence_features = use_sequence_features
        self.task_type = task_type

        # MLP to convert pairwise embeddings to sequence-level representation
        self.bbb_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Sequence feature projector (optional)
        if use_sequence_features:
            self.seq_feature_mlp = nn.Sequential(
                nn.Linear(input_token_s, input_token_s),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            combined_dim = input_token_s * 2
        else:
            combined_dim = input_token_s

        # Prediction heads
        if task_type == "regression":
            # Regression head: predicts continuous permeability score
            self.to_bbb_pred_value = nn.Sequential(
                nn.Linear(combined_dim, input_token_s),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_token_s, input_token_s // 2),
                nn.ReLU(),
                nn.Linear(input_token_s // 2, 1),
            )
            # Also provide binary classification as auxiliary output
            self.to_bbb_logits_binary = nn.Linear(1, 1)
        else:
            # Binary classification head
            self.to_bbb_pred_value = None
            self.to_bbb_logits_binary = nn.Sequential(
                nn.Linear(combined_dim, input_token_s),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_token_s, input_token_s // 2),
                nn.ReLU(),
                nn.Linear(input_token_s // 2, 1),
            )

    def forward(
        self,
        z,
        s_inputs=None,
        feats=None,
        peptide_mask=None,
        multiplicity=1,
    ):
        """
        Forward pass for BBB heads.
        
        Parameters
        ----------
        z : Tensor
            Pairwise embeddings (B*mult, L, L, token_z)
        s_inputs : Tensor, optional
            Sequence embeddings (B*mult, L, token_s)
        feats : dict
            Feature dictionary
        peptide_mask : Tensor
            Mask indicating peptide tokens (B*mult, L)
        multiplicity : int
            Multiplicity of conformations
            
        Returns
        -------
        dict
            BBB predictions
        """
        pad_token_mask = (
            feats["token_pad_mask"].repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        
        # Apply peptide mask
        if peptide_mask is not None:
            mask = peptide_mask.unsqueeze(-1) * pad_token_mask
        else:
            mask = pad_token_mask

        # Create pairwise mask
        pair_mask = (
            mask[:, :, None] * mask[:, None, :]
        ) * (
            1
            - torch.eye(mask.shape[1], device=mask.device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # Pool pairwise embeddings to sequence-level representation
        # Average over peptide tokens
        g_pairwise = torch.sum(z * pair_mask, dim=(1, 2)) / (
            torch.sum(pair_mask, dim=(1, 2)) + 1e-7
        )  # (B*mult, token_z)

        # Project to token_s dimension
        g = self.bbb_out_mlp(g_pairwise)  # (B*mult, token_s)

        # Optionally incorporate sequence features
        if self.use_sequence_features and s_inputs is not None:
            # Pool sequence embeddings over peptide tokens
            s_pooled = torch.sum(
                s_inputs * mask, dim=1
            ) / (torch.sum(mask, dim=1) + 1e-7)  # (B*mult, token_s)
            s_features = self.seq_feature_mlp(s_pooled)
            # Concatenate pairwise and sequence features
            g = torch.cat([g, s_features], dim=-1)  # (B*mult, 2*token_s)

        # Generate predictions
        out_dict = {}
        
        if self.task_type == "regression":
            # Regression: continuous permeability score
            bbb_pred_value = self.to_bbb_pred_value(g).reshape(-1, 1)
            # Binary logits as auxiliary output (from regression score)
            bbb_logits_binary = self.to_bbb_logits_binary(bbb_pred_value).reshape(-1, 1)
            out_dict["bbb_pred_value"] = bbb_pred_value
        else:
            # Binary classification
            bbb_logits_binary = self.to_bbb_logits_binary(g).reshape(-1, 1)
            out_dict["bbb_pred_value"] = bbb_logits_binary  # For compatibility

        out_dict["bbb_logits_binary"] = bbb_logits_binary
        out_dict["bbb_probability"] = torch.sigmoid(bbb_logits_binary)

        return out_dict

