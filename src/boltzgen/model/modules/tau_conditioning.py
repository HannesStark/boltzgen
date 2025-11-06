"""
Tau Conditioning Module for BoltzGen.

This module enables conditioning peptide generation on Tau protein fragments
using protein language model embeddings (ESM-2 or ProtT5). This allows the
model to learn sequence–sequence interactions and generate Tau-specific binders.
"""

import torch
from torch import nn
from typing import Dict, Optional, Tuple
from torch import Tensor


class TauEmbeddingProjector(nn.Module):
    """
    Projects protein language model embeddings (ESM-2/ProtT5) to BoltzGen dimensions.
    
    Handles different embedding dimensions from various protein language models:
    - ESM-2: typically 1280-dim per-token embeddings
    - ProtT5: typically 1024-dim per-token embeddings
    - Can also accept pre-computed embeddings of any dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        token_s: int,
        token_z: int,
        tau_embedding_type: str = "esm2",
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize Tau embedding projector.
        
        Parameters
        ----------
        input_dim : int
            Input dimension of Tau embeddings (from language model)
        token_s : int
            Target sequence embedding dimension
        token_z : int
            Target pairwise embedding dimension
        tau_embedding_type : str, default='esm2'
            Type of embedding: 'esm2', 'prott5', or 'custom'
        use_layer_norm : bool, default=True
            Whether to use layer normalization
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.token_s = token_s
        self.token_z = token_z
        self.tau_embedding_type = tau_embedding_type
        
        # Project to sequence embedding dimension
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, token_s * 2),
            nn.LayerNorm(token_s * 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_s * 2, token_s),
        )
        
        # Project to pairwise embedding dimension
        self.pair_proj = nn.Sequential(
            nn.Linear(input_dim, token_z * 2),
            nn.LayerNorm(token_z * 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_z * 2, token_z),
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.seq_proj[0].weight)
        nn.init.zeros_(self.seq_proj[0].bias)
        nn.init.xavier_uniform_(self.pair_proj[0].weight)
        nn.init.zeros_(self.pair_proj[0].bias)
    
    def forward(
        self,
        tau_embeddings: Tensor,
        tau_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project Tau embeddings to BoltzGen dimensions.
        
        Parameters
        ----------
        tau_embeddings : Tensor
            Tau protein language model embeddings (B, L_tau, input_dim)
        tau_mask : Tensor, optional
            Mask for Tau tokens (B, L_tau), 1=valid, 0=padded
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            - tau_seq_emb: Sequence embeddings (B, L_tau, token_s)
            - tau_pair_emb: Pairwise embeddings (B, L_tau, token_z)
        """
        # Project to sequence and pairwise dimensions
        tau_seq_emb = self.seq_proj(tau_embeddings)  # (B, L_tau, token_s)
        tau_pair_emb = self.pair_proj(tau_embeddings)  # (B, L_tau, token_z)
        
        # Apply mask if provided
        if tau_mask is not None:
            tau_mask = tau_mask.unsqueeze(-1).float()
            tau_seq_emb = tau_seq_emb * tau_mask
            tau_pair_emb = tau_pair_emb * tau_mask.unsqueeze(-1)
        
        return tau_seq_emb, tau_pair_emb


class TauCrossAttention(nn.Module):
    """
    Cross-attention module for Tau–peptide interactions.
    
    Allows peptide tokens to attend to Tau embeddings, enabling
    sequence–sequence interaction learning.
    """
    
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize Tau cross-attention.
        
        Parameters
        ----------
        token_s : int
            Sequence embedding dimension
        token_z : int
            Pairwise embedding dimension
        num_heads : int, default=8
            Number of attention heads
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.token_s = token_s
        self.token_z = token_z
        self.num_heads = num_heads
        self.head_dim_s = token_s // num_heads
        self.head_dim_z = token_z // num_heads
        
        assert token_s % num_heads == 0, f"token_s ({token_s}) must be divisible by num_heads ({num_heads})"
        assert token_z % num_heads == 0, f"token_z ({token_z}) must be divisible by num_heads ({num_heads})"
        
        # Sequence-level cross-attention
        self.seq_cross_attn = nn.MultiheadAttention(
            embed_dim=token_s,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Pairwise-level cross-attention (for pairwise embeddings)
        # This attends peptide pairs to Tau pairs
        self.pair_q_proj = nn.Linear(token_z, token_z)
        self.pair_k_proj = nn.Linear(token_z, token_z)
        self.pair_v_proj = nn.Linear(token_z, token_z)
        self.pair_out_proj = nn.Linear(token_z, token_z)
        self.pair_norm = nn.LayerNorm(token_z)
        self.pair_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        peptide_seq: Tensor,  # (B, L_peptide, token_s)
        peptide_pair: Tensor,  # (B, L_peptide, L_peptide, token_z)
        tau_seq: Tensor,  # (B, L_tau, token_s)
        tau_pair: Tensor,  # (B, L_tau, L_tau, token_z)
        peptide_mask: Optional[Tensor] = None,  # (B, L_peptide)
        tau_mask: Optional[Tensor] = None,  # (B, L_tau)
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply cross-attention between peptide and Tau embeddings.
        
        Parameters
        ----------
        peptide_seq : Tensor
            Peptide sequence embeddings
        peptide_pair : Tensor
            Peptide pairwise embeddings
        tau_seq : Tensor
            Tau sequence embeddings
        tau_pair : Tensor
            Tau pairwise embeddings
        peptide_mask : Tensor, optional
            Mask for peptide tokens
        tau_mask : Tensor, optional
            Mask for Tau tokens
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            - conditioned_seq: Tau-conditioned peptide sequence embeddings
            - conditioned_pair: Tau-conditioned peptide pairwise embeddings
        """
        # Sequence-level cross-attention
        # Peptide queries attend to Tau keys/values
        attn_mask = None
        if tau_mask is not None:
            # Create attention mask (invert for torch: True = mask out)
            attn_mask = (~tau_mask.bool()).unsqueeze(1)  # (B, 1, L_tau)
        
        conditioned_seq, _ = self.seq_cross_attn(
            query=peptide_seq,
            key=tau_seq,
            value=tau_seq,
            key_padding_mask=attn_mask if attn_mask is not None else None,
        )  # (B, L_peptide, token_s)
        
        # Pairwise-level cross-attention
        B, L_pep, L_pep, Z = peptide_pair.shape
        _, L_tau, _, _ = tau_pair.shape
        
        # Reshape for attention: (B, L_pep^2, Z) -> query
        peptide_pair_flat = peptide_pair.reshape(B, L_pep * L_pep, Z)
        
        # Reshape Tau pairwise: (B, L_tau^2, Z) -> key/value
        tau_pair_flat = tau_pair.reshape(B, L_tau * L_tau, Z)
        
        # Compute pairwise attention
        q = self.pair_q_proj(peptide_pair_flat)  # (B, L_pep^2, Z)
        k = self.pair_k_proj(tau_pair_flat)  # (B, L_tau^2, Z)
        v = self.pair_v_proj(tau_pair_flat)  # (B, L_tau^2, Z)
        
        # Reshape for multi-head attention
        q = q.view(B, L_pep * L_pep, self.num_heads, self.head_dim_z).transpose(1, 2)
        k = k.view(B, L_tau * L_tau, self.num_heads, self.head_dim_z).transpose(1, 2)
        v = v.view(B, L_tau * L_tau, self.num_heads, self.head_dim_z).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim_z ** 0.5)
        
        # Apply mask if provided
        if tau_mask is not None:
            # Create pairwise mask from sequence mask
            tau_pair_mask = tau_mask.unsqueeze(-1) * tau_mask.unsqueeze(-2)  # (B, L_tau, L_tau)
            tau_pair_mask_flat = tau_pair_mask.reshape(B, 1, 1, L_tau * L_tau)
            scores = scores.masked_fill(~tau_pair_mask_flat.bool(), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.pair_dropout(attn_weights)
        
        conditioned_pair_flat = torch.matmul(attn_weights, v)  # (B, num_heads, L_pep^2, head_dim_z)
        conditioned_pair_flat = conditioned_pair_flat.transpose(1, 2).contiguous()
        conditioned_pair_flat = conditioned_pair_flat.view(B, L_pep * L_pep, Z)
        
        conditioned_pair_flat = self.pair_out_proj(conditioned_pair_flat)
        conditioned_pair_flat = self.pair_norm(conditioned_pair_flat + peptide_pair_flat)
        conditioned_pair_flat = self.pair_dropout(conditioned_pair_flat)
        
        # Reshape back to pairwise format
        conditioned_pair = conditioned_pair_flat.reshape(B, L_pep, L_pep, Z)
        
        return conditioned_seq, conditioned_pair


class TauConditioning(nn.Module):
    """
    Main Tau conditioning module.
    
    Integrates Tau protein embeddings into the BoltzGen model,
    enabling Tau-conditioned peptide generation.
    """
    
    def __init__(
        self,
        token_s: int,
        token_z: int,
        tau_embedding_dim: int = 1280,  # ESM-2 default
        tau_embedding_type: str = "esm2",
        use_cross_attention: bool = True,
        num_cross_attention_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize Tau conditioning module.
        
        Parameters
        ----------
        token_s : int
            Sequence embedding dimension
        token_z : int
            Pairwise embedding dimension
        tau_embedding_dim : int, default=1280
            Input dimension of Tau embeddings
        tau_embedding_type : str, default='esm2'
            Type of embedding: 'esm2', 'prott5', or 'custom'
        use_cross_attention : bool, default=True
            Whether to use cross-attention for Tau–peptide interaction
        num_cross_attention_layers : int, default=1
            Number of cross-attention layers
        num_heads : int, default=8
            Number of attention heads
        dropout : float, default=0.1
            Dropout rate
        """
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        # Project Tau embeddings to BoltzGen dimensions
        self.tau_projector = TauEmbeddingProjector(
            input_dim=tau_embedding_dim,
            token_s=token_s,
            token_z=token_z,
            tau_embedding_type=tau_embedding_type,
            dropout=dropout,
        )
        
        # Cross-attention layers for Tau–peptide interaction
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                TauCrossAttention(
                    token_s=token_s,
                    token_z=token_z,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_cross_attention_layers)
            ])
        else:
            # Simple addition/gating mechanism
            self.tau_gate_seq = nn.Sequential(
                nn.Linear(token_s * 2, token_s),
                nn.Sigmoid(),
            )
            self.tau_gate_pair = nn.Sequential(
                nn.Linear(token_z * 2, token_z),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        peptide_seq: Tensor,  # (B, L_peptide, token_s)
        peptide_pair: Tensor,  # (B, L_peptide, L_peptide, token_z)
        tau_embeddings: Tensor,  # (B, L_tau, tau_embedding_dim)
        tau_mask: Optional[Tensor] = None,  # (B, L_tau)
        peptide_mask: Optional[Tensor] = None,  # (B, L_peptide)
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply Tau conditioning to peptide embeddings.
        
        Parameters
        ----------
        peptide_seq : Tensor
            Peptide sequence embeddings
        peptide_pair : Tensor
            Peptide pairwise embeddings
        tau_embeddings : Tensor
            Tau protein language model embeddings
        tau_mask : Tensor, optional
            Mask for Tau tokens
        peptide_mask : Tensor, optional
            Mask for peptide tokens
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            - conditioned_seq: Tau-conditioned peptide sequence embeddings
            - conditioned_pair: Tau-conditioned peptide pairwise embeddings
        """
        # Project Tau embeddings
        tau_seq, tau_pair_proj = self.tau_projector(tau_embeddings, tau_mask)
        
        # Create Tau pairwise matrix from sequence embeddings
        # tau_pair[i,j] = interaction between Tau residue i and j
        B, L_tau, Z = tau_pair_proj.shape
        # Create pairwise matrix: (B, L_tau, L_tau, Z)
        tau_pair = tau_pair_proj.unsqueeze(-2).expand(-1, -1, L_tau, -1)  # (B, L_tau, L_tau, Z)
        # Add symmetric interactions
        tau_pair = tau_pair + tau_pair.transpose(-3, -2)
        
        conditioned_seq = peptide_seq
        conditioned_pair = peptide_pair
        
        if self.use_cross_attention:
            # Apply cross-attention layers
            for layer in self.cross_attention_layers:
                conditioned_seq, conditioned_pair = layer(
                    peptide_seq=conditioned_seq,
                    peptide_pair=conditioned_pair,
                    tau_seq=tau_seq,
                    tau_pair=tau_pair,
                    peptide_mask=peptide_mask,
                    tau_mask=tau_mask,
                )
        else:
            # Simple gating mechanism
            # Pool Tau embeddings (e.g., mean pooling)
            if tau_mask is not None:
                tau_mask_expanded = tau_mask.unsqueeze(-1).float()
                tau_seq_pooled = (tau_seq * tau_mask_expanded).sum(dim=1) / (
                    tau_mask_expanded.sum(dim=1) + 1e-7
                )  # (B, token_s)
            else:
                tau_seq_pooled = tau_seq.mean(dim=1)  # (B, token_s)
            
            # Gate peptide embeddings with Tau information
            tau_seq_pooled_expanded = tau_seq_pooled.unsqueeze(1).expand(-1, peptide_seq.shape[1], -1)
            gated_seq = torch.cat([peptide_seq, tau_seq_pooled_expanded], dim=-1)
            gate_seq = self.tau_gate_seq(gated_seq)
            conditioned_seq = peptide_seq + gate_seq * tau_seq_pooled_expanded
        
        return conditioned_seq, conditioned_pair

