from typing import Dict, Tuple, List
import warnings

import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot

from boltzgen.data import const

# Replace torch_scatter with native PyTorch implementation
from boltzgen.model.modules.scatter_utils import scatter_sum, scatter_softmax
import torch.nn.functional as F


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = (dist.unsqueeze(-1) - self.offset.view(1, 1, -1)).flatten(start_dim=1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def softmax_dropout(
    attn_weight: Tensor, softmax_dropout: float, dst_idx: Tensor = None
):
    for _ in range(10):
        dropout_mask = torch.rand_like(attn_weight) < softmax_dropout
        not_all_drop_mask = scatter_sum(~dropout_mask, index=dst_idx, dim=0)
        if not_all_drop_mask.all():
            attn_weight = attn_weight.masked_fill(dropout_mask, -float("inf"))
            return attn_weight
    raise RuntimeError(
        "Softmax dropout failed to keep at least one edge for each node after 10 attempts."
    )


class MLPAttnGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.edge_FFN = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.pair_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.pair_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        z = z + self.edge_FFN(torch.cat([s[src_idx], s[dst_idx], z], dim=1))

        attn_weight = self.attn_weight_mlp(
            torch.cat([s[dst_idx], s[src_idx], z], dim=1)
        )
        attn_value = self.attn_value_mlp(torch.cat([s[src_idx], z], dim=1))

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s, z


class MLPAttnGNNDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s

    def sample(self, s, z):
        dst_idx = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s


class InverseFoldingEncoder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        enable_input_embedder: bool = False,
        **kwargs, # old checkpoint compatibility
    ):
        """Initialize the Inverse Folding Encoder."""
        super().__init__()

        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.num_heads = num_heads
        self.r_max = 32
        self.enable_input_embedder = enable_input_embedder
        self.edge_input_dim = (
            256 + 2 * self.r_max + 2 + 1 + 1 + len(const.bond_types) + 1 + 1
        )

        self.linear_token_to_node = nn.Linear(
            135 if not self.enable_input_embedder else self.token_s, self.node_dim
        )
        self.linear_token_to_pair = nn.Linear(self.edge_input_dim, self.pair_dim)

        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_encoder_layers):
            layer = MLPAttnGNN(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.encoder_layers.append(layer)

        self.distance_gaussian_smearing = GaussianSmearing(
            start=0.0,
            stop=20.0,
            num_gaussians=16,
        )

    @torch.no_grad()
    def init_knn_graph(self, feats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        valid_mask = (
            feats["token_resolved_mask"].bool() & feats["token_pad_mask"].bool()
        )
        B, N = valid_mask.shape
        valid_mask_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        token_index = (valid_mask.flatten().cumsum(dim=0) - 1).view(B, N)
        topk = min(self.topk, N)

        coords = feats["center_coords"]
        dists = torch.cdist(coords, coords)
        dists = dists.masked_fill(~valid_mask_pair, float("inf"))
        src_idx = torch.topk(dists, topk, largest=False).indices
        dst_idx = torch.arange(N, device=src_idx.device)[None, :, None].expand(
            B, -1, topk
        )

        assert feats["token_bonds"].shape[-1] == 1
        token_bonds = feats["token_bonds"][..., 0]
        type_bonds = feats["type_bonds"]
        token_bonds = torch.gather(token_bonds, 2, src_idx)
        type_bonds = torch.gather(type_bonds, 2, src_idx)

        src_idx, dst_idx = src_idx.flatten(start_dim=1), dst_idx.flatten(start_dim=1)
        token_bonds, type_bonds = token_bonds.flatten(), type_bonds.flatten()

        src_valid_mask = torch.gather(valid_mask, 1, src_idx).flatten()
        dst_valid_mask = torch.gather(valid_mask, 1, dst_idx).flatten()
        edge_valid_mask = src_valid_mask & dst_valid_mask

        src_idx = torch.gather(token_index, 1, src_idx).flatten()[edge_valid_mask]
        dst_idx = torch.gather(token_index, 1, dst_idx).flatten()[edge_valid_mask]
        edge_idx = torch.stack([src_idx, dst_idx], dim=0)

        token_bonds, type_bonds = (
            token_bonds[edge_valid_mask],
            type_bonds[edge_valid_mask],
        )
        type_bonds = one_hot(type_bonds, num_classes=len(const.bond_types) + 1)
        bond = torch.cat([token_bonds[..., None], type_bonds], dim=-1)

        return edge_idx, valid_mask, bond

    @torch.no_grad()
    def extract_geo_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        token_to_bb4_atoms = feats["token_to_bb4_atoms"]
        r = feats["coords"]
        if self.training and self.inverse_fold_noise > 0:
            noise = torch.randn_like(r) * self.inverse_fold_noise
            r = r + noise
        B, N = valid_mask.shape
        r_repr = torch.bmm(
            token_to_bb4_atoms.float().view(B, N * 4, -1), r.view(B, -1, 3)
        )
        r_repr = r_repr.reshape(B, N, 4, 3)
        r_repr = r_repr[valid_mask]

        dist = torch.norm(
            r_repr[src_idx, None] - r_repr[dst_idx, :, None], dim=-1
        ).flatten(start_dim=1)
        dist = self.distance_gaussian_smearing(dist)
        return dist

    @torch.no_grad()
    def extract_attr_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        feature_asym_id = feats["feature_asym_id"][valid_mask]
        b_same_chain = feature_asym_id[src_idx] == feature_asym_id[dst_idx]

        feature_residue_index = feats["feature_residue_index"][valid_mask]
        b_same_residue = (
            feature_residue_index[src_idx] == feature_residue_index[dst_idx]
        )
        d_residue = feature_residue_index[dst_idx] - feature_residue_index[src_idx]
        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1,
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        edge_attr = torch.cat(
            [
                a_rel_pos.float(),
                b_same_chain[..., None].float(),
                b_same_residue[..., None].float(),
            ],
            dim=-1,
        )

        b_standard = feats["is_standard"].bool()[valid_mask][..., None]
        mol_type = feats["mol_type"][valid_mask]
        mol_type_one_hot = F.one_hot(mol_type, num_classes=len(const.chain_type_ids))
        nonpolymer_mask = mol_type == const.chain_type_ids["NONPOLYMER"]
        modified = feats["modified"].unsqueeze(-1)[valid_mask]

        atom_feats = torch.cat(
            [
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
            ],
            dim=-1,
        )

        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_feats)

            atom_mask = feats["atom_pad_mask"]
            atom_padding_sum_to_token = torch.bmm(
                atom_to_token.transpose(1, 2), 1 - atom_mask.unsqueeze(-1)
            )
            assert atom_padding_sum_to_token[valid_mask].sum() == 0
        a = a[valid_mask]

        node_attr = torch.cat(
            [
                b_standard.float(),
                modified,
                mol_type_one_hot.float(),
                a,
            ],
            dim=-1,
        )

        return node_attr, edge_attr

    def forward(self, feats):
        edge_idx, valid_mask, bond = self.init_knn_graph(feats)
        geo_feat = self.extract_geo_feat(feats, edge_idx, valid_mask)
        node_attr, edge_attr = self.extract_attr_feat(feats, edge_idx, valid_mask)
        if self.enable_input_embedder:
            node_attr = feats["s_inputs"][valid_mask]

        N = valid_mask.sum()
        s = self.linear_token_to_node(node_attr)
        z = self.linear_token_to_pair(torch.cat([geo_feat, edge_attr, bond], dim=-1))

        for layer in self.encoder_layers:
            s, z = layer(s, z, edge_idx)

        return edge_idx, valid_mask, s, z



class InverseFoldingDecoder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        num_decoder_layers: int = 3,
        inverse_fold_restriction: List[str] = [],
        sampling_temperature: float = 0.1,
        tie_symmetric_sequences: bool = True,
        **kwargs, # old checkpoint compatibility
    ):
        super().__init__()
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.num_heads = num_heads
        self.num_res_type = const.num_tokens
        self.num_decoder_layers = num_decoder_layers
        self.inverse_fold_restriction = inverse_fold_restriction
        self.sampling_temperature = sampling_temperature
        self.tie_symmetric_sequences = tie_symmetric_sequences

        self.decoder_layers = nn.ModuleList()
        self.inf = 10**6
        for i in range(self.num_decoder_layers):
            layer = MLPAttnGNNDecoder(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.decoder_layers.append(layer)

        self.seq_to_s = nn.Linear(const.num_tokens, self.node_dim)
        self.predictor = nn.Linear(self.node_dim, const.num_tokens, bias=False)

        with torch.no_grad():
            # init the output of the predictor to be zero
            self.predictor.weight.zero_()

    def _build_symmetric_groups(
        self,
        feats: Dict[str, Tensor],
        valid_mask: Tensor,
        design_mask: Tensor,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """Build mapping from positions to symmetric groups for homomer tying."""
        from collections import defaultdict

        symmetric_group = feats["symmetric_group"][valid_mask]
        res_idx = feats["feature_residue_index"][valid_mask]

        # Group by (symmetric_group, res_idx) - positions that should share sequence
        key_to_positions = defaultdict(list)

        num_nodes = symmetric_group.shape[0]
        for i in range(num_nodes):
            if design_mask[i]:
                group = symmetric_group[i].item()
                if group > 0:  # 0 = no group
                    key = (group, res_idx[i].item())
                    key_to_positions[key].append(i)

        # Build symmetric groups (only groups with >1 member)
        sym_groups = {}
        position_to_group = {}
        group_id = 0

        for positions in key_to_positions.values():
            if len(positions) > 1:
                sym_groups[group_id] = positions
                for pos in positions:
                    position_to_group[pos] = group_id
                group_id += 1

        return sym_groups, position_to_group

    def forward(self, s, z, edge_idx, valid_mask, feats):
        with torch.no_grad():
            src_idx, dst_idx = edge_idx[0], edge_idx[1]
            rand = torch.rand(valid_mask.sum(), device=valid_mask.device)
            res_type_edge_visibility = rand[src_idx] < rand[dst_idx]
            res_type_clone = feats["res_type_clone"].bool()
            res_type_clone = res_type_clone[valid_mask][src_idx]
            res_type_clone = res_type_clone * res_type_edge_visibility[:, None]
            res_type_clone = res_type_clone.to(z)
        res_rep = self.seq_to_s(res_type_clone)
        neighbors_rep = torch.concat([z, s[src_idx] + res_rep], dim=-1)
        for layer in self.decoder_layers:
            s = layer(s, neighbors_rep, edge_idx)

        logits = self.predictor(s)

        B, N = valid_mask.shape
        logist_dense = torch.zeros(B, N, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits

        out_dict = {
            "logits": logist_dense,
            "res_type": logist_dense,
            "valid_mask": valid_mask,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, s, z, edge_idx, valid_mask, feats):
        """Sample the output from the decoder."""
        num_nodes = s.shape[0]

        if "inverse_fold_design_mask" in feats:
            design_mask = feats["inverse_fold_design_mask"].bool()[valid_mask]
        else:
            design_mask = feats["design_mask"].bool()[valid_mask]
        num_not_design = (~design_mask).sum().item()
        num_design = design_mask.sum().item()
        assert num_design == num_nodes - num_not_design, (
            f"num_design: {num_design}, num_not_design: {num_not_design}"
        )

        # Create per-residue restriction mask
        # Initialize with per-residue constraints from YAML if available
        if "aa_constraint_mask" in feats:
            constraint_mask = feats["aa_constraint_mask"][valid_mask]
            # Validate shape: should be (num_nodes, 20) for 20 canonical amino acids
            expected_shape = (num_nodes, len(const.canonical_tokens))
            if constraint_mask.shape != expected_shape:
                warnings.warn(
                    f"aa_constraint_mask shape mismatch: "
                    f"got {constraint_mask.shape}, expected {expected_shape}. "
                    f"Ignoring per-residue constraints.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                per_residue_mask = torch.zeros(num_nodes, len(const.canonical_tokens), device=s.device)
            else:
                # Check for positions with ALL AAs blocked (would leave no valid options).
                # This can happen when per-residue 'allowed' and global '--inverse_fold_avoid'
                # combine to eliminate every amino acid at a position.
                all_blocked = constraint_mask.sum(dim=1) >= len(const.canonical_tokens)
                if all_blocked.any():
                    blocked_positions = torch.where(all_blocked)[0].tolist()
                    warnings.warn(
                        f"Positions {blocked_positions} have ALL amino acids blocked "
                        f"(likely conflict between per-residue 'allowed' and global "
                        f"'--inverse_fold_avoid'). Relaxing constraints for these positions "
                        f"to allow all amino acids.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    constraint_mask[all_blocked] = 0  # Allow all AAs for fully-blocked positions

                # Per-residue constraints: values 0=allowed, 1=disallowed
                per_residue_mask = constraint_mask.clone()
                # Convert binary mask to logit bias: 0 -> 0, 1 -> -inf
                per_residue_mask = per_residue_mask * (-self.inf)
        else:
            # No per-residue constraints: all AAs allowed
            per_residue_mask = torch.zeros(num_nodes, len(const.canonical_tokens), device=s.device)

        # Add global restriction (backward compatible with --inverse_fold_avoid CLI)
        if len(self.inverse_fold_restriction) > 0:
            global_mask = torch.zeros(len(const.canonical_tokens), device=s.device)
            for res_type in self.inverse_fold_restriction:
                global_mask[const.canonical_tokens.index(res_type)] = -self.inf
            # Combine: global restrictions apply to ALL positions additively
            per_residue_mask = per_residue_mask + global_mask.unsqueeze(0)

        order = torch.randperm(num_nodes, device=s.device).cpu().numpy().tolist()
        # Non-design residues are not sampled and used as the condition. So the order should filter them out.
        if num_not_design > 0:
            id_not_design = torch.where(~design_mask)[0].cpu().numpy().tolist()
            for i in id_not_design:
                order.remove(i)
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            decoded_seq[~design_mask] = logits[~design_mask] = feats["res_type_clone"][
                valid_mask
            ][~design_mask].float()
        else:
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)

        # Build symmetric groups for homomer tying
        if self.tie_symmetric_sequences and "symmetric_group" in feats:
            sym_groups, position_to_group = self._build_symmetric_groups(
                feats, valid_mask, design_mask
            )
            sampled = set(torch.where(~design_mask)[0].cpu().numpy().tolist()) if num_not_design > 0 else set()
        else:
            sym_groups, position_to_group = {}, {}
            sampled = set()

        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        # decoding in order
        for i in order:
            # Skip if already sampled (symmetric position was processed earlier)
            if self.tie_symmetric_sequences and i in sampled:
                continue

            # Get symmetric positions (or just [i] if no symmetry)
            if self.tie_symmetric_sequences and i in position_to_group:
                positions = sym_groups[position_to_group[i]]
            else:
                positions = [i]

            # Aggregate logits from all symmetric positions
            aggregated_logits = None
            for pos in positions:
                s_pos = s[pos : pos + 1]
                edge_mask_pos = dst_idx == pos
                z_pos = z[edge_mask_pos]
                src_idx_pos = src_idx[edge_mask_pos]
                res_type_pos = decoded_seq[src_idx_pos]
                res_rep = self.seq_to_s(res_type_pos)
                neighbors_rep_pos = torch.concat([z_pos, s[src_idx_pos] + res_rep], dim=-1)

                s_temp = s_pos
                for layer in self.decoder_layers:
                    s_temp = layer.sample(s_temp, neighbors_rep_pos)

                logits_pos = self.predictor(s_temp)
                if aggregated_logits is None:
                    aggregated_logits = logits_pos
                else:
                    aggregated_logits = aggregated_logits + logits_pos

            # Average logits across symmetric positions
            aggregated_logits = aggregated_logits / len(positions)

            # Sample from aggregated logits
            pred_canonical = (
                aggregated_logits[
                    :,
                    const.canonicals_offset : len(const.canonical_tokens)
                    + const.canonicals_offset,
                ]
                + per_residue_mask[i : i + 1]  # Position-specific mask
            )
            if self.sampling_temperature is None:
                ids_canonical = torch.argmax(pred_canonical, dim=-1)
            else:
                ids_canonical = torch.multinomial(
                    F.softmax(pred_canonical / self.sampling_temperature, dim=-1),
                    num_samples=1,
                ).squeeze(-1)

            ids = ids_canonical + const.canonicals_offset
            pred_one_hot = F.one_hot(ids, num_classes=const.num_tokens)

            # Apply same residue to all symmetric positions
            for pos in positions:
                decoded_seq[pos] = pred_one_hot
                logits[pos] = aggregated_logits
                if self.tie_symmetric_sequences:
                    sampled.add(pos)

        n_tokens = valid_mask.shape[1]
        res_type = torch.zeros(1, n_tokens, self.num_res_type, device=s.device)
        res_type[valid_mask] = decoded_seq
        unk_ids = torch.full(
            [(~valid_mask).sum().item()], const.tokens.index("UNK"), device=s.device
        )
        unk_value = F.one_hot(unk_ids, num_classes=const.num_tokens)
        res_type[~valid_mask] = unk_value.float()
        feats["res_type"] = res_type.long()

        logist_dense = torch.zeros(1, n_tokens, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits
        logist_dense[~valid_mask] = feats["res_type_clone"][~valid_mask].float()

        out_dict = {
            "logits": logist_dense,
            "res_type": res_type,
            "valid_mask": valid_mask,
        }
        out_dict["sample_atom_coords"] = feats["coords"][0]
        return out_dict
