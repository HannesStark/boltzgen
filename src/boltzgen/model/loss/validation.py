from typing import Dict
import torch
from torch import Tensor, nn

from boltzgen.data import const
from boltzgen.model.loss.confidence import (
    compute_frame_pred,
    express_coordinate_in_frame,
    lddt_dist,
)
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.model.modules.utils import get_autocast_device_type


def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
    exclude_ions=False,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        # extract necessary features
        B = atom_mask.shape[0]

        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(multiplicity, 0)

        chain_id = feats["asym_id"]
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )
        atom_chain_id = atom_chain_id.repeat_interleave(multiplicity, 0)
        same_chain_mask = (
            atom_chain_id[:, :, None] == atom_chain_id[:, None, :]
        ).float()

        pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )

        ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
        if exclude_ions:
            ions = (torch.sum(same_chain_mask * pair_mask, dim=-1) == 1).float()
            ligand_mask = ligand_mask * (1 - ions)

    dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
    design_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()
    protein_mask = protein_mask * (1 - design_mask)

    nucleotide_mask = dna_mask + rna_mask

    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )
    del nucleotide_mask

    # compute different lddts
    design_protein_mask = pair_mask * (
        design_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * design_mask[:, None, :]
    )
    design_protein_lddt, design_protein_total = lddt_dist(
        pred_d, true_d, design_protein_mask, cutoff
    )
    del design_protein_mask

    design_ligand_mask = pair_mask * (
        design_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * design_mask[:, None, :]
    )
    design_ligand_lddt, design_ligand_total = lddt_dist(
        pred_d, true_d, design_ligand_mask, cutoff
    )
    del design_ligand_mask

    design_dna_mask = pair_mask * (
        design_mask[:, :, None] * dna_mask[:, None, :]
        + dna_mask[:, :, None] * design_mask[:, None, :]
    )
    design_dna_lddt, design_dna_total = lddt_dist(
        pred_d, true_d, design_dna_mask, cutoff
    )
    del design_dna_mask

    design_rna_mask = pair_mask * (
        design_mask[:, :, None] * rna_mask[:, None, :]
        + rna_mask[:, :, None] * design_mask[:, None, :]
    )
    design_rna_lddt, design_rna_total = lddt_dist(
        pred_d, true_d, design_rna_mask, cutoff
    )
    del design_rna_mask

    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask
    del dna_mask

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask
    del rna_mask

    # compute modified residues lddt
    modified_residue = (
        feats["mol_type"] != const.chain_type_ids["NONPOLYMER"]
    ).float() * (
        feats["res_type"][:, :, const.token_ids["UNK"]]
        + feats["res_type"][:, :, const.token_ids["DN"]]
        + feats["res_type"][:, :, const.token_ids["N"]]
    )
    modified_atom_mask = (
        torch.bmm(
            feats["atom_to_token"].float(), modified_residue.unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    modified_atom_mask = modified_atom_mask.repeat_interleave(multiplicity, 0)
    modified_vs_all_mask = pair_mask * modified_atom_mask[:, :, None]
    modified_lddt, modified_total = lddt_dist(
        pred_d, true_d, modified_vs_all_mask, cutoff
    )
    del modified_vs_all_mask
    del modified_atom_mask
    del modified_residue

    intra_ligand_mask = (
        pair_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    intra_design_mask = (
        pair_mask
        * same_chain_mask
        * (design_mask[:, :, None] * design_mask[:, None, :])
    )
    intra_design_lddt, intra_design_total = lddt_dist(
        pred_d, true_d, intra_design_mask, cutoff
    )
    del intra_design_mask

    design_design_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (design_mask[:, :, None] * design_mask[:, None, :])
    )
    design_design_lddt, design_design_total = lddt_dist(
        pred_d, true_d, design_design_mask, cutoff
    )
    del design_design_mask

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
        "design_protein": design_protein_lddt,
        "design_ligand": design_ligand_lddt,
        "design_dna": design_dna_lddt,
        "design_rna": design_rna_lddt,
        "intra_design": intra_design_lddt,
        "design_design": design_design_lddt,
        "modified": modified_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
        "design_protein": design_protein_total,
        "design_ligand": design_ligand_total,
        "design_dna": design_dna_total,
        "design_rna": design_rna_total,
        "intra_design": intra_design_total,
        "design_design": design_design_total,
        "modified": modified_total,
    }
    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def factored_lddt_loss_ensemble(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
    exclude_ions=False,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        # DEPRECATED
        # extract necessary features
        K, L = true_atom_coords.shape[1:3]

        true_atom_coords = true_atom_coords.reshape(K * multiplicity, L, 3)
        pred_atom_coords = pred_atom_coords.repeat_interleave(K, 0)

        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(multiplicity * K, 0)

        chain_id = feats["asym_id"]
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )
        atom_chain_id = atom_chain_id.repeat_interleave(multiplicity * K, 0)
        same_chain_mask = (
            atom_chain_id[:, :, None] == atom_chain_id[:, None, :]
        ).float()

        pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )

        ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
        if exclude_ions:
            ions = (torch.sum(same_chain_mask * pair_mask, dim=-1) == 1).float()
            ligand_mask = ligand_mask * (1 - ions)

        dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
        rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
        protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()

        nucleotide_mask = dna_mask + rna_mask

        true_d = torch.cdist(true_atom_coords, true_atom_coords)
        pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

        cutoff = 15 + 15 * (
            1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
        )

        # compute different lddts
        dna_protein_mask = pair_mask * (
            dna_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * dna_mask[:, None, :]
        )
        dna_protein_lddt, dna_protein_total = lddt_dist(
            pred_d, true_d, dna_protein_mask, cutoff
        )
        del dna_protein_mask

        rna_protein_mask = pair_mask * (
            rna_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * rna_mask[:, None, :]
        )
        rna_protein_lddt, rna_protein_total = lddt_dist(
            pred_d, true_d, rna_protein_mask, cutoff
        )
        del rna_protein_mask

        ligand_protein_mask = pair_mask * (
            ligand_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * ligand_mask[:, None, :]
        )
        ligand_protein_lddt, ligand_protein_total = lddt_dist(
            pred_d, true_d, ligand_protein_mask, cutoff
        )
        del ligand_protein_mask

        dna_ligand_mask = pair_mask * (
            dna_mask[:, :, None] * ligand_mask[:, None, :]
            + ligand_mask[:, :, None] * dna_mask[:, None, :]
        )
        dna_ligand_lddt, dna_ligand_total = lddt_dist(
            pred_d, true_d, dna_ligand_mask, cutoff
        )
        del dna_ligand_mask

        rna_ligand_mask = pair_mask * (
            rna_mask[:, :, None] * ligand_mask[:, None, :]
            + ligand_mask[:, :, None] * rna_mask[:, None, :]
        )
        rna_ligand_lddt, rna_ligand_total = lddt_dist(
            pred_d, true_d, rna_ligand_mask, cutoff
        )
        del rna_ligand_mask

        intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
        intra_dna_lddt, intra_dna_total = lddt_dist(
            pred_d, true_d, intra_dna_mask, cutoff
        )
        del intra_dna_mask

        intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
        intra_rna_lddt, intra_rna_total = lddt_dist(
            pred_d, true_d, intra_rna_mask, cutoff
        )
        del intra_rna_mask

        intra_ligand_mask = (
            pair_mask
            * same_chain_mask
            * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
        )
        intra_ligand_lddt, intra_ligand_total = lddt_dist(
            pred_d, true_d, intra_ligand_mask, cutoff
        )
        del intra_ligand_mask

        intra_protein_mask = (
            pair_mask
            * same_chain_mask
            * (protein_mask[:, :, None] * protein_mask[:, None, :])
        )
        intra_protein_lddt, intra_protein_total = lddt_dist(
            pred_d, true_d, intra_protein_mask, cutoff
        )
        del intra_protein_mask

        protein_protein_mask = (
            pair_mask
            * (1 - same_chain_mask)
            * (protein_mask[:, :, None] * protein_mask[:, None, :])
        )
        protein_protein_lddt, protein_protein_total = lddt_dist(
            pred_d, true_d, protein_protein_mask, cutoff
        )
        del protein_protein_mask

        lddt_dict = {
            "dna_protein": dna_protein_lddt,
            "rna_protein": rna_protein_lddt,
            "ligand_protein": ligand_protein_lddt,
            "dna_ligand": dna_ligand_lddt,
            "rna_ligand": rna_ligand_lddt,
            "intra_ligand": intra_ligand_lddt,
            "intra_dna": intra_dna_lddt,
            "intra_rna": intra_rna_lddt,
            "intra_protein": intra_protein_lddt,
            "protein_protein": protein_protein_lddt,
        }

        for k in lddt_dict:
            lddt_dict[k] = lddt_dict[k].reshape(multiplicity, K)

        total_dict = {
            "dna_protein": dna_protein_total,
            "rna_protein": rna_protein_total,
            "ligand_protein": ligand_protein_total,
            "dna_ligand": dna_ligand_total,
            "rna_ligand": rna_ligand_total,
            "intra_ligand": intra_ligand_total,
            "intra_dna": intra_dna_total,
            "intra_rna": intra_rna_total,
            "intra_protein": intra_protein_total,
            "protein_protein": protein_protein_total,
        }
        if not cardinality_weighted:
            for key in total_dict:
                total_dict[key] = (total_dict[key] > 0.0).float()

        for k in total_dict:
            total_dict[k] = total_dict[k].reshape(multiplicity, K)

    return lddt_dict, total_dict


def factored_token_lddt_dist_loss(true_d, pred_d, feats, cardinality_weighted=False):
    # extract necessary features
    token_type = feats["mol_type"]

    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()
    design_mask = feats["design_mask"].float()
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    protein_mask = protein_mask * (1 - design_mask)
    nucleotide_mask = dna_mask + rna_mask

    token_mask = feats["token_disto_mask"]
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    design_protein_mask = token_mask * (
        design_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * design_mask[:, None, :]
    )
    design_protein_lddt, design_protein_total = lddt_dist(
        pred_d, true_d, design_protein_mask, cutoff
    )
    del design_protein_mask

    design_ligand_mask = token_mask * (
        design_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * design_mask[:, None, :]
    )
    design_ligand_lddt, design_ligand_total = lddt_dist(
        pred_d, true_d, design_ligand_mask, cutoff
    )
    del design_ligand_mask

    design_dna_mask = token_mask * (
        design_mask[:, :, None] * dna_mask[:, None, :]
        + dna_mask[:, :, None] * design_mask[:, None, :]
    )
    design_dna_lddt, design_dna_total = lddt_dist(
        pred_d, true_d, design_dna_mask, cutoff
    )
    del design_dna_mask

    design_rna_mask = token_mask * (
        design_mask[:, :, None] * rna_mask[:, None, :]
        + rna_mask[:, :, None] * design_mask[:, None, :]
    )
    design_rna_lddt, design_rna_total = lddt_dist(
        pred_d, true_d, design_rna_mask, cutoff
    )
    del design_rna_mask

    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    # compute modified residues lddt
    modified_residue = (
        feats["mol_type"] != const.chain_type_ids["NONPOLYMER"]
    ).float() * (
        feats["res_type"][:, :, const.token_ids["UNK"]]
        + feats["res_type"][:, :, const.token_ids["DN"]]
        + feats["res_type"][:, :, const.token_ids["N"]]
    )
    modified_vs_all_mask = token_mask * modified_residue[:, :, None]
    modified_lddt, modified_total = lddt_dist(
        pred_d, true_d, modified_vs_all_mask, cutoff
    )
    del modified_vs_all_mask
    del modified_residue

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
        "modified": modified_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
        "modified": modified_total,
    }

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def factored_token_lddt_dist_loss_ensemble(
    true_d, pred_d, feats, cardinality_weighted=False
):
    # DEPRECATED for memmory issues with large K
    # extract necessary features
    token_type = feats["mol_type"]

    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    nucleotide_mask = dna_mask + rna_mask

    token_mask = feats["token_disto_mask"]
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask)

    # Expand for multiple conformers
    B, K = feats["coords"].shape[0:2]
    assert B == 1, "Batch size must be 1 for token lddt loss"

    # (B, L * L) -> (B * K, L * L) ; L = # tokens
    pred_d = pred_d.repeat_interleave(K, 0)

    # (B, L) -> (B * K, L)
    ligand_mask = ligand_mask.repeat_interleave(K, 0)
    dna_mask = dna_mask.repeat_interleave(K, 0)
    rna_mask = rna_mask.repeat_interleave(K, 0)
    protein_mask = protein_mask.repeat_interleave(K, 0)
    nucleotide_mask = nucleotide_mask.repeat_interleave(K, 0)
    token_mask = token_mask.repeat_interleave(K, 0)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    # compute modified residues lddt
    modified_residue = (
        feats["mol_type"] != const.chain_type_ids["NONPOLYMER"]
    ).float() * (
        feats["res_type"][:, :, const.token_ids["UNK"]]
        + feats["res_type"][:, :, const.token_ids["DN"]]
        + feats["res_type"][:, :, const.token_ids["N"]]
    )
    modified_vs_all_mask = token_mask * modified_residue[:, :, None]
    modified_lddt, modified_total = lddt_dist(
        pred_d, true_d, modified_vs_all_mask, cutoff
    )
    del modified_vs_all_mask
    del modified_residue

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    intra_design_mask = (
        token_mask
        * same_chain_mask
        * (design_mask[:, :, None] * design_mask[:, None, :])
    )
    intra_design_lddt, intra_design_total = lddt_dist(
        pred_d, true_d, intra_design_mask, cutoff
    )
    del intra_design_mask

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    design_design_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (design_mask[:, :, None] * design_mask[:, None, :])
    )
    design_design_lddt, design_design_total = lddt_dist(
        pred_d, true_d, design_design_mask, cutoff
    )
    del design_design_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
        "design_design": design_design_lddt,
        "design_protein": design_protein_lddt,
        "design_ligand": design_ligand_lddt,
        "design_dna": design_dna_lddt,
        "design_rna": design_rna_lddt,
        "intra_design": intra_design_lddt,
        "modified": modified_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
        "design_design": design_design_total,
        "design_protein": design_protein_total,
        "design_ligand": design_ligand_total,
        "design_dna": design_dna_total,
        "design_rna": design_rna_total,
        "intra_design": intra_design_total,
        "modified": modified_total,
    }

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    # Take average across conformers
    for k in total_dict:
        total_dict[k] = total_dict[k].reshape(
            K
        )  # total_dict[k].reshape(B, K).mean(dim=1)

    for k in lddt_dict:
        lddt_dict[k] = lddt_dict[k].reshape(K)  # lddt_dict[k].reshape(B, K).mean(dim=1)

    return lddt_dict, total_dict


def compute_plddt_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_lddt,
    true_coords_resolved_mask,
    token_level_confidence=False,
    multiplicity=1,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        # extract necessary features
        atom_mask = true_coords_resolved_mask
        R_set_to_rep_atom = feats["r_set_to_rep_atom"]
        R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
            token_type == const.chain_type_ids["RNA"]
        ).float()

        B = true_atom_coords.shape[0]

        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"].float()
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

        if token_level_confidence:
            true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords.float())
            pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords.float())

        # compute true lddt
        true_d = torch.cdist(
            true_token_coords if token_level_confidence else true_atom_coords,
            torch.bmm(R_set_to_rep_atom, true_atom_coords.float()),
        )
        pred_d = torch.cdist(
            pred_token_coords if token_level_confidence else pred_atom_coords,
            torch.bmm(R_set_to_rep_atom, pred_atom_coords.float()),
        )

        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )
        pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask.float(), R_set_to_rep_atom)

        if token_level_confidence:
            pair_mask = torch.bmm(token_to_rep_atom, pair_mask.float())
            atom_mask = torch.bmm(
                token_to_rep_atom, atom_mask.unsqueeze(-1).float()
            ).squeeze(-1)
        is_nucleotide_R_element = torch.bmm(
            R_set_to_rep_atom,
            torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1).float()),
        ).squeeze(-1)
        cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
            1, true_d.shape[1], 1
        )

        target_lddt, mask_no_match = lddt_dist(
            pred_d, true_d, pair_mask, cutoff, per_atom=True
        )

        if not token_level_confidence:
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].float().unsqueeze(-1),
                )
                .squeeze(-1)
                .long()
            )
        protein_mask = (
            (
                token_type == const.chain_type_ids["PROTEIN"]
                if token_level_confidence
                else atom_type == const.chain_type_ids["PROTEIN"]
            ).float()
            * atom_mask
            * mask_no_match
        )
        ligand_mask = (
            (
                token_type == const.chain_type_ids["NONPOLYMER"]
                if token_level_confidence
                else atom_type == const.chain_type_ids["NONPOLYMER"]
            ).float()
            * atom_mask
            * mask_no_match
        )
        dna_mask = (
            (
                token_type == const.chain_type_ids["DNA"]
                if token_level_confidence
                else atom_type == const.chain_type_ids["DNA"]
            ).float()
            * atom_mask
            * mask_no_match
        )
        rna_mask = (
            (
                token_type == const.chain_type_ids["RNA"]
                if token_level_confidence
                else atom_type == const.chain_type_ids["RNA"]
            ).float()
            * atom_mask
            * mask_no_match
        )

        protein_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * protein_mask) / (
            torch.sum(protein_mask) + 1e-5
        )
        protein_total = torch.sum(protein_mask)
        ligand_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * ligand_mask) / (
            torch.sum(ligand_mask) + 1e-5
        )
        ligand_total = torch.sum(ligand_mask)
        dna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * dna_mask) / (
            torch.sum(dna_mask) + 1e-5
        )
        dna_total = torch.sum(dna_mask)
        rna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * rna_mask) / (
            torch.sum(rna_mask) + 1e-5
        )
        rna_total = torch.sum(rna_mask)

        mae_plddt_dict = {
            "protein": protein_mae,
            "ligand": ligand_mae,
            "dna": dna_mae,
            "rna": rna_mae,
        }
        total_dict = {
            "protein": protein_total,
            "ligand": ligand_total,
            "dna": dna_total,
            "rna": rna_total,
        }

    return mae_plddt_dict, total_dict


def compute_pde_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pde,
    true_coords_resolved_mask,
    multiplicity=1,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        # extract necessary features
        token_to_rep_atom = feats["token_to_rep_atom"].float()
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

        token_mask = torch.bmm(
            token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
        ).squeeze(-1)

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)

        B = true_atom_coords.shape[0]

        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords.float())
        pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords.float())

        # compute true pde
        true_d = torch.cdist(true_token_coords, true_token_coords)
        pred_d = torch.cdist(pred_token_coords, pred_token_coords)
        target_pde = (
            torch.clamp(
                torch.floor(torch.abs(true_d - pred_d) * 64 / 32).long(), max=63
            ).float()
            * 0.5
            + 0.25
        )

        pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )

        protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
        ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
        dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
        rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

        # compute different pdes
        dna_protein_mask = pair_mask * (
            dna_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * dna_mask[:, None, :]
        )
        dna_protein_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * dna_protein_mask
        ) / (torch.sum(dna_protein_mask) + 1e-5)
        dna_protein_total = torch.sum(dna_protein_mask)
        del dna_protein_mask

        rna_protein_mask = pair_mask * (
            rna_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * rna_mask[:, None, :]
        )
        rna_protein_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * rna_protein_mask
        ) / (torch.sum(rna_protein_mask) + 1e-5)
        rna_protein_total = torch.sum(rna_protein_mask)
        del rna_protein_mask

        ligand_protein_mask = pair_mask * (
            ligand_mask[:, :, None] * protein_mask[:, None, :]
            + protein_mask[:, :, None] * ligand_mask[:, None, :]
        )
        ligand_protein_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * ligand_protein_mask
        ) / (torch.sum(ligand_protein_mask) + 1e-5)
        ligand_protein_total = torch.sum(ligand_protein_mask)
        del ligand_protein_mask

        dna_ligand_mask = pair_mask * (
            dna_mask[:, :, None] * ligand_mask[:, None, :]
            + ligand_mask[:, :, None] * dna_mask[:, None, :]
        )
        dna_ligand_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * dna_ligand_mask
        ) / (torch.sum(dna_ligand_mask) + 1e-5)
        dna_ligand_total = torch.sum(dna_ligand_mask)
        del dna_ligand_mask

        rna_ligand_mask = pair_mask * (
            rna_mask[:, :, None] * ligand_mask[:, None, :]
            + ligand_mask[:, :, None] * rna_mask[:, None, :]
        )
        rna_ligand_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * rna_ligand_mask
        ) / (torch.sum(rna_ligand_mask) + 1e-5)
        rna_ligand_total = torch.sum(rna_ligand_mask)
        del rna_ligand_mask

        intra_ligand_mask = pair_mask * (
            ligand_mask[:, :, None] * ligand_mask[:, None, :]
        )
        intra_ligand_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * intra_ligand_mask
        ) / (torch.sum(intra_ligand_mask) + 1e-5)
        intra_ligand_total = torch.sum(intra_ligand_mask)
        del intra_ligand_mask

        intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
        intra_dna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_dna_mask) / (
            torch.sum(intra_dna_mask) + 1e-5
        )
        intra_dna_total = torch.sum(intra_dna_mask)
        del intra_dna_mask

        intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
        intra_rna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_rna_mask) / (
            torch.sum(intra_rna_mask) + 1e-5
        )
        intra_rna_total = torch.sum(intra_rna_mask)
        del intra_rna_mask

        chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

        intra_protein_mask = (
            pair_mask
            * same_chain_mask
            * (protein_mask[:, :, None] * protein_mask[:, None, :])
        )
        intra_protein_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * intra_protein_mask
        ) / (torch.sum(intra_protein_mask) + 1e-5)
        intra_protein_total = torch.sum(intra_protein_mask)
        del intra_protein_mask

        protein_protein_mask = (
            pair_mask
            * (1 - same_chain_mask)
            * (protein_mask[:, :, None] * protein_mask[:, None, :])
        )
        protein_protein_mae = torch.sum(
            torch.abs(target_pde - pred_pde) * protein_protein_mask
        ) / (torch.sum(protein_protein_mask) + 1e-5)
        protein_protein_total = torch.sum(protein_protein_mask)
        del protein_protein_mask

        mae_pde_dict = {
            "dna_protein": dna_protein_mae,
            "rna_protein": rna_protein_mae,
            "ligand_protein": ligand_protein_mae,
            "dna_ligand": dna_ligand_mae,
            "rna_ligand": rna_ligand_mae,
            "intra_ligand": intra_ligand_mae,
            "intra_dna": intra_dna_mae,
            "intra_rna": intra_rna_mae,
            "intra_protein": intra_protein_mae,
            "protein_protein": protein_protein_mae,
        }
        total_pde_dict = {
            "dna_protein": dna_protein_total,
            "rna_protein": rna_protein_total,
            "ligand_protein": ligand_protein_total,
            "dna_ligand": dna_ligand_total,
            "rna_ligand": rna_ligand_total,
            "intra_ligand": intra_ligand_total,
            "intra_dna": intra_dna_total,
            "intra_rna": intra_rna_total,
            "intra_protein": intra_protein_total,
            "protein_protein": protein_protein_total,
        }

    return mae_pde_dict, total_pde_dict


def compute_pae_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pae,
    true_coords_resolved_mask,
    multiplicity=1,
):
    # Retrieve frames and resolved masks
    frames_idx_original = feats["frames_idx"]
    mask_frame_true = feats["frame_resolved_mask"]

    # Adjust the frames for nonpolymers after symmetry correction!
    # NOTE: frames of polymers do not change under symmetry!
    frames_idx_true, mask_collinear_true = compute_frame_pred(
        true_atom_coords,
        frames_idx_original,
        feats,
        multiplicity,
        resolved_mask=true_coords_resolved_mask,
    )

    frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
        frames_idx_true[:, :, :, 0],
        frames_idx_true[:, :, :, 1],
        frames_idx_true[:, :, :, 2],
    )
    # Compute token coords in true frames
    B, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    true_coords_transformed = express_coordinate_in_frame(
        true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
    )

    # Compute pred frames and mask
    frames_idx_pred, mask_collinear_pred = compute_frame_pred(
        pred_atom_coords, frames_idx_original, feats, multiplicity
    )
    frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c = (
        frames_idx_pred[:, :, :, 0],
        frames_idx_pred[:, :, :, 1],
        frames_idx_pred[:, :, :, 2],
    )
    # Compute token coords in pred frames
    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    pred_coords_transformed = express_coordinate_in_frame(
        pred_atom_coords, frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c
    )

    target_pae_continuous = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )
    target_pae = (
        torch.clamp(torch.floor(target_pae_continuous * 64 / 32).long(), max=63).float()
        * 0.5
        + 0.25
    )

    # Compute mask for the pae loss
    b_true_resolved_mask = true_coords_resolved_mask[
        torch.arange(B // multiplicity)[:, None, None].to(
            pred_coords_transformed.device
        ),
        frame_true_atom_b,
    ]

    pair_mask = (
        mask_frame_true[:, None, :, None]  # if true frame is invalid
        * mask_collinear_true[:, :, :, None]  # if true frame is invalid
        * mask_collinear_pred[:, :, :, None]  # if pred frame is invalid
        * b_true_resolved_mask[:, :, None, :]  # If atom j is not resolved
        * feats["token_pad_mask"][:, None, :, None]
        * feats["token_pad_mask"][:, None, None, :]
    )

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different paes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)
    del dna_protein_mask

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)
    del rna_protein_mask

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)
    del ligand_protein_mask

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)
    del dna_ligand_mask

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)
    del rna_ligand_mask

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)
    del intra_ligand_mask

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)
    del intra_dna_mask

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)
    del intra_rna_mask

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)
    del intra_protein_mask

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)
    del protein_protein_mask

    mae_pae_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pae_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pae_dict, total_pae_dict


def align_to_reference(
    ensemble_atom_coords,
    reference_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    representative_atoms=False,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        ensemble_atom_coords_ref = reference_atom_coords.unsqueeze(0).repeat_interleave(
            multiplicity, 0
        )

        if representative_atoms:
            atom_mask = feats["token_to_rep_atom"].sum(dim=1)
        else:
            atom_mask = feats["atom_resolved_mask"]

        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        L = atom_mask.shape[1]
        align_weights = torch.ones(multiplicity, L).to(atom_mask.device)
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_weight
            * (
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
            )
            + ligand_weight
            * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
        )

        # Align the ensemble coordinates to the the reference frame
        with torch.no_grad():
            ensemble_atom_coords_aligned = weighted_rigid_align(
                ensemble_atom_coords,
                ensemble_atom_coords_ref,
                align_weights,
                mask=atom_mask,
            )

    return (
        ensemble_atom_coords_ref,
        ensemble_atom_coords_aligned,
        align_weights,
        atom_mask,
    )


def get_rmsd_diversity(
    ensemble_atom_coords,
    reference_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    representative_atoms=False,
):
    # Align the ensemble coordinates to the the reference frame
    ensemble_atom_coords_ref, ensemble_atom_coords_aligned, align_weights, atom_mask = (
        align_to_reference(
            ensemble_atom_coords,
            reference_atom_coords,
            feats,
            multiplicity=multiplicity,
            nucleotide_weight=nucleotide_weight,
            ligand_weight=ligand_weight,
            representative_atoms=representative_atoms,
        )
    )

    mse_loss = ((ensemble_atom_coords_ref - ensemble_atom_coords_aligned) ** 2).sum(
        dim=-1
    )
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )

    return rmsd


def get_rmsf_score(
    ensemble_atom_coords_pred,
    reference_atom_coords_pred,
    ensemble_atom_coords_gt,
    reference_atom_coords_gt,
    feats,
    multiplicity_pred=1,
    multiplicity_gt=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    representative_atoms=False,
):
    ## Compute RMSF for pred ##

    # Align the ensemble coordinates to the the reference frame
    _, ensemble_atom_coords_aligned_pred, _, _ = align_to_reference(
        ensemble_atom_coords_pred,
        reference_atom_coords_pred,
        feats,
        multiplicity=multiplicity_pred,
        nucleotide_weight=nucleotide_weight,
        ligand_weight=ligand_weight,
        representative_atoms=representative_atoms,
    )

    # Take the mean of the ensemble coordinates per atom
    ensemble_atom_coords_aligned_mean_pred = ensemble_atom_coords_aligned_pred.mean(
        dim=0
    )[None]  # (S, L, 3)

    diff_pred = (
        ensemble_atom_coords_aligned_pred - ensemble_atom_coords_aligned_mean_pred
    ) ** 2
    rmsf_pred = torch.sqrt(diff_pred.sum(-1).sum(0) / multiplicity_pred)  # (L)

    ## Compute RMSF for gt ##

    # Align the ensemble coordinates to the the reference frame
    _, ensemble_atom_coords_aligned_gt, _, atom_mask = align_to_reference(
        ensemble_atom_coords_gt,
        reference_atom_coords_gt,
        feats,
        multiplicity=multiplicity_gt,
        nucleotide_weight=nucleotide_weight,
        ligand_weight=ligand_weight,
        representative_atoms=representative_atoms,
    )

    # Take the mean of the ensemble coordinates per atom
    ensemble_atom_coords_aligned_mean_gt = ensemble_atom_coords_aligned_gt.mean(dim=0)[
        None
    ]  # (S, L, 3)

    ## Compute RMSF score ##
    atom_mask = atom_mask[0]  # (multiplicity, L) - > (L)

    diff_gt = (
        ensemble_atom_coords_aligned_gt - ensemble_atom_coords_aligned_mean_gt
    ) ** 2
    rmsf_gt = torch.sqrt(diff_gt.sum(-1).sum(0) / multiplicity_gt)  # (L)

    rmsf_score = torch.sqrt(
        (((rmsf_pred - rmsf_gt) ** 2) * atom_mask).sum(0) / torch.sum(atom_mask, dim=-1)
    )

    return rmsf_score


def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    representative_atoms=False,
    protein_lig_rmsd=False,
):
    with torch.autocast(get_autocast_device_type(), enabled=False):
        atom_coords = feats["coords"]
        B, K = atom_coords.shape[0:2]
        assert B == 1, "Validation is not supported for batch size > 1"
        atom_coords = atom_coords.squeeze(0).repeat((multiplicity, 1, 1))
        pred_atom_coords = pred_atom_coords.repeat_interleave(
            K, 0
        )  # (B=1 * multiplicity, L, 3) -> (multiplicity * K, L, 3)

        if representative_atoms:
            atom_mask = feats["token_to_rep_atom"].sum(dim=1)
        else:
            atom_mask = feats["atom_resolved_mask"]

        atom_mask = atom_mask.repeat_interleave(K * multiplicity, 0)

        align_weights = atom_coords.new_ones(atom_coords.shape[:2])
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(K * multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_weight
            * (
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
            )
            + ligand_weight
            * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad():
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords, pred_atom_coords, align_weights, mask=atom_mask
            )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    # Take best rmsd across diffusion samples (multiplicity) and average across
    # conformers in ensemble
    best_rmsd_recall = torch.min(rmsd.reshape(multiplicity, K), dim=0).values
    best_rmsd_recall = torch.mean(best_rmsd_recall, dim=0)

    # Take best rmsd across conformers in ensemble and average across
    # samples (multiplicity)
    best_rmsd_precision = torch.min(rmsd.reshape(multiplicity, K), dim=1).values
    best_rmsd_precision = torch.mean(best_rmsd_precision, dim=0)

    if not protein_lig_rmsd:
        return rmsd, best_rmsd

    # designed and target mask
    design_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    design_mask = design_mask.repeat_interleave(multiplicity, 0)
    design_chain_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["chain_design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    design_chain_mask = design_chain_mask.repeat_interleave(multiplicity, 0)
    target_mask = 1.0 - design_chain_mask

    try:
        # Rmsd of the designed part after aligning the designed part
        rmsd_design, best_rmsd_design = compute_subset_rmsd(
            atom_coords,
            pred_atom_coords,
            atom_mask,
            align_weights,
            design_mask,
            multiplicity,
        )
    except Exception as e:
        print(f"Warning: rmsd_design failed with error: {e}")
        rmsd_design, best_rmsd_design = torch.tensor(torch.nan), torch.tensor(torch.nan)

    try:
        # Rmsd of the target part after aligning the target part
        rmsd_target, best_rmsd_target = compute_subset_rmsd(
            atom_coords,
            pred_atom_coords,
            atom_mask,
            align_weights,
            target_mask,
            multiplicity,
        )
    except Exception as e:
        print(f"Warning: rmsd_target failed with error: {e}")
        rmsd_target, best_rmsd_target = torch.tensor(torch.nan), torch.tensor(torch.nan)

    rmsd_design_target, best_rmsd_design_target = (
        torch.tensor(torch.nan),
        torch.tensor(torch.nan),
    )
    target_aligned_rmsd_design = torch.tensor(torch.nan)
    best_target_aligned_rmsd_design = torch.tensor(torch.nan)
    if (
        torch.any(
            feats["mol_type"][~feats["chain_design_mask"]]
            != const.chain_type_ids["NONPOLYMER"]
        )
        or (~feats["chain_design_mask"]).float().sum() > 3
    ):
        try:
            # align the whole complex and only compute RMSD for the designed part
            rmsd_design_target, best_rmsd_design_target = compute_subset_rmsd(
                atom_coords,
                pred_atom_coords,
                atom_mask,
                align_weights,
                atom_mask,
                multiplicity,
                rmsd_mask=design_mask,
            )
        except Exception as e:
            print(f"Warning: rmsd_design_target failed with error: {e}")
        try:
            target_aligned_rmsd_design, best_target_aligned_rmsd_design = (
                compute_subset_rmsd(
                    atom_coords,
                    pred_atom_coords,
                    atom_mask,
                    target_mask,
                    atom_mask,
                    multiplicity,
                    rmsd_mask=design_mask,
                )
            )
        except Exception as e:
            print(f"Warning: target_aligned_rmsd_design failed with error: {e}")

    return (
        rmsd,
        best_rmsd,
        rmsd_design,
        best_rmsd_design,
        rmsd_target,
        best_rmsd_target,
        rmsd_design_target,
        best_rmsd_design_target,
        target_aligned_rmsd_design,
        best_target_aligned_rmsd_design,
    )


def compute_subset_rmsd(
    atom_coords,
    pred_atom_coords,
    atom_mask,
    align_weights,
    subset_mask,
    multiplicity,
    rmsd_mask=None,
):
    used_mask = atom_mask * subset_mask
    used_weights = align_weights * subset_mask

    if used_mask.sum() == 0:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)

    with torch.no_grad():
        aligned_coords = weighted_rigid_align(
            atom_coords, pred_atom_coords, used_weights, mask=used_mask
        )

    mse = ((pred_atom_coords - aligned_coords) ** 2).sum(dim=-1)
    rmsd_mask = used_weights * used_mask if rmsd_mask is None else rmsd_mask * atom_mask
    rmsd = torch.sqrt(
        torch.sum(mse * rmsd_mask, dim=-1)
        / torch.sum(rmsd_mask, dim=-1).clamp_min(1e-7)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values
    return rmsd, best_rmsd


def weighted_minimum_rmsd_single(
    pred_atom_coords,
    atom_coords,
    atom_mask,
    atom_to_token,
    mol_type,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    return rmsd, atom_coords_aligned_ground_truth, align_weights
