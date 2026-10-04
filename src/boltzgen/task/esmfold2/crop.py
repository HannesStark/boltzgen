"""Slice ESMFold2 structural features while preserving source-chain identities."""

from dataclasses import replace

import torch

TOKEN_KEYS = {
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "res_type",
    "input_ids",
    "token_attention_mask",
    "pocket_feature",
    "deletion_mean",
    "distogram_atom_idx",
    "frames_idx",
}
PAIR_KEYS = {"token_bonds", "disto_cond", "disto_cond_mask"}
MSA_KEYS = {"msa", "has_deletion", "deletion_value", "msa_attention_mask"}
ATOM_KEYS = {
    "ref_pos",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_attention_mask",
    "atom_to_token",
    "is_resolved",
}


def crop_features(
    features: dict, chain_infos: list, chains: list[dict]
) -> tuple[dict, list, torch.Tensor, dict]:
    """Select whole residues, remapping atom/token references and atom padding.

    One residue may have multiple tokens. The inference tensors retain original
    residue/asym/entity/sym indices; only dense storage indices are reset.
    """
    expected = TOKEN_KEYS | PAIR_KEYS | MSA_KEYS | ATOM_KEYS | {"gt_coords"}
    if set(features) != expected:
        raise ValueError(f"ESMFold2 feature schema changed: {set(features) ^ expected}")
    by_id = {c["id"]: c for c in chains}
    if set(by_id) != {c.chain_id for c in chain_infos}:
        raise ValueError("ESMFold2 changed input chain identities")
    names = features["ref_atom_name_chars"][0].tolist()
    omitted = set()
    for info in chain_infos:
        atom_lookup = {
            (token.residue_index, "".join(chr(n + 32) for n in names[i] if n)): i
            for token in info.tokens
            for i in range(token.atom_start, token.atom_start + token.atom_count)
        }
        for position, name in by_id[info.chain_id].get("omitted_atoms", []):
            if "smiles" in by_id[info.chain_id]:
                name = name.upper()
            if position not in by_id[info.chain_id]["indices"]:
                raise ValueError("An omitted atom is outside the structural crop")
            if (position, name) not in atom_lookup:
                # Native ESM preparation already removes CCD leaving atoms on
                # covalently bonded ligands. An explicit removal is then done.
                if info.mol_type == 3 and "smiles" not in by_id[info.chain_id]:
                    from esm.models.esmfold2.conformers import get_ccd_leaving_atoms

                    residue = by_id[info.chain_id]["residue_names"][position]
                    if name in get_ccd_leaving_atoms(residue):
                        continue
                raise ValueError(
                    f"Cannot resolve omitted atom {info.chain_id}:{position + 1}:{name}"
                )
            omitted.add(atom_lookup[position, name])
    selected = []
    for info in chain_infos:
        wanted = by_id[info.chain_id]["indices"]
        if wanted != sorted(set(wanted)) or not wanted:
            raise ValueError("Crop positions must be nonempty, unique, and increasing")
        available = {t.residue_index for t in info.tokens}
        if not set(wanted) <= available:
            raise ValueError(f"Crop exceeds prepared sequence for {info.chain_id}")
        selected.extend(
            t.token_index
            for t in info.tokens
            if t.residue_index in set(wanted)
            and any(
                i not in omitted
                for i in range(t.atom_start, t.atom_start + t.atom_count)
            )
        )
    selected.sort()
    device = features["token_index"].device
    tokens = torch.tensor(selected, device=device, dtype=torch.long)
    full_tokens = features["token_index"].shape[1]
    token_map = torch.full((full_tokens,), -1, device=device, dtype=torch.long)
    token_map[tokens] = torch.arange(len(tokens), device=device)
    atom_to_token = features["atom_to_token"][0].long()
    keep_atoms = features["atom_attention_mask"][0].bool() & (
        token_map[atom_to_token] >= 0
    )
    if omitted:
        keep_atoms[list(omitted)] = False
    atoms = torch.where(keep_atoms)[0]
    atom_map = torch.full_like(atom_to_token, -1)
    atom_map[atoms] = torch.arange(len(atoms), device=device)
    padded_atoms = ((len(atoms) + 31) // 32) * 32

    cropped = {key: features[key].index_select(1, tokens) for key in TOKEN_KEYS}
    cropped.update(
        {
            key: features[key].index_select(1, tokens).index_select(2, tokens)
            for key in PAIR_KEYS
        }
    )
    cropped.update({key: features[key].index_select(2, tokens) for key in MSA_KEYS})
    for key in ATOM_KEYS | {"gt_coords"}:
        axis = 2 if key == "gt_coords" else 1
        value = features[key].index_select(axis, atoms)
        shape = list(value.shape)
        shape[axis] = padded_atoms - len(atoms)
        cropped[key] = torch.cat((value, value.new_zeros(shape)), dim=axis)
    cropped["token_index"] = torch.arange(len(tokens), device=device)[None]
    cropped["atom_to_token"][0, : len(atoms)] = token_map[atom_to_token[atoms]]
    for key in ("distogram_atom_idx", "frames_idx"):
        indices = atom_map[cropped[key].long()]
        if (indices < 0).any():
            raise ValueError(f"Crop removed an atom required by {key}")
        cropped[key] = indices
    # prepare_request verifies the native query row against the input sequence.
    # Modified protein residues carry their parent amino acid in that row while
    # their atom-tokenized structural res_type is UNK, so those need not match.
    if features["msa"].shape != (1, 1, full_tokens) or cropped["msa"].shape != (
        1,
        1,
        len(tokens),
    ):
        raise ValueError("ESMFold2 scoring requires exactly the native query MSA row")
    if any(
        cropped[k].count_nonzero()
        for k in (
            "has_deletion",
            "deletion_value",
            "deletion_mean",
            "is_resolved",
            "gt_coords",
            "disto_cond_mask",
        )
    ):
        raise ValueError("Unexpected MSA deletion or coordinate conditioning")

    infos = []
    for info in chain_infos:
        retained = []
        for token in info.tokens:
            if token_map[token.token_index] >= 0:
                kept_atoms = [
                    i
                    for i in range(
                        token.atom_start, token.atom_start + token.atom_count
                    )
                    if keep_atoms[i]
                ]
                retained.append(
                    replace(
                        token,
                        token_index=int(token_map[token.token_index]),
                        atom_start=int(atom_map[kept_atoms[0]]),
                        atom_count=len(kept_atoms),
                    )
                )
        if retained:
            infos.append(replace(info, tokens=retained))
    audit = {
        "full_tokens": full_tokens,
        "crop_tokens": len(tokens),
        "crop_atoms": len(atoms),
        "crop_padded_atoms": padded_atoms,
        "selected_global_token_indices": selected,
        "residue_indices": cropped["residue_index"][0].tolist(),
        "asym_ids": cropped["asym_id"][0].tolist(),
        "msa_depth": 1,
        "coordinate_conditioning": False,
        "explicitly_omitted_atoms": len(omitted),
    }
    return cropped, infos, tokens, audit


def polymer_representatives(features: dict, infos: list) -> dict[str, list[int]]:
    """Use CA/C1' token representatives for atom-tokenized modified residues."""
    names = features["ref_atom_name_chars"][0].cpu().tolist()
    result = {}
    for chain in infos:
        if chain.mol_type == 3:
            continue
        residues = {}
        for token in chain.tokens:
            residues.setdefault(token.residue_index, []).append(token)
        indices = []
        for tokens in residues.values():
            representative = "CA" if chain.mol_type == 0 else "C1'"
            matches = [
                t.token_index
                for t in tokens
                if any(
                    "".join(chr(n + 32) for n in names[i] if n) == representative
                    for i in range(t.atom_start, t.atom_start + t.atom_count)
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Modified residue in {chain.chain_id} lacks a unique {representative} token"
                )
            indices.append(matches[0])
        result[chain.chain_id] = indices
    return result
