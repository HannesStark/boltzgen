import copy
import math
import re
from typing import Dict, List, Optional, Tuple
import numba
import numpy as np
import numpy.typing as npt
import pydssp
import torch
from numba import types
from rdkit.Chem import Mol
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot as one_hot_torch

from boltzgen.data import const
from boltzgen.data.data import (
    MSA,
    Atom,
    Input,
    MSADeletion,
    MSAResidue,
    MSASequence,
    convert_atom_name,
    convert_ccd,
    elem_from_name,
)
from boltzgen.data.mol import (
    get_amino_acids_symmetries,
    get_chain_symmetries,
    get_ligand_symmetries,
    get_symmetries,
)
from boltzgen.data.pad import pad_dim
from boltzgen.model.modules.utils import center_random_augmentation

####################################################################################################
# CONSTANTS
####################################################################################################

possible_fake_atoms = []
for i in range(37):
    possible_fake_atoms.append(
        (
            const.fake_element + str(i),
            [0.0, 0.0, 0.0],
            True,
            0,
            0,
        )
    )
possible_fake_atoms = np.array(possible_fake_atoms, dtype=Atom)

####################################################################################################
# HELPERS
####################################################################################################


def one_hot_bool(tensor, num_classes=-1):
    """
    Efficient one-hot encoding that returns bool tensor instead of long tensor.

    Args:
        tensor: Input tensor of indices
        num_classes: Number of classes. If -1, inferred from tensor.max() + 1

    Returns:
        Boolean tensor with one-hot encoding
    """
    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1

    # Create output tensor with bool dtype for memory efficiency
    output_shape = list(tensor.shape) + [num_classes]
    result = torch.zeros(output_shape, dtype=torch.bool, device=tensor.device)

    # Use scatter_ for efficient assignment
    # scatter_(dim, index, src) - dim is the last dimension
    result.scatter_(-1, tensor.unsqueeze(-1), True)

    return result


def sample_d(
    min_d: float,
    max_d: float,
    n_samples: int,
    random: np.random.Generator,
) -> np.ndarray:
    """Generate samples from a 1/d distribution between min_d and max_d.

    Parameters
    ----------
    min_d : float
        Minimum value of d
    max_d : float
        Maximum value of d
    n_samples : int
        Number of samples to generate
    random : numpy.random.Generator
        Random number generator

    Returns
    -------
    numpy.ndarray
        Array of samples drawn from the distribution

    Notes
    -----
    The probability density function is:
    f(d) = 1/(d * ln(max_d/min_d)) for d in [min_d, max_d]

    The inverse CDF transform is:
    d = min_d * (max_d/min_d)**u where u ~ Uniform(0,1)

    """
    # Generate n_samples uniform random numbers in [0, 1]
    u = random.random(n_samples)
    # Transform u using the inverse CDF
    return min_d * (max_d / min_d) ** u


def compute_frames_nonpolymer(
    data: Input,
    coords,
    resolved_mask,
    atom_to_token,
    frame_data: List,
    resolved_frame_data: List,
) -> Tuple[List, List]:
    """Get the frames for non-polymer tokens.

    Parameters
    ----------
    data : Input
        The input data to the model.
    frame_data : List
        The frame data.
    resolved_frame_data : List
        The resolved frame data.

    Returns
    -------
    Tuple[List, List]
        The frame data and resolved frame data.

    """

    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0
    for id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == id
        mask_chain_atom = asym_id_atom == id
        num_tokens = mask_chain_token.sum()
        num_atoms = mask_chain_atom.sum()
        if (
            data.tokens[token_idx]["mol_type"] != const.chain_type_ids["NONPOLYMER"]
            or num_atoms < 3  # noqa: PLR2004
        ):
            token_idx += num_tokens
            atom_idx += num_atoms
            continue
        dist_mat = (
            (
                coords.reshape(-1, 3)[mask_chain_atom][:, None, :]
                - coords.reshape(-1, 3)[mask_chain_atom][None, :, :]
            )
            ** 2
        ).sum(-1) ** 0.5
        resolved_pair = 1 - (
            resolved_mask[mask_chain_atom][None, :]
            * resolved_mask[mask_chain_atom][:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )
        frame_data[token_idx : token_idx + num_atoms, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms] = resolved_mask[
            frames
        ].all(axis=1)
        token_idx += num_tokens
        atom_idx += num_atoms
    frames_expanded = coords.reshape(-1, 3)[frame_data]

    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    return frame_data, resolved_frame_data & mask_collinear


def compute_collinear_mask(v1, v2):
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a dummy MSA for a chain.

    Parameters
    ----------
    residues : np.ndarray
        The residues for the chain.

    Returns
    -------
    MSA
        The dummy MSA.

    """
    residues = [res["res_type"] for res in residues]
    deletions = []
    sequences = [(0, -1, 0, len(residues), 0, 0)]
    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def construct_paired_msa(  # noqa: C901, PLR0915, PLR0912
    data: Input,
    random: np.random.Generator,
    max_seqs: int,
    max_pairs: int = 8192,
    max_total: int = 16384,
    random_subset: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pair the MSA data.

    Parameters
    ----------
    data : Input
        The input data to the model.

    Returns
    -------
    Tensor
        The MSA data.
    Tensor
        The deletion data.
    Tensor
        Mask indicating paired sequences.

    """
    # Get unique chains (ensuring monotonicity in the order)
    assert np.all(np.diff(data.tokens["asym_id"], n=1) >= 0)
    chain_ids = np.unique(data.tokens["asym_id"])

    # Get relevant MSA, and create a dummy for chains without
    msa: Dict[int, MSA] = {}
    for chain_id in chain_ids:
        # Get input sequence
        chain = data.structure.chains[chain_id]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        residues = data.structure.residues[res_start:res_end]

        # Check if we have an MSA, and that the
        # first sequence matches the input sequence
        if chain_id in data.msa:
            # Set the MSA
            msa[chain_id] = data.msa[chain_id]

            # Run length and residue type checks
            first = data.msa[chain_id].sequences[0]
            first_start = first["res_start"]
            first_end = first["res_end"]
            msa_residues = data.msa[chain_id].residues
            first_residues = msa_residues[first_start:first_end]

            warning = "Warning: MSA does not match input sequence, creating dummy."
            if len(residues) == len(first_residues):
                # If there is a mismatch, check if it is between MET & UNK
                # If so, replace the first sequence with the input sequence.
                # Otherwise, replace with a dummy MSA for this chain.
                mismatches = residues["res_type"] != first_residues["res_type"]
                if mismatches.sum().item():
                    idx = np.where(mismatches)[0]
                    is_met = residues["res_type"][idx] == const.token_ids["MET"]
                    is_unk = residues["res_type"][idx] == const.token_ids["UNK"]
                    is_msa_unk = (
                        first_residues["res_type"][idx] == const.token_ids["UNK"]
                    )
                    if (np.all(is_met) and np.all(is_msa_unk)) or np.all(is_unk):
                        msa_residues[first_start:first_end]["res_type"] = residues[
                            "res_type"
                        ]
                    else:
                        print(
                            warning,
                            "1",
                            residues["res_type"],
                            first_residues["res_type"],
                            data.record.id,
                        )
                        msa[chain_id] = dummy_msa(residues)
            else:
                print(
                    warning,
                    "2",
                    residues["res_type"],
                    first_residues["res_type"],
                    data.record.id,
                )
                msa[chain_id] = dummy_msa(residues)
        else:
            msa[chain_id] = dummy_msa(residues)

    # Map taxonomies to (chain_id, seq_idx)
    taxonomy_map: Dict[str, List] = {}
    for chain_id, chain_msa in msa.items():
        sequences = chain_msa.sequences
        sequences = sequences[sequences["taxonomy"] != -1]
        for sequence in sequences:
            seq_idx = sequence["seq_idx"]
            taxon = sequence["taxonomy"]
            taxonomy_map.setdefault(taxon, []).append((chain_id, seq_idx))

    # Remove taxonomies with only one sequence and sort by the
    # number of chain_id present in each of the taxonomies
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    taxonomy_map = sorted(
        taxonomy_map.items(),
        key=lambda x: len({c for c, _ in x[1]}),
        reverse=True,
    )

    # Keep track of the sequences available per chain, keeping the original
    # order of the sequences in the MSA to favor the best matching sequences
    visited = {(c, s) for c, items in taxonomy_map for s in items}
    available = {}
    for c in chain_ids:
        available[c] = [
            i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
        ]

    # Create sequence pairs
    is_paired = []
    pairing = []

    # Start with the first sequence for each chain
    is_paired.append({c: 1 for c in chain_ids})
    pairing.append({c: 0 for c in chain_ids})

    # Then add up to 8191 paired rows
    for _, pairs in taxonomy_map:
        # Group occurences by chain_id in case we have multiple
        # sequences from the same chain and same taxonomy
        chain_occurences = {}
        for chain_id, seq_idx in pairs:
            chain_occurences.setdefault(chain_id, []).append(seq_idx)

        # We create as many pairings as the maximum number of occurences
        max_occurences = max(len(v) for v in chain_occurences.values())
        for i in range(max_occurences):
            row_pairing = {}
            row_is_paired = {}

            # Add the chains present in the taxonomy
            for chain_id, seq_idxs in chain_occurences.items():
                # Roll over the sequence index to maximize diversity
                idx = i % len(seq_idxs)
                seq_idx = seq_idxs[idx]

                # Add the sequence to the pairing
                row_pairing[chain_id] = seq_idx
                row_is_paired[chain_id] = 1

            # Add any missing chains
            for chain_id in chain_ids:
                if chain_id not in row_pairing:
                    row_is_paired[chain_id] = 0
                    if available[chain_id]:
                        # Add the next available sequence
                        seq_idx = available[chain_id].pop(0)
                        row_pairing[chain_id] = seq_idx
                    else:
                        # No more sequences available, we place a gap
                        row_pairing[chain_id] = -1

            pairing.append(row_pairing)
            is_paired.append(row_is_paired)

            # Break if we have enough pairs
            if len(pairing) >= max_pairs:
                break

        # Break if we have enough pairs
        if len(pairing) >= max_pairs:
            break

    # Now add up to 16384 unpaired rows total
    max_left = max(len(v) for v in available.values())
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for chain_id in chain_ids:
            row_is_paired[chain_id] = 0
            if available[chain_id]:
                # Add the next available sequence
                seq_idx = available[chain_id].pop(0)
                row_pairing[chain_id] = seq_idx
            else:
                # No more sequences available, we place a gap
                row_pairing[chain_id] = -1

        pairing.append(row_pairing)
        is_paired.append(row_is_paired)

        # Break if we have enough sequences
        if len(pairing) >= max_total:
            break

    # Randomly sample a subset of the pairs
    # ensuring the first row is always present
    if random_subset:
        num_seqs = len(pairing)
        if num_seqs > max_seqs:
            indices = random.choice(
                np.arange(1, num_seqs), size=max_seqs - 1, replace=False
            )  # noqa: NPY002
            pairing = [pairing[0]] + [pairing[i] for i in indices]
            is_paired = [is_paired[0]] + [is_paired[i] for i in indices]
    else:
        # Deterministic downsample to max_seqs
        pairing = pairing[:max_seqs]
        is_paired = is_paired[:max_seqs]

    # Map (chain_id, seq_idx, res_idx) to deletion
    deletions = {}
    for chain_id, chain_msa in msa.items():
        chain_deletions = chain_msa.deletions
        for sequence in chain_msa.sequences:
            del_start = sequence["del_start"]
            del_end = sequence["del_end"]
            chain_deletions = chain_msa.deletions[del_start:del_end]
            for deletion_data in chain_deletions:
                seq_idx = sequence["seq_idx"]
                res_idx = deletion_data["res_idx"]
                deletion = deletion_data["deletion"]
                deletions[(chain_id, seq_idx, res_idx)] = deletion

    # Add all the token MSA data
    msa_data, del_data, paired_data = prepare_msa_arrays(
        data.tokens, pairing, is_paired, deletions, msa
    )
    msa_data = torch.tensor(msa_data, dtype=torch.long)
    del_data = torch.tensor(del_data, dtype=torch.float)
    paired_data = torch.tensor(paired_data, dtype=torch.float)
    return msa_data, del_data, paired_data


def prepare_msa_arrays(
    tokens,
    pairing: list[dict[int, int]],
    is_paired: list[dict[int, int]],
    deletions: dict[tuple[int, int, int], int],
    msa: dict[int, MSA],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Reshape data to play nicely with numba jit."""
    token_asym_ids_arr = np.array([t["asym_id"] for t in tokens], dtype=np.int64)
    token_res_idxs_arr = np.array([t["res_idx"] for t in tokens], dtype=np.int64)

    chain_ids = sorted(msa.keys())

    # chain_ids are not necessarily contiguous (e.g. they might be 0, 24, 25).
    # This allows us to look up a chain_id by it's index in the chain_ids list.
    chain_id_to_idx = {chain_id: i for i, chain_id in enumerate(chain_ids)}
    token_asym_ids_idx_arr = np.array(
        [chain_id_to_idx[asym_id] for asym_id in token_asym_ids_arr], dtype=np.int64
    )

    pairing_arr = np.zeros((len(pairing), len(chain_ids)), dtype=np.int64)
    is_paired_arr = np.zeros((len(is_paired), len(chain_ids)), dtype=np.int64)

    for i, row_pairing in enumerate(pairing):
        for chain_id in chain_ids:
            pairing_arr[i, chain_id_to_idx[chain_id]] = row_pairing[chain_id]

    for i, row_is_paired in enumerate(is_paired):
        for chain_id in chain_ids:
            is_paired_arr[i, chain_id_to_idx[chain_id]] = row_is_paired[chain_id]

    max_seq_len = max(len(msa[chain_id].sequences) for chain_id in chain_ids)

    # we want res_start from sequences
    msa_sequences = np.full((len(chain_ids), max_seq_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        for i, seq in enumerate(msa[chain_id].sequences):
            msa_sequences[chain_id_to_idx[chain_id], i] = seq["res_start"]

    max_residues_len = max(len(msa[chain_id].residues) for chain_id in chain_ids)
    msa_residues = np.full((len(chain_ids), max_residues_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        residues = msa[chain_id].residues.astype(np.int64)
        idxs = np.arange(len(residues))
        chain_idx = chain_id_to_idx[chain_id]
        msa_residues[chain_idx, idxs] = residues

    deletions_dict = numba.typed.Dict.empty(
        key_type=numba.types.Tuple(
            [numba.types.int64, numba.types.int64, numba.types.int64]
        ),
        value_type=numba.types.int64,
    )
    deletions_dict.update(deletions)

    return _prepare_msa_arrays_inner(
        token_asym_ids_arr,
        token_res_idxs_arr,
        token_asym_ids_idx_arr,
        pairing_arr,
        is_paired_arr,
        deletions_dict,
        msa_sequences,
        msa_residues,
        const.token_ids["-"],
        const.token_ids["UNK"],
    )


deletions_dict_type = types.DictType(types.UniTuple(types.int64, 3), types.int64)


@numba.njit(
    [
        types.Tuple(
            (
                types.int64[:, ::1],  # msa_data
                types.int64[:, ::1],  # del_data
                types.int64[:, ::1],  # paired_data
            )
        )(
            types.int64[::1],  # token_asym_ids
            types.int64[::1],  # token_res_idxs
            types.int64[::1],  # token_asym_ids_idx
            types.int64[:, ::1],  # pairing
            types.int64[:, ::1],  # is_paired
            deletions_dict_type,  # deletions
            types.int64[:, ::1],  # msa_sequences
            types.int64[:, ::1],  # msa_residues
            types.int64,  # gap_token
            types.int64,  # unk_token
        )
    ],
    cache=True,
)
def _prepare_msa_arrays_inner(
    token_asym_ids: npt.NDArray[np.int64],
    token_res_idxs: npt.NDArray[np.int64],
    token_asym_ids_idx: npt.NDArray[np.int64],
    pairing: npt.NDArray[np.int64],
    is_paired: npt.NDArray[np.int64],
    deletions: dict[tuple[int, int, int], int],
    msa_sequences: npt.NDArray[np.int64],
    msa_residues: npt.NDArray[np.int64],
    gap_token: int,
    unk_token: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # If there is an index out of bounds error when indexing into any of the arrays in here, then it does not throw an error but instead just places a random seeming high value into the array.
    n_tokens = len(token_asym_ids)
    n_pairs = len(pairing)
    msa_data = np.full((n_tokens, n_pairs), gap_token, dtype=np.int64)
    paired_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)
    del_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)

    # Add all the token MSA data
    res_idx_contiguous = 0
    chain_id_tracker = token_asym_ids[0]
    res_idx_tracker = token_res_idxs[0]
    for token_idx in range(n_tokens):
        chain_id_idx = token_asym_ids_idx[token_idx]
        chain_id = token_asym_ids[token_idx]
        res_idx = token_res_idxs[token_idx]

        if res_idx != res_idx_tracker:
            res_idx_contiguous += 1
        res_idx_tracker = res_idx
        if chain_id != chain_id_tracker:
            res_idx_contiguous = 0
        chain_id_tracker = chain_id

        for pair_idx in range(n_pairs):
            seq_idx = pairing[pair_idx, chain_id_idx]
            paired_data[token_idx, pair_idx] = is_paired[pair_idx, chain_id_idx]

            # Add residue type
            if seq_idx != -1:
                res_start = msa_sequences[chain_id_idx, seq_idx]
                # TODO this only works for MSAs that are cropped
                seq = msa_sequences[chain_id_idx]
                res = msa_residues[chain_id_idx]
                if (
                    seq[0] == 0
                    and np.all(seq[1:] == -1)
                    and not np.all((res == -1) | (res == unk_token))
                ):
                    # if a chain has no MSA, and is not a chain of small molecules, we use the contiguous residue index
                    res_type = msa_residues[
                        chain_id_idx, res_start + res_idx_contiguous
                    ]
                else:
                    # if a chain has an MSA, or is a chain of small molecules, we use the residue index
                    res_type = msa_residues[chain_id_idx, res_start + res_idx]
                k = (chain_id, seq_idx, res_idx)
                if k in deletions:
                    del_data[token_idx, pair_idx] = deletions[k]
                msa_data[token_idx, pair_idx] = res_type

    return msa_data, del_data, paired_data


####################################################################################################
# FEATURES
####################################################################################################


def select_subset_from_mask(mask, p, random: np.random.Generator) -> np.ndarray:
    num_true = np.sum(mask)
    v = random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def get_range_bin(value: float, range_dict: Dict[Tuple[float, float], int], default=0):
    """Get the bin of a value given a range dictionary."""
    value = float(value)
    for k, idx in range_dict.items():
        if k == "other":
            continue
        low, high = k
        if low <= value < high:
            return idx
    return default


def process_token_features(  # noqa: C901, PLR0915, PLR0912
    data: Input,
    random: np.random.Generator,
    max_tokens: Optional[int] = None,
    override_method: Optional[str] = None,
    disulfide_prob: Optional[float] = 1.0,
    disulfide_on: Optional[bool] = False,
) -> Dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Input
        The input data to the model.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    Dict[str, Tensor]
        The token features.

    """
    # Token data
    one_hot = one_hot_torch
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"]).long()
    asym_id = from_numpy(token_data["asym_id"]).long()
    entity_id = from_numpy(token_data["entity_id"]).long()
    sym_id = from_numpy(token_data["sym_id"]).long()
    mol_type = from_numpy(token_data["mol_type"]).long()
    res_type = from_numpy(token_data["res_type"]).long()
    is_standard = from_numpy(token_data["is_standard"])
    design = from_numpy(token_data["design_mask"]).long()
    # Per-residue amino acid constraint mask (shape: num_tokens x 20)
    if "aa_constraint_mask" in token_data.dtype.names:
        aa_constraint_mask = from_numpy(token_data["aa_constraint_mask"]).float()
    else:
        # Default: no constraints (all zeros = all AAs allowed)
        aa_constraint_mask = torch.zeros(len(token_data), len(const.canonical_tokens))
    res_type = one_hot(res_type, num_classes=const.num_tokens)
    modified = from_numpy(token_data["modified"]).long()
    ccd = from_numpy(token_data["ccd"]).long()
    binding_type = from_numpy(token_data["binding_type"]).long()
    structure_group = from_numpy(token_data["structure_group"]).long()
    center_coords = from_numpy(token_data["center_coords"]).float()
    target_msa_mask = from_numpy(token_data["target_msa_mask"])
    design_ss_mask = from_numpy(token_data["design_ss_mask"])
    feature_residue_index = from_numpy(token_data["feature_res_idx"]).long()
    feature_asym_id = from_numpy(token_data["feature_asym_id"]).long()
    symmetric_group = from_numpy(token_data["symmetric_group"]).long()
    token_to_res = from_numpy(data.token_to_res).long()

    method = (
        np.zeros(len(token_data))
        + const.method_types_ids[
            ("other" if override_method is None else override_method.lower())
        ]
    )
    default_T = const.temperature_bins_ids["other"]
    default_pH = const.ph_bins_ids["other"]
    temp_feature = np.zeros(len(token_data)) + default_T
    ph_feature = np.zeros(len(token_data)) + default_pH
    if data.record is not None:
        if (
            override_method is None
            and data.record.structure.method is not None
            and data.record.structure.method.lower() in const.method_types_ids
        ):
            method = (method * 0) + const.method_types_ids[
                data.record.structure.method.lower()
            ]

        if data.record.structure.temperature is not None:
            T = data.record.structure.temperature
            temp_feature = (temp_feature * 0) + get_range_bin(
                T, const.temperature_bins_ids, default=default_T
            )
        if data.record.structure.pH is not None:
            pH = data.record.structure.pH
            ph_feature = (ph_feature * 0) + get_range_bin(
                pH, const.ph_bins_ids, default=default_pH
            )
    method_feature = from_numpy(method).long()
    temp_feature = from_numpy(temp_feature).long()
    ph_feature = from_numpy(ph_feature).long()

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"]).float()
    disto_mask = from_numpy(token_data["disto_mask"]).float()

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    bonds_type = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
    skip_prob = random.random()
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        # TODO: Add a check to not include head-to-tail connections for cyclic peptides here
        bond_type = token_bond["type"]
        if (
            bond_type == 1 + const.bond_type_ids["COVALENT"]
            and token_data[token_1]["res_name"] == "CYS"
            and token_data[token_2]["res_name"] == "CYS"
            and token_data[token_1]["asym_id"] == token_data[token_2]["asym_id"]
            and abs(token_1 - token_2) != token_data[token_1]["cyclic_period"]
            and (not disulfide_on or skip_prob < 1 - disulfide_prob)
        ):
            continue

        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1
        bond_type = token_bond["type"]
        bonds_type[token_1, token_2] = bond_type
        bonds_type[token_2, token_1] = bond_type
    bonds = bonds.unsqueeze(-1)

    contact_conditioning = (
        np.zeros((len(token_data), len(token_data)))
        + const.contact_conditioning_info["UNSELECTED"]
    )
    contact_threshold = np.zeros((len(token_data), len(token_data)))
    contact_threshold = from_numpy(contact_threshold).float()

    if np.all(contact_conditioning == const.contact_conditioning_info["UNSELECTED"]):
        contact_conditioning = (
            contact_conditioning
            - const.contact_conditioning_info["UNSELECTED"]
            + const.contact_conditioning_info["UNSPECIFIED"]
        )
    contact_conditioning = from_numpy(contact_conditioning).long()
    contact_conditioning = one_hot(
        contact_conditioning, num_classes=len(const.contact_conditioning_info)
    )

    # compute cyclic polymer mask
    cyclic_ids = {}
    for idx_chain, asym_id_iter in enumerate(data.structure.chains["asym_id"]):
        for connection in data.structure.bonds:
            if (
                idx_chain == connection["chain_1"] == connection["chain_2"]
                and data.structure.chains[connection["chain_1"]]["res_num"] > 2
                and connection["res_1"]
                != connection["res_2"]  # Avoid same residue bonds!
            ):
                if (
                    data.structure.chains[connection["chain_1"]]["res_num"]
                    == (connection["res_2"] + 1)
                    and connection["res_1"] == 0
                ) or (
                    data.structure.chains[connection["chain_1"]]["res_num"]
                    == (connection["res_1"] + 1)
                    and connection["res_2"] == 0
                ):
                    cyclic_ids[asym_id_iter] = data.structure.chains[
                        connection["chain_1"]
                    ]["res_num"]
    cyclic = from_numpy(
        np.array(
            [
                (cyclic_ids[asym_id_iter] if asym_id_iter in cyclic_ids else 0)
                for asym_id_iter in token_data["asym_id"]
            ]
        )
    ).float()

    # Compute token_distance_mask
    token_distance_mask = torch.zeros((len(token_data), len(token_data)))
    if all(token_data["structure_group"] == 1):
        # equal to the elif case but more efficient
        token_distance_mask += 1
    elif any(token_data["structure_group"] > 0):
        for i, token_i in enumerate(token_data):
            for j, token_j in enumerate(token_data):
                if (
                    token_i["structure_group"] > 0
                    and token_i["structure_group"] == token_j["structure_group"]
                ):
                    token_distance_mask[i, j] = 1

    ligand_affinity_mask = mol_type == const.chain_type_ids["NONPOLYMER"]

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            is_standard = pad_dim(is_standard, 0, pad_len)
            design = pad_dim(design, 0, pad_len)
            aa_constraint_mask = pad_dim(aa_constraint_mask, 0, pad_len)
            binding_type = pad_dim(binding_type, 0, pad_len)
            structure_group = pad_dim(structure_group, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            contact_conditioning = pad_dim(contact_conditioning, 0, pad_len)
            contact_conditioning = pad_dim(contact_conditioning, 1, pad_len)
            contact_threshold = pad_dim(contact_threshold, 0, pad_len)
            contact_threshold = pad_dim(contact_threshold, 1, pad_len)
            method_feature = pad_dim(method_feature, 0, pad_len)
            temp_feature = pad_dim(temp_feature, 0, pad_len)
            ph_feature = pad_dim(ph_feature, 0, pad_len)
            modified = pad_dim(modified, 0, pad_len)
            ccd = pad_dim(ccd, 0, pad_len)
            cyclic = pad_dim(cyclic, 0, pad_len)
            ligand_affinity_mask = pad_dim(ligand_affinity_mask, 0, pad_len)
            center_coords = pad_dim(center_coords, 0, pad_len)
            token_distance_mask = pad_dim(
                pad_dim(token_distance_mask, 0, pad_len), 1, pad_len
            )
            target_msa_mask = pad_dim(target_msa_mask, 0, pad_len)
            design_ss_mask = pad_dim(design_ss_mask, 0, pad_len)
            feature_residue_index = pad_dim(feature_residue_index, 0, pad_len)
            feature_asym_id = pad_dim(feature_asym_id, 0, pad_len)
            symmetric_group = pad_dim(symmetric_group, 0, pad_len)
            token_to_res = pad_dim(token_to_res, 0, pad_len)
    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "res_type_clone": res_type.clone(),
        "is_standard": is_standard,
        "design_mask": design,
        "aa_constraint_mask": aa_constraint_mask,
        "binding_type": binding_type,
        "structure_group": structure_group,
        "token_bonds": bonds,
        "type_bonds": bonds_type,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "token_pair_mask": disto_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "method_feature": method_feature,
        "temp_feature": temp_feature,
        "ph_feature": ph_feature,
        "modified": modified,
        "ccd": ccd,
        "cyclic": cyclic,
        "center_coords": center_coords,
        "token_distance_mask": token_distance_mask,
        "target_msa_mask": target_msa_mask,
        "design_ss_mask": design_ss_mask,
        "feature_residue_index": feature_residue_index,
        "feature_asym_id": feature_asym_id,
        "symmetric_group": symmetric_group,
        "ligand_affinity_mask": ligand_affinity_mask,
        "token_to_res": token_to_res,
    }

    return token_features


def process_atom_features(
    data: Input,
    random: np.random.Generator,
    molecules: Dict[str, Mol],
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
    override_bfactor: bool = False,
    compute_frames: bool = True,
    backbone_only: bool = False,
    atom14: bool = False,
    atom37: bool = False,
    design: bool = False,
    inverse_fold: bool = False,
) -> Dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Input
        The input to the model.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    Dict[str, Tensor]
        The atom features.

    """
    if inverse_fold:
        one_hot = one_hot_bool
    else:
        one_hot = one_hot_torch

    if backbone_only or atom14 or atom37:
        assert design, "Design must be true if backbone_only or atom14 or atom37"

    # Filter to tokens' atoms
    atom_data = []
    new_to_old_atomidx = []
    atom_name = []
    atom_element = []
    atom_charge = []
    atom_conformer = []
    atom_chirality = []
    ref_space_uid = []
    coord_data = []
    frame_data = []
    resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []  # index on cropped atom table
    r_set_to_rep_atom = []
    disto_coords = []
    token_to_bb4_atoms = []
    backbone_mask = []
    fake_atom_mask = []
    masked_ref_atom_name_chars = []

    e_offsets = data.structure.ensemble["atom_coord_idx"]
    atom_idx = 0

    # Start atom idx in full atom table for structures chosen. Up to points.

    # Set unk chirality id
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    chain_res_ids = {}
    res_index_to_conf_id = {}

    for token_id, token in enumerate(data.tokens):
        original_atom_idx = token["atom_idx"]
        original_atom_num = token["atom_num"]

        # Get the chain residue ids
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]
        res_type = const.tokens[token["res_type"]]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Get the molecule and conformer
        mol = molecules[token["res_name"]]
        atom_name_to_ref = {}
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
        # Sample a random conformer

        if (chain_idx, res_id) not in res_index_to_conf_id:
            conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
            conf_id = int(random.choice(conf_ids))
            res_index_to_conf_id[(chain_idx, res_id)] = conf_id

        conf_id = res_index_to_conf_id[(chain_idx, res_id)]
        conformer = mol.GetConformer(conf_id)

        # get atom_num
        if bool(token["design_mask"]) and backbone_only:
            if token["mol_type"] == const.chain_type_ids["PROTEIN"]:
                atom_num = 4
            elif token["mol_type"] == const.chain_type_ids["DNA"]:
                atom_num = 11
            elif token["mol_type"] == const.chain_type_ids["RNA"]:
                atom_num = 12
            else:
                atom_num = token["atom_num"]
            assert token["atom_num"] >= atom_num
        elif bool(token["design_mask"]) and (atom14 or atom37):
            if token["mol_type"] == const.chain_type_ids["PROTEIN"]:
                atom_num = 14 if atom14 else 37
            elif token["mol_type"] == const.chain_type_ids["DNA"]:
                atom_num = 22
            elif token["mol_type"] == const.chain_type_ids["RNA"]:
                atom_num = 23
            else:
                atom_num = token["atom_num"]
        else:
            atom_num = token["atom_num"]

        # make backbone mask
        is_backbone = np.zeros(atom_num)
        if token["mol_type"] == const.chain_type_ids["PROTEIN"]:
            is_backbone[:4] = 1
        elif token["mol_type"] == const.chain_type_ids["DNA"]:
            is_backbone[:11] = 1
        elif token["mol_type"] == const.chain_type_ids["RNA"]:
            is_backbone[:12] = 1
        else:
            is_backbone = np.zeros(atom_num)
        backbone_mask.append(is_backbone)

        # get token rep_atom_offset
        if design:
            rep_atom_offset = token["center_idx"] - token["atom_idx"]
        else:
            rep_atom_offset = token["disto_idx"] - token["atom_idx"]

        assert rep_atom_offset >= 0, (
            f"rep_atom_offset: {rep_atom_offset} is less than 0"
        )
        # get atoms and fake atoms
        is_protein = token["mol_type"] == const.chain_type_ids["PROTEIN"]

        if bool(token["design_mask"]) and atom14 and is_protein:
            real_atoms = data.structure.atoms[
                token["atom_idx"] : token["atom_idx"] + token["atom_num"]
            ]
            fake_atoms = possible_fake_atoms[token["atom_num"] : atom_num]
            token_atoms = np.concatenate([real_atoms, fake_atoms])
            placements = np.array(const.fake_atom_placements[res_type])
            oxygen = real_atoms[const.ref_atoms[res_type].index("O")]
            token_atoms["coords"][placements == "O"] = oxygen["coords"]
            token_atoms["is_present"][placements == "O"] = oxygen["is_present"]
            token_atoms["plddt"][placements == "O"] = oxygen["plddt"]

            nitrogen = real_atoms[const.ref_atoms[res_type].index("N")]
            token_atoms["coords"][placements == "N"] = nitrogen["coords"]
            token_atoms["is_present"][placements == "N"] = nitrogen["is_present"]
            token_atoms["plddt"][placements == "N"] = nitrogen["plddt"]

            fake_atom_mask.append([0] * len(real_atoms))
            fake_atom_mask.append([1] * len(fake_atoms))
            assert len(token_atoms) == atom_num
        elif bool(token["design_mask"]) and atom37 and is_protein:
            real_atoms = data.structure.atoms[
                token["atom_idx"] : token["atom_idx"] + token["atom_num"]
            ]
            real_atoms_name_to_idx = {
                real_atoms[i][0]: i for i in range(len(real_atoms))
            }
            fake_atoms = [
                possible_fake_atoms[i]
                for i, name in enumerate(const.atom_types)
                if name in real_atoms_name_to_idx
            ]

            token_atoms = np.array(
                [
                    real_atoms[real_atoms_name_to_idx[name]]
                    if name in real_atoms_name_to_idx
                    else possible_fake_atoms[i]
                    for i, name in enumerate(const.atom_types)
                ]
            )
            fake_atom_mask.append(
                [
                    0 if name in real_atoms_name_to_idx else 1
                    for name in const.atom_types
                ]
            )
            placements = np.array(
                [
                    "." if name in real_atoms_name_to_idx else "CA"
                    for name in const.atom_types
                ]
            )

            ca = real_atoms[const.atom_types.index("CA")]
            token_atoms["coords"][placements == "CA"] = ca["coords"]
            token_atoms["is_present"][placements == "CA"] = ca["is_present"]
            token_atoms["plddt"][placements == "CA"] = ca["plddt"]
            assert len(token_atoms) == atom_num
        else:
            real_atoms = data.structure.atoms[
                token["atom_idx"] : token["atom_idx"] + atom_num
            ]
            token_atoms = real_atoms
            fake_atoms = []
            fake_atom_mask.append([0] * atom_num)

        # Add atom ref data
        # element, charge, conformer, chirality
        token_atom_name = np.array([convert_atom_name(a["name"]) for a in token_atoms])
        token_atoms_element = []
        token_atoms_charge = []
        token_atoms_conformer = []
        token_atoms_chirality = []
        for atom in token_atoms:
            if const.fake_element in atom["name"]:
                token_atoms_element.append(const.fake_element_id)
                token_atoms_charge.append(0)
                token_atoms_conformer.append(token_atoms_conformer[1])  # CA coord
                token_atoms_chirality.append(unk_chirality)
            else:
                assert atom["name"] in atom_name_to_ref, (
                    f"Atom {atom['name']} not found in {atom_name_to_ref}"
                )
                atom_ref = atom_name_to_ref[atom["name"]]
                token_atoms_element.append(atom_ref.GetAtomicNum())
                token_atoms_charge.append(atom_ref.GetFormalCharge())
                token_atoms_conformer.append(
                    (
                        conformer.GetAtomPosition(atom_ref.GetIdx()).x,
                        conformer.GetAtomPosition(atom_ref.GetIdx()).y,
                        conformer.GetAtomPosition(atom_ref.GetIdx()).z,
                    )
                )
                token_atoms_chirality.append(
                    const.chirality_type_ids.get(
                        atom_ref.GetChiralTag().name, unk_chirality
                    )
                )
        token_atoms_element = np.array(token_atoms_element)
        token_atoms_charge = np.array(token_atoms_charge)
        token_atoms_conformer = np.array(token_atoms_conformer)
        token_atoms_chirality = np.array(token_atoms_chirality)
        # Map atoms to token indices
        ref_space_uid.extend([new_idx] * atom_num)
        atom_to_token.extend([token_id] * atom_num)

        # Map token to representative atom
        token_to_rep_atom.append(atom_idx + rep_atom_offset)
        if (chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]) and token[
            "resolved_mask"
        ]:
            r_set_to_rep_atom.append(atom_idx + rep_atom_offset)

        # Get token coordinates across sampled ensembles  and apply transforms
        coords_set = data.structure.coords[
            token["atom_idx"] : token["atom_idx"] + atom_num
        ]["coords"]

        # Get coordinates of fake atoms
        if bool(token["design_mask"]) and atom14 and len(fake_atoms) > 0:
            coords_set = np.concatenate(
                [
                    coords_set[: len(real_atoms)],
                    np.zeros((len(fake_atoms), 3), dtype=coords_set.dtype),
                ]
            )
            o_coord = coords_set[const.ref_atoms[res_type].index("O")]
            coords_set[placements == "O"] = o_coord
            n_coord = coords_set[const.ref_atoms[res_type].index("N")]
            coords_set[placements == "N"] = n_coord
        elif bool(token["design_mask"]) and atom37 and len(fake_atoms) > 0:
            ca_coord = coords_set[const.ref_atoms[res_type].index("CA")]
            coords_set = np.array(
                [
                    coords_set[const.ref_atoms[res_type].index(name)]
                    if name in const.ref_atoms[res_type]
                    else ca_coord
                    for name in const.atom_types
                ]
            )
        token_coords_list = []
        token_coords_list.append(coords_set)

        token_coords = np.array(token_coords_list)
        coord_data.append(token_coords)

        # Get frame data
        res_type = const.tokens[token["res_type"]]
        res_name = str(token["res_name"])

        if atom_num < 3 or res_type in ["PAD", "UNK", "-", "DN", "N"]:
            idx_frame_a, idx_frame_b, idx_frame_c, idx_frame_d = 0, 0, 0, 0
            mask_frame = False
        elif (token["mol_type"] == const.chain_type_ids["PROTEIN"]) and (
            res_name in const.ref_atoms
        ):
            idx_frame_a, idx_frame_b, idx_frame_c, idx_frame_d = (
                const.ref_atoms[res_name].index("N"),
                const.ref_atoms[res_name].index("CA"),
                const.ref_atoms[res_name].index("C"),
                const.ref_atoms[res_name].index("O"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (
            token["mol_type"] == const.chain_type_ids["DNA"]
            or token["mol_type"] == const.chain_type_ids["RNA"]
        ) and (res_name in const.ref_atoms):
            idx_frame_a, idx_frame_b, idx_frame_c, idx_frame_d = (
                const.ref_atoms[res_name].index("C1'"),
                const.ref_atoms[res_name].index("C3'"),
                const.ref_atoms[res_name].index("C4'"),
                const.ref_atoms[res_name].index("P"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif token["mol_type"] == const.chain_type_ids["PROTEIN"]:
            # Try to look for the atom nams in the modified residue
            is_ca = token_atoms["name"] == "CA"
            idx_frame_a = is_ca.argmax()
            ca_present = (
                token_atoms[idx_frame_a]["is_present"] if is_ca.any() else False
            )

            is_n = token_atoms["name"] == "N"
            idx_frame_b = is_n.argmax()
            n_present = token_atoms[idx_frame_b]["is_present"] if is_n.any() else False

            is_c = token_atoms["name"] == "C"
            idx_frame_c = is_c.argmax()
            c_present = token_atoms[idx_frame_c]["is_present"] if is_c.any() else False
            mask_frame = ca_present and n_present and c_present

        elif (token["mol_type"] == const.chain_type_ids["DNA"]) or (
            token["mol_type"] == const.chain_type_ids["RNA"]
        ):
            # Try to look for the atom nams in the modified residue
            is_c1 = token_atoms["name"] == "C1'"
            idx_frame_a = is_c1.argmax()
            c1_present = (
                token_atoms[idx_frame_a]["is_present"] if is_c1.any() else False
            )

            is_c3 = token_atoms["name"] == "C3'"
            idx_frame_b = is_c3.argmax()
            c3_present = (
                token_atoms[idx_frame_b]["is_present"] if is_c3.any() else False
            )

            is_c4 = token_atoms["name"] == "C4'"
            idx_frame_c = is_c4.argmax()
            c4_present = (
                token_atoms[idx_frame_c]["is_present"] if is_c4.any() else False
            )
            mask_frame = c1_present and c3_present and c4_present
        else:
            idx_frame_a, idx_frame_b, idx_frame_c, idx_frame_d = 0, 0, 0, 0
            mask_frame = False
        frame_data.append(
            [
                idx_frame_a + atom_idx,
                idx_frame_b + atom_idx,
                idx_frame_c + atom_idx,
            ]
        )
        token_to_bb4_atoms.append(
            [
                idx_frame_a + atom_idx,
                idx_frame_b + atom_idx,
                idx_frame_c + atom_idx,
                idx_frame_d + atom_idx,
            ]
        )
        resolved_frame_data.append(mask_frame)

        # Get distogram coordinates
        disto_coords_tok = data.structure.coords[token["atom_idx"] + rep_atom_offset][
            "coords"
        ]
        disto_coords.append(disto_coords_tok)

        # produce masked_ref_atom_name_chars
        for i in range(atom_num):
            masked_ref_atom_name_chars.append(
                convert_atom_name(const.mask_element + str(i))
            )

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        # atom has a copy of first coords in ensemble
        token_atoms["coords"] = token_coords[0]

        atom_data.append(token_atoms)
        atom_name.append(token_atom_name)
        atom_element.append(token_atoms_element)
        atom_charge.append(token_atoms_charge)
        atom_conformer.append(token_atoms_conformer)
        atom_chirality.append(token_atoms_chirality)
        new_to_old_atomidx.extend(
            range(token["atom_idx"], token["atom_idx"] + len(real_atoms))
        )
        new_to_old_atomidx.extend([-1] * len(fake_atoms))
        atom_idx += len(token_atoms)

    backbone_mask = np.concatenate(backbone_mask)
    fake_atom_mask = np.concatenate(fake_atom_mask)
    disto_coords = np.array(disto_coords)
    new_to_old_atomidx = np.array(new_to_old_atomidx)

    # Compute ensemble distogram
    L = len(data.tokens)

    # Create distogram
    if not inverse_fold:
        disto_target = torch.zeros(L, L, 1, num_bins)
        t_center = torch.Tensor(disto_coords)
        t_dists = torch.cdist(t_center, t_center)
        boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
        distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
        disto_target[:, :, 0, :] = one_hot(distogram, num_classes=num_bins).bool()
    else:
        disto_target = None

    # Normalize distogram
    atom_data = np.concatenate(atom_data)
    atom_name = np.concatenate(atom_name)
    atom_element = np.concatenate(atom_element)
    atom_charge = np.concatenate(atom_charge)
    atom_conformer = np.concatenate(atom_conformer)
    atom_chirality = np.concatenate(atom_chirality)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # Compute features
    disto_coords = from_numpy(disto_coords).unsqueeze(0)
    ref_atom_name_chars = from_numpy(atom_name).long()
    masked_ref_atom_name_chars = torch.tensor(masked_ref_atom_name_chars).long()
    ref_element = from_numpy(atom_element).long()
    ref_charge = from_numpy(atom_charge).float()
    ref_pos = from_numpy(atom_conformer).float()
    ref_space_uid = from_numpy(ref_space_uid)
    ref_chirality = from_numpy(atom_chirality).long()
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"])
    backbone_mask = from_numpy(backbone_mask)
    fake_atom_mask = from_numpy(fake_atom_mask)
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)
    token_to_bb4_atoms = torch.tensor(token_to_bb4_atoms, dtype=torch.long)
    new_to_old_atomidx = torch.tensor(new_to_old_atomidx, dtype=torch.long)
    bfactor = from_numpy(atom_data["bfactor"].copy())
    plddt = from_numpy(atom_data["plddt"].copy())

    # Override bfactor when e.g. training on distillation data that does not have bfactor
    if override_bfactor:
        bfactor = bfactor * 0.0

    # We compute frames within ensemble
    if compute_frames:
        frames = []
        frame_resolved_mask = []
        for i in range(coord_data.shape[0]):
            frame_data_, resolved_frame_data_ = compute_frames_nonpolymer(
                data,
                coord_data[i],
                atom_data["is_present"],
                atom_to_token,
                frame_data,
                resolved_frame_data,
            )  # Compute frames for NONPOLYMER tokens
            frames.append(frame_data_.copy())
            frame_resolved_mask.append(resolved_frame_data_.copy())
        frames = from_numpy(np.stack(frames))  # (N_ENS, N_TOK, 3)
        frame_resolved_mask = from_numpy(np.stack(frame_resolved_mask))

    # Convert to one-hot
    ref_atom_name_chars = one_hot(ref_atom_name_chars, num_classes=64)
    masked_ref_atom_name_chars = one_hot(masked_ref_atom_name_chars, num_classes=64)
    ref_element = one_hot(ref_element, num_classes=const.num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1).bool()
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data)).bool()
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data)).bool()
    token_to_bb4_atoms = one_hot(token_to_bb4_atoms, num_classes=len(atom_data)).bool()

    # Center the ground truth coordinates
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]

    # Apply random roto-translation to the input conformers
    for i in range(torch.max(ref_space_uid)):
        included = ref_space_uid == i
        # TODO: replace this code with the commented out code below when training a new model
        if torch.sum(included) > 0 and torch.any(resolved_mask[included]):
            ref_pos[included] = center_random_augmentation(
                ref_pos[included][None], resolved_mask[included][None], centering=True
            )[0]
        # TODO: use this code instead of the above when training a new model (we are keeping the old code for now to avoid a training vs. inference difference)
        # if torch.sum(included) > 0:
        #     ref_pos[included] = center_random_augmentation(
        #         ref_pos[included][None], torch.ones_like(resolved_mask[included][None]), centering=True
        #     )[0]

    # Compute padding and apply
    if max_atoms is None or (atom14 or atom37):
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)
    else:
        assert max_atoms % atoms_per_window_queries == 0
        pad_len = max_atoms - len(atom_data)
        if pad_len < 0:
            # make sure len(atom_data) can be divided by atoms_per_window_queries
            pad_len = (
                len(atom_data) // atoms_per_window_queries + 1
            ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        masked_ref_atom_name_chars = pad_dim(masked_ref_atom_name_chars, 0, pad_len)
        backbone_mask = pad_dim(backbone_mask, 0, pad_len)
        fake_atom_mask = pad_dim(fake_atom_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_chirality = pad_dim(ref_chirality, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len)
        new_to_old_atomidx = pad_dim(new_to_old_atomidx, 0, pad_len)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len)
        bfactor = pad_dim(bfactor, 0, pad_len)
        plddt = pad_dim(plddt, 0, pad_len)
        token_to_bb4_atoms = pad_dim(token_to_bb4_atoms, 2, pad_len)

    if max_tokens is not None:
        pad_len = max_tokens - token_to_rep_atom.shape[0]
        if pad_len > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len)
            token_to_bb4_atoms = pad_dim(token_to_bb4_atoms, 0, pad_len)
            if not inverse_fold:
                disto_target = pad_dim(pad_dim(disto_target, 0, pad_len), 1, pad_len)
            disto_coords = pad_dim(disto_coords, 1, pad_len)
            new_to_old_atomidx = pad_dim(new_to_old_atomidx, 0, pad_len)

            if compute_frames:
                frames = pad_dim(frames, 1, pad_len)
                frame_resolved_mask = pad_dim(frame_resolved_mask, 1, pad_len)
    extra_mols = {k: v for k, v in molecules.items() if re.match(r"^LIG\d+", k)}
    atom_features = {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "new_to_old_atomidx": new_to_old_atomidx,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "disto_target": disto_target,
        "disto_coords": disto_coords,
        "bfactor": bfactor,
        "plddt": plddt,
        "masked_ref_atom_name_chars": masked_ref_atom_name_chars,
        "backbone_mask": backbone_mask,
        "fake_atom_mask": fake_atom_mask,
        "token_to_bb4_atoms": token_to_bb4_atoms,
        "extra_mols": extra_mols,
    }

    if compute_frames:
        atom_features["frames_idx"] = frames
        atom_features["frame_resolved_mask"] = frame_resolved_mask

    return atom_features


def process_msa_features(
    data: Input,
    random: np.random.Generator,
    max_seqs_batch: int,
    max_seqs: int,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
    msa_sampling: bool = False,
    affinity: bool = False,
) -> Dict[str, Tensor]:
    """Get the MSA features.

    Parameters
    ----------
    data : Input
        The input to the model.
    random : np.random.Generator
        The random number generator.
    max_seqs : int
        The maximum number of MSA sequences.
    max_tokens : int
        The maximum number of tokens.
    pad_to_max_seqs : bool
        Whether to pad to the maximum number of sequences.
    msa_sampling : bool
        Whether to sample the MSA.

    Returns
    -------
    Dict[str, Tensor]
        The MSA features.

    """
    # Created paired MSA
    msa, deletion, paired = construct_paired_msa(
        data=data,
        random=random,
        max_seqs=max_seqs_batch,
        random_subset=msa_sampling,
    )

    msa, deletion, paired = (
        msa.transpose(1, 0),
        deletion.transpose(1, 0),
        paired.transpose(1, 0),
    )  # (N_MSA, N_RES, N_AA)

    # Prepare features
    assert torch.all(msa >= 0) and torch.all(msa < const.num_tokens)
    msa_one_hot = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
    msa_mask = torch.ones_like(msa)
    profile = msa_one_hot.float().mean(dim=0)
    has_deletion = deletion > 0
    deletion = np.pi / 2 * np.arctan(deletion / 3)
    deletion_mean = deletion.mean(axis=0)

    # Pad in the MSA dimension (dim=0)
    if pad_to_max_seqs:
        pad_len = max_seqs - msa.shape[0]
        if pad_len > 0:
            msa = pad_dim(msa, 0, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 0, pad_len)
            msa_mask = pad_dim(msa_mask, 0, pad_len)
            has_deletion = pad_dim(has_deletion, 0, pad_len)
            deletion = pad_dim(deletion, 0, pad_len)

    # Pad in the token dimension (dim=1)
    if max_tokens is not None:
        pad_len = max_tokens - msa.shape[1]
        if pad_len > 0:
            msa = pad_dim(msa, 1, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 1, pad_len)
            msa_mask = pad_dim(msa_mask, 1, pad_len)
            has_deletion = pad_dim(has_deletion, 1, pad_len)
            deletion = pad_dim(deletion, 1, pad_len)
            profile = pad_dim(profile, 0, pad_len)
            deletion_mean = pad_dim(deletion_mean, 0, pad_len)

    if affinity:
        return {
            "deletion_mean_affinity": deletion_mean,
            "profile_affinity": profile,
        }
    else:
        return {
            "msa": msa,
            "msa_paired": paired,
            "deletion_value": deletion,
            "has_deletion": has_deletion,
            "deletion_mean": deletion_mean,
            "profile": profile,
            "msa_mask": msa_mask,
        }


def process_symmetry_features(
    cropped: Input,
    symmetries: Dict,
    backbone_only: bool = False,
    atom14: bool = False,
    atom37: bool = False,
) -> Dict[str, Tensor]:
    """Get the symmetry features.

    Parameters
    ----------
    data : Input
        The input to the model.

    Returns
    -------
    Dict[str, Tensor]
        The symmetry features.

    """
    # TODO this does not work with multiple conformers
    features = get_chain_symmetries(cropped, backbone_only, atom14, atom37)
    features.update(get_amino_acids_symmetries(cropped, backbone_only, atom14, atom37))
    features.update(get_ligand_symmetries(cropped, symmetries))

    return features


def repopulate_res_type(
    feat,
    mol_types: List[str] = ["PROTEIN", "DNA", "RNA"],
    tokens: List[str] = ["GLY", "DC", "C"],
    repopulate_non_standards: bool = False,
    design_only: bool = True,
):
    feat = copy.deepcopy(feat)
    if isinstance(mol_types, str):
        mol_types = [mol_types]
    if isinstance(tokens, str):
        tokens = [tokens]

    for mol_type, token in zip(mol_types, tokens):
        if repopulate_non_standards:
            mask = feat["mol_type"] == const.chain_type_ids[mol_type]
        else:
            mask = (feat["mol_type"] == const.chain_type_ids[mol_type]) & feat[
                "is_standard"
            ].bool()

        if design_only:
            mask &= feat["design_mask"].bool()
        name = convert_atom_name(token)
        name = torch.tensor(name).to(feat["res_type"])
        ccd_name = convert_ccd(token)
        ccd_name = torch.tensor(ccd_name).to(feat["res_type"])
        old = feat["res_type"][mask]
        new = torch.zeros_like(old)
        new[..., const.token_ids[token]] = 1
        feat["res_type"][mask] = new
        feat["ccd"][mask] = ccd_name
    return feat


def res_from_atom14(
    feat: Dict[str, Tensor],
    threshold: float = 0.5,
    invalid_token: str = "UNK",
):
    """Returns a new dict of features with residue types inferred from atom14 coords geometry. Always updates res_type, ref_atom_name_chars_full, and ref_element. If num_bins is not None, then ref_atom_name_name_chars are also updated.

    Parameters
    ----------
    feat : Dict[str, Tensor]
        The features.

    feat : float
        The value up to which coordinates are still counted as belonging to a backbone atom and therefore contributing to its count.

    num_bins: int
        If this is not None, then ref_atom_name_chars are also updated.

    Returns
    -------
    Dict[str, Tensor]
        The ensemble features.
    """
    one_hot = one_hot_torch
    feat = copy.deepcopy(feat)

    # Only attempt residue-type inference on designed protein residues
    design_mask = (
        feat["design_mask"].bool()
        & (feat["mol_type"] == const.chain_type_ids["PROTEIN"]).bool()
    )
    if design_mask.sum() == 0:
        return feat

    # Get designed atom coordinates in shape N//14 x 14 x 3
    atom_to_token = torch.argmax(feat["atom_to_token"].int(), dim=-1)
    token_indices = feat["token_index"][design_mask]
    atom_design_mask = torch.isin(atom_to_token, token_indices)
    atom_design_mask = atom_design_mask & feat["atom_pad_mask"].bool()
    design_coords = feat["coords"][atom_design_mask].clone()
    design_coords = design_coords.view(len(design_coords) // 14, 14, 3)

    # For each sidechain atom, compute closest backbone atom and count them
    # while excluding those side chain atoms whose distance is above a threshold
    distances = torch.cdist(design_coords[:, :4], design_coords[:, 4:])
    value, argmin = torch.min(distances, dim=1)
    argmin[value > threshold] = -1
    arange = torch.arange(len(const.ref_atoms["GLY"]), device=argmin.device)
    counts = (argmin[:, :, None] == arange[None, None, :]).sum(1).long()

    # update res_type and name
    # counts has shape N x 4, now we extract which combination of counts corresponds to which restype
    # Invalid counts will be turned into UNK. Note that there also exists a valid count that encodes UNK.
    res_type_letters = [
        const.placement_count_to_token.get(tuple(count.tolist()), invalid_token)
        for count in counts
    ]
    res_type = [const.token_ids[ttype] for ttype in res_type_letters]
    res_type = torch.tensor(res_type).to(feat["res_type"])
    feat["res_type"][design_mask] = one_hot(res_type, len(const.tokens))
    ccds = [convert_ccd(res_name) for res_name in res_type_letters]
    ccds = torch.tensor(ccds).to(feat["res_type"])
    feat["ccd"][design_mask] = ccds

    # update ref_element, and ref_atom_name_chars, ref_charge,
    design_names = feat["ref_atom_name_chars"][atom_design_mask]
    design_elements = feat["ref_element"][atom_design_mask]
    for design_token_index, res_type_letter in enumerate(res_type_letters):
        for atom14_idx, atom_name in enumerate(const.ref_atoms[res_type_letter]):
            design_atom_idx = design_token_index * 14 + atom14_idx

            name = convert_atom_name(atom_name)
            name = torch.tensor(name).to(feat["ref_element"])
            design_names[design_atom_idx] = one_hot(name, num_classes=64)
            design_elements[design_atom_idx] = one_hot(
                torch.tensor(
                    const.element_to_atomic_num[
                        elem_from_name(atom_name, res_type_letter)
                    ]
                ),
                num_classes=const.num_elements,
            ).to(feat["ref_element"])
    feat["ref_atom_name_chars"][atom_design_mask] = design_names
    feat["ref_element"][atom_design_mask] = design_elements
    return feat


def res_from_atom37(
    feat: Dict[str, Tensor],
    threshold: float = 0.5,
    invalid_token: str = "UNK",
):
    """Returns a new dict of features with residue types inferred from atom37 coords geometry. Always updates res_type, ref_atom_name_chars_full, and ref_element. If num_bins is not None, then ref_atom_name_name_chars are also updated.

    Parameters
    ----------
    feat : Dict[str, Tensor]
        The features.

    feat : float
        The value up to which coordinates are still counted as belonging to a backbone atom and therefore contributing to its count.

    num_bins: int
        If this is not None, then ref_atom_name_chars are also updated.

    Returns
    -------
    Dict[str, Tensor]
        The ensemble features.
    """
    one_hot = one_hot_torch
    feat = copy.deepcopy(feat)

    # A residue being True in the design mask implies
    # that the residue is a standard residue and has moltype PROTEIN
    design_mask = feat["design_mask"].bool()
    if design_mask.sum() == 0:
        return feat

    # Get designed atom coordinates in shape N//37 x 37 x 3
    atom_to_token = torch.argmax(feat["atom_to_token"], dim=-1)
    token_indices = feat["token_index"][design_mask]
    atom_design_mask = torch.isin(atom_to_token, token_indices)
    atom_design_mask = atom_design_mask & feat["atom_pad_mask"].bool()
    design_coords = feat["coords"][atom_design_mask].clone()
    design_coords = design_coords.view(len(design_coords) // 37, 37, 3)

    # For each sidechain atom, compute closest backbone atom and count them
    # while excluding those side chain atoms whose distance is above a threshold
    distances = torch.cdist(design_coords[:, :4], design_coords[:, 4:])
    value, argmin = torch.min(distances, dim=1)
    argmin[value > threshold] = -1
    ca_placements = (argmin[:, :] == const.atom_types.index("CA")).long()

    res_names = [res for res in list(const.fake_atom_placements.keys()) if res != "UNK"]
    atom37_mask = torch.stack(
        [
            torch.tensor(
                [
                    1 if name in const.ref_atoms[res_type] else 0
                    for name in const.atom_types[4:]
                ],
                device=ca_placements.device,
            )
            for res_type in res_names
        ]
    )

    B = ca_placements.shape[0]
    atom37_comparison = torch.abs(
        ca_placements.unsqueeze(1).expand(-1, 20, -1)
        - (1 - atom37_mask.unsqueeze(0).expand(B, -1, -1))
    ).sum(axis=-1)
    decoded_idxs, res_idxs = np.where(atom37_comparison.cpu().numpy() == 0)
    _, n_decoded_res = np.unique(decoded_idxs, return_counts=True)
    if len(n_decoded_res) > 0:
        assert np.max(n_decoded_res) <= 1, "Ambiguous residue!"
    res_type_letters = ["UNK"] * B
    for i, res_i in zip(decoded_idxs, res_idxs):
        res_type_letters[i] = res_names[res_i]

    res_type = [const.token_ids[ttype] for ttype in res_type_letters]
    res_type = torch.tensor(res_type).to(feat["res_type"])
    feat["res_type"][design_mask] = one_hot(res_type, len(const.tokens))
    ccds = [convert_ccd(res_name) for res_name in res_type_letters]
    ccds = torch.tensor(ccds).to(feat["res_type"])
    feat["ccd"][design_mask] = ccds

    # update ref_element, and ref_atom_name_chars, ref_charge,
    design_names = feat["ref_atom_name_chars"][atom_design_mask]
    design_elements = feat["ref_element"][atom_design_mask]
    for design_token_index, res_type_letter in enumerate(res_type_letters):
        for atom14_idx, atom_name in enumerate(const.ref_atoms[res_type_letter]):
            atom37_idx = const.atom_types.index(atom_name)
            design_atom_idx = design_token_index * 37 + atom37_idx

            name = convert_atom_name(atom_name)
            name = torch.tensor(name).to(feat["ref_atom_name_chars"])
            design_names[design_atom_idx] = one_hot(name, num_classes=64)
            design_elements[design_atom_idx] = one_hot(
                torch.tensor(
                    const.element_to_atomic_num[
                        elem_from_name(atom_name, res_type_letter)
                    ]
                ),
                num_classes=const.num_elements,
            )
    feat["ref_atom_name_chars"][atom_design_mask] = design_names
    feat["ref_element"][atom_design_mask] = design_elements
    return feat


def res_all_gly(
    feat: Dict[str, Tensor],
):
    """Returns a new dict of features with residue types for only backbones

    Parameters
    ----------
    feat : Dict[str, Tensor]
        The features.

    feat : float
        The value up to which coordinates are still counted as belonging to a backbone atom and therefore contributing to its count.

    num_bins: int
        If this is not None, then ref_atom_name_chars are also updated.

    Returns
    -------
    Dict[str, Tensor]
        The ensemble features.
    """
    one_hot = one_hot_torch
    feat = copy.deepcopy(feat)

    # Only attempt residue-type inference on designed protein residues
    design_mask = (
        feat["design_mask"].bool()
        & (feat["mol_type"] == const.chain_type_ids["PROTEIN"]).bool()
    )
    if design_mask.sum() == 0:
        return feat

    # Get designed atom coordinates in shape N//4 x 4 x 3
    atom_to_token = torch.argmax(feat["atom_to_token"].int(), dim=-1)
    token_indices = feat["token_index"][design_mask]
    atom_design_mask = torch.isin(atom_to_token, token_indices)
    atom_design_mask = atom_design_mask & feat["atom_pad_mask"].bool()
    design_coords = feat["coords"][atom_design_mask].clone()
    design_coords = design_coords.view(len(design_coords) // 4, 4, 3)

    # update res_type and name
    # counts has shape N x 4, now we extract which combination of counts corresponds to which restype
    # Invalid counts will be turned into UNK. Note that there also exists a valid count that encodes UNK.
    res_type_letters = ["GLY"] * (len(design_coords))
    res_type = [const.token_ids[ttype] for ttype in res_type_letters]
    res_type = torch.tensor(res_type).to(feat["res_type"])
    feat["res_type"][design_mask] = one_hot(res_type, len(const.tokens))
    ccds = [convert_ccd(res_name) for res_name in res_type_letters]
    ccds = torch.tensor(ccds).to(feat["res_type"])
    feat["ccd"][design_mask] = ccds

    # update ref_element, and ref_atom_name_chars, ref_charge,
    design_names = feat["ref_atom_name_chars"][atom_design_mask]
    design_elements = feat["ref_element"][atom_design_mask]
    for design_token_index, res_type_letter in enumerate(res_type_letters):
        for bbatom_idx, atom_name in enumerate(const.ref_atoms[res_type_letter]):
            design_atom_idx = design_token_index * 4 + bbatom_idx

            name = convert_atom_name(atom_name)
            name = torch.tensor(name).to(feat["ref_element"])
            design_names[design_atom_idx] = one_hot(name, num_classes=64)
            design_elements[design_atom_idx] = one_hot(
                torch.tensor(
                    const.element_to_atomic_num[
                        elem_from_name(atom_name, res_type_letter)
                    ]
                ),
                num_classes=const.num_elements,
            ).to(feat["ref_element"])
    feat["ref_atom_name_chars"][atom_design_mask] = design_names
    feat["ref_element"][atom_design_mask] = design_elements
    return feat


def find_token_idx_for_atom(data: Input, atom_idx: int) -> int:
    """Find the token_idx that contains the given atom index.

    Parameters
    ----------
    data : Input
        The input data containing tokens and structure information
    atom_idx : int
        The atom index to find the corresponding token for

    Returns
    -------
    int
        The token_idx that contains the given atom index
    """
    # Iterate through tokens to find which one contains this atom
    for token in data.tokens:
        if token["atom_idx"] <= atom_idx < token["atom_idx"] + token["atom_num"]:
            return token["token_idx"]

    raise ValueError(f"No token found containing atom index {atom_idx}")


class Featurizer:
    """Featurizer for model training."""

    def process(
        self,
        data: Input,
        random: np.random.Generator,
        molecules: Dict[str, Mol],
        training: bool,
        max_seqs: int,
        backbone_only: bool = False,
        atom14: bool = False,
        atom37: bool = False,
        design: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        pad_to_max_seqs: bool = False,
        compute_symmetries: bool = False,
        disulfide_prob: Optional[float] = 1.0,
        disulfide_on: Optional[bool] = False,
        compute_affinity: Optional[bool] = False,
        single_sequence_prop: Optional[float] = 0.0,
        msa_sampling: bool = False,
        override_bfactor: float = False,
        override_method: Optional[str] = None,
        compute_frames: bool = True,
        inverse_fold: bool = False,
    ) -> Dict[str, Tensor]:
        """Compute features.

        Parameters
        ----------
        data : Input
            Input structure and metadata (e.g., coordinates, sequence, templates).
        random : np.random.Generator
            Random generator used for stochastic augmentations or sampling.
        molecules : Dict[str, Mol]
            Dictionary mapping CCD ids to rdkit molecule objects. These are loaded from the moldir. It is a cache of the rdkit molecules in the moldir such that we do not reload SMs every time
        training : bool
            Whether or not to do MSA sampling and single sequence sampling even when an MSA is present
        max_seqs : int
            Maximum number MSA sequences.
        backbone_only : bool, optional
            If True, only backbone atom coordinates are used for designed residues.
        atom14 : bool, optional
            If True, use the 14-atom representation for designed resideues (what is in the paper).
        atom37 : bool, optional
            If True, use the 37-atom representation for designed residues.
        design : bool, optional
            If True, use the design mask to decide which residues are processed as designed residues.
        atoms_per_window_queries : int, optional
            Number of atoms per local window used in neighborhood queries.
        min_dist : float, optional
            Distogram min dist in Å
        max_dist : float, optional
            Distogram max dist in Å
        num_bins : int, optional
            Distogram number of bins
        max_tokens : Optional[int], optional
            Num tokens to pad to
        max_atoms : Optional[int], optional
            Num atoms to pad to
        pad_to_max_seqs : bool, optional
            For MSA. If True, pad all sequence features to `max_seqs` length.
        compute_symmetries : bool, optional
            Whether to compute symmetry features and return those. These are necessary for computing symmetry corrected RMSD during validation.
        disulfide_prob : Optional[float], optional
            Probability with which disulfide bonds are included in the bond features. This should be 1 for inference with the BoltzGen Diffusion.
        disulfide_on : Optional[bool], optional
            Whether or not to include disulfide bonds in the bond featuers. This should be true for inference with BoltzGen diffusion. It should be false for inference with the refolding model or with the affinity model.
        compute_affinity : Optional[bool], optional
            Whether to compute the MSA in the style that the affinity model used during training.
        single_sequence_prop : Optional[float], optional
            How often to do single sequence training. Only relevant for training.
        msa_sampling : bool, optional
            Whether to randomly subsample sequences from the MSA for training. ONly relevant for training.
        override_bfactor : float, optional
            Value to override B-factors with (e.g., for uniform confidence).
        override_method : Optional[str], optional
            Method name for overriding B-factors or other structure attributes.
        compute_frames : bool, optional
            Whether to compute local residue frames and backbone geometry tensors.
        inverse_fold : bool, optional
            If True, generate features suitable for inverse folding tasks.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary mapping feature names to PyTorch tensors for model training.
        """
        # Compute random number of sequences
        if training and max_seqs is not None:
            if random.random() > single_sequence_prop:
                max_seqs_batch = random.integers(1, max_seqs + 1)
            else:
                max_seqs_batch = 1
        else:
            max_seqs_batch = max_seqs

        # Compute token features
        token_features = process_token_features(
            data=data,
            random=random,
            max_tokens=max_tokens,
            override_method=override_method,
            disulfide_prob=disulfide_prob,
            disulfide_on=disulfide_on,
        )

        # Compute atom features
        atom_features = process_atom_features(
            data=data,
            random=random,
            molecules=molecules,
            atoms_per_window_queries=atoms_per_window_queries,
            min_dist=min_dist,
            max_dist=max_dist,
            num_bins=num_bins,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            override_bfactor=override_bfactor,
            compute_frames=compute_frames,
            backbone_only=backbone_only,
            atom14=atom14,
            atom37=atom37,
            design=design,
            inverse_fold=inverse_fold,
        )

        msa_features = process_msa_features(
            data=data,
            random=random,
            max_seqs_batch=max_seqs_batch,
            max_seqs=max_seqs,
            max_tokens=max_tokens,
            pad_to_max_seqs=pad_to_max_seqs,
            msa_sampling=training and msa_sampling,
        )

        # Compute MSA features
        msa_features_affinity = {}
        if compute_affinity:
            msa_features_affinity = process_msa_features(
                data=data,
                random=random,
                max_seqs_batch=1,
                max_seqs=1,
                max_tokens=max_tokens,
                pad_to_max_seqs=pad_to_max_seqs,
                msa_sampling=training and msa_sampling,
                affinity=True,
            )

        # Compute symmetry features
        symmetry_features = {}
        if compute_symmetries:
            symmetries = get_symmetries(molecules)
            symmetry_features = process_symmetry_features(
                data, symmetries, backbone_only, atom14, atom37
            )

        # Compute secondary structure features
        design_mask = token_features["design_mask"].bool()
        ss = torch.zeros(len(design_mask), dtype=torch.long)
        design_ss_mask = token_features["design_ss_mask"].bool()
        if design_ss_mask.sum() > 0:
            atom_design_mask = (
                atom_features["atom_to_token"].float()
                @ design_mask.unsqueeze(-1).float()
            )
            atom_design_mask = atom_design_mask.squeeze().bool()
            bb_design_mask = (
                atom_features["atom_pad_mask"].bool()
                & atom_design_mask
                & atom_features["backbone_mask"].bool()
            )
            if bb_design_mask.sum() > 1:
                bb = atom_features["coords"][0][bb_design_mask]
                bb = bb.reshape(-1, 4, 3)
                assert len(bb) == design_mask.sum()
                try:
                    # 0: loop,  1: alpha-helix,  2: beta-strand
                    ss_out = pydssp.assign(bb, out_type="index")
                    # 1: loop,  2: alpha-helix,  3: beta-strand
                    ss_out += 1
                    ss[design_mask] = ss_out
                    ss *= design_ss_mask
                except:
                    print(
                        "Could not comptue secondary structure annotation. Leaving it unspecified"
                    )
        token_features.update({"ss_type": ss})

        return {
            "structure_bonds": data.structure.bonds,
            **token_features,
            **atom_features,
            **msa_features,
            **symmetry_features,
            **msa_features_affinity,
        }
