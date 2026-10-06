from dataclasses import replace
from typing import List, Optional, Set

import numpy as np
from scipy.spatial.distance import cdist

from boltzgen.data import const
from boltzgen.data.crop.cropper import Cropper
from boltzgen.data.data import Input, Token, TokenBond, Tokenized


def pick_random_token(
    tokens: np.ndarray,
    random: np.random.Generator,
) -> np.ndarray:
    """Pick a random token from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    random : np.random.Generator
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    return tokens[random.integers(len(tokens))]


def pick_chain_token(
    tokens: np.ndarray,
    chain_id: int,
    random: np.random.Generator,
) -> np.ndarray:
    """Pick a random token from a chain.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain_tokens = tokens[tokens["asym_id"] == chain_id]

    # Pick from chain, fallback to all tokens
    if chain_tokens.size:
        query = pick_random_token(chain_tokens, random)
    else:
        query = pick_random_token(tokens, random)

    return query


def pick_interface_token(
    tokens: np.ndarray,
    interface: np.ndarray,
    random: np.random.Generator,
) -> np.ndarray:
    """Pick a random token from an interface.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    interface : int
        The interface ID.
    random : np.random.Generator
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Sample random interface
    chain_1 = int(interface["chain_1"])
    chain_2 = int(interface["chain_2"])

    tokens_1 = tokens[tokens["asym_id"] == chain_1]
    tokens_2 = tokens[tokens["asym_id"] == chain_2]

    # If no interface, pick from the chains
    if tokens_1.size and (not tokens_2.size):
        query = pick_random_token(tokens_1, random)
    elif tokens_2.size and (not tokens_1.size):
        query = pick_random_token(tokens_2, random)
    elif (not tokens_1.size) and (not tokens_2.size):
        query = pick_random_token(tokens, random)
    else:
        # If we have tokens, compute distances
        tokens_1_coords = tokens_1["center_coords"]
        tokens_2_coords = tokens_2["center_coords"]

        dists = cdist(tokens_1_coords, tokens_2_coords)
        cuttoff = dists < const.interface_cutoff

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (const.interface_cutoff + 5.0)

        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Select random token
        candidates = np.concatenate([tokens_1, tokens_2])
        query = pick_random_token(candidates, random)

    return query


def pick_initial_crop_token(
    tokens: np.ndarray,
    initial_crop: List[int],
    random: np.random.Generator,
) -> np.ndarray:
    """Pick a random token from the initial crop.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    initial_crop : List[int]
        The initial crop.
    random : np.random.Generator
        The random state for reproducibility.

    Returns
    -------
    np.ndarray

    """
    # Compute crop centroid
    crop_centroid = np.mean(tokens[initial_crop]["center_coords"], axis=0)

    # Compute distances to all tokens
    dists = cdist(tokens["center_coords"], crop_centroid[None])

    # Pick the closest token
    return tokens[np.argmin(dists[:, 0])]


class MultimerCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(
        self, neighborhood_sizes: List[int], dna_double_helix: bool = False
    ) -> None:
        """Initialize the cropper.

        Parameters
        ----------
        neighborhood_sizes : List[int]
            Modulates the type of cropping to be performed.
            Smaller neighborhoods result in more spatial
            cropping. Larger neighborhoods result in more
            continuous cropping. A mix can be achieved by
            providing a list of sizes from which to sample.

        """
        self.neighborhood_sizes = neighborhood_sizes
        self.dna_double_helix = dna_double_helix

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.Generator,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
        max_atoms: Optional[int] = None,
        return_indices: bool = False,
        initial_crop: Optional[List[int]] = None,
        prefer_protein_queries: Optional[bool] = False,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.Generator
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Check inputs
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)

        # randomly select a neighborhood size
        neighborhood_size = random.choice(self.neighborhood_sizes)

        # Get token data
        token_data = data.tokens
        token_to_res = data.token_to_res
        token_bonds = data.bonds
        mask = data.structure.mask
        chains = data.structure.chains
        interfaces = data.structure.interfaces

        # Filter to valid chains
        valid_chains = chains[mask]

        # Filter to valid interfaces
        valid_interfaces = interfaces
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        is_protein_mask = valid_tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        if prefer_protein_queries and is_protein_mask.sum() > 0:
            query_choice_tokens = valid_tokens[is_protein_mask]
        else:
            query_choice_tokens = valid_tokens

        # Pick a random token, chain, or interface
        if initial_crop is not None:
            query = pick_initial_crop_token(token_data, initial_crop, random)
        elif chain_id is not None:
            query = pick_chain_token(query_choice_tokens, chain_id, random)
        elif interface_id is not None:
            interface = interfaces[interface_id]
            query = pick_interface_token(query_choice_tokens, interface, random)
        elif valid_interfaces.size:
            idx = random.integers(len(valid_interfaces))
            interface = valid_interfaces[idx]
            query = pick_interface_token(query_choice_tokens, interface, random)
        else:
            idx = random.integers(len(valid_chains))
            chain_id = valid_chains[idx]["asym_id"]
            query = pick_chain_token(query_choice_tokens, chain_id, random)

        cropped = self.select_cropped_indices(
            token_data,
            valid_tokens,
            query,
            neighborhood_size,
            max_atoms,
            max_tokens,
            initial_crop,
        )

        # Get the cropped tokens sorted by index
        token_data = token_data[cropped]
        token_to_res = token_to_res[cropped]

        # Only keep bonds within the cropped tokens and reindex token_idx
        old_indices = token_data["token_idx"]
        old_to_new = {old: new for new, old in enumerate(old_indices)}
        token_bonds = np.array(
            [
                (old_to_new[bond["token_1"]], old_to_new[bond["token_2"]], bond["type"])
                for bond in token_bonds
                if bond["token_1"] in old_indices and bond["token_2"] in old_indices
            ],
            dtype=TokenBond,
        )
        token_data["token_idx"] = np.arange(len(token_data))

        # Return the cropped tokens
        if return_indices:
            return replace(
                data, tokens=token_data, bonds=token_bonds, token_to_res=token_to_res
            ), cropped
        else:
            return replace(
                data, tokens=token_data, bonds=token_bonds, token_to_res=token_to_res
            )

    def select_cropped_indices(
        self,
        tokens: np.ndarray,
        valid_tokens: np.ndarray,
        query: np.ndarray,
        neighborhood_size: int,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        initial_crop: Optional[np.ndarray] = None,
    ):
        # Sort all tokens by distance to query_coords
        dists = valid_tokens["center_coords"] - query["center_coords"]
        indices = np.argsort(np.linalg.norm(dists, axis=1))

        # Select cropped indices
        cropped: Set[int] = set()
        total_atoms = 0

        if initial_crop is not None:
            cropped.update(initial_crop)
            total_atoms = sum(tokens[idx]["atom_num"] for idx in initial_crop)

        for idx in indices:
            # Get the token
            token = valid_tokens[idx]

            # Get all tokens from this chain
            chain_tokens = tokens[tokens["asym_id"] == token["asym_id"]]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= neighborhood_size:
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighboorhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.

                min_idx = token["res_idx"] - neighborhood_size
                max_idx = token["res_idx"] + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]
                if len(max_token_set) < neighborhood_size:
                    max_token_set = chain_tokens

                if len(max_token_set) < neighborhood_size:
                    print(
                        "WARNING! len(max_token_set) < neighborhood_size in cropper in selector"
                    )
                    new_tokens = max_token_set
                else:
                    # Start by adding just the query token
                    new_tokens = max_token_set[
                        max_token_set["res_idx"] == token["res_idx"]
                    ]

                    # Expand the neighborhood until we have enough tokens, one
                    # by one to handle some edge cases with non-standard chains.
                    # We switch to the res_idx instead of the token_idx to always
                    # include all tokens from modified residues or from ligands.
                    min_idx = max_idx = token["res_idx"]
                    counter = 0
                    while new_tokens.size < neighborhood_size:
                        counter += 1
                        min_idx = min_idx - 1
                        max_idx = max_idx + 1
                        new_tokens = max_token_set
                        new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                        new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]
                        if counter > 1000:
                            raise Exception("Infinite loop in cropper while loop")

            # Compute new tokens and new atoms
            new_token_mask = np.isin(tokens["token_idx"], new_tokens["token_idx"])
            new_indices = np.arange(len(tokens), dtype=tokens["token_idx"].dtype)[
                new_token_mask
            ]
            new_indices = set(new_indices) - cropped
            new_tokens = tokens[list(new_indices)]
            new_atoms = np.sum(new_tokens["atom_num"])

            # Stop if we exceed the max number of tokens or atoms
            if (len(new_indices) > (max_tokens - len(cropped))) or (
                (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)
            ):
                break

            # Add new indices
            cropped.update(new_indices)
            total_atoms += new_atoms
        return np.array(sorted(cropped))

    def crop_indices(  # noqa: PLR0915
        self,
        data: Tokenized,
        cropped_indices: List[int],
    ) -> Tokenized:
        token_data = data.tokens
        ligand_rows = np.flatnonzero(token_data["mol_type"] == 3).tolist()  # noqa: PLR2004
        cropped_indices = sorted({*ligand_rows, *cropped_indices})
        token_data = token_data[cropped_indices]
        indices = token_data["token_idx"]
        token_bonds = data.bonds
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        return replace(
            data,
            tokens=token_data,
            bonds=token_bonds,
            token_to_res=(
                data.token_to_res[cropped_indices]
                if data.token_to_res is not None
                else None
            ),
        )
