import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import string
from typing import Dict, List, Optional, Tuple, Union
import numpy.typing as npt
import numpy as np
import biotite
from mashumaro.mixins.dict import DataClassDictMixin

from rdkit.Chem import Mol
import torch

from boltzgen.data import const

####################################################################################################
# SERIALIZABLE
####################################################################################################


class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


class JSONSerializable(DataClassDictMixin):
    """Serializable datatype."""

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        with path.open("r") as f:
            return cls.from_dict(json.load(f))

    def dump(self, path: Path) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f)


####################################################################################################
# SOURCE
####################################################################################################


@dataclass(frozen=True, slots=True)
class PDB:
    """A raw MMCIF PDB file."""

    id: str
    path: str


@dataclass(frozen=True, slots=True)
class A3M:
    """A raw A3M MSA file."""

    id: str
    path: str


@dataclass(frozen=True, slots=True)
class HHR:
    """A raw HRR Template file."""

    id: str
    path: str


@dataclass(frozen=True, slots=True)
class PFAM:
    """A raw PFAM Template file."""

    entity_id: str
    seq_id: str


@dataclass(frozen=True, slots=True)
class PFamSet:
    """A set of raw PFAM Template files."""

    pdb_id: str
    entities: list[PFAM]


@dataclass(frozen=True, slots=True)
class PubChem:
    """A raw PubChem file."""

    id: str
    aid: int
    sid: int
    cid: int
    outcome: int
    activity_name: str
    activity_qualifier: str
    affinity: float
    normalized_protein_accession: str
    protein_cluster: str
    modify_date: str
    deposit_date: str
    pair_id: int
    assay_prot_id: int
    smiles: List[str]
    mols: List[Mol] = None
    protein_cluster_03: Optional[str] = None
    protein_cluster_06: Optional[str] = None
    protein_cluster_09: Optional[str] = None
    protein_sequences: List[str] = None


@dataclass(frozen=True, slots=True)
class MDData:
    """A raw MD file."""

    pdb_id: str
    id: str
    path: str


@dataclass(frozen=True, slots=True)
class UniProtData:
    """A raw Uniprot file."""

    pdb_ids: List[str]
    id: str
    uniprot_id: str
    path: str
    ref_seq: str


####################################################################################################
# HELPERS
####################################################################################################


def convert_ccd(name: str) -> Tuple[int, int, int, int, int]:
    """Convert a ccd code to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip().upper()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (5 - len(name))
    return tuple(name)


def convert_atom_name(name: str) -> Tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """

    name = name.strip().upper()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def elem_from_name(atom_name, res_name):
    atom_key = re.sub(r"\d", "", atom_name)
    if atom_key in const.ambiguous_atoms:
        if isinstance(const.ambiguous_atoms[atom_key], str):
            element = const.ambiguous_atoms[atom_key]
        elif res_name in const.ambiguous_atoms[atom_key]:
            element = const.ambiguous_atoms[atom_key][res_name]
        else:
            element = const.ambiguous_atoms[atom_key]["*"]
    else:
        element = atom_key[0]
    return element


def numeric_to_string(
    name: Union[Tuple[int, int, int, int], Tuple[int, int, int, int, int]],
) -> str:
    name = [chr(c + 32) for c in name if c != 0]
    name = "".join(name)
    return name


####################################################################################################
# STRUCTURE
####################################################################################################


Atom = [
    ("name", np.dtype("<U4")),
    ("coords", np.dtype("3f4")),  # first conformer will be duplicated in this field
    ("is_present", np.dtype("?")),
    ("bfactor", np.dtype("f4")),
    ("plddt", np.dtype("f4")),
]


Bond = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
    ("symmetric_group", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]

Coords = [
    ("coords", np.dtype("3f4")),
]

# This is completely superfluos and ideally would be removed out of the repository
Ensemble = [
    ("atom_coord_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
]


@dataclass(frozen=True, slots=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray
    coords: np.ndarray
    ensemble: np.ndarray

    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)

        # Temporary for adding a  cyclic period that is not yet in the preprocessed data
        if "cyclic_period" not in structure["chains"].dtype.names:
            chains = np.empty(structure["chains"].shape, dtype=Chain)
            for name in structure["chains"].dtype.names:
                chains[name] = structure["chains"][name]
            chains["cyclic_period"] = -1
        else:
            chains = structure["chains"]

        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=chains,
            interfaces=structure["interfaces"],
            mask=structure["mask"],
            coords=structure["coords"],
            ensemble=structure["ensemble"],
        )

    @classmethod
    def concatenate(
        self, str_1: "Structure", str_2: "Structure", return_renaming: bool = False
    ) -> "Structure":
        """Concatenate two structures
        Parameters
        ----------
        str_1 : Structure

        str_2 : Structure
        """
        if str_1.atoms.shape[0] == 0:
            return str_2, {} if return_renaming else str_2
        elif str_2.atoms.shape[0] == 0:
            return str_1, {} if return_renaming else str_1

        # get size protein
        num_atoms1 = str_1.atoms.shape[0]
        num_res1 = str_1.residues.shape[0]
        num_chains1 = str_1.chains.shape[0]

        # Build new atoms
        atoms = np.concatenate([str_1.atoms.copy(), str_2.atoms.copy()])

        # Build new residues
        residues_1 = str_1.residues.copy()
        residues_2 = str_2.residues.copy()
        residues_2["atom_idx"] = residues_2["atom_idx"] + num_atoms1
        residues_2["atom_center"] = residues_2["atom_center"] + num_atoms1
        residues_2["atom_disto"] = residues_2["atom_disto"] + num_atoms1
        residues = np.concatenate([residues_1, residues_2])

        # Build new chains
        chains_1 = str_1.chains.copy()
        chains_2 = str_2.chains.copy()
        chains_2["atom_idx"] = chains_2["atom_idx"] + num_atoms1

        str_1_seqres = []
        existing_chains_str_2 = []
        for chain_1 in chains_1:
            if (
                chain_1["mol_type"] == const.chain_type_ids["PROTEIN"]
                or chain_1["mol_type"] == const.chain_type_ids["RNA"]
                or chain_1["mol_type"] == const.chain_type_ids["DNA"]
            ):
                chain_1_seqres = "".join(
                    str_1.residues["name"][
                        chain_1["res_idx"] : chain_1["res_idx"] + chain_1["res_num"]
                    ]
                )
            if chain_1["mol_type"] == const.chain_type_ids["NONPOLYMER"]:
                chain_1_seqres = str_1.residues["name"][chain_1["res_idx"]]
            str_1_seqres.append(chain_1_seqres)
        for chain_idx, chain_2 in enumerate(chains_2):
            if (
                chain_2["mol_type"] == const.chain_type_ids["PROTEIN"]
                or chain_2["mol_type"] == const.chain_type_ids["RNA"]
                or chain_2["mol_type"] == const.chain_type_ids["DNA"]
            ):
                chain_2_seqres = "".join(
                    str_2.residues["name"][
                        chain_2["res_idx"] : chain_2["res_idx"] + chain_2["res_num"]
                    ]
                )
            if chain_2["mol_type"] == const.chain_type_ids["NONPOLYMER"]:
                chain_2_seqres = str_2.residues["name"][chain_2["res_idx"]]
            if chain_2_seqres in str_1_seqres:
                indices = [i for i, x in enumerate(str_1_seqres) if x == chain_2_seqres]
                chains_2["entity_id"][chain_idx] = chains_1[indices[0]]["entity_id"]
                chains_2["sym_id"][chain_idx] += max(chains_1[indices]["sym_id"]) + 1
                existing_chains_str_2.append(chain_idx)
        for chain_idx in range(len(chains_2)):
            if chain_idx not in existing_chains_str_2:
                # Count how many existing chains have indices smaller than current chain_idx
                smaller_existing_count = sum(
                    1 for i in existing_chains_str_2 if i < chain_idx
                )
                chains_2["entity_id"][chain_idx] = (
                    chains_2["entity_id"][chain_idx]
                    + len(np.unique(chains_1["entity_id"]))
                    - smaller_existing_count
                )
        chains_2["res_idx"] = (
            chains_2["res_idx"]
            + chains_1["res_idx"][-1]
            + chains_1["res_num"][-1]
            - chains_2["res_idx"][0]
        )
        chains_2["asym_id"] = (
            chains_2["asym_id"] + chains_1["asym_id"][-1] + 1 - chains_2["asym_id"][0]
        )
        chains = np.concatenate([chains_1, chains_2])

        # Find names for new chains
        names1 = chains_1["name"]
        names2 = chains_2["name"]
        all_letters = list(string.ascii_uppercase)
        used_letters = set(names1) | set(names2)
        replacement_iter = (ch for ch in all_letters if ch not in used_letters)
        replacements = {}
        new_names2 = []
        for ch in names2:
            if ch in names1:
                new_ch = next(replacement_iter)
                replacements[ch.item()] = new_ch
                new_names2.append(new_ch)
            else:
                new_names2.append(ch)
        new_names2 = np.array(new_names2)
        chains["name"] = np.concatenate([names1, new_names2])

        # Build new bonds
        bonds_1 = str_1.bonds.copy()
        bonds_2 = np.array(
            [
                (
                    num_chains1 + chain_1,
                    num_chains1 + chain_2,
                    num_res1 + res_1,
                    num_res1 + res_2,
                    atom_1 + num_atoms1,
                    atom_2 + num_atoms1,
                    type,
                )
                for chain_1, chain_2, res_1, res_2, atom_1, atom_2, type in str_2.bonds  # noqa: A001
            ],
            dtype=Bond,
        )
        bonds = np.concatenate([bonds_1, bonds_2])

        # Build new mask
        mask = np.concatenate([str_1.mask.copy(), str_2.mask.copy()])

        # Build new ensemble
        ensemble = str_1.ensemble.copy()
        ensemble["atom_num"] = atoms.shape[0]

        # Build new coords
        coords = np.concatenate([str_1.coords.copy(), str_2.coords.copy()])

        # Interfaces
        interfaces_1 = str_1.interfaces.copy()
        interfaces_2 = str_2.interfaces.copy()
        interfaces_2["chain_1"] = interfaces_2["chain_1"] + num_chains1
        interfaces_2["chain_2"] = interfaces_2["chain_2"] + num_chains1
        interfaces = np.concatenate([interfaces_1, interfaces_2])

        data = self(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
        )

        if return_renaming:
            return data, replacements
        return data

    @classmethod
    def insert(
        self, structure: "Structure", chain_name: int, res_idx: int, num_residues: int
    ) -> "Structure":
        """Insert number of residues into chain of a strucure object.
        This creates new residues and inserts them into the structure.residues at the index obtained from the specified chain and the res_idx that indexes the chain.
        The inserted residues are GLY and the corresponding atoms are inserted into the structure.atoms.
        The bonds are correctly reindexed.
        The residue and atom indices in the chains are also appropriately reindexed.
        The mask remains the same, and the inserted coords are 000 for the inserted atoms.

        Parameters
        ----------
        structure : Structure
            Structure in which to insert the residues.

        chain_name : int
            Index of the chain in `structure.chains` in which the residues should be inserted.

        res_idx : int
            Residue index (starts at 0 for the chain) for where the residue should be inserted in the chain.

        num_residues : int
            Number of residues that are inserted at the res_idx

        """
        # -1. Define Glycine properties using the const module
        gly_atom_names = const.ref_atoms["GLY"]
        num_atoms_per_gly = len(gly_atom_names)
        num_new_atoms = num_residues * num_atoms_per_gly

        # Make copies of the original data to avoid in-place modification
        atoms = structure.atoms.copy()
        residues = structure.residues.copy()
        chains = structure.chains.copy()
        bonds = structure.bonds.copy()
        coords = structure.coords.copy()
        ensemble = structure.ensemble.copy()

        target_chain_idx = np.where(chains["name"] == chain_name)[0]
        target_chain = chains[target_chain_idx]

        # Absolute residue index in the full `residues` array
        res_insert_idx = target_chain["res_idx"] + res_idx

        # Absolute atom index in the full `atoms` array
        if res_idx == target_chain["res_num"]:
            # Inserting at the very end of the chain
            atom_insert_idx = target_chain["atom_idx"] + target_chain["atom_num"]
        else:
            # Inserting before an existing residue
            atom_insert_idx = residues[res_insert_idx]["atom_idx"]
        atom_insert_idx = atom_insert_idx.item()

        # Create new atoms and residues to be inserted
        insert_atoms_list = []
        insert_residues_list = []
        atom_creation_idx = atom_insert_idx
        for i in range(num_residues):
            # Create the new residue tuple
            insert_residues_list.append(
                (
                    "GLY",  # name
                    const.token_ids["GLY"],  # res_type
                    res_idx + i,  # res_idx (within the chain)
                    atom_creation_idx,  # atom_idx (absolute)
                    num_atoms_per_gly,  # atom_num
                    atom_creation_idx
                    + const.res_to_center_atom_id["GLY"],  # atom_center
                    atom_creation_idx + const.res_to_disto_atom_id["GLY"],  # atom_disto
                    True,  # is_standard
                    True,  # is_present
                )
            )

            # Create the new atom tuples for this residue
            for atom_name in gly_atom_names:
                insert_atoms_list.append(
                    (
                        atom_name,
                        [0.0, 0.0, 0.0],
                        True,  # is_present
                        0.0,  # bfactor
                        0.0,  # plddt
                    )
                )
            atom_creation_idx += num_atoms_per_gly
        insert_atoms = np.array(insert_atoms_list, dtype=Atom)
        insert_residues = np.array(insert_residues_list, dtype=Residue)
        insert_coords = np.array(
            [(np.zeros(3, dtype=np.float32),) for _ in range(num_new_atoms)],
            dtype=Coords,
        )

        # Insert new data into main arrays
        final_atoms = np.insert(atoms, atom_insert_idx, insert_atoms)
        final_residues = np.insert(residues, res_insert_idx, insert_residues)
        final_coords = np.insert(coords, atom_insert_idx, insert_coords)

        # Update indices in all data structures

        # Update residues: Shift atom indices for all residues that now come after the new ones.
        residues_after_mask = np.arange(
            res_insert_idx + num_residues, len(final_residues)
        )
        if residues_after_mask.size > 0:
            final_residues["atom_idx"][residues_after_mask] += num_new_atoms
            final_residues["atom_center"][residues_after_mask] += num_new_atoms
            final_residues["atom_disto"][residues_after_mask] += num_new_atoms
        # Update res_idx for residues within the same chain that now come after the new ones.
        chain_res_start_orig = target_chain["res_idx"]
        chain_res_end_orig = chain_res_start_orig + target_chain["res_num"]
        in_chain_after_mask = np.arange(
            res_insert_idx + num_residues, chain_res_end_orig + num_residues
        )
        if in_chain_after_mask.size > 0:
            final_residues["res_idx"][in_chain_after_mask] += num_residues

        # Update chains: Update counts for the target chain and indices for all subsequent chains.
        chains["res_num"][target_chain_idx] += num_residues
        chains["atom_num"][target_chain_idx] += num_new_atoms
        chains_after_mask = np.arange(target_chain_idx + 1, len(chains))
        if chains_after_mask.size > 0:
            chains["res_idx"][chains_after_mask] += num_residues
            chains["atom_idx"][chains_after_mask] += num_new_atoms

        # Update bonds: Shift atom and residue indices for any bond affected by the insertion.
        if bonds.size > 0:
            bonds["atom_1"][bonds["atom_1"] >= atom_insert_idx] += num_new_atoms
            bonds["atom_2"][bonds["atom_2"] >= atom_insert_idx] += num_new_atoms
            bonds["res_1"][bonds["res_1"] >= res_insert_idx] += num_residues
            bonds["res_2"][bonds["res_2"] >= res_insert_idx] += num_residues
            # Bond chain indices do not change as we are not adding/removing chains.

        # Update ensemble: Reflect the new total number of atoms.
        if ensemble.size > 0:
            ensemble["atom_num"] += num_new_atoms
            # If there are multiple ensemble entries, update all of them.
            for i in range(1, len(ensemble)):
                ensemble[i]["atom_coord_idx"] = (
                    ensemble[i - 1]["atom_coord_idx"] + ensemble[i - 1]["atom_num"]
                )

        # Create and return the new Structure object
        result = Structure(
            atoms=final_atoms,
            bonds=bonds,
            residues=final_residues,
            chains=chains,
            interfaces=structure.interfaces.copy(),
            mask=structure.mask.copy(),
            coords=final_coords,
            ensemble=ensemble,
        )
        return result

    @classmethod
    def fuse(
        self,
        structure1: "Structure",
        structure2: "Structure",
        chain_name: str,
        res_reindex: bool = False,
    ) -> "Structure":
        """Fuses one structure to a chain in the first structure
        Parameters
        ---------
        structure1: Structure
            Structure where we fuse
        structure2: Structure
            Structure where we take the chain from, needs to be a single chain
        chain_id: chain id of there chain where we wish to
        """
        assert len(structure2.chains) == 1
        assert chain_name in structure1.chains["name"]

        # Make copies of the original data to avoid in-place modification
        atoms = structure1.atoms.copy()
        residues = structure1.residues.copy()
        chains = structure1.chains.copy()
        bonds = structure1.bonds.copy()
        coords = structure1.coords.copy()
        ensemble = structure1.ensemble.copy()
        num_new_atoms = len(structure2.atoms)
        num_new_residues = len(structure2.residues)

        target_chain_idx = np.where(chains["name"] == chain_name)[0]
        target_chain = chains[target_chain_idx]

        for idx in range(len(chains)):
            if chains["entity_id"][idx] >= target_chain["entity_id"]:
                chains["entity_id"][idx] += 1
        chains["entity_id"][target_chain_idx] = structure1.chains["entity_id"][
            target_chain_idx
        ]

        # Absolute residue index in the full `residues` array
        res_insert_idx = target_chain["res_idx"] + target_chain["res_num"]

        atom_insert_idx = target_chain["atom_idx"] + target_chain["atom_num"]

        insert_atoms = structure2.atoms.copy()
        insert_residues = structure2.residues.copy()
        insert_coords = structure2.coords.copy()
        for residue in insert_residues:
            if res_reindex:
                residue["res_idx"] += (
                    target_chain["res_num"] - structure2.residues["res_idx"][0]
                )
            residue["atom_idx"] += atom_insert_idx
            residue["atom_center"] += atom_insert_idx
            residue["atom_disto"] += atom_insert_idx

        final_atoms = np.insert(atoms, atom_insert_idx, insert_atoms)
        final_residues = np.insert(residues, res_insert_idx, insert_residues)
        final_coords = np.insert(coords, atom_insert_idx, insert_coords)

        # Update indices in all data structures
        chains["res_num"][target_chain_idx] += num_new_residues
        chains["atom_num"][target_chain_idx] += num_new_atoms
        residues_after_mask = np.arange(
            res_insert_idx + len(insert_residues), len(final_residues)
        )
        if residues_after_mask.size > 0:
            final_residues["atom_idx"][residues_after_mask] += len(insert_atoms)
            final_residues["atom_center"][residues_after_mask] += len(insert_atoms)
            final_residues["atom_disto"][residues_after_mask] += len(insert_atoms)

        chains_after_mask = np.arange(target_chain_idx + 1, len(chains))
        if chains_after_mask.size > 0:
            chains["res_idx"][chains_after_mask] += num_new_residues
            chains["atom_idx"][chains_after_mask] += num_new_atoms

        # TODO: the bonds of structure2 are ignored right now and completely dropped

        # Update bonds: Shift atom and residue indices for any bond affected by the insertion.
        if bonds.size > 0:
            bonds["atom_1"][bonds["atom_1"] >= atom_insert_idx] += num_new_atoms
            bonds["atom_2"][bonds["atom_2"] >= atom_insert_idx] += num_new_atoms
            bonds["res_1"][bonds["res_1"] >= res_insert_idx] += num_new_residues
            bonds["res_2"][bonds["res_2"] >= res_insert_idx] += num_new_residues

        # Handle the cyclic bond in case the fusion target is cyclic
        if target_chain["cyclic_period"] > 0:
            chains["cyclic_period"][target_chain_idx] = chains["res_num"][
                target_chain_idx
            ]
            cyclic_bond_idx = np.where(
                (structure1.bonds["chain_1"] == target_chain["asym_id"])
                & (structure1.bonds["chain_2"] == target_chain["asym_id"])
                & (structure1.bonds["res_1"] == target_chain["res_idx"])
                & (
                    structure1.bonds["res_2"]
                    == (target_chain["res_idx"] + target_chain["res_num"] - 1)
                )
                & (atoms[structure1.bonds["atom_1"]]["name"] == "N")
                & (atoms[structure1.bonds["atom_2"]]["name"] == "C")
            )[0].item()

            bonds[cyclic_bond_idx]["res_2"] = (
                chains["res_idx"][target_chain_idx]
                + chains["res_num"][target_chain_idx]
                - 1
            )

            # Get atom indices
            res2 = final_residues[structure1.bonds[cyclic_bond_idx]["res_2"]]
            atoms2 = final_atoms[res2["atom_idx"] : res2["atom_idx"] + res2["atom_num"]]
            assert "C" in atoms2["name"]
            idx_in_res2 = np.where(atoms2["name"] == "C")[0].item()
            atom_idx2 = res2["atom_idx"] + idx_in_res2
            bonds[cyclic_bond_idx]["atom_2"] = atom_idx2

        # Update ensemble: Reflect the new total number of atoms.
        if ensemble.size > 0:
            ensemble["atom_num"] += num_new_atoms
            # If there are multiple ensemble entries, update all of them.
            for i in range(1, len(ensemble)):
                ensemble[i]["atom_coord_idx"] = (
                    ensemble[i - 1]["atom_coord_idx"] + ensemble[i - 1]["atom_num"]
                )
        fused = Structure(
            atoms=final_atoms,
            bonds=bonds,
            residues=final_residues,
            chains=chains,
            interfaces=structure1.interfaces.copy(),
            mask=structure1.mask.copy(),
            coords=final_coords,
            ensemble=ensemble,
        )
        return fused

    @classmethod
    def extract_residues(
        self, structure, res_indices, res_reindex=False
    ) -> "Structure":
        """Extract residues with res_indices. Only retains the first ensemble.
        This does not reindex the asym_id in the chains attribute.
        This does not reindex the res_idx in the residues attribute.

        Parameters
        ----------
        structure : Structure
            Structure to extract from

        res_indices : np.ndarray
            Indices to extract, either as integer indices or as boolean mask

        """

        # Take subsets
        res_indices = np.arange(len(structure.residues))[res_indices]
        residues = structure.residues[res_indices].copy()
        atom_indices = np.concatenate(
            [
                np.arange(len(structure.atoms))[
                    res["atom_idx"] : res["atom_idx"] + res["atom_num"]
                ]
                for res in residues
            ]
        )
        return self.extract_atoms(structure, atom_indices, res_reindex)

    @classmethod
    def extract_atoms(self, structure, atom_indices, res_reindex=False) -> "Structure":
        """Extract atoms with atom_indices. Only retains the first ensemble.
        This does not reindex the asym_id in the chains attribute.
        This does not reindex the res_idx in the residues attribute.

        Parameters
        ----------
        structure : Structure
            Structure to extract from

        atom_indices : np.ndarray
            Indices to extract, either as integer indices or as boolean mask

        """

        # Take subsets
        atom_indices = np.arange(len(structure.atoms))[atom_indices]
        atoms = structure.atoms[atom_indices].copy()

        res_indices = np.arange(len(structure.residues))[
            np.isin(structure.residues["atom_idx"], atom_indices)
        ]
        residues = structure.residues[res_indices].copy()

        chain_indices = np.array(
            [
                i
                for i, chain in enumerate(structure.chains)
                if np.any(
                    np.isin(
                        np.arange(
                            chain["atom_idx"], chain["atom_idx"] + chain["atom_num"]
                        ),
                        residues["atom_idx"],
                    )
                )
            ]
        ).astype(int)
        assert len(chain_indices) > 0
        chains = structure.chains[chain_indices].copy()

        bonds = structure.bonds[
            np.isin(structure.bonds["atom_1"], atom_indices)
            & np.isin(structure.bonds["atom_2"], atom_indices)
        ].copy()

        interfaces = structure.interfaces[
            np.isin(structure.interfaces["chain_1"], chain_indices)
            & np.isin(structure.interfaces["chain_2"], chain_indices)
        ].copy()

        mask = structure.mask[chain_indices].copy()

        coords = structure.coords[atom_indices].copy()

        ensemble = structure.ensemble.copy()

        # Reindex atom_idx
        old_to_new = {old.item(): new for new, old in enumerate(atom_indices)}
        old_to_new_res = {old.item(): new for new, old in enumerate(res_indices)}

        res_chain_map = {}

        for i in range(len(residues)):
            original_atom_range = np.arange(
                residues[i]["atom_idx"],
                residues[i]["atom_idx"] + residues[i]["atom_num"],
            )
            selected_atoms_in_residue = np.intersect1d(
                original_atom_range, atom_indices
            )
            residues[i]["atom_num"] = len(selected_atoms_in_residue)
            residues[i]["atom_idx"] = old_to_new[residues[i]["atom_idx"].item()]
            residues[i]["atom_center"] = old_to_new.get(
                residues[i]["atom_center"].item(), -1
            )
            residues[i]["atom_disto"] = old_to_new.get(
                residues[i]["atom_disto"].item(), -1
            )

        ensemble["atom_num"] = len(atoms)

        for i in range(len(bonds)):
            atom_idx1 = bonds[i]["atom_1"].item()
            atom_idx2 = bonds[i]["atom_2"].item()
            if atom_idx1 in atom_indices and atom_idx2 in atom_indices:
                bonds[i]["atom_1"] = old_to_new[atom_idx1]
                bonds[i]["atom_2"] = old_to_new[atom_idx2]
                bonds[i]["res_1"] = old_to_new_res[bonds[i]["res_1"].item()]
                bonds[i]["res_2"] = old_to_new_res[bonds[i]["res_2"].item()]

        for i in range(len(ensemble)):
            ensemble[i]["atom_coord_idx"] = len(atoms) * i
            ensemble[i]["atom_num"] = len(atoms)

        # Reindex residue table idx (note: this is not the res_idx in the residues)
        for i, chain in enumerate(chains):
            chain_atom_range = np.arange(
                chain["atom_idx"], chain["atom_idx"] + chain["atom_num"]
            )
            chain_res_range = np.arange(
                chain["res_idx"], chain["res_idx"] + chain["res_num"]
            )
            selected_atoms_in_chain = np.intersect1d(chain_atom_range, atom_indices)
            selected_residues_in_chain = np.intersect1d(chain_res_range, res_indices)
            if (
                len(selected_atoms_in_chain) == 0
                or len(selected_residues_in_chain) == 0
            ):
                raise ValueError(f"No selected atoms/residues found in chain {i}")
            chains[i]["atom_idx"] = selected_atoms_in_chain.min()
            chains[i]["res_idx"] = selected_residues_in_chain.min()
            chains[i]["atom_num"] = len(selected_atoms_in_chain)
            chains[i]["res_num"] = len(selected_residues_in_chain)

        for i, chain in enumerate(chains):
            orig_chain = structure.chains[chain_indices[i]]
            chain_start = orig_chain["res_idx"]
            chain_end = orig_chain["res_idx"] + orig_chain["res_num"]
            chain_res_indices = [r for r in res_indices if chain_start <= r < chain_end]
            res_chain_map[i] = {
                old.item(): new for new, old in enumerate(chain_res_indices)
            }

        for i in range(len(chains)):
            chains[i]["atom_idx"] = old_to_new[chains[i]["atom_idx"].item()]
            chains[i]["res_idx"] = old_to_new_res[chains[i]["res_idx"].item()]
            chains[i]["asym_id"] = i

        # reassign chains entity id
        entity_atom_counts = {}
        new_entity_id = max(chains["entity_id"]) + 1
        for i, chain in enumerate(chains):
            entity_id = chain["entity_id"]
            atom_num = chain["atom_num"]
            if entity_id in entity_atom_counts:
                if atom_num != entity_atom_counts[entity_id]:
                    chains[i]["entity_id"] = new_entity_id
                    new_entity_id += 1
            else:
                entity_atom_counts[entity_id] = atom_num

        # reindex res_index if res_reindex is True
        if res_reindex:
            for i, res in enumerate(residues):
                for chain_idx, chain in enumerate(chains):
                    chain_atom_start = chain["atom_idx"]
                    chain_atom_end = chain["atom_idx"] + chain["atom_num"]
                    if chain_atom_start <= res["atom_idx"] < chain_atom_end:
                        res_idx_item = res_indices[i]
                        residues[i]["res_idx"] = res_chain_map[chain_idx].get(
                            res_idx_item
                        )

        old_to_new_chains = {old.item(): new for new, old in enumerate(chain_indices)}
        for i in range(len(interfaces)):
            interfaces[i]["chain_1"] = old_to_new_chains[
                interfaces[i]["chain_1"].item()
            ]
            interfaces[i]["chain_2"] = old_to_new_chains[
                interfaces[i]["chain_2"].item()
            ]

        for i in range(len(bonds)):
            bonds[i]["chain_1"] = old_to_new_chains[bonds[i]["chain_1"].item()]
            bonds[i]["chain_2"] = old_to_new_chains[bonds[i]["chain_2"].item()]

        data = self(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
        )
        return data

    @classmethod
    def add_side_chains(self, structure, residue_mask=None) -> "Structure":
        """Add side chains (if absent) for specified residues. Currently only supports amino acids.
        Parameters
        ----------
        structure : Structure
            Structure with potentially missing side chains
        residue_mask : np.ndarray
            Mask of residues to consider for side chain activation
        Returns
        -------
        Structure
            Structure with side chains added
        """
        if residue_mask is None:
            residue_mask = np.ones(len(structure.residues), dtype=bool)
        assert len(residue_mask) == len(structure.residues), (
            f"residue_mask.shape: {residue_mask.shape}, structure.residues.shape: {structure.residues.shape}"
        )

        # Update residues and atoms, also add old to new atom index mapping
        old_to_new_atom = {}
        residues_new = []
        atoms_new = []
        atom_idx = 0
        for i, res in enumerate(structure.residues):
            # Just copy residue and corresponding atoms if not a standard protein residue or UNK
            if res["name"] not in list(const.res_to_center_atom.keys())[:21]:
                residues_new.append(
                    (
                        res["name"],
                        res["res_type"],
                        res["res_idx"],
                        atom_idx,
                        res["atom_num"],
                        res["atom_center"],
                        res["atom_disto"],
                        res["is_standard"],
                        res["is_present"],
                    )
                )
                for j, atom in enumerate(
                    structure.atoms[res["atom_idx"] : res["atom_idx"] + res["atom_num"]]
                ):
                    old_to_new_atom[res["atom_idx"].item() + j] = atom_idx + j
                    atoms_new.append(atom)
                atom_idx += res["atom_num"]
                continue

            # Change num_atoms if residue is not masked and does not have a side chain
            adding_side_chains = residue_mask[i] and (
                res["atom_num"] == 4 and res["name"] != "GLY"
            )
            if adding_side_chains:
                ref_atoms = const.ref_atoms[res["name"]]
                atom_num = len(ref_atoms)
            else:
                atom_num = res["atom_num"]

            # Adjust other residue fields
            atom_center = atom_idx + const.res_to_center_atom_id[res["name"]]
            atom_disto = atom_idx + const.res_to_disto_atom_id[res["name"]]
            residues_new.append(
                (
                    res["name"],
                    res["res_type"],
                    res["res_idx"],
                    atom_idx,
                    atom_num,
                    atom_center,
                    atom_disto,
                    res["is_standard"],
                    res["is_present"],
                )
            )

            # Copy backbone atoms
            for j, atom in enumerate(
                structure.atoms[res["atom_idx"] : res["atom_idx"] + res["atom_num"]]
            ):
                old_to_new_atom[res["atom_idx"].item() + j] = atom_idx + j
                atoms_new.append(atom)

            # Add side chain atoms
            if adding_side_chains:
                for atom_name in ref_atoms[4:]:
                    atoms_new.append(
                        (
                            atom_name,
                            np.zeros(3),
                            True,  # is_present
                            100.0,  # bfactor
                            0.0,  # plddt
                        )
                    )

            atom_idx += atom_num

        # Adjust other structure fields based on new residues and atoms
        residues_new = np.array(residues_new, dtype=Residue)
        atoms_new = np.array(atoms_new, dtype=Atom)
        coords_new = np.array([(atom["coords"],) for atom in atoms_new], dtype=Coords)
        chains_new = structure.chains.copy()
        for i, chain in enumerate(chains_new):
            first_res = residues_new[chain["res_idx"]]
            last_res = residues_new[chain["res_idx"] + chain["res_num"] - 1]
            chain["atom_idx"] = first_res["atom_idx"]
            chain["atom_num"] = (
                last_res["atom_idx"] + last_res["atom_num"] - first_res["atom_idx"]
            )

        bonds_new = []
        for bond in structure.bonds:
            c1, c2, r1, r2, a1, a2, t = bond
            a1 = old_to_new_atom[a1]
            a2 = old_to_new_atom[a2]
            bonds_new.append((c1, c2, r1, r2, a1, a2, t))
        bonds_new = np.array(bonds_new, dtype=Bond)
        # Set coords for side chains present (this handles the case when side chains are already
        # in the structure but have is_present==False)
        for res in structure.residues[residue_mask]:
            side_chain_idx = slice(
                res["atom_idx"] + 4, res["atom_idx"] + res["atom_num"]
            )
            atoms_new[side_chain_idx]["is_present"] = True

        return self(
            atoms=atoms_new,
            bonds=bonds_new,
            residues=residues_new,
            chains=chains_new,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=coords_new,
            ensemble=structure.ensemble,
        )

    @classmethod
    def empty_protein(self, seq_len: int, res_name: str = "GLY") -> "Structure":
        assert res_name == "GLY", (
            "Not implemented for anything other than GLY yet. To make it work for others, you also have to add the correct atom charges to atom data instead of always adding 0."
        )
        res_data = []
        atom_data = []
        coords_data = []
        atom_num = len(const.ref_atoms[res_name])
        center_idx = const.res_to_center_atom_id[res_name]
        disto_idx = const.res_to_disto_atom_id[res_name]
        atom_idx = 0
        for idx in range(seq_len):
            res_data.append(
                (
                    res_name,
                    const.token_ids[res_name],
                    idx,
                    atom_idx,
                    atom_num,
                    atom_idx + center_idx,
                    atom_idx + disto_idx,
                    True,
                    True,
                )
            )

            # add atoms
            for jdx in range(atom_num):
                atom_name = const.ref_atoms[res_name][jdx]
                atom_data.append(
                    (
                        atom_name,
                        [0.0, 0.0, 0.0],
                        True,
                        0,
                        0,
                    )
                )
                coords_data.append(([0.0, 0.0, 0.0],))
                atom_idx += 1

        chain_data = [
            (
                "A",
                const.chain_type_ids["PROTEIN"],
                0,
                0,
                0,
                0,
                len(atom_data),
                0,
                len(res_data),
                0,  # cyclic_period
                0,  # symmetric_group
            )
        ]

        data = self(
            atoms=np.array(atom_data, dtype=Atom),
            bonds=np.array([], dtype=Bond),
            residues=np.array(res_data, dtype=Residue),
            chains=np.array(chain_data, dtype=Chain),
            interfaces=np.array([], dtype=Interface),
            mask=np.ones(len(chain_data), dtype=bool),
            coords=np.array(coords_data, dtype=Coords),
            ensemble=np.array([(0, len(atom_data))], dtype=Ensemble),
        )
        return data

    @classmethod
    def from_feat_batch(
        self, feat: Dict[str, torch.Tensor], res_atoms_only: bool = False
    ) -> "Structure":
        sample = {k: v.squeeze() for k, v in feat.items()}
        self.from_feat(sample, res_atoms_only=res_atoms_only)

    @classmethod
    def from_feat(
        self, feat: Dict[str, torch.Tensor], res_atoms_only: bool = False
    ) -> "Structure":
        return self._from_feat(
            id=feat["id"],
            entity_id=feat["entity_id"].cpu(),
            asym_id=feat["asym_id"].cpu(),
            sym_id=feat["sym_id"].cpu(),
            mol_type=feat["mol_type"].cpu(),
            res_type=torch.argmax(feat["res_type"], dim=-1).cpu(),
            coords=feat["coords"].squeeze().cpu(),
            type_bonds=feat["type_bonds"].cpu(),
            structure_bonds=feat["structure_bonds"],
            new_to_old_atomidx=feat["new_to_old_atomidx"].cpu(),
            ref_element=torch.argmax(feat["ref_element"].int(), dim=-1).squeeze().cpu(),
            ref_charge=feat["ref_charge"].cpu(),
            ref_atom_name_chars=torch.argmax(
                feat["ref_atom_name_chars"].int(), dim=-1
            ).cpu(),
            atom_to_token=torch.argmax(feat["atom_to_token"].int(), dim=-1).cpu(),
            residue_index=feat["residue_index"].cpu(),
            atom_resolved_mask=feat["atom_resolved_mask"].cpu(),
            token_resolved_mask=feat["token_resolved_mask"].cpu(),
            design_mask=feat["design_mask"].cpu(),
            atom_pad_mask=feat["atom_pad_mask"].cpu(),
            is_standard=feat["is_standard"].cpu(),
            ccd=feat["ccd"].cpu(),
            token_to_res_old=feat["token_to_res"].cpu(),
            res_atoms_only=res_atoms_only,
        )

    @classmethod
    def _from_feat(
        self,
        id: str,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
        mol_type: torch.Tensor,
        res_type: torch.Tensor,
        ref_element: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        coords: torch.Tensor,
        type_bonds: torch.Tensor,
        structure_bonds: np.ndarray,
        new_to_old_atomidx: torch.Tensor,
        atom_to_token: torch.Tensor,
        residue_index: torch.Tensor,
        atom_resolved_mask: torch.Tensor,
        token_resolved_mask: torch.Tensor,
        design_mask: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        is_standard: torch.Tensor,
        ccd: torch.Tensor,
        token_to_res_old: torch.Tensor,
        res_atoms_only: bool = False,
    ) -> "Structure":
        """Executing this function can take around 0.15 seconds."""

        assert len(coords.shape) == 2, f"coords.shape: {coords.shape}"
        assert len(ref_element.shape) == 1, f"ref_element.shape: {ref_element.shape}"
        assert len(atom_to_token.shape) == 1, (
            f"atom_to_token.shape: {atom_to_token.shape}"
        )
        assert len(res_type.shape) == 1, f"res_type.shape: {res_type.shape}"
        assert len(type_bonds.shape) == 2, f"type_bonds.shape: {type_bonds.shape}"

        assert not res_atoms_only, (
            "res_atoms_only for structure from features is not finished implementing yet. I started it (see res_atoms_only), but this still leaves some non res atoms behind"
        )
        # remove token padding if there is any
        not_padding_selector = torch.where(res_type != const.token_ids["<pad>"])[0]
        asym_id = asym_id[not_padding_selector]
        entity_id = entity_id[not_padding_selector]
        sym_id = sym_id[not_padding_selector]
        mol_type = mol_type[not_padding_selector]
        res_type = res_type[not_padding_selector]
        residue_index = residue_index[not_padding_selector]
        design_mask = design_mask[not_padding_selector]
        ccd = ccd[not_padding_selector]
        type_bonds = type_bonds[
            : len(not_padding_selector), : len(not_padding_selector)
        ]
        token_to_res_old = token_to_res_old[not_padding_selector]

        # remove atom padding if there is any
        ref_element = ref_element[atom_pad_mask.bool()]
        ref_charge = ref_charge[atom_pad_mask.bool()]
        ref_atom_name_chars = ref_atom_name_chars[atom_pad_mask.bool()]
        coords = coords[atom_pad_mask.bool()]
        atom_to_token = atom_to_token[atom_pad_mask.bool()]

        # create residue identifiers
        res_identifiers = []
        for asym, res_idx in zip(asym_id, residue_index):
            res_identifiers.append(f"asym{asym}_res_idx{res_idx}")
        res_identifiers = np.array(res_identifiers)

        # find unique chains to start counting how many res they have
        res_per_chain = {cid: 0 for cid in np.unique(asym_id)}

        # create atom and residue data
        atom_idx = 0
        atom_data = []
        designed_atoms = []
        designed_residues = []
        res_data = []
        coords_data = []
        token_to_restable = []
        atom_to_res = []
        res_table_idx = 0
        processed_res_identifiers = []
        res_chain_id = []
        res_chain_idx = 0
        res_chain_indices = []
        res_mol_type = []
        res_entity_id = []
        res_sym_id = []
        chain_ids = []

        for res_identifier in res_identifiers:
            # dont process the same residue twice. Make sure that it is still in the same order as in the tokens
            if res_identifier in processed_res_identifiers:
                continue
            processed_res_identifiers.append(res_identifier)

            # get tokens of residue. One token if a canonical residue, otherwise all atomized tokens.
            token_selector = np.where(res_identifier == res_identifiers)[0]
            num_tokens_in_res = len(token_selector)
            token_to_restable.extend([res_table_idx] * num_tokens_in_res)

            # get atom mask for all atoms in the residue
            atom_mask = torch.zeros_like(atom_to_token)
            for token_selector_elem in token_selector:
                if res_atoms_only and is_standard[token_selector_elem]:
                    # only take as many atoms as there are in the standard residue
                    token_letters = const.tokens[res_type[token_selector_elem]]
                    token_num_atoms = len(const.ref_atoms[token_letters])
                    part_atom_mask = atom_to_token == token_selector_elem
                    start = torch.arange(len(part_atom_mask))[part_atom_mask][0]
                    part_atom_mask[start + token_num_atoms :] = False
                    atom_mask = atom_mask + part_atom_mask
                else:
                    atom_mask = atom_mask + (atom_to_token == token_selector_elem)
            atom_num = atom_mask.sum()
            atom_to_res.extend([atom_idx] * atom_num)

            # add residue
            if const.chain_types[mol_type[token_selector[0]]] == "NONPOLYMER":
                center_idx = disto_idx = 0
            else:
                token_name = const.tokens[res_type[token_selector[0]]]
                center_idx = const.res_to_center_atom_id[token_name]
                disto_idx = const.res_to_disto_atom_id[token_name]
            res_data.append(
                (
                    numeric_to_string(ccd[token_selector[0]]),
                    res_type[token_selector[0]].item(),
                    residue_index[token_selector[0]].item(),
                    # res_per_chain[asym_id[token_selector[0]].item()],
                    atom_idx,
                    atom_num.item(),
                    atom_idx + center_idx,
                    atom_idx + disto_idx,
                    is_standard[token_selector[0]],
                    token_resolved_mask[token_selector[0]],
                )
            )

            chain_id = asym_id[token_selector[0]]
            if res_table_idx == 0:
                chain_ids.append(chain_id.item())
            if res_table_idx > 0 and res_chain_id[-1] != chain_id:
                res_chain_idx += 1
                chain_ids.append(chain_id.item())
            res_chain_indices.append(res_chain_idx)
            res_chain_id.append(chain_id)
            res_mol_type.append(mol_type[token_selector[0]])
            res_entity_id.append(entity_id[token_selector[0]])
            res_sym_id.append(sym_id[token_selector[0]])
            if design_mask[token_selector[0]].item():
                designed_residues.append(res_table_idx)

            # add atoms
            for _ in range(atom_num):
                atom_data.append(
                    (
                        numeric_to_string(ref_atom_name_chars[atom_idx]),
                        (coords[atom_idx] * atom_resolved_mask[atom_idx]).numpy(),
                        atom_resolved_mask[atom_idx],
                        0,
                        0,
                    )
                )
                coords_data.append(
                    ((coords[atom_idx] * atom_resolved_mask[atom_idx]).numpy(),)
                )
                if design_mask[token_selector[0]].item():
                    designed_atoms.append(atom_idx)
                atom_idx += 1
            res_table_idx += 1
            res_per_chain[asym_id[token_selector[0]].item()] += 1
        token_to_restable = np.array(token_to_restable)
        res_chain_id = np.array(res_chain_id)
        res_chain_indices = np.array(res_chain_indices)
        res_data_array = np.array(res_data, dtype=Residue)

        # create bond data
        bond_data = []
        for bond in structure_bonds:
            # check that the bond is in the crop in case we are performing cropping.
            if (
                bond["chain_1"].item() in chain_ids
                and bond["chain_2"].item() in chain_ids
                and bond["res_1"].item() in token_to_res_old
                and bond["res_2"].item() in token_to_res_old
                and bond["atom_1"].item() in new_to_old_atomidx
                and bond["atom_2"].item() in new_to_old_atomidx
            ):
                chain_1 = np.where(np.asarray(chain_ids) == bond["chain_1"].item())[
                    0
                ].item()
                chain_2 = np.where(np.asarray(chain_ids) == bond["chain_2"].item())[
                    0
                ].item()

                token_1 = np.where(token_to_res_old == bond["res_1"].item())[0][0]
                token_2 = np.where(token_to_res_old == bond["res_2"].item())[0][0]
                res_1 = token_to_restable[token_1]
                res_2 = token_to_restable[token_2]

                atom_1 = np.where(new_to_old_atomidx == bond["atom_1"].item())[0][0]
                atom_2 = np.where(new_to_old_atomidx == bond["atom_2"].item())[0][0]

                bond_data.append(
                    (
                        chain_1,
                        chain_2,
                        res_1,
                        res_2,
                        atom_1,
                        atom_2,
                        bond["type"],
                    )
                )

        # determine entity ids
        seqs_to_id = {}
        id_counter = 0
        res_entity_id = []
        for chain_id in chain_ids:  # np.unique(res_chain_id):
            chain_res_selector = np.where(chain_id == res_chain_id)[0]
            seq = "".join(res_data_array["name"][chain_res_selector])
            if seq not in seqs_to_id.keys():
                seqs_to_id[seq] = id_counter
                id_counter += 1
            res_entity_id.extend([seqs_to_id[seq]] * len(chain_res_selector))

        # make chain data
        chain_data = []
        total_res = 0
        total_atoms = 0
        for chain_id in chain_ids:  # np.unique(res_chain_id):
            chain_res_selector = np.where(chain_id == res_chain_id)[0]
            chain_number = chain_id // 26 + 1
            chain_letter = chr(65 + chain_id % 26)
            num_atoms = (
                np.array([res[4] for res in res_data_array[chain_res_selector]])
                .sum()
                .item()
            )
            chain_data.append(
                (
                    chain_letter + str(chain_number),
                    res_mol_type[chain_res_selector[0]].item(),
                    res_entity_id[chain_res_selector[0]],
                    res_sym_id[chain_res_selector[0]].item(),
                    chain_id,
                    res_data[chain_res_selector[0]][3],
                    num_atoms,
                    total_res,
                    len(chain_res_selector),
                    0,  # cyclic_period
                    0,  # symmetric_group
                )
            )
            total_res += len(chain_res_selector)
            total_atoms += num_atoms
        assert total_atoms == len(atom_data)
        assert total_res == len(res_data)

        chains = np.array(chain_data, dtype=Chain)

        # Check same entity id have same number of atoms
        if not all(
            len({c["atom_num"] for c in chains if c["entity_id"] == entity}) == 1
            for entity in {c["entity_id"] for c in chains}
        ):
            print(
                "Warning in Structure._from_feat(): There are two chains with the same entity_id, but with a different number of atoms."
            )

        data = self(
            atoms=np.array(atom_data, dtype=Atom),
            bonds=np.array(bond_data, dtype=Bond),
            residues=res_data_array,
            chains=chains,
            interfaces=np.array([], dtype=Interface),
            mask=np.ones(len(chain_data), dtype=bool),
            coords=np.array(coords_data, dtype=Coords),
            ensemble=np.array([(0, len(atom_data))], dtype=Ensemble),
        )
        return data, np.array(designed_atoms), np.array(designed_residues)

    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            chain_1 = bond["chain_1"]
            chain_2 = bond["chain_2"]
            res_1 = bond["res_1"]
            res_2 = bond["res_2"]
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["chain_1"] = chain_map[chain_1]
                new_bond["chain_2"] = chain_map[chain_2]
                new_bond["res_1"] = res_map[res_1]
                new_bond["res_2"] = res_map[res_2]
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Create arrays
        bonds = np.array(bonds, dtype=Bond)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)
        coords = [(x,) for x in atoms["coords"]]
        coords = np.array(coords, Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)

        return Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
        )


def biotite_array_from_feat(feat):
    struc, _, _ = Structure.from_feat(feat)
    biotite_atoms = []

    chain_names = [re.sub(r"\d+", "", c["name"]) for c in struc.chains]
    chain_id_pool = list(reversed(string.ascii_uppercase)) + list(
        reversed(string.digits)
    )
    used_names = []
    old_to_new_chainid = {}
    for chain in struc.chains:
        old_chainid = chain["name"].item()
        new_chainid = re.sub(r"\d+", "", old_chainid)
        if new_chainid in used_names:
            # Find next unused chain ID from the pool
            for candidate in chain_id_pool:
                if candidate not in chain_names and candidate not in used_names:
                    new_chainid = candidate
                    break
        old_to_new_chainid[old_chainid] = new_chainid
        used_names.append(new_chainid)

    for chain in struc.chains:
        old_chainid = chain["name"].item()
        chain_id = old_to_new_chainid[old_chainid]

        residues = struc.residues[
            chain["res_idx"] : chain["res_idx"] + chain["res_num"]
        ]

        for res in residues:
            # Missing residues are in the seqres but not in the residue table
            if not res["is_present"]:
                continue

            res_name = res["name"].item()

            atoms = struc.atoms[res["atom_idx"] : res["atom_idx"] + res["atom_num"]]
            coords = struc.coords["coords"][
                res["atom_idx"] : res["atom_idx"] + res["atom_num"]
            ]

            for atom, coord, atom_idx in zip(
                atoms,
                coords,
                np.arange(res["atom_idx"], res["atom_idx"] + res["atom_num"]),
            ):
                # Skip missing atoms
                if not atom["is_present"]:
                    continue

                atom_name = atom["name"].item()
                element = elem_from_name(atom_name, res_name)

                # Skip fake atoms
                if (
                    const.fake_element.upper() in atom_name
                    or const.mask_element.upper() in atom_name
                ):
                    assert (
                        element == const.fake_element or element == const.mask_element
                    ), "Atom name not consistent with element for possible fake atom."
                    continue

                if res_name in const.formal_charges:
                    charge = const.formal_charges[res_name][atom_name]
                else:
                    charge = feat["ref_charge"][atom_idx]
                biotite_atom = biotite.structure.Atom(
                    coord,
                    chain_id=chain_id,
                    res_id=res["res_idx"],
                    res_name=res_name,
                    atom_name=atom_name,
                    element=element,
                    charge=charge,
                )

                biotite_atoms.append(biotite_atom)
    atom_array = biotite.structure.array(biotite_atoms)
    atom_array.bonds = biotite.structure.connect_via_residue_names(atom_array)

    # add design mask
    design_mask = feat["design_mask"].bool()
    design_resolved_mask = design_mask & feat["token_resolved_mask"].bool()
    atom_design_resolved_mask = (
        (feat["atom_to_token"].float() @ design_resolved_mask.unsqueeze(-1).float())
        .bool()
        .squeeze()
    )
    atom_pad_mask = feat["atom_pad_mask"].bool()
    atom_resolved_mask = feat["atom_resolved_mask"].bool()
    atom_array.add_annotation("is_design", bool)
    atom_array.is_design = atom_design_resolved_mask[
        atom_pad_mask & atom_resolved_mask
    ].bool()

    # add chain design mask
    chain_design_mask = feat["chain_design_mask"].bool()
    atom_chain_design_mask = (
        (feat["atom_to_token"].float() @ chain_design_mask.unsqueeze(-1).float())
        .bool()
        .squeeze()
    )
    atom_array.add_annotation("is_chain_design", bool)
    atom_array.is_chain_design = atom_chain_design_mask[
        atom_pad_mask & atom_resolved_mask
    ].bool()

    return atom_array


####################################################################################################
# MSA
####################################################################################################


MSAResidue = [
    ("res_type", np.dtype("i1")),
]

MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]

MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


@dataclass(frozen=True, slots=True)
class MSA(NumpySerializable):
    """MSA datatype."""

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


####################################################################################################
# TEMPLATE
####################################################################################################

TemplateCoordinates = [
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("frame_rot", np.dtype("9f4")),
    ("frame_t", np.dtype("3f4")),
    ("coords_cb", np.dtype("3f4")),
    ("coords_ca", np.dtype("3f4")),
    ("mask_frame", np.dtype("?")),
    ("mask_cb", np.dtype("?")),
    ("mask_ca", np.dtype("?")),
]


@dataclass(frozen=True, slots=True)
class Template(NumpySerializable):
    """Template datatype."""

    coordinates: np.ndarray


####################################################################################################
# RECORD
####################################################################################################


@dataclass(frozen=True)
class StructureInfo:
    """StructureInfo datatype."""

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None
    pH: Optional[float] = None
    temperature: Optional[float] = None


@dataclass(frozen=False)
class ChainInfo:
    """ChainInfo datatype."""

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: Union[str, int]
    msa_id: Union[str, int]
    num_residues: int
    valid: bool = True
    entity_id: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class InterfaceInfo:
    """InterfaceInfo datatype."""

    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass(frozen=True)
class TemplateInfo:
    """InterfaceInfo datatype."""

    name: str
    query_chain: str
    query_st: int
    query_en: int
    template_chain: str
    template_st: int
    template_en: int


@dataclass(frozen=True, slots=True)
class ConfidenceInfo:
    """ConfidenceInfo datatype."""

    confidence_score: Optional[float] = None
    ptm: Optional[float] = None
    iptm: Optional[float] = None
    ligand_iptm: Optional[float] = None
    protein_iptm: Optional[float] = None
    complex_plddt: Optional[float] = None
    complex_iplddt: Optional[float] = None
    complex_pde: Optional[float] = None
    complex_ipde: Optional[float] = None
    chains_ptm: Optional[dict] = None
    pair_chains_iptm: Optional[dict] = None


@dataclass(frozen=True)
class Record(JSONSerializable):
    """Record datatype."""

    id: str
    structure: StructureInfo
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    templates: Optional[list[TemplateInfo]] = None


####################################################################################################
# DESIGN INFO
####################################################################################################


@dataclass(frozen=True)
class DesignInfo(NumpySerializable):
    """Design Info datatype."""

    res_design_mask: npt.NDArray[np.bool_]
    res_structure_groups: npt.NDArray[np.int_]
    res_ss_types: npt.NDArray[np.int_]
    res_binding_type: npt.NDArray[np.int_]
    res_aa_constraint_mask: npt.NDArray[np.float32]  # Shape: (num_residues, 20), 0=allowed, 1=disallowed

    @classmethod
    def is_valid(self, info: "DesignInfo") -> bool:
        """Check if design info is valid"""

        assert (
            len(info.res_design_mask) == len(info.res_structure_groups)
            and len(info.res_structure_groups) == len(info.res_ss_types)
            and len(info.res_ss_types) == len(info.res_binding_type)
            and len(info.res_aa_constraint_mask) == len(info.res_design_mask)
        ), (
            "There must be a bug in the code. All residue level design info objects should have the same length."
        )

        if any(info.res_design_mask.astype(bool) & (info.res_structure_groups != 0)):
            msg = "[WARNING]: There were residues that have a structure group specified and are set to be designed. Make sure that you want to specify the backbone structure of designed residues."
            print(msg)

        if any(info.res_design_mask.astype(bool) & (info.res_binding_type != 0)):
            msg = "Misspecified design info. There were residues that have a binding type specified and are set to be designed. Only target residues can have a binding type specified since this feature indicates where the design should bind."
            raise ValueError(msg)

        if any(~info.res_design_mask.astype(bool) & (info.res_ss_types != 0)):
            msg = "Misspecified design info. There were residues that have a secondary structure type specified but are not set to be designed."
            raise ValueError(msg)

        # Validate residue constraints
        has_constraints = info.res_aa_constraint_mask.any(axis=1)
        if any(has_constraints & ~info.res_design_mask.astype(bool)):
            warnings.warn(
                "Residue constraints specified for non-designed residues "
                "will be ignored during inverse folding.",
                UserWarning,
                stacklevel=2,
            )

        # Check if any designed position has ALL amino acids blocked
        all_blocked = info.res_aa_constraint_mask.all(axis=1)
        if any(all_blocked & info.res_design_mask.astype(bool)):
            msg = "Invalid residue constraints: some designed positions have all amino acids disallowed."
            raise ValueError(msg)

        return True


####################################################################################################
# TARGET
####################################################################################################


@dataclass(frozen=True)
class Target:
    """Target datatype."""

    record: Record
    structure: Structure
    design_info: Optional[DesignInfo] = None
    sequences: Optional[dict[str, str]] = None
    templates: Optional[dict[str, Structure]] = None
    extra_mols: Optional[dict[str, Mol]] = None


@dataclass(frozen=True, slots=True)
class Manifest(JSONSerializable):
    """Manifest datatype."""

    records: List[Record]

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        Raises
        ------
        TypeError
            If the file is not a valid manifest file.

        """
        with path.open("r") as f:
            data = json.load(f)
            # New format
            if isinstance(data, dict):
                manifest = cls.from_dict(data)

            # Backward compatibility
            elif isinstance(data, list):
                records = [Record.from_dict(r) for r in data]
                manifest = cls(records=records)
            else:
                msg = "Invalid manifest file."
                raise TypeError(msg)

        return manifest


####################################################################################################
# TOKENS
####################################################################################################

# These need to contain a multiple of 4 bytes
Token = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i4")),
    ("res_name", np.dtype("<U8")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i1")),
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("modified", np.dtype("?")),
    ("frame_rot", np.dtype("9f4")),
    ("frame_t", np.dtype("3f4")),
    ("frame_mask", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("design_mask", np.dtype("?")),
    ("binding_type", np.dtype("i4")),
    ("structure_group", np.dtype("i4")),
    ("aa_constraint_mask", np.dtype("20f4")),  # Per-residue AA constraints: 20 floats (one per canonical AA)
    ("ccd", np.dtype("5i4")),
    ("target_msa_mask", np.dtype("?")),
    ("design_ss_mask", np.dtype("?")),
    ("feature_asym_id", np.dtype("i4")),
    ("feature_res_idx", np.dtype("i4")),
    ("symmetric_group", np.dtype("i4")),
]

TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    token_to_res: Optional[np.ndarray] = None


####################################################################################################
# INPUT
####################################################################################################


@dataclass(frozen=True, slots=True)
class Input:
    """Input datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    token_to_res: np.ndarray
    structure: Structure
    msa: Dict[str, MSA]
    templates: Dict[str, list[Template]]
    record: Optional[Record] = None
