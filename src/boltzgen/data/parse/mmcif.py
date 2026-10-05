import contextlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple

import gemmi
import numpy as np
from rdkit import rdBase
import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from sklearn.neighbors import KDTree

from boltzgen.data import const
from boltzgen.data.data import (
    Atom,
    Bond,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    Structure,
    StructureInfo,
)
from boltzgen.data.mol import load_molecules
from collections import defaultdict


####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True, slots=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    coords: Tuple[float, float, float]
    is_present: bool
    bfactor: float
    plddt: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True, slots=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: List[ParsedAtom]
    bonds: List[ParsedBond]
    auth_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool


@dataclass(frozen=True, slots=True)
class ParsedChain:
    """A parsed chain object."""

    name: str
    entity: str
    type: int
    residues: List[ParsedResidue]
    sequence: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ParsedConnection:
    """A parsed connection object."""

    chain_1: str
    chain_2: str
    residue_index_1: int
    residue_index_2: int
    atom_index_1: str
    atom_index_2: str


@dataclass(frozen=True, slots=True)
class ParsedStructure:
    """A parsed structure object."""

    data: Structure
    info: StructureInfo
    sequences: Dict[str, str]


####################################################################################################
# HELPERS
####################################################################################################


def get_mol(ccd: str, mols: Dict, moldir: str) -> Mol:
    """Get mol from CCD code.

    Return mol with ccd from mols if it is in mols. Otherwise load it from moldir,
    add it to mols, and return the mol.
    """
    mol = mols.get(ccd)
    if mol is None:
        # Load molecule
        mol = load_molecules(moldir, [ccd])[ccd]

        # Add to resource/dict
        if isinstance(mols, dict):
            mols[ccd] = mol
        else:
            mols.set(ccd, mol)

    return mol


def get_dates(block: gemmi.cif.Block) -> Tuple[str, str, str]:
    """Get the deposited, released, and last revision dates.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    str
        The deposited date.
    str
        The released date.
    str
        The last revision date.

    """
    deposited = "_pdbx_database_status.recvd_initial_deposition_date"
    revision = "_pdbx_audit_revision_history.revision_date"
    deposit_date = revision_date = release_date = ""
    with contextlib.suppress(Exception):
        deposit_date = block.find([deposited])[0][0]
        release_date = block.find([revision])[0][0]
        revision_date = block.find([revision])[-1][0]

    return deposit_date, release_date, revision_date


def get_resolution(block: gemmi.cif.Block) -> float:
    """Get the resolution from a gemmi structure.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    float
        The resolution.

    """
    resolution = 0.0
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        with contextlib.suppress(Exception):
            resolution = float(block.find([res_key])[0].str(0))
            break
    return resolution


def get_method(block: gemmi.cif.Block) -> str:
    """Get the method from a gemmi structure.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    str
        The method.

    """
    method = ""
    method_key = "_exptl.method"
    with contextlib.suppress(Exception):
        methods = block.find([method_key])
        method = ",".join([m.str(0).lower() for m in methods])

    return method


def get_experiment_conditions(
    block: gemmi.cif.Block,
) -> Tuple[Optional[float], Optional[float]]:
    """Get temperature and pH.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    Tuple[float, float]
        The temperature and pH.
    """
    temperature = None
    ph = None

    keys_t = [
        "_exptl_crystal_grow.temp",
        "_pdbx_nmr_exptl_sample_conditions.temperature",
    ]
    for key in keys_t:
        with contextlib.suppress(Exception):
            temperature = float(block.find([key])[0][0])
            break

    keys_ph = ["_exptl_crystal_grow.pH", "_pdbx_nmr_exptl_sample_conditions.pH"]
    with contextlib.suppress(Exception):
        for key in keys_ph:
            ph = float(block.find([key])[0][0])
            break

    return temperature, ph


def get_unk_token(dtype: gemmi.PolymerType) -> str:
    """Get the unknown token for a given entity type.

    Parameters
    ----------
    dtype : gemmi.EntityType
        The entity type.

    Returns
    -------
    str
        The unknown token.

    """
    if dtype == gemmi.PolymerType.PeptideL:
        unk = const.unk_token["PROTEIN"]
    elif dtype == gemmi.PolymerType.Dna:
        unk = const.unk_token["DNA"]
    elif dtype == gemmi.PolymerType.Rna:
        unk = const.unk_token["RNA"]
    else:
        msg = f"Unknown polymer type: {dtype}"
        raise ValueError(msg)

    return unk


def compute_covalent_ligands(
    connections: List[gemmi.Connection],
    subchain_map: Dict[Tuple[str, int], str],
    entities: Dict[str, gemmi.Entity],
) -> Set[str]:
    """Compute the covalent ligands from a list of connections.

    Parameters
    ----------
    connections: List[gemmi.Connection]
        The connections to process.
    subchain_map: Dict[Tuple[str, int], str]
        The mapping from chain, residue index to subchain name.
    entities: Dict[str, gemmi.Entity]
        The entities in the structure.

    Returns
    -------
    set
        The covalent ligand subchains.

    """
    # Get covalent chain ids
    covalent_chain_ids = set()
    for connection in connections:
        if connection.type.name not in {"Covale", "Disulf"}:
            continue
        # Map to correct subchain
        chain_1_name = connection.partner1.chain_name
        chain_2_name = connection.partner2.chain_name

        res_1_id = connection.partner1.res_id.seqid
        res_1_id = str(res_1_id.num) + str(res_1_id.icode).strip()

        res_2_id = connection.partner2.res_id.seqid
        res_2_id = str(res_2_id.num) + str(res_2_id.icode).strip()

        subchain_1 = subchain_map[(chain_1_name, res_1_id)]
        subchain_2 = subchain_map[(chain_2_name, res_2_id)]

        # If non-polymer or branched, add to set
        entity_1 = entities[subchain_1].entity_type.name
        entity_2 = entities[subchain_2].entity_type.name

        if entity_1 in {"NonPolymer", "Branched"}:
            covalent_chain_ids.add(subchain_1)
        if entity_2 in {"NonPolymer", "Branched"}:
            covalent_chain_ids.add(subchain_2)

    return covalent_chain_ids


def compute_interfaces(atom_data: np.ndarray, chain_data: np.ndarray) -> np.ndarray:
    """Compute the chain-chain interfaces from a gemmi structure.

    Parameters
    ----------
    atom_data : List[Tuple]
        The atom data.
    chain_data : List[Tuple]
        The chain data.

    Returns
    -------
    List[Tuple[int, int]]
        The interfaces.

    """
    # Compute chain_id per atom
    chain_ids = []
    for idx, chain in enumerate(chain_data):
        chain_ids.extend([idx] * chain["atom_num"])
    chain_ids = np.array(chain_ids)

    # Filter to present atoms
    coords = atom_data["coords"]
    mask = atom_data["is_present"]

    coords = coords[mask]
    chain_ids = chain_ids[mask]

    # Compute the distance matrix
    tree = KDTree(coords, metric="euclidean")
    query = tree.query_radius(coords, const.atom_interface_cutoff)

    # Get unique chain pairs
    interfaces = set()
    for c1, pairs in zip(chain_ids, query):
        chains = np.unique(chain_ids[pairs])
        chains = chains[chains != c1]
        interfaces.update((c1, c2) for c2 in chains)

    # Get unique chain pairs
    interfaces = [(min(i, j), max(i, j)) for i, j in interfaces]
    interfaces = list({(int(i), int(j)) for i, j in interfaces})
    interfaces = np.array(interfaces, dtype=Interface)
    return interfaces


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue_from_smiles(
    name: str,
    ref_mol: Mol,
    res_idx: int,
    gemmi_mol: Optional[gemmi.Residue] = None,
    is_covalent: bool = False,
) -> Optional[ParsedResidue]:
    """Parse a ligand from SMILES instead of CCD."""
    is_present = gemmi_mol is not None
    auth_idx = (
        str(gemmi_mol.seqid.num) + str(gemmi_mol.seqid.icode).strip()
        if is_present
        else None
    )

    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        bfactor = 0
        if is_present:
            pos = (
                gemmi_mol[0].pos.x,
                gemmi_mol[0].pos.y,
                gemmi_mol[0].pos.z,
            )
            bfactor = gemmi_mol[0].b_iso
        ref_atom = ref_mol.GetAtoms()[0]
        atom_name = f"{ref_atom.GetSymbol()}1"
        atom = ParsedAtom(
            name=atom_name,
            coords=pos,
            is_present=is_present,
            bfactor=bfactor,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        return ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            auth_idx=auth_idx,
            atom_center=0,
            atom_disto=0,
            is_standard=False,
            is_present=is_present,
        )

    pdb_pos = {}
    bfactor = {}
    if is_present:
        for atom in gemmi_mol:
            pos = (atom.pos.x, atom.pos.y, atom.pos.z)
            pdb_pos[atom.name] = pos
            bfactor[atom.name] = atom.b_iso

    atoms = []
    atom_idx = 0
    idx_map = {}
    element_counts = defaultdict(int)

    for i, atom in enumerate(ref_mol.GetAtoms()):
        symbol = atom.GetSymbol()
        element_counts[symbol] += 1
        atom_name = f"{symbol}{element_counts[symbol]}"

        coords = pdb_pos.get(atom_name, (0, 0, 0))
        atom_is_present = atom_name in pdb_pos

        atoms.append(
            ParsedAtom(
                name=atom_name,
                coords=coords,
                is_present=atom_is_present,
                bfactor=bfactor.get(atom_name, 0),
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1

    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        start = min(idx_map[idx_1], idx_map[idx_2])
        end = max(idx_map[idx_1], idx_map[idx_2])
        bond_type = const.bond_type_ids.get(bond.GetBondType().name, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        auth_idx=auth_idx,
        is_standard=False,
        is_present=is_present,
    )


def parse_ccd_residue(  # noqa: PLR0915
    name: str,
    ref_mol: Mol,
    res_idx: int,
    gemmi_mol: Optional[gemmi.Residue] = None,
    is_covalent: bool = False,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    components : Dict
        The preprocessed PDB components dictionary.
    res_idx : int
        The residue index.
    gemmi_mol : Optional[gemmi.Residue]
        The PDB molecule, as a gemmi Residue object, if any.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    # Check if we have a PDB structure for this residue,
    # it could be a missing residue from the sequence
    is_present = gemmi_mol is not None

    # Save original index (required for parsing connections)
    if is_present:
        auth_idx = gemmi_mol.seqid
        auth_idx = str(auth_idx.num) + str(auth_idx.icode).strip()
    else:
        auth_idx = None

    # Remove hydrogens
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

    # Check if this is a single atom CCD residue
    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        bfactor = 0
        if is_present:
            pos = (
                gemmi_mol[0].pos.x,
                gemmi_mol[0].pos.y,
                gemmi_mol[0].pos.z,
            )
            bfactor = gemmi_mol[0].b_iso
        ref_atom = ref_mol.GetAtoms()[0]
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            coords=pos,
            is_present=is_present,
            bfactor=bfactor,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            auth_idx=auth_idx,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=is_present,
        )
        return residue

    # If multi-atom, start by getting the PDB coordinates
    pdb_pos = {}
    bfactor = {}
    if is_present:
        # Match atoms based on names
        for atom in gemmi_mol:
            atom: gemmi.Atom
            pos = (atom.pos.x, atom.pos.y, atom.pos.z)
            pdb_pos[atom.name] = pos
            bfactor[atom.name] = atom.b_iso
    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")

        # If the atom is a leaving atom, skip if not in the PDB and is_covalent
        if (
            atom.HasProp("leaving_atom")
            and int(atom.GetProp("leaving_atom")) == 1
            and is_covalent
            and (atom_name not in pdb_pos)
        ):
            continue

        # Get PDB coordinates, if any
        coords = pdb_pos.get(atom_name.upper())
        if coords is None:
            atom_is_present = False
            coords = (0, 0, 0)
        else:
            atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                coords=coords,
                is_present=atom_is_present,
                bfactor=bfactor.get(atom_name, 0),
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        auth_idx=auth_idx,
        is_standard=False,
        is_present=is_present,
    )


def parse_polymer(  # noqa: C901, PLR0915, PLR0912
    polymer: gemmi.ResidueSpan,
    polymer_type: gemmi.PolymerType,
    sequence: List[str],
    chain_id: str,
    entity: str,
    mols: Dict[str, Mol],
    moldir: str,
    use_original_res_idx: bool = False,
    entity_poly_seq: Optional[List[List]] = None,
) -> Optional[ParsedChain]:
    """Process a gemmi Polymer into a chain object.

    Performs alignment of the full sequence to the polymer
    residues. Loads coordinates and masks for the atoms in
    the polymer, following the ordering in const.atom_order.

    Parameters
    ----------
    polymer : gemmi.ResidueSpan
        The polymer to process.
    polymer_type : gemmi.PolymerType
        The polymer type.
    sequence : str
        The full sequence of the polymer.
    chain_id : str
        The chain identifier.
    entity : str
        The entity name.

    Returns
    -------
    ParsedChain, optional
        The output chain, if successful.

    Raises
    ------
    ValueError
        If the alignment fails.

    """
    assert entity_poly_seq is not None
    
    # Since the polymer object already contains the global idx, we don't need to perform the alignment
    sequence = [_entity[1] for _entity in entity_poly_seq]

    i = 0
    ref_res = set(const.tokens)
    parsed = []
    for j, res_name in entity_poly_seq:
        res = None
        name_to_atom = {}
        if i < len(polymer) and j == polymer[i].label_seq:  # residue is not missing.
            # double check.
            assert polymer[i].name == res_name
            res = polymer[i]
            name_to_atom = {a.name.upper(): a for a in res}
            i += 1

        # Add residue to parsed list
        if res is not None:
            auth_idx = res.seqid.num
            res_idx = res.label_seq - 1  # convert to 0 indexing
        else:
            auth_idx = None
            res_idx = j - 1

        # Map MSE to MET, put the selenium atom in the sulphur column
        if res_name == "MSE":
            res_name = "MET"
            if "SE" in name_to_atom:
                name_to_atom["SD"] = name_to_atom["SE"]

        # Handle non-standard residues
        elif res_name not in ref_res:
            modified_mol = get_mol(res_name, mols, moldir)
            if modified_mol is not None:
                residue = parse_ccd_residue(
                    name=res_name,
                    ref_mol=modified_mol,
                    res_idx=res_idx if use_original_res_idx else j - 1,
                    gemmi_mol=res,
                    is_covalent=True,
                )
                parsed.append(residue)
                continue
            else:  # noqa: RET507
                res_name = "UNK"

        # Load regular residues
        ref_mol = get_mol(res_name, mols, moldir)
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

        # Only use reference atoms set in constants
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_name]]

        # Iterate, always in the same order
        atoms: List[ParsedAtom] = []

        for ref_atom in ref_atoms:
            # Get atom name
            atom_name = ref_atom.GetProp("name")

            # Get coordinates from PDB
            if atom_name in name_to_atom:
                atom: gemmi.Atom = name_to_atom[atom_name]
                atom_is_present = True
                coords = (atom.pos.x, atom.pos.y, atom.pos.z)
                bfactor = atom.b_iso
            else:
                atom_is_present = False
                coords = (0, 0, 0)
                bfactor = 0

            # Add atom to list
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    coords=coords,
                    is_present=atom_is_present,
                    bfactor=bfactor,
                )
            )

        # Fix naming errors in arginine residues where NH2 is
        # incorrectly assigned to be closer to CD than NH1
        if (res is not None) and (res_name == "ARG"):
            ref_atoms: List[str] = const.ref_atoms["ARG"]
            cd = atoms[ref_atoms.index("CD")]
            nh1 = atoms[ref_atoms.index("NH1")]
            nh2 = atoms[ref_atoms.index("NH2")]

            cd_coords = np.array(cd.coords)
            nh1_coords = np.array(nh1.coords)
            nh2_coords = np.array(nh2.coords)

            if all(atom.is_present for atom in (cd, nh1, nh2)) and (
                np.linalg.norm(nh1_coords - cd_coords)
                > np.linalg.norm(nh2_coords - cd_coords)
            ):
                atoms[ref_atoms.index("NH1")] = replace(nh1, coords=nh2.coords)
                atoms[ref_atoms.index("NH2")] = replace(nh2, coords=nh1.coords)

        atom_center = const.res_to_center_atom_id[res_name]
        atom_disto = const.res_to_disto_atom_id[res_name]

        parsed.append(
            ParsedResidue(
                name=res_name,
                type=const.token_ids[res_name],
                atoms=atoms,
                bonds=[],
                idx=res_idx if use_original_res_idx else j - 1,
                atom_center=atom_center,
                atom_disto=atom_disto,
                is_standard=True,
                is_present=res is not None,
                auth_idx=auth_idx,
            )
        )

    res_indices = [r.idx for r in parsed]
    assert len(set(res_indices)) == len(res_indices), (
        "There was an instance where the residue index of a present residue is the same as the assigned residue index for a missing residue (which only has its residue type in the seq_res)"
    )

    # Get polymer class
    if polymer_type == gemmi.PolymerType.PeptideL:
        chain_type = const.chain_type_ids["PROTEIN"]
    elif polymer_type == gemmi.PolymerType.Dna:
        chain_type = const.chain_type_ids["DNA"]
    elif polymer_type == gemmi.PolymerType.Rna:
        chain_type = const.chain_type_ids["RNA"]

    # Return polymer object
    return ParsedChain(
        name=chain_id,
        entity=entity,
        residues=parsed,
        type=chain_type,
        sequence=gemmi.one_letter_code(sequence),
    )


def parse_connection(
    connection: gemmi.Connection,
    chains: List[ParsedChain],
    subchain_map: Dict[Tuple[str, int], str],
) -> ParsedConnection:
    """Parse (covalent) connection from a gemmi Connection.

    Parameters
    ----------
    connections : gemmi.ConnectionList
        The connection list to parse.
    chains : List[Chain]
        The parsed chains.
    subchain_map : Dict[Tuple[str, int], str]
        The mapping from chain, residue index to subchain name.

    Returns
    -------
    List[Connection]
        The parsed connections.

    """
    # Map to correct subchains
    chain_1_name = connection.partner1.chain_name
    chain_2_name = connection.partner2.chain_name

    res_1_id = connection.partner1.res_id.seqid
    res_1_id = str(res_1_id.num) + str(res_1_id.icode).strip()

    res_2_id = connection.partner2.res_id.seqid
    res_2_id = str(res_2_id.num) + str(res_2_id.icode).strip()

    subchain_1 = subchain_map[(chain_1_name, res_1_id)]
    subchain_2 = subchain_map[(chain_2_name, res_2_id)]

    # Get chain indices
    chain_1 = next(chain for chain in chains if (chain.name == subchain_1))
    chain_2 = next(chain for chain in chains if (chain.name == subchain_2))

    # Get residue indices
    res_1_idx, res_1 = next(
        (idx, res)
        for idx, res in enumerate(chain_1.residues)
        if (str(res.auth_idx) == res_1_id)
    )
    res_2_idx, res_2 = next(
        (idx, res)
        for idx, res in enumerate(chain_2.residues)
        if (str(res.auth_idx) == res_2_id)
    )

    # Get atom indices
    atom_index_1 = next(
        idx
        for idx, atom in enumerate(res_1.atoms)
        if atom.name == connection.partner1.atom_name
    )
    atom_index_2 = next(
        idx
        for idx, atom in enumerate(res_2.atoms)
        if atom.name == connection.partner2.atom_name
    )

    conn = ParsedConnection(
        chain_1=subchain_1,
        chain_2=subchain_2,
        residue_index_1=res_1_idx,
        residue_index_2=res_2_idx,
        atom_index_1=atom_index_1,
        atom_index_2=atom_index_2,
    )

    return conn


def parse_mmcif(  # noqa: C901, PLR0915, PLR0912
    path: str,
    mols: Optional[Dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    use_original_res_idx: bool = False,
) -> ParsedStructure:
    """Parse a structure in MMCIF format.

    Parameters
    ----------
    mmcif_file : PathLike
        Path to the MMCIF file.

    use_original_res_idx : bool
        Retained for compatibility. Polymer residue indices use zero-based
        label_seq_id positions for both values, not author residue numbers.

    Returns
    -------
    ParsedStructure
        The parsed structure.

    """

    # Set mols
    mols = {} if mols is None else mols

    # Parse MMCIF input file
    block = gemmi.cif.read(str(path))[0]
    try:
        parsed = mmcif_from_block(
            block,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            use_original_res_idx=use_original_res_idx,
        )
    except Exception as e:
        print(f"\n\nFailed parsing mmcif: {path} \n\n\n")
        raise e
    return parsed


def mmcif_from_block(  # noqa: C901, PLR0915, PLR0912
    block,
    mols: Dict[str, Mol],
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    use_original_res_idx: bool = False,
) -> ParsedStructure:
    """Parse a structure in MMCIF format.

    Parameters
    ----------
    block :
        Gemmi Block.

    use_original_res_idx : bool
        Retained for compatibility. Polymer residue indices use zero-based
        label_seq_id positions for both values, not author residue numbers.

    Returns
    -------
    ParsedStructure
        The parsed structure.

    """
    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # Extract medatadata
    deposit_date, release_date, revision_date = get_dates(block)
    resolution = get_resolution(block)
    method = get_method(block)
    temperature, ph = get_experiment_conditions(block)

    # We can parse the "_entity_poly_seq" directly from the block.
    # [[[index_id, mon_id], ...], [[index_id, mon_id], ...], ..]
    entity_poly_seq_block_idx = block.get_index("_entity_poly_seq.entity_id")
    eps_loop = block[entity_poly_seq_block_idx].loop

    eps_label_id_idx = eps_loop.tags.index("_entity_poly_seq.num")
    eps_entity_id_idx = eps_loop.tags.index("_entity_poly_seq.entity_id")
    eps_mon_id_idx = eps_loop.tags.index("_entity_poly_seq.mon_id")

    entity_poly_seq, curr_seqid_mon = {}, []
    curr_entity_id = eps_loop[0, eps_entity_id_idx]
    for i in range(eps_loop.length()):
        entity_id = eps_loop[i, eps_entity_id_idx]
        num = eps_loop[i, eps_label_id_idx]
        mon_id = eps_loop[i, eps_mon_id_idx]

        if entity_id != curr_entity_id:
            entity_poly_seq[str(curr_entity_id)] = curr_seqid_mon
            curr_entity_id = entity_id
            curr_seqid_mon = []
        curr_seqid_mon.append([int(num), mon_id])
    entity_poly_seq[str(curr_entity_id)] = curr_seqid_mon

    # Load structure object
    structure = gemmi.make_structure_from_block(block)

    structure.merge_chain_parts()

    # Expand before removing waters: assembly generators may reference their
    # subchains, which Gemmi 0.7 requires to exist during expansion.
    if use_assembly and structure.assemblies:
        how = gemmi.HowToNameCopiedChain.AddNumber
        assembly_name = structure.assemblies[0].name
        structure.transform_to_assembly(assembly_name, how=how)

    # Clean up the structure
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    # Parse entities
    # Create mapping from subchain id to entity
    entities: Dict[str, gemmi.Entity] = {}
    entity_ids: Dict[str, int] = {}

    for entity_id, entity in enumerate(structure.entities):
        entity: gemmi.Entity
        if entity.entity_type.name == "Water":
            continue

        for subchain_id in entity.subchains:
            entities[subchain_id] = entity
            entity_ids[subchain_id] = entity_id

    # Create mapping from chain, residue to subchains
    # since a Connection uses the chains and not subchains
    subchain_map = {}
    for chain in structure[0]:
        for residue in chain:
            seq_id = residue.seqid
            seq_id = str(seq_id.num) + str(seq_id.icode).strip()
            subchain_map[(chain.name, seq_id)] = residue.subchain

    # Find covalent ligands
    covalent_chain_ids = compute_covalent_ligands(
        connections=structure.connections,
        subchain_map=subchain_map,
        entities=entities,
    )

    # Parse chains
    chains: List[ParsedChain] = []
    for raw_chain in structure[0].subchains():
        # Check chain type
        subchain_id = raw_chain.subchain_id()
        entity: gemmi.Entity = entities[subchain_id]

        entity_type = entity.entity_type.name

        # Parse a polymer
        if entity_type == "Polymer":
            # Skip PeptideD, DnaRnaHybrid, Pna, Other
            if entity.polymer_type.name not in {
                "PeptideL",
                "Dna",
                "Rna",
            }:
                continue

            # Add polymer if successful
            parsed_polymer = parse_polymer(
                polymer=raw_chain,
                polymer_type=entity.polymer_type,
                sequence=entity.full_sequence,
                chain_id=subchain_id,
                entity=entity.name,
                mols=mols,
                moldir=moldir,
                use_original_res_idx=use_original_res_idx,
                entity_poly_seq=entity_poly_seq[entity.name],
            )
            if parsed_polymer is not None:
                chains.append(parsed_polymer)

        # Parse a non-polymer
        elif entity_type in {"NonPolymer", "Branched"}:
            # Skip UNL
            if any(lig.name == "UNL" for lig in raw_chain):
                continue

            residues = []
            for lig_idx, ligand in enumerate(raw_chain):
                # Check if ligand is covalent
                if entity_type == "Branched":
                    is_covalent = True
                else:
                    is_covalent = subchain_id in covalent_chain_ids

                ligand: gemmi.Residue
                ligand_mol = get_mol(ligand.name, mols, moldir)

                residue = parse_ccd_residue(
                    name=ligand.name,
                    ref_mol=ligand_mol,
                    res_idx=lig_idx,
                    gemmi_mol=ligand,
                    is_covalent=is_covalent,
                )
                residues.append(residue)

            if residues:
                chains.append(
                    ParsedChain(
                        name=subchain_id,
                        entity=entity.name,
                        residues=residues,
                        type=const.chain_type_ids["NONPOLYMER"],
                    )
                )

    # If no chains parsed fail
    if not chains:
        msg = "No chains parsed!"
        raise ValueError(msg)

    # Want to traverse subchains in same order as reference structure
    ref_chain_map = {ref_chain.name: i for i, ref_chain in enumerate(chains)}
    all_ensembles = [chains]

    # Loop through different structures in model
    for struct in list(structure)[1:]:
        struct: gemmi.Model
        ensemble_chains = {}

        for raw_chain in struct.subchains():
            # Check chain type
            subchain_id = raw_chain.subchain_id()
            entity: gemmi.Entity = entities[subchain_id]
            entity_type = entity.entity_type.name

            # Parse a polymer
            if entity_type == "Polymer":
                # Skip PeptideD, DnaRnaHybrid, Pna, Other
                if entity.polymer_type.name not in {
                    "PeptideL",
                    "Dna",
                    "Rna",
                }:
                    continue

                # Add polymer if successful
                parsed_polymer = parse_polymer(
                    polymer=raw_chain,
                    polymer_type=entity.polymer_type,
                    sequence=entity.full_sequence,
                    chain_id=subchain_id,
                    entity=entity.name,
                    mols=mols,
                    moldir=moldir,
                    use_original_res_idx=use_original_res_idx,
                    entity_poly_seq=entity_poly_seq[entity.name],
                )
                if parsed_polymer is not None:
                    ensemble_chains[ref_chain_map[subchain_id]] = parsed_polymer

            # Parse a non-polymer
            elif entity_type in {"NonPolymer", "Branched"}:
                # Skip UNL
                if any(lig.name == "UNL" for lig in raw_chain):
                    continue

                residues = []
                for lig_idx, ligand in enumerate(raw_chain):
                    # Check if ligand is covalent
                    if entity_type == "Branched":
                        is_covalent = True
                    else:
                        is_covalent = subchain_id in covalent_chain_ids

                    ligand: gemmi.Residue
                    ligand_mol = get_mol(ligand.name, mols, moldir)

                    residue = parse_ccd_residue(
                        name=ligand.name,
                        ref_mol=ligand_mol,
                        res_idx=lig_idx,
                        gemmi_mol=ligand,
                        is_covalent=is_covalent,
                    )
                    residues.append(residue)

                if residues:
                    parsed_non_polymer = ParsedChain(
                        name=subchain_id,
                        entity=entity.name,
                        residues=residues,
                        type=const.chain_type_ids["NONPOLYMER"],
                    )
                    ensemble_chains[ref_chain_map[subchain_id]] = parsed_non_polymer

        # Ensure ensemble chains are in the same order as reference structure
        ensemble_chains = [ensemble_chains[idx] for idx in range(len(ensemble_chains))]
        all_ensembles.append(ensemble_chains)

    # Parse covalent connections
    connections: List[ParsedConnection] = []
    for connection in structure.connections:
        # Skip non-covalent connections
        connection: gemmi.Connection
        if connection.type.name not in {"Covale", "Disulf"}:
            continue
        try:
            parsed_connection = parse_connection(
                connection=connection,
                chains=chains,
                subchain_map=subchain_map,
            )
        except Exception:  # noqa: S112, BLE001
            continue
        connections.append(parsed_connection)
    # Create tables
    atom_data = []
    bond_data = []
    res_data = []
    chain_data = []
    ensemble_data = []
    coords_data = defaultdict(list)

    # Convert parsed chains to tables
    atom_idx = 0
    res_idx = 0
    sym_count = {}
    chain_to_idx = {}
    res_to_idx = {}
    chain_to_seq = {}

    for asym_id, chain in enumerate(chains):
        # Compute number of atoms and residues
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)

        # Get same chain across models in ensemble
        ensemble_chains = [ensemble[asym_id] for ensemble in all_ensembles]
        assert len(ensemble_chains) == len(all_ensembles)
        for ensemble_chain in ensemble_chains:
            assert len(ensemble_chain.residues) == res_num
            assert sum(len(res.atoms) for res in ensemble_chain.residues) == atom_num

        # Find all copies of this chain in the assembly
        entity_id = entity_ids[chain.name]
        sym_id = sym_count.get(entity_id, 0)

        chain_data.append(
            (
                chain.name,
                chain.type,
                entity_id,
                sym_id,
                asym_id,
                atom_idx,
                atom_num,
                res_idx,
                res_num,
                0,  # cyclic period
                0,  # symmetric_group (default, can be overridden via YAML)
            )
        )
        chain_to_idx[chain.name] = asym_id
        sym_count[entity_id] = sym_id + 1
        if chain.sequence is not None:
            chain_to_seq[chain.name] = chain.sequence

        # Add residue, atom, bond, data
        for i, res in enumerate(chain.residues):
            # Get same residue across models in ensemble
            ensemble_residues = [
                ensemble_chain.residues[i] for ensemble_chain in ensemble_chains
            ]
            assert len(ensemble_residues) == len(all_ensembles)
            for ensemble_res in ensemble_residues:
                assert ensemble_res.name == res.name

            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (
                    res.name,
                    res.type,
                    res.idx,
                    atom_idx,
                    len(res.atoms),
                    atom_center,
                    atom_disto,
                    res.is_standard,
                    res.is_present,
                )
            )
            res_to_idx[(chain.name, i)] = (res_idx, atom_idx)

            for bond in res.bonds:
                chain_1 = asym_id
                chain_2 = asym_id
                res_1 = res_idx
                res_2 = res_idx
                atom_1 = atom_idx + bond.atom_1
                atom_2 = atom_idx + bond.atom_2
                bond_data.append(
                    (
                        chain_1,
                        chain_2,
                        res_1,
                        res_2,
                        atom_1,
                        atom_2,
                        bond.type,
                    )
                )

            for a_idx, atom in enumerate(res.atoms):
                # Get same atom across models in ensemble
                ensemble_atoms = [
                    ensemble_res.atoms[a_idx] for ensemble_res in ensemble_residues
                ]
                assert len(ensemble_atoms) == len(all_ensembles)
                for e_idx, ensemble_atom in enumerate(ensemble_atoms):
                    assert ensemble_atom.name == atom.name
                    assert atom.is_present == ensemble_atom.is_present

                    coords_data[e_idx].append(ensemble_atom.coords)

                atom_data.append(
                    (
                        atom.name,
                        atom.coords,
                        atom.is_present,
                        atom.bfactor,
                        1.0,  # plddt is 1 for real data
                    )
                )
                atom_idx += 1

            res_idx += 1

    # Create coordinates table
    coords_data_ = []
    for e_idx in range(len(coords_data)):
        ensemble_data.append((e_idx * atom_idx, atom_idx))
        coords_data_.append(coords_data[e_idx])
    coords_data = [(x,) for xs in coords_data_ for x in xs]

    # Convert connections to tables
    for conn in connections:
        chain_1_idx = chain_to_idx[conn.chain_1]
        chain_2_idx = chain_to_idx[conn.chain_2]
        res_1_idx, atom_1_offset = res_to_idx[(conn.chain_1, conn.residue_index_1)]
        res_2_idx, atom_2_offset = res_to_idx[(conn.chain_2, conn.residue_index_2)]
        atom_1_idx = atom_1_offset + conn.atom_index_1
        atom_2_idx = atom_2_offset + conn.atom_index_2
        bond_data.append(
            (
                chain_1_idx,
                chain_2_idx,
                res_1_idx,
                res_2_idx,
                atom_1_idx,
                atom_2_idx,
                const.bond_type_ids["COVALENT"],
            )
        )
        if (
            conn.chain_1 == conn.chain_2
            and conn.residue_index_1 == 0
            and conn.residue_index_2
            == len(chains[chain_to_idx[conn.chain_1]].residues) - 1
        ) or (
            conn.chain_1 == conn.chain_2
            and conn.residue_index_1
            == len(chains[chain_to_idx[conn.chain_1]].residues) - 1
            and conn.residue_index_2 == 0
        ):
            if atom_data[atom_1_idx][0] == "SG" and atom_data[atom_2_idx][0] == "SG":
                continue
            temp = list(chain_data[chain_1_idx])
            temp[9] = len(chains[chain_to_idx[conn.chain_1]].residues)
            chain_data[chain_1_idx] = tuple(temp)

    # Convert into datatypes
    atoms = np.array(atom_data, dtype=Atom)
    bonds = np.array(bond_data, dtype=Bond)
    residues = np.array(res_data, dtype=Residue)
    chains = np.array(chain_data, dtype=Chain)
    mask = np.ones(len(chain_data), dtype=bool)
    ensemble = np.array(ensemble_data, dtype=Ensemble)
    coords = np.array(coords_data, dtype=Coords)

    interfaces = np.array([], dtype=Interface)

    # Return parsed structure
    info = StructureInfo(
        deposited=deposit_date,
        revised=revision_date,
        released=release_date,
        resolution=resolution,
        method=method,
        num_chains=len(chains),
        num_interfaces=len(interfaces),
        temperature=temperature,
        pH=ph,
    )

    data = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        ensemble=ensemble,
        coords=coords,
    )

    return ParsedStructure(
        data=data,
        info=info,
        sequences=chain_to_seq,
    )


def compute_3d(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = rdkit.Chem.AllChem.ETKDGv3()
    elif version == "v2":
        options = rdkit.Chem.AllChem.ETKDGv2()
    else:
        options = rdkit.Chem.AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = rdkit.Chem.AllChem.EmbedMolecule(mol, options)
        rdkit.Chem.AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", ConformerType.Computed.name)
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False
